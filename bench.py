# Latency of one full RosVLA prediction (frame + command history -> next 6 commands) on the GPU.
#   python bench.py           full run
#   python bench.py --quick   fewer iterations
import argparse
import csv
import statistics
import time
from pathlib import Path

import torch

from data import folds, load_features, load_runs, test_inputs
from kernel import FusedSampler
from model import RosVLA, load_checkpoint

torch.set_grad_enabled(False)
device = "cuda"
LAUNCH_CALLS = ("cudaLaunchKernel", "cuLaunchKernel", "cudaLaunchKernelExC", "cuLaunchKernelEx", "cudaGraphLaunch")


def timeit(fn, iters):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
    return statistics.median(times) * 1000


def count_launches(fn):
    acts = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=acts) as prof:
        fn()
        torch.cuda.synchronize()
    return sum(1 for e in prof.events() if e.name in LAUNCH_CALLS)


def cuda_graph(fn, img, tel, noise):
    static = [img.clone(), tel.clone(), noise.clone()]
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn(*static)
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn(*static)

    def replay(img, tel, noise):
        for a, b in zip(static, [img, tel, noise]):
            a.copy_(b)
        graph.replay()
        return out
    return replay


def run(model, img, tel, steps, iters):
    noise = torch.randn(len(img), model.flow.latent_dim, device=device)
    fused = FusedSampler(model.flow)
    fused_tf32 = FusedSampler(model.flow, precision="tf32")
    per_step = FusedSampler(model.flow, one_launch=False)

    def pytorch(img, tel, noise):
        return model.predict(img, tel, steps, noise)[1]

    def triton(img, tel, noise):
        return model.predict(img, tel, steps, noise, sampler=fused)[1]

    def triton_tf32(img, tel, noise):
        return model.predict(img, tel, steps, noise, sampler=fused_tf32)[1]

    def triton_per_step(img, tel, noise):
        return model.predict(img, tel, steps, noise, sampler=per_step)[1]

    variants = [
        ("pytorch", lambda: pytorch),
        ("torch.compile", lambda: torch.compile(pytorch)),
        ("pytorch + cuda graph", lambda: cuda_graph(pytorch, img, tel, noise)),
        ("triton, 1 kernel per step", lambda: triton_per_step),
        ("triton, 1 kernel total", lambda: triton),
        ("triton tf32", lambda: triton_tf32),
        ("triton + cuda graph", lambda: cuda_graph(triton, img, tel, noise)),
    ]
    results = {}
    for name, make in variants:
        try:
            fn = make()
            ms = timeit(lambda: fn(img, tel, noise), iters)
            results[name] = (ms, count_launches(lambda: fn(img, tel, noise)))
        except Exception as e:
            print(f"  skipped {name}: {type(e).__name__}: {str(e).splitlines()[0][:100]}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    iters = 200 if args.quick else 1000

    print("GPU:", torch.cuda.get_device_name())
    checkpoints = folds()
    if checkpoints:
        path, scenario = checkpoints[0]
        model, ckpt = load_checkpoint(path, device)
        img, tel, _, _ = test_inputs(ckpt, scenario, load_runs(), load_features())
        img, tel = torch.tensor(img, device=device), torch.tensor(tel, device=device)
        print(f"model: {path.parent.name}, inputs: RosVLA frames from {scenario}")
    else:
        model = RosVLA().to(device).eval()
        img, tel = torch.randn(64, 576, device=device), torch.randn(64, 12, device=device)
        print("no checkpoints found, using a random model and random inputs")

    rows = []
    for batch, steps in [(1, 1), (1, 2), (1, 4), (1, 8), (16, 4), (64, 4), (256, 4)]:
        idx = torch.arange(batch, device=device) % len(img)
        print(f"\nbatch {batch}, {steps} flow steps")
        res = run(model, img[idx], tel[idx], steps, iters)
        base = res["pytorch"][0]
        for name, (ms, launches) in res.items():
            print(f"  {name:<28}{ms:8.3f} ms  {base / ms:5.1f}x  {launches:4d} launches")
            rows.append([batch, steps, name, round(ms, 4), round(base / ms, 2), launches])

    Path("results").mkdir(exist_ok=True)
    with open("results/bench.csv", "w", newline="") as f:
        csv.writer(f).writerows([["batch", "steps", "variant", "ms", "speedup", "launches"]] + rows)
    print("\nsaved results/bench.csv, make the plots with: python plot.py")


if __name__ == "__main__":
    main()
