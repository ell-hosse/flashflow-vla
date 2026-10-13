# Checks that the Triton sampler gives the same z as the PyTorch one.
# On a machine without a GPU: TRITON_INTERPRET=1 python test.py
import torch

from data import folds
from kernel import FusedSampler
from model import RosVLA, load_checkpoint

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)


def check(model, name):
    worst = 0.0
    for precision, tol in [("ieee", 1e-4), ("tf32", 2e-2)]:
        if device == "cpu" and precision == "tf32":
            continue
        for one_launch in [True, False]:
            fused = FusedSampler(model.flow, precision=precision, one_launch=one_launch)
            for batch in [1, 7, 64, 300]:
                for steps in [1, 2, 4, 8]:
                    cond = torch.randn(batch, 160, device=device)
                    noise = torch.randn(batch, model.flow.latent_dim, device=device)
                    ref = model.flow.sample(cond, steps, noise)
                    out = fused(cond, steps, noise)
                    err = (ref - out).abs().max().item()
                    if err > tol:
                        raise AssertionError(f"{name} {precision} B={batch} steps={steps}: max error {err:.2e}")
                    if precision == "ieee":
                        worst = max(worst, err)
    print(f"{name:<32} ok   (max error vs PyTorch: {worst:.1e})")


with torch.no_grad():
    checkpoints = folds()
    if not checkpoints:
        print("no checkpoints found, testing a randomly initialised model")
        check(RosVLA().to(device).eval(), "random model")
    for path, scenario in checkpoints:
        model, _ = load_checkpoint(path, device)
        check(model, path.parent.name)
print("all good")
