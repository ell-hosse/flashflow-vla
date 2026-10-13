# Runs both samplers on the held-out scenario of every RosVLA fold, with the same noise,
# and checks that the robot would get the same commands.
import json

import numpy as np
import torch

from data import ROOT, folds, load_features, load_runs, test_inputs
from kernel import FusedSampler
from model import load_checkpoint

STEPS = 4
device = "cuda" if torch.cuda.is_available() else "cpu"


def to_commands(v, w, ckpt):
    v = v.cpu().numpy() * ckpt["v_std"] + ckpt["v_mean"]
    w = w.cpu().numpy() * ckpt["w_std"] + ckpt["w_mean"]
    return np.stack([v, w], axis=2)


@torch.no_grad()
def main():
    runs, feats = load_runs(), load_features()
    rows = []
    print(f"{'held-out scenario':<20}{'n':>4}{'v MAE':>9}{'w MAE':>9}{'acc':>7}   |{'v MAE':>9}{'w MAE':>9}{'acc':>7}"
          f"{'same action':>13}{'max |dz|':>10}")
    print(f"{'':<24}{'---- PyTorch ----':>25}   |{'----- Triton -----':>25}")

    for path, scenario in folds():
        model, ckpt = load_checkpoint(path, device)
        img, tel, chunk, act = test_inputs(ckpt, scenario, runs, feats)
        img = torch.tensor(img, device=device)
        tel = torch.tensor(tel, device=device)
        g = torch.Generator(device="cpu").manual_seed(0)
        noise = torch.randn(len(img), model.flow.latent_dim, generator=g).to(device)

        fused = FusedSampler(model.flow)
        z_ref, v_ref, w_ref, l_ref = model.predict(img, tel, STEPS, noise)
        z_tri, v_tri, w_tri, l_tri = model.predict(img, tel, STEPS, noise, sampler=fused)

        res = {"scenario": scenario, "n": len(img), "max_dz": (z_ref - z_tri).abs().max().item()}
        for tag, v, w, logits in [("pytorch", v_ref, w_ref, l_ref), ("triton", v_tri, w_tri, l_tri)]:
            cmd = to_commands(v, w, ckpt)
            pred = logits.argmax(1).cpu().numpy()
            res[tag] = {
                "v_mae": float(np.abs(cmd[..., 0] - chunk[..., 0]).mean()),
                "w_mae": float(np.abs(cmd[..., 1] - chunk[..., 1]).mean()),
                "accuracy": float((pred == act).mean()),
            }
        res["same_action"] = float((l_ref.argmax(1) == l_tri.argmax(1)).float().mean().item())
        rows.append(res)

        p, t = res["pytorch"], res["triton"]
        print(f"{scenario:<20}{res['n']:>4}{p['v_mae']:>9.4f}{p['w_mae']:>9.4f}{p['accuracy']:>7.2f}   |"
              f"{t['v_mae']:>9.4f}{t['w_mae']:>9.4f}{t['accuracy']:>7.2f}{res['same_action']:>13.2f}{res['max_dz']:>10.1e}")

    if not rows:
        print("no checkpoints found in checkpoints/")
        return
    mean = lambda tag, k: np.mean([r[tag][k] for r in rows])
    print(f"{'mean':<24}{mean('pytorch', 'v_mae'):>9.4f}{mean('pytorch', 'w_mae'):>9.4f}{mean('pytorch', 'accuracy'):>7.2f}   |"
          f"{mean('triton', 'v_mae'):>9.4f}{mean('triton', 'w_mae'):>9.4f}{mean('triton', 'accuracy'):>7.2f}"
          f"{np.mean([r['same_action'] for r in rows]):>13.2f}{max(r['max_dz'] for r in rows):>10.1e}")

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "eval.json").write_text(json.dumps(rows, indent=2))
    print("saved results/eval.json")


if __name__ == "__main__":
    main()
