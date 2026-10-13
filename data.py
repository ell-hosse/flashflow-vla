import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
DATASET = ROOT / "dataset"
FEATURES = ROOT / "cache" / "img_mobilenet_v3_small_224.npz"
CHECKPOINTS = ROOT / "checkpoints"

ACTIONS = ["FORWARD", "RIGHT_TURN", "LEFT_TURN", "STOP", "REVERSE_LEFT"]
HIST = 6
HORIZON = 6


def load_runs():
    runs = {}
    for fp in sorted((DATASET / "annotations").glob("*.json")):
        name = fp.name.split("_vla_annotations")[0]
        recs = json.loads(fp.read_text())
        u = np.array([[r["robot_state"]["linear_velocity"], r["robot_state"]["angular_velocity"]] for r in recs],
                     dtype=np.float32)
        labels = np.array([ACTIONS.index(r["action"]["action_label"]) for r in recs])
        runs[name] = (u, labels)
    return runs


def load_features():
    if not FEATURES.exists():
        raise FileNotFoundError(f"{FEATURES} not found, run: python features.py")
    blob = np.load(FEATURES, allow_pickle=True)
    return {str(k): f for k, f in zip(blob["key"], blob["feat"])}


def windows(scenario, runs, feats):
    # same windows as the paper: past 6 commands in, next 6 commands out
    u, labels = runs[scenario]
    img, tel, chunk, act = [], [], [], []
    for t in range(HIST, len(u) - HORIZON):
        img.append(feats[f"{scenario}:{t}"])
        tel.append(u[t - HIST:t].ravel())
        chunk.append(u[t:t + HORIZON])
        act.append(labels[t])
    return np.stack(img), np.stack(tel), np.stack(chunk), np.array(act)


def folds():
    # checkpoints/fold_03_scenario_3_run_02/model.pt -> held-out scenario_3_run_02
    out = []
    for d in sorted(CHECKPOINTS.glob("fold_*")):
        out.append((d / "model.pt", d.name.split("_", 2)[2]))
    return out


def test_inputs(ckpt, scenario, runs, feats):
    img, tel, chunk, act = windows(scenario, runs, feats)
    img = (img - ckpt["img_mean"]) / ckpt["img_std"]
    tel = (tel - ckpt["tel_mean"]) / ckpt["tel_std"]
    return img.astype(np.float32), tel.astype(np.float32), chunk, act
