# Builds cache/img_mobilenet_v3_small_224.npz from the RosVLA h5 files.
# Only needed if you don't already have the cache from the RosVLA repo.
import h5py
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as tvm

from data import DATASET, FEATURES

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@torch.no_grad()
def main():
    net = tvm.mobilenet_v3_small(weights=tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1).features.eval()
    keys, feats = [], []
    for path in sorted((DATASET / "data").glob("*.h5")):
        with h5py.File(path, "r") as f:
            images = f["rgb_images"][:]
        for start in range(0, len(images), 16):
            x = torch.from_numpy(images[start:start + 16]).permute(0, 3, 1, 2).float() / 255.0
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
            out = net((x - MEAN) / STD).mean(dim=(2, 3))
            feats.append(out.numpy())
        keys += [f"{path.stem}:{i}" for i in range(len(images))]
        print(path.stem, len(images))
    FEATURES.parent.mkdir(exist_ok=True)
    np.savez_compressed(FEATURES, feat=np.concatenate(feats).astype(np.float32), key=np.array(keys))
    print("saved", FEATURES)


if __name__ == "__main__":
    main()
