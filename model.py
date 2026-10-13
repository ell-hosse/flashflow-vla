import math
import pathlib
import pickle
import types

import torch
import torch.nn as nn


def timestep_embedding(s, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(1000.0) * torch.arange(half, device=s.device, dtype=torch.float32) / half)
    ang = s[:, None] * freqs[None] * 1000.0
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


class FiLMResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.SiLU()

    def forward(self, x, scale, shift):
        h = self.norm(self.fc1(x))
        h = h * (1 + scale) + shift
        return x + self.fc2(self.act(h))


class FlowHead(nn.Module):
    def __init__(self, latent_dim, cond_dim, hidden=256, n_blocks=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.time_dim = 32
        self.hidden = hidden
        self.n_blocks = n_blocks
        n_film = 2 * hidden * n_blocks

        self.inp = nn.Linear(latent_dim, hidden)
        self.blocks = nn.ModuleList(FiLMResBlock(hidden) for _ in range(n_blocks))
        self.out = nn.Linear(hidden, latent_dim)
        self.film_cond = nn.Sequential(nn.Linear(cond_dim, hidden), nn.SiLU(), nn.Linear(hidden, n_film))
        self.film_time = nn.Linear(self.time_dim, n_film)

    def velocity(self, z, s, cond_film):
        film = cond_film + self.film_time(timestep_embedding(s, self.time_dim))
        p = film.view(-1, self.n_blocks, 2, self.hidden)
        h = self.inp(z)
        for i, blk in enumerate(self.blocks):
            h = blk(h, p[:, i, 0], p[:, i, 1])
        return self.out(h)

    def sample(self, cond, steps, noise):
        cond_film = self.film_cond(cond)
        z = noise
        dt = 1.0 / steps
        for i in range(steps):
            s = torch.full((z.shape[0],), i * dt, device=z.device)
            z = z + dt * self.velocity(z, s, cond_film)
        return z


class VelocityHead(nn.Module):
    def __init__(self, cond_dim, latent_dim, hidden, horizon, n_actions):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(cond_dim + latent_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.lin = nn.Linear(hidden, horizon)
        self.ang = nn.Linear(hidden, horizon)
        self.action = nn.Linear(hidden, n_actions)

    def forward(self, cond, z):
        h = self.trunk(torch.cat([cond, z], dim=-1))
        return self.lin(h), self.ang(h), self.action(h)


class RosVLA(nn.Module):
    def __init__(self, img_dim=576, tel_dim=12, latent_dim=7, hidden=256, n_blocks=3, horizon=6, n_actions=5):
        super().__init__()
        self.img_proj = nn.Sequential(nn.Linear(img_dim, 128), nn.SiLU())
        self.tel_proj = nn.Sequential(nn.Linear(tel_dim, 32), nn.SiLU())
        self.flow = FlowHead(latent_dim, 160, hidden, n_blocks)
        self.head = VelocityHead(160, latent_dim, hidden, horizon, n_actions)

    def predict(self, img, tel, steps, noise, sampler=None):
        cond = torch.cat([self.img_proj(img), self.tel_proj(tel)], dim=-1)
        if sampler is None:
            z = self.flow.sample(cond, steps, noise)
        else:
            z = sampler(cond, steps, noise)
        v, w, logits = self.head(cond, z)
        return z, v, w, logits


class _Unpickler(pickle.Unpickler):
    # the checkpoints were saved on Windows and hold WindowsPath objects in their config
    def find_class(self, module, name):
        if name == "WindowsPath":
            return pathlib.PurePosixPath
        return super().find_class(module, name)


_pickle = types.ModuleType("checkpoint_pickle")
_pickle.Unpickler = _Unpickler
_pickle.load = pickle.load


def load_checkpoint(path, device="cuda"):
    ckpt = torch.load(path, map_location="cpu", weights_only=False, pickle_module=_pickle)
    sd = ckpt["model_state"]
    model = RosVLA(
        img_dim=sd["img_proj.0.weight"].shape[1],
        tel_dim=sd["tel_proj.0.weight"].shape[1],
        latent_dim=sd["flow.inp.weight"].shape[1],
        horizon=sd["head.lin.weight"].shape[0],
    )
    model.load_state_dict(sd)
    return model.to(device).eval(), ckpt
