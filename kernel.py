import torch
import triton
import triton.language as tl

from model import timestep_embedding


@triton.jit
def flow_sample_kernel(
    z_ptr, cf_ptr, tf_ptr,
    win_ptr, bin_ptr, w1_ptr, b1_ptr, g_ptr, beta_ptr, w2_ptr, b2_ptr, wout_ptr, bout_ptr,
    hbuf_ptr, abuf_ptr,
    B, dt, eps,
    LAT: tl.constexpr, LATP: tl.constexpr, H: tl.constexpr, NB: tl.constexpr,
    STEPS: tl.constexpr, BLOCK_B: tl.constexpr, BLOCK_K: tl.constexpr, PREC: tl.constexpr,
):
    # one program = BLOCK_B rows of the batch, running every Euler step
    pid = tl.program_id(0)
    rows = pid * BLOCK_B + tl.arange(0, BLOCK_B)
    rmask = rows < B
    cols = tl.arange(0, H)
    lat = tl.arange(0, LATP)
    ks = tl.arange(0, BLOCK_K)

    zmask = rmask[:, None] & (lat[None, :] < LAT)
    z = tl.load(z_ptr + rows[:, None] * LAT + lat[None, :], mask=zmask, other=0.0)

    win = tl.load(win_ptr + lat[:, None] * H + cols[None, :])
    b_in = tl.load(bin_ptr + cols)
    b_out = tl.load(bout_ptr + lat)
    hbuf = hbuf_ptr + rows[:, None] * H
    abuf = abuf_ptr + rows[:, None] * H
    cf_row = cf_ptr + rows[:, None] * (NB * 2 * H)

    for i in range(STEPS):
        h = tl.dot(z, win, input_precision=PREC) + b_in[None, :]

        for j in range(NB):
            # h is in registers, but tl.dot needs it in K chunks, so it goes
            # through a small scratch buffer that stays in L1/L2
            tl.store(hbuf + cols[None, :], h)
            tl.debug_barrier()
            acc = tl.zeros((BLOCK_B, H), dtype=tl.float32)
            for k in range(0, H, BLOCK_K):
                x = tl.load(hbuf + k + ks[None, :])
                w = tl.load(w1_ptr + j * H * H + (k + ks)[:, None] * H + cols[None, :])
                acc += tl.dot(x, w, input_precision=PREC)
            a = acc + tl.load(b1_ptr + j * H + cols)[None, :]

            mean = tl.sum(a, axis=1) / H
            a = a - mean[:, None]
            var = tl.sum(a * a, axis=1) / H
            a = a * tl.rsqrt(var + eps)[:, None]
            a = a * tl.load(g_ptr + j * H + cols)[None, :] + tl.load(beta_ptr + j * H + cols)[None, :]

            off = j * 2 * H + cols
            scale = tl.load(cf_row + off[None, :], mask=rmask[:, None], other=0.0)
            shift = tl.load(cf_row + H + off[None, :], mask=rmask[:, None], other=0.0)
            scale += tl.load(tf_ptr + i * NB * 2 * H + off)[None, :]
            shift += tl.load(tf_ptr + i * NB * 2 * H + H + off)[None, :]
            a = a * (1 + scale) + shift
            a = a * tl.sigmoid(a)

            tl.store(abuf + cols[None, :], a)
            tl.debug_barrier()
            acc = tl.zeros((BLOCK_B, H), dtype=tl.float32)
            for k in range(0, H, BLOCK_K):
                x = tl.load(abuf + k + ks[None, :])
                w = tl.load(w2_ptr + j * H * H + (k + ks)[:, None] * H + cols[None, :])
                acc += tl.dot(x, w, input_precision=PREC)
            h = h + acc + tl.load(b2_ptr + j * H + cols)[None, :]
            tl.debug_barrier()

        tl.store(hbuf + cols[None, :], h)
        tl.debug_barrier()
        v = tl.zeros((BLOCK_B, LATP), dtype=tl.float32)
        for k in range(0, H, BLOCK_K):
            x = tl.load(hbuf + k + ks[None, :])
            w = tl.load(wout_ptr + (k + ks)[:, None] * LATP + lat[None, :])
            v += tl.dot(x, w, input_precision=PREC)
        z = z + dt * (v + b_out[None, :])
        tl.debug_barrier()

    tl.store(z_ptr + rows[:, None] * LAT + lat[None, :], z, mask=zmask)


class FusedSampler:
    """Drop-in replacement for FlowHead.sample that runs the whole Euler loop in one kernel."""

    def __init__(self, flow, block_b=16, block_k=32, num_warps=4, precision="ieee", one_launch=True):
        self.flow = flow
        self.block_b = block_b
        self.block_k = block_k
        self.num_warps = num_warps
        self.precision = precision
        self.one_launch = one_launch

        self.H = flow.hidden
        self.NB = flow.n_blocks
        self.LAT = flow.latent_dim
        self.LATP = max(16, triton.next_power_of_2(self.LAT))
        self.eps = flow.blocks[0].norm.eps

        dev = flow.inp.weight.device
        f32 = dict(device=dev, dtype=torch.float32)
        with torch.no_grad():
            self.win = torch.zeros(self.LATP, self.H, **f32)
            self.win[: self.LAT] = flow.inp.weight.t()
            self.b_in = flow.inp.bias.float().contiguous()
            self.w1 = torch.stack([b.fc1.weight.t() for b in flow.blocks]).float().contiguous()
            self.b1 = torch.stack([b.fc1.bias for b in flow.blocks]).float().contiguous()
            self.g = torch.stack([b.norm.weight for b in flow.blocks]).float().contiguous()
            self.beta = torch.stack([b.norm.bias for b in flow.blocks]).float().contiguous()
            self.w2 = torch.stack([b.fc2.weight.t() for b in flow.blocks]).float().contiguous()
            self.b2 = torch.stack([b.fc2.bias for b in flow.blocks]).float().contiguous()
            self.wout = torch.zeros(self.H, self.LATP, **f32)
            self.wout[:, : self.LAT] = flow.out.weight.t()
            self.b_out = torch.zeros(self.LATP, **f32)
            self.b_out[: self.LAT] = flow.out.bias

        self._time = {}
        self._scratch = {}

    @torch.no_grad()
    def time_film(self, steps):
        # the time FiLM only depends on the step index, not on the input,
        # so it is computed once per step count and reused forever
        if steps not in self._time:
            dt = 1.0 / steps
            s = torch.tensor([i * dt for i in range(steps)], device=self.win.device)
            emb = timestep_embedding(s, self.flow.time_dim)
            self._time[steps] = self.flow.film_time(emb).float().contiguous()
        return self._time[steps]

    def scratch(self, n_rows):
        if n_rows not in self._scratch:
            self._scratch[n_rows] = (
                torch.empty(n_rows, self.H, device=self.win.device),
                torch.empty(n_rows, self.H, device=self.win.device),
            )
        return self._scratch[n_rows]

    def launch(self, z, cf, tf, steps, dt):
        B = z.shape[0]
        grid = (triton.cdiv(B, self.block_b),)
        hbuf, abuf = self.scratch(grid[0] * self.block_b)
        flow_sample_kernel[grid](
            z, cf, tf,
            self.win, self.b_in, self.w1, self.b1, self.g, self.beta, self.w2, self.b2, self.wout, self.b_out,
            hbuf, abuf,
            B, dt, self.eps,
            LAT=self.LAT, LATP=self.LATP, H=self.H, NB=self.NB,
            STEPS=steps, BLOCK_B=self.block_b, BLOCK_K=self.block_k, PREC=self.precision,
            num_warps=self.num_warps,
        )

    @torch.no_grad()
    def __call__(self, cond, steps, noise):
        cf = self.flow.film_cond(cond).float().contiguous()
        tf = self.time_film(steps)
        z = noise.float().clone().contiguous()
        dt = 1.0 / steps
        if self.one_launch:
            self.launch(z, cf, tf, steps, dt)
        else:
            for i in range(steps):
                self.launch(z, cf, tf[i:i + 1], 1, dt)
        return z
