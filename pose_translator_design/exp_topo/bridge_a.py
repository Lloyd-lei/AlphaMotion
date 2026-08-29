"""BRIDGE Stage A — the "pen": temporal bridge codec over Greenwich code streams.

    bridge_a.py --out RUNS/BRIDGE_A_pilot_0810   (env: ardy)

First implementation of the agreed BRIDGE STACK v1.0 (0810):
  * B-route at last: input/output are SPATIAL CODE streams (frozen Greenwich=C1),
    not key-joint geometry. CE on code levels is the reconstruction loss.
  * CLOCK grid (arc-length regrid retired 0810), variable window n ~ U[15,120],
    frame budget n conditions the decoder (clock queries at t/n).
  * 32 content tokens, FSQ 6dim x 5lvl. No speed/style token in the pilot.
  * Endpoint pass-through, code-space version: the decoder only ever predicts the
    INTERIOR frames; both cut-point frames are known conditioning and are copied
    verbatim at render time. Exact by construction (the temporal vis_residual).
  * Endpoint condition = frame embedding of the two cut frames + a discrete
    velocity feature (embedding delta to the neighbor frame) ~ (q, v).
  * Optional geometry loss through the frozen spatial decoder on a few sampled
    interior frames (soft-expected codes -> proj_out -> decode).

Data: bridge_codes/<run>/codes_pose.npy [N,128,20] int8 + full-cache sequences.json.
Val split: clip-level rng(123) 10%, same convention as KeyData.
"""
from __future__ import annotations
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v5_common as V                                                # noqa: E402
from model import FSQ                                                # noqa: E402

FULL = "/var/tmp/alphamotion/topo_exp/cache_xwbc_amass_full"
CODES = "/var/tmp/alphamotion/topo_exp/bridge_codes"


def fourier_time(u, n_freq=8):
    """u [B,T] in [0,1] -> [B,T,2*n_freq]."""
    f = (2.0 ** torch.arange(n_freq, device=u.device, dtype=u.dtype)) * torch.pi
    a = u[..., None] * f
    return torch.cat([torch.sin(a), torch.cos(a)], -1)


class BridgeA(nn.Module):
    def __init__(self, slots=128, dims=20, levels=9, d=256, n_content=32,
                 enc_layers=5, dec_layers=3, n_heads=8, n_freq=8,
                 time_local=False, use_root=False):
        super().__init__()
        self.slots, self.dims, self.levels, self.d = slots, dims, levels, d
        self.n_content = n_content
        self.time_local, self.use_root = time_local, use_root
        self.streams = 2 if use_root else 1
        # frame embedding: normalized levels -> linear (the "learned embedding of ints")
        self.femb = nn.Sequential(nn.Linear(slots * dims * self.streams, d), nn.LayerNorm(d))
        self.q = nn.Parameter(torch.randn(n_content, d) * 0.02)
        el = nn.TransformerEncoderLayer(d, n_heads, 4 * d, batch_first=True,
                                        norm_first=True, dropout=0.0)
        self.enc = nn.TransformerEncoder(el, enc_layers)
        self.fsq = FSQ(d, (5, 5, 5, 5, 5, 5))
        self.tok_id = nn.Parameter(torch.randn(n_content, d) * 0.02)
        self.ep_id = nn.Parameter(torch.randn(2, d) * 0.02)
        self.vel_in = nn.Linear(d, d)
        self.time_in = nn.Linear(2 * n_freq + 1, d)
        dl = nn.TransformerDecoderLayer(d, n_heads, 4 * d, batch_first=True,
                                        norm_first=True, dropout=0.0)
        self.dec = nn.TransformerDecoder(dl, dec_layers)
        self.head = nn.Linear(d, slots * dims * levels * self.streams)

    def embed_frames(self, c):
        """c long [B,T,slots,dims] -> [B,T,d]. C1's FSQ emits ints 0..8 (the 9th is
        the tanh-saturation grid point 4/3.5=1.143 - a REAL level in the decode
        path, model.py edge case found 0810). Map ints to their true grid values."""
        x = (c.float() - 4.0) / 3.5
        return self.femb(x.reshape(*c.shape[:2], -1))

    def encode(self, c):
        """full window codes -> straight-through z tokens [B,32,d].
        time_local (v2, 完全体): token i may only attend frames of clock segment i
        (plus all tokens) — the code becomes TIME-LOCAL, which is what gives
        Stage B's forward/backward masks their meaning."""
        h = self.embed_frames(c)
        B, T = h.shape[0], h.shape[1]
        seq = torch.cat([self.q[None].expand(B, -1, -1), h], 1)
        mask = None
        if self.time_local:
            K = self.n_content
            L = K + T
            mask = torch.zeros(L, L, dtype=torch.bool, device=h.device)
            b = np.round(np.linspace(0, T, K + 1)).astype(int)
            for i in range(K):
                mask[i, K:] = True
                mask[i, K + b[i]:K + max(b[i + 1], b[i] + 1)] = False
                mask[i, :K] = False
        out = self.enc(seq, mask=mask)[:, :self.n_content]
        z, _ = self.fsq(out)
        return z

    def endpoints(self, c):
        """[B,T,slots,dims] -> endpoint kv [B,2,d] with (q,v)-style features."""
        e = self.embed_frames(torch.stack([c[:, 0], c[:, 1], c[:, -2], c[:, -1]], 1))
        q0, v0 = e[:, 0], e[:, 1] - e[:, 0]
        q1, v1 = e[:, 3], e[:, 3] - e[:, 2]
        ep = torch.stack([q0 + self.vel_in(v0), q1 + self.vel_in(v1)], 1)
        return ep + self.ep_id[None]

    def decode(self, z, ep, T_out, pad_mask=None):
        """-> interior logits [B,T_out,slots,dims,levels] (t=0/T-1 rows are junk;
        caller never trains or renders them)."""
        B = z.shape[0]
        u = torch.linspace(0, 1, T_out, device=z.device)[None].expand(B, -1)
        tf = torch.cat([fourier_time(u), torch.full_like(u[..., None], 0.0)
                        + np.log(T_out) / 5.0], -1)
        qs = self.time_in(tf)
        kv = torch.cat([z + self.tok_id[None], ep], 1)
        h = self.dec(qs, kv)
        return self.head(h).reshape(B, T_out, self.streams * self.slots, self.dims, -1)

    def forward(self, c, pad_mask=None):
        z = self.encode(c)
        return self.decode(z, self.endpoints(c), c.shape[1]), z


class CodeData:
    def __init__(self, run, use_root=False):
        self.c = np.load(f"{CODES}/{run}/codes_pose.npy", mmap_mode="r")
        self.use_root = use_root
        if use_root:
            self.cr = np.load(f"{CODES}/{run}/codes_root_packed.npy", mmap_mode="r")
        seqs = json.load(open(f"{FULL}/sequences.json"))
        self.clips = seqs
        rng = np.random.default_rng(V.SPLIT_SEED)
        val = set(rng.permutation(len(seqs))[: len(seqs) // 10].tolist())
        self.pools = {"train": [], "val": []}
        for i, s in enumerate(seqs):
            self.pools["val" if i in val else "train"].append(
                (s["start"], s["start"] + s["len"]))

    def sample(self, bs, g, split, n_lo=15, n_hi=120):
        """-> codes [bs, n, 128, 20] long, one shared n per batch."""
        n = int(g.integers(n_lo, n_hi + 1))
        pool = [(a, b) for a, b in self.pools[split] if b - a >= n]
        out = []
        for _ in range(bs):
            a, b = pool[int(g.integers(len(pool)))]
            s = int(g.integers(a, b - n + 1))
            cp = torch.from_numpy(np.ascontiguousarray(self.c[s:s + n]))
            if self.use_root:
                pk = torch.from_numpy(np.ascontiguousarray(self.cr[s:s + n]))
                r = torch.empty(n, cp.shape[1], cp.shape[2], dtype=cp.dtype)
                r[..., 0::2] = pk & 0x0F
                r[..., 1::2] = (pk >> 4) & 0x0F
                cp = torch.cat([cp, r], 1)          # [n, 256, D] pose||root slots
            out.append(cp)
        return torch.stack(out).long(), n


def load_bridge_a(run, dev):
    """Build BridgeA FROM ITS CONFIG (d_model/layers/flags) and load weights.
    0810 lesson: every consumer hardcoding d=256 died the moment A3 went d384."""
    import json as _j
    cfg = _j.load(open(f"{run}/config.json"))
    m = BridgeA(dims=20,
                d=int(cfg.get("d_model", 256)),
                enc_layers=int(cfg.get("enc_layers", 5)),
                dec_layers=int(cfg.get("dec_layers", 3)),
                time_local=bool(cfg.get("time_local", False)),
                use_root=bool(cfg.get("use_root", False)))
    m.load_state_dict(torch.load(f"{run}/model.pt", map_location=dev))
    return m.to(dev).eval(), cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="PROD_c1_91k_dim20_tail_0810")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=400)
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--enc_layers", type=int, default=5)
    ap.add_argument("--dec_layers", type=int, default=3)
    ap.add_argument("--bs2", type=int, default=0, help="override batch size")
    ap.add_argument("--time_local", action="store_true")
    ap.add_argument("--use_root", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    data = CodeData(args.run, use_root=args.use_root)
    m = BridgeA(time_local=args.time_local, use_root=args.use_root,
                d=args.d_model, enc_layers=args.enc_layers,
                dec_layers=args.dec_layers,
                dims=np.load(f"{CODES}/{args.run}/codes_pose.npy",
                             mmap_mode="r").shape[-1]).to(dev)
    n_par = sum(p.numel() for p in m.parameters())
    print(f"params {n_par/1e6:.2f}M  train_clips {len(data.pools['train'])} "
          f"val_clips {len(data.pools['val'])}", flush=True)
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / args.warmup))
    g = np.random.default_rng(args.seed)
    log = []
    t0 = time.time()
    for step in range(args.steps):
        c, n = data.sample(args.bs, g, "train")
        c = c.to(dev)
        logits, z = m(c)
        # interior CE only: endpoints are conditioning, never predicted
        ce = F.cross_entropy(
            logits[:, 1:-1].reshape(-1, m.levels),
            c[:, 1:-1].reshape(-1), reduction="mean")
        opt.zero_grad(set_to_none=True)
        ce.backward()
        nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 100 == 0:
            acc = (logits[:, 1:-1].argmax(-1) == c[:, 1:-1]).float().mean()
            print(f"step {step} n {n} ce {ce.item():.4f} acc {acc.item():.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        if step and step % args.eval_every == 0 or step == args.steps - 1:
            m.eval()
            with torch.no_grad():
                accs, ces = [], []
                gv = np.random.default_rng(1)
                for _ in range(8):
                    cv, nv = data.sample(args.bs, gv, "val")
                    cv = cv.to(dev)
                    lo, _ = m(cv)
                    ces.append(F.cross_entropy(
                        lo[:, 1:-1].reshape(-1, m.levels),
                        cv[:, 1:-1].reshape(-1)).item())
                    accs.append((lo[:, 1:-1].argmax(-1) == cv[:, 1:-1])
                                .float().mean().item())
            ev = {"step": step, "val_ce": round(float(np.mean(ces)), 4),
                  "val_acc": round(float(np.mean(accs)), 4)}
            log.append(ev)
            print("EVAL", json.dumps(ev), flush=True)
            json.dump(log, open(f"{args.out}/eval_log.json", "w"))
            m.train()
    torch.save(m.state_dict(), f"{args.out}/model.pt")
    json.dump(vars(args) | {"arch": "bridge_a", "params": n_par,
                            "n_content": 32, "fsq": "6x5", "grid": "clock",
                            "endpoint": "interior-only CE (exact pass-through)"},
              open(f"{args.out}/config.json", "w"), indent=2)
    print(f"DONE {args.out}", flush=True)


if __name__ == "__main__":
    main()
