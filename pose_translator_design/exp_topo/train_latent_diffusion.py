"""E1-B: chunked autoregressive LATENT DIFFUSION over the frozen tokenizer's
continuous pre-round features — the "discrete is the interface, continuous is the
compute" arm of the E1 duel. Its unique selling point vs the discrete AR-LM:
every denoising step is differentiable => the SE3 gradient-guidance path.

Data: tokenize_cache output ({emb}_f.pt [N,2,K,f] float16 pre-round features +
sequences.json). Per-frame latent = flattened [2*K*f] (=384 for K=32,f=6).
Model: causal next-chunk diffusion — context = 7 chunks of 8 frames (as clean
latents), target = 8th chunk, cosine-schedule DDPM, epsilon-prediction,
transformer denoiser. Window 64 frames total = E1-A's window (fair duel).

Self-contained DDPM (no ardy import): the decision experiment needs minimal
moving parts; ARDY component reuse happens at integration time, not here.

Usage: python train_latent_diffusion.py --tokens {run}/tokens_seq10k --out {run_dir}
"""
from __future__ import annotations
import os, sys, json, math, time, argparse
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(__file__))

CHUNK = 8


def cosine_alphas(T=1000, s=0.008):
    t = torch.linspace(0, T, T + 1)
    f = torch.cos((t / T + s) / (1 + s) * math.pi / 2) ** 2
    ab = (f / f[0]).clamp(1e-5, 1.0)
    return ab                                              # \bar{alpha}_t, [T+1]


class ChunkDenoiser(nn.Module):
    """Context chunks (clean) + noisy target chunk -> predict target noise.
    Frame-level tokens; causal structure via segment embeddings (ctx vs target)."""
    def __init__(self, in_dim, d_model=384, layers=6, heads=8, n_frames=64):
        super().__init__()
        self.inp = nn.Linear(in_dim, d_model)
        self.pos = nn.Parameter(torch.randn(n_frames, d_model) * 0.02)
        self.seg = nn.Parameter(torch.randn(2, d_model) * 0.02)   # 0=context, 1=noisy target
        self.tmb = nn.Sequential(nn.Linear(1, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        layer = nn.TransformerEncoderLayer(d_model, heads, 4 * d_model, batch_first=True,
                                           norm_first=True, activation="gelu", dropout=0.1)
        self.tr = nn.TransformerEncoder(layer, layers)
        self.out = nn.Linear(d_model, in_dim)

    def forward(self, ctx, x_t, t01):
        # ctx [B,Wc,D] clean; x_t [B,CHUNK,D] noisy; t01 [B] in (0,1]
        B = ctx.shape[0]
        h = torch.cat([self.inp(ctx) + self.seg[0], self.inp(x_t) + self.seg[1]], 1)
        h = h + self.pos[: h.shape[1]][None]
        h = h + self.tmb(t01[:, None, None].expand(-1, h.shape[1], 1))
        h = self.tr(h)
        return self.out(h[:, -x_t.shape[1]:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--emb", default="soma77")
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--d_model", type=int, default=384)
    ap.add_argument("--window", type=int, default=64)         # frames incl. target chunk
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda"; torch.manual_seed(0)

    F = torch.load(f"{args.tokens}/{args.emb}_f.pt").float()   # [N,2,K,f]
    N = F.shape[0]; D = F.shape[1] * F.shape[2] * F.shape[3]
    X = F.reshape(N, D)
    mu, sd = X.mean(0), X.std(0).clamp(min=1e-4)
    X = ((X - mu) / sd).to(device)
    torch.save({"mu": mu, "sd": sd}, f"{args.out}/norm.pt")
    seqs = json.load(open(f"{args.tokens}/sequences.json"))

    rng = np.random.default_rng(0)
    order = rng.permutation(len(seqs))
    nval = max(8, len(seqs) // 10)
    val_seqs = [seqs[i] for i in order[:nval] if seqs[i]["len"] > args.window]
    tr_seqs = [seqs[i] for i in order[nval:] if seqs[i]["len"] > args.window]
    print(f"D={D} seqs train={len(tr_seqs)} val={len(val_seqs)}")

    ab = cosine_alphas(args.T).to(device)
    model = ChunkDenoiser(D, args.d_model, args.layers, n_frames=args.window).to(device)
    print(f"params {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    gen = np.random.default_rng(1)

    def sample_windows(seq_list, bs, g):
        rows = []
        si = g.integers(0, len(seq_list), bs)
        for i in si:
            s = seq_list[i]
            t0 = g.integers(0, s["len"] - args.window + 1)
            rows.append(np.arange(s["start"] + t0, s["start"] + t0 + args.window))
        return torch.as_tensor(np.stack(rows), device=device)

    def batch_loss(seq_list, g):
        w = sample_windows(seq_list, args.bs, g)
        x = X[w]                                              # [B,W,D]
        ctx, tgt = x[:, :-CHUNK], x[:, -CHUNK:]
        t = torch.randint(1, args.T + 1, (x.shape[0],), device=device)
        a = ab[t][:, None, None]
        eps = torch.randn_like(tgt)
        x_t = a.sqrt() * tgt + (1 - a).sqrt() * eps
        pred = model(ctx, x_t, t.float() / args.T)
        return nn.functional.mse_loss(pred, eps)

    log = []; t0 = time.time(); best_val = float("inf")
    for step in range(args.steps):
        loss = batch_loss(tr_seqs, gen)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            model.eval()
            with torch.no_grad():
                vg = np.random.default_rng(2)
                v = float(np.mean([batch_loss(val_seqs, vg).item() for _ in range(8)]))
                tr = float(np.mean([batch_loss(tr_seqs, np.random.default_rng(3)).item() for _ in range(8)]))
            ev = {"step": step + 1, "val_eps_mse": round(v, 5), "train_eps_mse": round(tr, 5),
                  "sec": round(time.time() - t0)}
            model.train(); log.append(ev)
            if v < best_val:
                best_val = v
                torch.save(model.state_dict(), f"{args.out}/model_best.pt"); ev["best"] = True
            print("EVAL", json.dumps(ev)); json.dump(log, open(f"{args.out}/eval_log.json", "w"), indent=2)
    torch.save(model.state_dict(), f"{args.out}/model.pt")
    json.dump({**vars(args), "D": D, "chunk": CHUNK}, open(f"{args.out}/config.json", "w"), indent=2)
    print("DONE", args.out)


if __name__ == "__main__":
    main()
