"""Causal AR-LM over frozen-tokenizer FSQ codes. Two uses:
  1. small config (default)  = AR-predictability metric for the E0a codebook lock:
     held-out bits/frame vs the codebook's raw bit budget. A codebook that
     reconstructs well but yields an unpredictable token stream loses here.
  2. big config (--layers 6 --d_model 384) = E1-A discrete-AR arm.

Input per frame is a function of the DISCRETE code only (dequantized grid values),
matching what a rollout would see. Output: factorized per-position softmax over
levels; loss = sum CE; metric bits/frame = CE_nats/ln(2) summed over positions.

Usage:
  python train_ar_lm.py --tokens {run}/tokens_seq --out {run}/ar_lm
"""
from __future__ import annotations
import os, sys, json, math, time, argparse
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(__file__))


def load_tokens(tokens_dir, emb):
    meta = json.load(open(f"{tokens_dir}/meta.json"))
    ints = torch.load(f"{tokens_dir}/{emb}_ints.pt").long()      # [N,2,S,K,f]
    seqs = json.load(open(f"{tokens_dir}/sequences.json"))
    levels = torch.tensor(meta["levels"], dtype=torch.float32)
    return ints, seqs, levels, meta


def dequant(ints, levels):
    """Discrete-only input rep: (int - ceil(half)) / half per dim, flattened per frame."""
    half = (levels - 1) / 2
    x = (ints.float() - torch.ceil(half)) / half                  # broadcast over trailing f dim
    return x.reshape(x.shape[0], -1)                              # [N, P]


class ARLM(nn.Module):
    def __init__(self, in_dim, n_pos, n_lvl, d_model=256, layers=4, heads=8, max_w=256, dropout=0.0):
        super().__init__()
        self.inp = nn.Linear(in_dim, d_model)
        self.pos = nn.Parameter(torch.randn(max_w, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, heads, 4 * d_model, batch_first=True,
                                           norm_first=True, activation="gelu", dropout=dropout)
        self.tr = nn.TransformerEncoder(layer, layers)
        self.head = nn.Linear(d_model, n_pos * n_lvl)
        self.n_pos, self.n_lvl = n_pos, n_lvl

    def forward(self, x):                                         # x [B,W,P] -> logits [B,W,P,L]
        B, W, _ = x.shape
        h = self.inp(x) + self.pos[:W][None]
        mask = torch.triu(torch.ones(W, W, device=x.device, dtype=torch.bool), 1)
        h = self.tr(h, mask=mask)
        return self.head(h).view(B, W, self.n_pos, self.n_lvl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", required=True, help="{run}/tokens_seq dir")
    ap.add_argument("--emb", default="soma77")
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda"; torch.manual_seed(0)

    ints, seqs, levels, meta = load_tokens(args.tokens, args.emb)
    N = ints.shape[0]
    X = dequant(ints, levels).to(device)                          # [N,P] input rep
    Y = ints.reshape(N, -1).to(device)                            # [N,P] targets
    # vocab = observed max + 1: fp32 tanh saturates to exactly 1.0, adding one extra
    # saturation level beyond the nominal grid on even-level dims
    P = X.shape[1]; L = int(Y.max().item()) + 1
    raw_bits = float(np.sum(np.log2(meta["levels"])) * meta["stages"] * meta["n_latent"] * 2)

    # sequence-level 90/10 split; windows never cross sequence boundaries
    rng = np.random.default_rng(0)
    order = rng.permutation(len(seqs))
    nval = max(8, len(seqs) // 10)
    val_seqs = [seqs[i] for i in order[:nval]]
    tr_seqs = [s for s in (seqs[i] for i in order[nval:]) if s["len"] > args.window]
    val_seqs = [s for s in val_seqs if s["len"] > args.window]
    print(f"seqs train={len(tr_seqs)} val={len(val_seqs)}  P={P} L={L} raw_bits/frame={raw_bits:.0f}")

    def sample_windows(seq_list, bs, gen):
        starts, rows = [], []
        si = gen.integers(0, len(seq_list), bs)
        for i in si:
            s = seq_list[i]
            t0 = gen.integers(0, s["len"] - args.window)
            rows.append(np.arange(s["start"] + t0, s["start"] + t0 + args.window + 1))
        return torch.as_tensor(np.stack(rows), device=device)

    model = ARLM(P, P, L, d_model=args.d_model, layers=args.layers, max_w=args.window + 1,
                 dropout=args.dropout).to(device)
    print(f"params {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    gen = np.random.default_rng(1)
    vgen = np.random.default_rng(2)

    def bits_per_frame(seq_list, gen, n_batches=8):
        tot, cnt = 0.0, 0
        with torch.no_grad():
            for _ in range(n_batches):
                w = sample_windows(seq_list, args.bs, gen)
                logits = model(X[w[:, :-1]])
                ce = nn.functional.cross_entropy(logits.reshape(-1, L), Y[w[:, 1:]].reshape(-1),
                                                 reduction="sum")
                tot += float(ce); cnt += w.shape[0] * args.window
        return tot / cnt / math.log(2)                            # bits per frame (sum over P inside ce)

    log = []; t0 = time.time(); best_val = float("inf")
    for step in range(args.steps):
        w = sample_windows(tr_seqs, args.bs, gen)
        logits = model(X[w[:, :-1]])                              # predict frame t+1 codes
        loss = nn.functional.cross_entropy(logits.reshape(-1, L), Y[w[:, 1:]].reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            model.eval()
            ev = {"step": step + 1,
                  "val_bits_per_frame": round(bits_per_frame(val_seqs, np.random.default_rng(2)), 1),
                  "train_bits_per_frame": round(bits_per_frame(tr_seqs, np.random.default_rng(3)), 1),
                  "raw_bits_per_frame": round(raw_bits, 1),
                  "sec": round(time.time() - t0)}
            model.train(); log.append(ev)
            if ev["val_bits_per_frame"] < best_val:               # E1A overfit lesson: keep the best-val checkpoint
                best_val = ev["val_bits_per_frame"]
                torch.save(model.state_dict(), f"{args.out}/model_best.pt")
                ev["best"] = True
            print("EVAL", json.dumps(ev)); json.dump(log, open(f"{args.out}/eval_log.json", "w"), indent=2)
    torch.save(model.state_dict(), f"{args.out}/model.pt")
    json.dump({**vars(args), "P": P, "L": L, "raw_bits": raw_bits}, open(f"{args.out}/config.json", "w"), indent=2)
    print("DONE", args.out)


if __name__ == "__main__":
    main()
