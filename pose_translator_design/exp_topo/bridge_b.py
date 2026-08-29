"""BRIDGE Stage B — the "player": token prior over frozen Stage-A tokens.

    bridge_b.py --out RUNS/BRIDGE_B_pilot_0810

P(z_1..z_32 | start(q,v), goal(q,v), frame budget n)  — the preference table.

  * tokens come from the FROZEN Stage-A encoder on the fly (no token cache)
  * each token = one JOINT class in the 5^6 = 15,625 FSQ lattice (no factorized
    heads — within-token coupling preserved, agreed 0810)
  * mask curriculum (pilot): BRIDGE 70% (all 32 masked — the main use), SPARSE 30%
    (random subset masked). Time-direction modes need time-local tokens (Stage-A v2).
  * goal dropout 10% (weak forward mode + CFG knob)
  * inference: one-by-one confidence-ordered unmasking = exact chain rule (lloyd's
    "逐个吐"), never argmax-all-at-once.

Loss: CE on masked positions only. Val: masked-token top-1 acc + NLL, bridge mode.
"""
from __future__ import annotations
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bridge_a import BridgeA, CodeData, fourier_time, load_bridge_a                 # noqa: E402

RUNS = "/var/tmp/alphamotion/topo_exp/runs"
V5 = 5 ** 6                                                          # 15,625


class BridgeB(nn.Module):
    def __init__(self, d=256, n_tok=32, layers=6, n_heads=8, n_freq=8,
                 ep_dim=None):
        super().__init__()
        self.n_tok = n_tok
        self.tok_emb = nn.Embedding(V5 + 1, d)          # +1 = the <mask> id
        self.pos_id = nn.Parameter(torch.randn(n_tok, d) * 0.02)
        self.cond_in = nn.Linear(ep_dim or d, d)        # start/goal features (A's d!)
        self.cond_id = nn.Parameter(torch.randn(3, d) * 0.02)   # start/goal/n
        self.n_in = nn.Linear(2 * n_freq, d)
        el = nn.TransformerEncoderLayer(d, n_heads, 4 * d, batch_first=True,
                                        norm_first=True, dropout=0.0)
        self.trunk = nn.TransformerEncoder(el, layers)
        self.head = nn.Linear(d, V5)

    def forward(self, tok, mask, ep, n_frames, goal_drop=None):
        """tok [B,32] long (0..15624), mask [B,32] bool (True=hidden),
        ep [B,2,d] start/goal features, n_frames [B] float."""
        B = tok.shape[0]
        t = torch.where(mask, torch.full_like(tok, V5), tok)
        h = self.tok_emb(t) + self.pos_id[None]
        nf = fourier_time((n_frames[:, None] / 120.0).clamp(0, 1))[:, 0]
        cond = torch.stack([self.cond_in(ep[:, 0]),
                            self.cond_in(ep[:, 1]),
                            self.n_in(nf)], 1) + self.cond_id[None]
        if goal_drop is not None:
            cond = cond.clone()
            cond[goal_drop, 1] = 0.0
        h = self.trunk(torch.cat([cond, h], 1))[:, 3:]
        return self.head(h)


def tokens_of(ma, c, masked=False):
    """frozen Stage-A encode -> joint token ids [B,32] (0..15624).

    CONTRACT (audited 0812, runs/CONTRACT_AUDIT_0812.txt). bridge_a.encode()
    applies a time-local attention mask; this function historically did not.
    The two streams agree on 0.15% of tokens and B3 masked-CE differs by 15.2
    nats between them, so they are NOT interchangeable.
      masked=False  the pre-0812 path. B3/B6/B7/B8 and every eval that compares
                    against them MUST use this, or the numbers are nonsense.
      masked=True   bridge_a.encode's own contract, for contract-correct
                    retraining (B9 onwards).
    The default stays False so that existing checkpoints keep reproducing; flip
    it only together with the checkpoints that were trained on it."""
    with torch.no_grad():
        h = ma.embed_frames(c)
        B, T = h.shape[0], h.shape[1]
        seq = torch.cat([ma.q[None].expand(B, -1, -1), h], 1)
        am = None
        if masked and getattr(ma, "time_local", False):
            K, L = ma.n_content, ma.n_content + T
            am = torch.zeros(L, L, dtype=torch.bool, device=h.device)
            b = np.round(np.linspace(0, T, K + 1)).astype(int)
            for i in range(K):
                am[i, K:] = True
                am[i, K + b[i]:K + max(b[i + 1], b[i] + 1)] = False
                am[i, :K] = False
        out = ma.enc(seq, mask=am)[:, :ma.n_content]
        ints = ma.fsq.ints(ma.fsq.features(out))[0]      # [B,32,6] 0..4
        w = (5 ** torch.arange(6, device=ints.device))
        return (ints * w).sum(-1), ma.endpoints(c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage_a", default="BRIDGE_A_pilot_0810")
    ap.add_argument("--spatial", default="PROD_c1_91k_dim20_tail_0810")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=400)
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--curriculum", default="maskgit",
                    choices=("v1", "maskgit"),
                    help="v1 = bridge0.7/sparse0.3 (marginal-collapse bug); "
                         "maskgit = variable-ratio cos schedule x mode-sampled "
                         "patterns incl. fwd/bwd (needs time-local Stage A)")
    ap.add_argument("--seed", type=int, default=0)
    # phase-space verdict 0811: with goal_drop=0.10 the goal-free (autonomous)
    # flow collapses to a fixed point — the model interpolates but does not
    # generate. Raising the rate makes forward mode a trained regime, not a
    # side effect of a CFG knob.
    ap.add_argument("--goal_drop", type=float, default=0.10)
    ap.add_argument("--masked_tokens", action="store_true",
                    help="use bridge_a.encode's own time-local contract "
                         "(B9 onwards); omit to reproduce B3-era tokens")
    ap.add_argument("--init", default=None, help="warm-start from a run dir")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    ma, acfg = load_bridge_a(f"{RUNS}/{args.stage_a}", dev)
    a_root = bool(acfg.get("use_root", False))
    data = CodeData(args.spatial, use_root=a_root)
    for p in ma.parameters():
        p.requires_grad_(False)
    m = BridgeB(ep_dim=ma.d).to(dev)
    if args.init:
        m.load_state_dict(torch.load(f"{RUNS}/{args.init}/model.pt",
                                     map_location=dev))
        print(f"warm-start <- {args.init}", flush=True)
    n_par = sum(p.numel() for p in m.parameters())
    print(f"params {n_par/1e6:.2f}M (head {V5} classes)", flush=True)
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, s / args.warmup))
    g = np.random.default_rng(args.seed)
    log = []
    t0 = time.time()

    def batch(split, gg):
        c, n = data.sample(args.bs, gg, split)
        c = c.to(dev)
        tok, ep = tokens_of(ma, c, masked=args.masked_tokens)
        return tok, ep, torch.full((args.bs,), float(c.shape[1]), device=dev)

    for step in range(args.steps):
        tok, ep, nf = batch("train", g)
        B = tok.shape[0]
        mode = g.random()
        if args.curriculum == "v1":
            if mode < 0.7:
                mask = torch.ones(B, 32, dtype=torch.bool, device=dev)
            else:
                keep = torch.rand(B, 32, device=dev) < g.uniform(0.1, 0.9)
                mask = ~keep
                mask[mask.sum(1) == 0, 0] = True
        else:
            # MaskGIT cure for marginal collapse: per-sample ratio r = cos(pi/2*u),
            # u~U(0,1) -> dense coverage of PARTIAL visibility, so inter-token
            # conditionals actually receive gradient. Patterns ride on top; fwd/bwd
            # are meaningful because A2 tokens are time-local (token i = segment i).
            r = torch.cos(torch.rand(B, 1, device=dev) * (np.pi / 2))
            if mode < 0.2:                                # bridge: all hidden
                mask = torch.ones(B, 32, dtype=torch.bool, device=dev)
            elif mode < 0.6:                              # sparse, variable ratio
                mask = torch.rand(B, 32, device=dev) < r
            elif mode < 0.8:                              # forward: hide the future
                cut = (32 * (1 - r)).long().clamp(1, 31)
                ar = torch.arange(32, device=dev)[None]
                mask = ar >= cut
            else:                                         # backward: hide the past
                cut = (32 * r).long().clamp(1, 31)
                ar = torch.arange(32, device=dev)[None]
                mask = ar < cut
            mask[mask.sum(1) == 0, 0] = True
        gd = torch.rand(B, device=dev) < args.goal_drop   # goal dropout
        logits = m(tok, mask, ep, nf, goal_drop=gd)
        ce = F.cross_entropy(logits[mask], tok[mask])
        opt.zero_grad(set_to_none=True)
        ce.backward()
        nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 100 == 0:
            acc = (logits[mask].argmax(-1) == tok[mask]).float().mean()
            print(f"step {step} ce {ce.item():.4f} acc {acc.item():.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        if step and step % args.eval_every == 0 or step == args.steps - 1:
            m.eval()
            with torch.no_grad():
                gv = np.random.default_rng(1)
                ces, accs = [], []
                for _ in range(8):
                    tokv, epv, nfv = batch("val", gv)
                    maskv = torch.ones_like(tokv, dtype=torch.bool)
                    lo = m(tokv, maskv, epv, nfv)
                    ces.append(F.cross_entropy(lo[maskv], tokv[maskv]).item())
                    accs.append((lo[maskv].argmax(-1) == tokv[maskv])
                                .float().mean().item())
            ev = {"step": step, "val_bridge_ce": round(float(np.mean(ces)), 4),
                  "val_bridge_acc": round(float(np.mean(accs)), 4),
                  "val_ppl": round(float(np.exp(np.mean(ces))), 1)}
            log.append(ev)
            print("EVAL", json.dumps(ev), flush=True)
            json.dump(log, open(f"{args.out}/eval_log.json", "w"))
            m.train()
    torch.save(m.state_dict(), f"{args.out}/model.pt")
    json.dump(vars(args) | {"arch": "bridge_b", "params": n_par, "vocab": V5,
                            "curriculum": args.curriculum, "ep_dim": ma.d,
                            "token_contract": "masked" if args.masked_tokens
                                              else "legacy_unmasked"},
              open(f"{args.out}/config.json", "w"), indent=2)
    print(f"DONE {args.out}", flush=True)


if __name__ == "__main__":
    main()
