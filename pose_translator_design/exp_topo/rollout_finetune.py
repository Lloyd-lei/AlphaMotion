"""ROLLOUT FINETUNE — train the prior on states it produced itself.

    rollout_finetune.py --out RUNS/BRIDGE_B7_rollout_0811

Diagnosis being treated (PHASE_SPACE_B6_verdict_0811): masked-token CE is a
per-frame likelihood, so nothing penalises slow drift of a whole trajectory.
B3 froze; B6 (goal_drop 0.5) drifts — return-map slope +0.90 where the corpus
is -0.70.

The fix is exposure-bias correction, done in the model's own state space:
  1 take a real window of two chunks
  2 roll chunk 1 forward with the goal DROPPED, no grad
  3 decode the model's own chunk-1 output and quantise its last two frames —
    this is the state the model will actually be in at deployment
  4 condition chunk 2 on THAT state, and train masked CE against the REAL
    chunk-2 tokens
So the model is asked: from where you actually end up, predict what really
comes next. Drift is penalised because a drifted state must still map to the
true continuation.
Half the steps stay ordinary (real starts) to keep the prior anchored.

Acceptance is the phase-space ruler, not the loss: return-map slope must move
from +0.90 toward the corpus's -0.70, portrait amplitude must grow.
"""
from __future__ import annotations
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("V5_AMASS", "/var/tmp/alphamotion/topo_exp/cache_xwbc_amass_full")
from bridge_a import CodeData, load_bridge_a                          # noqa: E402
from bridge_b import BridgeB, tokens_of, V5                           # noqa: E402

RUNS = "/var/tmp/alphamotion/topo_exp/runs"


def sample_batch(mb, ep, nf, dev, goal_drop=True, temp=1.0, gen=None):
    B = ep.shape[0]
    gd = torch.ones(B, dtype=torch.bool, device=dev) if goal_drop else None
    tok = torch.zeros(B, 32, dtype=torch.long, device=dev)
    mask = torch.ones(B, 32, dtype=torch.bool, device=dev)
    ar = torch.arange(B, device=dev)
    for _ in range(32):
        logits = mb(tok, mask, ep, nf, goal_drop=gd)
        probs = F.softmax(logits / temp, -1)
        conf = probs.max(-1).values.masked_fill(~mask, -1)
        pos = conf.argmax(-1)
        tok[ar, pos] = torch.multinomial(probs[ar, pos], 1, generator=gen)[:, 0]
        mask[ar, pos] = False
    return tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage_a", default="BRIDGE_A3_prod_0810")
    ap.add_argument("--init", default="BRIDGE_B6_fwd_0811")
    ap.add_argument("--spatial", default="PROD_c1_91k_dim20_tail_0810")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--chunk", type=int, default=45)
    ap.add_argument("--p_rollout", type=float, default=0.5)
    ap.add_argument("--goal_drop", type=float, default=0.5)
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev, C = args.device, args.chunk
    torch.manual_seed(0)
    os.makedirs(args.out, exist_ok=True)

    ma, acfg = load_bridge_a(f"{RUNS}/{args.stage_a}", dev)
    for p in ma.parameters():
        p.requires_grad_(False)
    data = CodeData(args.spatial, use_root=bool(acfg.get("use_root")))
    bcfg = json.load(open(f"{RUNS}/{args.init}/config.json"))
    m = BridgeB(ep_dim=bcfg.get("ep_dim")).to(dev)
    m.load_state_dict(torch.load(f"{RUNS}/{args.init}/model.pt", map_location=dev))
    print(f"warm-start <- {args.init}", flush=True)
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=1e-2)
    g = np.random.default_rng(0)
    lv = (torch.arange(9, device=dev).float() - 4.0) / 3.5
    W = 5 ** torch.arange(6, device=dev)
    log = []
    t0 = time.time()
    SPAN = 2 * C + 2

    def window(split, bs):
        c = data.sample(bs, g, split, n_lo=SPAN, n_hi=SPAN)
        return (c[0] if isinstance(c, tuple) else c).to(dev)

    def own_state(c1):
        """roll chunk 1 forward with the goal dropped -> the model's own
        end state, quantised back to codes [B,2,slots,dims]."""
        ep = ma.endpoints(torch.stack([c1[:, 0], c1[:, 1],
                                       c1[:, 0], c1[:, 1]], 1))
        nf = torch.full((c1.shape[0],), float(C), device=dev)
        tok = sample_batch(m, ep, nf, dev, goal_drop=True)
        z = ma.fsq.from_ints(((tok[..., None] // W) % 5)[None].float())
        q = (F.softmax(ma.decode(z, ep, C), -1) * lv).sum(-1)      # [B,C,S,D]
        return (q[:, -2:] * 3.5 + 4.0).round().clamp(0, 8).long()

    def make_mask(B, gg):
        mode = gg.random()
        r = torch.cos(torch.rand(B, 1, device=dev) * (np.pi / 2))
        if mode < 0.2:
            mk = torch.ones(B, 32, dtype=torch.bool, device=dev)
        elif mode < 0.6:
            mk = torch.rand(B, 32, device=dev) < r
        elif mode < 0.8:
            cut = (32 * (1 - r)).long().clamp(1, 31)
            mk = torch.arange(32, device=dev)[None] >= cut
        else:
            cut = (32 * r).long().clamp(1, 31)
            mk = torch.arange(32, device=dev)[None] < cut
        mk[mk.sum(1) == 0, 0] = True
        return mk

    for step in range(args.steps):
        c = window("train", args.bs)
        use_rollout = g.random() < args.p_rollout
        tgt = c[:, C:C + C]                       # the REAL second chunk
        tok_t, ep_real = tokens_of(ma, tgt)
        if use_rollout:
            with torch.no_grad():
                st = own_state(c[:, :C + 2])      # the model's OWN end state
            ep = ma.endpoints(torch.stack([st[:, 0], st[:, 1],
                                           tgt[:, -2], tgt[:, -1]], 1))
        else:
            ep = ep_real
        nf = torch.full((args.bs,), float(C), device=dev)
        mk = make_mask(args.bs, g)
        gd = torch.rand(args.bs, device=dev) < args.goal_drop
        logits = m(tok_t, mk, ep, nf, goal_drop=gd)
        loss = F.cross_entropy(logits[mk], tok_t[mk])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if step % 200 == 0:
            print(f"step {step} loss {loss.item():.4f} "
                  f"{'ROLLOUT' if use_rollout else 'real'} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        if step and step % args.eval_every == 0 or step == args.steps - 1:
            m.eval()
            with torch.no_grad():
                gv = np.random.default_rng(1)
                ces = []
                for _ in range(6):
                    cv = window("val", args.bs)
                    tk, er = tokens_of(ma, cv[:, C:C + C])
                    nfv = torch.full((args.bs,), float(C), device=dev)
                    mkv = torch.ones_like(tk, dtype=torch.bool)
                    ces.append(float(F.cross_entropy(
                        m(tk, mkv, er, nfv)[mkv], tk[mkv])))
            ev = {"step": step, "val_bridge_ce": round(float(np.mean(ces)), 4),
                  "val_ppl": round(float(np.exp(np.mean(ces))), 1)}
            log.append(ev)
            print("EVAL", json.dumps(ev), flush=True)
            json.dump(log, open(f"{args.out}/eval_log.json", "w"))
            m.train()
    torch.save(m.state_dict(), f"{args.out}/model.pt")
    json.dump(vars(args) | {"arch": "bridge_b_rollout",
                            "ep_dim": bcfg.get("ep_dim")},
              open(f"{args.out}/config.json", "w"), indent=2)
    print(f"DONE {args.out}", flush=True)


if __name__ == "__main__":
    main()
