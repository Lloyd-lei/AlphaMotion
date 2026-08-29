"""SEMIGROUP FINETUNE — train Equator-B on its own composition law. No GT.

    semigroup_finetune.py --out RUNS/EQ_B5_semigroup_0811

The law: bridging start->goal in n frames must agree with bridging
start->mid in n/2 then mid->goal in n/2, where mid is the model's OWN
midpoint.  Measured 0811: gap 2.69cm vs sampling noise 3.89cm (ratio 0.69)
— already self-consistent, so this is a polish lever.

Loss = CE of the one-hop token distribution against the two-hop tokens
(teacher = the composed path, detached), mixed with the ordinary masked CE
so the prior cannot drift.  B3 stays frozen on disk; this is a copy.

Acceptance: semigroup gap down vs B3 on the frozen probe, masked-CE (val
bridge ppl) not worse.  Every metric needs zero ground truth.
"""
from __future__ import annotations
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bridge_a import CodeData, load_bridge_a                          # noqa: E402
from bridge_b import BridgeB, tokens_of, V5                           # noqa: E402

RUNS = "/var/tmp/alphamotion/topo_exp/runs"


def sample_tokens_batch(mb, ep, nf, dev, temp=1.0, gen=None):
    """confidence-ordered sampling, BATCHED. returns [B,32] ids."""
    B = ep.shape[0]
    tok = torch.zeros(B, 32, dtype=torch.long, device=dev)
    mask = torch.ones(B, 32, dtype=torch.bool, device=dev)
    ar = torch.arange(B, device=dev)
    for _ in range(32):
        logits = mb(tok, mask, ep, nf)
        probs = F.softmax(logits / temp, -1)
        conf = probs.max(-1).values.masked_fill(~mask, -1)
        pos = conf.argmax(-1)
        pick = torch.multinomial(probs[ar, pos], 1, generator=gen)[:, 0]
        tok[ar, pos] = pick
        mask[ar, pos] = False
    return tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage_a", default="BRIDGE_A3_prod_0810")
    ap.add_argument("--init", default="BRIDGE_B3_prod_0810")
    ap.add_argument("--spatial", default="PROD_c1_91k_dim20_tail_0810")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--w_semi", type=float, default=0.5)
    ap.add_argument("--n_total", type=int, default=60)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device
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
    N, H = args.n_total, args.n_total // 2
    lv = (torch.arange(9, device=dev).float() - 4.0) / 3.5
    W = 5 ** torch.arange(6, device=dev)
    log = []
    t0 = time.time()

    def endpoints_from(c4):
        return ma.endpoints(c4)

    def semigroup_batch(split, bs):
        """returns (ep_full, ep1, ep2, mid codes) built from real windows."""
        c = data.sample(bs, g, split, n_lo=N + 2, n_hi=N + 2)
        c = c[0] if isinstance(c, tuple) else c
        c = c.to(dev)
        ep_full = endpoints_from(torch.stack(
            [c[:, 0], c[:, 1], c[:, N], c[:, N + 1]], 1))
        return c, ep_full

    for step in range(args.steps):
        c, ep_full = semigroup_batch("train", args.bs)
        nfN = torch.full((args.bs,), float(N), device=dev)
        nfH = torch.full((args.bs,), float(H), device=dev)
        with torch.no_grad():
            # teacher: the model's own two-hop composition through its midpoint
            tok1_probe = sample_tokens_batch(m, ep_full, nfN, dev)
            z = ma.fsq.from_ints(((tok1_probe[..., None] // W) % 5)[None].float())
            q = (F.softmax(ma.decode(z, ep_full, N), -1) * lv).sum(-1)
            midc = (q[:, H - 1:H + 1] * 3.5 + 4.0).round().clamp(0, 8).long()
            ep1 = endpoints_from(torch.stack(
                [c[:, 0], c[:, 1], midc[:, 0], midc[:, 1]], 1))
            ep2 = endpoints_from(torch.stack(
                [midc[:, 0], midc[:, 1], c[:, N], c[:, N + 1]], 1))
            t1 = sample_tokens_batch(m, ep1, nfH, dev)
            t2 = sample_tokens_batch(m, ep2, nfH, dev)
            # compose: first half of hop-1's tokens + second half of hop-2's
            teacher = torch.cat([t1[:, :16], t2[:, 16:]], 1)
        allmask = torch.ones(args.bs, 32, dtype=torch.bool, device=dev)
        logits = m(teacher, allmask, ep_full, nfN)
        semi = F.cross_entropy(logits.reshape(-1, V5), teacher.reshape(-1))
        # anchor: ordinary masked CE on real tokens (keeps the prior honest)
        tok_r, ep_r = tokens_of(ma, c[:, :N])
        r = torch.cos(torch.rand(args.bs, 1, device=dev) * (np.pi / 2))
        mk = torch.rand(args.bs, 32, device=dev) < r
        mk[mk.sum(1) == 0, 0] = True
        lo_r = m(tok_r, mk, ep_r, nfN)
        anchor = F.cross_entropy(lo_r[mk], tok_r[mk])
        loss = anchor + args.w_semi * semi
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if step % 100 == 0:
            print(f"step {step} anchor {anchor.item():.4f} semi {semi.item():.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        if step and step % args.eval_every == 0 or step == args.steps - 1:
            m.eval()
            with torch.no_grad():
                gv = np.random.default_rng(1)
                ces = []
                for _ in range(6):
                    cv, epv = semigroup_batch("val", args.bs)
                    tk, er = tokens_of(ma, cv[:, :N])
                    nf = torch.full((args.bs,), float(N), device=dev)
                    mkv = torch.ones_like(tk, dtype=torch.bool)
                    ces.append(F.cross_entropy(m(tk, mkv, er, nf)[mkv],
                                               tk[mkv]).item())
            ev = {"step": step, "val_bridge_ce": round(float(np.mean(ces)), 4),
                  "val_ppl": round(float(np.exp(np.mean(ces))), 1)}
            log.append(ev)
            print("EVAL", json.dumps(ev), flush=True)
            json.dump(log, open(f"{args.out}/eval_log.json", "w"))
            m.train()
    torch.save(m.state_dict(), f"{args.out}/model.pt")
    json.dump(vars(args) | {"arch": "bridge_b_semigroup",
                            "ep_dim": bcfg.get("ep_dim")},
              open(f"{args.out}/config.json", "w"), indent=2)
    print(f"DONE {args.out}", flush=True)


if __name__ == "__main__":
    main()
