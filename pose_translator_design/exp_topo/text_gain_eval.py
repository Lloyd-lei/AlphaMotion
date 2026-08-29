"""Paired text-gain duel: does the prompt reduce the NLL of a held-out clip?

    text_gain_eval.py --n_batches 120

The training logs bounce between 0.047 and 0.072 nats because each in-loop eval
uses 8 batches of 48. That spread is as large as the effect, so the logs cannot
settle a B4-vs-B5 comparison.  This runs both arms over the SAME val windows in
the SAME order (one RNG, replayed), so the comparison is paired and the
difference gets a bootstrap CI over windows.

Reported per arm, on captioned val_amass, full mask, goal dropped:
    ce_null - ce_text   in nats/token, mean and 95% CI
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("V5_AMASS", "/var/tmp/alphamotion/topo_exp/cache_xwbc_amass_full")
from bridge_a import load_bridge_a                                   # noqa: E402
from bridge_b import tokens_of, V5                                   # noqa: E402
from bridge_b5 import BridgeB5, CaptionedWindows, Critic             # noqa: E402

RUNS = "/var/tmp/alphamotion/topo_exp/runs"
TA = "/var/tmp/alphamotion/topo_exp/text_align"


def per_window_ce(logits, tok):
    """CE averaged within each window -> [B], so the bootstrap is over windows."""
    ce = F.cross_entropy(logits.reshape(-1, V5), tok.reshape(-1),
                         reduction="none").reshape(tok.shape)
    return ce.mean(1)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="BRIDGE_B4v5_fixed_0812:b4,"
                                      "BRIDGE_B5_xattn_0814:b5")
    ap.add_argument("--stage_a", default="BRIDGE_A3_prod_0810")
    ap.add_argument("--spatial", default="PROD_c1_91k_dim20_tail_0810")
    ap.add_argument("--critic", default="TEXT_ALIGN_v3_fixed_0812")
    ap.add_argument("--n_batches", type=int, default=120)
    ap.add_argument("--bs", type=int, default=48)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device

    ma, acfg = load_bridge_a(f"{RUNS}/{args.stage_a}", dev)
    seq = torch.load(f"{TA}/caption_seq.pt", map_location="cpu")
    keyidx = {k: i for i, k in enumerate(seq["keys"])}
    pooled_all = torch.load(f"{TA}/caption_emb.pt", map_location="cpu")
    pool = torch.stack([pooled_all[k] for k in seq["keys"]])
    va = CaptionedWindows("val_amass", args.spatial,
                          bool(acfg.get("use_root", False)), seq, keyidx, pool)
    critic = Critic(args.critic, dev)
    W = 5 ** torch.arange(6, device=dev)

    # replay the identical window stream for every arm
    batches = []
    g = np.random.default_rng(args.seed)
    for _ in range(args.n_batches):
        c, s_, l_, pv = va.batch(args.bs, g)
        batches.append((c, s_, l_, pv))
    print(f"val windows {args.n_batches * args.bs} "
          f"({len(va.items)} captioned clips)", flush=True)

    out = {}
    for spec in args.arms.split(","):
        run, arch = spec.split(":")
        if arch == "b5":
            m = BridgeB5(ep_dim=ma.d).to(dev)
        else:
            from bridge_b4 import BridgeB4
            m = BridgeB4(ep_dim=ma.d).to(dev)
        m.load_state_dict(torch.load(f"{RUNS}/{run}/model.pt",
                                     map_location=dev)); m.eval()
        gains = []
        for c, s_, l_, pv in batches:
            c, s_, l_, pv = c.to(dev), s_.to(dev), l_.to(dev), pv.to(dev)
            tok, ep = tokens_of(ma, c)
            nf = torch.full((c.shape[0],), float(c.shape[1]), device=dev)
            mk = torch.ones_like(tok, dtype=torch.bool)
            gd = torch.ones(c.shape[0], dtype=torch.bool, device=dev)   # no goal
            if arch == "b5":
                z = torch.zeros(c.shape[0], dtype=torch.bool, device=dev)
                lo_t = m(tok, mk, ep, nf, s_, l_, goal_drop=gd, text_drop=z)
                lo_0 = m(tok, mk, ep, nf, s_, l_, goal_drop=gd, text_drop=~z)
            else:
                tx = (critic.tc.codes(pv).long() * W).sum(-1)
                nul = torch.full_like(tx, V5)
                lo_t = m(tok, mk, ep, nf, goal_drop=gd, text=tx)
                lo_0 = m(tok, mk, ep, nf, goal_drop=gd, text=nul)
            gains.append((per_window_ce(lo_0, tok) -
                          per_window_ce(lo_t, tok)).cpu().numpy())
        gv = np.concatenate(gains)
        bs_ = np.array([gv[np.random.default_rng(k).integers(0, len(gv), len(gv))]
                        .mean() for k in range(2000)])
        lo95, hi95 = np.percentile(bs_, [2.5, 97.5])
        pos = float((gv > 0).mean())
        out[run] = {"arch": arch, "text_gain": float(gv.mean()),
                    "ci95": [float(lo95), float(hi95)],
                    "frac_windows_helped": pos, "n": int(len(gv))}
        print(f"{run:28s} [{arch}] text_gain {gv.mean():+.4f} nats  "
              f"95%CI [{lo95:+.4f}, {hi95:+.4f}]  "
              f"帮到 {pos*100:.1f}% 的窗口", flush=True)
        del m
        torch.cuda.empty_cache()
    json.dump(out, open(f"{RUNS}/TEXT_GAIN_DUEL_0814.json", "w"), indent=2)
    ks = list(out)
    if len(ks) == 2:
        d = out[ks[1]]["text_gain"] - out[ks[0]]["text_gain"]
        print(f"\n差值 {ks[1]} - {ks[0]} = {d:+.4f} nats "
              f"({d / max(out[ks[0]]['text_gain'], 1e-9):+.1%})")
    print(f"-> {RUNS}/TEXT_GAIN_DUEL_0814.json", flush=True)


if __name__ == "__main__":
    main()
