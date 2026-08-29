"""BRIDGE exam — the real transition: clip A ... LONG GAP ... clip B, gap invented.

    bridge_exam.py --runs T32_arclen_0809,T32_jepa_0809 --gaps 30,60

lloyd 0810: "我说的transition是两个clip之间有长时间的空白这种" — the original
continuous-control decider. There is NO GT inside the gap, so accuracy is undefined;
what is measurable is whether the invented bridge is a plausible, continuous motion:

  junction snap  |velocity| jump at the two joins (A->bridge, bridge->B), each
                 normalized by the neighboring segments' own median |v|. 1.0 = the
                 join moves like the motion around it; big = visible pop.
  bridge jerk    3rd-diff RMS inside the gap vs inside A/B (over-smooth shows here
                 as a LOW ratio, popping as high).
  bridge energy  mean |v| in gap / mean |v| in A+B. mush = low.
  arrival err    pose distance of bridge's last frame to B's first frame [cm] —
                 does the invented motion actually LAND on B's entry pose?

Tokens = interp mode (encode the straight-line baseline; no donor leakage), so the
model invents from endpoints alone — the honest continuous-control setting.
Output: runs/BRIDGE_verdict_0810.txt (+ videos via v7_bridge_mesh separately).
"""
from __future__ import annotations
import os, sys, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v5_common as V                                                # noqa: E402
import v7_bridge as B7                                               # noqa: E402
from train_v5 import load_run, to_states                             # noqa: E402

EMB = "human_smpl"
PAIRS = [("walk", "kick"), ("run", "jacks"), ("dance", "walk"), ("punch", "run")]
LA = 30


def vel(p):
    return (p[1:] - p[:-1]).norm(dim=-1).mean(-1)        # [T-1] mean over anchors


def jerk_rms(p):
    j = p[3:] - 3 * p[2:-1] + 3 * p[1:-2] - p[:-3]
    return float(j.norm(dim=-1).mean(-1).pow(2).mean().sqrt())


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="T32_arclen_0809,T32_jepa_0809")
    ap.add_argument("--labels", default="MSBA,JEPA")
    ap.add_argument("--gaps", default="30,60")
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device
    labels = args.labels.split(",")
    kd = V.KeyData(V.AMASS, [EMB], device="cpu")
    models = {lab: load_run(f"{V.RUNS}/{r}", dev)[0]
              for lab, r in zip(labels, args.runs.split(","))}
    reach = kd.reach[EMB]
    L = ["=" * 100,
         "BRIDGE EXAM — A ... GAP(invented, interp-tokens) ... B. No GT in the gap.",
         "junction snap ~1.0 = join moves like its surroundings; arrival err = lands on B?",
         "baseline row = straight-line interp (0 bits) — the floor any model must beat.",
         "=" * 100]
    for gap in [int(x) for x in args.gaps.split(",")]:
        agg = {k: {"snapA": [], "snapB": [], "jerk_ratio": [], "energy": [],
                   "arrive": []} for k in ["interp"] + labels}
        g = np.random.default_rng(args.seed)
        used = 0
        for i in range(args.n):
            fa, fb = PAIRS[i % len(PAIRS)]
            try:
                sa, _ = kd.sample_starts(1, g, "val", span=LA, families=[fa])
                sb, _ = kd.sample_starts(1, g, "val", span=LA, families=[fb])
            except Exception:
                continue
            A = kd.window(EMB, sa, LA, LA)[0]
            B = B7.yaw_align(kd.window(EMB, sb, LA, LA)[0], A[-1])
            cur, tgt = A[-1][None].to(dev).float(), B[0][None].to(dev).float()
            kn = torch.ones(1, device=dev)
            m0 = next(iter(models.values()))
            interp = m0.baseline(cur, tgt, kn, gap).float()
            outs = {"interp": interp[0].cpu()}
            for lab in labels:
                m = models[lab]
                e = m.encode(interp, cur, tgt, kn)
                o = m.decode(e["path_z"], e["time_z"], cur, tgt, kn, gap)
                outs[lab] = to_states(o, m, cur, tgt, kn).float()[0].cpu()
            pA, pB = A[..., 6:9], B[..., 6:9]
            vA, vB = vel(pA), vel(pB)
            ref_v = float(torch.cat([vA, vB]).median() + 1e-9)
            jerk_ab = 0.5 * (jerk_rms(pA) + jerk_rms(pB))
            for k, s in outs.items():
                p = s[..., 6:9]
                snapA = float((p[0] - pA[-1]).norm(dim=-1).mean() / ref_v)
                snapB = float((pB[0] - p[-1]).norm(dim=-1).mean() / ref_v)
                agg[k]["snapA"].append(snapA)
                agg[k]["snapB"].append(snapB)
                agg[k]["jerk_ratio"].append(jerk_rms(p) / (jerk_ab + 1e-9))
                agg[k]["energy"].append(float(vel(p).mean() /
                                              (torch.cat([vA, vB]).mean() + 1e-9)))
                agg[k]["arrive"].append(float((p[-1] - pB[0]).norm(dim=-1).mean()
                                              * reach))
            used += 1
        L.append(f"\n-- GAP = {gap} frames ({gap / 30:.0f}s invented)  n={used} --")
        L.append(f"{'metric':<26}" + "".join(f"{k:>12}" for k in ["interp"] + labels))
        for name, key in (("junction snap A->gap", "snapA"),
                          ("junction snap gap->B", "snapB"),
                          ("jerk ratio (gap/AB)", "jerk_ratio"),
                          ("bridge energy ratio", "energy"),
                          ("arrival err [cm]", "arrive")):
            L.append(f"{name:<26}" + "".join(
                f"{np.mean(agg[k][key]):>12.3f}" for k in ["interp"] + labels))
    txt = "\n".join(L)
    print(txt, flush=True)
    open(f"{V.RUNS}/BRIDGE_verdict_0810.txt", "w").write(txt + "\n")
    print(f"verdict: {V.RUNS}/BRIDGE_verdict_0810.txt", flush=True)


if __name__ == "__main__":
    main()
