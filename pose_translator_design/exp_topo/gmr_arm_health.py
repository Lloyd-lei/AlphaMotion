"""GMR ARM-LABEL HEALTH CHECK — is the paired robot GT's arm channel noisy?

    gmr_arm_health.py

Same human windows -> every embodiment's paired GT.  Three fingerprints,
arms vs legs (legs = known-clean control), robot vs human (human = source):

  rough_ratio   second-difference magnitude of end-effector trajectory,
                robot / human.  IK-injected jitter shows up here.
  vel_cos       cosine between robot and human end-effector velocity
                (root-frame, per frame, len-weighted).  Noise decorrelates.
  eflip/1k      elbow-branch flips per 1000 frames: the elbow deviation
                vector (elbow vs shoulder->wrist chord) turning >90deg in
                one frame while the arm is bent = the multi-solution jump.

Writes runs/GMR_ARM_HEALTH_0811.txt.
"""
from __future__ import annotations
import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("V5_AMASS", "/var/tmp/alphamotion/topo_exp/cache_xwbc_amass_full")
import v5_common as V                                                # noqa: E402
from bridge_a import CodeData, FULL                                  # noqa: E402
from train_global import build_global                                # noqa: E402
from common import rot6d_to_matrix                                   # noqa: E402

OUT = "/var/tmp/alphamotion/topo_exp/runs/GMR_ARM_HEALTH_0811.txt"
EMBS = ["human_smpl", "booster_k1", "booster_t1", "booster_t1_29dof",
        "engineai_pm01", "fourier_gr3", "fourier_n1", "hightorque_hi",
        "kuavo_s45", "pnd_adam_lite", "tienkung", "unitree_g1_29dof",
        "unitree_g1_23dof", "unitree_h1", "unitree_h1_2", "unitree_h2",
        "unitree_r1"]
WIN, NWIN = 90, 120


ARM_KW = ("arm", "shoulder", "elbow", "wrist", "hand", "palm", "clavicle")
LEG_KW = ("ankle", "foot", "shank", "knee", "leg")


def chain(names, parents, side, kws):
    """joints on one side matching kws, sorted shallow->deep by tree depth."""
    nl = [n.lower() for n in names]
    depth = []
    for i in range(len(nl)):
        d, j = 0, i
        while parents[j] >= 0:
            j = parents[j]; d += 1
        depth.append(d)
    import re
    def onside(n):
        if ("left" if side == "l" else "right") in n:
            return True
        return re.search(rf"(^|_){side}(?=[\d_]|$)", n) is not None
    sel = [i for i, n in enumerate(nl)
           if onside(n) and any(k in n for k in kws)]
    if not sel:
        raise KeyError(f"{side}/{kws} not in {names}")
    return sorted(sel, key=lambda i: depth[i])


def ee_pos(l6, spec, rows, dev):
    """root-frame positions [T,J,3] cm for contiguous rows."""
    g6, _, _ = build_global(l6[rows], spec, dev)
    pos = torch.as_tensor(V.fk_pos(g6, spec)).float()
    R0 = rot6d_to_matrix(g6[:, 0].float())
    rel = pos - pos[:, :1]
    return torch.einsum("nji,nkj->nki", R0, rel)


def main():
    dev = "cuda"
    data = CodeData("PROD_c1_91k_dim20_tail_0810", use_root=True)
    g = np.random.default_rng(0)
    pool = [(a, b) for sp in ("train", "val") for a, b in data.pools[sp]
            if b - a >= WIN + 2]
    wins = [int(g.integers(a, b - WIN)) for a, b in
            (pool[i] for i in g.choice(len(pool), NWIN, replace=False))]

    stats = {}
    hum = None
    for emb in EMBS:
        l6 = torch.load(f"{FULL}/{emb}_paired_local6d.pt", mmap=True)
        spec = V.load_spec(FULL, emb)
        names = list(spec.joint_names)
        idx = {}
        par = list(spec.parents)
        try:
            for side in ("l", "r"):
                ch = chain(names, par, side, ARM_KW)
                idx[f"{side}s"], idx[f"{side}w"] = ch[0], ch[-1]
                idx[f"{side}e"] = ch[len(ch) // 2] if len(ch) > 2 else ch[-1]
                idx[f"{side}a"] = chain(names, par, side, LEG_KW)[-1]
        except KeyError:
            print(f"[{emb}] SKIP (anonymous joint names)", flush=True)
            continue
        rough = {"arm": [], "leg": []}
        vel = {"arm": [], "leg": []}
        flips = 0
        frames = 0
        for s0 in wins:
            rows = torch.arange(s0, s0 + WIN)
            p = ee_pos(l6, spec, rows, dev)
            for side in ("l", "r"):
                w, a = p[:, idx[f"{side}w"]], p[:, idx[f"{side}a"]]
                rough["arm"].append(float((w[2:] - 2 * w[1:-1] + w[:-2])
                                          .norm(dim=-1).mean()))
                rough["leg"].append(float((a[2:] - 2 * a[1:-1] + a[:-2])
                                          .norm(dim=-1).mean()))
                vel["arm"].append(w[1:] - w[:-1])
                vel["leg"].append(a[1:] - a[:-1])
                # elbow branch flip
                sj, e = p[:, idx[f"{side}s"]], p[:, idx[f"{side}e"]]
                chord = w - sj
                t = ((e - sj) * chord).sum(-1, keepdim=True) / \
                    (chord.norm(dim=-1, keepdim=True) ** 2 + 1e-8)
                dvec = e - (sj + t * chord)                    # deviation
                dn = torch.nn.functional.normalize(dvec, dim=-1)
                bent = dvec.norm(dim=-1) > 2.0                 # cm: arm not straight
                cos = (dn[1:] * dn[:-1]).sum(-1)
                flips += int(((cos < 0) & bent[1:] & bent[:-1]).sum())
                frames += WIN - 1
        stats[emb] = {"rough": {k: float(np.mean(v)) for k, v in rough.items()},
                      "vel": {k: torch.cat([x.reshape(-1, 3) for x in v])
                              for k, v in vel.items()},
                      "eflip": 1000.0 * flips / frames}
        if emb == "human_smpl":
            hum = stats[emb]
        print(f"[{emb}] rough arm {stats[emb]['rough']['arm']:.3f} "
              f"leg {stats[emb]['rough']['leg']:.3f} "
              f"eflip/1k {stats[emb]['eflip']:.1f}", flush=True)

    lines = [f"{'embodiment':20s} {'rough_ratio':>11s} {'(leg)':>7s} "
             f"{'vel_cos':>8s} {'(leg)':>7s} {'eflip/1k':>9s} {'(hum)':>7s}"]
    hv = {k: hum["vel"][k] for k in ("arm", "leg")}
    for emb in EMBS[1:]:
        if emb not in stats:
            continue
        st = stats[emb]
        rr = {k: st["rough"][k] / max(hum["rough"][k], 1e-6)
              for k in ("arm", "leg")}
        vc = {}
        for k in ("arm", "leg"):
            a, b = st["vel"][k], hv[k]
            num = (a * b).sum(-1)
            den = a.norm(dim=-1) * b.norm(dim=-1) + 1e-6
            m = b.norm(dim=-1) > 0.3                           # moving frames
            vc[k] = float((num / den)[m].mean())
        lines.append(f"{emb:20s} {rr['arm']:11.2f} {rr['leg']:7.2f} "
                     f"{vc['arm']:8.3f} {vc['leg']:7.3f} "
                     f"{st['eflip']:9.1f} {hum['eflip']:7.1f}")
    body = "\n".join(lines)
    print("\n" + body)
    open(OUT, "w").write(
        "GMR ARM-LABEL HEALTH (0811). rough_ratio = robot/human 2nd-diff at "
        "end-effector; vel_cos = robot-vs-human EE velocity cosine (moving "
        f"frames); eflip = elbow-branch flips per 1k frames.\n{body}\n")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
