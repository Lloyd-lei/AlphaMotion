"""DECODER POST-TRAIN DUEL: C1 vs PT_c1dec_arm on one yardstick, plus the
code-invariance proof.

    duel_decoder_pt.py --pt PT_c1dec_arm_0811

  1 CODE INVARIANCE  encode ints of both models on identical rows must be
    bit-identical (the zero-cascade guarantee).
  2 PER-JOINT DUEL   cross directions: arm-chain vs leg-chain pos error,
    mean AND p95, per model.  Acceptance: arms down, legs within 5%.
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dual_common import load_paired_data, split_indices, build_model, geom

RUNS = "/var/tmp/alphamotion/topo_exp/runs"
ARM_KW = ("shoulder", "elbow", "wrist", "arm", "hand")
LEG_KW = ("hip", "knee", "ankle", "foot", "leg", "toe")


def load(run, data_cfg_from, device):
    cfg = json.load(open(f"{RUNS}/{run}/config.json"))
    m = build_model(cfg, device)
    m.load_state_dict(torch.load(f"{RUNS}/{run}/model.pt", map_location=device))
    m.eval()
    return m


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="PROD_c1_91k_dim20_tail_0810")
    ap.add_argument("--pt", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n_rows", type=int, default=4096)
    args = ap.parse_args()
    dev = args.device

    cfg = json.load(open(f"{RUNS}/{args.base}/config.json"))
    torch.manual_seed(0)
    data, names, N = load_paired_data(cfg["cache"], dev,
                                      sem_encoder=cfg["sem_encoder"])
    val_idx, _ = split_indices(N)
    val = val_idx[torch.randperm(len(val_idx))[:args.n_rows]]
    mb = load(args.base, cfg, dev)
    mp = load(args.pt, cfg, dev)

    # ---- 1 code invariance ----
    same = True
    for src in ("human_smpl", "booster_t1", "unitree_h1"):
        g_ = geom(data, src, len(val))
        ib_p, ib_r = mb.ints_from_features(*mb.encode_features(data[src]["pose"][val], g_))
        ip_p, ip_r = mp.ints_from_features(*mp.encode_features(data[src]["pose"][val], g_))
        ok = bool(torch.equal(ib_p, ip_p) and torch.equal(ib_r, ip_r))
        same &= ok
        print(f"[codes] {src}: {'IDENTICAL' if ok else 'MISMATCH!!'} "
              f"({ib_p.numel()} ints)")
    print(f"CODE INVARIANCE: {'PASS' if same else 'FAIL'}\n")

    # ---- 2 per-joint duel ----
    DIRS = [("human_smpl", "unitree_g1_29dof"), ("human_smpl", "fourier_gr3"),
            ("human_smpl", "booster_t1"), ("human_smpl", "kuavo_s45"),
            ("booster_t1", "unitree_h1")]
    print(f"{'direction':32s} {'model':4s} {'arm mean/p95':>16s} "
          f"{'leg mean/p95':>16s} {'wrist mean/p95':>16s}")
    verdict = []
    for src, tgt in DIRS:
        spec = data[tgt]["spec"]
        nl = [n.lower() for n in spec.joint_names]
        armm = torch.tensor([any(k in n for k in ARM_KW) for n in nl], device=dev)
        legm = torch.tensor([any(k in n for k in LEG_KW) for n in nl], device=dev)
        wrm = torch.tensor(["wrist" in n for n in nl], device=dev)
        row = {}
        for label, m in (("C1", mb), ("PT", mp)):
            zp, zr = m.encode(data[src]["pose"][val], geom(data, src, len(val)))
            _, ppos = m.decode(zp, zr, geom(data, tgt, len(val)))
            err = (ppos - data[tgt]["pose"][val][..., 6:9]).norm(dim=-1) \
                * data[tgt]["reach"]
            st = {}
            for gname, gm in (("arm", armm), ("leg", legm), ("wrist", wrm)):
                e = err[:, gm].reshape(-1) if gm.any() else err[:, :1] * float("nan")
                st[gname] = (float(e.mean()), float(e.quantile(0.95)))
            row[label] = st
            print(f"{src+'->'+tgt:32s} {label:4s} "
                  f"{st['arm'][0]:7.2f}/{st['arm'][1]:7.2f} "
                  f"{st['leg'][0]:7.2f}/{st['leg'][1]:7.2f} "
                  f"{st['wrist'][0]:7.2f}/{st['wrist'][1]:7.2f}")
        verdict.append({
            "dir": f"{src}->{tgt}",
            "arm_p95_delta_pct": round(100 * (row["PT"]["arm"][1] / row["C1"]["arm"][1] - 1), 1),
            "leg_p95_delta_pct": round(100 * (row["PT"]["leg"][1] / row["C1"]["leg"][1] - 1), 1),
            "wrist_p95_delta_pct": round(100 * (row["PT"]["wrist"][1] / row["C1"]["wrist"][1] - 1), 1)})
    print("\n=== DELTA (PT vs C1, negative = better) ===")
    for v in verdict:
        print(f"  {v['dir']:32s} arm_p95 {v['arm_p95_delta_pct']:+6.1f}%  "
              f"leg_p95 {v['leg_p95_delta_pct']:+6.1f}%  "
              f"wrist_p95 {v['wrist_p95_delta_pct']:+6.1f}%")
    json.dump({"code_invariance": same, "verdict": verdict},
              open(f"{RUNS}/{args.pt}/duel_vs_c1.json", "w"), indent=2)


if __name__ == "__main__":
    main()
