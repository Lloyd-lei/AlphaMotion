"""Single-frame sparse SE3 constraint demo (the product primitive, layer 1 of the
precision ladder): give the encoder ONLY a few constrained joints (e.g. root +
one wrist), let the masked-trained tokenizer complete the full body, and measure
how well the completion honors the given joints.

Reads GT frames from the paired cache, hides everything except --keep joints,
completes, reports per-kept-joint position/rotation error (constraint
satisfaction, soft) + full-body error vs the GT frame (completion plausibility;
GT is ONE valid completion, not the only one).

Usage:
  python constraint_demo.py --run .../runs/E0b_mask25_0803 --keep Root LeftHand
  python constraint_demo.py --run ... --keep Root LeftHand RightFoot --emb soma77
Outputs a JSON summary + renders completion vs GT via render_mesh-style capsules.
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from common import rot6d_to_matrix, geodesic_deg
from dual_common import load_paired_data, split_indices, build_model, geom


def resolve_joints(spec, patterns):
    """Exact name match first (so 'LeftHand' is the wrist, not 24 finger joints);
    substring fallback. Root always kept."""
    keep = set()
    for pat in patterns:
        exact = [j for j, n in enumerate(spec.joint_names) if pat.lower() == n.lower()]
        hits = exact or [j for j, n in enumerate(spec.joint_names) if pat.lower() in n.lower()]
        if not hits:
            raise SystemExit(f"no joint matches '{pat}' in {spec.name}; names: {spec.joint_names}")
        keep.update(hits)
    keep.add(int(np.where(spec.parents < 0)[0][0]))
    return sorted(keep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="a mask-trained run")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--emb", default="soma77")
    ap.add_argument("--keep", nargs="+", default=["LeftHand"], help="joint name substrings to keep visible")
    ap.add_argument("--n", type=int, default=512, help="val frames to evaluate")
    args = ap.parse_args()
    device = "cuda"

    cfg = json.load(open(f"{args.run}/config.json"))
    torch.manual_seed(0)
    data, names, N = load_paired_data(args.cache or cfg["cache"], device)
    val_idx, _ = split_indices(N)
    idx = val_idx[:args.n]
    model = build_model(cfg, device)
    model.load_state_dict(torch.load(f"{args.run}/model.pt", map_location=device), strict=False)
    model.eval()

    spec = data[args.emb]["spec"]
    keep = resolve_joints(spec, args.keep)
    mask = torch.ones(len(idx), spec.J, dtype=torch.bool, device=device)
    mask[:, keep] = False                                      # False = visible
    print(f"constraint joints ({len(keep)}): {[spec.joint_names[j] for j in keep]}")

    x = data[args.emb]["pose"][idx]; g = geom(data, args.emb, len(idx))
    with torch.no_grad():
        zp, zr = model.encode(x, g, joint_mask=mask)
        prot, ppos = model.decode(zp, zr, g)
    reach = data[args.emb]["reach"]
    pos_err = (ppos - x[..., 6:9]).norm(dim=-1) * reach        # [B,J] cm
    rot_err = geodesic_deg(rot6d_to_matrix(prot), rot6d_to_matrix(x[..., :6]))
    kept = torch.zeros(spec.J, dtype=torch.bool); kept[keep] = True

    out = {
        "run": args.run, "emb": args.emb,
        "constraint_joints": [spec.joint_names[j] for j in keep],
        "constraint_satisfaction": {                            # how well completion honors the givens
            "pos_cm": round(float(pos_err[:, kept].mean()), 2),
            "rot_deg": round(float(rot_err[:, kept].mean()), 1)},
        "completion_vs_gt": {                                   # plausibility vs one valid answer
            "pos_cm": round(float(pos_err[:, ~kept].mean()), 2),
            "rot_deg": round(float(rot_err[:, ~kept].mean()), 1)},
        "n_frames": int(len(idx)),
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))
    path = f"{args.run}/constraint_demo_{'_'.join(args.keep)}.json"
    json.dump(out, open(path, "w"), indent=2)
    print("WROTE", path)


if __name__ == "__main__":
    main()
