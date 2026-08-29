"""Single-frame sparse SE3 constraint demo (the product primitive, layer 1 of the
precision ladder): give the encoder ONLY a few constrained joints (e.g. root +
one wrist), let the masked-trained tokenizer complete the full body, and measure
how well the completion honors the given joints.

Reads GT frames from the paired cache, hides everything except --keep joints,
completes, reports per-kept-joint position/rotation error (constraint
satisfaction, soft) + full-body error vs the GT frame (completion plausibility;
GT is ONE valid completion, not the only one).

--project turns the SOFT conditioning into a HARD one: the completion is pushed
through manifold_project.project_constrained, which (a) projects onto the target
robot's mechanical manifold and (b) runs damped least squares with the constrained
joints as task-space goals and the projected pose as the null-space reference.
Constraint satisfaction then stops being a model property and becomes a solver
property -> millimetres, while the rest of the body stays where the model put it.

Usage:
  python constraint_demo.py --run .../runs/E0b_mask25_0803 --keep Root LeftHand
  python constraint_demo.py --run ... --keep Root LeftHand RightFoot --emb soma77
  python constraint_demo.py --run ... --keep L_Wrist --emb human_smpl --project
Outputs a JSON summary + renders completion vs GT via render_mesh-style capsules.
"""
from __future__ import annotations
import os, sys, json, argparse, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from common import rot6d_to_matrix, geodesic_deg
from dual_common import load_paired_data, split_indices, build_model, geom
import manifold_project as MP


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


def geo_deg(Ra, Rb):
    """Unclamped geodesic. common.geodesic_deg floors at 0.081 deg (its acos clamp),
    which would hide a millimetre-grade solve. atan2 form, stable near 0."""
    Rr = Ra.transpose(-1, -2) @ Rb
    A = (Rr - Rr.transpose(-1, -2)) / 2
    v = torch.stack([A[..., 2, 1], A[..., 0, 2], A[..., 1, 0]], -1)
    return torch.rad2deg(torch.atan2(v.norm(dim=-1).clamp(max=1.0),
                                     (Rr.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2))


def fk_pos_from_global(R, spec):
    """Root-relative joint positions implied by GLOBAL rotations (spec offset units)."""
    off = torch.as_tensor(np.asarray(spec.rest_offsets), dtype=R.dtype, device=R.device)
    gp = [None] * spec.J
    for j, p in enumerate(spec.parents):
        gp[j] = torch.zeros(R.shape[0], 3, dtype=R.dtype, device=R.device) if int(p) < 0 \
            else gp[int(p)] + (R[:, int(p)] @ off[j])
    return torch.stack(gp, 1)


def limit_violation(q, dof, tol=1e-9):
    """-> (bad slot-frames, total limited slot-frames, worst overshoot deg)."""
    v = dof[:, :18].reshape(-1, 3, 6)
    lo = v[:, :, 5] * np.pi - v[:, :, 4] * np.pi
    hi = v[:, :, 5] * np.pi + v[:, :, 4] * np.pi
    nd = (dof[:, 18] * 3).round().long()
    act = (torch.arange(3, device=q.device)[None, :] < nd[:, None]) & (v[:, :, 3] > 0.5)
    if not bool(act.any()):
        return 0, 0, 0.0
    bad = ((q < lo[None] - tol) | (q > hi[None] + tol)) & act[None]
    over = (q - q.clamp(lo[None], hi[None])).abs()[act[None].expand_as(q)]
    return int(bad.sum()), int(act.sum()) * q.shape[0], float(over.max() * 180 / np.pi)


def hard_pin(args, cache, spec, prot, x, keep, reach):
    """Pin the constrained joints with task-space DLS on top of the feasibility
    projection. Everything in float64 on CPU (36-63 params/frame; trivially fast).
    Returns a dict of before/after constraint-satisfaction numbers."""
    dof, rest = MP.load_dof(cache, args.emb, dtype=torch.float64)
    Rp = rot6d_to_matrix(prot.detach().double().cpu())            # model completion
    Rg = rot6d_to_matrix(x[..., :6].detach().double().cpu())      # GT (the givens)
    Pg = (x[..., 6:9].detach().double().cpu()) * reach            # GT positions, cm
    Pp = fk_pos_from_global(Rp, spec)
    kj = list(keep)
    t0 = time.time()
    rot6, pos, q, q0 = MP.project_constrained(
        Rp, spec, dof, kj, target_pos=Pg[:, kj], target_rot=Rg[:, kj], rest=rest,
        soft_margin=args.soft_margin, method=args.method, lm_iters=args.lm_iters,
        w_task_pos=args.w_task / max(reach, 1e-6) ** 2, w_task_rot=args.w_task,
        w_ref=args.w_ref)
    dt = (time.time() - t0) / max(len(Rp), 1)
    Rn = rot6d_to_matrix(rot6)
    kmask = torch.zeros(spec.J, dtype=torch.bool); kmask[kj] = True

    def blk(R, P):
        pe = (P - Pg).norm(dim=-1); re = geo_deg(R, Rg)
        return {"constraint_pos_mm": round(float(pe[:, kmask].mean()) * 10, 4),
                "constraint_pos_p99_mm": round(float(pe[:, kmask].flatten().quantile(.99)) * 10, 4),
                "constraint_pos_max_mm": round(float(pe[:, kmask].max()) * 10, 4),
                "constraint_frac_over_1mm": round(float((pe[:, kmask] * 10 > 1).float().mean()), 4),
                "constraint_rot_deg": round(float(re[:, kmask].mean()), 5),
                "constraint_rot_max_deg": round(float(re[:, kmask].max()), 5),
                "rest_pos_cm": round(float(pe[:, ~kmask].mean()), 2),
                "rest_rot_deg": round(float(re[:, ~kmask].mean()), 1)}

    nb0, nt0, w0 = limit_violation(MP.fit_angles(Rp, spec, dof, rest=rest, clamp=False)[0], dof)
    nb1, nt1, w1 = limit_violation(q, dof)
    return {
        "method": args.method, "w_task": args.w_task, "w_ref": args.w_ref,
        "lm_iters": args.lm_iters, "n_par_frame": int(q.shape[1] * 3),
        "before_fk_of_prediction": blk(Rp, Pp),
        "after_hard_pin": blk(Rn, pos),
        "limit_violation_pct_before": round(100.0 * nb0 / max(nt0, 1), 3),
        "limit_violation_pct_after": round(100.0 * nb1 / max(nt1, 1), 3),
        "worst_overshoot_deg_before": round(w0, 3),
        "worst_overshoot_deg_after": round(w1, 6),
        "ms_per_frame": round(1000 * dt, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="a mask-trained run")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--emb", default="soma77")
    ap.add_argument("--keep", nargs="+", default=["LeftHand"], help="joint name substrings to keep visible")
    ap.add_argument("--n", type=int, default=512, help="val frames to evaluate")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--project", action="store_true",
                    help="hard-pin the constrained joints (feasibility projection + task-space DLS)")
    ap.add_argument("--method", default="global", choices=["global", "greedy"],
                    help="manifold projection used as the null-space reference")
    ap.add_argument("--w_task", type=float, default=1e4, help="task-space weight")
    ap.add_argument("--w_ref", type=float, default=1e-2, help="null-space pull toward the projected pose")
    ap.add_argument("--lm_iters", type=int, default=60)
    ap.add_argument("--soft_margin", type=float, default=1.0)
    args = ap.parse_args()
    device = args.device

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

    if args.project:
        out["hard_constraint"] = hard_pin(args, args.cache or cfg["cache"], spec,
                                          prot, x, keep, reach)
    print(json.dumps(out, indent=2, ensure_ascii=False))
    path = (f"{args.run}/constraint_demo_{'_'.join(args.keep)}"
            f"{'_pinned' if args.project else ''}.json")
    json.dump(out, open(path, "w"), indent=2)
    print("WROTE", path)


if __name__ == "__main__":
    main()
