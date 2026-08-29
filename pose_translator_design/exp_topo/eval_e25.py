"""E25 verdict — does masking the redundant rotational DOF buy anything?

Same yardstick as every run this week: zero-shot human_smpl -> unitree_g1_29dof on the
cache_xwbc_lafan VAL split (G1 is held out of training in all three arms), mean geodesic
+ position, per part (legs = hip/knee/ankle/toe, arms = shoulder/elbow, wrists =
wrist/hand). Plus the three E25-specific numbers:

  manifold_dist_deg   how far the RAW prediction sits off the mechanical manifold
                      (local per-joint, hinge joints only) and how much the pose moves
                      when it is snapped there (global). Arm A must be ~0.
  limit_viol          fraction of (frame, hinge) pairs whose unclamped angle is outside
                      the mechanical limit box, and fraction of frames with any such
                      hinge. Arm A must be exactly 0.
  projected           the error AFTER manifold_project.project(method="global") — the
                      number a real robot would actually track.

Every metric is computed with the SAME code for all three runs. CPU by default so it
can run while the GPU queue is busy.

Usage:
  python eval_e25.py --runs E19_K128_0805 E25_dofhead_0805 E25_dofloss_0805
"""
from __future__ import annotations
import os, sys, json, time, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import rot6d_to_matrix, geodesic_deg
from dual_common import load_paired_data, split_indices, build_model, geom
import dof_head as DH

ROOT = "/var/tmp/alphamotion/topo_exp"
RUNS = f"{ROOT}/runs"
PARTS = {"legs": ("hip", "knee", "ankle", "toe"),
         "arms": ("shoulder", "elbow"),
         "wrists": ("wrist", "hand")}
DIRS = [("human_smpl", "unitree_g1_29dof")]


def part_masks(spec, device):
    names = [n.lower() for n in spec.joint_names]
    m = {p: torch.tensor([any(k in n for k in kws) for n in names], device=device)
         for p, kws in PARTS.items()}
    m["other"] = ~(m["legs"] | m["arms"] | m["wrists"])
    return m


@torch.no_grad()
def decode_all(model, data, src, tgt, idx, chunk=512):
    """-> (pred_rot6d [n,J,6], pred_pos [n,J,3]) on CPU."""
    R, P = [], []
    for i in range(0, len(idx), chunk):
        sub = idx[i:i + chunk]
        zp, zr = model.encode(data[src]["pose"][sub], geom(data, src, len(sub)))
        prot, ppos = model.decode(zp, zr, geom(data, tgt, len(sub)))
        R.append(prot.cpu()); P.append(ppos.cpu())
    return torch.cat(R), torch.cat(P)


def err_table(prot, ppos, gt, reach, masks):
    rot_j = geodesic_deg(rot6d_to_matrix(prot), rot6d_to_matrix(gt[..., :6]))
    pos_j = (ppos - gt[..., 6:9]).norm(dim=-1) * reach
    out = {"all": {"rot_deg": round(float(rot_j.mean()), 2),
                   "pos_cm": round(float(pos_j.mean()), 2)}}
    for p, m in masks.items():
        out[p] = {"rot_deg": round(float(rot_j[:, m].mean()), 2),
                  "pos_cm": round(float(pos_j[:, m].mean()), 2)}
    return out, rot_j


@torch.no_grad()
def mechanical_report(prot, kin, spec, dof, rest, n_proj, chunk=256):
    """Off-manifold distance, limit violations, and the post-projection error inputs."""
    hinge = ~kin.free
    md = DH.manifold_distance_deg(prot, kin)                     # [n,J] local
    snap = DH.project_global6d(prot, kin)                        # global snap
    snap_deg = geodesic_deg(snap, rot6d_to_matrix(prot))         # [n,J]
    dpct, fpct, over = DH.limit_violation(prot, kin)
    return {"manifold_dist_local_deg": round(float(md[:, hinge].mean()), 3),
            "manifold_dist_local_p99_deg": round(float(md[:, hinge].flatten().kthvalue(
                max(1, int(0.99 * md[:, hinge].numel())))[0]), 3),
            "hinge_snap_global_deg": round(float(snap_deg.mean()), 3),
            "limit_viol_hinge_pct": round(dpct, 3),
            "limit_viol_frame_pct": round(fpct, 2),
            "limit_worst_overshoot_deg": round(over, 2)}


@torch.no_grad()
def projected_error(prot, gt, spec, dof, rest, masks, n_proj, chunk=128):
    """Error after the real feasibility projection (whole-chain LM, joint limits on)."""
    from manifold_project import project
    idx = torch.arange(min(n_proj, prot.shape[0]))
    rot_all, per = [], {}
    t0 = time.time()
    outs = []
    for i in range(0, len(idx), chunk):
        s = idx[i:i + chunk]
        r6, _, _ = project(prot[s], spec, dof, rest=rest, method="global", lm_iters=25)
        outs.append(r6)
    r6 = torch.cat(outs)
    rot_j = geodesic_deg(rot6d_to_matrix(r6), rot6d_to_matrix(gt[idx, ..., :6]))
    out = {"all": round(float(rot_j.mean()), 2), "n_frames": int(len(idx)),
           "sec": round(time.time() - t0, 1)}
    for p, m in masks.items():
        out[p] = round(float(rot_j[:, m].mean()), 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=f"{ROOT}/cache_xwbc_lafan")
    ap.add_argument("--runs", nargs="+",
                    default=["E19_K128_0805", "E25_dofhead_0805", "E25_dofloss_0805"])
    ap.add_argument("--n_eval", type=int, default=5000)
    ap.add_argument("--n_proj", type=int, default=1024)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max_rows", type=int, default=0)
    ap.add_argument("--reach_from", default="",
                    help="pin `reach` to this cache's value (needed when evaluating on a "
                         "cache the models were not trained on; '' = use the eval cache's own)")
    ap.add_argument("--out", default=f"{RUNS}/E25_verdict.json")
    args = ap.parse_args()
    device = args.device

    torch.manual_seed(0)                                  # split contract: seed -> load -> perm
    data, names, N = load_paired_data(args.cache, device, max_rows=args.max_rows)
    val_idx, _ = split_indices(N)                         # the training-time val split
    idx = val_idx[:args.n_eval]
    if args.reach_from:                                   # same fix as eval_data_axis.pin_reach
        from common import SkeletonSpec
        from train_global import build_global
        for name in sorted({n for d in DIRS for n in d}):
            dd = np.load(f"{args.reach_from}/{name}_spec.npz", allow_pickle=True)
            sp = SkeletonSpec(str(dd["name"]), dd["parents"], dd["rest_offsets"], list(dd["joint_names"]))
            _, _, rref = build_global(torch.load(f"{args.reach_from}/{name}_paired_local6d.pt"),
                                      sp, device)
            old = data[name]["reach"]
            data[name]["pose"][..., 6:9] *= old / rref
            data[name]["reach"] = rref
            print(f"[reach] {name}: {old:.2f} -> {rref:.2f} cm (pinned to {args.reach_from})")
    print(f"[eval] cache={args.cache} rows={N} val={len(val_idx)} n_eval={len(idx)}")

    from manifold_project import load_dof
    masks = {t: part_masks(data[t]["spec"], device) for _, t in DIRS}
    for p, m in masks[DIRS[0][1]].items():
        print(f"  part {p:7s} J={int(m.sum())}")

    out = {"cache": args.cache, "n_eval": int(len(idx)), "n_proj": args.n_proj,
           "dirs": ["->".join(d) for d in DIRS], "runs": {}}
    for run in args.runs:
        rd = run if os.path.isabs(run) else f"{RUNS}/{run}"
        if not (os.path.exists(f"{rd}/config.json") and os.path.exists(f"{rd}/model.pt")):
            print(f"[skip] {run}: not finished"); continue
        cfg = json.load(open(f"{rd}/config.json"))
        model = build_model(cfg, device)
        sd = torch.load(f"{rd}/model.pt", map_location=device)
        miss, unexp = model.load_state_dict(sd, strict=True)
        model.eval()
        r = {"params": sum(p.numel() for p in model.parameters()),
             "rot_head": cfg.get("rot_head", "free"), "dof_loss": bool(cfg.get("dof_loss", False)),
             "n_latent": cfg.get("n_latent"), "fsq_levels": cfg.get("fsq_levels"),
             "steps": cfg.get("steps"), "cache": cfg.get("cache"), "seed": cfg.get("seed", 0)}
        for src, tgt in DIRS:
            prot, ppos = decode_all(model, data, src, tgt, idx)
            gt = data[tgt]["pose"][idx].cpu()
            tab, _ = err_table(prot, ppos, gt, data[tgt]["reach"], masks[tgt])
            kin = data[tgt]["kin"].to("cpu")
            dof, rest = load_dof(args.cache, tgt, device="cpu")
            spec = data[tgt]["spec"]
            tab["mechanical"] = mechanical_report(prot, kin, spec, dof, rest, args.n_proj)
            tab["projected_rot_deg"] = projected_error(prot, gt, spec, dof, rest,
                                                       masks[tgt], args.n_proj)
            r[f"{src}->{tgt}"] = tab
            k = r[f"{src}->{tgt}"]
            print(f"[{run}] {src}->{tgt}  rot={k['all']['rot_deg']:.2f} pos={k['all']['pos_cm']:.2f}cm "
                  f"| legs {k['legs']['rot_deg']:.2f} arms {k['arms']['rot_deg']:.2f} "
                  f"wrists {k['wrists']['rot_deg']:.2f} | off-manifold "
                  f"{k['mechanical']['manifold_dist_local_deg']:.2f}deg "
                  f"viol {k['mechanical']['limit_viol_frame_pct']:.1f}%frames "
                  f"| projected {k['projected_rot_deg']['all']:.2f}", flush=True)
        out["runs"][run] = r
        del model
    json.dump(out, open(args.out, "w"), indent=2)
    print("WROTE", args.out)

    # side-by-side table
    rs = [r for r in args.runs if r in out["runs"]]
    if rs:
        d = "->".join(DIRS[0])
        rows = ["all", "legs", "arms", "wrists", "other"]
        w = max(len(r) for r in rs) + 2
        print(f"\n{'part':8s} " + "".join(f"{r:>{w}s}" for r in rs))
        for k in rows:
            print(f"{k+' rot':8s} " + "".join(
                f"{out['runs'][r][d][k]['rot_deg']:>{w}.2f}" for r in rs))
            print(f"{k+' pos':8s} " + "".join(
                f"{out['runs'][r][d][k]['pos_cm']:>{w}.2f}" for r in rs))
        for k in ["manifold_dist_local_deg", "hinge_snap_global_deg",
                  "limit_viol_hinge_pct", "limit_viol_frame_pct"]:
            print(f"{k:24s} " + "".join(
                f"{out['runs'][r][d]['mechanical'][k]:>{w}.3f}" for r in rs))
        print(f"{'projected rot':24s} " + "".join(
            f"{out['runs'][r][d]['projected_rot_deg']['all']:>{w}.2f}" for r in rs))


if __name__ == "__main__":
    main()
