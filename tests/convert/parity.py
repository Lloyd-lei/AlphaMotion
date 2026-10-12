"""Parity of the GPU converter against the CPU converter on the same clips.

usage: parity.py --gpu-root R --cpu-root /home/arenalabs/umr_factory_data/smplx/hiphi [--kind hiphi] --out parity.json
                 [--control betas_control.npy]   (CPU float32 beta fits on 1e-5 m-perturbed targets, see beta_control.py)
Per clip: array diffs of the npz, per-frame SMPL-X body-joint (22) and vertex errors (mm) from the same GPU SMPL-X
forward, betas difference, qa.json flag/field comparison; plus a betas-only error (CPU pose with GPU betas) and the
control arm (CPU pose with the CPU path's own perturbed-input betas).
"""
from __future__ import annotations
from surface_retargeting.config import CFG as SRCFG  # machine paths: configs/sr.toml

import argparse
import json
import sys
from pathlib import Path

import numpy as np



def load(kind, d):
    if kind == "hiphi":
        z = np.load(d / "motion_actor_smplx.npz")
        return {k: z[k] for k in z.files}
    return dict(poses=np.load(d / "poses.npy"), trans=np.load(d / "transl.npy"), betas=np.load(d / "betas.npy"),
                per_frame_residuals=np.load(d / "per_frame_residuals.npy"))


def qa_cmp(a, b):
    out = dict(ok_equal=a.get("ok") == b.get("ok"), problems_equal=a.get("problems") == b.get("problems"),
               layout_equal=a.get("bvh_layout") == b.get("bvh_layout"), frames_equal=a.get("frames") == b.get("frames"))
    fa, fb = a.get("floor", {}), b.get("floor", {})
    out["floor_y_mm_diff"] = round(1000 * abs(fa.get("floor_y_m", 0) - fb.get("floor_y_m", 0)), 3)
    out["sink_gt1cm_frac_diff"] = round(abs(fa.get("sink_gt1cm_frac", 0) - fb.get("sink_gt1cm_frac", 0)), 4)
    out["floor_fields_equal"] = fa == fb
    out["objects_equal"] = a.get("objects") == b.get("objects")
    ra, rb = a.get("ik_residual_cm", {}), b.get("ik_residual_cm", {})
    out["ik_residual_cm_maxdiff"] = max([abs(ra.get(k, 0) - rb.get(k, 0)) for k in ("mean", "p95", "max")] or [0])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-root", required=True)
    ap.add_argument("--cpu-root", default=f'{SRCFG.data_root}/smplx/hiphi')
    ap.add_argument("--kind", default="hiphi")
    ap.add_argument("--out", required=True)
    ap.add_argument("--control", default=None)
    args = ap.parse_args()
    from surface_retargeting.ingest.convert_gpu import gpu_smplx
    sm = gpu_smplx.GpuSmplx("cuda")
    groot, croot = Path(args.gpu_root), Path(args.cpu_root)
    pat = "data/*/*/*/qa.json" if args.kind == "hiphi" else "*/*/*/qa.json"
    ctrl = np.load(args.control, allow_pickle=True).item() if args.control else {}
    rows = []
    for q in sorted(groot.glob(pat)):
        rel = q.parent.relative_to(groot)
        cdir = croot / rel
        if not (cdir / "qa.json").exists():
            continue
        qa_g, qa_c = json.loads(q.read_text()), json.loads((cdir / "qa.json").read_text())
        if not (qa_g.get("ok") and qa_c.get("ok")):
            rows.append(dict(clip=str(rel), ok_g=qa_g.get("ok"), ok_c=qa_c.get("ok"), err=qa_g.get("error"))); continue
        g, c = load(args.kind, q.parent), load(args.kind, cdir)
        T = len(c["poses"])
        r = dict(clip=str(rel), frames=T, frames_equal=len(g["poses"]) == T)
        for k in ("poses", "trans", "betas", "fitted_joints", "per_frame_residuals", "final_losses"):
            if k in g and k in c and g[k].shape == c[k].shape:
                r[f"{k}_bitwise_equal"] = bool(np.array_equal(g[k], c[k]))
                r[f"{k}_maxabs"] = float(np.abs(g[k].astype(np.float64) - c[k]).max())
        r["betas_maxabs"] = float(np.abs(g["betas"] - c["betas"]).max())
        e = sm.compare(g["poses"], g["trans"], g["betas"], c["poses"], c["trans"], c["betas"])
        mm = lambda x: float(x) * 1000
        r.update(vert_mm_median=mm(np.median(e["v_median"])), vert_mm_mean=mm(e["v_mean"].mean()), vert_mm_p95=mm(np.percentile(e["v_max"], 95)),
                 vert_mm_max=mm(e["v_max"].max()), joint_mm_median=mm(np.median(e["j_mean"])), joint_mm_p95=mm(np.percentile(e["j_max"], 95)),
                 joint_mm_max=mm(e["j_max"].max()))
        # betas-only effect (CPU pose + GPU betas) and the control arm (CPU pose + CPU perturbed-input betas)
        eb = sm.compare(c["poses"], c["trans"], g["betas"], c["poses"], c["trans"], c["betas"])
        r.update(betas_only_vert_mm_max=mm(eb["v_max"].max()), betas_only_vert_mm_median=mm(np.median(eb["v_median"])))
        key = str(rel)
        if key in ctrl:
            ec = sm.compare(c["poses"], c["trans"], ctrl[key], c["poses"], c["trans"], c["betas"])
            r.update(control_betas_maxabs=float(np.abs(ctrl[key] - c["betas"]).max()), control_vert_mm_max=mm(ec["v_max"].max()),
                     control_vert_mm_median=mm(np.median(ec["v_median"])))
        r.update(qa_cmp(qa_g, qa_c))
        if "gpu_convert" in qa_g:
            r["beta_fit"] = qa_g["gpu_convert"].get("beta_fit")
            r["gpu_beta_loss_w5"] = qa_g["gpu_convert"].get("beta_loss_w5")
        rows.append(r)
    ok = [r for r in rows if "vert_mm_max" in r]
    agg = lambda k, f: (round(float(f([r[k] for r in ok if k in r])), 4) if any(k in r for r in ok) else None)
    summ = dict(n_clips=len(rows), n_compared=len(ok),
                frames=int(sum(r["frames"] for r in ok)),
                poses_bitwise_equal_clips=sum(bool(r.get("poses_bitwise_equal")) for r in ok),
                fitted_joints_bitwise_equal_clips=sum(bool(r.get("fitted_joints_bitwise_equal")) for r in ok),
                vert_mm_median_of_clip_medians=agg("vert_mm_median", np.median), vert_mm_max_over_clips=agg("vert_mm_max", np.max),
                vert_mm_p95_median_over_clips=agg("vert_mm_p95", np.median),
                joint_mm_median_of_clip_medians=agg("joint_mm_median", np.median), joint_mm_max_over_clips=agg("joint_mm_max", np.max),
                betas_maxabs_median=agg("betas_maxabs", np.median), betas_maxabs_max=agg("betas_maxabs", np.max),
                betas_only_vert_mm_max_median=agg("betas_only_vert_mm_max", np.median), betas_only_vert_mm_max_max=agg("betas_only_vert_mm_max", np.max),
                control_vert_mm_max_median=agg("control_vert_mm_max", np.median), control_vert_mm_max_max=agg("control_vert_mm_max", np.max),
                control_betas_maxabs_median=agg("control_betas_maxabs", np.median),
                qa_ok_equal=all(r.get("ok_equal") for r in ok), qa_problems_equal=all(r.get("problems_equal") for r in ok),
                qa_layout_equal=all(r.get("layout_equal") for r in ok),
                qa_floor_fields_equal_clips=sum(bool(r.get("floor_fields_equal")) for r in ok),
                qa_objects_equal_clips=sum(bool(r.get("objects_equal")) for r in ok),
                floor_y_mm_diff_max=agg("floor_y_mm_diff", np.max), ik_residual_cm_maxdiff_max=agg("ik_residual_cm_maxdiff", np.max))
    Path(args.out).write_text(json.dumps(dict(summary=summ, clips=rows), indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
