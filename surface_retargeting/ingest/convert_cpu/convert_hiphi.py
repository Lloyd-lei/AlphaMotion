"""HiPHI motion package (BVH + metadata + object_tracks) -> UMR HiPHI layout.

usage: convert_hiphi.py <raw_motion_dir> [--force]
out:   /home/arenalabs/umr_factory_data/smplx/hiphi/data/<frame>/<lu>/<motion_id>/
         motion_actor_smplx.npz, metadata.json, object_tracks/*.csv, qa.json
       object meshes are shared at smplx/hiphi/object_meshes (copied once by the batch script)
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
torch.set_num_threads(1)

sys.path.insert(0, str(Path(__file__).parent))
import bvh2smplx  # noqa: E402
import smplx_qa  # noqa: E402

OUT = Path("/home/arenalabs/umr_factory_data/smplx/hiphi")


def main():
    src = Path(sys.argv[1]).resolve()
    force = "--force" in sys.argv
    meta = json.loads((src / "metadata.json").read_text())
    mid = meta["motion_id"]
    dst = OUT / "data" / meta["frame"] / meta["lu"] / mid
    if (dst / "qa.json").exists() and not force:
        print("skip", dst); return
    dst.mkdir(parents=True, exist_ok=True)
    rec = {"clip_id": f"hiphi/{meta['frame']}/{meta['lu']}/{mid}", "source": "HiPHI", "raw_dir": str(src),
           "actor_id": meta.get("actor_id"), "actor_metadata": meta.get("actor_metadata"),
           "is_hoi": meta.get("is_hoi"), "mirrored": meta.get("mirrored")}
    t0 = time.time()
    try:
        res, layout = bvh2smplx.convert(str(src / "motion_actor.bvh"))
        T = len(res.poses)
        poses = res.poses.reshape(T, 165).astype(np.float32)
        trans = res.trans.astype(np.float32)
        fps = float(1.0 / res.frame_time)
        rec.update(bvh_layout=layout, frames=T, fps=round(fps, 4), frames_meta=meta.get("frame_count"))
        problems = []
        if meta.get("frame_count") != T:
            problems.append(f"metadata frame_count {meta.get('frame_count')} != bvh {T}")
        tmp = dst / ".motion_actor_smplx.tmp.npz"
        np.savez_compressed(
            tmp, poses=poses, trans=trans, transl=trans, betas=res.betas.astype(np.float32),
            gender=np.asarray("neutral"), model_type=np.asarray("smplx"),
            mocap_framerate=np.asarray(fps, np.float32), fps=np.asarray(fps, np.float32),
            frame_time=np.asarray(res.frame_time), coordinate_up=np.asarray("y"), output_up=np.asarray("y"),
            num_joints=np.asarray(55, np.int32), fitted_joints=res.fitted_joints.astype(np.float32),
            per_frame_residuals=res.residuals.astype(np.float32), final_losses=res.final_losses.astype(np.float32),
            pipeline=np.asarray("hiphi2smplx@65e5131 (body only, fingers zero) + umr_factory wrapper"),
            source_bvh=np.asarray(str(src / "motion_actor.bvh")))
        tmp.replace(dst / "motion_actor_smplx.npz")
        shutil.copy2(src / "metadata.json", dst / "metadata.json")
        objs = []
        for o in meta.get("objects", []):
            tp = src / o["trajectory_path"]
            (dst / tp.relative_to(src)).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(tp, dst / tp.relative_to(src))
            rows = np.genfromtxt(tp, delimiter=",", names=True)
            tr = np.stack([rows[k] for k in ("px", "py", "pz", "qx", "qy", "qz", "qw")], 1)
            if len(tr) != T:
                problems.append(f"{o['object_id']}: track rows {len(tr)} != frames {T}")
            if not np.array_equal(rows["frame"], np.arange(len(tr))):
                problems.append(f"{o['object_id']}: non-contiguous frame index")
            qn = np.linalg.norm(tr[:, 3:], axis=1)
            if np.abs(qn - 1).max() > 1e-3 or not np.isfinite(tr).all():
                problems.append(f"{o['object_id']}: bad quaternion/NaN rows {(np.abs(qn - 1) > 1e-3).sum()}")
            mesh = OUT / o["mesh_path"]
            if not mesh.exists():
                problems.append(f"missing mesh {mesh}")
            objs.append((o["object_id"], o.get("object_category"), mesh, tr))
        fl, _ = smplx_qa.body_floor(poses, trans, res.betas, fps)
        rec["floor"] = fl
        rec["objects"] = []
        for oid, cat, mesh, tr in objs:
            d = {"object_id": oid, "category": cat, "mesh_id": mesh.stem}
            if mesh.exists():
                ov, _of = smplx_qa.load_obj_cm(mesh)
                d.update(smplx_qa.object_floor_stats(ov, tr, fl["floor_y_m"], fps))
            rec["objects"].append(d)
        rec["ik_residual_cm"] = {"mean": round(float(res.residuals.mean() * 100), 2),
                                 "p95": round(float(np.percentile(res.residuals, 95) * 100), 2),
                                 "max": round(float(res.residuals.max() * 100), 2)}
        rec["betas"] = [round(float(x), 4) for x in res.betas]
        rec["problems"] = problems
        rec["ok"] = True
    except Exception as e:
        rec["ok"] = False
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["trace"] = traceback.format_exc()[-2000:]
    rec["seconds"] = round(time.time() - t0, 1)
    (dst / "qa.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: rec.get(k) for k in ("clip_id", "ok", "frames", "seconds", "problems", "error")}))


if __name__ == "__main__":
    main()
