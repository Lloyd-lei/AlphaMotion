"""OmniContact raw capture -> UMR HSI/HOI flat SMPL-X folder (sample_data/omnicontact layout).

usage: convert_omni.py <raw_capture_dir> [--max-frames N] [--force]
out:   /home/arenalabs/umr_factory_data/smplx/omnicontact/<domain>/<case>/<capture_id>/
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
import csv
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

RAW = Path("/home/arenalabs/umr_factory_data/raw/omnicontact")
OUT = Path("/home/arenalabs/umr_factory_data/smplx/omnicontact")

XML_TMPL = """<?xml version='1.0' encoding='utf-8'?>
<mujoco model="{low}">
  <compiler angle="radian" autolimits="true" />
  <option timestep="0.002" gravity="0 0 -9.81" />
  <asset>
    <material name="obj_visual" rgba="0.75 0.72 0.66 1" />
    <mesh name="{low}_mesh" file="{name}.obj" scale="0.01 0.01 0.01" />
  </asset>
  <worldbody>
    <body name="{low}" pos="0 0 0">
      <inertial pos="{com}" mass="{mass}" diaginertia="{I}" />
      <geom name="{low}_geom" type="mesh" mesh="{low}_mesh" material="obj_visual" friction="1 0.005 0.0001" contype="1" conaffinity="1" quat="0.707106781 0.707106781 0 0" />
    </body>
  </worldbody>
</mujoco>
"""
# geom quat = +90deg about X (Y-up -> Z-up), copied from UMR's released
# sample_data/omnicontact/soccer/.../ScoccerBall_A_1.xml.


def write_xml(dst_dir: Path, name: str, obj_path: Path):
    import trimesh
    m = trimesh.load(obj_path, force="mesh")
    v = np.asarray(m.vertices) * 0.01
    hull = trimesh.Trimesh(v, process=True).convex_hull
    mass = 0.45 if "ball" in name.lower() else 3.0  # nominal; not used by kinematic retarget
    com = hull.center_mass
    # convert COM from Y-up mesh frame into the rotated geom frame (x, -z, y)
    com_g = np.array([com[0], -com[2], com[1]])
    ext = hull.extents
    I = mass / 12.0 * np.array([ext[1] ** 2 + ext[2] ** 2, ext[0] ** 2 + ext[2] ** 2, ext[0] ** 2 + ext[1] ** 2])
    (dst_dir / f"{name}.xml").write_text(XML_TMPL.format(
        low=name.lower(), name=name, com=" ".join(f"{x:.6f}" for x in com_g), mass=mass,
        I=" ".join(f"{x:.6g}" for x in I)))


def read_csv(path):
    with open(path) as f:
        r = csv.reader(f)
        head = next(r)
        rows = np.asarray([[float(x) for x in row] for row in r], np.float64)
    return head, rows


def main():
    src = Path(sys.argv[1]).resolve()
    max_frames = None
    if "--max-frames" in sys.argv:
        max_frames = int(sys.argv[sys.argv.index("--max-frames") + 1])
    force = "--force" in sys.argv
    meta = json.loads((src / "capture_meta.json").read_text())
    dom, case, cid = meta["domain"], meta["normalized_case"], meta["capture_id"]
    dst = OUT / dom / case / cid
    if (dst / "qa.json").exists() and not force:
        print("skip", dst); return
    dst.mkdir(parents=True, exist_ok=True)
    rec = {"clip_id": f"omnicontact/{dom}/{case}/{cid}", "source": "OmniContact", "raw_dir": str(src)}
    t0 = time.time()
    try:
        bvh = src / meta["human_motion_file"]
        _, ts = read_csv(src / meta["human_timestamps_file"])
        res, layout = bvh2smplx.convert(str(bvh), max_frames=max_frames)
        T = len(res.poses)
        rec["bvh_layout"] = layout
        poses = res.poses.reshape(T, 165).astype(np.float32)
        trans = res.trans.astype(np.float32)
        fps = float(1.0 / res.frame_time)
        rec.update(frames=T, fps=round(fps, 4), bvh_frames_meta=meta.get("frame_count"))
        rec["timestamps_rows"] = int(len(ts))
        problems = []
        if max_frames is None and len(ts) != T:
            problems.append(f"timestamps rows {len(ts)} != bvh frames {T}")
        np.save(dst / "poses.npy", poses)
        np.save(dst / "transl.npy", trans)
        np.save(dst / "betas.npy", res.betas.astype(np.float32))
        np.save(dst / "gender.npy", np.asarray("neutral"))
        np.save(dst / "model_type.npy", np.asarray("smplx"))
        np.save(dst / "mocap_framerate.npy", np.asarray(fps, np.float32))
        np.save(dst / "output_up.npy", np.asarray("y"))
        np.save(dst / "per_frame_residuals.npy", res.residuals.astype(np.float32))
        objs = []
        for o in meta.get("objects", []):
            name = o["object_name"].removeprefix("prop_").removeprefix("object_pose_")
            head, rows = read_csv(src / o["source_file"])
            cols = {h: i for i, h in enumerate(head)}
            if max_frames is not None:
                rows = rows[:max_frames]
            if len(rows) != T:
                problems.append(f"{name}: track rows {len(rows)} != frames {T}")
            if "frame_id" in cols and len(ts) and max_frames is None:
                if not np.array_equal(rows[:, cols["frame_id"]], ts[:len(rows), 0]):
                    problems.append(f"{name}: frame_id mismatch vs motion_actor.csv")
            track = rows[:, [cols[k] for k in ("px", "py", "pz", "qx", "qy", "qz", "qw")]]
            qn = np.linalg.norm(track[:, 3:], axis=1)
            if np.abs(qn - 1).max() > 1e-3 or not np.isfinite(track).all():
                problems.append(f"{name}: bad quaternion/NaN rows {(np.abs(qn - 1) > 1e-3).sum()}")
            with open(dst / f"prop_{name}.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(head)
                for r in rows:
                    w.writerow([str(int(x)) if h == "frame_id" else f"{x:.6f}" if h == "timestamp" else f"{x:.9g}"
                                for h, x in zip(head, r)])
            obj_src = RAW / "assets" / "objects" / f"{name}.obj"
            if not obj_src.exists():
                problems.append(f"{name}: missing asset {obj_src}")
            else:
                shutil.copy2(obj_src, dst / f"{name}.obj")
                write_xml(dst, name, obj_src)
            objs.append((name, track))
        # QA: floor + object
        fl, _low = smplx_qa.body_floor(poses, trans, res.betas, fps)
        rec["floor"] = fl
        rec["objects"] = []
        for name, track in objs:
            ov, of = smplx_qa.load_obj_cm(dst / f"{name}.obj")
            d = {"object_id": name, **smplx_qa.object_floor_stats(ov, track, fl["floor_y_m"], fps)}
            rec["objects"].append(d)
        rec["ik_residual_cm"] = {"mean": round(float(res.residuals.mean() * 100), 2),
                                 "p95": round(float(np.percentile(res.residuals, 95) * 100), 2),
                                 "max": round(float(res.residuals.max() * 100), 2)}
        rec["betas"] = [round(float(x), 4) for x in res.betas]
        rec["problems"] = problems
        rec["ok"] = True
    except Exception as e:  # record and continue
        rec["ok"] = False
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["trace"] = traceback.format_exc()[-2000:]
    rec["seconds"] = round(time.time() - t0, 1)
    (dst / "qa.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: rec.get(k) for k in ("clip_id", "ok", "frames", "seconds", "problems", "error")}))


if __name__ == "__main__":
    main()
