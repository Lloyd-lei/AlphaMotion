"""CPU-side stages of the GPU converter (run in worker processes). Everything here is upstream
hiphi2smplx / umr_factory code or a byte-identical re-implementation of it:

  prep_clip   : BVH parse (same values as hiphi2smplx.bvh.load_bvh, ~6x faster numeric parse), omni51 SpineMid
                insertion (bvh2smplx), skeleton map, prepare_body_inputs (targets + copied rotations), first-frame
                beta-fit targets (_prepare_target_joints with the neutral body)
  shape_clip  : load_smplx_body(betas) (cached model arrays, same float64 ops) + equivalent_foot_targets
  write_hiphi : convert_hiphi.py output layout (npz keys, metadata/object_tracks copies, qa.json with the same
                checks, floor/object QA through smplx_qa) + optional 30 fps variant
  write_omni  : convert_omni.py output layout
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, "/home/arenalabs/umr_factory_data/scripts")
import bvh2smplx  # noqa: E402  (also patches the omni51 skeleton map)
import smplx_qa  # noqa: E402
from hiphi2smplx import skeleton as S  # noqa: E402
from hiphi2smplx.beta_fitting import _prepare_target_joints  # noqa: E402
from hiphi2smplx.bvh import _BvhJoint, load_bvh as upstream_load_bvh  # noqa: E402
from hiphi2smplx.ik_solver import equivalent_foot_targets, prepare_body_inputs  # noqa: E402
from hiphi2smplx.smplx import SmplxBody, _sole_pitch_rotation, load_smplx_body  # noqa: E402

import torch  # noqa: E402

torch.set_num_threads(1)

MODEL = bvh2smplx.MODEL
GPU_PIPELINE = ("gpu_convert v1 (/home/arenalabs/gpu_convert): batched GPU re-implementation of hiphi2smplx@65e5131 "
                "body IK (bit-exact Mink/MuJoCo math, float64) + float64 multi-start Newton beta fit; body only, fingers zero")
_NEUTRAL = None
_MODEL_CACHE = None


def neutral():
    global _NEUTRAL
    if _NEUTRAL is None:
        _NEUTRAL = load_smplx_body(MODEL, np.zeros(10, dtype=np.float32))
    return _NEUTRAL


# ------------------------------------------------------------------------------------------------ BVH
def fast_load_bvh(path, unit_scale=0.01) -> S.MotionClip:
    """hiphi2smplx.bvh.load_bvh with the motion block parsed by np.fromstring (strtod, correctly rounded
    float64 -> float32, identical to float() -> np.asarray(float32)). Falls back to upstream on any mismatch."""
    source = Path(path)
    lines = source.read_text(encoding="utf-8", errors="ignore").splitlines()
    joints, stack = [], []
    channel_cursor = 0
    cursor = 0
    while cursor < len(lines):
        line = lines[cursor].strip()
        if line == "MOTION":
            cursor += 1
            break
        if line.startswith(("ROOT ", "JOINT ")):
            parent = next((item for item in reversed(stack) if isinstance(item, int)), -1)
            joints.append(_BvhJoint(line.split(maxsplit=1)[1].strip(), int(parent)))
            stack.append(len(joints) - 1)
        elif line.startswith("End Site"):
            stack.append("END")
        elif line.startswith("OFFSET ") and stack and isinstance(stack[-1], int):
            joints[stack[-1]].offset = np.asarray(line.split()[1:4], dtype=np.float32)
        elif line.startswith("CHANNELS ") and stack and isinstance(stack[-1], int):
            parts = line.split()
            count = int(parts[1])
            joints[stack[-1]].channels = parts[2:2 + count]
            joints[stack[-1]].channel_indices = list(range(channel_cursor, channel_cursor + count))
            channel_cursor += count
        elif line == "}" and stack:
            stack.pop()
        cursor += 1
    if not joints or cursor + 1 >= len(lines):
        return upstream_load_bvh(path, unit_scale=unit_scale)
    frames = int(lines[cursor].split(":", 1)[1])
    frame_time = float(lines[cursor + 1].split(":", 1)[1])
    motion_lines = lines[cursor + 2: cursor + 2 + frames]
    flat = np.fromstring("\n".join(motion_lines), dtype=np.float64, sep=" ")
    nonblank = sum(1 for ln in motion_lines if ln.strip())
    if flat.size != frames * channel_cursor or nonblank != frames:
        return upstream_load_bvh(path, unit_scale=unit_scale)
    values = flat.astype(np.float32).reshape(frames, channel_cursor)
    local_rotations = np.zeros((frames, len(joints), 3), dtype=np.float32)
    local_translations = np.broadcast_to(np.stack([j.offset for j in joints])[None], (frames, len(joints), 3)).copy()
    root_translations = np.zeros((frames, 3), dtype=np.float32)
    for joint_index, joint in enumerate(joints):
        joint_values = values[:, joint.channel_indices]
        rotation_indices = [i for i, ch in enumerate(joint.channels) if ch.lower().endswith("rotation")]
        order = "".join(joint.channels[i][0].upper() for i in rotation_indices)
        if order:
            local_rotations[:, joint_index] = Rotation.from_euler(
                order, joint_values[:, rotation_indices], degrees=True).as_rotvec().astype(np.float32)
        position_indices = {ch[0].lower(): i for i, ch in enumerate(joint.channels) if ch.lower().endswith("position")}
        destination = root_translations if joint_index == 0 else local_translations[:, joint_index]
        if joint_index == 0:
            destination[:] = joint.offset
        for axis, component in (("x", 0), ("y", 1), ("z", 2)):
            if axis in position_indices:
                destination[:, component] = joint_values[:, position_indices[axis]]
    offsets = np.stack([j.offset for j in joints]).astype(np.float32)
    offsets *= unit_scale
    offsets[0] = 0.0
    local_translations *= unit_scale
    local_translations[:, 0] = 0.0
    root_translations *= unit_scale
    skeleton = S.Skeleton(tuple(j.name for j in joints), np.asarray([j.parent for j in joints], dtype=np.int64), offsets)
    return S.MotionClip(skeleton=skeleton, local_rotations=local_rotations, root_translations=root_translations,
                        frame_time=frame_time, local_translations=local_translations)


def load_bvh_any(path):
    clip = fast_load_bvh(path, unit_scale=0.01)
    if "Spine4" not in clip.skeleton.joint_names and "Spine1" in clip.skeleton.joint_names:
        clip = bvh2smplx._insert_spinemid(clip)
    return clip


def prep_clip(bvh_path: str, max_frames=None):
    """-> dict(targets f32 (F,22,3), reference f32 (F,22,3), beta_target f32 (22,3), frame_time, layout, frames)."""
    t0 = time.time()
    layout = bvh2smplx.bvh_layout(bvh_path)
    clip = load_bvh_any(bvh_path)
    if max_frames is not None:
        clip = clip.sliced(min(clip.num_frames, max_frames))
    mapping = S.map_skeleton(clip.skeleton, layout)
    nb = neutral()
    targets, reference = prepare_body_inputs(clip, mapping, nb.skeleton)
    beta_target = _prepare_target_joints(targets[0], nb.rest_joints)
    return dict(targets=targets, reference=reference, beta_target=beta_target, frame_time=float(clip.frame_time),
                layout=layout, frames=int(clip.num_frames), prep_s=time.time() - t0)


# ------------------------------------------------------------------------------------------------ shape
def _model_arrays():
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        with np.load(Path(MODEL), allow_pickle=True) as model:
            template = np.asarray(model["v_template"], dtype=np.float64)
            shapedirs = np.asarray(model["shapedirs"], dtype=np.float64)
            weights = np.asarray(model["weights"], dtype=np.float64)
            regressor = model["J_regressor"]
            if hasattr(regressor, "toarray"):
                regressor = regressor.toarray()
            parents = np.asarray(model["kintree_table"], dtype=np.int64)[0].copy()
        _MODEL_CACHE = (template, shapedirs, weights, np.asarray(regressor, dtype=np.float64), parents)
    return _MODEL_CACHE


def load_smplx_body_cached(betas) -> SmplxBody:
    """hiphi2smplx.smplx.load_smplx_body with the model arrays read once per process (same float64 ops)."""
    coefficients = np.asarray(betas, dtype=np.float64).reshape(-1)
    template, shapedirs, weights, regressor, parents = _model_arrays()
    parents = parents.copy()
    beta_count = min(coefficients.size, shapedirs.shape[-1])
    vertices = template + np.einsum("vcn,n->vc", shapedirs[..., :beta_count], coefficients[:beta_count])
    joints = regressor @ vertices
    parents[parents > 1_000_000] = -1
    body_joints = joints[:22].astype(np.float32)
    body_parents = parents[:22]
    offsets = body_joints.copy()
    for joint, parent in enumerate(body_parents):
        if parent >= 0:
            offsets[joint] -= body_joints[parent]
    skeleton = S.Skeleton(S.SMPLX_BODY_JOINT_NAMES, body_parents, offsets)
    sole = (_sole_pitch_rotation(vertices, weights, (7, 10)), _sole_pitch_rotation(vertices, weights, (8, 11)))
    return SmplxBody(skeleton=skeleton, rest_joints=body_joints, sole_rotations=sole)


def shape_clip(betas, targets, reference):
    body = load_smplx_body_cached(np.asarray(betas, np.float32))
    fit_targets = equivalent_foot_targets(targets, body.rest_joints, reference, body.sole_rotations)
    return fit_targets, body.rest_joints


# ------------------------------------------------------------------------------------------------ writers
def _floor_and_objects(lows, fps, objs):
    fl = smplx_qa.floor_stats(lows[:, 1].astype(np.float32), lows, fps)
    out = []
    for oid, cat, mesh, tr in objs:
        d = {"object_id": oid, "category": cat, "mesh_id": mesh.stem}
        if mesh.exists():
            ov, _of = smplx_qa.load_obj_cm(mesh)
            d.update(smplx_qa.object_floor_stats(ov, tr, fl["floor_y_m"], fps))
        out.append(d)
    return fl, out


def _residual_block(residuals):
    return {"mean": round(float(residuals.mean() * 100), 2), "p95": round(float(np.percentile(residuals, 95) * 100), 2),
            "max": round(float(residuals.max() * 100), 2)}


def write_hiphi(job):
    """job: dict(src, out_root, meta, rec_base, res(poses55 f32 (T,55,3), trans, betas, joints, losses, fit_targets,
    frame_time, layout), lows (T,3) f32, gpu (dict), seconds, fps30_root or None)."""
    src = Path(job["src"])
    out_root = Path(job["out_root"])
    meta = job["meta"]
    dst = out_root / "data" / meta["frame"] / meta["lu"] / meta["motion_id"]
    dst.mkdir(parents=True, exist_ok=True)
    rec = dict(job["rec_base"])
    t0 = time.time()
    try:
        if job.get("error"):
            raise RuntimeError(job["error"])
        r = job["res"]
        T = len(r["poses55"])
        poses = r["poses55"].reshape(T, 165).astype(np.float32)
        trans = r["trans"].astype(np.float32)
        residuals = np.linalg.norm(r["joints"] - r["fit_targets"], axis=-1).mean(axis=1).astype(np.float32)
        fps = float(1.0 / r["frame_time"])
        rec.update(bvh_layout=r["layout"], frames=T, fps=round(fps, 4), frames_meta=meta.get("frame_count"))
        problems = []
        if meta.get("frame_count") != T:
            problems.append(f"metadata frame_count {meta.get('frame_count')} != bvh {T}")
        arrays = dict(poses=poses, trans=trans, transl=trans, betas=r["betas"].astype(np.float32),
                      gender=np.asarray("neutral"), model_type=np.asarray("smplx"),
                      mocap_framerate=np.asarray(fps, np.float32), fps=np.asarray(fps, np.float32),
                      frame_time=np.asarray(r["frame_time"]), coordinate_up=np.asarray("y"), output_up=np.asarray("y"),
                      num_joints=np.asarray(55, np.int32), fitted_joints=r["joints"].astype(np.float32),
                      per_frame_residuals=residuals, final_losses=r["losses"].astype(np.float32),
                      pipeline=np.asarray(GPU_PIPELINE), source_bvh=np.asarray(str(src / "motion_actor.bvh")))
        tmp = dst / ".motion_actor_smplx.tmp.npz"
        np.savez_compressed(tmp, **arrays)
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
            mesh = out_root / o["mesh_path"]
            if not mesh.exists():
                problems.append(f"missing mesh {mesh}")
            objs.append((o["object_id"], o.get("object_category"), mesh, tr, tp))
        lows = job["lows"]
        fl, ostats = _floor_and_objects(lows, fps, [o[:4] for o in objs])
        rec["floor"] = fl
        rec["objects"] = ostats
        rec["ik_residual_cm"] = _residual_block(residuals)
        rec["betas"] = [round(float(x), 4) for x in r["betas"]]
        rec["problems"] = problems
        rec["ok"] = True
        rec["gpu_convert"] = job["gpu"]
        if job.get("fps30_root"):
            rec["variant_30fps"] = _write_30fps(job, dst, arrays, objs, residuals, rec)
    except Exception as e:
        rec["ok"] = False
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["trace"] = traceback.format_exc()[-2000:]
    rec["seconds"] = round(job.get("seconds", 0.0) + time.time() - t0, 2)
    (dst / "qa.json").write_text(json.dumps(rec, indent=1))
    return {k: rec.get(k) for k in ("clip_id", "ok", "frames", "seconds", "problems", "error")}


def _write_30fps(job, dst_master, arrays, objs, residuals, rec_master, stride=3):
    """Stride-3 subsample of the 90 fps master (no re-solve): same layout under <fps30_root>/data/..."""
    meta = dict(job["meta"])
    root = Path(job["fps30_root"])
    dst = root / "data" / meta["frame"] / meta["lu"] / meta["motion_id"]
    dst.mkdir(parents=True, exist_ok=True)
    T = len(arrays["poses"])
    idx = np.arange(0, T, stride)
    fps = float(arrays["fps"]) / stride
    a = dict(arrays)
    for k in ("poses", "trans", "transl", "fitted_joints", "per_frame_residuals", "final_losses"):
        a[k] = arrays[k][idx]
    a["fps"] = np.asarray(fps, np.float32)
    a["mocap_framerate"] = np.asarray(fps, np.float32)
    a["frame_time"] = np.asarray(float(arrays["frame_time"]) * stride)
    a["pipeline"] = np.asarray(str(arrays["pipeline"]) + f"; 30fps variant = master[::{stride}]")
    tmp = dst / ".motion_actor_smplx.tmp.npz"
    np.savez_compressed(tmp, **a)
    tmp.replace(dst / "motion_actor_smplx.npz")
    src_meta = json.loads((Path(job["src"]) / "metadata.json").read_text())
    src_meta.update(fps=round(src_meta["fps"] / stride, 6), frame_count=int(len(idx)),
                    downsampled_from={"fps": src_meta["fps"], "frame_count": src_meta["frame_count"], "stride": stride})
    (dst / "metadata.json").write_text(json.dumps(src_meta, indent=2, sort_keys=True))
    objs30 = []
    for oid, cat, mesh, tr, tp in objs:
        lines = tp.read_text().splitlines()
        head, body = lines[0], lines[1:]
        sel = [body[i] for i in idx if i < len(body)]
        out = [head] + [",".join([str(n)] + ln.split(",")[1:]) for n, ln in enumerate(sel)]
        p = dst / tp.relative_to(Path(job["src"]))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(out) + "\n")
        mesh30 = root / meta["objects"][[o["object_id"] for o in meta["objects"]].index(oid)]["mesh_path"]
        objs30.append((oid, cat, mesh30, tr[idx[idx < len(tr)]]))
    fl, ostats = _floor_and_objects(job["lows"][idx], fps, objs30)
    rec = {k: v for k, v in rec_master.items() if k not in ("floor", "objects", "ik_residual_cm", "variant_30fps")}
    rec.update(frames=int(len(idx)), fps=round(fps, 4), frames_meta=int(len(idx)), floor=fl, objects=ostats,
               ik_residual_cm=_residual_block(residuals[idx]), variant=f"30fps = 90fps master[::{stride}]",
               master_dir=str(dst_master))
    (dst / "qa.json").write_text(json.dumps(rec, indent=1))
    return str(dst)


def write_omni(job):
    """convert_omni.py layout: flat .npy files + prop_<name>.csv + <name>.obj/.xml + qa.json."""
    import convert_omni as CO  # noqa: E402  (only helper functions are used)
    src = Path(job["src"])
    out_root = Path(job["out_root"])
    meta = job["meta"]
    dst = out_root / meta["domain"] / meta["normalized_case"] / meta["capture_id"]
    dst.mkdir(parents=True, exist_ok=True)
    rec = dict(job["rec_base"])
    t0 = time.time()
    try:
        if job.get("error"):
            raise RuntimeError(job["error"])
        r = job["res"]
        _, ts = CO.read_csv(src / meta["human_timestamps_file"])
        T = len(r["poses55"])
        rec["bvh_layout"] = r["layout"]
        poses = r["poses55"].reshape(T, 165).astype(np.float32)
        trans = r["trans"].astype(np.float32)
        residuals = np.linalg.norm(r["joints"] - r["fit_targets"], axis=-1).mean(axis=1).astype(np.float32)
        fps = float(1.0 / r["frame_time"])
        rec.update(frames=T, fps=round(fps, 4), bvh_frames_meta=meta.get("frame_count"))
        rec["timestamps_rows"] = int(len(ts))
        problems = []
        if len(ts) != T:
            problems.append(f"timestamps rows {len(ts)} != bvh frames {T}")
        np.save(dst / "poses.npy", poses)
        np.save(dst / "transl.npy", trans)
        np.save(dst / "betas.npy", r["betas"].astype(np.float32))
        np.save(dst / "gender.npy", np.asarray("neutral"))
        np.save(dst / "model_type.npy", np.asarray("smplx"))
        np.save(dst / "mocap_framerate.npy", np.asarray(fps, np.float32))
        np.save(dst / "output_up.npy", np.asarray("y"))
        np.save(dst / "per_frame_residuals.npy", residuals)
        objs = []
        import csv
        for o in meta.get("objects", []):
            name = o["object_name"].removeprefix("prop_").removeprefix("object_pose_")
            head, rows = CO.read_csv(src / o["source_file"])
            cols = {h: i for i, h in enumerate(head)}
            if len(rows) != T:
                problems.append(f"{name}: track rows {len(rows)} != frames {T}")
            if "frame_id" in cols and len(ts):
                if not np.array_equal(rows[:, cols["frame_id"]], ts[:len(rows), 0]):
                    problems.append(f"{name}: frame_id mismatch vs motion_actor.csv")
            track = rows[:, [cols[k] for k in ("px", "py", "pz", "qx", "qy", "qz", "qw")]]
            qn = np.linalg.norm(track[:, 3:], axis=1)
            if np.abs(qn - 1).max() > 1e-3 or not np.isfinite(track).all():
                problems.append(f"{name}: bad quaternion/NaN rows {(np.abs(qn - 1) > 1e-3).sum()}")
            with open(dst / f"prop_{name}.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(head)
                for rr in rows:
                    w.writerow([str(int(x)) if h == "frame_id" else f"{x:.6f}" if h == "timestamp" else f"{x:.9g}"
                                for h, x in zip(head, rr)])
            obj_src = CO.RAW / "assets" / "objects" / f"{name}.obj"
            if not obj_src.exists():
                problems.append(f"{name}: missing asset {obj_src}")
            else:
                shutil.copy2(obj_src, dst / f"{name}.obj")
                CO.write_xml(dst, name, obj_src)
            objs.append((name, track))
        fl = smplx_qa.floor_stats(job["lows"][:, 1].astype(np.float32), job["lows"], fps)
        rec["floor"] = fl
        rec["objects"] = []
        for name, track in objs:
            ov, of = smplx_qa.load_obj_cm(dst / f"{name}.obj")
            rec["objects"].append({"object_id": name, **smplx_qa.object_floor_stats(ov, track, fl["floor_y_m"], fps)})
        rec["ik_residual_cm"] = _residual_block(residuals)
        rec["betas"] = [round(float(x), 4) for x in r["betas"]]
        rec["problems"] = problems
        rec["ok"] = True
        rec["gpu_convert"] = job["gpu"]
    except Exception as e:
        rec["ok"] = False
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["trace"] = traceback.format_exc()[-2000:]
    rec["seconds"] = round(job.get("seconds", 0.0) + time.time() - t0, 2)
    (dst / "qa.json").write_text(json.dumps(rec, indent=1))
    return {k: rec.get(k) for k in ("clip_id", "ok", "frames", "seconds", "problems", "error")}


def selftest(bvh_path):
    a = upstream_load_bvh(bvh_path)
    b = fast_load_bvh(bvh_path)
    same = all(np.array_equal(getattr(a, k), getattr(b, k)) for k in ("local_rotations", "root_translations", "local_translations"))
    same &= np.array_equal(a.skeleton.rest_offsets, b.skeleton.rest_offsets) and a.frame_time == b.frame_time
    return same
