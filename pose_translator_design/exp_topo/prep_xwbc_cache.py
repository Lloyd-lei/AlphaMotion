"""Build the XWBC-LAFAN strictly clip-paired cache: 17 robots + 1 human (human_smpl),
77 LAFAN1 clips each, from TonyZhang1999/xwbc-retarget-data (extracted npz).

Sources (see pose_translator_design/XWBC_ASSESSMENT.md):
  /var/tmp/alphamotion/xwbc_lafan/extracted/{emb}/{clip}.npz   30 fps
  robot npz : joint_pos [T,ndof] (Isaac-Lab BFS order, remap by NAME), body_pos_w,
              body_quat_w (wxyz), joint_names (fallback: xwbc_joint_names.json).
              World frame Z-up.
  human npz : human_motion_v1 schema, 22 SMPL bodies (body_pos_w/body_quat_w only,
              no joint angles). LAFAN human is ALSO Z-up (AMASS human is Y-up — not
              used here). Canonical skeleton: verified identical rest offsets across
              subjects (0.0 mm), so ONE spec + the exact FK round-trip gate apply.

Pipeline per robot (reuses prep_gmr_cache machinery):
  qpos rebuild  [root pos, root quat wxyz, hinge angles remapped by name]
  -> MuJoCo FK (GMR XMLs; unitree_h2/r1 from unitree_ros generated MJCFs;
     unitree_g1_23dof runs on the 29dof XML with the 6 absent joints zero-filled —
     same kinematic tree with those joints locked at 0 on the 23dof hardware)
  -> Y-up harmonization (AX conjugation, prep_gmr_cache v2 convention)
  -> LOCAL rot6d with the chain-end frame attribution fix (build_merge/build_spec).
Human: body_quat_w -> AX-conjugated globals -> local6d against the SMPL parent table.

Frame alignment: clips are same length across embodiments up to a ~2-frame human
tail; every clip is truncated to the min length over all embodiments. Full takes are
kept (no capping); sequences.json carries per-clip {name,start,len}.

Gates (VALIDATION.json): per-embodiment FK round-trip < 0.1 cm (human included),
NaN == 0, rows identical across embodiments, uprightness (top joint peaks on axis 1
= Y) for ALL embodiments, per-clip length spread, plus a data-vs-XML probe:
FK body positions must match stored body_pos_w (nearest-neighbor) to < 5 mm, which
catches any wrong/incompatible MJCF. Robots failing the probe or joint mapping are
SKIPPED and recorded, never block the build.

Excluded: berkeley_humanoid_lite (lloyd-flagged below-ground retargets).

A/B pilot: manifest.json keeps ALL embodiments. Model A excludes the g1 variants at
train time via  --holdout unitree_g1_29dof,unitree_g1_23dof  (train_dual comma list);
model B trains on everything. pilot_ab.json documents both lists.

CPU only (nice 10), multiprocessing. No trainings started here.
"""
from __future__ import annotations
import os, sys, json, time, glob, argparse, zipfile, importlib.util
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SkeletonSpec, fk_global, CACHE_ROOT
import prep_gmr_cache as PG   # AX, CM, build_merge, build_spec, qpos_to_global,
                              # to_local6d, save_spec, upright_check, _worker_init

EXT = "/var/tmp/alphamotion/xwbc_lafan/extracted"
OUT_DEFAULT = f"{CACHE_ROOT}/cache_xwbc_lafan"
JN_JSON = "/var/tmp/alphamotion/xwbc_joint_names.json"
UNITREE_ROS = "/var/tmp/alphamotion/robot_models/unitree_ros/robots"
HUMAN_SRC, HUMAN_NAME = "human_motion", "human_smpl"
EXCLUDE = {"berkeley_humanoid_lite"}
FK_GATE_CM, PROBE_GATE_MM = PG.FK_GATE_CM, 5.0

# dataset embodiment -> GMR params.py key ("@" prefix = explicit path)
XML_KEY = {
    "unitree_g1_29dof": "unitree_g1",     # g1_mocap_29dof.xml
    "unitree_g1_23dof": "unitree_g1",     # 29dof tree, absent joints locked at 0
    "unitree_h1": "unitree_h1", "unitree_h1_2": "unitree_h1_2",
    "unitree_h2": f"@{UNITREE_ROS}/h2_description/unitree_h2_mjcf.xml",
    "unitree_r1": f"@{UNITREE_ROS}/r1_description/unitree_r1_mjcf.xml",
    "booster_k1": "booster_k1", "booster_t1": "booster_t1",
    "booster_t1_29dof": "booster_t1_29dof", "engineai_pm01": "engineai_pm01",
    "fourier_gr3": "fourier_gr3", "fourier_n1": "fourier_n1",
    "hightorque_hi": "hightorque_hi", "kuavo_s45": "kuavo_s45",
    "pal_talos": "pal_talos", "pnd_adam_lite": "pnd_adam_lite",
    "tienkung": "tienkung",
}

# data joints with no counterpart in the (mocap) XML — leaf head/gripper DOFs,
# dropped from the cache for these robots (motion of those leaves is lost, noted):
DROP_JOINTS = {"booster_t1_29dof": {"AAHead_yaw", "Head_pitch"},
               "pal_talos": {"gripper_left_joint", "gripper_right_joint"}}
# robots whose available MJCF disagrees geometrically with the data (h2: ~40 mm on
# several bodies for BOTH unitree_ros MJCFs; pal_talos: 20.6 mm vs GMR talos.xml
# even with gripper bodies excused) -> skeleton taken directly from the stored body
# kinematics (human-style path, tree inferred from rigid-offset constancy, joint
# names borrowed from the nearest MJCF body where unambiguous). Side benefit for
# talos: gripper motion is kept instead of dropped.
DATA_TREE = {"unitree_h2", "pal_talos"}

# Joint LIMITS to encode into {emb}_dof.npz when the MJCF we do kinematics from is
# not the vendor's own model.  GMR's g1_mocap_29dof.xml hand-narrows 15 of G1's 29
# ranges (hips 130 deg lost, waist_yaw 120, shoulders 105/87/57, elbow 22.6, ...) to
# keep its IK retargeter tame; the geometry, axes and joint set are identical to
# Unitree's URDF.  The limits are fed to the network as INPUT FEATURES, so they must
# describe the real robot, not GMR's solver box.  See LIMITS_AUDIT.md 1.
# Verified byte-identical to upstream unitreerobotics/unitree_ros g1_29dof.urdf,
# g1_29dof_rev_1_0.urdf and the g1_29dof_mode_{11,13,15,18} descriptions.
_G1_VENDOR_URDF = f"{PG.GMR_REPO}/assets/unitree_g1/g1_custom_collision_29dof.urdf"
LIMIT_URDF = {"unitree_g1_29dof": _G1_VENDOR_URDF,
              "unitree_g1_23dof": _G1_VENDOR_URDF}


def urdf_ranges(path):
    """URDF -> {joint_name: (lower, upper)} in radians, movable joints only."""
    import xml.etree.ElementTree as ET
    out = {}
    for j in ET.parse(path).getroot().iter("joint"):
        lim = j.find("limit")
        if j.get("type") == "fixed" or lim is None:
            continue
        out[j.get("name")] = (float(lim.get("lower")), float(lim.get("upper")))
    return out


def limit_override(emb):
    """-> {joint_name: (lo, hi)} rad for `emb`, or None when the MJCF is the vendor's."""
    p = LIMIT_URDF.get(emb)
    return urdf_ranges(p) if p else None

SMPL_NAMES = ['Pelvis', 'L_Hip', 'R_Hip', 'Torso', 'L_Knee', 'R_Knee', 'Spine',
              'L_Ankle', 'R_Ankle', 'Chest', 'L_Toe', 'R_Toe', 'Neck', 'L_Thorax',
              'R_Thorax', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow',
              'L_Wrist', 'R_Wrist']
SMPL_PARENTS = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9,
                         12, 13, 14, 16, 17, 18, 19], np.int64)


def resolve_xml(emb):
    k = XML_KEY[emb]
    if k.startswith("@"):
        return k[1:]
    p = f"{PG.GMR_REPO}/general_motion_retargeting/params.py"
    s = importlib.util.spec_from_file_location("gmr_params", p)
    mod = importlib.util.module_from_spec(s)
    s.loader.exec_module(mod)
    return str(mod.ROBOT_XML_DICT[k])


def npz_shape(path, key):
    """Array shape from the zip header only (avoids reading GB of data for lengths)."""
    with zipfile.ZipFile(path) as z, z.open(key + ".npy") as f:
        ver = np.lib.format.read_magic(f)
        shape, _, _ = np.lib.format._read_array_header(f, ver)
    return shape


def quat_to_mat(q):
    """wxyz [...,4] -> [...,3,3], numpy."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def clip_names():
    names = None
    for d in sorted(os.listdir(EXT)):
        s = {f[:-4] for f in os.listdir(f"{EXT}/{d}") if f.endswith(".npz")}
        names = s if names is None else names & s
    return sorted(names)


# --------------------------------- human side ---------------------------------

def build_human_spec(ref_clip):
    d = np.load(f"{EXT}/{HUMAN_SRC}/{ref_clip}.npz", allow_pickle=True)
    assert [str(n) for n in d["body_names"]] == SMPL_NAMES, list(d["body_names"])
    p = d["body_pos_w"].astype(np.float64)
    R = quat_to_mat(d["body_quat_w"].astype(np.float64))
    off = np.zeros((22, 3))
    for j, par in enumerate(SMPL_PARENTS):
        if par >= 0:
            off[j] = np.einsum("tab,ta->tb", R[:, par], p[:, j] - p[:, par]).mean(0)
    off[0] = [0.0, 0.0, float(p[:, 0, 2].mean())]         # standing height, Z-up
    off = (off @ PG.AX.T) * PG.CM                          # Z-up -> Y-up, m -> cm
    spec = SkeletonSpec(HUMAN_NAME, SMPL_PARENTS, off.astype(np.float32), SMPL_NAMES)
    spec.order_root_first()
    return spec


def human_globals(npz, K):
    d = np.load(npz)
    assert int(np.asarray(d["fps"]).ravel()[0]) == 30
    R = quat_to_mat(d["body_quat_w"][:K].astype(np.float64))
    P = d["body_pos_w"][:K].astype(np.float64)
    R = (PG.AX[None, None] @ R @ PG.AX.T[None, None]).astype(np.float32)
    P = (P @ PG.AX.T).astype(np.float32)
    P = (P - P[:, :1]) * PG.CM
    return R, P


# --------------------------------- robot side ---------------------------------

def data_joint_names(emb):
    return json.load(open(JN_JSON))[emb]["joint_names"]


def joint_map(model, emb):
    """(qpos column of each npz joint or -1-> zero-fill, per model hinge). Returns
    (hinge qpos addrs [nh], source npz columns [nh] with -1 = fill 0, n_zero).
    Name match is exact first, case-insensitive fallback (engineai_pm01 data names
    are lowercased XML names). DROP_JOINTS[emb] columns are intentionally unused."""
    import mujoco
    jn = data_joint_names(emb)
    src = {n: i for i, n in enumerate(jn)}
    src_lc = {n.lower(): i for i, n in enumerate(jn)}
    addrs, cols, used = [], [], set()
    assert model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE, emb
    for j in range(1, model.njnt):
        assert model.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE,
                                     mujoco.mjtJoint.mjJNT_SLIDE), emb
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        c = src.get(name, src_lc.get(name.lower(), -1))
        addrs.append(int(model.jnt_qposadr[j]))
        cols.append(c)
        if c >= 0:
            used.add(jn[c])
    missing = set(jn) - used - DROP_JOINTS.get(emb, set())
    if missing:
        raise RuntimeError(f"{emb}: data joints not in XML: {sorted(missing)}")
    return np.array(addrs), np.array(cols), int((np.array(cols) < 0).sum())


# ---- DATA_TREE robots: skeleton straight from stored body kinematics ----

def infer_parents(P, R, n_probe=200):
    """[T,B,3]/[T,B,3,3] world kinematics -> parent table by rigid-offset constancy:
    parent(j) = argmin_{i<j} std_t | R_i^T (p_j - p_i) |. Returns (parents, max_std_m)."""
    T = P.shape[0]
    ts = np.linspace(0, T - 1, min(n_probe, T)).astype(int)
    P, R = P[ts].astype(np.float64), R[ts].astype(np.float64)
    B = P.shape[1]
    parents, worst = [-1], 0.0
    for j in range(1, B):
        loc = np.einsum("tiab,tia->tib", R[:, :j], P[:, [j]] - P[:, :j])  # [t,j,3]
        s = loc.std(0).max(-1)                                            # [j]
        parents.append(int(s.argmin()))
        worst = max(worst, float(s.min()))
    return np.array(parents, np.int64), worst


def build_data_spec(emb, ref_clip):
    d = np.load(f"{EXT}/{emb}/{ref_clip}.npz")
    P = d["body_pos_w"]
    R = quat_to_mat(d["body_quat_w"].astype(np.float64))
    parents, drift_m = infer_parents(P, R)
    assert drift_m < 1e-3, (emb, drift_m)
    B = P.shape[1]
    off = np.zeros((B, 3))
    for j in range(1, B):
        p = parents[j]
        off[j] = np.einsum("tab,ta->tb", R[:, p], P[:, j].astype(np.float64)
                           - P[:, p].astype(np.float64)).mean(0)
    # root bone = the robot's REAL standing (sole -> pelvis) height from the MJCF's
    # mesh geometry.  Was the mean pelvis z over the retargeted motion, which is the
    # height of a MOVING (crouched) pelvis: -15.4 cm on pal_talos, -6.1 cm on
    # unitree_h2 (LIMITS_AUDIT.md 5b).  The MJCF is only geometrically unreliable for
    # these two at the ~2-4 cm bone level, which is why DATA_TREE bypasses it for the
    # tree; its overall standing height is still by far the best estimate we have.
    root_h = float(P[:, 0, 2].mean())
    try:
        import mujoco
        root_h = PG.standing_height(mujoco.MjModel.from_xml_path(resolve_xml(emb)))
    except Exception as e:
        print(f"[{emb}] standing_height failed ({e}); using mean pelvis z", flush=True)
    off[0] = [0.0, 0.0, root_h]
    off = (off @ PG.AX.T) * PG.CM
    names = [f"body{j:02d}" for j in range(B)]
    try:                                     # borrow names from the MJCF where unambiguous
        import mujoco
        m = mujoco.MjModel.from_xml_path(resolve_xml(emb))
        dd = mujoco.MjData(m)
        addrs, cols, _ = joint_map(m, emb)
        dd.qpos[:] = rebuild_qpos(m, d, addrs, cols, 1)[0]
        mujoco.mj_kinematics(m, dd)
        nn = np.linalg.norm(P[0][:, None] - dd.xpos[None], axis=-1).argmin(1)
        if len(set(nn.tolist())) == B:
            names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(b)) or names[j]
                     for j, b in enumerate(nn)]
    except Exception:
        pass
    spec = SkeletonSpec(emb, parents, off.astype(np.float32), names)
    spec.order_root_first()
    return spec


def data_tree_globals(emb, npz, K):
    d = np.load(npz)
    assert int(np.asarray(d["fps"]).ravel()[0]) == 30
    R = quat_to_mat(d["body_quat_w"][:K].astype(np.float64))
    P = d["body_pos_w"][:K].astype(np.float64)
    R = (PG.AX[None, None] @ R @ PG.AX.T[None, None]).astype(np.float32)
    P = (P @ PG.AX.T).astype(np.float32)
    P = (P - P[:, :1]) * PG.CM
    return R, P


def load_robot(emb):
    import mujoco
    model = mujoco.MjModel.from_xml_path(resolve_xml(emb))
    data = mujoco.MjData(model)
    keep, frame = PG.build_merge(model)
    spec = PG.build_spec(model, emb, keep)
    addrs, cols, n_zero = joint_map(model, emb)
    return model, data, spec, keep, frame, addrs, cols, n_zero


def rebuild_qpos(model, npz_d, addrs, cols, K):
    jp = npz_d["joint_pos"][:K].astype(np.float64)
    q = np.zeros((K, model.nq))
    q[:, 0:3] = npz_d["body_pos_w"][:K, 0].astype(np.float64)
    q[:, 3:7] = npz_d["body_quat_w"][:K, 0].astype(np.float64)   # wxyz = MuJoCo order
    for a, c in zip(addrs, cols):
        if c >= 0:
            q[:, a] = jp[:, c]
    return q


def probe_robot(emb, clip):
    """Data-vs-XML gate: FK vs stored body_pos_w (nearest-neighbor over model bodies,
    3 probe frames). Catches wrong MJCF / joint-order mistakes. The bodies moved by
    DROP_JOINTS (head/grippers absent from the XML) are excused: the worst
    n_dropped npz bodies are reported separately, the gate applies to the rest.
    DATA_TREE robots skip the XML entirely; their gate is the FK round-trip of the
    inferred skeleton. Returns (gated err_mm, n_zero, J, note)."""
    import mujoco
    if emb in DATA_TREE:
        spec = build_data_spec(emb, clip)
        R, P = data_tree_globals(emb, f"{EXT}/{emb}/{clip}.npz", 10 ** 9)
        l6 = PG.to_local6d(R, spec.parents)
        _, gp = fk_global(l6, spec)
        err_mm = float((gp - torch.as_tensor(P)).norm(dim=-1).max()) * 10.0
        return err_mm, 0, spec.J, "data-tree (MJCF bypassed)"
    model, data, spec, keep, frame, addrs, cols, n_zero = load_robot(emb)
    d = np.load(f"{EXT}/{emb}/{clip}.npz")
    assert int(np.asarray(d["fps"]).ravel()[0]) == 30
    T = d["joint_pos"].shape[0]
    q = rebuild_qpos(model, d, addrs, cols, T)
    ts = [0, T // 2, T - 1]
    fk = np.empty((len(ts), model.nbody, 3))
    for i, t in enumerate(ts):
        data.qpos[:] = q[t]
        mujoco.mj_kinematics(model, data)
        fk[i] = data.xpos
    stored = d["body_pos_w"][ts].astype(np.float64)               # [3,nb_npz,3]
    dist = np.linalg.norm(fk[:, None] - stored[:, :, None], axis=-1)  # [3,nb_npz,nb_model]
    per_body = np.sort(dist.min(2).max(0))                        # worst over frames, asc
    n_drop = len(DROP_JOINTS.get(emb, ()))
    err_mm = float(per_body[:len(per_body) - n_drop].max() * 1000.0)
    note = (f"{n_drop} dropped-joint bodies excused, worst excused "
            f"{per_body[-1] * 1000:.1f}mm" if n_drop else "")
    return err_mm, n_zero, spec.J, note


# ----------------------------------- workers -----------------------------------

def process_chunk(task):
    emb, chunk_id, clips, out_npy = task
    human = emb == HUMAN_NAME
    if human:
        spec = build_human_spec(REF_CLIP)
    elif emb in DATA_TREE:
        spec = build_data_spec(emb, REF_CLIP)
    else:
        model, data, spec, keep, frame, addrs, cols, _ = load_robot(emb)
    parts, fk_err, drift, nan_frames, rows = [], 0.0, 0.0, 0, 0
    for name, K in clips:
        if human:
            R, P = human_globals(f"{EXT}/{HUMAN_SRC}/{name}.npz", K)
        elif emb in DATA_TREE:
            R, P = data_tree_globals(emb, f"{EXT}/{emb}/{name}.npz", K)
        else:
            d = np.load(f"{EXT}/{emb}/{name}.npz")
            assert int(np.asarray(d["fps"]).ravel()[0]) == 30
            q = rebuild_qpos(model, d, addrs, cols, K)
            R, P = PG.qpos_to_global(model, data, q, keep, frame)
        l6 = PG.to_local6d(R, spec.parents)
        nan_frames += int(torch.isnan(l6).any(-1).any(-1).sum())
        P_t = torch.as_tensor(P)
        _, gp = fk_global(l6, spec)
        fk_err = max(fk_err, float((gp - P_t).norm(dim=-1).max()))
        blen = (P_t[:, 1:] - P_t[:, [max(int(p), 0) for p in spec.parents[1:]]]).norm(dim=-1)
        drift = max(drift, float(blen.std(0).max()))
        parts.append(l6.numpy().astype(np.float32))
        rows += K
    np.save(out_npy, np.concatenate(parts, 0))
    return {"emb": emb, "chunk": chunk_id, "rows": rows, "fk_err_cm": fk_err,
            "bone_drift_cm": drift, "nan_frames": nan_frames, "npy": out_npy}


REF_CLIP = None      # set in main before forking (inherited by workers)


# ------------------------------------ main -------------------------------------

def main():
    global REF_CLIP
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--chunk_clips", type=int, default=6)
    ap.add_argument("--limit_clips", type=int, default=0)
    ap.add_argument("--probe_only", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    robots = sorted(d for d in os.listdir(EXT)
                    if d != HUMAN_SRC and d not in EXCLUDE)
    clips = clip_names()
    assert len(clips) == 77, len(clips)
    if args.limit_clips:
        clips = clips[:args.limit_clips]
    REF_CLIP = clips[0]
    print(f"[src] {len(robots)} robots + human, {len(clips)} paired clips", flush=True)

    # ---- 0. data-vs-XML probe; failing robots are skipped, not fatal ----
    probe, skipped = {}, {}
    for r in robots:
        try:
            err_mm, n_zero, J, note = probe_robot(r, REF_CLIP)
            probe[r] = {"probe_err_mm": round(err_mm, 3), "zero_filled_joints": n_zero,
                        "J": J, "note": note, "pass": err_mm < PROBE_GATE_MM}
            print(f"[probe] {r}: err={err_mm:.3f}mm zero_fill={n_zero} J={J} "
                  f"{'OK' if err_mm < PROBE_GATE_MM else 'FAIL'} {note}", flush=True)
            if err_mm >= PROBE_GATE_MM:
                skipped[r] = f"probe {err_mm:.1f}mm >= {PROBE_GATE_MM}mm"
        except Exception as e:
            probe[r] = {"error": str(e), "pass": False}
            skipped[r] = str(e)
            print(f"[probe] {r}: ERROR {e}", flush=True)
    robots = [r for r in robots if r not in skipped]
    if args.probe_only:
        print(json.dumps(probe, indent=2))
        return

    os.makedirs(args.out, exist_ok=True)
    tmp = f"{args.out}/_chunks"
    os.makedirs(tmp, exist_ok=True)
    emb_all = [HUMAN_NAME] + robots
    src_dir = {HUMAN_NAME: HUMAN_SRC, **{r: r for r in robots}}

    # ---- 1. per-clip lengths (zip headers only) -> min-truncated row layout ----
    key = {HUMAN_NAME: "body_pos_w"}
    lengths = {e: [npz_shape(f"{EXT}/{src_dir[e]}/{c}.npz",
                             key.get(e, "joint_pos"))[0] for c in clips]
               for e in emb_all}
    Ks = [min(lengths[e][i] for e in emb_all) for i in range(len(clips))]
    spread = [max(lengths[e][i] for e in emb_all) - Ks[i] for i in range(len(clips))]
    n_rows = int(sum(Ks))
    print(f"[align] rows={n_rows}  max tail spread={max(spread)} frames "
          f"(truncated to per-clip min)", flush=True)

    # ---- 2. FK -> local rot6d, chunked multiprocessing ----
    clip_kl = list(zip(clips, Ks))
    tasks = []
    for e in emb_all:
        for ci in range(0, len(clip_kl), args.chunk_clips):
            cid = ci // args.chunk_clips
            tasks.append((e, cid, clip_kl[ci:ci + args.chunk_clips],
                          f"{tmp}/{e}_{cid:04d}.npy"))
    print(f"[fk] {len(tasks)} tasks ({args.workers} workers)", flush=True)

    import multiprocessing as mp
    stats = {e: {"fk_err_cm": 0.0, "bone_drift_cm": 0.0, "nan_frames": 0, "rows": 0}
             for e in emb_all}
    chunks = {e: {} for e in emb_all}
    with mp.Pool(args.workers, initializer=PG._worker_init) as pool:
        done = 0
        for res in pool.imap_unordered(process_chunk, tasks):
            e = res["emb"]
            chunks[e][res["chunk"]] = res["npy"]
            stats[e]["fk_err_cm"] = max(stats[e]["fk_err_cm"], res["fk_err_cm"])
            stats[e]["bone_drift_cm"] = max(stats[e]["bone_drift_cm"], res["bone_drift_cm"])
            stats[e]["nan_frames"] += res["nan_frames"]
            stats[e]["rows"] += res["rows"]
            done += 1
            if done % 20 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)} ({time.time()-t0:.0f}s)", flush=True)

    for e in emb_all:
        spec = (build_human_spec(REF_CLIP) if e == HUMAN_NAME
                else build_data_spec(e, REF_CLIP) if e in DATA_TREE
                else load_robot(e)[2])
        arr = np.concatenate([np.load(chunks[e][c]) for c in sorted(chunks[e])], 0)
        assert arr.shape == (n_rows, spec.J, 6), (e, arr.shape, n_rows, spec.J)
        torch.save(torch.from_numpy(arr), f"{args.out}/{e}_paired_local6d.pt")
        PG.save_spec(spec, f"{args.out}/{e}_spec.npz")
        stats[e]["J"] = spec.J
        print(f"[{e}] J={spec.J} rows={arr.shape[0]} fk_err={stats[e]['fk_err_cm']:.4f}cm "
              f"drift={stats[e]['bone_drift_cm']:.4f}cm nan={stats[e]['nan_frames']}",
              flush=True)

    # ---- 3. manifest / sequences / pilot lists ----
    new_seq, start = [], 0
    for (c, K) in clip_kl:
        new_seq.append({"name": c, "start": start, "len": int(K)})
        start += int(K)
    json.dump(new_seq, open(f"{args.out}/sequences.json", "w"))
    json.dump({"schema": 2, "embodiments": emb_all, "rows": n_rows,
               "clips": len(clips), "mode": "fps", "fps": 30.0,
               "human_name": HUMAN_NAME, "src": EXT,
               "dataset": "TonyZhang1999/xwbc-retarget-data (lafan)",
               "excluded": sorted(EXCLUDE), "skipped": skipped},
              open(f"{args.out}/manifest.json", "w"), indent=2)
    g1_variants = [e for e in emb_all if "unitree_g1" in e]
    json.dump({"model_A": {"embodiments": [e for e in emb_all if e not in g1_variants],
                           "holdout_arg": ",".join(g1_variants)},
               "model_B": {"embodiments": emb_all, "holdout_arg": ""}},
              open(f"{args.out}/pilot_ab.json", "w"), indent=2)

    # ---- 4. validation gates ----
    for f in glob.glob(f"{tmp}/*.npy"):
        os.remove(f)
    os.rmdir(tmp)
    upright = {}
    for e in emb_all:
        upright[e] = PG.upright_check(args.out, e, new_seq)
        u = upright[e]
        print(f"[upright] {e}: axis={u['up_axis']} top={u['top_joint']} "
              f"peak_xyz={u['peak_cm_xyz']} {'OK' if u['pass'] else 'FAIL'}", flush=True)
    disk = sum(os.path.getsize(f"{args.out}/{f}") for f in os.listdir(args.out))
    ok = all(s["fk_err_cm"] < FK_GATE_CM and s["nan_frames"] == 0
             and s["rows"] == n_rows for s in stats.values()) \
        and all(u["pass"] for u in upright.values()) \
        and all(probe[r]["pass"] for r in robots)
    val = {"pass": bool(ok), "fk_gate_cm": FK_GATE_CM, "probe_gate_mm": PROBE_GATE_MM,
           "clips": len(clips), "rows": n_rows, "embodiments": emb_all,
           "excluded": sorted(EXCLUDE), "skipped": skipped, "probe": probe,
           "dropped_joints": {k: sorted(v) for k, v in DROP_JOINTS.items()},
           "data_tree_robots": sorted(DATA_TREE),
           "stats": stats, "upright": upright,
           "max_tail_spread_frames": int(max(spread)),
           "clip_lengths": {c: int(k) for c, k in clip_kl},
           "disk_bytes": int(disk), "wall_s": round(time.time() - t0, 1)}
    json.dump(val, open(f"{args.out}/VALIDATION.json", "w"), indent=2)
    print(f"[validate] PASS={ok}  disk={disk/1e9:.2f}GB  wall={time.time()-t0:.0f}s")
    print("SAVED", args.out)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
