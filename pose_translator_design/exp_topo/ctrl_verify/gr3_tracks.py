"""Controller-readiness check for Greenwich on Fourier GR3.

Two reference tracks for the SAME AMASS walk clips:
  A  human_smpl (xwbc AMASS) -> Greenwich encode -> decode on fourier_gr3 ->
     golden global projection (hinge angles) ; root position = data passthrough
     (fourier_gr3_root_world), root orientation = decoded pelvis rotation.
  B  xwbc native GR3 retarget (raw npz: joint_pos + body pose, Isaac schema).
Both are written as beyondmimic/GMR-style CSV (root xyz, quat xyzw, joints in
MJCF order) + npz (beyondmimic schema, via MuJoCo FK) and scored with the same
physical-plausibility metrics on the GMR GR3 MJCF.  Also renders both side by
side (MuJoCo, EGL) for lloyd to judge.
"""
import os, sys, json, argparse, numpy as np, torch
sys.path.insert(0, "/home/arenalabs/Desktop/AlphaMotion/src")
torch.set_grad_enabled(False)
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco as mj
from scipy.spatial.transform import Rotation as Rot
from alphamotion.engine.greenwich import Greenwich
from alphamotion.engine.spatial import SkeletonSpec, build_global
from alphamotion.engine.descriptor import build_from_mjcf
from alphamotion.engine import constraints as MP
from alphamotion.engine.nets.rotations import rot6d_to_matrix

C = "/var/tmp/alphamotion/topo_exp/cache_xwbc_amass_full/"
RAW = "/var/tmp/alphamotion/ctrl_verify/raw/fourier_gr3/"
OUT = "/var/tmp/alphamotion/ctrl_verify/"
XML = "/var/tmp/alphamotion/config_harvest/GMR/assets/fourier_gr3v2_1_1/mjcf/gr3v2_1_1.xml"
AX = np.array([[0., 1, 0], [0, 0, 1], [1, 0, 0]])     # Z-up -> Y-up (prep_gmr_cache v2)
FPS = 30
CLIPS = ["KIT__11__WalkingStraightForwards01_stageii", "KIT__11__WalkingStraightForwards03_stageii",
         "KIT__11__WalkingStraightForwards05_stageii", "ACCAD__Female1Walking_c3d__B3_-_walk1_stageii",
         "ACCAD__Female1Walking_c3d__B1_-_stand_to_walk_stageii", "HumanEva__S2__Walking_3_stageii"]

def load_spec(p):
    d = np.load(p, allow_pickle=True); return SkeletonSpec(str(d["name"]), d["parents"], d["rest_offsets"], [str(n) for n in d["joint_names"]])

# ---------------------------------------------------------------- model ----
_spec = mj.MjSpec.from_file(XML)
if not any(j.type == mj.mjtJoint.mjJNT_FREE for j in _spec.joints):
    _spec.bodies[1].add_freejoint()                      # GMR ships GR3 fixed-base
_spec.worldbody.add_geom(name="_floor", type=mj.mjtGeom.mjGEOM_PLANE, size=[20, 20, 0.1], rgba=[0.9, 0.9, 0.92, 1])
_spec.worldbody.add_light(name="_sun", pos=[0, -1, 3], dir=[0, 0.25, -1], diffuse=[0.6, 0.6, 0.6], castshadow=True)
model = _spec.compile(); data = mj.MjData(model)
assert model.jnt_type[0] == mj.mjtJoint.mjJNT_FREE, [int(t) for t in model.jnt_type[:3]]
hinge = [j for j in range(1, model.njnt)]
hnames = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, j) for j in hinge]
hadr = np.array([model.jnt_qposadr[j] for j in hinge]); hdof = np.array([model.jnt_dofadr[j] for j in hinge])
jrange = model.jnt_range[hinge]           # [n,2]
body_names = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, b) for b in range(model.nbody)]
foot_geoms = {s: [g for g in range(model.ngeom) if model.geom_bodyid[g] > 0 and s in body_names[model.geom_bodyid[g]].lower() and ("ankle" in body_names[model.geom_bodyid[g]].lower() or "foot" in body_names[model.geom_bodyid[g]].lower())] for s in ["left", "right"]}
print("GR3 model: nq", model.nq, "hinges", len(hinge), "foot geoms", {k: len(v) for k, v in foot_geoms.items()})

def qpos_from(root_p, root_q_wxyz, jp_by_name):
    """[T,nq] from root pose + {joint_name: [T]}."""
    T = root_p.shape[0]; qp = np.tile(model.qpos0, (T, 1))
    qp[:, 0:3] = root_p; qp[:, 3:7] = root_q_wxyz
    for a, n in zip(hadr, hnames):
        qp[:, a] = jp_by_name[n]
    return qp

_mesh_cache = {}
def _geom_low_z(g):
    """Exact lowest world-z of a geom (mesh vertices for meshes, AABB for primitives)."""
    R = data.geom_xmat[g].reshape(3, 3); p = data.geom_xpos[g]
    if model.geom_type[g] == mj.mjtGeom.mjGEOM_MESH:
        mid = model.geom_dataid[g]
        if mid not in _mesh_cache:
            a = model.mesh_vertadr[mid]; n = model.mesh_vertnum[mid]; _mesh_cache[mid] = model.mesh_vert[a:a + n].copy()
        v = _mesh_cache[mid] @ R.T + p
        return float(v[:, 2].min())
    return float(p[2] - np.abs(R[2]).dot(model.geom_size[g]))
def fk_all(qp):
    """MuJoCo FK -> dict of beyondmimic-schema arrays + foot geom lows."""
    T = qp.shape[0]
    bp = np.zeros((T, model.nbody, 3)); bq = np.zeros((T, model.nbody, 4)); low = {k: np.zeros(T) for k in foot_geoms}
    fxy = {k: np.zeros((T, 2)) for k in foot_geoms}
    for t in range(T):
        data.qpos[:] = qp[t]; mj.mj_kinematics(model, data)
        bp[t] = data.xpos; bq[t] = data.xquat
        for k, gs in foot_geoms.items():
            low[k][t] = min(_geom_low_z(g) for g in gs) if gs else np.nan
            fxy[k][t] = np.mean([data.geom_xpos[g, :2] for g in gs], 0) if gs else np.nan
    dt = 1.0 / FPS
    jp = qp[:, hadr]; jv = np.gradient(jp, dt, axis=0)
    blv = np.gradient(bp, dt, axis=0)
    # angular velocity (world) from consecutive quats
    bav = np.zeros_like(blv)
    for b in range(model.nbody):
        R = Rot.from_quat(bq[:, b][:, [1, 2, 3, 0]])       # xyzw
        dR = (R[1:] * R[:-1].inv()).as_rotvec() / dt
        bav[1:, b] = dR; bav[0, b] = dR[0] if T > 1 else 0
    return dict(fps=np.array([FPS]), joint_pos=jp.astype(np.float32), joint_vel=jv.astype(np.float32),
                body_pos_w=bp.astype(np.float32), body_quat_w=bq.astype(np.float32),
                body_lin_vel_w=blv.astype(np.float32), body_ang_vel_w=bav.astype(np.float32),
                joint_names=np.array(hnames), body_names=np.array(body_names)), low, fxy

def metrics(qp, fk, low, fxy):
    jp, jv = fk["joint_pos"], fk["joint_vel"]; dt = 1 / FPS
    over = np.maximum(jp - jrange[:, 1], 0) + np.maximum(jrange[:, 0] - jp, 0)
    ja = np.gradient(jv, dt, axis=0); jerk = np.gradient(ja, dt, axis=0)
    m = {"frames": int(jp.shape[0]),
         "limit_violation_frac": float((over > 1e-4).mean()), "limit_excess_max_deg": float(np.degrees(over.max())),
         "joint_vel_p95_rad_s": float(np.percentile(np.abs(jv), 95)), "joint_vel_max_rad_s": float(np.abs(jv).max()),
         "vel_spike_frac_gt8rad_s": float((np.abs(jv) > 8).mean()), "frames_with_spike": int((np.abs(jv) > 8).any(1).sum()),
         "joint_acc_p95_rad_s2": float(np.percentile(np.abs(ja), 95)), "jerk_rms_rad_s3": float(np.sqrt((jerk ** 2).mean())),
         "root_height_mean_m": float(qp[:, 2].mean()), "root_travel_m": float(np.linalg.norm(qp[-1, :2] - qp[0, :2])),
         "root_speed_mean_m_s": float(np.linalg.norm(np.diff(qp[:, :2], axis=0), axis=1).mean() * FPS)}
    # ground = lowest foot vertex over the clip (xwbc aligns body ORIGINS to z=0, so
    # meshes sit ~12 cm below z=0 in the native data too — common-mode, calibrate it out)
    floor = min(np.min(low[k]) for k in low)
    skate, stance_frac, clear = [], [], []
    for k in low:
        z = low[k] - floor; v = np.linalg.norm(np.diff(fxy[k], axis=0), axis=1) * FPS
        stance = (z[1:] < 0.02); skate.append(v[stance] if stance.any() else np.array([0.0])); stance_frac.append(stance.mean())
        clear.append(z.max())
    m["foot_floor_offset_cm"] = float(floor * 100)
    m["stance_frac"] = float(np.mean(stance_frac)); m["swing_clearance_max_cm"] = float(np.max(clear) * 100)
    m["stance_skate_mean_cm_s"] = float(np.mean(np.concatenate(skate)) * 100); m["stance_skate_p95_cm_s"] = float(np.percentile(np.concatenate(skate), 95) * 100)
    # robust proxy for walking: the LOWER foot each frame should be planted -> its horizontal speed = skate
    ks = list(low); zz = np.stack([low[k] for k in ks], 1); vv = np.stack([np.linalg.norm(np.diff(fxy[k], axis=0), axis=1) * FPS for k in ks], 1)
    lower = zz[1:].argmin(1); vlow = vv[np.arange(len(vv)), lower]
    m["lower_foot_speed_mean_cm_s"] = float(vlow.mean() * 100); m["lower_foot_speed_p95_cm_s"] = float(np.percentile(vlow, 95) * 100)
    m["feet_height_diff_mean_cm"] = float(np.abs(zz[:, 0] - zz[:, 1]).mean() * 100)
    return m

def write_csv(path, qp):
    """GMR/beyondmimic CSV: root xyz, quat xyzw, joints (MJCF hinge order)."""
    q = qp[:, 3:7][:, [1, 2, 3, 0]]
    np.savetxt(path, np.concatenate([qp[:, :3], q, qp[:, hadr]], 1), delimiter=",", fmt="%.6f")

# ---------------------------------------------------------------- Greenwich ---
G = Greenwich.load()
hs = load_spec(C + "human_smpl_spec.npz"); hd = np.load(C + "human_smpl_dof.npz", allow_pickle=True)["dof"]
gspec, gdof, grest, gqnames, _ = build_from_mjcf(XML, "fourier_gr3")
cspec = load_spec(C + "fourier_gr3_spec.npz")
print("gr3 spec from MJCF J", gspec.J, "cache J", cspec.J, "names equal", list(gspec.joint_names) == list(cspec.joint_names))
L = torch.load(C + "human_smpl_paired_local6d.pt", mmap=True); RW = torch.load(C + "fourier_gr3_root_world.pt", mmap=True)
seqs = {s["name"]: s for s in json.load(open(C + "sequences.json"))}
root_j = int(np.where(np.asarray(gspec.parents) < 0)[0][0])
dof_t = torch.as_tensor(gdof, dtype=torch.float64); rest_t = torch.as_tensor(grest, dtype=torch.float64)

report = {}
for name in CLIPS:
    s = seqs[name]; st, n = s["start"], s["len"]
    # ---- A: Greenwich
    l6 = L[st:st + n].float(); g6, gp, reach = build_global(l6, hs, "cpu")
    codes = G.encode(torch.cat([g6, gp / reach], -1).to(G.device), hs, hd)
    rot, _ = G.decode_full(codes, gspec, gdof)
    Rg = rot6d_to_matrix(rot).double().cpu()
    _, _, q = MP.project(Rg, gspec, dof_t, rest=rest_t, method="global", lm_iters=20)   # q [T,J,3]
    q = q.detach().numpy()
    jp = {}
    for j, names3 in enumerate(gqnames):
        for k, nm in enumerate(names3):
            if nm: jp[nm] = q[:, j, k]
    missing = [h for h in hnames if h not in jp]; assert not missing, missing
    R_root_zup = AX.T @ Rg[:, root_j].numpy() @ AX
    quat_wxyz = Rot.from_matrix(R_root_zup).as_quat()[:, [3, 0, 1, 2]]
    # owner design: the ROOT TRAJECTORY (position AND orientation) is data passthrough;
    # Greenwich supplies the root-relative body pose. Decoded root orientation was
    # 21 deg off while lying -> bodies 30+ cm through the floor on fall clips.
    ROOT_ORI_PASSTHROUGH = True
    root_yup = RW[st:st + n].numpy().astype(np.float64)                     # cm, Y-up, world
    root_zup = (root_yup @ AX) / 100.0                                       # p_zup = AX^T p_yup
    # ---- B: native
    d = np.load(RAW + name + ".npz", allow_pickle=True); jn = [str(x) for x in d["joint_names"]]
    T = min(n, d["joint_pos"].shape[0])
    if ROOT_ORI_PASSTHROUGH:
        quat_wxyz = quat_wxyz.copy(); quat_wxyz[:T] = d["body_quat_w"][:T, 0].astype(np.float64)
    qpA = qpos_from(root_zup, quat_wxyz, jp)
    jpB = {nm: d["joint_pos"][:T, jn.index(nm)].astype(np.float64) for nm in hnames}
    qpB = qpos_from(d["body_pos_w"][:T, 0].astype(np.float64), d["body_quat_w"][:T, 0].astype(np.float64), jpB)
    qpA = qpA[:T]
    fkA, lowA, fxyA = fk_all(qpA); fkB, lowB, fxyB = fk_all(qpB)
    mA, mB = metrics(qpA, fkA, lowA, fxyA), metrics(qpB, fkB, lowB, fxyB)
    # ---- A vs B agreement
    dj = np.degrees(np.abs(fkA["joint_pos"] - fkB["joint_pos"]))
    agree = {"joint_abs_err_mean_deg": float(dj.mean()), "joint_abs_err_p95_deg": float(np.percentile(dj, 95)),
             "root_pos_err_mean_cm": float(np.linalg.norm(qpA[:, :3] - qpB[:, :3], axis=1).mean() * 100),
             "root_pos_err_first_frame_cm": float(np.linalg.norm(qpA[0, :3] - qpB[0, :3]) * 100),
             "root_yaw_err_mean_deg": float(np.degrees(np.abs((Rot.from_quat(qpA[:, 3:7][:, [1,2,3,0]]) * Rot.from_quat(qpB[:, 3:7][:, [1,2,3,0]]).inv()).magnitude()).mean())),
             "body_pos_err_mean_cm": float(np.linalg.norm(fkA["body_pos_w"] - fkB["body_pos_w"], axis=-1).mean() * 100),
             "worst_joints_deg": {hnames[i]: round(float(dj.mean(0)[i]), 1) for i in np.argsort(-dj.mean(0))[:5]}}
    os.makedirs(OUT + "tracks", exist_ok=True)
    write_csv(f"{OUT}tracks/{name}__A_greenwich.csv", qpA); write_csv(f"{OUT}tracks/{name}__B_native.csv", qpB)
    np.savez(f"{OUT}tracks/{name}__A_greenwich.npz", **fkA); np.savez(f"{OUT}tracks/{name}__B_native.npz", **fkB)
    np.save(f"{OUT}tracks/{name}__qposA.npy", qpA); np.save(f"{OUT}tracks/{name}__qposB.npy", qpB)
    report[name] = {"A_greenwich": mA, "B_native": mB, "A_vs_B": agree}
    print(f"\n== {name} ({T} f)")
    for k in mA: print(f"  {k:28s} A {mA[k]:9.3f}   B {mB[k]:9.3f}")
    print("  A vs B:", {k: v for k, v in agree.items() if k != "worst_joints_deg"}); print("  worst joints:", agree["worst_joints_deg"])
json.dump(report, open(OUT + "report.json", "w"), indent=1)
print("saved", OUT + "report.json")
