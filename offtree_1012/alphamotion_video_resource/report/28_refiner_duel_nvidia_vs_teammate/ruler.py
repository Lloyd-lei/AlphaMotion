"""One ruler for every retargeter: MuJoCo FK on the vendor model, per-foot lowest collision point,
stance-phase slip, floor penetration, airborne frames, root height, joint jerk, joint-limit excess.
Loaders: NVIDIA/BoneSeed CSV (cm Z-up, Euler deg, joints deg, header names), teammate dataset npz (43-node q +
native_slots), teammate toolkit motion.npz (q + joint_names + rot6d + root_t via the roster xml descriptor), mjlab csv."""
import sys, json, csv, numpy as np, mujoco
from scipy.spatial.transform import Rotation as Rot
AX = np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
G1_XML = '/home/arenalabs/alphamotion_ctrl/.venv/lib/python3.12/site-packages/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml'
H2_XML = '/var/tmp/alphamotion/robot_models/unitree_ros/robots/h2_description/unitree_h2_mjcf.xml'
FEET = {'g1': ('left_ankle_roll_link', 'right_ankle_roll_link'), 'h2': ('left_ankle_pitch_link', 'right_ankle_pitch_link')}
XML = {'g1': G1_XML, 'h2': H2_XML}
_models = {}
def model(robot):
    if robot not in _models:
        m = mujoco.MjModel.from_xml_path(XML[robot]); _models[robot] = (m, mujoco.MjData(m))
    return _models[robot]
def hinge_names(m): return [m.joint(i).name for i in range(m.njnt) if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]

def foot_lowest_points(m, d, robot):
    """per foot: lowest z (m) over that body's collision geoms (mesh vertices, boxes, spheres, capsules), and the xy of that point"""
    out = []
    for side in FEET[robot]:
        b = m.body(side).id; zmin, xy = 1e9, None
        for g in range(m.ngeom):
            if m.geom_bodyid[g] != b or not (m.geom_contype[g] or m.geom_conaffinity[g]): continue
            R = d.geom_xmat[g].reshape(3, 3); c = d.geom_xpos[g]; s = m.geom_size[g]; t = m.geom_type[g]
            if t == mujoco.mjtGeom.mjGEOM_MESH:
                mid = m.geom_dataid[g]; v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]; w = v @ R.T + c; k = w[:, 2].argmin(); z, p = w[k, 2], w[k, :2]
            elif t == mujoco.mjtGeom.mjGEOM_BOX:
                corners = np.array([[sx, sy, sz] for sx in (-s[0], s[0]) for sy in (-s[1], s[1]) for sz in (-s[2], s[2])]) @ R.T + c; k = corners[:, 2].argmin(); z, p = corners[k, 2], corners[k, :2]
            elif t == mujoco.mjtGeom.mjGEOM_SPHERE: z, p = c[2] - s[0], c[:2]
            elif t == mujoco.mjtGeom.mjGEOM_CAPSULE: z, p = c[2] - abs(R[2, 2]) * s[1] - s[0], c[:2]
            else: z, p = c[2] - m.geom_rbound[g], c[:2]
            if z < zmin: zmin, xy = z, p
        out.append((zmin, xy))
    return out

def audit(robot, pos_m, quat_xyzw, qj, fps, label=''):
    """pos [T,3] m Z-up, quat xyzw [T,4], qj [T,nhinge] rad in the vendor model's hinge order."""
    m, d = model(robot); T = len(pos_m); sole = np.zeros((T, 2)); xy = np.zeros((T, 2, 2)); lo, hi = m.jnt_range[1:, 0], m.jnt_range[1:, 1]
    for t in range(T):
        d.qpos[:3] = pos_m[t]; d.qpos[3:7] = quat_xyzw[t][[3, 0, 1, 2]]; d.qpos[7:] = qj[t]; mujoco.mj_kinematics(m, d)
        for k, (z, p) in enumerate(foot_lowest_points(m, d, robot)): sole[t, k] = z * 100; xy[t, k] = d.xpos[m.body(FEET[robot][k]).id][:2] * 100
    from scipy.ndimage import minimum_filter1d
    stance = minimum_filter1d((sole < 1.0).astype(np.uint8), size=5, axis=0, mode='constant').astype(bool)   # lowest point within 1 cm of the floor, eroded by 2 frames each side (touchdown/lift-off excluded)
    speed = np.linalg.norm(np.diff(xy, axis=0), axis=-1) * fps           # cm/s of the foot body origin (fixed material point), per foot
    st = stance[1:] & stance[:-1]
    slip = speed[st] if st.any() else np.array([0.])
    pen = np.clip(-sole, 0, None)
    excess = np.maximum(np.maximum(lo - qj, qj - hi), 0)
    jerk = np.abs(np.diff(qj, n=3, axis=0)).max(1) * fps ** 3 if T > 3 else np.zeros(1)
    return {'label': label, 'robot': robot, 'frames': T, 'fps': fps,
            'slip_p95_cm_s': float(np.percentile(slip, 95)), 'slip_p50_cm_s': float(np.percentile(slip, 50)), 'stance_frac': float(stance.any(1).mean()),
            'pen_p95_cm': float(np.percentile(pen.max(1), 95)), 'pen_max_cm': float(pen.max()), 'pen_frames_over_0p5cm': float((pen.max(1) > 0.5).mean()),
            'airborne_frames': float((sole.min(1) > 2.0).mean()), 'sole_min_cm': float(sole.min()),
            'root_h_cm_min': float(pos_m[:, 2].min() * 100), 'root_h_cm_max': float(pos_m[:, 2].max() * 100), 'root_step_p95_cm': float(np.percentile(np.abs(np.diff(pos_m[:, 2])) * 100, 95)),
            'joint_limit_excess_max_deg': float(np.degrees(excess.max())), 'joint_limit_frames': float((excess.max(1) > 1e-3).mean()),
            'joint_jerk_p95_deg_s3': float(np.degrees(np.percentile(jerk, 95))), 'joint_speed_p95_deg_s': float(np.degrees(np.percentile(np.abs(np.diff(qj, axis=0)).max(1) * fps, 95)))}

def load_boneseed_csv(path, robot, fps_in=120.0, fps_out=30.0, euler='xyz'):
    """NVIDIA soma-retargeter and BoneSeed official share this layout: Frame, root_translateXYZ cm Z-up, root_rotateXYZ deg, joints deg by name."""
    rows = list(csv.reader(open(path))); hdr = rows[0]; data = np.array(rows[1:], float); m, _ = model(robot); names = hinge_names(m)
    col = {h.replace('_dof', ''): i for i, h in enumerate(hdr)}
    stride = int(round(fps_in / fps_out)); data = data[::stride]
    pos = data[:, 1:4] / 100.0; q = Rot.from_euler(euler, data[:, 4:7], degrees=True).as_quat()
    qj = np.radians(np.stack([data[:, col[n]] for n in names], 1))
    return pos, q, qj, fps_out

def load_teammate_dataset_npz(path, robot='h2'):
    z = np.load(path, allow_pickle=True); meta = json.load(open('/var/tmp/arena/raw/hf_soma_h2/metadata/robot_h2.json'))
    slot = {s['native_joint_name']: (s['node_index'], s['slot_index']) for s in meta['native_slots']}
    return _from_q(z['h2__q'], z['h2__root_rot6d'], z['h2__root_t_cm'], float(z['fps']), slot, robot)

def _rot6d(x):
    a, b = x[:, :3], x[:, 3:]; a = a / np.linalg.norm(a, axis=1, keepdims=True); b = b - (a * b).sum(1, keepdims=True) * a; b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.stack([a, b, np.cross(a, b)], 2)

def _from_q(q, root_rot6d, root_t_cm, fps, slot, robot):
    m, _ = model(robot); names = hinge_names(m); R = _rot6d(root_rot6d.astype(np.float64))
    pos = (root_t_cm.astype(np.float64) @ AX) / 100.0; quat = np.stack([Rot.from_matrix(AX.T @ R[t] @ AX).as_quat() for t in range(len(R))])
    qj = np.stack([q[:, slot[n][0], slot[n][1]] if n in slot else np.zeros(len(q)) for n in names], 1).astype(np.float64)
    missing = [n for n in names if n not in slot]
    return pos, quat, qj, fps, missing

def load_teammate_toolkit_npz(path, robot, xml, body):
    """toolkit noncompact motion.npz: q [T,J,3] in the descriptor slot convention + rot6d (global) + root_t (cm canonical)."""
    from alphamotion.engine.descriptor import build_from_mjcf
    z = np.load(path, allow_pickle=True); spec, D, R, qn, _ = build_from_mjcf(xml, body)
    slot = {n: (k, s) for k, names in enumerate(qn) for s, n in enumerate(names)}
    return _from_q(z['q'], z['rot6d'][:, 0], z['root_t'], float(z['fps']), slot, robot)

if __name__ == '__main__':
    # smoke: NVIDIA reference CSVs for G1 and H2 with the floor test picking the Euler convention
    import glob, os
    base = os.path.dirname(os.path.abspath(__file__)) + '/../soma-retargeter/assets/motions/csv'
    res = []
    for robot, sub in (('g1', 'unitree_g1'), ('h2', 'unitree_h2')):
        for f in sorted(glob.glob(f'{base}/{sub}/*.csv')):
            pos, q, qj, fps = load_boneseed_csv(f, robot); a = audit(robot, pos, q, qj, fps, label='nvidia:' + os.path.basename(f)[:-4]); res.append(a)
            print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in a.items()}), flush=True)
    json.dump(res, open(os.path.dirname(os.path.abspath(__file__)) + '/nvidia_ruler.json', 'w'), indent=1)
