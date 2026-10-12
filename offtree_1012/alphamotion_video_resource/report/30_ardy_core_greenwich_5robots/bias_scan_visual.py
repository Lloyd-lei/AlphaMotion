"""Label-free vertical bias: per-frame lowest collision point on the real MJCF for every run (Core vs G1 source), then
median (= root-height bias), spread after removing the bias, and geometry-stance slip after the shift."""
import os, sys, json, numpy as np, torch, mujoco
os.environ.setdefault('MUJOCO_GL', 'egl')
sys.path.insert(0, '/home/arenalabs/Desktop/alphamotion video/resource/report/29_ardy_greenwich_5robots')
from scipy.ndimage import minimum_filter1d
from arena.eval.visual import RobotRenderer, resolve_xml
from alphamotion.engine.descriptor import build_from_mjcf
from arena.refiner.robots import profiles
H2_XML = '/var/tmp/alphamotion/robot_models/unitree_ros/robots/h2_description/unitree_h2_mjcf.xml'
meta = json.load(open('/var/tmp/arena/raw/hf_soma_h2/metadata/robot_h2.json')); hslot = {x['native_joint_name']: (x['node_index'], x['slot_index']) for x in meta['native_slots']}
ROBOTS = ['unitree_g1_29dof', 'unitree_h2', 'agibot_a3_t3d0', 'fourier_gr3', 'pnd_adam_lite']; PROMPTS = ['p1_box', 'p2_hurdle', 'p3_wave', 'p4_kick', 'p5_sit']
RUNS = {'core': sys.argv[1], 'g1': sys.argv[2]}
def q_for_renderer(qdict, r):
    q = np.zeros((len(next(iter(qdict.values()))), len(r.qnames), 3), np.float32)
    for k, names in enumerate(r.qnames):
        for s, n in enumerate(names):
            if n in qdict: q[:, k, s] = qdict[n]
    return q
def low_and_feet(r, feet_bodies):
    m, d = r.model, r.data; zmin = 1e9; fxy = []
    for gi in range(m.ngeom):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gi) or ''
        if (m.geom_contype[gi] or m.geom_conaffinity[gi]) or m.geom_type[gi] != mujoco.mjtGeom.mjGEOM_MESH or nm.startswith('_tg'): continue   # VISUAL meshes only
        c = d.geom_xpos[gi]; s = m.geom_size[gi]; R = d.geom_xmat[gi].reshape(3, 3); t = m.geom_type[gi]
        if t == mujoco.mjtGeom.mjGEOM_SPHERE: z = c[2] - s[0]
        elif t == mujoco.mjtGeom.mjGEOM_CAPSULE: z = c[2] - abs(R[2, 2]) * s[1] - s[0]
        elif t == mujoco.mjtGeom.mjGEOM_BOX: z = c[2] - np.abs(R[2]).dot(s)
        elif t == mujoco.mjtGeom.mjGEOM_CYLINDER: z = c[2] - abs(R[2, 2]) * s[1] - s[0] * np.sqrt(max(0, 1 - R[2, 2] ** 2))
        elif t == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[gi]; v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]; z = (v @ R.T + c)[:, 2].min()
        else: z = c[2] - m.geom_rbound[gi]
        zmin = min(zmin, z)
    for b in feet_bodies: fxy.append(d.xpos[b][:2] * 100)
    return zmin * 100, np.array(fxy)
out = {}
for name in ROBOTS:
    xml = H2_XML if name == 'unitree_h2' else resolve_xml(name); r = RobotRenderer(name, width=320, height=180, xml=xml)
    qn = None if name == 'unitree_h2' else build_from_mjcf(xml, name + 'x')[3]
    prof = profiles()[name]; feet = []
    for side in ('left', 'right'):
        i = mujoco.mj_name2id(r.model, mujoco.mjtObj.mjOBJ_BODY, 'rb_' + prof[side]['foot_name']); feet.append(i)
    for src, base in RUNS.items():
        for p in PROMPTS:
            z = np.load(f'{base}/{p}/{name}.npz'); q, R, root, fps = z['q'], z['root_R'], z['root_world_cm'], float(z['fps']); T = len(q)
            qd = {n: q[:, k, s] for n, (k, s) in hslot.items()} if name == 'unitree_h2' else {n: q[:, k, s] for k, names in enumerate(qn) for s, n in enumerate(names)}
            qr = q_for_renderer(qd, r); lz = np.zeros(T); fxy = np.zeros((T, 2, 2))
            for t in range(T):
                r._set_frame(qr[t], R[t], root[t], None, follow=False); lz[t], fxy[t] = low_and_feet(r, feet)
            bias = float(np.median(lz)); lzs = lz - bias
            st = minimum_filter1d((lzs < 1.0).astype(np.uint8), size=5, axis=0, mode='constant').astype(bool)  # any-body stance after removing the bias
            # per-foot stance needs per-foot lowest points; approximate with the body-origin height rank: the lower foot is the stance foot
            sp = np.linalg.norm(np.diff(fxy, axis=0), axis=-1) * fps   # [T-1, 2]
            low_foot = np.argmin(np.stack([np.zeros(T), np.zeros(T)], 1), 1)  # placeholder, replaced below
            # slip of the lower foot (by body origin z) during stance frames
            fz = np.zeros((T, 2))
            for t in range(T):
                r._set_frame(qr[t], R[t], root[t], None, follow=False)
                for k, b in enumerate(feet): fz[t, k] = r.data.xpos[b][2]
            lowk = np.argmin(fz, 1)[1:]; mask = st[1:] & st[:-1]
            slip = sp[np.arange(T - 1), lowk][mask]
            out[f'{src}|{p}|{name}'] = dict(bias_cm=bias, lz_min=float(lz.min()), lz_p10=float(np.percentile(lz, 10)), lz_p90=float(np.percentile(lz, 90)), spread_p90_p10=float(np.percentile(lz, 90) - np.percentile(lz, 10)),
                                            pen_after_shift_p95=float(np.percentile(np.clip(-lzs, 0, None), 95)), stance_frac_after_shift=float(st.mean()),
                                            slip_p95_after_shift=float(np.percentile(slip, 95)) if mask.any() else None, slip_p50_after_shift=float(np.percentile(slip, 50)) if mask.any() else None)
            print(src, p, name, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in out[f'{src}|{p}|{name}'].items()}, flush=True)
json.dump(out, open(f'{sys.argv[1]}/bias_scan_visual.json', 'w'), indent=1); print('BIAS DONE')
