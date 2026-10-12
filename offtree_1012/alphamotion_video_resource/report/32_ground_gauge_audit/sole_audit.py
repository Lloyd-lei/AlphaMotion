"""For each embodiment: the labeler's sole_drop (joint centre -> assumed sole, cm) vs the TRUE drop measured on the vendor MJCF
(joint-body origin height minus the lowest collision point of the foot bodies at the model's rest pose). Predicted decoder floor bias = label - true."""
import os; os.environ.setdefault('MUJOCO_GL', 'egl')
import numpy as np, json, mujoco
from arena.eval.visual import resolve_xml
from alphamotion.engine.descriptor import build_from_mjcf, build_from_cache
CACHE = '/var/tmp/alphamotion/topo_exp/cache_lafan_v3'; H2_XML = '/var/tmp/alphamotion/robot_models/unitree_ros/robots/h2_description/unitree_h2_mjcf.xml'
man = json.load(open(f'{CACHE}/manifest.json'))['embodiments']
def lowest_point_of_body_tree(m, d, root_body):
    """lowest z over collision geoms of root_body and its descendants (mesh vertices / boxes / spheres / capsules)."""
    kids = {root_body}; changed = True
    while changed:
        changed = False
        for b in range(m.nbody):
            if b not in kids and m.body_parentid[b] in kids: kids.add(b); changed = True
    zmin = None
    for g in range(m.ngeom):
        if m.geom_bodyid[g] not in kids or not (m.geom_contype[g] or m.geom_conaffinity[g]): continue
        R = d.geom_xmat[g].reshape(3, 3); c = d.geom_xpos[g]; s = m.geom_size[g]; t = m.geom_type[g]
        if t == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]; v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]; z = (v @ R.T + c)[:, 2].min()
        elif t == mujoco.mjtGeom.mjGEOM_BOX: z = c[2] - np.abs(R[2]).dot(s)
        elif t == mujoco.mjtGeom.mjGEOM_SPHERE: z = c[2] - s[0]
        elif t == mujoco.mjtGeom.mjGEOM_CAPSULE: z = c[2] - abs(R[2, 2]) * s[1] - s[0]
        elif t == mujoco.mjtGeom.mjGEOM_CYLINDER: z = c[2] - abs(R[2, 2]) * s[1] - s[0] * np.sqrt(max(0, 1 - R[2, 2] ** 2))
        else: z = c[2] - m.geom_rbound[g]
        zmin = z if zmin is None else min(zmin, z)
    return zmin
out = {}
for emb in man:
    row = {}
    try:
        hdr = json.loads(str(np.load(f'{CACHE}/{emb}_contact.npz', allow_pickle=True)['header']))
    except Exception as e: row['label'] = f'no contact npz: {e}'; out[emb] = row; print(emb, row); continue
    row['label_feet'] = hdr['sole_joint_names']; row['label_sole_drop_cm'] = hdr['sole_drop_cm']; row['j_floor_name'] = hdr['j_floor_name']; row['h0_lafan_cm'] = hdr['h0_m'] * 100
    try:
        xml = H2_XML if emb == 'unitree_h2' else resolve_xml(emb); m = mujoco.MjModel.from_xml_path(xml); d = mujoco.MjData(m); d.qpos[:] = m.qpos0; mujoco.mj_kinematics(m, d)
        row['mjcf'] = xml
        # which MJCF bodies correspond to the labeler's feet joints? by name, else fall back to the cache spec's low joint name
        names = hdr['sole_joint_names'] or [hdr['j_floor_name'], hdr['j_low_name']]
        drops = {}
        for n in names:
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
            if bid < 0: drops[n] = 'body not in MJCF'; continue
            z = lowest_point_of_body_tree(m, d, bid); drops[n] = None if z is None else float((d.xpos[bid][2] - z) * 100)
        row['true_drop_cm'] = drops
        # whole-robot lowest collision point vs the lowest-foot-body origin (fallback when names differ)
        zall = min(lowest_point_of_body_tree(m, d, b) or 1e9 for b in range(1, m.nbody))
        row['pelvis_z_rest_cm'] = float(d.xpos[1][2] * 100); row['lowest_point_rest_cm'] = float(zall * 100)
    except Exception as e: row['mjcf'] = f'ERR {str(e)[:150]}'
    vals = [v for v in row.get('true_drop_cm', {}).values() if isinstance(v, float)]
    lab = row['label_sole_drop_cm'][0] if row['label_sole_drop_cm'] else 0.0
    row['predicted_bias_cm'] = (lab - float(np.mean(vals))) if vals else None
    out[emb] = row; print(f"{emb:18s} label sole_drop {row['label_sole_drop_cm']} feet {row['label_feet'] or [hdr['j_floor_name']]} | true drop {row.get('true_drop_cm')} | predicted bias {row['predicted_bias_cm']}", flush=True)
json.dump(out, open('sole_audit.json', 'w'), indent=1)
