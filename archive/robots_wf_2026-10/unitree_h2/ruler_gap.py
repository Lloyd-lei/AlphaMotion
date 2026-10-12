"""Dense-point ruler vs exact visual-vertex ruler for the feet: how much of the measured 'hover' is the 4096-point
sampling gap (lowest dense point above the true lowest mesh vertex)?  Usage: ruler_gap.py <cid> -> prints + json"""
import json, sys
import mujoco, numpy as np
sys.path.insert(0, '/home/arenalabs/umr_factory_data/robots/unitree_h2/pipeline')
import h2common as H2
cid = sys.argv[1]
o = np.load(H2.PD / 'out' / cid / 'unitree_h2.npz', allow_pickle=True)
pf = np.load(H2.QD / f'{cid}.perframe.npz')
m = mujoco.MjModel.from_xml_path(str(H2.RD / 'unitree_h2.xml')); d = mujoco.MjData(m)
Q = o['qpos'].astype(float); T = len(Q)
out = {}
for s, side, r in (('L', 'left', 5), ('R', 'right', 6)):
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f'{side}_ankle_pitch_link')
    gs = [g for g in range(m.ngeom) if m.geom_bodyid[g] == b and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and m.geom_group[g] == 1]
    V = [m.mesh_vert[m.mesh_vertadr[m.geom_dataid[g]]:m.mesh_vertadr[m.geom_dataid[g]] + m.mesh_vertnum[m.geom_dataid[g]]].astype(float) for g in gs]
    ex = np.zeros(T)
    for t in range(T):
        d.qpos[:] = Q[t]; mujoco.mj_kinematics(m, d)
        ex[t] = min((v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])[:, 2].min() for g, v in zip(gs, V))
    H = pf['H_g'][:, r]; dense = pf['cg_patch'][:, r]
    gap = (dense - ex)[H]
    hov = H & (dense > 0.001)
    out[f'{s}_foot'] = dict(human_contact=int(H.sum()), exact_vertex_in_0_1mm=int((H & (ex >= -0.0005) & (ex <= 0.001)).sum()),
                            dense_in_0_1mm=int((H & (dense >= 0) & (dense <= 0.001)).sum()),
                            dense_minus_exact_mm_p50_p95_max=np.round(np.percentile(gap, [50, 95, 100]) * 1000, 2).tolist(),
                            hover_1_2mm=int((hov & (dense <= 0.002)).sum()), hover_2_5mm=int((hov & (dense > 0.002) & (dense <= 0.005)).sum()),
                            hover_5_20mm=int((hov & (dense > 0.005) & (dense <= 0.02)).sum()), far_gt20mm=int((H & (dense > 0.02)).sum()),
                            exact_vertex_hover_gt5mm=int((H & (ex > 0.005)).sum()))
print(json.dumps(out, indent=1))
json.dump(out, open(H2.QD / f'{cid}.ruler_gap.json', 'w'), indent=1)
