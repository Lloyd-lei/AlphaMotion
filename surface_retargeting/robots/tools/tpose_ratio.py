"""Compute robot bbox center ratio of point_cloud_center body in T-pose (UMR convention check)."""
import sys, json, numpy as np, mujoco
sys.path.insert(0, '/var/tmp/alphamotion/umr_factory/umr/UMR/scripts')
from mujoco_geom_surface import surface_geom_ids  # noqa
from retarget_smpl_to_humanoid_surface_vector import geom_local_mesh  # noqa

def ratio(xml, center_body, tpose, extra=None):
    m = mujoco.MjModel.from_xml_path(xml); d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    for j, v in tpose.items():
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        if jid < 0: print('missing joint', j); continue
        d.qpos[m.jnt_qposadr[jid]] = v
    mujoco.mj_forward(m, d)
    gids = surface_geom_ids(m, 'auto')
    V = []
    for g in gids:
        lv, _ = geom_local_mesh(m, int(g))
        V.append(np.asarray(lv) @ d.geom_xmat[g].reshape(3,3).T + d.geom_xpos[g])
    V = np.concatenate(V)
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, center_body)
    zc = d.xpos[b][2]; lo, hi = V[:,2].min(), V[:,2].max()
    return (zc-lo)/(hi-lo), hi-lo, V, m, d

if __name__ == '__main__':
    cfg = json.load(open(sys.argv[1]))['robot']
    import os
    xml = os.path.normpath(os.path.join(os.path.dirname(sys.argv[1]), cfg['xml']))
    r, h, *_ = ratio(xml, cfg['point_cloud_center'].split(':',1)[1], cfg.get('tpose_qpos', {}))
    print(sys.argv[1], 'ratio=%.6f height=%.3f' % (r, h))
