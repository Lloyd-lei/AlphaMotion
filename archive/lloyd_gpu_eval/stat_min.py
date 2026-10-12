import sys, json
sys.path.insert(0, '/home/arenalabs/umr_factory_code')
import numpy as np, mujoco
from umrf import surface as S, geometry as G
from umrf.paths import OUT
clip, robot = sys.argv[1], sys.argv[2]
o = np.load(f'cpu_{robot}.npz', allow_pickle=True)
g = np.load(f'{clip}__{robot}.gpu.npz', allow_pickle=True)
m = mujoco.MjModel.from_xml_path(str(o['robot_xml'])); geo = G.RobotGeometry(m, S.load_surface(robot), 0.0)
qg = g['q_out'].copy(); qg[:, 2] -= float(g['floor'][0])
res = {}
for name, Q in [('cpu', o['qpos']), ('gpu', qg), ('gpu_nofloorshift', g['q_out'])]:
    mn = []
    for t in range(0, len(Q), 3):
        geo.set(Q[t].astype(float)); mn.append(geo.dense_world()[:, 2].min() * 1000)
    mn = np.array(mn)
    res[name] = dict(p1=round(float(np.percentile(mn, 1)), 3), p50=round(float(np.median(mn)), 3), p99=round(float(np.percentile(mn, 99)), 3),
                     touching_frac=round(float(((mn >= -0.5) & (mn <= 1.0)).mean()), 3), pen_frac=round(float((mn < -0.5).mean()), 3),
                     hover_1_5mm=round(float(((mn > 1.0) & (mn <= 5)).mean()), 3), air_gt5mm=round(float((mn > 5).mean()), 3))
hc = o['human_ground_contact'][::3].any(1)
res['human_any_ground_contact_frac'] = round(float(hc.mean()), 3)
print(json.dumps(res, indent=1))
