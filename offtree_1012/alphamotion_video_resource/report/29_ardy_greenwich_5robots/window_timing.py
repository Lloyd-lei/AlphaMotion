"""Window-mode timing: ARDY emits 52-frame windows; time pose9+encode+decode+project per 52-frame window per robot (median of 5 after warm-up)."""
import os, sys, json, time, statistics, numpy as np, torch
os.environ.setdefault("MUJOCO_GL", "egl"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ardy_in import load_ardy, SPEC as SPEC_G, DOF as DOF_G, NAMES as NAMES_G
from alphamotion.engine.greenwich import Greenwich
from alphamotion.engine.descriptor import build_from_mjcf
from alphamotion.engine.constraints import project
from alphamotion.engine.nets.rotations import SkeletonSpec, matrix_to_rot6d
from arena.eval.visual import resolve_xml
from arena.refiner.robots import load_robot
g = Greenwich.load(model='megatron', device='cuda'); sync = torch.cuda.synchronize
packs = {}
for nm in ['unitree_g1_29dof', 'fourier_gr3', 'pnd_adam_lite']:
    s, d, r = g.embodiment(nm); packs[nm] = (s, torch.as_tensor(d).float(), torch.as_tensor(r).float())
rob = load_robot('agibot_a3_t3d0'); sx, dx, rx, _, _ = build_from_mjcf(resolve_xml('agibot_a3_t3d0'), 'agibot_a3_t3d0'); packs['agibot_a3_t3d0'] = (rob.spec, torch.as_tensor(dx).float(), torch.as_tensor(rx).float())
ms = json.load(open('/var/tmp/arena/raw/hf_soma_h2/metadata/robot_h2.json'))['merged_skeleton']
packs['unitree_h2'] = (SkeletonSpec('h2_soma43', np.array(ms['parents']), np.array(ms['rest_offsets_cm'], np.float32), list(ms['joint_names'])), torch.as_tensor(np.array(ms['dof_features'], np.float32)), torch.as_tensor(np.array(ms['rest_rotations'], np.float32)))
for s, d, _ in packs.values(): g.geom(s, d)
a = load_ardy(sys.argv[1]); gR6 = matrix_to_rot6d(a['gR']).float(); T = len(gR6); W = 52
con = torch.zeros(T, SPEC_G.J)
for c, side in enumerate(['left', 'left', 'right', 'right']):
    for n in (f'{side}_ankle_roll_link', f'{side}_toe_link'): con[:, NAMES_G.index(n)] = torch.maximum(con[:, NAMES_G.index(n)], torch.as_tensor(a['contacts'][:, c] > 0.5).float())
out = {'window_frames': W, 'window_seconds': W / a['fps'], 'robots': {}}
def enc(t0):
    pose, _ = g.pose9(gR6[t0:t0 + W], SPEC_G, is_global=True); return g.encode(pose, SPEC_G, DOF_G, contact=con[t0:t0 + W].cuda())
for nm, (spec, dof, rest) in packs.items():
    xs_e, xs_d, xs_p = [], [], []
    for i in range(6):
        t0 = (i * W) % (T - W)
        sync(); s0 = time.perf_counter(); c = enc(t0); sync(); s1 = time.perf_counter()
        r6, pos = g.decode_full(c, spec, dof); sync(); s2 = time.perf_counter()
        project(r6.cpu(), spec, dof, rest=rest); sync(); s3 = time.perf_counter()
        if i: xs_e.append(s1 - s0); xs_d.append(s2 - s1); xs_p.append(s3 - s2)
    e, d_, p = [1000 * statistics.median(x) for x in (xs_e, xs_d, xs_p)]
    out['robots'][nm] = dict(encode_ms=e, decode_ms=d_, project_ms=p, total_ms=e + d_ + p, rtf=(e + d_ + p) / 1000 / (W / a['fps']))
    print(nm, {k: round(v, 2) for k, v in out['robots'][nm].items()}, flush=True)
json.dump(out, open(os.path.dirname(os.path.abspath(__file__)) + '/runs/window_timing.json', 'w'), indent=1)
