"""Synthetic tests of the strict post-pass (run: python tests/test_postpass.py [robot]).
 0. arm at its roll limit, 95 mm into the torso (G1): fixed within deviation bounds, else FLAGGED
 1. penetration: standing pose sunk 2 cm into the floor + arm rolled into the body    -> all pens <= 0.5 mm
 2. hover: feet 6 mm above the floor with both feet REQUIRED in contact            -> foot clearance in [0, 1 mm]
 3. object: ball pushed 1.5 cm into the right hand + hand REQUIRED in contact          -> obj pen <= 0.5 mm, contact [0,1mm]
 4. known-clean: feet exactly touching (0.3 mm), no requirement violated               -> qpos bit-identical
 5. runtime per frame on a 60-frame sequence.
"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco, numpy as np, trimesh
from surface_retargeting.robots import surface as S, robots as R
from surface_retargeting.strict import geometry as G, postpass as PP
from surface_retargeting.strict.objects import PosedObject
from surface_retargeting.paths import ROBOTS

name = sys.argv[1] if len(sys.argv) > 1 else 'unitree_g1_29dof'
st = json.load(open(ROBOTS / name / 'staging.json'))
m = mujoco.MjModel.from_xml_path(st['staged_mjcf']); surf = S.load_surface(name)
NR = len(S.REGIONS); RID = S.RID
res = {'robot': name}


def stand_pose():
    q = m.qpos0.copy(); q[0:3] = [0, 0, 1.0]; q[3:7] = [1, 0, 0, 0]
    geo = G.RobotGeometry(m, surf, 0.0); geo.set(q); z, _, _ = geo.ground()
    q[2] -= z.min()                       # lowest point exactly at z=0
    return q


def report(tag, geo, q, t, rg, ro):
    mm = geo.measure(q, t)
    out = dict(ground_min_mm=round(mm['ground_min'] * 1000, 3), self_min_mm=round(min(mm['self_min'], 1) * 1000, 3),
               obj_min_mm=round(mm['obj_min'] * 1000, 3) if np.isfinite(mm['obj_min']) else None,
               req_ground_mm={S.REGIONS[r]: round(mm['ground_reg_min'][r] * 1000, 3) for r in np.flatnonzero(rg)},
               req_obj_mm={S.REGIONS[r]: round(mm['obj_reg_min'][r] * 1000, 3) for r in np.flatnonzero(ro)})
    return out


def one(tag, q_ref, rg, ro, obj=None, extra_check=None):
    geo = G.RobotGeometry(m, surf, 0.0, obj)
    pj = PP.Projector(geo)
    t0 = time.time()
    q, f, it, f0 = pj.solve_frame(0, q_ref, q_ref, None, rg, ro)
    dt = time.time() - t0
    r = dict(before=report(tag, geo, q_ref, 0, rg, ro), after=report(tag, geo, q, 0, rg, ro), flags_before=int(f0),
             flags_after=int(f), iters=it, seconds=round(dt, 3),
             max_joint_change_deg=round(float(np.degrees(np.abs(pj.diff(q_ref, q)[6:]).max())), 2),
             root_change_cm=round(float(np.linalg.norm(q[:3] - q_ref[:3]) * 100), 3))
    res[tag] = r
    print(tag, json.dumps(r), flush=True)
    return q, r


none = np.zeros(NR, bool)
# 0. deep self-penetration (arm at roll limit, 95 mm into torso on G1): fixed inside the bounds or FLAGGED, never silent
q = stand_pose()
j = R.mid(m, mujoco.mjtObj.mjOBJ_JOINT, 'left_shoulder_roll_joint')
if j >= 0 and name.startswith('unitree_g1'):
    q[m.jnt_qposadr[j]] = m.jnt_range[j][0]
    qq, r0 = one('deep_self_pen', q, none, none)
    assert r0['flags_after'] != 0 or r0['after']['self_min_mm'] >= -0.5, r0
    res['deep_self_pen']['note'] = 'either fixed (<=0.5 mm) inside the deviation bounds or flagged; never silent'
# 1. penetration
q = stand_pose(); q[2] -= 0.02
# drive left arm into torso: shoulder roll towards body
for jn, v in (('left_shoulder_roll_joint', -0.25), ('left_elbow_joint', 0.0)):
    j = R.mid(m, mujoco.mjtObj.mjOBJ_JOINT, jn)
    if j >= 0:
        q[m.jnt_qposadr[j]] = np.clip(v, *m.jnt_range[j])
_, r1 = one('penetration', q, none, none)
assert r1['after']['ground_min_mm'] >= -0.5 and r1['after']['self_min_mm'] >= -0.5 and r1['flags_after'] == 0, r1
# 2. hover
q = stand_pose(); q[2] += 0.006
rg = none.copy(); rg[RID['L_foot']] = rg[RID['R_foot']] = True
_, r2 = one('hover_snap', q, rg, none)
assert all(0 <= v <= 1.0 + 1e-9 for v in r2['after']['req_ground_mm'].values()) and r2['flags_after'] == 0, r2
# 3. object (sphere 8 cm) pushed 1.5 cm into the right hand
q = stand_pose()
geo = G.RobotGeometry(m, surf, 0.0); geo.set(q)
Xd = geo.dense_world(); hand = Xd[surf['region'] == RID['R_hand']]
c_hand = hand.mean(0); out_dir = np.array([0, -1.0, 0])          # outward from the right hand (lateral)
tip = hand[np.argmax(hand @ out_dir)]
rad = 0.08
centre = tip + out_dir * (rad - 0.015)
ball = trimesh.creation.icosphere(subdivisions=4, radius=rad)
obj = PosedObject('ball', ball, np.eye(3)[None], centre[None])
ro = none.copy(); ro[RID['R_hand']] = True
_, r3 = one('object_pen_and_contact', q, none, ro, obj)
assert r3['after']['obj_min_mm'] >= -0.5 and 0 <= list(r3['after']['req_obj_mm'].values())[0] <= 1.0 and r3['flags_after'] == 0, r3
# 4. known clean
q = stand_pose(); q[2] += 0.0003
rg = none.copy(); rg[RID['L_foot']] = rg[RID['R_foot']] = True
q4, r4 = one('known_clean', q, rg, none)
assert np.array_equal(q4, q), 'clean pose changed'
r4['bit_identical'] = True
# 5. runtime on a 60-frame sequence: stand sunk by a time-varying 0..2 cm, feet required
T = 60
Q = np.repeat(stand_pose()[None], T, 0); Q[:, 2] -= 0.02 * np.sin(np.linspace(0, np.pi, T)) ** 2 - 0.003
RG = np.zeros((T, NR), bool); RG[:, RID['L_foot']] = RG[:, RID['R_foot']] = True
geo = G.RobotGeometry(m, surf, 0.0); pj = PP.Projector(geo)
t0 = time.time()
Qo, fl, fl0, info = PP.run_projection(pj, Q, RG, np.zeros_like(RG), latch_regions=RG)
info['frames'] = T; info['wall_s'] = round(time.time() - t0, 2)
gm = [geo.measure(Qo[t], t)['ground_min'] for t in range(T)]
info['ground_min_after_mm_min'] = round(min(gm) * 1000, 3)
res['sequence_runtime'] = info
print('sequence', json.dumps(info), flush=True)
assert info['frames_flagged'] == 0
out = Path('/home/arenalabs/umr_factory_data/out/tests'); out.mkdir(parents=True, exist_ok=True)
json.dump(res, open(out / f'test_postpass_{name}.json', 'w'), indent=1, default=float)
print('ALL TESTS PASSED', name)
