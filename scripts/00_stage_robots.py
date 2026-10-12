"""Stage all core robots: standalone MJCF, validated/fixed T-pose, documented rest-overlap exclusions, UMR config.
Usage: python scripts/stage_robots.py [--only name1,name2]
Writes DATA/robots/<name>/{<name>.xml, umr_robot.json, staging.json} and DATA/robots/staging_summary.json
"""
from surface_retargeting.config import CFG as SRCFG  # machine paths: configs/sr.toml
import argparse, json, sys, time
from pathlib import Path
import mujoco, numpy as np
from surface_retargeting.robots import robots as R
from surface_retargeting.paths import ROBOTS, DRAFT_CONFIGS

# T-poses shipped by UMR's authors (robot_configs/*_example.json) are kept as-is even if our metric is stricter
UMR_VERIFIED = {'unitree_g1_29dof', 'unitree_g1_23dof', 'unitree_h2', 'booster_k1'}
ap = argparse.ArgumentParser(); ap.add_argument('--only', default='')
a = ap.parse_args()
inv = R.load_inventory()
if a.only:
    inv = [r for r in inv if r['name'] in a.only.split(',')]
summary = {}
for rec in inv:
    t0 = time.time(); name = rec['name']
    src = rec['mjcf']
    draft = SRCFG.load_json(DRAFT_CONFIGS / f'humanoid_retarget_{name}.json')['robot']
    p, m = R.write_staged(src, name)
    # facing convention: left hand must be on +y at qpos0
    d = mujoco.MjData(m); mujoco.mj_resetData(m, d); mujoco.mj_forward(m, d)
    hb = {s: R.mid(m, mujoco.mjtObj.mjOBJ_BODY, rec['hands'][s]['hand_body']) for s in 'LR'}
    facing_ok = bool(d.xpos[hb['L']][1] > d.xpos[hb['R']][1])
    tpose = {k: v for k, v in (draft.get('tpose_qpos') or {}).items() if R.mid(m, mujoco.mjtObj.mjOBJ_JOINT, k) >= 0}
    met0 = R.tpose_metrics(m, tpose, rec)
    fixed = False
    if not met0['pass'] and name not in UMR_VERIFIED:
        tpose = R.solve_tpose(m, rec, tpose)
        fixed = True
    met = R.tpose_metrics(m, tpose, rec)
    excl, flagged = R.rest_overlap_exclusions(m, tpose, manual=R.MANUAL_EXCLUSIONS.get(name, ()))
    p, m2 = R.write_staged(src, name, [(e['body1'], e['body2']) for e in excl])
    # verify: no remaining overlaps among non-excluded pairs at T-pose
    d2 = mujoco.MjData(m2); R.set_tpose(m2, d2, tpose)
    remaining = R.overlapping_pairs(m2, d2, d2.qpos.copy())
    names = R.body_names(m2)
    ratio = R.bbox_center_ratio(m2, tpose, rec['umr_point_cloud_center'])
    cfg = {'robot': {'name': name, 'xml': str(p), 'point_cloud_center': rec['umr_point_cloud_center'],
                     'tpose_qpos': tpose},
           'correspondence': {'dataset': {'smpl_center': 'bbox_ratio', 'bbox_center_ratio': round(ratio, 6)}}}
    json.dump(cfg, open(ROBOTS / name / 'umr_robot.json', 'w'), indent=1)
    st = dict(name=name, src_mjcf=src, staged_mjcf=str(p), nq=int(m2.nq), nv=int(m2.nv), facing_plus_x=facing_ok,
              locked_joints=R.LOCKED_JOINTS.get(name, []), tpose=tpose, tpose_fixed=fixed,
              tpose_metrics_draft=met0, tpose_metrics_final=met, bbox_center_ratio=ratio,
              bbox_center_ratio_draft=rec.get('pcc_bbox_ratio_tpose'),
              exclusions=excl, flagged_nonadjacent_rest_overlaps=flagged,
              remaining_tpose_overlaps=[(names[k[0]], names[k[1]], round(-v * 1000, 2)) for k, v in remaining.items()],
              n_collision_geoms=len(R.robot_geoms(m2)), seconds=round(time.time() - t0, 1))
    json.dump(st, open(ROBOTS / name / 'staging.json', 'w'), indent=1)
    summary[name] = {k: st[k] for k in ('nq', 'facing_plus_x', 'tpose_fixed', 'bbox_center_ratio', 'seconds')}
    summary[name]['tpose_pass'] = met['pass']
    summary[name]['n_excl'] = len(excl); summary[name]['n_flagged'] = len(flagged)
    summary[name]['remaining'] = len(remaining)
    print(name, json.dumps(summary[name]), 'L', met['L'], 'R', met['R'], flush=True)
sp = ROBOTS / 'staging_summary.json'
old = json.load(open(sp)) if sp.exists() else {}
old.update(summary)
json.dump(old, open(sp, 'w'), indent=1)
