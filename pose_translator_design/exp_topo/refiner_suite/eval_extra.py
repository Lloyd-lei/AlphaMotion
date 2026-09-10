"""补跑条件 (用 suite_core.Robot, 同一批行), 结果并入 out/<robot>.json. 用法: eval_extra.py <robot> <cond,...>"""
import sys, json, time, numpy as np, torch
import suite_core as SC
from refine import geo_deg
robot, conds = sys.argv[1], sys.argv[2].split(',')
out = f"/home/arenalabs/tmp/refiner_suite/{__import__('os').environ.get('OUT','out')}/{robot}.json"; res = json.load(open(out))
z = np.load(out.replace('.json', '_q.npz')); rows = torch.as_tensor(z['rows']); cid = z['cid']
R = SC.Robot(robot, rows, log=lambda *a: None); P = R.P; HANDS, FEET, ARM, LEG, NONROOT = R.HANDS, R.FEET, R.ARM, R.LEG, R.NONROOT
def metrics(q, gR, gp):
    m = {}
    m['hand_mm'] = torch.stack([(gp[:, j] - R.tgt_pos[j]).norm(dim=-1) for j in HANDS], 1).mean(1) * 10
    m['hand_deg'] = torch.stack([geo_deg(gR[:, j], R.tgt_rot[j]) for j in HANDS], 1).mean(1)
    m['foot_mm'] = torch.stack([(gp[:, j] - R.tgt_pos[j]).norm(dim=-1) for j in FEET], 1).mean(1) * 10
    m['foot_h_mm'] = torch.stack([(gp[:, j, 1] - R.tgt_pos[j][:, 1]).abs() for j in FEET], 1).mean(1) * 10
    m['foot_deg'] = torch.stack([geo_deg(gR[:, j], R.tgt_rot[j]) for j in FEET], 1).mean(1)
    def axdeg(j, side, key):
        a_ = torch.as_tensor(P[side][key], dtype=gR.dtype); u = gR[:, j] @ a_; v = R.tgt_rot[j] @ a_
        return torch.rad2deg(torch.arccos((u * v).sum(-1).clamp(-1, 1)))
    m['foot_normal_deg'] = torch.stack([axdeg(FEET[0], 'left', 'sole_normal_local'), axdeg(FEET[1], 'right', 'sole_normal_local')], 1).mean(1)
    m['foot_fwd_deg'] = torch.stack([axdeg(FEET[0], 'left', 'foot_forward_local'), axdeg(FEET[1], 'right', 'foot_forward_local')], 1).mean(1)
    m['pelvis_h'] = -torch.minimum(gp[:, FEET[0], 1], gp[:, FEET[1], 1])
    if not R.ZS:
        Rg, Pg, q_gt = R.Rg, R.Pg, R.q_gt
        m['rot_all'] = geo_deg(gR[:, NONROOT], Rg[:, NONROOT]).mean(1)
        m['rot_arm'] = geo_deg(gR[:, ARM], Rg[:, ARM]).mean(1); m['rot_leg'] = geo_deg(gR[:, LEG], Rg[:, LEG]).mean(1)
        m['pos_all'] = (gp[:, NONROOT] - Pg[:, NONROOT]).norm(dim=-1).mean(1)
        m['pos_arm'] = (gp[:, ARM] - Pg[:, ARM]).norm(dim=-1).mean(1); m['pos_leg'] = (gp[:, LEG] - Pg[:, LEG]).norm(dim=-1).mean(1)
        m['q_deg'] = torch.rad2deg((q - q_gt).abs()).flatten(1).mean(1)
        m['pelvis_h_err'] = (m['pelvis_h'] - (-torch.minimum(Pg[:, FEET[0], 1], Pg[:, FEET[1], 1]))).abs()
    return {k: v.numpy() for k, v in m.items()}
# 自检: suite_core 复现 eval_suite 的一个条件 (hands) 到 1e-3°
if 'hands' in res['conds'] and 'hands' in z:
    q, gR, gp = R.run_cond('hands'); m = metrics(q, gR, gp)
    print(f'[{robot}] selfcheck hands: rot_all suite_core={m.get("rot_all", np.zeros(1)).mean():.3f} eval_suite={res["conds"]["hands"].get("rot_all", {}).get("mean", float("nan")):.3f} hand_mm {m["hand_mm"].mean():.3f} vs {res["conds"]["hands"]["hand_mm"]["mean"]:.3f} | q maxdiff {np.abs(q.float().numpy() - z["hands"]).max():.2e}', flush=True)
QS = {}
for name in conds:
    t1 = time.time(); q, gR, gp = R.run_cond(name); m = metrics(q, gR, gp); dt = time.time() - t1
    summ = {k: {'mean': float(v.mean()), 'p95': float(np.percentile(v, 95))} for k, v in m.items()}
    summ['per_clip'] = {k: [float(v[cid == c].mean()) for c in range(cid.max() + 1)] for k, v in m.items() if k in ('rot_all', 'rot_arm', 'rot_leg', 'hand_mm', 'foot_mm', 'pelvis_h_err')}
    summ['sec_per_frame_ms'] = dt / R.B * 1000; res['conds'][name] = summ; QS[name] = q.float().numpy()
    line = ' '.join(f'{k}={summ[k]["mean"]:.2f}' for k in ('rot_all', 'rot_arm', 'rot_leg', 'pos_all', 'pelvis_h_err', 'hand_mm', 'hand_deg', 'foot_mm', 'foot_deg') if k in summ)
    print(f'[{robot}] {name:14s} {line}  ({dt:.0f}s)', flush=True)
json.dump(res, open(out, 'w'), indent=1)
d = dict(z); d.update(QS); np.savez_compressed(out.replace('.json', '_q.npz'), **d)
print(f'[{robot}] merged', flush=True)
