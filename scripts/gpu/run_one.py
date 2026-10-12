"""Run the full GPU pipeline on one captured problem (whole clip) and save outputs."""
import sys, json, time, argparse, numpy as np, torch
from surface_retargeting.gpu.bank import Problem, Bank
from surface_retargeting.gpu.pipeline import Retargeter
ap = argparse.ArgumentParser(); ap.add_argument('problem'); ap.add_argument('--robot-dir', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--frames', type=int, default=0); ap.add_argument('--no-strict', action='store_true'); ap.add_argument('--log', type=int, default=500)
a = ap.parse_args()
P = Problem(a.problem); bank = Bank([P])
R = Retargeter(bank, a.robot_dir + '/surface.npz', strict=not a.no_strict)
dev = bank.device; T = P.T if a.frames <= 0 else a.frames
tm = {}
out = R.run(torch.zeros(1, dtype=torch.long, device=dev), torch.zeros(1, dtype=torch.long, device=dev), torch.tensor([T], device=dev), log_every=a.log, timings=tm)
res = {k: (v[0, :T].cpu().numpy() if torch.is_tensor(v) else v) for k, v in out.items() if k not in ('strict_info', 'latch')}
for k in ('req_g', 'req_o'):
    if k in out: res[k] = out[k][0, :T]
if out.get('lift_g') is not None: res['lift_required'] = out['lift_g'][0, :T].cpu().numpy()
# naming follows the CPU factory: qpos is floor-relative (z = 0 is the calibrated ground); floor_offset_removed is
# the calibrated floor height in UMR's raw world (qpos_raw_world = qpos + [0,0,floor_offset_removed] on the root)
if 'qpos' in res:
    res['floor_offset_removed'] = np.array(out['floor'][0], np.float32); res['floor_height'] = np.array(0.0, np.float32)
    res['qpos_raw_world'] = res.pop('q_strict_world')
    res['convention'] = np.array('qpos: z=0 is the calibrated ground (floor-relative). qpos_raw_world / q_umr*: UMR raw world.')
res.pop('floor', None)
np.savez_compressed(a.out, **{k: v for k, v in res.items() if isinstance(v, np.ndarray)},
                    timings=json.dumps(tm), strict_info=json.dumps(out.get('strict_info', {})), problem=a.problem)
print(json.dumps(dict(timings={k: round(v, 2) for k, v in tm.items()}, frames=T,
      flagged=int((res.get('flags', np.zeros(1)) > 0).sum()), flags_hist={str(b): int(((res['flags'] & b) > 0).sum()) for b in (1, 2, 4, 8, 16, 32, 64, 128)} if 'flags' in res else None,
      fixed=int((res['flags_initial'] > 0).sum()) if 'flags_initial' in res else None, info=out.get('strict_info'))))
