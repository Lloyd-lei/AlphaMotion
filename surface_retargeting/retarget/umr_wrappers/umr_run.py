"""Wrapper around UMR retarget scripts (upstream code untouched).
Usage: umr_run.py <upstream_script.py> [args...]
Env:
  UMR_CAPTURE=path.npz   save scaled source surface slots [T,4096,3] (retarget frame, z-up, ground z=0)
  UMR_QP_FALLBACK=1      on Clarabel failure retry with all inequality rows softened (cost 1e3), then dq=0
"""
from surface_retargeting.config import CFG as SRCFG  # machine paths: configs/sr.toml
import os, sys, json, runpy, atexit
import numpy as np
SCRIPTS = f'{SRCFG.umr_root}/scripts'
sys.path.insert(0, SCRIPTS); sys.path.insert(0, os.path.dirname(SCRIPTS))
import smpl_surface_retarget_common as common

cap = os.environ.get('UMR_CAPTURE')
if cap:
    _orig_gc = common.compute_source_slot_ground_contact
    def _gc(source_slots, snap_threshold=0.005):
        np.savez_compressed(cap, source_slots=np.asarray(source_slots, np.float32))
        print(f'[umr_run] captured source slots {np.shape(source_slots)} -> {cap}')
        return _orig_gc(source_slots, snap_threshold=snap_threshold)
    common.compute_source_slot_ground_contact = _gc

stats = {'qp_calls': 0, 'qp_fail_soft_ok': 0, 'qp_fail_zero': 0, 'statuses': {}}
if os.environ.get('UMR_QP_FALLBACK') == '1':
    _orig_qp = common.solve_clarabel_qp_step
    def _qp(J, residual, damping, lower, upper, ineq_A=None, ineq_b=None, ineq_soft_cost=0.0, ineq_soft_costs=None, **kw):
        stats['qp_calls'] += 1
        try:
            return _orig_qp(J, residual, damping, lower, upper, ineq_A, ineq_b, ineq_soft_cost, ineq_soft_costs, **kw)
        except RuntimeError as e:
            s = str(e).split('=')[-1]; stats['statuses'][s] = stats['statuses'].get(s, 0) + 1
            if ineq_A is not None and np.asarray(ineq_A).shape[0] > 0:
                try:
                    n = np.asarray(ineq_A).shape[0]
                    r = _orig_qp(J, residual, damping, lower, upper, ineq_A, ineq_b, 0.0, np.full(n, 1e3), **kw)
                    stats['qp_fail_soft_ok'] += 1
                    return r
                except RuntimeError:
                    pass
            stats['qp_fail_zero'] += 1
            return np.zeros(np.asarray(J).shape[1])
    common.solve_clarabel_qp_step = _qp
    def _dump():
        out = os.environ.get('UMR_STATS', '')
        print('[umr_run] qp stats', json.dumps(stats))
        if out: json.dump(stats, open(out, 'w'))
    atexit.register(_dump)

script = sys.argv[1]
sys.argv = [script] + sys.argv[2:]
runpy.run_path(script, run_name='__main__')
