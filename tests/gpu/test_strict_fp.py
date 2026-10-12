"""Frame-parallel strict pass on the UMR-CPU output of a clip: GPU flags + timing (warm)."""
from surface_retargeting.config import CFG as SRCFG  # machine paths: configs/sr.toml
import sys, time, json, numpy as np, torch
from surface_retargeting.gpu.bank import Problem, Bank
from surface_retargeting.gpu.solver import UMRSolver
from surface_retargeting.strict.strict import RobotPoints, StrictProjector
from surface_retargeting.gpu.geometry import SelfCollision
from surface_retargeting.strict import strict_fp
from surface_retargeting.gpu import labels as L
from surface_retargeting.gpu.pipeline import Retargeter
P = Problem(sys.argv[1]); robot = sys.argv[2]
bank = Bank([P]); sv = UMRSolver(bank); dev = 'cuda'
pts = RobotPoints(sv.kt, f'{SRCFG.work}/robots/{robot}/surface.npz', dev)
proj = StrictProjector(sv.kt, pts, SelfCollision(sv.kt, dev), None, lim_lo=sv.j_lo, lim_hi=sv.j_hi, jq=sv.jq, jd=sv.jd, k_ground=96, k_self=32)
proj.qp_dtype = torch.float32; proj.qp_eps = 1e-6; proj.qp_iters = 24
R = Retargeter(bank, f'{SRCFG.work}/robots/{robot}/surface.npz', strict=False)
h = R.human(0)
fin = torch.as_tensor(np.load(P.args['out'], allow_pickle=True)['qpos'].astype(np.float64), device=dev)
T = len(fin); reps = int(sys.argv[3]) if len(sys.argv) > 3 else 1
Q = fin[None].repeat(reps, 1, 1)
piece = torch.arange(reps, device=dev)[:, None].expand(reps, T).contiguous(); tl = torch.arange(T, device=dev)[None].expand(reps, T).contiguous()
floor = torch.full((reps, T), h['floor'], device=dev)
hg = torch.as_tensor(h['g_contact'], device=dev)[None].expand(reps, -1, -1)
strict_fp.strict_chains(proj, Q[:1, :200], piece[:1, :200], tl[:1, :200], floor[:1, :200], hg[:1, :200])   # warm-up/compile
torch.cuda.synchronize(); t0 = time.time()
Qs, flags, f0, req_g, req_o, info = strict_fp.strict_chains(proj, Q, piece, tl, floor, hg)
torch.cuda.synchronize(); dt = time.time() - t0
info.update(seconds=round(dt, 2), fps=round(reps * T / dt), flags_hist={str(b): int(((flags & b) > 0).sum()) for b in (1, 2, 4, 8, 16)})
print(json.dumps(info))
