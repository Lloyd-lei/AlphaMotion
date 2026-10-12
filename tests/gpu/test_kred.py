import sys, numpy as np, torch, time
from surface_retargeting.gpu.bank import Problem, Bank
from surface_retargeting.gpu.solver import UMRSolver
P = Problem(sys.argv[1]); bank = Bank([P]); sv = UMRSolver(bank)
dev = bank.device; T = P.T; W = 400
starts = torch.arange(0, T, W, device=dev); B = len(starts); L = torch.clamp(T - starts, max=W)
import itertools
for (dt, eps), cap in itertools.product([(torch.float64, 1e-9), (torch.float32, 1e-6), (torch.float32, 1e-5)], [False, True]):
        kred = str(dt) + ' eps%g' % eps
        sv.qp_dtype = dt; sv.qp_eps = eps; sv._capturing = cap
        Q, st = sv.run(torch.zeros(B, dtype=torch.long, device=dev), starts, L, teacher=True)
        idx = (starts[:, None] + torch.arange(W, device=dev)[None]).clamp(max=T - 1)
        m = (torch.arange(W, device=dev)[None] < L[:, None]) & ((torch.arange(W, device=dev)[None] >= 30) | (starts[:, None] == 0))
        dj = (Q[:, :, sv.jq] - bank.q_cpu[idx][:, :, sv.jq]).abs().amax(-1)[m]
        print(f'kred={kred} fixed24={cap}: joint diff p50 {float(dj.median()):.2e} p99 {float(torch.quantile(dj, .99)):.2e} max {float(dj.max()):.2e} ipm iters {np.mean(st["qp_iters"]):.1f}', flush=True)
sv._capturing = False
