"""Teacher-forced parity over a whole clip (windows of W frames solved as a batch; window-start frames excluded
because UMR's latched ground anchors are history dependent)."""
import sys, json, numpy as np, torch
from surface_retargeting.gpu.bank import Problem, Bank
from surface_retargeting.gpu.solver import UMRSolver
P = Problem(sys.argv[1]); bank = Bank([P]); sv = UMRSolver(bank)
dev = bank.device; T = P.T; W = int(sys.argv[2]) if len(sys.argv) > 2 else 400; skip = 30
starts = torch.arange(0, T, W, device=dev); B = len(starts); L = torch.clamp(T - starts, max=W)
Q, st = sv.run(torch.zeros(B, dtype=torch.long, device=dev), starts, L, teacher=True)
idx = (starts[:, None] + torch.arange(W, device=dev)[None]).clamp(max=T - 1)
Qc = bank.q_cpu[idx]
m = (torch.arange(W, device=dev)[None] < L[:, None]) & ((torch.arange(W, device=dev)[None] >= skip) | (starts[:, None] == 0))
dj = (Q[:, :, sv.jq] - Qc[:, :, sv.jq]).abs().amax(-1)[m].cpu().numpy()
dr = (Q[:, :, :3] - Qc[:, :, :3]).norm(dim=-1)[m].cpu().numpy()
print(json.dumps(dict(frames=int(m.sum()), joint_maxabs_p50=float(np.median(dj)), p99=float(np.percentile(dj, 99)), max=float(dj.max()),
      root_p50=float(np.median(dr)), root_max=float(dr.max()), ipm_iters_mean=float(np.mean(st['qp_iters'])), ipm_iters_max=int(np.max(st['qp_iters'])))))
