"""One-step (teacher-forced) parity: each frame starts from UMR-CPU's own warm start/history."""
import sys, time, numpy as np, torch
from surface_retargeting.gpu.bank import Problem, Bank
from surface_retargeting.gpu.solver import UMRSolver
path = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 200
kdt = torch.float64 if (len(sys.argv) > 3 and sys.argv[3] == '64') else torch.float32
bank = Bank([Problem(path)])
sv = UMRSolver(bank, kin_dtype=kdt)
dev = bank.device
prob = torch.zeros(1, dtype=torch.long, device=dev); start = torch.zeros(1, dtype=torch.long, device=dev)
L = torch.tensor([n], device=dev)
Q, stats = sv.run(prob, start, L, teacher=True)
Qg = Q[0].cpu().numpy(); Qc = bank.q_cpu[:n].cpu().numpy()
jq = sv.jq.cpu().numpy()
dj = np.abs(Qg[:, jq] - Qc[:, jq]); dr = np.linalg.norm(Qg[:, :3] - Qc[:, :3], axis=1)
print('joint |dq| max per frame: p50 %.2e p95 %.2e max %.2e' % (np.median(dj.max(1)), np.percentile(dj.max(1), 95), dj.max()))
print('root pos diff m: p50 %.2e max %.2e' % (np.median(dr), dr.max()))
print('qp iters mean', np.mean(stats['qp_iters']), 'max', np.max(stats['qp_iters']), 'rprim max', np.max(stats['qp_rprim']), 'sec', stats['seconds'])
worst = np.argsort(-dj.max(1))[:5]; print('worst frames', worst, dj.max(1)[worst])
