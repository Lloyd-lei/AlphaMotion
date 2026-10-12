import sys, numpy as np, torch
sys.path.insert(0, '/home/arenalabs/umr_gpu')
from surface_retargeting.gpu.bank import Problem, Bank
from surface_retargeting.gpu.solver import UMRSolver, LQRGPU, lqr_filter
P = Problem(sys.argv[1]); bank = Bank([P]); sv = UMRSolver(bank)
Q = bank.q_cpu[None].clone(); T = Q.shape[1]
L = LQRGPU(sv.a, T)
a = L(Q, sv.jq.tolist(), sv.clamp)
b = lqr_filter(Q, torch.tensor([T], device='cuda'), sv.jq.tolist(), sv.a, sv.clamp)
c = np.load(P.args['out'], allow_pickle=True)['qpos']
print('gpu-vs-scipy', float((a - b).abs().max()), ' gpu-vs-UMR-output', float(np.abs(a[0].cpu().numpy() - c).max()))
