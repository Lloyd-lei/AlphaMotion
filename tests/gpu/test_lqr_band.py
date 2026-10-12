import sys, numpy as np, torch
from surface_retargeting.gpu.bank import Problem, Bank
from surface_retargeting.gpu.solver import UMRSolver, LQRGPU
from surface_retargeting.gpu.lqr_band import lqr_chains
P = Problem(sys.argv[1]); bank = Bank([P]); sv = UMRSolver(bank)
Q = bank.q_cpu.clone(); T = Q.shape[0]
# chain 0: whole clip; chain 1: two pieces [0,1000) and [1000,2500) then idle
piece = torch.full((2, T), -1, dtype=torch.long, device='cuda'); tl = piece.clone(); rem = piece.clone()
piece[0] = 0; tl[0] = torch.arange(T); rem[0] = T - torch.arange(T)
piece[1, :1000] = 1; tl[1, :1000] = torch.arange(1000); rem[1, :1000] = 1000 - torch.arange(1000)
piece[1, 1000:2500] = 2; tl[1, 1000:2500] = torch.arange(1500); rem[1, 1000:2500] = 1500 - torch.arange(1500)
QQ = Q[None].repeat(2, 1, 1)
import time; t0 = time.time()
F = lqr_chains(QQ, piece, tl, rem, sv.a, sv.jq.tolist(), sv.clamp); torch.cuda.synchronize(); print('band lqr s', round(time.time() - t0, 2))
ref0 = LQRGPU(sv.a, T)(Q[None], sv.jq.tolist(), sv.clamp)[0]
ref1 = LQRGPU(sv.a, 1000)(Q[None, :1000], sv.jq.tolist(), sv.clamp)[0]
ref2 = LQRGPU(sv.a, 1500)(Q[None, 1000:2500], sv.jq.tolist(), sv.clamp)[0]
print('whole clip', float((F[0] - ref0).abs().max()), ' piece1', float((F[1, :1000] - ref1).abs().max()), ' piece2', float((F[1, 1000:2500] - ref2).abs().max()))
