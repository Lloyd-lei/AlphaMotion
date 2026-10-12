import sys, time, numpy as np, torch
sys.path.insert(0, '/home/arenalabs/umr_gpu')
from surface_retargeting.gpu.bank import Problem, Bank
from surface_retargeting.gpu.solver import UMRSolver
P = Problem(sys.argv[1]); bank = Bank([P]); sv = UMRSolver(bank)
dev = bank.device
for B in [int(x) for x in sys.argv[2].split(',')]:
    T = int(sys.argv[3])
    starts = torch.linspace(0, P.T - T - 1, B, device=dev).long()
    torch.cuda.reset_peak_memory_stats()
    Q, st = sv.run(torch.zeros(B, dtype=torch.long, device=dev), starts, torch.full((B,), T, device=dev), log_every=0, store=True)
    t0 = time.time(); Q, st = sv.run(torch.zeros(B, dtype=torch.long, device=dev), starts, torch.full((B,), T, device=dev)); dt = time.time() - t0
    print(f'B={B} T={T} {dt:.2f}s  agg fps={B*T/dt:.0f}  ms/step={dt/T*1000:.1f} peakGB={torch.cuda.max_memory_allocated()/1e9:.1f} ipm iters={np.mean(st["qp_iters"]):.1f}', flush=True)
