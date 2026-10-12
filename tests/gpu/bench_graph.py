import sys, time, numpy as np, torch
from surface_retargeting.gpu.bank import Problem, Bank
from surface_retargeting.gpu.solver import UMRSolver
P = Problem(sys.argv[1]); bank = Bank([P]); sv = UMRSolver(bank)
import os
if os.environ.get("QP32"): sv.qp_dtype = torch.float32; sv.qp_eps = 1e-6
if os.environ.get("QPIT"): sv.qp_iters_graph = int(os.environ["QPIT"])
dev = bank.device
T = int(sys.argv[3])
for B in [int(x) for x in sys.argv[2].split(',')]:
    starts = torch.linspace(0, P.T - T - 1, B, device=dev).long()
    torch.cuda.reset_peak_memory_stats()
    Q, st = sv.run_graph(torch.zeros(B, dtype=torch.long, device=dev), starts, torch.full((B,), T, device=dev))
    dt = st['seconds']
    if B <= 64:
        Qe, _ = sv.run(torch.zeros(B, dtype=torch.long, device=dev), starts, torch.full((B,), T, device=dev))
        print('  graph vs eager max |dq|', float((Q - Qe).abs().max()))
    print(f'B={B} T={T} {dt:.2f}s agg fps={B*T/dt:.0f} ms/frame-step={dt/T*1000:.1f} peakGB={torch.cuda.max_memory_allocated()/1e9:.1f}', flush=True)
