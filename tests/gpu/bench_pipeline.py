import sys, time, json, numpy as np, torch
from surface_retargeting.gpu.bank import Problem, Bank
from surface_retargeting.gpu.pipeline import Retargeter
P = Problem(sys.argv[1]); bank = Bank([P]); R = Retargeter(bank, sys.argv[2] + '/surface.npz')
dev = bank.device; W = int(sys.argv[4])
R.human(0)
for B in [int(x) for x in sys.argv[3].split(',')]:
    starts = torch.linspace(0, P.T - W - 1, B, device=dev).long()
    tm = {}
    torch.cuda.synchronize(); t0 = time.time()
    out = R.run(torch.zeros(B, dtype=torch.long, device=dev), starts, torch.full((B,), W, device=dev), timings=tm)
    torch.cuda.synchronize(); dt = time.time() - t0
    fl = out['flags']
    print(json.dumps(dict(B=B, W=W, sec=round(dt, 2), fps=round(B * W / dt), timings={k: round(v, 2) for k, v in tm.items()},
          flagged_frac=float((fl > 0).float().mean()), fixed_frac=float((out['flags_initial'] > 0).float().mean()),
          iters_mean=float(out['iters'].float().mean()), peakGB=round(torch.cuda.max_memory_allocated() / 1e9, 1))), flush=True)
