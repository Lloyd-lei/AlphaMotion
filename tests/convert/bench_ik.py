"""IK step-time vs lane count. usage: bench_ik.py B [K] [steps]"""
from surface_retargeting.config import CFG as SRCFG  # machine paths: configs/sr.toml
import os
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", f'{SRCFG.convert_work}/.inductor_cache')
import sys, time, json
from pathlib import Path
import numpy as np
import torch
from surface_retargeting.ingest.convert_gpu import prep, gpu_ik

B = int(sys.argv[1]); K = int(sys.argv[2]) if len(sys.argv) > 2 else 8; steps = int(sys.argv[3]) if len(sys.argv) > 3 else 64
clip = f'{SRCFG.data_root}/raw/hiphi/extracted/part_0002/HiPHI/data/Adjusting/adjust/Adjusting-adjust_0078'
cpu = f'{SRCFG.data_root}/smplx/hiphi/data/Adjusting/adjust/Adjusting-adjust_0078/motion_actor_smplx.npz'
p = prep.prep_clip(clip + "/motion_actor.bvh")
betas = np.load(cpu)["betas"]
ft, rest = prep.shape_clip(betas, p["targets"], p["reference"])
F = len(ft)
ik = gpu_ik.FastMinkIK(np.repeat(rest[None], B, 0), device="cuda")
t0 = time.time()
run = gpu_ik.GraphRunner(ik, K=K)
tc = time.time() - t0
T = torch.tensor(ft, dtype=torch.float64, device="cuda"); R = torch.tensor(p["reference"], dtype=torch.float64, device="cuda")
off = (torch.arange(B, device="cuda") * 37) % (F - K * steps - 1)
ts = []
for s in range(steps):
    idx = off[:, None] + s * K + torch.arange(K, device="cuda")[None]
    run.tgt.copy_(T[idx]); run.ref.copy_(R[idx])
    run.active.fill_(True); run.first.copy_((idx == off[:, None]) & (s == 0))
    torch.cuda.synchronize(); t = time.time()
    run.run(); torch.cuda.synchronize(); ts.append(time.time() - t)
ms = 1000 * np.median(ts[2:]) / K
print(json.dumps(dict(B=B, K=K, compile_s=round(tc, 1), ms_per_frame_step=round(ms, 3), frames_per_s=round(B / ms * 1000),
                      mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2))), flush=True)
