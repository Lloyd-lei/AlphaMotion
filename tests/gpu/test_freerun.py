"""Free-running parity: GPU streams the whole clip from its own warm starts; compare to UMR-CPU (pre- and post-LQR)."""
import sys, time, numpy as np, torch
from surface_retargeting.gpu.bank import Problem, Bank
from surface_retargeting.gpu.solver import UMRSolver, lqr_filter
path = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 0
P = Problem(path); bank = Bank([P])
sv = UMRSolver(bank)
dev = bank.device
T = P.T if n <= 0 else n
prob = torch.zeros(1, dtype=torch.long, device=dev); start = torch.zeros(1, dtype=torch.long, device=dev)
L = torch.tensor([T], device=dev)
Q, stats = sv.run(prob, start, L, teacher=False, log_every=1000)
Qf = lqr_filter(Q, L, sv.jq.tolist(), sv.a, sv.clamp)
Qg = Q[0].cpu().numpy(); Qc = bank.q_cpu[:T].cpu().numpy()
jq = sv.jq.cpu().numpy()
def rep(name, A, Bm):
    dj = np.abs(A[:, jq] - Bm[:, jq]); dr = np.linalg.norm(A[:, :3] - Bm[:, :3], axis=1)
    print(f'{name}: joint maxabs/frame p50 {np.median(dj.max(1)):.2e} p95 {np.percentile(dj.max(1),95):.2e} max {dj.max():.2e} | mean abs {dj.mean():.2e} | root p50 {np.median(dr):.2e} max {dr.max():.2e}')
rep('pre-LQR ', Qg, Qc)
if n <= 0:
    fin = np.load(str(P.z['umr_out']) or '', allow_pickle=True) if str(P.z['umr_out']) else None
    import glob, json
    out = json.loads(str(P.z['args_json']))['out']
    Fc = np.load(out, allow_pickle=True)['qpos']
    rep('post-LQR', Qf[0].cpu().numpy(), Fc)
print('seconds', round(stats['seconds'], 1), 'fps', round(T / stats['seconds'], 1), 'qp iters mean', np.mean(stats['qp_iters']))
