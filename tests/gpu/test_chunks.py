"""Chunk-boundary error: chunked (Lc, Ov) vs unchunked GPU solve of the same clip, both fed by GPU preprocessing;
plus end-to-end parity of the unchunked GPU run vs UMR-CPU (pre- and post-LQR)."""
import sys, json, time, argparse, numpy as np, torch
from surface_retargeting.gpu.bank import Problem, Bank
from surface_retargeting.gpu.solver import UMRSolver, LQRGPU
from surface_retargeting.gpu.source_gpu import SourcePrep, load_smplx_sequence
from surface_retargeting.gpu.chunks import ChunkRunner, make_chunks, stitch
ap = argparse.ArgumentParser(); ap.add_argument('problem'); ap.add_argument('--Lc', type=int, default=512); ap.add_argument('--Ov', type=int, default=32)
ap.add_argument('--qp', default='fp32'); ap.add_argument('--out', default='')
a = ap.parse_args()
P = Problem(a.problem); dev = 'cuda'
seq = load_smplx_sequence(P); T = P.T
pose = torch.as_tensor(np.asarray(seq['pose_aa'], np.float32)[:T], device=dev); tr = torch.as_tensor(np.asarray(seq['trans_orig'], np.float32)[:T], device=dev)

def solver():
    bank = Bank([P]); sv = UMRSolver(bank)
    if a.qp == 'fp32':
        sv.qp_dtype = torch.float32; sv.qp_eps = 1e-6
    return sv

res = {}
sv1 = solver(); prep = SourcePrep(P, dev)
r1 = ChunkRunner(sv1, prep, 1, T, Tb=64)
t0 = time.time(); Qf = r1.run([pose], [tr], torch.zeros(1, dtype=torch.long, device=dev), torch.zeros(1, dtype=torch.long, device=dev),
                               seq.get('gender', 'neutral'), seq.get('beta'))[0]
res['unchunked_s'] = round(time.time() - t0, 1)
ch = make_chunks(T, a.Lc, a.Ov)
sv2 = solver(); r2 = ChunkRunner(sv2, SourcePrep(P, dev), len(ch), a.Lc, Tb=64)
cs = torch.as_tensor([c[0] for c in ch], device=dev)
Qc = r2.run([pose], [tr], torch.zeros(len(ch), dtype=torch.long, device=dev), cs, seq.get('gender', 'neutral'), seq.get('beta'))
Qs = stitch(ch, Qc, T)
L_full = LQRGPU(sv1.a, T); L_ch = LQRGPU(sv1.a, a.Lc)
Ff = L_full(Qf[None], sv1.jq.tolist(), sv1.clamp)[0]
Fs = stitch(ch, L_ch(Qc, sv1.jq.tolist(), sv1.clamp), T)
jq = sv1.jq
def surf(Q):
    K = sv1.kt.fk(Q); Pw, _, _, _ = sv1.slots_world(K, torch.zeros(len(Q), dtype=torch.long, device=dev)); return Pw
sw = [ (s + (ch[k-1][0] + ch[k-1][1])) // 2 for k, (s, L) in enumerate(ch) if k > 0]
bmask = torch.zeros(T, dtype=torch.bool, device=dev)
for w in sw: bmask[max(0, w - 8):w + 8] = True
for name, A, Bq in (('raw', Qs, Qf), ('lqr', Fs, Ff)):
    dj = (A[:, jq] - Bq[:, jq]).abs().amax(-1)
    dp = (surf(A) - surf(Bq)).norm(dim=-1).amax(-1) * 1000
    res[name] = dict(joint_max_rad=float(dj.max()), joint_max_at_boundaries=float(dj[bmask].max()) if len(sw) else 0.0,
                     joint_p99=float(torch.quantile(dj, .99)), surf_max_mm=float(dp.max()), surf_max_at_boundaries_mm=float(dp[bmask].max()) if len(sw) else 0.0,
                     surf_p99_mm=float(torch.quantile(dp, .99)))
# parity of the unchunked end-to-end GPU run vs UMR-CPU
qc = torch.as_tensor(P.z['q_cpu'], device=dev)
fc = torch.as_tensor(np.load(P.args['out'], allow_pickle=True)['qpos'].astype(np.float64), device=dev)
for name, A, Bq in (('vs_cpu_raw', Qf, qc), ('vs_cpu_final', Ff, fc)):
    dj = (A[:, jq] - Bq[:, jq]).abs().amax(-1)
    dp = (surf(A) - surf(Bq)).norm(dim=-1)
    res[name] = dict(joint_p50=float(dj.median()), joint_p99=float(torch.quantile(dj, .99)), joint_max=float(dj.max()),
                     root_max_mm=float((A[:, :3] - Bq[:, :3]).norm(dim=-1).max() * 1000),
                     surf_mean_mm=float(dp.mean() * 1000), surf_p99_mm=float(torch.quantile(dp.amax(-1), .99) * 1000), surf_max_mm=float(dp.max() * 1000))
res['chunks'] = len(ch); res['switch_frames'] = sw
print(json.dumps(res, indent=1))
if a.out:
    np.savez_compressed(a.out, q_full=Ff.cpu().numpy(), q_chunked=Fs.cpu().numpy(), q_full_raw=Qf.cpu().numpy(), q_chunked_raw=Qs.cpu().numpy())
# convergence of a chunk's warm-up towards the unchunked solution: error vs frames since chunk start
curve = []
for k, (s, L) in enumerate(ch):
    if k == 0: continue
    e = (Qc[k, :, jq] - Qf[s:s + L, jq]).abs().amax(-1)
    curve.append(e.cpu().numpy())
curve = np.stack(curve)
for t in [0, 8, 16, 32, 64, 96, 128, 192, 256, 384]:
    if t < curve.shape[1]:
        print(f'  t={t:4d} since chunk start: max over chunks {curve[:, t].max():.2e}  median {np.median(curve[:, t]):.2e}')
