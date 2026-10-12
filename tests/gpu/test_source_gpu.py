"""GPU source preprocessing vs UMR-CPU captured solver inputs (same clip, same robot)."""
import sys, time, json, numpy as np, torch
sys.path.insert(0, '/home/arenalabs/umr_gpu')
from surface_retargeting.gpu.bank import Problem
from surface_retargeting.gpu.source_gpu import SMPLXForward, SourcePrep, load_smplx_sequence
P = Problem(sys.argv[1]); z = P.z
assert 'x_bind_face_ids' in z.files, 'capture lacks static extras'
seq = load_smplx_sequence(P)
dev = 'cuda'
pose = torch.as_tensor(np.asarray(seq['pose_aa'], np.float32), device=dev); tr = torch.as_tensor(np.asarray(seq['trans_orig'], np.float32), device=dev)
T = P.T
pose, tr = pose[:T], tr[:T]
sm = SMPLXForward(seq.get('gender', 'neutral'), seq.get('beta'), batch=2048)
prep = SourcePrep(P)
torch.cuda.synchronize(); t0 = time.time()
V, J = sm(pose, tr)
torch.cuda.synchronize(); t1 = time.time()
o = prep.frames(V, J)
torch.cuda.synchronize(); t2 = time.time()
res = dict(frames=T, smplx_s=round(t1 - t0, 3), prep_s=round(t2 - t1, 3), scale=prep.scale)
src = torch.as_tensor(z['src'], device=dev)
res['src_maxabs_mm'] = float((o['src'] - src).abs().max() * 1e3)
nt = torch.as_tensor(z['ntgt'], device=dev)
res['ntgt_max_deg'] = float(torch.rad2deg(torch.acos((o['ntgt'] * nt).sum(-1).clamp(-1, 1))).max())
res['gd_maxabs_mm'] = float((o['gd'] - torch.as_tensor(z['gd'], device=dev)).abs().max() * 1e3)
res['gw_maxabs_mm'] = float((o['gw'] - torch.as_tensor(z['gw'], device=dev)).abs().max() * 1e3)
res['gd_snap_mismatch'] = int(((o['gd'] == 0) != (torch.as_tensor(z['gd'], device=dev) == 0)).sum())
qi = o['q_init'][0].cpu().numpy(); res['q_init0_maxabs'] = float(np.abs(qi - z['q_init'][0]).max())
if 'sc_pairs' in o:
    sel = z['selected']; pos = np.full(4096, -1); pos[sel] = np.arange(len(sel))
    gp = o['sc_pairs'].cpu().numpy(); gd_ = o['sc_d'].cpu().numpy(); gw_ = o['sc_w'].cpu().numpy()
    cp = z['sc_pairs']; cd = z['sc_d']; cw = z['sc_w']
    nset = 0; nfr = 0; dmax = 0; wmax = 0
    for t in range(T):
        A = {tuple(sorted(p)): (d, w) for p, d, w in zip(gp[t].tolist(), gd_[t], gw_[t]) if p[0] >= 0}
        Bm = {tuple(sorted((pos[p[0]], pos[p[1]]))): (d, w) for p, d, w in zip(cp[t].tolist(), cd[t], cw[t]) if p[0] >= 0}
        if set(A) != set(Bm):
            nfr += 1; nset += len(set(A) ^ set(Bm))
        for k in set(A) & set(Bm):
            dmax = max(dmax, abs(A[k][0] - Bm[k][0])); wmax = max(wmax, abs(A[k][1] - Bm[k][1]))
    res.update(sc_frames_with_set_diff=nfr, sc_pairs_symdiff=nset, sc_dist_maxabs_mm=dmax * 1e3, sc_weight_maxabs=wmax)
print(json.dumps(res))
