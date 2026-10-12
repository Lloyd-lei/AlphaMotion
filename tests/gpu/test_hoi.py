"""GPU object motion + object-contact maps vs UMR-CPU capture (OmniContact soccer sample)."""
from surface_retargeting.config import CFG as SRCFG  # machine paths: configs/sr.toml
import sys, json, glob, numpy as np, torch
from surface_retargeting.gpu.bank import Problem
from surface_retargeting.gpu.hoi import ObjectAsset, object_contact_maps
P = Problem(sys.argv[1]); z = P.z; a = P.args
d = f'{SRCFG.umr_root}/sample_data/omnicontact/soccer/case3_kick_right/20260330000586_1_1775269234'
xml = glob.glob(d + '/*.xml')[0]; prop = glob.glob(d + '/prop_*.csv')[0]
import time; _t = time.time()
asset = ObjectAsset(xml); print("asset", time.time() - _t, flush=True)
T = P.T
stride = int(sys.argv[2]) if len(sys.argv) > 2 else 1
fids = np.arange(T) * stride
# smpl scale from the captured object position vs the unit motion
m1 = ObjectAsset.motion(prop, fids, 1.0, 'y')
cap = z['obj_pose']
s = float(np.median(np.linalg.norm(cap[:, :3], axis=1) / np.maximum(np.linalg.norm(m1[:, :3], axis=1), 1e-9)))
best = None
for up in ('y', 'z'):
    mo = ObjectAsset.motion(prop, fids, s, up)
    e = np.abs(mo - cap).max()
    print('output_up', up, 'scale %.6f' % s, 'pose max abs err', e)
    if best is None or e < best[0]: best = (e, up, mo)
mo = best[2]
pose8 = torch.as_tensor(np.concatenate([mo, np.full((T, 1), s)], 1), device='cuda')
src = torch.as_tensor(z['src'], device='cuda')
_t = time.time(); r = object_contact_maps(asset, src, pose8, a); torch.cuda.synchronize(); print("maps", time.time() - _t, flush=True)
gs, gd = r['oc_slot'].cpu().numpy(), r['oc_d'].cpu().numpy()
cs, cn, cd = z['oc_slot'], z['oc_n'], z['oc_d']; cvec = z['oc_vec']; cpt = z['oc_pt']
setdiff = 0; frames_diff = 0; dmax = 0; vmax = 0; ptmax = 0
gv, gp = r['oc_vec'].cpu().numpy(), r['oc_pt'].cpu().numpy()
for t in range(T):
    A = {int(x): i for i, x in enumerate(gs[t]) if x >= 0}; Bc = {int(x): i for i, x in enumerate(cs[t, :cn[t]])}
    if set(A) != set(Bc):
        frames_diff += 1; setdiff += len(set(A) ^ set(Bc))
    for k in set(A) & set(Bc):
        dmax = max(dmax, abs(gd[t, A[k]] - cd[t, Bc[k]])); vmax = max(vmax, np.abs(gv[t, A[k]] - cvec[t, Bc[k]]).max())
        ptmax = max(ptmax, np.abs(gp[t, A[k]] - cpt[t, Bc[k]]).max())
print(json.dumps(dict(frames=T, active_frames_cpu=int((cn > 0).sum()), active_frames_gpu=int((gs >= 0).any(1).sum()), frames_with_set_diff=frames_diff,
                      slot_symdiff=setdiff, dist_maxabs_mm=dmax * 1e3, vec_maxabs_mm=vmax * 1e3, objpt_maxabs_mm=ptmax * 1e3, object_kind=asset.sdf.kind)))
