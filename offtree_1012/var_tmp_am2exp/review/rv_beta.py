"""Is a constant beta_vis right off-stance?  Per robot: lowest posed surface point (h = y + 100 rh - beta) on
stance frames (lowest joint is a foot joint) vs ground frames (lowest surface region not a foot)."""
import sys, json, importlib.util, numpy as np, torch
sys.path.insert(0, '/home/arenalabs/Desktop/"be water, robot"/alpha_motion2/research/alphamotion_scaling_20261004/am2exp_v1')
from assets import surface_v2 as S
R='/home/arenalabs/Desktop/"be water, robot"/alpha_motion2/research/alphamotion_scaling_20261004'
spec=importlib.util.spec_from_file_location('tr',R+'/data/trajectory_reader.py'); tr=importlib.util.module_from_spec(spec); spec.loader.exec_module(tr)
bodies = sys.argv[1:]
clips=['fallAndGetUp1_subject4','ground1_subject5','ground2_subject3','walk1_subject1']
out={}
for b in bodies:
    A=S.load_surface(b); reg=A.region.numpy(); RN=S.REGIONS
    foot=np.isin(reg,[RN.index('L_foot'),RN.index('R_foot')])
    tm = A.extra.get('point_data_residual_p95_cm'); trust = (tm<=3) if tm is not None else np.ones(len(reg),bool)
    st=[]; gr=[]; grreg={}
    for c in clips:
        d=tr.load_xwbc_trajectory(R+'/data/data_manifest_v2.json','lafan::'+c,b,with_contact=True)
        idx=np.arange(0,len(d['position_native']),4)
        Rm=torch.as_tensor(d['global_rotation_matrices'][idx].astype(np.float32)); P=torch.as_tensor(d['position_native'][idx].astype(np.float32))
        rh=np.asarray(d['root_height_m'])[idx]; ok=np.asarray(d['root_height_valid'])[idx]&np.isfinite(rh)
        with torch.no_grad(): p,_=S.pose_surface(A,Rm,P)
        h=p.numpy()[...,1]+100*rh[:,None]-A.beta_vis_cm
        h[:,~trust]=np.inf
        lo=h.argmin(1); m=h.min(1)
        isfoot=foot[lo]
        st+=list(m[ok&isfoot]); gr+=list(m[ok&~isfoot])
        for r in reg[lo][ok&~isfoot]: grreg[RN[r]]=grreg.get(RN[r],0)+1
    st=np.array(st); gr=np.array(gr)
    out[b]=dict(beta=float(A.beta_vis_cm), stance_n=len(st), stance_min_p50=float(np.median(st)) if len(st) else None,
                ground_n=len(gr), ground_min_p10_p50_p90=[float(x) for x in np.percentile(gr,[10,50,90])] if len(gr) else None,
                ground_lowest_regions=dict(sorted(grreg.items(),key=lambda x:-x[1])[:5]))
    print(b, json.dumps(out[b]), flush=True)
json.dump(out, open('beta_offstance_'+bodies[0]+'.json','w'), indent=1)
