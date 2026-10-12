"""Reviewer: human_smpl bone_revolution on real LAFAN clips: frame flips, degeneracy, stance soles, sinking."""
import sys, json, importlib.util, numpy as np, torch
sys.path.insert(0, '/home/arenalabs/Desktop/"be water, robot"/alpha_motion2/research/alphamotion_scaling_20261004/am2exp_v1')
from assets import surface_v2 as S
R='/home/arenalabs/Desktop/"be water, robot"/alpha_motion2/research/alphamotion_scaling_20261004'
spec=importlib.util.spec_from_file_location('tr',R+'/data/trajectory_reader.py'); tr=importlib.util.module_from_spec(spec); spec.loader.exec_module(tr)
A = S.load_surface('human_smpl'); beta = A.beta_vis_cm
reg = A.region.numpy(); RN = S.REGIONS
NAMES=['Pelvis','L_Hip','R_Hip','Torso','L_Knee','R_Knee','Spine','L_Ankle','R_Ankle','Chest','L_Toe','R_Toe','Neck','L_Thorax','R_Thorax','Head','L_Shoulder','R_Shoulder','L_Elbow','R_Elbow','L_Wrist','R_Wrist']
FOOTJ=[7,8,10,11]
clips = sys.argv[1:] or ['fallAndGetUp1_subject4','walk1_subject1','ground1_subject5','ground2_subject3','pushAndStumble1_subject2','multipleActions1_subject3','run1_subject2','jumps1_subject1','dance1_subject1','obstacles1_subject1','sprint1_subject2','fight1_subject2']
def ang(a,b):
    return np.degrees(np.arccos(np.clip((a*b).sum(-1),-1,1)))
out={}
for c in clips:
    try:
        d = tr.load_xwbc_trajectory(R+'/data/data_manifest_v2.json','lafan::'+c,'human_smpl',with_contact=True)
    except Exception as e:
        out[c]={'error':repr(e)}; print(c,'ERR',e); continue
    P = torch.as_tensor(d['position_native'].astype(np.float32)); T=len(P)
    rh = np.asarray(d['root_height_m'],np.float64); rv=np.asarray(d['root_height_valid'],bool)&np.isfinite(rh)
    ch = np.asarray(d['contact_hard'],bool); cv=np.asarray(d['contact_valid'],bool)
    with torch.no_grad():
        u,v,w,L = S.bone_frames(A, P)
        pts=[];nrm=[]
        for k in range(0,T,512):
            p,n=S.pose_surface(A,None,P[k:k+512]); pts.append(p.numpy()); nrm.append(n.numpy())
    pts=np.concatenate(pts); nrm=np.concatenate(nrm); u=u.numpy(); v=v.numpy()
    # frame continuity
    du = ang(u[1:],u[:-1]); dv = ang(v[1:],v[:-1])
    pop = (dv>30)&(du<10)
    pops = {NAMES[j]:int(pop[:,j].sum()) for j in range(22) if pop[:,j].sum()}
    # degeneracy of the primary reference for each bone: |perp_u(ref)| small
    Pn=P.numpy()
    ref_feet={}
    for j,kn in ((7,4),(8,5),(10,4),(11,5)):
        r = Pn[:,kn]-Pn[:,7 if j in (7,10) else 8]; r/=np.linalg.norm(r,axis=-1,keepdims=True)
        perp = np.linalg.norm(r-(r*u[:,j]).sum(-1,keepdims=True)*u[:,j],axis=-1)
        ref_feet[NAMES[j]]=dict(frac_perp_lt_0p26=float((perp<0.26).mean()), min=float(perp.min()))
    # point jumps
    dp = np.linalg.norm(pts[1:]-pts[:-1],axis=-1).max(1); dj=np.linalg.norm(Pn[1:]-Pn[:-1],axis=-1).max(1)
    jump_ratio = dp/np.maximum(dj,0.5)
    # heights (only rows with valid root height)
    ok = rv
    h = pts[...,1] + 100*rh[:,None] - beta
    hj = Pn[...,1] + 100*rh[:,None] - beta
    reg_min = {RN[r]: h[:, reg==r].min(1) for r in np.unique(reg)}
    res = dict(T=int(T), valid_rows=int(ok.sum()), v_pops=pops, v_pop_frames=int(pop.any(1).sum()),
               foot_frame_degenerate=ref_feet, point_jump_ratio_p99=float(np.percentile(jump_ratio,99)), point_jump_ratio_max=float(jump_ratio.max()),
               point_jump_max_cm=float(dp.max()), joint_jump_at_max_cm=float(dj[dp.argmax()]))
    # stance: valid & any foot contact & no non-foot contact
    footc = (ch[:,FOOTJ]&cv[:,FOOTJ]).any(1); nonfoot = (ch&cv)[:, [j for j in range(22) if j not in FOOTJ]].any(1)
    st = ok & footc & ~nonfoot
    if st.sum():
        lo = h[st].argmin(1)
        lr = reg[lo]; ln = nrm[st][np.arange(st.sum()), lo]
        footmin = np.minimum(reg_min['L_foot'], reg_min['R_foot'])[st]
        res['stance']=dict(n=int(st.sum()), lowest_is_foot=float(np.isin(lr,[RN.index('L_foot'),RN.index('R_foot')]).mean()),
                           lowest_normal_down=float((ln[:,1]<-0.7).mean()), foot_min_p10_p50_p90=[float(x) for x in np.percentile(footmin,[10,50,90])],
                           nonfoot_min_p5=float(np.percentile(np.min(np.stack([reg_min[r][st] for r in reg_min if 'foot' not in r]),0),5)))
    # sinking: per region frac of valid frames with min < -1 cm, and < -5 cm
    res['sink_frac_lt_m1']={r: round(float((reg_min[r][ok]< -1).mean()),3) for r in reg_min}
    res['sink_frac_lt_m5']={r: round(float((reg_min[r][ok]< -5).mean()),3) for r in reg_min}
    res['region_min_p1']={r: round(float(np.percentile(reg_min[r][ok],1)),1) if ok.any() else None for r in reg_min}
    # pelvis: pelvis surface min vs pelvis joint and hip joints
    pm = reg_min['pelvis'][ok]
    if ok.any():
        sel = pm < -1
        res['pelvis_sink']=dict(frac=float(sel.mean()), surf_min_med=float(np.median(pm[sel])) if sel.any() else None,
                                pelvis_joint_h_med=float(np.median(hj[ok][sel,0])) if sel.any() else None,
                                hip_joint_h_med=float(np.median(hj[ok][sel][:,[1,2]].min(1))) if sel.any() else None,
                                surf_minus_joint_med=float(np.median(pm[sel]-hj[ok][sel,0])) if sel.any() else None,
                                pelvis_contact_label_frac=float((ch[ok][sel,0]).mean()) if sel.any() else None)
        # wrists
        for side,j,rn in (('L',20,'L_hand'),('R',21,'R_hand')):
            res[f'{side}_wrist_joint_min']=float(hj[ok][:,j].min()); res[f'{side}_hand_surf_min']=float(reg_min[rn][ok].min())
    out[c]=res
    np.savez_compressed(f'/var/tmp/alphamotion/am2exp/review/human_{c}.npz', h_regmin=np.stack([reg_min[r] for r in reg_min]), regs=np.array(list(reg_min)), hj=hj, rh=rh, ok=ok, dv=dv, du=du)
    print(c, json.dumps(res), flush=True)
json.dump(out, open('/var/tmp/alphamotion/am2exp/review/human_review.json','w'), indent=1)
