import sys, json, importlib.util, numpy as np, torch
sys.path.insert(0, '/home/arenalabs/Desktop/"be water, robot"/alpha_motion2/research/alphamotion_scaling_20261004/am2exp_v1')
from assets import surface_v2 as S
R='/home/arenalabs/Desktop/"be water, robot"/alpha_motion2/research/alphamotion_scaling_20261004'
spec=importlib.util.spec_from_file_location('tr',R+'/data/trajectory_reader.py'); tr=importlib.util.module_from_spec(spec); spec.loader.exec_module(tr)
A=S.load_surface('human_smpl'); reg=A.region.numpy(); RN=S.REGIONS; bp=A.bone_parent.numpy()
NAMES=['Pelvis','L_Hip','R_Hip','Torso','L_Knee','R_Knee','Spine','L_Ankle','R_Ankle','Chest','L_Toe','R_Toe','Neck','L_Thorax','R_Thorax','Head','L_Shoulder','R_Shoulder','L_Elbow','R_Elbow','L_Wrist','R_Wrist']
# rest geometry
rp=A.rest_points.numpy(); cj=A.extra['calib_joint_positions']
for r in ['pelvis','L_thigh','trunk','L_hand']:
    m=reg==RN.index(r); bones=np.unique(bp[m])
    print('rest',r,'bones',[NAMES[b] for b in bones],'ymin-pelvisjoint',round(float(rp[m,1].min()-cj[0,1]),1),'radial max',round(float(A.radial.numpy()[m].max()),1),'t range',np.round([A.t.numpy()[m].min(),A.t.numpy()[m].max()],2))
print('pelvis joint y',cj[0],'hips',cj[1],cj[2],'torso',cj[3])
for clip in ['ground1_subject5','fallAndGetUp1_subject4']:
    d=tr.load_xwbc_trajectory(R+'/data/data_manifest_v2.json','lafan::'+clip,'human_smpl',with_contact=True)
    P=torch.as_tensor(d['position_native'].astype(np.float32))
    with torch.no_grad():
        p,_=S.pose_surface(A,None,P); u,v,w,L=S.bone_frames(A,P)
    p=p.numpy(); Pn=P.numpy()
    dp=np.linalg.norm(p[1:]-p[:-1],axis=-1); dj=np.linalg.norm(Pn[1:]-Pn[:-1],axis=-1)
    t=int(np.argmax(dp.max(1)/np.maximum(dj.max(1),0.5))); i=int(dp[t].argmax())
    print(clip,'worst frame',t,'point',i,'region',RN[reg[i]],'bone',NAMES[bp[i]],'dp',round(float(dp[t,i]),1),'max joint move',round(float(dj[t].max()),1),'joint',NAMES[int(dj[t].argmax())])
    b=bp[i]
    print('   u change deg',np.degrees(np.arccos(np.clip((u[t+1,b]*u[t,b]).sum().item(),-1,1))),'v change deg',np.degrees(np.arccos(np.clip((v[t+1,b]*v[t,b]).sum().item(),-1,1))))
    # pelvis worst sink frame
    rh=d['root_height_m']; h=p[...,1]+100*rh[:,None]-A.beta_vis_cm
    pm=h[:,reg==0].min(1); f=int(pm.argmin()); k=np.flatnonzero(reg==0)[h[f,reg==0].argmin()]
    print('   pelvis worst frame',f,'pelvis min',round(float(pm[f]),1),'point bone',NAMES[bp[k]],'t',round(float(A.t[k]),2),'radial',round(float(A.radial[k]),1),
          'joint heights', {NAMES[j]:round(float(Pn[f,j,1]+100*rh[f]-A.beta_vis_cm),1) for j in (0,1,2,3,6,4,5)})
    np.save(f'h2_{clip}_frames.npy', np.array([t,f]))
