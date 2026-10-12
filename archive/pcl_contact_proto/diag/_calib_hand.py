import sys; sys.path.insert(0,'.')
import numpy as np, torch, json
from pcl_contact import sources as SR
S=SR.S
a=S.load_surface('human_smpl')
d=SR.load_traj('lafan::walk1_subject5','human_smpl')
names=[str(x) for x in d['joint_names']]
pos=torch.tensor(d['position_native'],dtype=torch.float32)
u,v,w,L=S.bone_frames(a,pos)
out={}
for s in 'LR':
    j=names.index(s+'_Wrist'); Rw=d['global_rotation_matrices'][:,j]
    ul=d['rest_offsets'][j]/np.linalg.norm(d['rest_offsets'][j])
    vl=np.einsum('tji,tj->ti',Rw,v[:,j].numpy())   # R^T v
    vl=vl-(vl@ul)[:,None]*ul; vl/=np.linalg.norm(vl,axis=1,keepdims=True)
    m=vl.mean(0); spread=np.degrees(np.arccos(np.clip(vl@ (m/np.linalg.norm(m)),-1,1)))
    m=m-(m@ul)*ul; m/=np.linalg.norm(m)
    out[s]=dict(u_local=ul.tolist(),v_local=m.tolist(),spread_deg_p50_p95=np.percentile(spread,[50,95]).round(1).tolist(),n=len(vl))
    print(s,out[s])
json.dump(out,open('pcl_contact/hand_frame_calib.json','w'),indent=1)
