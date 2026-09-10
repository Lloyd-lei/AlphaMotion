import json, numpy as np, torch
from scipy.spatial.transform import Rotation as R
CF="/var/tmp/alphamotion/gmr_upstream/general_motion_retargeting/ik_configs"
L=json.load(open(f"{CF}/bvh_lafan1_to_g1.json"))["ik_match_table1"]; A=json.load(open(f"{CF}/smplx_to_g1.json"))["ik_match_table1"]
AX=np.array([[0.,1,0],[0,0,1],[1,0,0]])
# lafan bvh name -> our smpl name ; smplx name -> our smpl name
lmap={"Hips":"Pelvis","LeftUpLeg":"L_Hip","RightUpLeg":"R_Hip","LeftLeg":"L_Knee","RightLeg":"R_Knee","LeftFootMod":"L_Ankle","RightFootMod":"R_Ankle","Spine2":"Chest","LeftArm":"L_Shoulder","RightArm":"R_Shoulder","LeftForeArm":"L_Elbow","RightForeArm":"R_Elbow","LeftHand":"L_Wrist","RightHand":"R_Wrist"}
amap={"pelvis":"Pelvis","left_hip":"L_Hip","right_hip":"R_Hip","left_knee":"L_Knee","right_knee":"R_Knee","left_foot":"L_Toe","right_foot":"R_Toe","spine3":"Chest","left_shoulder":"L_Shoulder","right_shoulder":"R_Shoulder","left_elbow":"L_Elbow","right_elbow":"R_Elbow","left_wrist":"L_Wrist","right_wrist":"R_Wrist"}
OL={lmap[v[0]]:(k,R.from_quat(v[4],scalar_first=True).as_matrix()) for k,v in L.items()}
OA={amap[v[0]]:(k,R.from_quat(v[4],scalar_first=True).as_matrix()) for k,v in A.items()}
fit=torch.load("/home/arenalabs/tmp/dialect/Q_unitree_g1_29dof.pt"); names=fit["joint_names"]; Qf=fit["Q"].numpy()
def geo(a,b): return np.degrees(np.arccos(np.clip((np.trace(a.T@b)-1)/2,-1,1)))
Qc={}
for j in OL:
    if j not in OA: continue
    kl,ol=OL[j]; ka,oa=OA[j]
    q=AX@(oa@ol.T)@AX.T
    Qc[j]=q
    print(f"{j:11s} link L={kl:22s} A={ka:22s} same={kl==ka}  |Qcfg-Qfit| = {geo(q,Qf[names.index(j)]):5.1f}°")
np.save("/home/arenalabs/tmp/dialect/Qcfg.npy",{k:v for k,v in Qc.items()},allow_pickle=True)
