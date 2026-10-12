from surface_retargeting.config import CFG as SRCFG  # machine paths: configs/sr.toml
import numpy as np, mujoco, json
from scipy.optimize import least_squares
xml='assets/fourier_gr3v2_1_1/mjcf/gr3v2_1_1_dummy_hand.xml'
m=mujoco.MjModel.from_xml_path(xml); d=mujoco.MjData(m)
def jadr(n): return m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n)]
def bpos(n): return d.xpos[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,n)]
def bmat(n): return d.xmat[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,n)].reshape(3,3)
out={}
for side,sgn in (('left',1),('right',-1)):
    J=[f'{side}_shoulder_pitch_joint',f'{side}_shoulder_roll_joint',f'{side}_shoulder_yaw_joint',f'{side}_elbow_pitch_joint',f'{side}_wrist_yaw_joint',f'{side}_wrist_pitch_joint',f'{side}_wrist_roll_joint']
    lo=[m.jnt_range[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,j)][0] for j in J]
    hi=[m.jnt_range[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,j)][1] for j in J]
    def res(x):
        mujoco.mj_resetData(m,d)
        for j,v in zip(J,x): d.qpos[jadr(j)]=v
        mujoco.mj_forward(m,d)
        s=bpos(f'{side}_upper_arm_pitch_link'); e=bpos(f'{side}_lower_arm_pitch_link'); w=bpos(f'{side}_hand_pitch_link'); h=bpos(f'{side}_end_effector_link')
        t=np.array([0,sgn,0.])
        u=(e-s)/np.linalg.norm(e-s); f=(w-e)/np.linalg.norm(w-e); g=(h-w)/np.linalg.norm(h-w+1e-9)
        ax=bmat(f'{side}_lower_arm_pitch_link')[:,1]
        return np.concatenate([u-t, f-t, 0.5*(g-t), [0.5*(1-abs(ax[2]))], 0.05*np.asarray(x)])
    x0=np.zeros(len(J)); x0[1]=sgn*1.5
    best=None
    for seed in range(40):
        xi=np.random.default_rng(seed).uniform(lo,hi) if seed else np.clip(x0,lo,hi)
        r=least_squares(res,xi,bounds=(lo,hi))
        if best is None or r.cost<best.cost: best=r
    r=best
    print(side, np.round(r.x,4), 'resid', np.round(np.linalg.norm(r.fun[:10]),4))
    out.update({j:float(v) for j,v in zip(J,r.x)})
json.dump(out,open(f'{SRCFG.umr_tools}/gr3_tpose.json','w'),indent=1)
