from surface_retargeting.config import CFG as SRCFG  # machine paths: configs/sr.toml
from pathlib import Path
import os
os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2'
import numpy as np,mujoco,json,time
from scipy.optimize import least_squares
R=Path(f'{SRCFG.research_root}/alpha_motion2/research/hand_surface_case_20261004')
m=mujoco.MjModel.from_xml_path(str(R/'source_assets/wonik_allegro/right_hand.xml'));d=mujoco.MjData(m)
geoms=[]
for g in range(m.ngeom):
 if m.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH and m.geom_contype[g]==0:
  k=int(m.geom_dataid[g]);v=m.mesh_vert[m.mesh_vertadr[k]:m.mesh_vertadr[k]+m.mesh_vertnum[k]].copy();f=m.mesh_face[m.mesh_faceadr[k]:m.mesh_faceadr[k]+m.mesh_facenum[k]].copy();tri=v[f];points=np.concatenate([v,tri.mean(1)])
  name=m.body(int(m.geom_bodyid[g])).name
  geoms.append((g,name,points[::max(1,len(points)//120)]))
q0=np.array([0,.7,.7,.3]*3+[1.1,.8,.5,.4]);x0=np.r_[q0,[.05,0,.06]]
lo=np.r_[m.jnt_range[:,0],[.005,-.02,.032]];hi=np.r_[m.jnt_range[:,1],[.085,.02,.095]]
x0=np.clip(x0,lo+1e-5,hi-1e-5)
def residual(x,ret=False):
 d.qpos[:]=x[:16];mujoco.mj_forward(m,d);c=x[16:];gaps=[];mins=[];names=[]
 for g,n,p in geoms:
  w=p@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g];gap=np.linalg.norm(w-c,axis=1)-.03;gaps.append(gap);names.append(n)
  if n.endswith('_tip'):mins.append(gap.min())
 allgap=np.concatenate(gaps)
 if ret:return dict(center=c.tolist(),radius_m=.03,qpos=x[:16].tolist(),tip_minimum_gap_m=dict(zip([n for n in names if n.endswith('_tip')],mins)),sampled_mesh_minimum_gap_m=float(allgap.min()))
 return np.r_[(np.array(mins)-.0006)*3,np.minimum(allgap-.00015,0)*4,(x[:16]-q0)*.00015]
t=time.time();sol=least_squares(residual,x0,bounds=(lo,hi),max_nfev=170,ftol=1e-8,xtol=1e-9,gtol=1e-8,diff_step=1e-5)
out=residual(sol.x,True);out.update(pose_status='constructed geometric illustration; static bounded kinematic fit, no UMR/model/grasp simulation',solver_evaluations=sol.nfev,seconds=time.time()-t,objective=float(np.sum(sol.fun**2)),joint_names=[m.joint(j).name for j in range(m.njnt)])
(R/'constructed_pose.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
