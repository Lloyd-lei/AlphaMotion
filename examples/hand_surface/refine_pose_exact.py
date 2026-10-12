from pathlib import Path
import os
os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2'
import numpy as np,mujoco,trimesh,json,time
from scipy.optimize import least_squares
R=Path('/home/arenalabs/Desktop/"be water, robot"/alpha_motion2/research/hand_surface_case_20261004')
m=mujoco.MjModel.from_xml_path(str(R/'source_assets/wonik_allegro/right_hand.xml'));d=mujoco.MjData(m);pose=json.loads((R/'constructed_pose.json').read_text());old=dict(pose)
geoms=[]
for g in range(m.ngeom):
 if m.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH and m.geom_contype[g]==0:
  k=int(m.geom_dataid[g]);v=m.mesh_vert[m.mesh_vertadr[k]:m.mesh_vertadr[k]+m.mesh_vertnum[k]].astype(float);f=m.mesh_face[m.mesh_faceadr[k]:m.mesh_faceadr[k]+m.mesh_facenum[k]].copy();geoms.append((g,m.body(int(m.geom_bodyid[g])).name,trimesh.Trimesh(v,f,process=False)))
x0=np.r_[pose['qpos'],pose['center']];lo=np.r_[m.jnt_range[:,0],[.005,-.02,.032]];hi=np.r_[m.jnt_range[:,1],[.085,.02,.095]]
def residual(x,ret=False):
 d.qpos[:]=x[:16];mujoco.mj_forward(m,d);gaps=[];tips=[]
 for g,n,mesh in geoms:
  pt=(x[16:]-d.geom_xpos[g])@d.geom_xmat[g].reshape(3,3);_,dist,_=trimesh.proximity.closest_point_naive(mesh,pt[None]);gap=float(dist[0]-.03);gaps.append(gap)
  if n.endswith('_tip'):tips.append(gap)
 if ret:return gaps,tips
 return np.r_[(np.array(tips)-.0008)*3,np.minimum(np.array(gaps)-.00035,0)*8,(x[:16]-x0[:16])*.00007]
t=time.time();sol=least_squares(residual,x0,bounds=(lo,hi),max_nfev=100,ftol=1e-8,xtol=1e-9,gtol=1e-8,diff_step=1e-5)
gap,tips=residual(sol.x,True);pose.update(qpos=sol.x[:16].tolist(),center=sol.x[16:].tolist(),tip_minimum_gap_m=dict(zip([n for _,n,_ in geoms if n.endswith('_tip')],tips)),minimum_exact_surface_gap_m=min(gap),solver_evaluations=sol.nfev,seconds=time.time()-t,exact_refinement=True,construction_method='bounded least squares using exact closest point over every visual-mesh triangle; no forces/dynamics/neural model',objective=float(np.sum(sol.fun**2)))
(R/'constructed_pose.json').write_text(json.dumps(pose,indent=2)+'\n');(R/'construction_initial_coarse_pose.json').write_text(json.dumps(old,indent=2)+'\n');print(json.dumps(pose,indent=2))
