#!/usr/bin/env python3
"""CPU geometry from recorded qpos. Never steps simulator or invents motion.
Model poses are own-link global rotations relative to fixed qpos0 rest frames.
Z-up metres -> Y-up metres by AX; pose9 position = root-relative metres / static reach.
"""
from pathlib import Path
import json,hashlib,argparse
import numpy as np
try:
 import mujoco
except ImportError:
 mujoco=None
AX=np.array([[0.,1.,0.],[0.,0.,1.],[1.,0.,0.]])
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rot6(R):return np.concatenate((R[..., :,0],R[..., :,1]),axis=-1)
def fk_global(R,offsets,parents):
 p=np.zeros(R.shape[:-2]+(3,),dtype=R.dtype)
 for j,pa in enumerate(parents):
  if pa>=0:p[...,j,:]=p[...,pa,:]+np.einsum('...ab,b->...a',R[...,pa,:,:],offsets[j])
 return p
def static_reach(parents,offsets):
 d=np.zeros(len(parents))
 for j,p in enumerate(parents):d[j]=0 if p<0 else d[p]+np.linalg.norm(offsets[j])
 if d.max()<=0:raise ValueError('degenerate rest geometry')
 return float(d.max())
def group_split(keys,seed=20261004):
 ks=sorted(set(keys),key=lambda x:hashlib.sha256(f'{seed}|{x}'.encode()).hexdigest())
 if len(ks)<2:return {k:2 for k in ks}
 n=max(1,round(.2*len(ks)));ev=set(ks[:n]);return {k:int(k in ev) for k in ks}

class MujocoMotionAdapter:
 """Exact saved-qpos decoder. `project_free` is intentionally not fabricated here."""
 def __init__(self,xml,root_body,excluded_bodies=()):
  self.xml=str(xml);self.model=mujoco.MjModel.from_xml_path(self.xml);m=self.model
  root=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,root_body)
  if root<1:raise ValueError('named robot root missing')
  self.body_ids=[]
  for b in range(root,m.nbody):
   p=b
   while p>0 and p!=root:p=int(m.body_parentid[p])
   name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,b)
   if p==root and name not in excluded_bodies:self.body_ids.append(b)
  ids=self.body_ids;lookup={b:i for i,b in enumerate(ids)}
  self.parents=np.array([-1 if b==root else lookup[int(m.body_parentid[b])] for b in ids],np.int64)
  self.names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,b) for b in ids]
  self.data=mujoco.MjData(m);self.data.qpos[:]=m.qpos0;mujoco.mj_forward(m,self.data)
  self.rest_R=self.data.xmat[ids].reshape(-1,3,3).copy();self.rest_P=self.data.xpos[ids].copy()
  off=np.zeros((len(ids),3))
  for j,p in enumerate(self.parents):
   if p>=0:off[j]=AX@(self.rest_P[j]-self.rest_P[p])
  self.offsets=off;self.reach=static_reach(self.parents,off)
  self.mechanics=[];dof=np.zeros((len(ids),19),np.float32);valid=np.ones(len(ids),bool)
  for j,b in enumerate(ids):
   js=[k for k in range(m.njnt) if int(m.jnt_bodyid[k])==b];hinges=[]
   for k in js:
    tp=int(m.jnt_type[k]);axis=AX@self.rest_R[j]@m.jnt_axis[k]
    r={'skeleton_index':j,'body_name':self.names[j],'joint_id':k,'joint_name':mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,k),'type':{3:'hinge',2:'slide',1:'ball',0:'free'}[tp],'qpos_address':int(m.jnt_qposadr[k]),'dof_address':int(m.jnt_dofadr[k]),'axis_local_body':m.jnt_axis[k].tolist(),'axis_rest_canonical':axis.tolist(),'origin_local_body_m':m.jnt_pos[k].tolist(),'limited':bool(m.jnt_limited[k]),'range':m.jnt_range[k].tolist(),'model_qpos0':float(m.qpos0[m.jnt_qposadr[k]])}
    self.mechanics.append(r)
    if tp==3:hinges.append(r)
    else:valid[j]=False
   if len(hinges)>3:valid[j]=False
   if valid[j]:
    for slot,r in enumerate(hinges):
     lo,hi=r['range'];dof[j,slot*6:slot*6+6]=[*r['axis_rest_canonical'],float(r['limited']),(hi-lo)/(2*np.pi) if r['limited'] else 0,(hi+lo)/(2*np.pi) if r['limited'] else 0]
    dof[j,18]=len(hinges)/3
  self.dof_features=dof;self.dof_valid=valid
 def decode_q(self,qpos):
  qpos=np.asarray(qpos,np.float64)
  if qpos.ndim==1:qpos=qpos[None]
  if qpos.shape[1]!=self.model.nq:raise ValueError('full model qpos required, with explicit unobserved components recorded in metadata')
  rs=[];ps=[];wr=[];wp=[]
  for q in qpos:
   self.data.qpos[:]=q;mujoco.mj_forward(self.model,self.data)
   R=self.data.xmat[self.body_ids].reshape(-1,3,3).copy();P=self.data.xpos[self.body_ids].copy()
   rs.append(AX@(R@self.rest_R.transpose(0,2,1))@AX.T);ps.append((P-P[0])@AX.T);wr.append(R);wp.append(P)
  return {'global_matrices':np.asarray(rs),'position_native':np.asarray(ps),'world_rotation_matrices_zup':np.asarray(wr),'world_positions_m_zup':np.asarray(wp)}
 def metadata(self):
  return {'model_xml':self.xml,'model_xml_sha256':sha(self.xml),'model_nq':self.model.nq,'body_ids':self.body_ids,'joint_names':self.names,'parents':self.parents.tolist(),'rest_offsets_m':self.offsets.tolist(),'rest_body_rotations_world_zup':self.rest_R.tolist(),'rest_body_positions_world_zup_m':self.rest_P.tolist(),'reach_m':self.reach,'mechanical_joints':self.mechanics,'fixed_body_nodes':[{'skeleton_index':j,'body_id':b,'body_name':self.names[j],'parent_skeleton_index':int(self.parents[j]),'evidence':'MuJoCo body_jntnum==0; explicit fixed source-model transform','source_site':'end_effector' if self.names[j]=='end_effector_site' else None} for j,b in enumerate(self.body_ids) if int(self.model.body_jntnum[b])==0],'rotation_definition':'AX @ (R_world(q) @ R_world(qpos0).T) @ AX.T','AX_zup_to_yup':AX.tolist(),'length_unit':'m','descriptor_scale':100.0,'descriptor_length_unit':'cm','geometry_scale_policy':'static maximum root-to-joint path length; no motion fitting','direct_free_rotation_is_physically_executable':False}

def build_adroit(out):
 import h5py
 out=Path(out);xml=out/'models/Gymnasium-Robotics/gymnasium_robotics/envs/assets/adroit_hand/adroit_pen.xml';src=out/'minari/pen/human-v2/data/main_data.hdf5'
 a=MujocoMotionAdapter(xml,'forearm');assert len(a.names)==25
 m=a.model;assert m.nq==30 and all(x['type']=='hinge' for x in a.mechanics) and len(a.mechanics)==24
 dt=float(m.opt.timestep)*5;assert abs(dt-.01)<1e-10
 episodes=[]
 with h5py.File(src,'r') as f:
  for ep in sorted(f,key=lambda x:int(x.split('_')[-1])):
   q=f[ep]['observations'][:,:24];n=len(q);assert n==len(f[ep]['actions'])+1
   # Official observation includes exact robot qpos; object state is deliberately excluded.
   assert np.allclose(q[0],f[ep].attrs['options/initial_state_dict/qpos'][:24],atol=1e-6)
   h=hashlib.sha256(np.ascontiguousarray(q).tobytes()).hexdigest()
   episodes.append({'name':ep,'q':q,'group':'adroit_pen:qpos_sha256:'+h,'steps':n-1,'source_numeric_sha256':h})
 splits=group_split([e['group'] for e in episodes]);groups=sorted(splits);allG=[];allP=[];allQ=[];allW=[];records=[];sp=[];gi=[];fr=[];times=[];row=0
 for e in episodes:
  q=np.repeat(m.qpos0[None],len(e['q']),axis=0);q[:,:24]=e['q'];d=a.decode_q(q);R=d['global_matrices'];P=d['position_native'];err=np.linalg.norm(fk_global(R,a.offsets,a.parents)-P,axis=-1)
  assert err.max()<1e-9
  n=len(q);r={'episode':e['name'],'source_group':e['group'],'source_numeric_sha256':e['source_numeric_sha256'],'split':splits[e['group']],'row_range':[row,row+n],'frames':n,'steps':e['steps'],'fps':1/dt,'dt_seconds':dt,'fk_max_error_m':float(err.max()),'qpos_range_rad':np.ptp(e['q'],axis=0).tolist(),'first_qpos_rad':e['q'][0].tolist(),'last_qpos_rad':e['q'][-1].tolist()};records.append(r);row+=n
  allG.append(rot6(R).astype('float32'));allP.append(P.astype('float32'));allQ.append(e['q']);allW.append(d['world_positions_m_zup'][:,0]);sp.extend([r['split']]*n);gi.extend([groups.index(e['group'])]*n);fr.extend(range(n));times.extend(np.arange(n)*dt)
 G=np.concatenate(allG);P=np.concatenate(allP);N=len(P);J=len(a.names)
 payload={'global_rot6':G,'position_native':P,'position_canonical':P,'pose9':np.concatenate([G,P/a.reach],axis=-1),'parents':a.parents,'rest_offsets':a.offsets.astype('float32'),'rest_offsets_descriptor':(100*a.offsets).astype('float32'),'joint_names':np.array(a.names,dtype='U'),'reach':np.array(a.reach,np.float32),'reach_descriptor':np.array(a.reach*100,np.float32),'dof_features':a.dof_features,'dof_valid':a.dof_valid,'ee_indices':np.array([j for j in range(J) if j not in a.parents]),'split':np.array(sp,np.int8),'source_group_index':np.array(gi,np.int64),'frame_index':np.array(fr,np.int64),'timestamp_seconds':np.array(times),'fps':np.full(N,1/dt,np.float32),'metric_length_valid':np.array(True),'ground_valid':np.array(False),'root_height_valid':np.array(False),'contact_valid':np.zeros(J,bool),'position_valid':np.ones((N,J),bool),'rotation_valid':np.ones((N,J),bool),'recorded_robot_qpos':np.concatenate(allQ),'world_root_m_zup':np.concatenate(allW)}
 folder=out/'payloads';folder.mkdir(exist_ok=True);p=folder/'adroit_pen.npz';np.savez_compressed(p,**payload)
 meta={'schema':'recorded_mujoco_motion_v1','body_id':'adroit_pen','source_dataset':'D4RL/pen/human-v2','source_kind':'recorded_simulated_human_demonstrations','task_family':'unpaired_same_body_reconstruction_and_sparse_completion','paired_cross_body_mapping_gt':False,'source_hdf5':str(src),'source_sha256':sha(src),'source_license':'CC-BY-4.0','model':a.metadata(),'frames':N,'independent_groups':len(groups),'episodes':records,'frame_counts_by_split':{str(i):int(np.sum(payload['split']==i)) for i in [0,1,2]},'group_counts_by_split':{str(i):sum(v==i for v in splits.values()) for i in [0,1,2]},'payload_path':str(p),'payload_sha256':sha(p),'payload_shapes':{k:list(v.shape) for k,v in payload.items()},'observation_to_qpos':'observations[:,0:24] exactly qpos[:-6] in official AdroitHandPen._get_obs; per-episode initial state equality verified','unobserved_object_qpos':'six excluded object DOFs set model.qpos0 only for mj_forward; robot subtree FK independent of object; no object/contact truth emitted','time_source':'official frame_skip=5 times MJCF timestep=.002; nominal uniform recorded simulation control interval, no wallclock timestamps','validity':{'metric':True,'ground':False,'root_height_supervision':False,'contact':False,'object_pose':False,'physical_robot_execution':False},'split_seed':20261004,'all_five_digit_families_present':True,'actuated_hinge_count':24,'finger_hinge_count':22,'model_patch_binding':'rigid link transforms; surface points must use actual MuJoCo body frames and rest_R rebase','old_checkpoint_exposure':'main scaling from scratch; no old checkpoint initialization'}
 (folder/'adroit_pen.json').write_text(json.dumps(meta,indent=2)+'\n');(out/'all_required_qwen_names.json').write_text(json.dumps({'bodies':{'adroit_pen':a.names},'deduplicated_joint_names':sorted(set(a.names))},indent=2)+'\n')
 print(json.dumps({'body':'adroit_pen','frames':N,'episodes':len(episodes),'groups':len(groups),'payload':str(p),'manifest':str(folder/'adroit_pen.json'),'splits':meta['frame_counts_by_split'],'fk_max_error_m':max(e['fk_max_error_m'] for e in records)}))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args();build_adroit(args.out)
