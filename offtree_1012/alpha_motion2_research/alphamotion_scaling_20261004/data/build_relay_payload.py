#!/usr/bin/env python3
"""Exact recorded MJL arm FK; archive read member-by-member, no simulator stepping."""
from pathlib import Path
import json,hashlib,struct,zipfile,copy,xml.etree.ElementTree as ET
import numpy as np
from mujoco_dataset_adapter import MujocoMotionAdapter,rot6,fk_global,group_split,sha
ROOT=Path('/home/arenalabs/Desktop')/'"be water, robot"'/'alpha_motion2/research/alphamotion_scaling_20261004/data'
def prepare_model(out):
 src=out/'models/relay-policy-learning';a=src/'third_party/franka/assets/assets.xml';chain=src/'third_party/franka/assets/chain0.xml';scene=src/'adept_envs/adept_envs/franka/assets/franka_kitchen_jntpos_act_ab.xml'
 s=ET.parse(scene).getroot();robot=next(b for b in s.find('worldbody').findall('body') if any('chain0.xml' in i.get('file','') for i in b.findall('include')))
 tree=ET.Element('mujoco',model='relay_recorded_panda_robot_only_derived')
 ET.SubElement(tree,'compiler',angle='radian',meshdir=str(src/'adept_models/kitchen'),inertiafromgeom='auto',inertiagrouprange='3 5')
 for el in ET.parse(a).getroot():
  if el.tag!='compiler':tree.append(copy.deepcopy(el))
 w=ET.SubElement(tree,'worldbody');b=ET.SubElement(w,'body',attrib=dict(robot.attrib));c=copy.deepcopy(ET.parse(chain).getroot().find('body'));b.append(c)
 site_parent=next(el for el in c.iter('body') if any(st.get('name')=='end_effector' for st in el.findall('site')))
 site=next(st for st in site_parent.findall('site') if st.get('name')=='end_effector')
 ET.SubElement(site_parent,'body',name='end_effector_site',**{k:v for k,v in site.attrib.items() if k in ['pos','quat','euler','axisangle','xyaxes','zaxis']})
 dst=out/'models/relay_robot_only_derived.xml';ET.ElementTree(tree).write(dst,encoding='unicode')
 evidence={'derived_xml':str(dst),'derived_xml_sha256':sha(dst),'source_files':[{'path':str(p),'sha256':sha(p)} for p in [a,chain,scene]],'derivation':['Original Franka assets/defaults and chain copied without changing joint axes/origins/ranges.','Original enclosing robot base pos/euler copied from source kitchen scene.','Objects, controllers, actuators and environmental scene omitted for independent robot FK.','Actual named end_effector site duplicated as fixed skeleton body end_effector_site with identical local pos/euler.'],'original_scene_current_mujoco_compile_error':"top-level default class 'main' cannot be renamed",'dynamics_equivalence_claim':False}
 (out/'relay_model_derivation.json').write_text(json.dumps(evidence,indent=2)+'\n');return dst

def parse_mjl(data):
 h=struct.unpack('<7i',data[:28]);nq,nv,nu,nmocap,nsens,nuser,nname=h;name=data[28:28+nname].decode();width=1+nq+nv+nu+7*nmocap+nsens+nuser
 floats=np.frombuffer(data[28+nname:],dtype='<f4');assert len(floats)%width==0
 x=floats.reshape(-1,width);assert np.isfinite(x).all();assert np.all(np.diff(x[:,0])>0)
 return {'model_name':name,'header':h,'time':x[:,0].astype('float64'),'qpos':x[:,1:1+nq].copy(),'qvel':x[:,1+nq:1+nq+nv].copy(),'ctrl':x[:,1+nq+nv:1+nq+nv+nu].copy(),'mocap':x[:,1+nq+nv+nu:1+nq+nv+nu+7*nmocap].copy()}

def main(out=ROOT):
 xml=prepare_model(out);a=MujocoMotionAdapter(xml,'panda0_link0',excluded_bodies=['panda0_leftfinger','panda0_rightfinger']);assert len(a.names)==9 and a.model.nq==9
 assert len(a.mechanics)==7 and all(r['type']=='hinge' for r in a.mechanics)
 archive=out/'relay_policy_learning/kitchen_demos_multitask.zip';z=zipfile.ZipFile(archive);members=sorted(n for n in z.namelist() if n.endswith('.mjl') and 'microwave_kettle_switch_slide' in n);assert len(members)==19
 eps=[]
 for n in members:
  b=z.read(n);d=parse_mjl(b);assert d['header'][:4]==(30,29,9,1) and d['model_name']=='franka_mocap_studyTable_buttons'
  # Every native saved frame retained. File/content identity is the whole-episode grouping unit.
  group='relay_mjl:sha256:'+hashlib.sha256(b).hexdigest();eps.append((n,group,d,hashlib.sha256(b).hexdigest()))
 splits=group_split([e[1] for e in eps]);groups=sorted(splits);G=[];P=[];Q=[];FULL=[];QV=[];CTRL=[];MOCAP=[];W=[];SP=[];GI=[];FR=[];TS=[];records=[];row=0
 for n,g,d,h in eps:
  q=d['qpos'][:,:9];dec=a.decode_q(q);err=np.linalg.norm(fk_global(dec['global_matrices'],a.offsets,a.parents)-dec['position_native'],axis=-1);assert err.max()<1e-9
  F=len(q);ts=d['time'];split=splits[g];records.append({'member':n,'source_group':g,'source_sha256':h,'split':split,'row_range':[row,row+F],'frames':F,'native_frame_stride':1,'timestamp_range_seconds':[float(ts[0]),float(ts[-1])],'dt_quantiles_seconds':np.quantile(np.diff(ts),[0,.5,1]).tolist(),'fk_max_error_m':float(err.max()),'qpos_range':np.ptp(q,axis=0).tolist(),'first_robot_qpos':q[0].tolist(),'last_robot_qpos':q[-1].tolist()});row+=F
  G.append(rot6(dec['global_matrices']).astype('float32'));P.append(dec['position_native'].astype('float32'));Q.append(q);FULL.append(d['qpos']);QV.append(d['qvel']);CTRL.append(d['ctrl']);MOCAP.append(d['mocap']);W.append(dec['world_positions_m_zup'][:,0]);SP.extend([split]*F);GI.extend([groups.index(g)]*F);FR.extend(range(F));TS.extend(ts-ts[0]);print(json.dumps({'episode':n,'frames':F,'fk_error_m':float(err.max())}),flush=True)
 G=np.concatenate(G);P=np.concatenate(P);N=len(P);J=len(a.names)
 data={'global_rot6':G,'position_native':P,'position_canonical':P,'pose9':np.concatenate([G,P/a.reach],axis=-1),'parents':a.parents,'rest_offsets':a.offsets.astype('float32'),'rest_offsets_descriptor':(100*a.offsets).astype('float32'),'joint_names':np.array(a.names,dtype='U'),'reach':np.array(a.reach,np.float32),'reach_descriptor':np.array(100*a.reach,np.float32),'dof_features':a.dof_features,'dof_valid':a.dof_valid,'ee_indices':np.array([a.names.index('end_effector_site')]),'split':np.array(SP,np.int8),'source_group_index':np.array(GI,np.int64),'frame_index':np.array(FR,np.int64),'timestamp_seconds':np.array(TS),'fps':np.full(N,500,np.float32),'metric_length_valid':np.array(True),'ground_valid':np.array(False),'root_height_valid':np.array(False),'contact_valid':np.zeros(J,bool),'position_valid':np.ones((N,J),bool),'rotation_valid':np.ones((N,J),bool),'recorded_robot_qpos':np.concatenate(Q),'recorded_scene_qpos':np.concatenate(FULL),'recorded_scene_qvel':np.concatenate(QV),'recorded_ctrl':np.concatenate(CTRL),'recorded_mocap':np.concatenate(MOCAP),'world_root_m_zup':np.concatenate(W)}
 p=out/'payloads/relay_franka_arm.npz';np.savez_compressed(p,**data)
 excluded=[]
 import mujoco
 for k in range(a.model.njnt):
  if a.model.jnt_type[k]==mujoco.mjtJoint.mjJNT_SLIDE:
   excluded.append({'joint_name':mujoco.mj_id2name(a.model,mujoco.mjtObj.mjOBJ_JOINT,k),'body_name':mujoco.mj_id2name(a.model,mujoco.mjtObj.mjOBJ_BODY,int(a.model.jnt_bodyid[k])),'type':'slide','qpos_address':int(a.model.jnt_qposadr[k]),'axis_local_body':a.model.jnt_axis[k].tolist(),'origin_local_body_m':a.model.jnt_pos[k].tolist(),'range_m':a.model.jnt_range[k].tolist(),'retained_in_recorded_robot_qpos':True,'excluded_from_rotation_only_training_skeleton_and_surface_loss':True})
 meta={'schema':'recorded_mujoco_motion_v1','body_id':'relay_franka_arm','source_dataset':'google-research/relay-policy-learning kitchen demos complete-task original MJL','source_kind':'recorded_simulated_human_VR_demonstrations','task_family':'unpaired_same_body_reconstruction_and_sparse_completion','paired_cross_body_mapping_gt':False,'archive':str(archive),'archive_sha256':sha(archive),'revision':'cd70ac9334f584f86db281a1ffd3e5cbc3e5e293','source_license':'Apache-2.0','model':a.metadata(),'frames':N,'independent_groups':len(groups),'episodes':records,'frame_counts_by_split':{str(i):int(np.sum(data['split']==i)) for i in [0,1,2]},'group_counts_by_split':{str(i):sum(v==i for v in splits.values()) for i in [0,1,2]},'payload_path':str(p),'payload_sha256':sha(p),'payload_shapes':{k:list(v.shape) for k,v in data.items()},'unsupported_prismatic_joints':excluded,'scope':'Seven exact robot hinges plus one fixed source-defined end_effector site; full gripper capability NOT supported by rotation-only skeleton. Two actual finger slides and full scene qpos/qvel retained separately, not used as fake fixed-offset supervision.','mimic_constraint':'No mimic equality introduced; actual separate two recorded finger slides retained.','time_source':'All native MJL saved frames retained at nominal500Hz; exact float32 recorded time converted float64, timestamps preserve actual quantization. Evaluate using timestamps, not assumed dt.','observation_quality':'Minari Franka observations have noise and are explicitly not used for exact qpos training/evaluation.','validity':{'metric':True,'ground':False,'contact':False,'root_height_supervision':False,'physical_robot_execution':False,'scene_qpos':True,'scene_fk_semantics_verified':False},'split_seed':20261004,'old_checkpoint_exposure':'main scaling from scratch; no old checkpoint initialization'}
 (p.with_suffix('.json')).write_text(json.dumps(meta,indent=2)+'\n')
 namesfile=out/'all_required_qwen_names.json';names=json.loads(namesfile.read_text());names['bodies']['relay_franka_arm']=a.names;names['deduplicated_joint_names']=sorted({n for ns in names['bodies'].values() for n in ns});namesfile.write_text(json.dumps(names,indent=2)+'\n')
 print(json.dumps({'complete':True,'frames':N,'groups':len(groups),'payload':str(p),'splits':meta['frame_counts_by_split'],'fk_max_error_m':max(x['fk_max_error_m'] for x in records)}))
if __name__=='__main__':main()
