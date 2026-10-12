#!/usr/bin/env python3
"""Versioned full arm+gripper truth from immutable recorded qpos, no new download."""
from pathlib import Path
import json,hashlib
import numpy as np
from mujoco_dataset_adapter import MujocoMotionAdapter,rot6,AX,sha
D=Path('/home/arenalabs/Desktop')/'"be water, robot"'/'alpha_motion2/research/alphamotion_scaling_20261004/data'
def mixed_fk(R,offsets,parents,axis,slide,reference):
 p=np.zeros(R.shape[:-2]+(3,),dtype=R.dtype)
 for j,pa in enumerate(parents):
  if pa>=0:
   local=offsets[j]+axis[j]*(slide[:,j]-reference[j])[:,None]
   p[:,j]=p[:,pa]+np.einsum('nab,nb->na',R[:,pa],local)
 return p

def main():
 old=D/'payloads/relay_franka_arm.npz';om=D/'payloads/relay_franka_arm.json';oldhash=sha(old);omhash=sha(om);meta=json.loads(om.read_text())
 a=MujocoMotionAdapter(meta['model']['model_xml'],'panda0_link0');J=len(a.names);assert J==11
 with np.load(old) as z:
  q=z['recorded_robot_qpos'];unchanged={k:z[k] for k in ['split','source_group_index','frame_index','timestamp_seconds','fps','recorded_robot_qpos','recorded_scene_qpos','recorded_scene_qvel','recorded_ctrl','recorded_mocap','world_root_m_zup']}
 N=len(q);typ=np.ones(J,np.int8);typ[0]=4;axis=np.zeros((J,3));lower=np.zeros(J);upper=np.zeros(J);ref=np.zeros(J);indices=np.full(J,-1,np.int64);slide=np.zeros((N,J));mask=np.zeros(J,bool);jointq=np.zeros((N,J));jlo=np.zeros(J);jhi=np.zeros(J)
 for r in a.mechanics:
  j=r['skeleton_index'];indices[j]=r['qpos_address'];jointq[:,j]=q[:,indices[j]];jlo[j],jhi[j]=r['range']
  if r['type']=='hinge':typ[j]=2
  elif r['type']=='slide':
   typ[j]=3;mask[j]=True;axis[j]=r['axis_rest_canonical'];lower[j],upper[j]=r['range'];ref[j]=r['model_qpos0'];slide[:,j]=q[:,indices[j]]
  else:raise ValueError('unsupported unexpected joint type')
 assert np.flatnonzero(mask).tolist()==[8,9]
 dec=a.decode_q(q);R=dec['global_matrices'];P=dec['position_native'];err=np.linalg.norm(mixed_fk(R,a.offsets,a.parents,axis,slide,ref)-P,axis=-1);assert err.max()<1e-9
 rotconst=max(float(np.max(np.abs(R[:,j]-R[:,a.parents[j]]))) for j in [8,9,10]);assert rotconst<1e-12
 norm=np.zeros_like(slide);norm[:,mask]=(slide[:,mask]-lower[mask])/(upper[mask]-lower[mask]);G=rot6(R).astype('float32')
 aperture=np.linalg.norm(P[:,8]-P[:,9],axis=-1);signed=np.sum((P[:,8]-P[:,9])*np.einsum('nab,b->na',R[:,7],axis[8]),axis=-1);aperr=float(np.max(np.abs(signed-q[:,7]-q[:,8])));assert aperr<1e-10
 payload={**unchanged,'global_rot6':G,'position_native':P.astype('float32'),'position_canonical':P.astype('float32'),'pose9':np.concatenate([G,P.astype('float32')/a.reach],axis=-1),'parents':a.parents,'rest_offsets':a.offsets.astype('float32'),'rest_offsets_descriptor':(a.offsets*100).astype('float32'),'joint_names':np.array(a.names,dtype='U'),'reach':np.array(a.reach,np.float32),'reach_descriptor':np.array(a.reach*100,np.float32),'dof_features':a.dof_features,'dof_valid':a.dof_valid,'legacy_rotation_dof_valid':a.dof_valid,'mechanical_valid':np.ones(J,bool),'joint_type':typ,'qpos_indices':indices,'joint_position_native':jointq.astype('float32'),'joint_position_valid':np.broadcast_to(indices>=0,(N,J)),'joint_lower_native':jlo.astype('float32'),'joint_upper_native':jhi.astype('float32'),'prismatic_axis':axis.astype('float32'),'slide_lower':lower.astype('float32'),'slide_upper':upper.astype('float32'),'slide_reference':ref.astype('float32'),'slide_mask':mask,'slide_valid':np.broadcast_to(mask,(N,J)),'slide_position_m':slide.astype('float32'),'slide_normalized':norm.astype('float32'),'prismatic_fixed_rotation':np.broadcast_to(np.eye(3),(J,3,3)).astype('float32'),'aperture_m':aperture.astype('float32'),'aperture_signed_m':signed.astype('float32'),'ee_indices':np.array([8,9,10]),'metric_length_valid':np.array(True),'ground_valid':np.array(False),'root_height_valid':np.array(False),'contact_valid':np.zeros(J,bool),'position_valid':np.ones((N,J),bool),'rotation_valid':np.ones((N,J),bool)}
 out=D/'payloads/relay_franka_full_v2.npz';assert not out.exists(),'v2 payload already exists; refuse overwrite';np.savez_compressed(out,**payload)
 # Independent re-read validates serialized float32 mixed FK, not just in-memory doubles.
 with np.load(out) as z:
  r=z['global_rot6'].astype('float64');u=r[...,:3];v=r[...,3:];u/=np.linalg.norm(u,axis=-1,keepdims=True);v-=np.sum(u*v,axis=-1,keepdims=True)*u;v/=np.linalg.norm(v,axis=-1,keepdims=True);RR=np.stack([u,v,np.cross(u,v)],axis=-1)
  qcerr=float(np.max(np.linalg.norm(mixed_fk(RR,z['rest_offsets'],z['parents'],z['prismatic_axis'],z['slide_position_m'],z['slide_reference'])-z['position_native'],axis=-1)));assert qcerr<2e-6
  for k,vv in unchanged.items():assert np.array_equal(z[k],vv),k
 selected=unchanged['frame_index']%40==0;outside=mask[None,:]&((slide<lower)|(slide>upper));seps=[]
 for ep in meta['episodes']:
  lo,hi=ep['row_range'];curve=aperture[lo:hi];dt=np.diff(unchanged['timestamp_seconds'][lo:hi]);delta=np.diff(curve)
  seps.append({'source_group':ep['source_group'],'member':ep['member'],'row_range':[lo,hi],'split':ep['split'],'aperture_min_m':float(curve.min()),'aperture_max_m':float(curve.max()),'aperture_first_m':float(curve[0]),'aperture_last_m':float(curve[-1]),'closing_intervals_gt_1um':int((delta < -1e-6).sum()),'opening_intervals_gt_1um':int((delta>1e-6).sum()),'aperture_speed_abs_max_m_per_s':float(np.max(np.abs(delta/dt)))})
 m={**meta,'schema':'recorded_mixed_joint_motion_v2','body_id':'relay_franka_full_v2','supersedes_for_new_six_run_training':'relay_franka_arm v1 (unchanged archived arm-only extraction)','model':a.metadata(),'payload_path':str(out),'payload_sha256':sha(out),'payload_shapes':{k:list(v.shape) for k,v in payload.items()},'unsupported_prismatic_joints':[],'scope':'All seven source arm hinges, both dynamic source finger slides, fixed root and exact source tool site. Mixed-joint FK required; rotation-only FK is invalid for fingers. No physical execution or task-success claim.','native_frame_identity_equals_v1':True,'original_v1_payload_sha256':oldhash,'original_v1_manifest_sha256':omhash,'mixed_joint_contract':{'joint_type_codes':{'unknown':0,'fixed':1,'revolute':2,'prismatic':3,'root':4},'units':'rest_offsets,slide_position_m,slide_lower,slide_upper,slide_reference: m; descriptor offsets: cm. Convert all slider length fields by100 when using descriptor cm in model FK.','slide_normalization':'(q-lower)/(upper-lower), independently per true slide; no target clipping. Fixed/nonprism slots zero with mask=false.','prismatic_axis_frame':'canonical rest-world basis before parent G; axis=AX@Rrest_child@axis_local_body','prismatic_rotation':'G_child=G_parent@prismatic_fixed_rotation; actual C=I measured all frames','position_formula':'P_child=P_parent+G_parent@(rest_offset+prismatic_axis*(q-slide_reference))','legacy19D':'dof_features encodes hinge slots only; dof_valid=false for sliders means unsupported in legacy19D, while mechanical_valid=true and explicit joint_type/axis/limits are known','coupling':'No MJCF mimic/equality constraint between fingers; two independently recorded sliders. No inferred symmetry enforced.'},'aperture_definition':'Signed projection of left-origin minus right-origin onto actual parent-rotated left slide axis; equals q_left+q_right (verified). Euclidean origin separation saved separately. This is model finger-origin opening, not measured contact-pad clearance.','aperture_episode_stats':seps,'source_soft_limit_violations':{'native_count_per_slide':[int(outside[:,j].sum()) for j in [8,9]],'selected_count_per_slide':[int(outside[selected,j].sum()) for j in [8,9]],'q_min_m':slide[:,[8,9]].min(0).tolist(),'q_max_m':slide[:,[8,9]].max(0).tolist(),'max_overshoot_m':float(max(np.max(slide[:,mask]-upper[mask]),np.max(lower[mask]-slide[:,mask]))),'handling':'Preserved exact simulation truth. A sigmoid/strict-limit prediction cannot reproduce tiny soft-limit excursions; report this nonzero representational floor instead of clipping GT.'},'validation':{'all_native_frames':N,'all_frame_mujoco_vs_mixed_fk_max_error_m':float(err.max()),'all_frame_serialized_mixed_fk_max_error_m':qcerr,'fixed_and_slide_rotation_inheritance_max_abs_error':rotconst,'aperture_equals_qsum_max_error_m':aperr,'v1_unchanged':sha(old)==oldhash and sha(om)==omhash,'selected_frames':int(selected.sum()),'selected_frame_counts_by_split':{str(i):int(np.sum(selected&(unchanged['split']==i))) for i in [0,1,2]}}}
 out.with_suffix('.json').write_text(json.dumps(m,indent=2)+'\n')
 nq=json.loads((D/'all_required_qwen_names.json').read_text());nq['bodies']['relay_franka_full_v2']=a.names;nq['deduplicated_joint_names']=sorted({n for names in nq['bodies'].values() for n in names});nq['note']='v1 names retained; two actual finger body names added. Use full_v2 in new six-run model.';(D/'all_required_qwen_names_v2.json').write_text(json.dumps(nq,indent=2)+'\n')
 print(json.dumps({'payload':str(out),'manifest':str(out.with_suffix('.json')),'J':J,'names':a.names,'bytes':out.stat().st_size,'validation':m['validation'],'violations':m['source_soft_limit_violations']}))
if __name__=='__main__':main()
