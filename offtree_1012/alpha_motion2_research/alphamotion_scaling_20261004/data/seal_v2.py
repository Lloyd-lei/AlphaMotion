#!/usr/bin/env python3
from pathlib import Path
import json,hashlib,datetime,shutil
import numpy as np
import mujoco
D=Path('/home/arenalabs/Desktop')/'"be water, robot"'/'alpha_motion2/research/alphamotion_scaling_20261004/data'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 evidence={}
 for name in ['adroit_pen','relay_franka_full_v2']:
  p=D/'payloads'/(name+'.json');m=json.loads(p.read_text());model=mujoco.MjModel.from_xml_path(m['model']['model_xml']);b=m['model']['body_ids'][0];anc=[]
  while b>0:anc.append({'body_id':b,'name':mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,b),'joint_count':int(model.body_jntnum[b])});b=int(model.body_parentid[b])
  assert all(x['joint_count']==0 for x in anc)
  with np.load(m['payload_path']) as z:
   gr=z['global_rot6'][:,0];err=float(np.max(np.abs(gr-np.array([1,0,0,0,1,0]))));rw=z['world_root_m_zup'];trans=float(np.max(np.abs(rw-rw[:1])))
  assert err<1e-12 and trans==0
  evidence[name]={'root_fixed':True,'root_global_rebased_rotation':'identity by static model root/ancestor topology, never GT readback at inference','root_and_ancestors':anc,'all_native_frame_root_rotation_max_abs_I_error':err,'all_native_root_world_translation_max_change_m':trans,'payload_sha256':m['payload_sha256'],'model_xml_sha256':m['model']['model_xml_sha256']}
  if name.endswith('v2'):
   m['model']['root_fixed']=True;m['model']['root_fixed_evidence']=evidence[name]
   m['aperture_indices']={'left_finger':8,'right_finger':9,'parent':7,'definition':'finger-origin signed separation along parentG @ prismatic_axis[8]','not_contact_pad_clearance':True}
   p.write_text(json.dumps(m,indent=2)+'\n')
 (D/'root_fixed_evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
 old=json.loads((D/'external_manifest.json').read_text());v=json.loads((D/'payloads/relay_franka_full_v2.json').read_text());p=Path(v['payload_path']);assert sha(p)==v['payload_sha256']
 with np.load(p) as z:
  selected=z['frame_index']%40==0;n=int(selected.sum());J=len(z['parents']);q=z['slide_position_m'][:,[8,9]];lo=z['slide_lower'][[8,9]];hi=z['slide_upper'][[8,9]];gap=np.maximum(lo-q,0)+np.maximum(q-hi,0);floor=gap/(hi-lo)
  ap=z['aperture_m'];stats={'native_aperture_min_max_m':[float(ap.min()),float(ap.max())],'selected_aperture_min_max_m':[float(ap[selected].min()),float(ap[selected].max())],'irreducible_selected_strict_limit_slide_L1_normalized':float(floor[selected].mean()),'irreducible_selected_strict_limit_slide_L1_m':float(gap[selected].mean()),'strict_limit_prediction_vs_recorded_gt_note':'minimum if each predicted q is constrained to [lower,upper]; original soft-limit excursions preserved in GT','valid_slide_scalar_count_native':int(len(q)*2),'valid_slide_scalar_count_selected':int(n*2),'rotation_position_joint_frames_selected':int(n*J)}
 record={'body_id':'relay_franka_full_v2','manifest':str(p.with_suffix('.json')),'manifest_sha256':sha(p.with_suffix('.json')),'payload':str(p),'payload_sha256':sha(p),'payload_bytes':p.stat().st_size,'source_dataset':v['source_dataset'],'license':v['source_license'],'J':11,'native_frames':v['frames'],'native_independent_episodes':v['independent_groups'],'native_frame_counts_by_split':v['frame_counts_by_split'],'episode_counts_by_split':v['group_counts_by_split'],'training_frame_rule':'frame_index % 40 == 0','selected_frames':3699,'selected_frames_by_split':v['validation']['selected_frame_counts_by_split'],'selected_labels':{'recorded_robot_qpos_scalar_count':33291,'revolute_q_scalar_count':25893,'slide_q_scalar_count':7398,'rotation_joint_frames':40689,'position_joint_frames':40689,'aperture_frames':3699,'ground_valid_frames':0,'contact_valid_joint_frames':0,'root_height_valid_frames':0},'mechanical_joint_counts':{'fixed':2,'hinge':7,'prismatic':2},'metric_length_valid':True,'mixed_fk_required':True,'joint_type_codes':v['mixed_joint_contract']['joint_type_codes'],'model_root_fixed':True,'serialized_float32_fk_max_error_m':v['validation']['all_frame_serialized_mixed_fk_max_error_m'],'aperture_and_bound_qc':stats,'soft_limit_violations':v['source_soft_limit_violations'],'source_frame_identity_equals_v1':True,'unsupported_prismatic_joints':[],'simulated':True,'task_family':'unpaired_same_body_reconstruction_and_sparse_completion','zero_cross_body_mapping_gt':True}
 out={**old,'schema':'scaling_external_seal_v2','sealed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'bodies':[old['bodies'][0],record],'archived_body_ids_excluded_from_formal_training':['relay_franka_arm'],'archived_v1_manifest':{'path':str(D/'external_manifest.json'),'sha256':sha(D/'external_manifest.json')},'fixed_root_evidence':{'path':str(D/'root_fixed_evidence.json'),'sha256':sha(D/'root_fixed_evidence.json')},'joint_names_v2':{'path':str(D/'all_required_qwen_names_v2.json'),'sha256':sha(D/'all_required_qwen_names_v2.json')},'prismatic_contract':v['mixed_joint_contract'],'no_new_download_for_v2':True,'frozen_payloads_and_per_body_manifests':True}
 regular=sum(x.stat().st_size for x in D.rglob('*') if x.is_file() and not x.is_symlink());links=sum(x.stat().st_size for x in D.rglob('*') if x.is_symlink());out['storage']={'new_data_directory_regular_file_bytes':regular,'HF_cache_linked_bytes_additional':links,'sum_no_double_count':regular+links,'cap_bytes':2**31,'full_zip_extracted':False,'free_bytes_now':shutil.disk_usage(D).free};assert regular+links<2**31
 for b in old['bodies']:assert sha(b['payload'])==b['payload_sha256'] and sha(b['manifest'])==b['manifest_sha256']
 (D/'external_manifest_v2.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
