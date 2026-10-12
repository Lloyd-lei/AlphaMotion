#!/usr/bin/env python3
"""Seal independent external data and report true labels. No source or payload edits."""
from pathlib import Path
import json,hashlib,shutil,datetime
import numpy as np
ROOT=Path('/home/arenalabs/Desktop')/'"be water, robot"'/'alpha_motion2/research/alphamotion_scaling_20261004/data'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 bodies=[]
 for name,stride in [('adroit_pen',1),('relay_franka_arm',40)]:
  p=ROOT/'payloads'/(name+'.json');m=json.loads(p.read_text());npz=Path(m['payload_path']);assert sha(npz)==m['payload_sha256']
  with np.load(npz) as z:
   use=z['frame_index']%stride==0;s=z['split'];sg=z['source_group_index'];fr=z['frame_index'];N,J=z['global_rot6'].shape[:2];q=z['recorded_robot_qpos'];times=z['timestamp_seconds']
   assert np.isfinite(q).all() and all(len(np.unique(s[sg==g]))==1 for g in np.unique(sg))
   for g in np.unique(sg):
    rows=sg==g;assert np.array_equal(fr[rows],np.arange(rows.sum()));assert np.all(np.diff(times[rows])>0)
   assert z['position_valid'].all() and z['rotation_valid'].all() and not z['contact_valid'].any() and not bool(z['ground_valid'])
   rotation=z['global_rot6'].astype('float64');aa=rotation[...,:3];bb=rotation[...,3:];aa/=np.linalg.norm(aa,axis=-1,keepdims=True);bb-=np.sum(aa*bb,axis=-1,keepdims=True)*aa;bb/=np.linalg.norm(bb,axis=-1,keepdims=True);R=np.stack([aa,bb,np.cross(aa,bb)],axis=-1);pred=np.zeros((N,J,3))
   for j,pa in enumerate(z['parents']):
    if pa>=0:pred[:,j]=pred[:,pa]+np.einsum('nab,b->na',R[:,pa],z['rest_offsets'][j])
   err=float(np.max(np.linalg.norm(pred-z['position_native'],axis=-1)));assert err<2e-6
   item={'body_id':name,'manifest':str(p),'manifest_sha256':sha(p),'payload':str(npz),'payload_sha256':sha(npz),'payload_bytes':npz.stat().st_size,'source_dataset':m['source_dataset'],'license':m['source_license'],'J':J,'native_frames':N,'native_independent_episodes':m['independent_groups'],'native_frame_counts_by_split':m['frame_counts_by_split'],'episode_counts_by_split':m['group_counts_by_split'],'training_frame_rule':f'frame_index % {stride} == 0','selected_frames':int(use.sum()),'selected_frames_by_split':{str(i):int(np.sum(use&(s==i))) for i in [0,1,2]},'native_labels':{'recorded_robot_qpos_scalar_count':int(q.size),'rotation_joint_frames':N*J,'position_joint_frames':N*J,'known_dof_joint_count':int(z['dof_valid'].sum()),'metric_length_valid':True,'ground_valid_frames':0,'contact_valid_joint_frames':0,'root_height_valid_frames':0},'selected_labels':{'recorded_robot_qpos_scalar_count':int(use.sum()*q.shape[1]),'rotation_joint_frames':int(use.sum()*J),'position_joint_frames':int(use.sum()*J),'ground_valid_frames':0,'contact_valid_joint_frames':0,'root_height_valid_frames':0},'qpos_native_range':np.ptp(q,axis=0).tolist(),'serialized_float32_fk_max_error_m':err,'max_original_double_fk_error_m':max(e['fk_max_error_m'] for e in m['episodes']),'unsupported_prismatic_joints':m.get('unsupported_prismatic_joints',[]),'task_family':'unpaired_same_body_reconstruction_and_sparse_completion','simulated':True,'from_scratch':True,'zero_cross_body_mapping_gt':True}
   bodies.append(item)
 records=[]
 for n in ['minari_acquisition.json','model_acquisition.json','relay_acquisition.json','relay_model_acquisition.json','relay_model_derivation.json','all_required_qwen_names.json']:
  p=ROOT/n;records.append({'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size})
 archive_bytes=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file() and not p.is_symlink());cache_link_bytes=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_symlink())
 result={'schema':'scaling_external_seal_v1','sealed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'bodies':bodies,'artifacts':records,'frozen_payloads_and_per_body_manifests':True,'same_data_and_stride_all_six_runs':True,'dataset_split_unit':'entire independent episode or exact content-hash alias group; no frame split','source_rights':[{'dataset':'D4RL/pen/human-v2','license':'CC-BY-4.0','evidence_url':'https://github.com/Farama-Foundation/D4RL#licenses','attribution':['DAPG/Rajeswaran et al., RSS 2018','D4RL/Fu et al., 2020','Farama conversion/Rodrigo de Lazcano'],'revision':'a0fd465d1d9d253b075d532b270e03d8fb4f46e6','source_url':'https://huggingface.co/datasets/farama-minari/D4RL/tree/a0fd465d1d9d253b075d532b270e03d8fb4f46e6/pen/human-v2'},{'dataset':'relay-policy-learning original kitchen MJL','license':'Apache-2.0','evidence_url':'https://github.com/google-research/relay-policy-learning/tree/cd70ac9334f584f86db281a1ffd3e5cbc3e5e293','attribution':['Google Research relay-policy-learning, 2020','Vikash Kumar Franka model and original contributors'],'revision':'cd70ac9334f584f86db281a1ffd3e5cbc3e5e293'},{'geometry':'Adroit Gymnasium-Robotics','revision':'4d1ebecbc6436806cfbc0e42ebc36f594d05844e','top_level_license':'MIT','asset_license_notice':'Apache-2.0 and source attribution retained in assets/LICENSE.md','evidence_url':'https://github.com/Farama-Foundation/Gymnasium-Robotics/blob/4d1ebecbc6436806cfbc0e42ebc36f594d05844e/gymnasium_robotics/envs/assets/LICENSE.md'}],'excluded_source':{'dataset':'D4RL/kitchen/complete-v2','reason':'observed qpos carries simulator observation noise; no precise per-frame qpos. Download kept for audit only, not training/evaluation.'},'storage':{'new_data_directory_regular_file_bytes':archive_bytes,'HF_cache_linked_bytes_additional':cache_link_bytes,'sum_no_double_count':archive_bytes+cache_link_bytes,'cap_bytes':2*2**30,'full_zip_extracted':False,'free_bytes_now':shutil.disk_usage(ROOT).free},'checksum_level':'all added raw downloads pinned and SHA256 verified; all derived payload bytes hashed; every included original/serialized frame FK and time/group contracts numerically verified; original multi-GB XWBC caches not fully scanned'}
 assert archive_bytes+cache_link_bytes<2*2**30
 p=ROOT/'external_manifest.json';p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
