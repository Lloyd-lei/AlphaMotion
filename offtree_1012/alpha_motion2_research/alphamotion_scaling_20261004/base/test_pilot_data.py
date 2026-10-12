"""Small offline fixtures cover terminal acceptance and corruption/leakage refusal."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import shutil
from unittest.mock import patch
import unittest
import numpy as np
import torch
from xwbc_contract import (SPLIT_CHECKS,validate_xwbc,validate_payload,sha256)
from pilot_data import Body,make_inputs

CONTRACT=dict(body_joints={'body_a':3,'body_b':3},rows=4,train_rows=2,
 cache_rows=(32,32),fps=30.,max_frames=96,min_stride=3,group_counts={(1,0):1,(1,1):1})


def write(path,obj):path.write_text(json.dumps(obj,indent=2)+'\n')
def fixture(root):
 (root/'payloads').mkdir();(root/'static').mkdir()
 row=dict(cache_index=np.ones(4,np.int8),source_row_index=np.array([0,3,8,11],np.int64),
 row_ids=np.array([32,35,40,43],np.int64),group_index=np.array([0,0,1,1],np.int64),
 frame_index=np.array([0,3,0,3],np.int64),timestamp_seconds=np.array([0,.1,0,.1],np.float64),
 split=np.array([0,0,1,1],np.int8),pretrain_seen=np.array([1,1,0,0],bool))
 np.savez_compressed(root/'row_table.npz',**row)
 tab=np.stack([row[k] for k in ['cache_index','source_row_index','row_ids','group_index','frame_index','split','pretrain_seen']],-1).astype(np.int64)
 rowhash=hashlib.sha256(tab.tobytes()).hexdigest()
 groups=[]
 for i in range(2):groups.append(dict(group_id=f'amass::clip{i}',clip=f'clip{i}',source_dataset='fixture',cache_index=1,start=8*i,length=4,split=i,pretrain_seen=not bool(i),pretrain_excluded=bool(i),frame_indices=[0,3],chosen_frame_count=2))
 split=dict(groups=groups,body_names=list(CONTRACT['body_joints']),rows=4,fps=30.,max_frames_per_clip=96,minimum_source_frame_stride=3,
 row_table_sha256=sha256(root/'row_table.npz'),row_table_content_sha256=rowhash,
 counts=dict(amass_train_groups=1,amass_eval_groups=1,lafan_train_groups=0,lafan_eval_groups=0,train_rows=2,eval_rows=2),
 no_leakage_checks=SPLIT_CHECKS,pretrain_heldout_provenance=[dict(exclude_set='original/exclude.json',config_sha256='a'*64,launch_args_sha256='b'*64) for _ in range(2)],exclude_set_sha256='c'*64,
 source_sequence_files=[dict(manifest_sha256='d'*64),dict(manifest_sha256='e'*64)])
 write(root/'chosen_split.json',split)
 entries=[];checks=[]
 parents=np.array([-1,0,0],np.int64);off=np.array([[0,4,0],[1,0,0],[-1,0,0]],np.float32);names=np.array(['root','left_wrist','right_wrist'])
 for name in CONTRACT['body_joints']:
  gr=np.tile(np.array([1,0,0,0,1,0],np.float32),(4,3,1));pos=np.tile(np.array([[0,0,0],[1,0,0],[-1,0,0]],np.float32),(4,1,1));reach=2.
  arrays=dict(pose9=np.concatenate([gr,pos/reach],-1),global_rot6=gr,position_cm=pos,reach_cm=np.array(reach,np.float64),
   root_height_m=np.full(4,np.nan,np.float32),root_height_valid=np.zeros(4,bool),contact_sdist=np.full((4,3),np.nan,np.float32),
   contact_valid=np.zeros((4,3),bool),contact_soft=np.full((4,3),np.nan,np.float32),contact_hard=np.zeros((4,3),bool),contact_trust=np.zeros(4,bool),ground_valid=np.zeros(4,bool))
  arrays.update({('source_cache_index' if k=='cache_index' else k):v for k,v in row.items()})
  path=root/'payloads'/f'{name}.npz';np.savez_compressed(path,**arrays)
  spec=root/'static'/f'{name}_spec.npz';dof=root/'static'/f'{name}_dof.npz';desc=root/'static'/f'{name}_descriptor.json'
  np.savez_compressed(spec,name=np.asarray(name),parents=parents,rest_offsets=off,joint_names=names.astype(object))
  np.savez_compressed(dof,dof=np.zeros((3,19),np.float32),joint_names=names.astype(object),slot_joints=np.asarray(['','',''],dtype=object))
  write(desc,dict(body=name,J=3,joint_names=names.tolist(),parents=parents.tolist(),rest_offsets_cm=off.tolist(),
   norm_bonelen=[4/6,1/6,1/6],depth=[0,1,1],hop=[[0,1,1],[1,0,2],[1,2,0]],dof_shape=[3,19],reach_cm=reach,root_index=0,source_spec_sha256=sha256(spec)))
  files={}
  for kind,p in [('spec',spec),('dof',dof),('descriptor',desc)]:
   files[kind]=dict(path=str(p.relative_to(root)),sha256=sha256(p))
   if kind!='descriptor':files[kind].update(source_path='original/'+p.name,source_sha256=sha256(p))
  contacts=[dict(exists=True,path='fixture/contact.npz',sha256='c'*64,header=dict(N=32,J=3,thr_m=.04,D_used_m=1.,h0_m=0.,sole_applied=[],sole_joint_names=[])) for _ in range(2)]
  ground=dict(schema_version=1,body=name,J=3,source_spec_sha256=sha256(spec),ground_geometry_verified=True,
   sole_j=[],sole_off_m=[],sole_joint_names=[],mapping_status='verified_empty_legacy_mapping',
   sole_source=dict(path='fixture/sole_offsets3.json',sha256='f'*64),sole_function_source=dict(path='fixture/contact_label.py',sha256='a'*64),
   threshold_source_cache_index=1,contact_threshold_m=.04,contact_tau_m=.04/3,source_contacts=contacts)
  gp=root/'static'/f'{name}_ground.json';write(gp,ground)
  entries.append(dict(body=name,J=3,payload=str(path.relative_to(root)),payload_sha256=sha256(path),payload_bytes=path.stat().st_size,reach_cm=reach,
   source_rotations=[dict(body=name,rows=32,shape=[32,3,6],reach_cm=reach) for _ in range(2)],static_descriptor=str(desc.relative_to(root)),static_files=files,source_contacts=contacts,ground_static=dict(path=str(gp.relative_to(root)),sha256=sha256(gp))))
  checks.append(dict(body=name,rows=4,J=3,pose9_all_finite=True,position_root_max_abs_cm=0,pose9_position_formula_max_abs_error=0,invalid_contact_is_nan=True,invalid_root_height_is_nan=True,contact_valid_values_finite=True,row_content_sha256=rowhash))
 manifest=dict(ground_recipe_seal=dict(schema_version=1,sole_json_sha256='f'*64,source_contact_label_sha256='a'*64),status='complete',rows=4,bodies=2,fps=30.,body_payloads=entries,checks=checks,chosen_split_sha256=sha256(root/'chosen_split.json'),row_table_sha256=sha256(root/'row_table.npz'),static_seal=dict(schema_version=1,mode='verified_during_extraction',source_cache_manifest_sha256='d'*64))
 write(root/'manifest.json',manifest);write(root/'checks.json',dict(all_pass=True,checks=checks,split_checks=SPLIT_CHECKS))
 return manifest,row,arrays,parents,off,names


class ContractTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
  self.manifest,self.row,self.arrays,self.parents,self.offsets,self.names=fixture(self.root)
 def tearDown(self):self.tmp.cleanup()
 def validate(self):return validate_xwbc(self.root,contract=CONTRACT)
 def update_manifest(self):write(self.root/'manifest.json',self.manifest)
 def test_complete_bundle_and_payload_accept(self):
  m,checks,split,row,entries,artifacts=self.validate()
  self.assertEqual(len(entries),2);self.assertIn('static/body_a_dof.npz',artifacts)
  validate_payload('body_b',self.arrays,row,entries['body_b'])
 def test_manifest_missing_body_rejected(self):
  self.manifest['body_payloads'].pop();self.update_manifest()
  with self.assertRaisesRegex(ValueError,'missing, duplicate or unexpected body'):self.validate()
 def test_payload_bad_hash_rejected(self):
  p=self.root/'payloads/body_a.npz'
  with p.open('ab') as f:f.write(b'corruption')
  with self.assertRaisesRegex(ValueError,'SHA256 mismatch payloads/body_a'):self.validate()
 def test_static_bad_hash_rejected(self):
  p=self.root/'static/body_a_dof.npz';np.savez_compressed(p,dof=np.ones((3,19),np.float32))
  with self.assertRaisesRegex(ValueError,'SHA256 mismatch static/body_a_dof'):self.validate()
 def test_static_seal_missing_rejected(self):
  del self.manifest['static_seal'];self.update_manifest()
  with self.assertRaisesRegex(ValueError,'static seal missing'):self.validate()
 def test_checks_false_rejected(self):
  p=self.root/'checks.json';obj=json.loads(p.read_text());obj['all_pass']=False;write(p,obj)
  with self.assertRaisesRegex(ValueError,'terminal checks did not pass'):self.validate()
 def test_cross_split_group_rejected_even_with_refreshed_sha(self):
  p=self.root/'chosen_split.json';obj=json.loads(p.read_text());obj['groups'][1]['clip']='clip0';obj['groups'][1]['group_id']='amass::clip0';write(p,obj)
  self.manifest['chosen_split_sha256']=sha256(p);self.update_manifest()
  with self.assertRaisesRegex(ValueError,'source group duplicated/cross-split'):self.validate()
 def test_timestamp_mismatch_rejected_even_with_refreshed_sha(self):
  self.row['timestamp_seconds'][2]=.5;np.savez_compressed(self.root/'row_table.npz',**self.row)
  digest=sha256(self.root/'row_table.npz');p=self.root/'chosen_split.json';obj=json.loads(p.read_text());obj['row_table_sha256']=digest;write(p,obj)
  self.manifest.update(row_table_sha256=digest,chosen_split_sha256=sha256(p));self.update_manifest()
  with self.assertRaisesRegex(ValueError,'timestamps disagree'):self.validate()
 def test_payload_pair_swap_rejected(self):
  self.arrays['source_row_index']=self.arrays['source_row_index'][::-1]
  with self.assertRaisesRegex(ValueError,'payload row pairing'):validate_payload('body_b',self.arrays,self.row,self.manifest['body_payloads'][1])
 def body(self,arrays=None,off=None):
  a=copy.deepcopy(self.arrays if arrays is None else arrays);a['group']=a['group_index']
  offsets=self.offsets if off is None else off
  return Body('body_b','xwbc',a,self.parents,offsets,offsets,self.names.tolist(),2.,np.zeros((3,19),np.float32),np.ones(3,bool),True,{})
 def test_nonfinite_static_and_valid_height_rejected(self):
  off=self.offsets.copy();off[1,0]=np.nan
  with self.assertRaisesRegex(ValueError,'nonfinite static'):self.body(off=off)
  a=copy.deepcopy(self.arrays);a['root_height_valid'][0]=True
  with self.assertRaisesRegex(ValueError,'valid measurement must be finite'):self.body(arrays=a)
 def test_fk_uses_global_rotations_and_ignores_root_offset(self):
  self.assertEqual(self.body().audit_fk(),0.)
 def prepare_unsealed_sources(self):
  source=self.root/'original';source.mkdir();write(source/'manifest.json',{'fixture':True})
  digest=sha256(source/'manifest.json')
  for name in CONTRACT['body_joints']:
   for kind in ['spec','dof']:shutil.copy2(self.root/'static'/f'{name}_{kind}.npz',source/f'{name}_{kind}.npz')
  for entry in self.manifest['body_payloads']:del entry['static_files']
  del self.manifest['static_seal']
  p=self.root/'chosen_split.json';obj=json.loads(p.read_text());obj['source_sequence_files'][0]['manifest_sha256']=digest;write(p,obj)
  self.manifest['chosen_split_sha256']=sha256(p);self.update_manifest()
  return source,digest
 def test_completed_legacy_extract_can_be_sealed_without_reextracting(self):
  import seal_xwbc_static
  source,digest=self.prepare_unsealed_sources()
  payload_before=sha256(self.root/'payloads/body_a.npz')
  with patch.object(seal_xwbc_static,'validate_xwbc',side_effect=lambda root,**kw:validate_xwbc(root,contract=CONTRACT,**kw)):
   result=seal_xwbc_static.seal_static(self.root,source,digest)
  self.assertEqual(result['mode'],'source_revalidated_at_seal_time')
  self.assertEqual(payload_before,sha256(self.root/'payloads/body_a.npz'))
  self.validate()
 def test_seal_refuses_static_copy_mismatch_without_writing_manifest(self):
  import seal_xwbc_static
  source,digest=self.prepare_unsealed_sources();before=sha256(self.root/'manifest.json')
  np.savez_compressed(source/'body_b_dof.npz',dof=np.ones((3,19),np.float32))
  with patch.object(seal_xwbc_static,'validate_xwbc',side_effect=lambda root,**kw:validate_xwbc(root,contract=CONTRACT,**kw)):
   with self.assertRaisesRegex(ValueError,'differs from original source'):seal_xwbc_static.seal_static(self.root,source,digest)
  self.assertEqual(before,sha256(self.root/'manifest.json'))
 def test_one_condition_is_two_condition_prefix(self):
  body=self.body();rows=np.arange(4)
  one,_=make_inputs(body,body,rows,np.random.default_rng(781),'cpu','masked',condition_count=1,condition_dropout=0.,contact_dropout=0.)
  two,_=make_inputs(body,body,rows,np.random.default_rng(781),'cpu','masked',condition_count=2,condition_dropout=0.,contact_dropout=0.)
  self.assertTrue(torch.equal(one['condition']['joint_indices'][:,0],two['condition']['joint_indices'][:,0]))
  self.assertTrue(torch.equal(one['source_hidden'],two['source_hidden']))
if __name__=='__main__':unittest.main()
