"""CPU synthetic tests of full restored step; never loads remote data or steps optimizer."""
import copy,json,unittest,contextlib,tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from test_train_pilot import synthetic_body,small_model
import train_pilot as trainer
from recipe_training import make_reference_inputs,xwbc_combos,backward_training_step
from pilot_data import make_inputs
from xwbc_contract import BODY_JOINTS
from common import SkeletonSpec
from masking import parse_mix,sample_joint_mask

torch.set_num_threads(1)

def config():return json.loads(Path(__file__).with_name('candidate_config.json').read_text())

def bodies():
 out=[]
 for name in BODY_JOINTS:
  b=synthetic_body(name,'xwbc');a=b.arrays;N,J=a['pose9'].shape[:2]
  a.update(row_ids=np.arange(N,dtype=np.int64),source_cache_index=np.zeros(N,np.int64),source_row_index=np.arange(N,dtype=np.int64),group_index=a['group'].copy(),frame_index=np.arange(N),timestamp_seconds=np.arange(N)/30.,root_height_valid=np.ones(N,bool),root_height_m=np.full(N,.05,np.float32),contact_valid=np.ones((N,J),bool),contact_soft=np.full((N,J),.8,np.float32),contact_hard=np.ones((N,J),bool),contact_sdist=np.full((N,J),.3,np.float32),contact_trust=np.ones(N,bool),ground_valid=np.ones(N,bool))
  b.metadata['ground_static']={'ground_geometry_verified':True,'contact_threshold_m':.08,'sole_j':[2,4],'sole_off_m':[[0,-.4,0],[0,-.4,0]],'mapping_status':'SYNTHETIC TEST ONLY'}
  out.append(b)
 out.append(synthetic_body('synthetic_unimate','unimate'));return out

class RestoredStep(unittest.TestCase):
 def setUp(self):
  torch.manual_seed(33);self.b=bodies();self.c=config();self.c['batch_size']=4;self.c['judge_body_order']=list(BODY_JOINTS);self.device=torch.device('cpu')
 def test_reference_mask_and_contact_match_original_sampling(self):
  src=self.b[0];rows=np.array([0,1,2,3]);c=self.c
  torch.manual_seed(44)
  actual,_=make_reference_inputs(src,src,rows,np.random.default_rng(5),self.device,c)
  torch.manual_seed(44);g=src.ground_batch(rows,self.device)
  expected=sample_joint_mask(SkeletonSpec(src.name,src.parents,src.offsets,src.names),4,parse_mix(c['recipe']['mask_mix']),device='cpu',contact=g['contact_soft'],contact_bin=g['contact_hard'])
  expected &= torch.rand(4,1)<.3
  drop=(torch.rand(4,1)<.3)&~expected.any(1,keepdim=True)
  self.assertTrue(torch.equal(actual['source_hidden'],expected));self.assertFalse(bool(expected[:,0].any()))
  con=torch.where(drop,torch.full((4,5),float('nan')),torch.full((4,5),.3))
  torch.testing.assert_close(actual['source_contact'],con,equal_nan=True)
 def test_contact_preserved_when_pose_masked_but_not_in_pure_se3(self):
  c=copy.deepcopy(self.c);c['recipe'].update(mask_prob=1.,mask_mix='root:1');c['contact_dropout']=1.
  src=self.b[0];rows=np.array([0,1,2])
  i,_=make_reference_inputs(src,src,rows,np.random.default_rng(1),self.device,c)
  self.assertTrue(bool(i['source_hidden'][:,1:].all()));self.assertTrue(bool(torch.isfinite(i['source_contact']).all()))
  j,_=make_inputs(src,src,rows,np.random.default_rng(1),self.device,'sparse_only',1)
  self.assertIsNone(j['source_contact']);self.assertTrue(bool(j['source_hidden'].all()))
 def test_eight_directions_unique_and_include_self_in_universe(self):
  combos=xwbc_combos(['a','b','c'],np.random.default_rng(1),9)
  self.assertEqual(set(combos),{(a,b) for a in 'abc' for b in 'abc'})
  self.assertEqual(len(set(xwbc_combos(list(BODY_JOINTS),np.random.default_rng(4)))),8)
 def test_full_restored_step_backward_all_losses_no_optimizer(self):
  model=small_model();before={n:p.detach().clone() for n,p in model.named_parameters()}
  report=backward_training_step(model,self.b,self.c,np.random.default_rng(901),np.random.default_rng(902),np.random.default_rng(903),device=self.device,domain='xwbc',forward_fn=trainer.forward,objective_fn=trainer.objective,autocast_fn=trainer.autocast,force_all_losses=True)
  main=[d for d in report['directions'] if d['mode']=='reference_mask_mix'];extra=[d for d in report['directions'] if d['mode']=='sparse_only_extra']
  self.assertEqual(len(main),8);self.assertEqual(sum(d['weight'] for d in main),1.);self.assertEqual(extra[0]['weight'],1.)
  self.assertTrue({'pen','cgeo','root_height_smooth_l1_m','contrast','judge','reenc'}<=set(report['terms']))
  self.assertTrue(report['auxiliary']['reenc_active']);self.assertEqual(report['auxiliary']['contrast']['pairs'],15)
  self.assertTrue(all(np.isfinite(v) for v in report['terms'].values()))
  self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
  for prefix in ['inp.','hpos.','judge.','condition_projection.']:
   self.assertGreater(sum(float(p.grad.abs().sum()) for n,p in model.named_parameters() if n.startswith(prefix) and p.grad is not None),0.)
  self.assertTrue(all(torch.equal(p,before[n]) for n,p in model.named_parameters()))
 def test_full_preflight_profiles_are_recorded_but_cpu_cannot_issue_receipt(self):
  model=small_model();c=copy.deepcopy(self.c);before={n:p.detach().clone() for n,p in model.named_parameters()}
  with tempfile.TemporaryDirectory(prefix='SYNTHETIC-full-recipe-') as tmp:
   report=trainer.preflight(model,self.b,c,Path(tmp))
   self.assertEqual(len(report['full_recipe_profiles']),2)
   self.assertFalse(report['full_reference_recipe_verified']);self.assertEqual(report['optimizer_steps'],0)
   self.assertEqual(report['full_recipe_profiles'][0]['domain'],'xwbc')
   self.assertTrue(all(torch.equal(p,before[n]) for n,p in model.named_parameters()))
   self.assertTrue(all(p.grad is None for p in model.parameters()))
   from recipe_gate import write_validation_receipt
   with self.assertRaises((ValueError,RuntimeError)):write_validation_receipt(c,tmp,tmp,report)
   self.assertFalse((Path(tmp)/'recipe_validation_receipt.json').exists())
 def test_unimate_no_aux_or_metric_ground_terms(self):
  model=small_model()
  with patch('recipe_auxiliary.auxiliary_losses',side_effect=AssertionError('fake pairing')):
   r=backward_training_step(model,self.b,self.c,np.random.default_rng(1),np.random.default_rng(2),np.random.default_rng(3),device=self.device,domain='unimate',forward_fn=trainer.forward,objective_fn=trainer.objective,autocast_fn=trainer.autocast)
  self.assertFalse(r['auxiliary']['cross_body_auxiliary_applied'])
  self.assertFalse({'pen','cgeo','root_height_smooth_l1_m','contrast','judge','reenc'}&set(r['terms']))
  self.assertTrue(all(p.grad is None for p in model.judge.parameters()))
 def test_main_mean_and_auxiliary_not_divided_by_eight(self):
  model=torch.nn.Linear(1,1,bias=False);model.weight.data.fill_(2.);c=copy.deepcopy(self.c);c['xwbc_sparse_extra']['probability']=0
  selected=xwbc_combos(c['judge_body_order'],np.random.default_rng(902),8);by={n:i+1 for i,n in enumerate(c['judge_body_order'])}
  def obj(out,target,tgt,rows,condition,weights):return model.weight.square().sum()*by[tgt.name],{}
  def auxiliary(*a,**kw):return {'total':3*model.weight.sum(),'contrast':None,'judge':None,'reenc':None,'diagnostics':{}}
  with patch('recipe_auxiliary.auxiliary_losses',side_effect=auxiliary):
   r=backward_training_step(model,self.b,c,np.random.default_rng(901),np.random.default_rng(902),np.random.default_rng(903),device=self.device,domain='xwbc',forward_fn=lambda *a:None,objective_fn=obj,autocast_fn=lambda d:contextlib.nullcontext())
  expected=4*np.mean([by[t] for _,t in selected])+3
  self.assertAlmostEqual(float(model.weight.grad),expected,places=5)

if __name__=='__main__':unittest.main(verbosity=2)
