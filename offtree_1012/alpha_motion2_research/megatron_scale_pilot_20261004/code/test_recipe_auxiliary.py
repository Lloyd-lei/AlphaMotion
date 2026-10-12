"""Offline numerical/gradient diagnostics; synthetic data/weights, no optimizer."""
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'legacy'));sys.path.insert(0,str(HERE))
from model_dual import DualTranslator
from pilot_model import ConditionalMegatron,load_reference_weights,model_parameter_counts,initialize_new_judge
from recipe_auxiliary import (AuxiliaryRecipe,contrast_loss,judge_loss,cycle_consistency,auxiliary_losses)
from xwbc_contract import BODY_JOINTS

torch.set_num_threads(1)
ORDER=tuple(BODY_JOINTS) # Synthetic current-label mapping only; no historical-order claim.

def small_model():
 return ConditionalMegatron(d_model=16,n_heads=4,enc_layers=1,dec_layers=1,n_latent=3,n_latent_p=2,fsq_stages=1,fsq_levels=(8,8,8),sem_dim=1024)

def geom(batch,joints=3):
 depth=torch.arange(joints)[None].expand(batch,-1)
 return dict(rest_off=torch.randn(batch,joints,3),bonelen=torch.ones(batch,joints)/joints,
 depth=depth,hop=(depth[:,:,None]-depth[:,None,:]).abs(),sem=F.normalize(torch.randn(batch,joints,1024),dim=-1),
 dof=torch.zeros(batch,joints,19),metadata_missing=torch.zeros(batch,joints,2))

class TinyBody:
 def __init__(self,name,batch=16,joints=3,domain='xwbc'):
  self.name=name;self.domain=domain
  self.arrays={'pose9':np.random.default_rng(1).normal(size=(batch,joints,9)).astype(np.float32),'split':np.zeros(batch,np.int8),
   'row_ids':np.arange(batch,dtype=np.int64),'source_cache_index':np.zeros(batch,np.int8),'source_row_index':np.arange(batch,dtype=np.int64),
   'group_index':np.zeros(batch,np.int64),'frame_index':np.arange(batch,dtype=np.int64),'timestamp_seconds':np.arange(batch,dtype=np.float64)/30}
  self.g=geom(1,joints)
 def geom(self,batch):return {k:v.expand(batch,*v.shape[1:]) for k,v in self.g.items()}
 def tensor(self,key,rows,device):return torch.as_tensor(self.arrays[key][rows],device=device)

class ToyCycle(nn.Module):
 def __init__(self):
  super().__init__();self.a=nn.Parameter(torch.tensor(.6,dtype=torch.float64));self.b=nn.Parameter(torch.tensor(.4,dtype=torch.float64));self.c=nn.Parameter(torch.tensor(.7,dtype=torch.float64));self.inputs=[]
 def encode(self,x,g):
  self.inputs.append(x);z=self.a*x.mean((1,2))[:,None,None];return z,2*z
 def decode(self,zp,zr,g):return self.b*zp.expand(-1,1,6),self.c*zr.expand(-1,1,3)

class AuxiliaryNumerics(unittest.TestCase):
 def setUp(self):torch.manual_seed(77)
 def latents(self,batch=5):
  return {n:(torch.randn(batch,2,4,requires_grad=True),torch.randn(batch,3,4,requires_grad=True)) for n in ORDER[:3]}
 def test_default_recipe_matches_original_run_config(self):
  import json
  cfg=json.loads((HERE/'reference_recipe/L_M20k_0903.config.json').read_text())
  self.assertEqual(AuxiliaryRecipe.from_mapping(cfg),AuxiliaryRecipe())
 def test_unique_infonce_matches_old_value_and_gradient(self):
  zs=self.latents();actual,diag=contrast_loss(zs,np.arange(5),.1);names=list(zs);terms=[]
  for i,a in enumerate(names):
   for b in names[i+1:]:
    va=F.normalize(torch.cat([z.reshape(5,-1) for z in zs[a]],-1),dim=-1)
    vb=F.normalize(torch.cat([z.reshape(5,-1) for z in zs[b]],-1),dim=-1)
    logits=va@vb.T/.1;labels=torch.arange(5)
    terms.append(.5*(F.cross_entropy(logits,labels)+F.cross_entropy(logits.T,labels)))
  expected=torch.stack(terms).mean();self.assertTrue(torch.allclose(actual,expected,atol=1e-7,rtol=1e-7))
  inputs=[x for z in zs.values() for x in z]
  ga=torch.autograd.grad(actual,inputs,retain_graph=True);ge=torch.autograd.grad(expected,inputs)
  self.assertTrue(all(torch.allclose(a,b,atol=1e-7,rtol=1e-6) for a,b in zip(ga,ge)))
  self.assertEqual(diag['pairs'],3)
 def test_duplicate_source_rows_are_not_false_negatives(self):
  zs=self.latents();ids=np.array([41,17,41,17,90]);actual,diag=contrast_loss(zs,ids)
  unique={n:(zp[[0,1,4]],zr[[0,1,4]]) for n,(zp,zr) in zs.items()}
  expected,_=contrast_loss(unique,np.array([41,17,90]))
  self.assertTrue(torch.allclose(actual,expected,atol=1e-7));self.assertEqual(diag['duplicate_rows'],2)
  actual.backward()
  for zp,zr in zs.values():
   self.assertEqual(float(zp.grad[[2,3]].abs().max()),0.)
   self.assertEqual(float(zr.grad[[2,3]].abs().max()),0.)
 def test_one_unique_source_row_has_zero_contrast(self):
  zs=self.latents();loss,diag=contrast_loss(zs,np.full(5,13,np.int64));loss.backward()
  self.assertEqual(float(loss.detach()),0.);self.assertFalse(diag['active'])
  self.assertTrue(all(torch.isfinite(x.grad).all() for z in zs.values() for x in z))
 def test_judge_grad_and_observe_match_old_bce(self):
  for mode in ['grad','observe']:
   model=small_model();zs={ORDER[0]:(torch.randn(5,2,16,requires_grad=True),torch.randn(5,3,16,requires_grad=True))}
   loss=judge_loss(model,zs,ORDER,mode)
   zp,zr=zs[ORDER[0]];logits=model.judge(zp,zr);target=torch.zeros_like(logits);target[:,0]=1
   self.assertTrue(torch.allclose(loss,F.binary_cross_entropy_with_logits(logits,target),atol=1e-7))
   loss.backward();self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.judge.parameters()))
   if mode=='grad':self.assertGreater(float(zp.grad.abs().sum()),0.)
   else:self.assertIsNone(zp.grad);self.assertIsNone(zr.grad)
 def test_judge_current_label_order_is_used_and_cannot_silently_change(self):
  model=small_model();zp=torch.randn(2,2,16);zr=torch.randn(2,3,16);order=ORDER[::-1]
  loss=judge_loss(model,{ORDER[0]:(zp,zr)},order);logits=model.judge(zp,zr);target=torch.zeros_like(logits);target[:,-1]=1
  self.assertTrue(torch.allclose(loss,F.binary_cross_entropy_with_logits(logits,target)))
  with self.assertRaisesRegex(ValueError,'reorder'):model.set_judge_body_order(ORDER)
 def test_judge_off_has_no_head_gradient(self):
  model=small_model();zp=torch.randn(2,2,16,requires_grad=True);zr=torch.randn(2,3,16,requires_grad=True)
  loss=judge_loss(model,{ORDER[0]:(zp,zr)},ORDER,'off');loss.backward()
  self.assertTrue(all(p.grad is None for p in model.judge.parameters()))
 def test_cycle_raw_hpos_and_both_latent_sides_have_exact_gradient(self):
  model=ToyCycle();x=torch.tensor([1.,2.],dtype=torch.float64)[:,None,None].expand(-1,1,9)
  loss=cycle_consistency(model,x,{},{});u=x.mean((1,2));a=model.a.detach();b=model.b.detach();c=model.c.detach()
  delta=(2/3)*(b+c)*a*a*u-a*u;expected=5*delta.square().mean()
  self.assertTrue(torch.allclose(loss,expected,atol=1e-12,rtol=1e-12))
  loss.backward();expected_da=10*(delta*((4/3)*(b+c)*a*u-u)).mean()
  self.assertTrue(torch.allclose(model.a.grad,expected_da,atol=1e-12,rtol=1e-12))
  self.assertTrue(torch.allclose(model.inputs[1][...,6:],2*c*a*u[:,None,None].expand(-1,1,3)))
  self.assertGreater(float(model.c.grad.abs()),0.)
 def test_cycle_actual_model_hpos_receives_finite_gradient_without_condition(self):
  model=small_model();loss=cycle_consistency(model,torch.randn(8,3,9),geom(8),geom(8,4));loss.backward()
  self.assertTrue(torch.isfinite(loss));self.assertGreater(float(model.hpos[-1].weight.grad.abs().sum()),0.)
  self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
  self.assertTrue(all(p.grad is None for p in model.condition_projection.parameters()))
 def test_orchestration_six_bodies_original_half_batch_and_single_total(self):
  model=small_model();bodies={n:TinyBody(n) for n in ORDER[:8]};rows=np.arange(16)
  out=auxiliary_losses(model,bodies,rows,device='cpu',judge_body_order=ORDER,combo_rng=np.random.default_rng(4),reenc_rng=np.random.default_rng(77),recipe={'reenc_prob':1.})
  self.assertEqual(len(out['diagnostics']['contrast_bodies']),6);self.assertEqual(out['diagnostics']['contrast']['pairs'],15)
  self.assertEqual(out['diagnostics']['reenc_frames'],8)
  expected=.5*out['contrast']+.1*out['judge']+out['reenc'];self.assertTrue(torch.equal(out['total'],expected))
  out['total'].backward();self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
  self.assertGreater(float(model.inp.weight.grad.abs().sum()),0.);self.assertGreater(float(model.judge.net[-1].weight.grad.abs().sum()),0.)
  self.assertTrue(all(p.grad is None for p in model.condition_projection.parameters()))
 def test_cycle_probability_zero_and_unimate_or_unpaired_rows_rejected(self):
  model=small_model();bodies={n:TinyBody(n) for n in ORDER[:2]};kwargs=dict(device='cpu',judge_body_order=ORDER,combo_rng=np.random.default_rng(4),reenc_rng=np.random.default_rng(77),recipe={'reenc_prob':0.})
  out=auxiliary_losses(model,bodies,np.arange(8),**kwargs);self.assertFalse(out['diagnostics']['reenc_active'])
  bodies[ORDER[1]].domain='unimate'
  with self.assertRaisesRegex(ValueError,'XWBC-only'):auxiliary_losses(model,bodies,np.arange(8),**kwargs)
  bodies[ORDER[1]].domain='xwbc';bodies[ORDER[1]].arrays['timestamp_seconds'][0]=9.
  with self.assertRaisesRegex(ValueError,'pairing mismatch'):auxiliary_losses(model,bodies,np.arange(8),**kwargs)
 def test_new_judge_seed_does_not_change_global_rng(self):
  model=small_model();state=torch.random.get_rng_state().clone();initialize_new_judge(model,31)
  self.assertTrue(torch.equal(state,torch.random.get_rng_state()));first={k:v.clone() for k,v in model.judge.state_dict().items()}
  initialize_new_judge(model,31);self.assertTrue(all(torch.equal(v,model.judge.state_dict()[k]) for k,v in first.items()))
  initialize_new_judge(model,32);self.assertTrue(any(not torch.equal(v,model.judge.state_dict()[k]) for k,v in first.items()))
 def test_warmstart_discards_old_judge_and_preserves_compatible_core(self):
  common=dict(d_model=256,n_heads=8,n_latent=128,n_latent_p=4,fsq_stages=1,fsq_levels=(32,)*13,use_sem=True,use_dof=True,contact_chan=True)
  reference=DualTranslator(enc_layers=4,dec_layers=4,n_judges=16,sem_dim=768,**common).state_dict()
  model=ConditionalMegatron(enc_layers=6,dec_layers=6,sem_dim=1024,**common)
  sem={k:v.clone() for k,v in model.state_dict().items() if k.startswith('jf.sem_proj.')}
  for k,v in reference.items():
   if k.startswith('judge.'):v.fill_(9.)
  report=load_reference_weights(model,reference,judge_body_order=ORDER,judge_init_seed=901)
  self.assertEqual(len(report['judge_dropped_reference_keys']),4);self.assertEqual(len(report['judge_new_keys']),4)
  self.assertEqual(len(report['semantic_projection_new_keys']),4)
  self.assertEqual(len(report['loaded_reference_keys']),375)
  self.assertTrue(all(torch.equal(model.state_dict()[k],reference[k]) for k in report['loaded_reference_keys']))
  self.assertTrue(all(torch.equal(model.state_dict()[k],v) for k,v in sem.items()))
  self.assertTrue(all(not torch.equal(model.state_dict()[k],reference[k]) for k in report['judge_new_keys']))
  counts=model_parameter_counts(model);self.assertEqual(counts['judge_auxiliary'],266512)
  self.assertEqual(counts['total'],counts['main_excluding_judge']+counts['judge_auxiliary'])
  self.assertEqual(model.judge_body_order,ORDER)
  print('\nPARAMETER_COUNTS',counts)

if __name__=='__main__':unittest.main(verbosity=2)
