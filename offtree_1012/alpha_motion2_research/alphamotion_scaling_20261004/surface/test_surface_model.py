"""CPU synthetic numerical tests only; never measured surface or grasp evidence."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import json,unittest
import numpy as np
import torch
from torch.nn import functional as F
from surface_model import SurfaceMegatron,ConditionalMegatron
from surface_contract import prepare_static_surface,expand_surface,scale_surface_units
from surface_geometry import fk_global_positions,bound_surface,predict_bound_surface,surface_losses,surface_metrics

torch.set_num_threads(1)
METRICS={}

def payload(joints=4,patches=8):
 rng=np.random.default_rng(5);anchor=np.arange(patches)%joints
 local=rng.normal(size=(patches,3)).astype(np.float32)*.15
 normal=np.tile([0.,1.,0.],(patches,1)).astype(np.float32)
 sem=rng.normal(size=(patches,1024)).astype(np.float32);sem/=np.linalg.norm(sem,axis=-1,keepdims=True)
 return dict(schema_version='stable_surface_v1',canonical_up='Y',scale_policy='geometry_only',binding_kind='rigid',
 evidence_kind='synthetic_test',geometry_hashes={'fixture':'0'*64},patch_ids=[f'fixture:face:{i}:bary:1/3' for i in range(patches)],
 valid=np.ones(patches,bool),rest_xyz=local+np.stack([anchor,np.zeros(patches),np.zeros(patches)],-1),rest_normal=normal,
 area=np.ones(patches,np.float32)*.02,anchor_joint=anchor,local_xyz=local,local_normal=normal,
 qwen_sem=sem,semantic_valid=np.ones(patches,bool),geometry_scale=3.,pose_scale=3.,metric_valid=True,meters_per_unit=.01)

def geometry(batch=2,joints=4,with_surface=True):
 parents=np.arange(joints,dtype=np.int64)-1;off=torch.zeros(batch,joints,3);off[:,1:,0]=1.
 depth=torch.arange(joints)[None].expand(batch,-1)
 g=dict(rest_off=off,kinematic_rest_off=off,bonelen=off.norm(dim=-1)/(joints-1),depth=depth,hop=(depth[:,:,None]-depth[:,None,:]).abs(),
 sem=F.normalize(torch.randn(batch,joints,1024),dim=-1),dof=torch.zeros(batch,joints,19),metadata_missing=torch.zeros(batch,joints,2),parents=parents,pose_scale=torch.full((batch,),3.))
 if with_surface:g['surface']=expand_surface(prepare_static_surface(payload(joints),joints),batch)
 return g

def motion(batch,joints,g):
 rot=torch.randn(batch,joints,6);pos=fk_global_positions(rot,g['parents'],g['rest_off'])
 return torch.cat([rot,pos/g['pose_scale'][:,None,None]],-1)

def model():return SurfaceMegatron(d_model=32,n_heads=4,enc_layers=2,dec_layers=2,n_latent=8,n_latent_p=4,fsq_stages=1,fsq_levels=(32,)*13,sem_dim=1024,n_judges=12)

class SurfaceNumerics(unittest.TestCase):
 def setUp(self):
  torch.manual_seed(123);self.m=model().eval();self.g=geometry();self.x=motion(2,4,self.g)
 def test_fixed_rotation_fk_and_rigid_mesh_transform(self):
  rot=torch.tensor([1.,0.,0.,0.,1.,0.]).expand(2,4,6)
  pos=fk_global_positions(rot,self.g['parents'],self.g['rest_off']);surface=bound_surface(rot,pos,self.g['surface'])
  anchor=self.g['surface']['binding_weights'].argmax(-1)
  expected=torch.gather(pos,1,anchor[...,None].expand(-1,-1,3))+(self.g['surface']['binding_weights'][...,None]*self.g['surface']['bind_local_xyz']).sum(2)
  self.assertTrue(torch.allclose(surface['position'],expected,atol=1e-7));self.assertTrue(torch.allclose(surface['normal'][...,1],torch.ones_like(surface['normal'][...,1])))
 def test_forward_fsq_shapes_no_surface_coordinate_head(self):
  out=self.m(self.x,self.g,self.g);zp,zr,codes=self.m.encode(self.x,self.g,return_codes=True)
  self.assertEqual(out[0].shape,(2,4,6));self.assertEqual(zp.shape,(2,4,32));self.assertEqual(zr.shape,(2,8,32))
  self.assertEqual(codes[0].shape,(2,4,13));self.assertTrue(all(torch.isfinite(v).all() for v in out))
  self.assertFalse(any('surface_head' in n for n,_ in self.m.named_parameters()))
  bound=predict_bound_surface(out[0],self.g);self.assertEqual(bound['surface']['position'].shape,(2,8,3))
 def test_surface_losses_train_geometry_text_motion_attention_and_core(self):
  rot,raw,con=self.m(self.x,self.g,self.g)
  losses=surface_losses(rot,self.x[...,:6],self.g)
  losses['total'].backward();norms={}
  for name in ['patch_geometry','patch_semantic','patch_motion','target_surface_attention','inp','hrot']:
   module=getattr(self.m,name);grads=[p.grad for p in module.parameters() if p.grad is not None]
   self.assertTrue(grads,name);self.assertTrue(all(torch.isfinite(g).all() for g in grads));norms[name]=sum(float(g.abs().sum()) for g in grads);self.assertGreater(norms[name],0.)
  self.assertFalse(self.g['surface']['qwen_sem'].requires_grad)
  METRICS['surface_loss_gradient_l1']=norms
 def test_geometry_and_semantic_perturbations_and_drop_ablation_affect_output(self):
  with torch.no_grad():
   base=self.m(self.x,self.g,self.g)[0]
   changes={}
   for field in ['rest_xyz','qwen_sem']:
    g=dict(self.g);g['surface']=dict(self.g['surface']);v=g['surface'][field].clone()
    if field=='rest_xyz':v=v*1.8+1.
    else:v=v.roll(137,-1)
    g['surface'][field]=v;new=self.m(self.x,g,g)[0];changes[field]=float((base-new).abs().max());self.assertGreater(changes[field],1e-5)
   for mode in ['drop_surface','static_only','no_patch_semantics','no_attachment_bias']:
    with self.m.ablate_surface(mode):new=self.m(self.x,self.g,self.g)[0]
    changes[mode]=float((base-new).abs().max());self.assertGreater(changes[mode],1e-5)
  METRICS['ablation_max_abs_output_delta']=changes
 def test_hidden_dynamic_surface_does_not_leak_ground_truth(self):
  hidden=torch.zeros(2,4,dtype=torch.bool);hidden[:,[1,3]]=True
  changed=self.x.clone();changed[hidden]=float('nan')
  with torch.no_grad():
   a=self.m(self.x,self.g,self.g,source_hidden=hidden)[0];b=self.m(changed,self.g,self.g,source_hidden=hidden)[0]
  self.assertTrue(torch.allclose(a,b,atol=1e-6));METRICS['hidden_source_nan_delta']=float((a-b).abs().max())
  x=self.x.clone().requires_grad_();self.m(x,self.g,self.g,source_hidden=hidden)[0].square().mean().backward()
  self.assertEqual(float(x.grad[hidden].abs().max()),0.)
 def test_pure_sparse_has_no_dynamic_surface_gt_leak(self):
  hidden=torch.ones(2,4,dtype=torch.bool)
  with torch.no_grad():
   a=self.m(self.x,self.g,self.g,source_hidden=hidden)[0];b=self.m(torch.full_like(self.x,float('nan')),self.g,self.g,source_hidden=hidden)[0]
  self.assertTrue(torch.allclose(a,b,atol=1e-6))
 def test_patch_order_permutation_preserves_prediction(self):
  g=dict(self.g);s=dict(g['surface']);perm=torch.tensor([7,2,6,1,5,0,4,3])
  for k in ['valid','rest_xyz','rest_normal','area','binding_weights','bind_local_xyz','bind_local_normal','qwen_sem','semantic_valid']:s[k]=s[k][:,perm]
  s['patch_ids']=tuple(s['patch_ids'][int(i)] for i in perm);g['surface']=s
  with torch.no_grad():a=self.m(self.x,self.g,self.g)[0];b=self.m(self.x,g,g)[0]
  self.assertTrue(torch.allclose(a,b,atol=3e-6,rtol=3e-6));METRICS['patch_permutation_max_abs']=float((a-b).abs().max())
 def test_missing_surface_matches_skeleton_model_shared_weights(self):
  a=ConditionalMegatron(d_model=32,n_heads=4,enc_layers=2,dec_layers=2,n_latent=8,n_latent_p=4,fsq_stages=1,fsq_levels=(32,)*13,sem_dim=1024,n_judges=12).eval()
  a.load_state_dict({k:v for k,v in self.m.state_dict().items() if k in a.state_dict()},strict=True)
  with torch.no_grad(),self.m.ablate_surface('drop_surface'):
   x=self.m(self.x,self.g,self.g)[0];y=a(self.x,self.g,self.g)[0]
  self.assertTrue(torch.allclose(x,y,atol=1e-6))
 def test_all_invalid_surface_is_finite_and_has_zero_aux_loss(self):
  g=dict(self.g);g['surface']=dict(g['surface']);g['surface']['valid']=torch.zeros_like(g['surface']['valid'])
  out=self.m(self.x,g,g)[0];loss=surface_losses(out,self.x[...,:6],g)
  self.assertTrue(torch.isfinite(out).all());self.assertEqual(float(loss['total'].detach()),0.)
 def test_unknown_ground_contact_cannot_be_negative_labels(self):
  rot=self.m(self.x,self.g,self.g)[0];rh=torch.full((2,),float('nan'),requires_grad=True);rv=torch.zeros(2,dtype=torch.bool)
  out=surface_losses(rot,self.x[...,:6],self.g,root_height_m=rh,root_height_valid=rv,
   contact_targets=torch.full((2,8),float('nan')),contact_valid=torch.zeros(2,8,dtype=torch.bool),contact_evidence_kind='cache_geometry_derived')
  self.assertEqual(float(out['contact'].detach()),0.);self.assertEqual(float(out['penetration'].detach()),0.)
  out['total'].backward();self.assertTrue(torch.isfinite(rh.grad).all());self.assertEqual(float(rh.grad.abs().sum()),0.)
 def test_valid_ground_contact_has_root_and_rotation_gradient(self):
  rot=self.m(self.x,self.g,self.g)[0];rot.retain_grad();rh=torch.full((2,),-.02,requires_grad=True)
  out=surface_losses(rot,self.x[...,:6],self.g,root_height_m=rh,root_height_valid=torch.ones(2,dtype=torch.bool),
   contact_targets=torch.ones(2,8),contact_valid=torch.ones(2,8,dtype=torch.bool),contact_evidence_kind='simulated',contact_partner_valid=torch.ones(2,dtype=torch.bool),contact_provenance={'source_sha256':'0'*64,'partner_kind':'ground_plane'},w_position=0.,w_normal=0.)
  out['total'].backward();self.assertGreater(float(rh.grad.abs().sum()),0.);self.assertGreater(float(rot.grad.abs().sum()),0.)
  self.assertEqual(out['diagnostics']['contact_points'],16)
 def test_prismatic_and_holdout_fitted_scale_rejected(self):
  p=payload();p['joint_motion_kind']='prismatic'
  with self.assertRaisesRegex(ValueError,'prismatic'):prepare_static_surface(p,4)
  p=payload();p.update(heldout_body=True,scale_policy='train_only')
  with self.assertRaisesRegex(ValueError,'geometry-only'):prepare_static_surface(p,4)
 def test_contact_labels_without_verified_partner_rejected(self):
  rot=self.m(self.x,self.g,self.g)[0]
  with self.assertRaisesRegex(ValueError,'source/partner'):
   surface_losses(rot,self.x[...,:6],self.g,root_height_m=torch.zeros(2),root_height_valid=torch.ones(2,dtype=torch.bool),contact_targets=torch.ones(2,8),contact_valid=torch.ones(2,8,dtype=torch.bool),contact_evidence_kind='simulated')
 def test_descriptor_offsets_cannot_silently_replace_kinematic_offsets(self):
  g=dict(self.g);del g['kinematic_rest_off']
  with self.assertRaisesRegex(ValueError,'kinematic_rest_off'):predict_bound_surface(self.x[...,:6],g)
 def test_surface_cost_reports_real_extra_tokens(self):
  cost=self.m.surface_cost(self.g,self.g,2);self.assertEqual(cost['encoder_tokens'],12);self.assertEqual(cost['target_patch_query_cells'],32);METRICS['fixture_cost']=cost
 def test_explicit_units_keep_common_A_B_metrics_and_losses_invariant(self):
  prediction=self.x[...,:6]+.1
  native=surface_metrics(prediction,self.x[...,:6],self.g)
  g=dict(self.g);g['surface']=scale_surface_units(g['surface'],100.,destination_unit='centi-native')
  g['kinematic_rest_off']=g['kinematic_rest_off']*100;g['pose_scale']=g['pose_scale']*100
  converted=surface_metrics(prediction,self.x[...,:6],g)
  for key in ('position_m_per_frame','position_normalized_per_frame','normal_cosine_error_per_frame'):
   self.assertTrue(torch.allclose(native[key],converted[key],atol=1e-6),key)
  self.assertTrue(torch.allclose(native['position_native_per_frame']*100,converted['position_native_per_frame'],atol=1e-5))
  self.assertEqual(native['active_patches'],16)
  l1=surface_losses(prediction,self.x[...,:6],self.g)['total'];l2=surface_losses(prediction,self.x[...,:6],g)['total']
  self.assertTrue(torch.allclose(l1,l2,atol=1e-6))
 def test_common_metrics_unknown_metric_and_invalid_patch_counts(self):
  g=dict(self.g);g['surface']=dict(g['surface']);g['surface']['metric_valid']=torch.zeros(2,dtype=torch.bool)
  g['surface']['meters_per_unit']=torch.full((2,),float('nan'));g['surface']['valid']=g['surface']['valid'].clone();g['surface']['valid'][0]=False
  r=surface_metrics(self.x[...,:6],self.x[...,:6],g)
  self.assertEqual(r['valid_patch_count_per_frame'].tolist(),[0,8]);self.assertTrue(torch.isnan(r['position_m_per_frame']).all());self.assertTrue(torch.isnan(r['position_normalized_per_frame'][0]));self.assertEqual(float(r['position_normalized_per_frame'][1]),0.)

if __name__=='__main__':
 suite=unittest.defaultTestLoader.loadTestsFromTestCase(SurfaceNumerics)
 result=unittest.TextTestRunner(verbosity=2).run(suite)
 report={'passed':result.wasSuccessful(),'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'metrics':METRICS,
 'evidence_scope':'CPU synthetic mesh patches, synthetic Qwen unit vectors and random model weights. Diagnostic backward only; no optimizer, real data, grasp or physical execution.'}
 (HERE/'numerical_review.json').write_text(json.dumps(report,indent=2)+'\n')
 sys.exit(not result.wasSuccessful())
