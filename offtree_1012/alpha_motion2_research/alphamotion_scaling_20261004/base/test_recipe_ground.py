"""Offline numerical and source-supplement fixtures; no real data or training."""
import copy,json,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from recipe_ground import ground_losses
from seal_xwbc_static import read_contact_rows,build_ground_static,backfill_ground
from xwbc_contract import sha256,validate_xwbc
from test_pilot_data import fixture,CONTRACT,write


def batch():
 B,J=3,2
 rot=torch.tensor([1.,0,0,0,1,0]).repeat(B,J,1).requires_grad_()
 pos=torch.zeros(B,J,3);pos[:,1,1]=-3.;pos.requires_grad_()
 root=torch.tensor([.01,.08,float('nan')]);valid=torch.tensor([True,True,False])
 soft=torch.tensor([[.7,.9],[.2,.1],[float('nan'),float('nan')]])
 return rot,pos,dict(metric_length_valid=True,ground_geometry_verified=True,root_height_m=root,root_height_valid=valid,
  ground_valid=valid,contact_trust=valid,contact_soft=soft,contact_hard=torch.tensor([[True,True],[False,False],[False,False]]),
  contact_valid=valid[:,None].expand(B,J),sole_j=torch.tensor([1]),sole_off_m=torch.tensor([[.1,-.04,0]]),contact_threshold_m=torch.tensor(.04))


class GroundRecipeTests(unittest.TestCase):
 def test_original_geometric_formula_sole_rotation_and_weights(self):
  rot,pos,g=batch();out=ground_losses(rot,pos,None,g,w_rootpos=0)
  h=pos[:2,:,1]/100+g['root_height_m'][:2,None];h=h.clone();h[:,1]-=.04
  pen=torch.relu(-h).square().mean();cgeo=F.binary_cross_entropy_with_logits((.04-h)/(.04/3),g['contact_soft'][:2])
  torch.testing.assert_close(out['terms']['pen'],pen);torch.testing.assert_close(out['terms']['cgeo'],cgeo)
  torch.testing.assert_close(out['total'],200*pen+cgeo)
  # Rotating the foot changes the sole height, even with fixed joint centres.
  r=rot.detach().clone();r[:,1]=torch.tensor([0.,1,0,-1.,0,0])
  rotated=ground_losses(r,pos,None,g,w_rootpos=0)
  self.assertNotEqual(float(rotated['total'].detach()),float(out['total'].detach()))
  out['total'].backward();self.assertTrue(torch.isfinite(rot.grad).all());self.assertGreater(float(rot.grad.abs().sum()),0.)
 def test_direct_original_contact_terms_value_and_gradient_parity(self):
  import ast,types
  from common import rot6d_to_matrix
  source=Path(__file__).with_name('reference_recipe')/'source'/'dual_common.py'
  tree=ast.parse(source.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='contact_terms')
  namespace={'torch':torch,'rot6d_to_matrix':rot6d_to_matrix,'CONTACT_UP':1}
  exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),namespace)
  rot,pos,g=batch()
  store=types.SimpleNamespace(sole_j=g['sole_j'],sole_off=g['sole_off_m'],thr=.04,tau=.04/3,
   gap_root=lambda idx:g['root_height_m'][idx],trust=lambda idx:g['contact_trust'][idx],
   soft=lambda idx:g['contact_soft'][idx],hard=lambda idx:g['contact_hard'][idx])
  original={};namespace['contact_terms'](original,{'test':{'con_ok':True,'con':store}},'test',torch.arange(3),rot,fk_cm=pos)
  actual=ground_losses(rot,pos,None,g,w_rootpos=0)
  for key in ['pen','cgeo']:torch.testing.assert_close(actual['terms'][key],original[key],atol=1e-7,rtol=1e-6)
  old_loss=200*original['pen']+original['cgeo']
  old_grad=torch.autograd.grad(old_loss,(rot,pos));new_grad=torch.autograd.grad(actual['total'],(rot,pos))
  for old,new in zip(old_grad,new_grad):torch.testing.assert_close(old,new,atol=2e-6,rtol=2e-6)
 def test_invalid_nan_rows_have_no_gradient_and_root_is_independent(self):
  rot,pos,g=batch();rot=rot.detach();pos=pos.detach();rot[2]=float('nan');pos[2]=float('nan');rot.requires_grad_();pos.requires_grad_()
  pr=torch.tensor([.04,.05,float('nan')],requires_grad=True)
  out=ground_losses(rot,pos,pr,g);out['total'].backward()
  self.assertTrue(torch.isfinite(rot.grad).all());self.assertTrue(torch.isfinite(pos.grad).all());self.assertTrue(torch.isfinite(pr.grad).all())
  self.assertEqual(float(pr.grad[2]),0.)
  g['ground_valid']=torch.zeros(3,dtype=torch.bool)
  only=ground_losses(rot,pos,pr,g);self.assertEqual(set(only['terms']),{'rootpos'})
  no_double=ground_losses(rot,pos,pr,g,w_rootpos=0);self.assertEqual(no_double['terms'],{});self.assertEqual(float(no_double['total'].detach()),0.)
 def test_nonmetric_has_no_floor_or_fake_targets(self):
  rot,pos,g=batch();g['metric_length_valid']=False;g['ground_valid']=torch.zeros(3,dtype=torch.bool);g['root_height_valid']=torch.zeros(3,dtype=torch.bool);g['contact_valid']=torch.zeros((3,2),dtype=torch.bool)
  out=ground_losses(rot,pos,None,g);self.assertEqual(out['terms'],{});self.assertEqual(float(out['total'].detach()),0.)
  g['root_height_valid'][0]=True
  with self.assertRaisesRegex(ValueError,'nonmetric'):ground_losses(rot,pos,None,g)
 def test_missing_sole_table_does_not_fall_back_to_centres(self):
  with tempfile.TemporaryDirectory() as tmp:
   with self.assertRaisesRegex(ValueError,'original contact_label.py and sole_offsets3.json required'):
    build_ground_static('x',{'parents':[-1]},'a'*64,[],Path(tmp)/'missing.json',tmp)
 def test_completed_extraction_backfill_preserves_pose_and_restores_native_labels(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);manifest,row,arrays,parents,offsets,names=fixture(root)
   caches=[root/'original_lafan',root/'original_amass']
   split=json.loads((root/'chosen_split.json').read_text())
   for ci,c in enumerate(caches):
    c.mkdir();write(c/'manifest.json',{'fixture_cache':ci});split['source_sequence_files'][ci]['manifest_sha256']=sha256(c/'manifest.json')
    for name in CONTRACT['body_joints']:
     import shutil;shutil.copy2(root/'static'/f'{name}_spec.npz',c/f'{name}_spec.npz')
     gap=np.full((32,3),.1,np.float16);gap[0,1]=-.01
     soft=np.full((32,3),.25,np.float16);hard=np.zeros((32,3),bool);hard[3,2]=True
     hdr=dict(N=32,J=3,joint_names=names.tolist(),thr_m=.03+.01*ci,D_used_m=1.,h0_m=0.,sole_applied=[1],sole_joint_names=['left_wrist'])
     # Deliberately no trust field: old compatibility policy is all True.
     np.savez(c/f'{name}_contact.npz',header=json.dumps(hdr),gap=gap,contact_soft=soft,contact=hard)
   write(root/'chosen_split.json',split);manifest['chosen_split_sha256']=sha256(root/'chosen_split.json')
   manifest['static_seal']['source_cache_manifest_sha256']=split['source_sequence_files'][0]['manifest_sha256'];write(root/'manifest.json',manifest)
   sole=root/'sole_offsets3.json';write(sole,{name:{'a':[dict(body='left_wrist',off_ardy=[.02,-.03,0])]} for name in CONTRACT['body_joints']})
   source=root/'source';source.mkdir();(source/'contact_label.py').write_text("""SOLE_ALIAS = {}
def sole_offsets(emb,spec):
 with open(SOLE_JSON) as f:tbl=json.load(f)
 out={};names=list(spec.joint_names)
 for side in ('a','b'):
  for c in tbl.get(SOLE_ALIAS.get(emb,emb),{}).get(side,[]):
   if c['body'] in names:
    out[names.index(c['body'])]=np.asarray(c['off_ardy'],np.float32);break
 return out
""")
   before=arrays['pose9'].copy();result=backfill_ground(root,caches,source,sole,contract=CONTRACT)
   self.assertEqual(result['status'],'complete');validate_xwbc(root,contract=CONTRACT)
   with np.load(root/'payloads/body_a.npz') as z:
    np.testing.assert_array_equal(z['pose9'],before);self.assertTrue(z['ground_valid'].all())
    self.assertTrue(z['contact_hard'][0,1]);self.assertEqual(float(z['contact_soft'][0,1]),1.)
    self.assertTrue(z['contact_hard'][1,2]);self.assertEqual(float(z['contact_soft'][1,2]),.25)
   ground=json.loads((root/'static/body_a_ground.json').read_text());self.assertEqual(ground['sole_j'],[1]);self.assertEqual(ground['contact_threshold_m'],.04)
   self.assertEqual(ground['contact_header_thresholds_m'],[.03,.04])

if __name__=='__main__':unittest.main()
