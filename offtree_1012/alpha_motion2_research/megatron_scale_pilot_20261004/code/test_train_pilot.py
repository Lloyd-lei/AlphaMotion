#!/usr/bin/env python3
"""Local CPU interface tests. Every Body, motion, descriptor and text vector is synthetic.
No real pilot training/data/weights, network, SSH or CUDA; a three-update tiny optimizer
probe is explicitly scoped to synthetic data and is not a pilot result.
"""
from __future__ import annotations
import ast,datetime,hashlib,inspect,json,os,sys,tempfile,time,unittest
from pathlib import Path
from unittest import mock
os.environ['PYTHONDONTWRITEBYTECODE']='1';os.environ['CUDA_VISIBLE_DEVICES']='';sys.dont_write_bytecode=True
import numpy as np
import torch
import torch.nn.functional as F
HERE=Path(__file__).resolve().parent
LEGACY=Path(os.environ.get('PILOT_LEGACY_SOURCE',str(HERE/'legacy')))
sys.path.insert(0,str(LEGACY));sys.path.insert(0,str(HERE))
SOURCE_HASHES={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/'train_pilot.py',HERE/'pilot_data.py',HERE/'pilot_model.py']}
import pilot_data as data
import train_pilot as trainer
from pilot_model import ConditionalMegatron
from common import matrix_to_rot6d

torch.set_num_threads(1);METRICS={};EVALUATION_CAPTURE={}
WEIGHTS=dict(position=5.,condition_position=10.,condition_rotation=1.,root_height=3.)

def small_model():
 return ConditionalMegatron(d_model=32,n_heads=4,enc_layers=2,dec_layers=2,n_latent=4,n_latent_p=2,fsq_stages=1,fsq_levels=(8,8,8,5,5,5),sem_dim=1024,use_sem=True,use_dof=True,n_judges=16,contact_chan=True).cpu()

def synthetic_body(name,domain,joints=5,holdout=False):
 N=12;metric=domain=='xwbc';reach=100. if metric else 1.5
 if joints==5:parents=np.array([-1,0,1,0,3]);names=['root','left_arm','left_wrist','right_arm','right_wrist']
 elif joints==6:parents=np.array([-1,0,1,0,3,0]);names=['root','node1','node2','node3','node4','node5']
 else:parents=np.array([-1,0,0,0]);names=['root','part1','part2','part3']
 offsets=np.zeros((joints,3),np.float32)
 for j in range(1,joints):offsets[j]=np.array([(-1 if j%2 else 1)*.10,.12,.025],np.float32)*reach
 angle=torch.arange(N)[:,None]*.04+torch.arange(joints)[None,:]*.025
 co,si=torch.cos(angle),torch.sin(angle);R=torch.zeros(N,joints,3,3)
 R[...,0,0]=co;R[...,0,2]=si;R[...,1,1]=1;R[...,2,0]=-si;R[...,2,2]=co
 rot=matrix_to_rot6d(R).numpy().astype('float32')
 pos=data.fk(torch.from_numpy(rot),parents,offsets).numpy()/reach
 pose=np.concatenate([rot,pos],-1).astype('float32')
 split=np.full(N,2,np.int64) if holdout else np.array([0]*6+[1]*6,np.int64)
 groups=np.repeat(np.arange(4),3)
 hvalid=(np.arange(N)%2==0) if metric else np.zeros(N,bool)
 height=np.where(hvalid,.8+np.arange(N)*.01,np.nan).astype('float32')
 valid=np.full((N,joints),metric,bool);valid[:,0]=False
 contact=np.where(valid,np.sin(np.arange(N)[:,None]+np.arange(joints)[None,:])*.1,np.nan).astype('float32')
 arrays=dict(pose9=pose,global_rot6=rot,split=split,group=groups,root_height_valid=hvalid,root_height_m=height,contact_valid=valid,contact_sdist=contact,pretrain_seen=np.arange(N)%2==0)
 body=data.Body(name,domain,arrays,parents,offsets,offsets.copy(),names,reach,np.zeros((joints,19),np.float32),np.full(joints,metric,bool),metric,{'fixture':'SYNTHETIC ONLY; not loaded from source motion or Qwen cache'})
 cache={n:F.normalize(torch.randn(1024),dim=0) for n in names};body.setup(cache,torch.device('cpu'))
 return body

def fixtures():
 return [synthetic_body('human_smpl','xwbc'),synthetic_body('unitree_g1_29dof','xwbc'),synthetic_body('synthetic_unimate_seen','unimate',6),synthetic_body('synthetic_unimate_holdout','unimate',4,True)]

def config():return dict(seed=31,eval=dict(quick_frames=4,frames_per_body=4,batch_size=2),loss=WEIGHTS)

def assert_gradients(test,model):
 grads=[p.grad for p in model.parameters() if p.grad is not None]
 test.assertTrue(grads);test.assertTrue(all(bool(torch.isfinite(x).all()) for x in grads))
 return len(grads)

class TrainerContracts(unittest.TestCase):
 def setUp(self):
  torch.manual_seed(2026);self.bodies=fixtures();self.model=small_model();self.cpu=torch.device('cpu')

 def test_body_fk_consistency_and_missingness(self):
  errors={b.name:b.audit_fk() for b in self.bodies}
  self.assertEqual(max(errors.values()),0.)
  for b in self.bodies:
   if b.domain=='unimate':
    self.assertTrue(bool((b.static['metadata_missing']==1).all()))
    self.assertFalse(b.metric);self.assertFalse(bool(b.arrays['root_height_valid'].any()))
   else:self.assertTrue(bool((b.static['metadata_missing']==0).all()))
  METRICS['synthetic_body_fk']=dict(max_error_over_reach=max(errors.values()),body_count=len(self.bodies),all_data_synthetic=True)

 def test_objective_uses_fk_not_auxiliary_xyz(self):
  b=self.bodies[2];rows=np.array([0,1,2]);target=b.tensor('pose9',rows,self.cpu)
  condition=dict(joint_indices=torch.full((3,2),-1,dtype=torch.long),valid=torch.zeros(3,2,dtype=torch.bool),values=torch.full((3,2,9),float('nan')))
  rot=target[...,:6].clone().requires_grad_(True)
  # Deliberately nonsensical aux XYZ: position term must still be exactly FK truth.
  aux=torch.full((3,len(b.parents),3),12345.,requires_grad=True)
  loss,terms=trainer.objective((rot,aux,None),target,b,rows,condition,WEIGHTS)
  self.assertEqual(float(terms['position_over_reach'].detach()),0.)
  self.assertNotIn('root_height_smooth_l1_m',terms)
  loss.backward();self.assertTrue(bool(torch.isfinite(rot.grad).all()));self.assertIsNone(aux.grad)
  perturbed=rot.detach().clone();perturbed[:,0]=torch.tensor([0.,0.,-1.,0.,1.,0.])
  _,changed=trainer.objective((perturbed,aux,None),target,b,rows,condition,WEIGHTS)
  self.assertGreater(float(changed['position_over_reach']),.01)
  METRICS['fk_loss_contract']=dict(auxiliary_not_used_for_xyz=True,matching_fk_position_error=0.,rotating_parent_changes_position_error=float(changed['position_over_reach']))

 def test_root_height_mask_removes_invalid_nan_before_loss(self):
  b=self.bodies[1];rows=np.array([0,1,2,3]);b.arrays['root_height_valid'][2]=True;b.arrays['root_height_m'][2]=np.nan
  target=b.tensor('pose9',rows,self.cpu);aux=torch.full((4,5,3),float('nan'))
  aux[0,0,0]=.2;aux.requires_grad_(True)
  condition=dict(joint_indices=torch.full((4,2),-1,dtype=torch.long),valid=torch.zeros(4,2,dtype=torch.bool),values=torch.full((4,2,9),float('nan')))
  loss,terms=trainer.objective((target[...,:6].clone().requires_grad_(True),aux,None),target,b,rows,condition,WEIGHTS)
  expected=F.smooth_l1_loss(aux[0,0,0],torch.tensor(float(b.arrays['root_height_m'][0])))
  torch.testing.assert_close(terms['root_height_smooth_l1_m'],expected)
  loss.backward();self.assertTrue(bool(torch.isfinite(aux.grad).all()))
  mask=torch.ones_like(aux,dtype=torch.bool);mask[0,0,0]=False;self.assertEqual(float(aux.grad[mask].abs().max()),0.)
  METRICS['root_height_mask']=dict(valid_flag_plus_finite_height=True,invalid_aux_gradient_max_abs=0.,loss=float(loss.detach()))

 def test_all_domains_modes_dropout_and_finite_gradients(self):
  results=[]
  for body in [self.bodies[1],self.bodies[2]]:
   for mode in ['full','masked','sparse_only']:
    rows=data.sample_rows(body,np.random.default_rng(10),3)
    for c in [0,1,2]:
     self.model.zero_grad(set_to_none=True)
     inputs,target=data.make_inputs(body,body,rows,np.random.default_rng(99),self.cpu,mode,c,contact_dropout=0.)
     self.assertEqual(int(inputs['condition']['valid'].sum()),len(rows)*c)
     if mode=='sparse_only':self.assertTrue(bool(inputs['source_hidden'].all()));self.assertIsNone(inputs['source_contact'])
     else:self.assertTrue(bool(torch.isnan(inputs['source_contact'][inputs['source_hidden']]).all()))
     out=trainer.forward(self.model,body,body,inputs);loss,terms=trainer.objective(out,target,body,rows,inputs['condition'],WEIGHTS)
     self.assertTrue(bool(torch.isfinite(loss)));loss.backward();n=assert_gradients(self,self.model)
     results.append(dict(domain=body.domain,mode=mode,count=c,loss=float(loss.detach()),finite_gradient_tensors=n))
    drop,_=data.make_inputs(body,body,rows,np.random.default_rng(99),self.cpu,mode,None,condition_dropout=1.,contact_dropout=1.)
    self.assertFalse(bool(drop['condition']['valid'].any()))
    if drop['source_contact'] is not None:self.assertTrue(bool(torch.isnan(drop['source_contact']).all()))
  METRICS['domains_modes']=results

 def test_explicit_condition_counts_share_source_rows_mask_and_contact(self):
  for body in [self.bodies[1],self.bodies[2]]:
   for mode in ['full','masked','sparse_only']:
    rows=np.array([6,7,8,9]);items=[]
    for c in [0,1,2]:
     i,t=data.make_inputs(body,body,rows,np.random.default_rng(121),self.cpu,mode,c,contact_dropout=1.);items.append((i,t))
    for i,t in items[1:]:
     self.assertTrue(torch.equal(i['source'],items[0][0]['source']));self.assertTrue(torch.equal(i['source_hidden'],items[0][0]['source_hidden']));self.assertTrue(torch.equal(t,items[0][1]))
     if i['source_contact'] is not None:torch.testing.assert_close(i['source_contact'],items[0][0]['source_contact'],equal_nan=True)
    # Nested prefix prevents varying evaluation anchor selection when adding the second condition.
    self.assertTrue(torch.equal(items[1][0]['condition']['joint_indices'][:,0],items[2][0]['condition']['joint_indices'][:,0]))
  METRICS['same_rows_source_mask']=dict(exact=True,condition_endpoint_prefix_nested=True)

 def test_pure_condition_interface_has_no_hidden_contact_or_pose_dependency(self):
  b=self.bodies[2];rows=np.array([0,1,2]);inputs,target=data.make_inputs(b,b,rows,np.random.default_rng(7),self.cpu,'sparse_only',2)
  self.assertIsNone(inputs['source_contact']);self.assertTrue(bool(inputs['source_hidden'].all()))
  self.model.eval()
  with torch.no_grad():
   a=trainer.forward(self.model,b,b,inputs);inputs['source']=torch.full_like(inputs['source'],float('nan'));z=trainer.forward(self.model,b,b,inputs)
  self.assertTrue(all(torch.equal(x,y) for x,y in zip(a,z)))
  METRICS['pure_condition_input']=dict(source_hidden_all=True,contact_none=True,output_max_abs_change=0.)

 def test_preflight_profiles_proposed_batch_largest_topology_cpu_synthetic(self):
  # Call the production helper on explicitly synthetic CPU fixtures only.
  # The helper's real_batch_preflight.json exists only inside this temporary directory.
  small=synthetic_body('synthetic_unimate_smaller_train','unimate',4)
  bodies=[self.bodies[0],self.bodies[1],small,*self.bodies[2:]]
  cfg=config();cfg['batch_size']=4;cfg['optimizer']={'clip_norm':1.0}
  before={n:p.detach().clone() for n,p in self.model.named_parameters()}
  with tempfile.TemporaryDirectory(prefix='SYNTHETIC-ONLY-preflight-cpu-') as tmp:
   with mock.patch.object(trainer,'forward',wraps=trainer.forward) as forward_spy, mock.patch.object(torch.nn.utils,'clip_grad_norm_',wraps=torch.nn.utils.clip_grad_norm_) as clip_spy:
    report=trainer.preflight(self.model,bodies,cfg,Path(tmp))
   saved=json.loads((Path(tmp)/'real_batch_preflight.json').read_text());self.assertEqual(report,saved)
  self.assertEqual(report['status'],'passed');self.assertEqual(report['optimizer_steps'],0)
  self.assertEqual(report['gpu'],'cpu');self.assertTrue(report['hidden_source_perturbation_invariant'])
  self.assertEqual(len(report['real_payload_batch_checks']),6)
  profiles=report['proposed_batch_profiles'];self.assertEqual(len(profiles),2)
  for profile,call in zip(profiles,forward_spy.call_args_list[-2:]):
   eligible=[b for b in bodies if b.domain==profile['domain'] and len(b.eligible(True))]
   maximum=max(len(b.parents) for b in eligible)
   self.assertEqual(profile['joints'],maximum);self.assertEqual(profile['batch_size'],4)
   self.assertEqual(profile['optimizer_steps'],0);self.assertIsNone(profile['peak_cuda_allocated_bytes'])
   self.assertTrue(np.isfinite(profile['gradient_norm']));self.assertGreaterEqual(profile['forward_backward_seconds'],0.)
   self.assertEqual(call.args[2].name,profile['body']);self.assertEqual(tuple(call.args[3]['source'].shape[:2]),(4,maximum))
  self.assertEqual(clip_spy.call_count,2)
  self.assertTrue(all(call.args[1]==1.0 and call.kwargs['error_if_nonfinite'] for call in clip_spy.call_args_list))
  self.assertTrue(all(torch.equal(before[n],p) for n,p in self.model.named_parameters()))
  self.assertTrue(all(p.grad is None for p in self.model.parameters()))
  METRICS['preflight_synthetic_cpu_profile']=dict(profiles=profiles,proposed_batch=4,clip_norm=1.0,optimizer_steps=0,all_parameters_unchanged=True,temporary_report_roundtrip=True,real_payloads_used=False,remote_cuda_profile_validated=False)

 def test_three_synthetic_optimizer_steps_are_finite(self):
  opt=torch.optim.AdamW(self.model.parameters(),lr=1e-4);initial=self.model.inp.weight.detach().clone();logs=[]
  for step,(domain,mode,count) in enumerate([('xwbc','full',0),('unimate','masked',2),('xwbc','sparse_only',1)],1):
   tgt=self.bodies[1] if domain=='xwbc' else self.bodies[2];src=self.bodies[0] if domain=='xwbc' and mode=='full' else tgt
   rows=data.sample_rows(tgt,np.random.default_rng(step),3)
   inputs,target=data.make_inputs(src,tgt,rows,np.random.default_rng(step+99),self.cpu,mode,count)
   opt.zero_grad(set_to_none=True);out=trainer.forward(self.model,src,tgt,inputs);loss,terms=trainer.objective(out,target,tgt,rows,inputs['condition'],WEIGHTS)
   loss.backward();assert_gradients(self,self.model);grad=torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.,error_if_nonfinite=True);opt.step()
   logs.append(dict(step=step,domain=domain,mode=mode,count=count,loss=float(loss.detach()),gradient_norm=float(grad)))
  change=float((self.model.inp.weight.detach()-initial).abs().max());self.assertGreater(change,0.)
  METRICS['synthetic_optimizer_probe']=dict(steps=3,logs=logs,parameter_max_abs_change=change,real_pilot_training=False)

 def test_evaluate_persists_readable_metrics_and_same_conditions(self):
  records=[];original=trainer.make_inputs
  def spy(src,tgt,rows,rng,device,mode,condition_count,*args,**kwargs):
   inputs,target=original(src,tgt,rows,rng,device,mode,condition_count,*args,**kwargs)
   records.append(dict(src=src.name,tgt=tgt.name,mode=mode,count=condition_count,rows=np.asarray(rows).copy(),source=inputs['source'].clone(),hidden=inputs['source_hidden'].clone(),target=target.clone(),ci=inputs['condition']['joint_indices'].clone(),cv=inputs['condition']['valid'].clone()))
   return inputs,target
  self.model.train()
  with tempfile.TemporaryDirectory(prefix='synthetic-trainer-eval-') as tmp:
   output=Path(tmp)
   with mock.patch.object(trainer,'make_inputs',side_effect=spy):report=trainer.evaluate(self.model,self.bodies,config(),output,'synthetic_test',quick=False)
   self.assertTrue(self.model.training)
   saved=json.loads((output/'evaluation_synthetic_test.json').read_text());self.assertEqual(report,saved)
   self.assertEqual(len(report['cases']),36)
   by={}
   for r in records:by.setdefault((r['src'],r['tgt'],r['mode'],tuple(r['rows'])),{})[r['count']]=r
   for trio in by.values():
    self.assertEqual(set(trio),{0,1,2})
    for count in [1,2]:
     for field in ['source','hidden','target']:self.assertTrue(torch.equal(trio[0][field],trio[count][field]))
   example_paths=list(output.glob('*examples.npz'));self.assertEqual(len(example_paths),36)
   for p in example_paths:
    with np.load(p,allow_pickle=False) as d:
     self.assertTrue(np.isfinite(d['predicted_fk_positions']).all());self.assertTrue(np.isfinite(d['predicted_rot6']).all())
     self.assertTrue(np.isnan(d['condition_values'][~d['condition_valid']]).all())
   for case in report['cases']:
    self.assertEqual(len(case['rows']),4)
    if case['condition_count']==0:self.assertIsNone(case['metrics']['selected_position_over_reach'])
    if case['domain']=='xwbc':
     n=case['metrics']['old_pretrain_seen_position_over_reach']['count']+case['metrics']['old_excluded_motion_position_over_reach']['count'];self.assertEqual(n,case['metrics']['all_position_over_reach']['count'])
   EVALUATION_CAPTURE['report']=report
   EVALUATION_CAPTURE['error_arrays']={p.name:{k:v.copy() for k,v in np.load(p,allow_pickle=False).items()} for p in output.glob('*errors.npz')}
   METRICS['evaluate_io']=dict(cases=len(report['cases']),same_row_mask_batches=len(by),example_npz_files=len(example_paths),error_npz_files=len(EVALUATION_CAPTURE['error_arrays']),all_npz_read_allow_pickle_false=True,model_train_mode_restored=True,synthetic_only=True)
   case_map={(c['source'],c['target'],c['mode'],c['condition_count']):c for c in report['cases']}
   paired=[]
   for c in report['cases']:
    if c['condition_count']==0:continue
    base=case_map[(c['source'],c['target'],c['mode'],0)]
    z=EVALUATION_CAPTURE['error_arrays'][base['error_file']];v=EVALUATION_CAPTURE['error_arrays'][c['error_file']]
    np.testing.assert_array_equal(z['rows'],v['rows']);mask=v['selected']
    self.assertEqual(int(mask.sum()),len(v['rows'])*c['condition_count'])
    self.assertFalse(bool(z['selected'].any()))
    self.assertAlmostEqual(float(v['position_over_reach'][mask].mean()),c['metrics']['selected_position_over_reach']['mean'],places=6)
    self.assertTrue(np.isfinite(z['position_over_reach'][mask]).all())
    if c['condition_count']==2:
     one=case_map[(c['source'],c['target'],c['mode'],1)]
     one_mask=EVALUATION_CAPTURE['error_arrays'][one['error_file']]['selected']
     self.assertTrue(bool(np.all(~one_mask|mask)))
    paired.append(dict(target=c['target'],mode=c['mode'],count=c['condition_count'],matched_samples=int(mask.sum()),baseline_selected_position=float(z['position_over_reach'][mask].mean()),conditioned_selected_position=float(v['position_over_reach'][mask].mean()),baseline_remaining_position=float(z['position_over_reach'][~mask].mean()),conditioned_remaining_position=float(v['position_over_reach'][~mask].mean())))
   METRICS['paired_metric_reconstruction']=dict(case_count=len(paired),baseline_uses_conditioned_mask=True,same_reference_rows=True,nested_condition_prefix=True,values=paired,note="Raw c0 selected metrics are empty by definition; paired gains must use the conditioned mask on c0 full error arrays, as reconstructed here.")


 def test_evaluate_rootheight_ignores_true_flag_nonfinite_values(self):
  # Objective already filters isfinite(height); evaluation must use the same validity policy.
  b=self.bodies[1];b.arrays['root_height_valid'][6:]=True;b.arrays['root_height_m'][6:]=np.nan
  with tempfile.TemporaryDirectory(prefix='synthetic-height-eval-') as tmp:
   report=trainer.evaluate(self.model,self.bodies,config(),Path(tmp),'synthetic_nanheight',quick=True)
  cases=[c for c in report['cases'] if c['target']==b.name]
  self.assertTrue(all(c['metrics'].get('root_height_abs_error_m') is None for c in cases))
  METRICS['eval_rootheight_nonfinite']=dict(no_crash=True,invalid_heights_excluded=True)

 def test_reference_interface_static_contract(self):
  source=inspect.getsource(trainer.evaluate_reference);tree=ast.parse(source)
  calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
  strict=[n for n in calls if isinstance(n.func,ast.Attribute) and n.func.attr=='load_state_dict']
  self.assertEqual(len(strict),1);self.assertTrue(any(k.arg=='strict' and isinstance(k.value,ast.Constant) and k.value.value is True for k in strict[0].keywords))
  loads=[n for n in calls if isinstance(n.func,ast.Attribute) and n.func.attr=='load' and isinstance(n.func.value,ast.Name) and n.func.value.id=='torch']
  self.assertEqual(len(loads),2)
  self.assertTrue(all(any(k.arg=='weights_only' and isinstance(k.value,ast.Constant) and k.value.value is True for k in n.keywords) for n in loads))
  eval_calls=[n for n in calls if isinstance(n.func,ast.Name) and n.func.id=='evaluate']
  self.assertEqual(len(eval_calls),1)
  self.assertTrue(any(k.arg=='condition_counts' and ast.literal_eval(k.value)==(0,) for k in eval_calls[0].keywords))
  self.assertNotIn('from_pretrained',source);self.assertNotIn('hf_hub_download',source)
  self.assertIn("sem_dim=768",source);self.assertIn("ret_con=True",source)
  METRICS['reference_interface_static']=dict(strict_state_load=True,weights_only_cache_and_checkpoint=True,count_zero_only=True,existing_siglip768_only=True,download_api_absent=True,actual_reference_checkpoint_loaded=False)

 def test_reference_restores_qwen_descriptors_after_missing_siglip_failure(self):
  original={b.name:b.static['sem'] for b in self.bodies if b.domain=='xwbc'}
  self.bodies[1].names[-1]='synthetic_missing_siglip_joint'
  lut={n:F.normalize(torch.randn(768),dim=0) for n in self.bodies[0].names}
  with tempfile.TemporaryDirectory(prefix='synthetic-reference-cache-') as tmp:
   p=Path(tmp)/'synthetic_siglip.pt';torch.save(lut,p)
   c=config();c['reference_siglip_cache']=str(p);c['reference_checkpoint']=str(Path(tmp)/'must_not_be_loaded.pt')
   with self.assertRaisesRegex(ValueError,'missing reference SigLIP'):
    trainer.evaluate_reference(self.bodies,c,Path(tmp))
  for b in self.bodies:
   if b.domain=='xwbc':self.assertIs(b.static['sem'],original[b.name]);self.assertEqual(b.static['sem'].shape[-1],1024)
  METRICS['reference_error_cleanup']=dict(synthetic_cache_only=True,reference_model_not_instantiated=True,all_original_descriptor_objects_restored=True)

class Recorder(unittest.TextTestResult):
 def __init__(self,*a,**kw):super().__init__(*a,**kw);self.records=[];self.starts={}
 def startTest(self,test):self.starts[str(test)]=time.monotonic();super().startTest(test)
 def mark(self,test,status,err=None):self.records.append(dict(test=str(test),status=status,seconds=time.monotonic()-self.starts[str(test)],detail=self._exc_info_to_string(err,test) if err else None))
 def addSuccess(self,test):self.mark(test,'passed');super().addSuccess(test)
 def addFailure(self,test,err):self.mark(test,'failed',err);super().addFailure(test,err)
 def addError(self,test,err):self.mark(test,'error',err);super().addError(test,err)

def main():
 start=time.monotonic();suite=unittest.defaultTestLoader.loadTestsFromTestCase(TrainerContracts);result=unittest.TextTestRunner(verbosity=2,resultclass=Recorder).run(suite)
 sources=[HERE/'train_pilot.py',HERE/'pilot_data.py',HERE/'pilot_model.py',Path(__file__)]
 report=dict(schema='trainer-cpu-numerical-review-v1',created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),success=result.wasSuccessful(),tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),wall_seconds=time.monotonic()-start,tests=result.records,metrics=METRICS,evidence_scope={'data':'synthetic Body fixtures for xwbc-like and unimate-like interfaces; names are fixture labels only','text_features':'synthetic normalized random1024-D placeholders; not Qwen inference','weights':'random small ConditionalMegatron','device':'CPU','synthetic_optimizer_updates':3,'real_pilot_training':False,'real_payload_loaded':False,'real_checkpoint_loaded':False,'network_used':False,'ssh_used':False,'gpu_used':False},sources=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),sha256_at_import=SOURCE_HASHES.get(str(p))) for p in sources],not_validated=['Actual pilot extraction schema and remote contents','True Qwen/SigLIP checkpoint eval or learned quality','CUDA/autocast runtime and optimizer capacity','Rootheight/condition performance on real heldout motions'])
 path=HERE/'trainer_numerical_review.json';path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 print(json.dumps({'report':str(path),'success':result.wasSuccessful(),'tests_run':result.testsRun,'seconds':report['wall_seconds']}));return 0 if result.wasSuccessful() else 1
if __name__=='__main__':raise SystemExit(main())
