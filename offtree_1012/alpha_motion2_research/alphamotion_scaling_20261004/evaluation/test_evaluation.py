"""CPU numerical tests; every pose/trajectory/model is explicitly synthetic."""
import copy,hashlib,json,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from eval_engine import *
from trajectory_refinement import *
from curve_statistics import *
from dof_protocol import VerifiedDOFAdapter,three_stage_refinement
torch.set_num_threads(1)


def fixture():
 p=np.zeros((5,3,3));p[:,:,0]=np.arange(3)[None];r=np.broadcast_to(np.eye(3),(5,3,3,3)).copy()
 return TrajectoryCase('case','physicalclip','source','target','paired_mapping','heldout_body','heldout_clip',np.arange(5),np.arange(5)*.1,['root','mid','end'],True,2.,p,r,source_frame_count=5,role_endpoints={'effector':[2]},provenance={'kind':'synthetic'})


def conditions(target=1.):
 return VisibleSE3(torch.ones(4,1,dtype=torch.long),torch.full((4,1,3),target),torch.eye(3).expand(4,1,3,3).clone(),torch.ones(4,1,dtype=torch.bool),torch.ones(4,1,dtype=torch.bool),('root','middle','end'),'synthetic toy condition','cm')


def decode(s):return s,torch.eye(3,device=s.device,dtype=s.dtype).expand(*s.shape[:2],3,3)


def refine(**kw):return refine_trajectory(torch.zeros(4,3,3),decode,conditions(),config=RefinementConfig(steps=100,learning_rate=.1,position_tolerance=.05,state_representation='synthetic independent XYZ'),**kw)


class Metrics(unittest.TestCase):
 def test_units_root_relative_endpoints(self):
  c=fixture();pred=PosePrediction(c.reference_positions+10,c.reference_rotations)
  o=evaluate_case(c,pred);self.assertEqual(o['metrics']['position']['mean'],0);self.assertEqual(o['metrics']['rotation']['mean'],0)
  self.assertEqual(o['metrics']['position']['unit'],'cm');self.assertEqual(o['endpoints']['effector']['joint_names'],['end'])
  c.metric=False;self.assertEqual(evaluate_case(c,pred)['metrics']['position']['unit'],'canonical_body_units')
 def test_yaw_alignment_single_constant_transform(self):
  c=fixture();A=np.array([[0.,0,1],[0,1,0],[-1,0,0]])
  pred=PosePrediction(np.einsum('ij,tkj->tki',A,c.reference_positions),A@c.reference_rotations)
  self.assertAlmostEqual(evaluate_case(c,pred)['metrics']['rotation']['mean'],90)
  self.assertAlmostEqual(evaluate_case(c,pred,alignment='first_frame_yaw_aligned')['metrics']['position']['mean'],0,places=12)
 def test_world_requires_explicit_anchors(self):
  c=fixture();pred=PosePrediction(c.reference_positions,c.reference_rotations)
  with self.assertRaises(ValueError):evaluate_case(c,pred,alignment='world')
  c.reference_world_positions=c.reference_positions+5;c.reference_world_rotations=c.reference_rotations
  pred.world_positions=c.reference_world_positions+1;pred.world_rotations=c.reference_rotations
  self.assertAlmostEqual(evaluate_case(c,pred,alignment='world')['metrics']['position']['mean'],np.sqrt(3))
 def test_failures_retained_and_invalid_rotation(self):
  c=fixture();p=c.reference_positions.copy();p[2,1]=np.nan;r=c.reference_rotations.copy();r[0,2]=0
  o=evaluate_case(c,PosePrediction(p,r));self.assertEqual(o['failed_frames'],2);self.assertIsNone(o['metrics']['position']['mean'])
  self.assertEqual(o['metrics']['position']['failed_values'],2)
 def test_case_contract_rejects_frame_drop_false_mapping_and_exposure(self):
  c=fixture();c.frame_ids=np.array([0,1,3,4,5])
  with self.assertRaises(ValueError):c.validate()
  c=fixture();c.reference_kind='same_body'
  with self.assertRaises(ValueError):c.validate()
  c=fixture()
  with self.assertRaises(ValueError):validate_exposure([c],training_body_ids=['target'],training_clip_ids=[])
  validate_exposure([c],training_body_ids=[],training_clip_ids=[])
 def test_constraints_only_has_no_fullbody_error(self):
  c=fixture();p,r=c.reference_positions,c.reference_rotations;c.reference_positions=c.reference_rotations=None;c.task_kind=c.reference_kind='constraints_only'
  self.assertEqual(evaluate_case(c,PosePrediction(p,r))['metrics'],{})
 def test_slip_true_time_world_metric_only(self):
  c=fixture();pred=PosePrediction(c.reference_positions,c.reference_rotations)
  self.assertEqual(evaluate_case(c,pred)['foot_slip']['status'],'unsupported')
  c.contact=np.ones((5,1),bool);c.contact_valid=np.ones((5,1),bool);pred.sole_world_positions=np.zeros((5,1,3));pred.sole_world_positions[:,0,0]=np.arange(5)*3
  self.assertAlmostEqual(evaluate_case(c,pred)['foot_slip']['horizontal_speed']['mean'],30)
  c.complete_clip=False;self.assertEqual(evaluate_case(c,pred)['foot_slip']['status'],'unsupported')
 def test_linear_time_has_zero_jitter(self):
  t=np.arange(5)*.1;p=np.zeros((5,1,3));p[:,0,0]=t*3;o=temporal_metrics(p,t,'cm')
  self.assertAlmostEqual(o['speed']['mean'],3);self.assertLess(o['jerk']['mean'],1e-10)
 def test_single_frame_mapping_timing_and_count(self):
  c=fixture();calls=[]
  def fun(i):calls.append(i);return c.reference_positions[i],c.reference_rotations[i]
  p,t=map_single_frames(5,fun);self.assertEqual(calls,list(range(5)));self.assertEqual(len(t['frame_latency_seconds']),5)
  self.assertEqual(p.positions.shape,(5,3,3));self.assertGreaterEqual(t['wall_seconds'],0)


class Refiner(unittest.TestCase):
 def test_single_optimizer_all_101_steps_and_baseline(self):
  r=refine();self.assertEqual([x['step'] for x in r['records']],list(range(101)));self.assertEqual(r['executed_optimizer_steps'],100)
  self.assertGreater(r['records'][0]['metrics']['selected_position_mean'],1);self.assertLess(r['records'][-1]['metrics']['selected_position_mean'],.05)
  self.assertEqual(r['protocol']['optimizer_instances'],1);self.assertEqual(r['protocol']['feasibility_level'],'raw_free_posture')
 def test_eval_gt_never_in_optimizer(self):
  a=refine(observer=lambda step,p,r:float((p-100).square().mean()));b=refine(observer=lambda step,p,r:float((p+999).square().mean()))
  torch.testing.assert_close(a['final_state'],b['final_state'],atol=0,rtol=0)
  self.assertFalse(a['protocol']['ground_truth_fullbody_loss'])
 def test_illegal_gt_loss_and_unknown_limits(self):
  c=LegalConstraint('bad','fullbody_GT',lambda s,p,r:s.abs(),'synthetic GT',validity_description='all')
  with self.assertRaises(ValueError):refine(constraints=[c])
  r=refine_trajectory(torch.zeros(4,3,3),decode,conditions(100.),config=RefinementConfig(steps=1))
  self.assertEqual(r['status'],'target_tolerances_not_met');self.assertIn('unknown',r['protocol']['limits_status'])
 def test_invalid_padding_never_enters_loss(self):
  c=conditions();c.joint_indices=torch.cat([c.joint_indices,torch.full((4,1),-99,dtype=torch.long)],1)
  c.positions=torch.cat([c.positions,torch.full((4,1,3),float('nan'))],1);c.rotations=torch.cat([c.rotations,torch.full((4,1,3,3),float('nan'))],1)
  c.position_valid=torch.cat([c.position_valid,torch.zeros(4,1,dtype=torch.bool)],1);c.rotation_valid=c.position_valid.clone()
  r=refine_trajectory(torch.zeros(4,3,3),decode,c,config=RefinementConfig(steps=2));self.assertTrue(np.isfinite(r['records'][-1]['loss']))
 def test_failure_tail_not_dropped(self):
  calls=[0]
  def f(s):
   calls[0]+=1;p,r=decode(s);return (p*float('nan'),r) if calls[0]>1 else (p,r)
  r=refine_trajectory(torch.zeros(4,3,3),f,conditions(),config=RefinementConfig(steps=4))
  self.assertEqual(len(r['records']),5);self.assertEqual(r['executed_optimizer_steps'],1);self.assertEqual(r['records'][-1]['status'],'not_executed_after_failure')
 def test_decoder_weights_frozen_and_restored(self):
  m=torch.nn.Linear(3,3,bias=False);m.weight.data.copy_(torch.eye(3));before=m.weight.detach().clone()
  def f(s):return m(s),torch.eye(3).expand(4,3,3,3)
  refine_trajectory(torch.zeros(4,3,3),f,conditions(),config=RefinementConfig(steps=3),frozen=[m])
  torch.testing.assert_close(m.weight,before);self.assertIsNone(m.weight.grad);self.assertTrue(m.training);self.assertTrue(m.weight.requires_grad)
 def test_roundtrip_derived_curves_arrays(self):
  r=refine()
  with tempfile.TemporaryDirectory() as d:
   report=save_refinement(d,r);self.assertEqual(len(refinement_curve({'synthetic':report})),101)
   self.assertEqual(len(json.loads((Path(d)/'refinement_curve.json').read_text())['records']),101)
   with np.load(Path(d)/'refinement_arrays.npz',allow_pickle=False) as a:self.assertEqual(a['final_positions'].shape,(4,3,3))
 def test_missing_true_dof_adapter_is_unsupported(self):
  r=three_stage_refinement(torch.zeros(4,3,3),decode,conditions())
  self.assertEqual(r['dof_projected']['status'],'unsupported');self.assertEqual(r['constrained_refinement']['status'],'unsupported')
 def test_verified_toy_dof_three_stages_and_hard_limits(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'synthetic_model.json';p.write_text('{"kind":"synthetic scalar slider","limits":[0,0.5]}')
   def qdecode(q):
    positions=torch.stack([torch.zeros_like(q).expand(-1,3),q.expand(-1,3),torch.zeros_like(q).expand(-1,3)],1)
    return positions,torch.eye(3).expand(len(q),3,3,3)
   adapter=VerifiedDOFAdapter(('synthetic_slider',),('metre',),torch.tensor([0.]),torch.tensor([.5]),lambda s:s[:,1,:1],qdecode,
      {str(p):hashlib.sha256(p.read_bytes()).hexdigest()},'explicit synthetic one-slider mapping','synthetic identity inverse')
   r=three_stage_refinement(torch.zeros(4,3,3),decode,conditions(),adapter=adapter,config=RefinementConfig(steps=10,learning_rate=.2,position_tolerance=.1))
   final=r['constrained_refinement'];self.assertTrue((final['final_state']<=.5).all());self.assertEqual(final['status'],'target_tolerances_not_met')
   self.assertEqual(final['protocol']['feasibility_level'],'constrained_joint_coordinates')
   p.write_text('changed')
   with self.assertRaises(ValueError):adapter.validate()


class Stats(unittest.TestCase):
 def row(self,b,c,v,n=10):return {'case_id':b+'/'+c,'clip_id':c,'target_body':b,'task_kind':'paired_mapping','body_exposure':'heldout_body','clip_exposure':'heldout_clip','alignment':'canonical_root_relative','failed_frames':0,'frame_count':5,'metrics':{'position_over_reach':{'mean':v,'total_values':n,'unit':'reach_fraction'}}}
 def test_macro_prevents_joint_count_dominance(self):
  r=summarize_cases([self.row('a','c',1,100),self.row('b','c',10,1)])[0];self.assertEqual(r['body_macro'],5.5);self.assertAlmostEqual(r['joint_time_micro'],110/101)
 def test_clip_bootstrap_minimum_and_reproducibility(self):
  self.assertEqual(bootstrap_clips([self.row('a','c',1)],samples=20)['status'],'insufficient_independent_clips')
  r=[self.row(b,str(i),float(i)) for b in ['a','b'] for i in range(5)]
  a=bootstrap_clips(r,samples=100,seed=17);self.assertEqual(a,bootstrap_clips(r,samples=100,seed=17));self.assertEqual(a['source_clips'],5)
 def test_failed_case_not_silently_dropped(self):
  rows=[self.row('a',str(i),1.) for i in range(3)];rows[-1]['metrics']['position_over_reach']['mean']=None;rows[-1]['failed_frames']=5
  r=bootstrap_clips(rows,samples=20);self.assertEqual(r['status'],'unavailable_including_failed_cases');self.assertAlmostEqual(r['aggregate']['failure_frame_fraction'],1/3)
 def test_paired_same_cases_and_task_strata(self):
  a=[self.row('a',str(i),float(i)) for i in range(3)];b=copy.deepcopy(a)
  for r in b:r['metrics']['position_over_reach']['mean']+=2
  self.assertEqual(paired_clip_difference(a,b,samples=20)['estimate'],-2.)
  with self.assertRaises(ValueError):paired_clip_difference(a,b[:1])
  b[0]['body_exposure']='seen_body'
  with self.assertRaises(ValueError):bootstrap_clips(b,samples=20)
 def test_fixed_budget_axes_and_evalset(self):
  c=[dict(run_id=r,checkpoint_id=r+'ck',examples_seen=100,training_compute_seconds=t,hardware_id='GPU',evaluation_set_sha256='fixed') for r,t in [('S',5),('L',20)]]
  self.assertTrue(all(x['status']=='matched' for x in budget_comparison(c,axis='examples_seen',budget=100,evaluation_set_sha256='fixed')['selected']))
  self.assertIn('under_budget_not_equal',{x['status'] for x in budget_comparison(c,axis='training_compute_seconds',budget=20,evaluation_set_sha256='fixed')['selected']})
  c[1]['hardware_id']='other'
  with self.assertRaises(ValueError):budget_comparison(c,axis='training_compute_seconds',budget=20,evaluation_set_sha256='fixed')
 def test_actual_nested_training_schema_and_digest(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'train.jsonl';p.write_text(json.dumps({'step':1,'terms':{'total':2,'rotation_deg':3,'position_over_reach':.1}})+'\n'+json.dumps({'step':2,'loss':1})+'\n')
   r=training_curve(p);self.assertEqual(r[0]['loss'],2);self.assertEqual(r[0]['schema'],'terms.total');self.assertEqual(r[1]['schema'],'legacy.loss')
  c=fixture();first=evaluation_set_digest([c]);c.body_exposure='seen_body';self.assertNotEqual(first,evaluation_set_digest([c]))

class Integration(unittest.TestCase):
 def test_same_fk_adapter_ignores_raw_aux_and_rejects_prismatic(self):
  import sys
  base=Path(__file__).parent.parent/'base';sys.path.insert(0,str(base));sys.path.insert(0,str(base/'legacy'))
  from pilot_data import fk
  from common import rot6d_to_matrix
  from fk_adapter import global_rot6_decoder,prediction_from_model_output
  parents=np.array([-1,0,1]);offsets=np.array([[0.,0,0],[1,0,0],[1,0,0]])
  r=torch.tensor([1.,0,0,0,1,0]).expand(4,3,6).clone().requires_grad_(True)
  p,R=global_rot6_decoder(parents,offsets,fk_fn=fk,rotation_matrix_fn=rot6d_to_matrix)(r)
  (p.square().sum()+R.square().sum()).backward();self.assertTrue(torch.isfinite(r.grad).all())
  a=prediction_from_model_output((r,torch.full((4,3,3),999.)),parents=parents,offsets=offsets,fk_fn=fk,rotation_matrix_fn=rot6d_to_matrix)
  np.testing.assert_allclose(a.positions,p.detach().numpy());self.assertLess(a.positions.max(),999)
  with self.assertRaises(ValueError):global_rot6_decoder(parents,offsets,fk_fn=fk,rotation_matrix_fn=rot6d_to_matrix,prismatic_joints=[1])
 def test_fixed_eval_selection_preserves_whole_paired_clip(self):
  cases=[]
  for ci in range(4):
   for bi in range(2):
    c=fixture();c.case_id=f'{ci}-{bi}';c.clip_id=str(ci);c.target_body=str(bi);cases.append(c)
  chosen,m=select_fixed_eval_cases(cases,max_source_clips_per_stratum=2,seed=1)
  self.assertEqual(len(chosen),4);self.assertEqual(len({c.clip_id for c in chosen}),2)
  self.assertEqual(m['evaluation_set_sha256'],evaluation_set_digest(chosen))
 def test_plot_reads_nested_actual_schema_and_preserves_refinement_baseline(self):
  from plot_curves import plot_training,plot_refinement
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'train.jsonl';p.write_text(json.dumps({'step':1,'terms':{'total':2.},'training_elapsed_seconds':.1})+'\n')
   plot_training({'synthetic':p},Path(d)/'train.png')
   result=refine_trajectory(torch.zeros(4,3,3),decode,conditions(),config=RefinementConfig(steps=3))
   plot_refinement({'synthetic':result},Path(d)/'refine.png')
   self.assertGreater((Path(d)/'train.png').stat().st_size,1000);self.assertGreater((Path(d)/'refine.png').stat().st_size,1000)

class HingeAdapterTests(unittest.TestCase):
 def metadata(self,path):
  entries=[]
  for i,axis in [(1,[0.,0.,1.]),(2,[1.,0.,0.])]:
   entries.append(dict(skeleton_index=i,body_name='link'+str(i),type='hinge',origin_local_body_m=[0.,0.,0.],qpos_address=i-1,joint_name='q'+str(i),axis_rest_canonical=axis,model_qpos0=0.,limited=True,range=[-2.,2.]))
  value={'model':dict(parents=[-1,0,1],joint_names=['fixedroot','link1','link2'],length_unit='m',rest_offsets_m=[[0,0,0],[1,0,0],[1,0,0]],mechanical_joints=entries,fixed_body_nodes=[dict(skeleton_index=0,body_name='fixedroot',parent_skeleton_index=-1,evidence='synthetic fixed root')])}
  path.write_text(json.dumps(value));return value
 def test_true_axis_fk_analytic_positions_and_projection_roundtrip(self):
  from hinge_dof_adapter import SingleHingeTree,rot6_from_matrix
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'synthetic_model.json';self.metadata(path);tree=SingleHingeTree(path)
   q=torch.tensor([[np.pi/2,.3]],dtype=torch.float64,requires_grad=True);p,R=tree.decode_q(q)
   torch.testing.assert_close(p[0,2],torch.tensor([100.,100.,0.],dtype=torch.float64),atol=1e-12,rtol=1e-12)
   torch.testing.assert_close(tree.project_free(rot6_from_matrix(R)),q,atol=1e-12,rtol=1e-12)
   p[0,2,0].backward();self.assertAlmostEqual(float(q.grad[0,0]),-100.,places=9)
   self.assertEqual(float(q.grad[0,1]),0.)
 def test_single_hinge_adapter_rejects_missing_actual_mapping_or_pivot(self):
  from hinge_dof_adapter import SingleHingeTree
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'synthetic_model.json';value=self.metadata(path);value['model']['mechanical_joints'][0]['origin_local_body_m']=[.1,0,0];path.write_text(json.dumps(value))
   with self.assertRaises(ValueError):SingleHingeTree(path)
   value=self.metadata(path);value['model']['mechanical_joints'][0]['type']='slide';path.write_text(json.dumps(value))
   with self.assertRaises(ValueError):SingleHingeTree(path)
 def test_no_actual_validation_receipt_means_no_verified_adapter(self):
  from hinge_dof_adapter import SingleHingeTree
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'synthetic_model.json';self.metadata(path);tree=SingleHingeTree(path);receipt=Path(d)/'receipt.json';receipt.write_text('{"status":"passed"}')
   with self.assertRaises(ValueError):tree.verified_adapter(receipt)

class MatchedControls(unittest.TestCase):
 def test_default_q_baseline_same_protocol_threshold_timing(self):
  from dof_protocol import compare_learned_vs_default
  with tempfile.TemporaryDirectory() as d:
   evidence=Path(d)/'toy.json';evidence.write_text('{"source":"synthetic actual slider"}')
   def qdecode(q):
    p=torch.stack([torch.zeros_like(q).expand(-1,3),q.expand(-1,3),torch.zeros_like(q).expand(-1,3)],1)
    return p,torch.eye(3).expand(len(q),3,3,3)
   adapter=VerifiedDOFAdapter(('toy',),('metre',),torch.tensor([0.]),torch.tensor([2.]),lambda s:s[:,1,:1],qdecode,
      {str(evidence):hashlib.sha256(evidence.read_bytes()).hexdigest()},'synthetic one-slider','synthetic inverse',default_q=torch.tensor([0.]),default_source='synthetic model qpos0')
   initial=torch.zeros(4,3,3);initial[:,1]=.5
   r=compare_learned_vs_default(initial,decode,conditions(),adapter=adapter,config=RefinementConfig(steps=10,learning_rate=.2,position_tolerance=.1))
   self.assertEqual(r['default_valid_q_baseline']['requested_steps'],10)
   self.assertEqual(r['default_valid_q_baseline']['protocol']['learning_rate'],r['learned']['constrained_refinement']['protocol']['learning_rate'])
   c=r['comparison'];self.assertTrue(c['same_conditions']);self.assertIn('not a comparison against a complete IK',c['claim'])
   self.assertIsNotNone(c['default_initialization']['step0'])
   self.assertEqual(len(r['learned']['dof_projected']['records']),11)
 def test_surface_uses_identical_fixed_mesh_points_and_missing_is_na(self):
  from surface_metrics import evaluate_fixed_surface
  self.assertEqual(evaluate_fixed_surface(None,None,{},bind_fn=None,expected_patch_sha256='')['status'],'NA')
  valid=torch.ones(2,2,dtype=torch.bool);surf={'_validated':True,'asset_sha256':'a'*64,'patch_ids':('face0','face1'),
   'valid':valid,'area':torch.tensor([[1.,3.],[1.,3.]]),'metric_valid':torch.ones(2,dtype=torch.bool),
   'meters_per_unit':torch.tensor([.01,.01]),'pose_scale':torch.tensor([10.,10.])}
  def bind(x,g):return {'surface':{'position':x,'normal':torch.tensor([0.,1.,0.]).expand(2,2,3),'valid':valid,'patch_ids':surf['patch_ids']}}
  p=torch.zeros(2,2,3);p[:,1,0]=2
  r=evaluate_fixed_surface(p,torch.zeros_like(p),{'surface':surf},bind_fn=bind,expected_patch_sha256='a'*64)
  self.assertEqual(r['position_area_mean']['mean'],1.5);self.assertAlmostEqual(r['position_over_reach_area_mean']['mean'],.15,places=6)
  with self.assertRaises(ValueError):evaluate_fixed_surface(p,p,{'surface':surf},bind_fn=bind,expected_patch_sha256='b'*64)
 def test_independent_frame_batch_mapping_preserves_complete_sequence(self):
  c=fixture();calls=[]
  def f(ids):calls.extend(ids.tolist());return c.reference_positions[ids],c.reference_rotations[ids]
  p,t=map_independent_frame_batches(5,2,f)
  self.assertEqual(calls,list(range(5)));self.assertEqual([b['frames'] for b in t['batches']],[2,2,1])
  np.testing.assert_array_equal(p.positions,c.reference_positions);self.assertIn('amortized',t['execution'])

if __name__=='__main__':unittest.main(verbosity=2)
