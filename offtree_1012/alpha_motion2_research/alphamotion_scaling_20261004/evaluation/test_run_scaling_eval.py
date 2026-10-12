"""CPU-only synthetic integration. These are not real dataset/model evaluations."""
from pathlib import Path
import copy, json, tempfile, unittest
import numpy as np
import torch
import run_scaling_eval as run
from mixed_dof_adapter import MixedJointTree, validate_saved_qpos
from dof_protocol import VerifiedDOFAdapter
from trajectory_refinement import RefinementConfig
from hinge_dof_adapter import file_sha,rot6_from_matrix
from prismatic.kinematics import mixed_forward_kinematics

torch.set_num_threads(1)


def metadata(path):
    names=['root','arm','left','right','tool'];parents=[-1,0,1,1,1]
    entries=[]
    for slot,j,typ,axis,limits in [(0,1,'hinge',[0.,0.,1.],[-1.,1.]),(1,2,'slide',[0.,1.,0.],[0.,.04]),(2,3,'slide',[0.,-1.,0.],[0.,.04])]:
        entries.append(dict(skeleton_index=j,body_name=names[j],joint_name='q'+str(slot),qpos_address=slot,type=typ,
            origin_local_body_m=[0.,0.,0.],axis_rest_canonical=axis,model_qpos0=0.,limited=True,range=limits))
    fixed=[dict(skeleton_index=j,body_name=names[j],parent_skeleton_index=parents[j],evidence='synthetic fixed-node source') for j in (0,4)]
    value={'frames':4,'model':dict(parents=parents,joint_names=names,length_unit='m',rest_offsets_m=[[0,0,0],[.1,0,0],[.2,0,0],[.2,0,0],[.25,0,0]],mechanical_joints=entries,fixed_body_nodes=fixed),
        'mixed_joint_contract':{'coupling':'No MJCF mimic/equality constraint; synthetic independent sliders','prismatic_rotation':'actual C=I synthetic fixture'},
        'aperture_definition':'Signed projection of left-origin minus right-origin onto actual parent-rotated left slide axis; SYNTHETIC',
        'aperture_indices':{'left_finger':2,'right_finger':3,'parent':1}}
    path.write_text(json.dumps(value));return value


def geom(tree,B):
    J=len(tree.names);typ=torch.tensor([4,2,3,3,1]);axis=torch.zeros(J,3);axis[2]=tree.axis[1].float();axis[3]=tree.axis[2].float()
    return dict(parents=tree.parents,kinematic_rest_off=tree.offsets.float(),joint_type=typ,
        prismatic_axis=axis,slide_lower=torch.zeros(J),slide_upper=torch.tensor([0.,0.,4.,4.,0.]),slide_reference=torch.zeros(J),
        kinematic_fixed_rotation=torch.eye(3).expand(J,3,3),root_fixed=True,surface=None)


class FakeBody:
    """Four toy independent clips (one train, three heldout), not robot evidence."""
    def __init__(self,tree):
        self.name='synthetic_slide_body';self.names=list(tree.names);self.parents=tree.parents;self.offsets=tree.offsets.numpy();self.reach=40.;self.tree=tree
        q=torch.tensor([[.2+i*.03,.025,.035] for i in range(6)],dtype=torch.float32).repeat(4,1);P,R=tree.decode_q(q);rot=rot6_from_matrix(R).numpy()
        self.arrays={'global_rot6':rot,'pose9':np.concatenate([rot,P.numpy()/self.reach],-1),
            'group':np.repeat(np.arange(4),6),'split':np.repeat([0,1,1,1],6),'frame_index':np.tile(np.arange(6),4),
            'timestamp_seconds':np.tile(np.arange(6)*.1,4),'recorded_robot_qpos':q.numpy(),
            'slide_position_m':tree.pack_q(q).numpy()[...,6]/100.}
        self.metadata={'payload_sha256':'0'*64,'training_stride':1,'source_dataset':'SYNTHETIC',
            'episodes':[dict(source_group='synthetic_clip_'+str(i),frames=6) for i in range(4)]}
        self.endpoint={'indices':[2,3]}
    def eligible(self,train):return np.flatnonzero(self.arrays['split']==0 if train else self.arrays['split']>0)
    def geom(self,B):return geom(self.tree,B)


class FakeModel(torch.nn.Module):
    def __init__(self):super().__init__();self.anchor=torch.nn.Parameter(torch.tensor(0.));self.calls=[]
    def forward_kinematic(self,source,sg,tg,source_hidden=None,source_contact=None,source_slide_normalized=None,condition=None):
        assert torch.count_nonzero(source)==0 and bool(source_hidden.all()) and source_contact is None and source_slide_normalized is None
        assert condition['values'].shape[1]<=2 and condition['valid'].all()
        self.calls.append(tuple(source.shape));B,J=source.shape[:2]
        rot=source.new_tensor([1.,0.,0.,0.,1.,0.]).expand(B,J,6).clone()
        slide=source.new_zeros(B,J);slide[:,2:4]=2.
        return dict(**mixed_forward_kinematics(rot,tg,slide),raw_hpos=source.new_full((B,J,3),999.))


def adapter(tree,path):
    return VerifiedDOFAdapter(tree.q_names,tree.units,tree.lower,tree.upper,tree.project_free,tree.decode_q,
        {str(path):file_sha(path)},'SYNTHETIC exact toy tree','SYNTHETIC axis+slide inverse',tree.q0,'SYNTHETIC model default')


class MixedAdapter(unittest.TestCase):
    def test_actual_mixed_metadata_internal_roundtrip_not_simulator_validation(self):
        path=run.PACKAGE/'data/relay_franka_full_v2.json'
        if not path.exists():self.skipTest('local source metadata optional')
        tree=MixedJointTree(path);self.assertEqual(tree.units,('radian',)*7+('metre',)*2)
        q=tree.q0[None].clone();q[0,7:]=torch.tensor([.013,.027],dtype=q.dtype);P,R=tree.decode_q(q)
        torch.testing.assert_close(tree.project_free(tree.pack_q(q)),q,atol=1e-10,rtol=0)
        self.assertAlmostEqual(float(tree.aperture_cm(P,R)),4.,places=7)
    def test_analytic_slider_units_fixed_marker_and_projection(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'synthetic.json';metadata(p);tree=MixedJointTree(p)
            q=torch.tensor([[0.,.01,.03],[.2,.041,.02]],dtype=torch.float64);P,R=tree.decode_q(q)
            np.testing.assert_allclose(P[0,2].numpy(),[30,1,0]);np.testing.assert_allclose(P[0,3].numpy(),[30,-3,0]);np.testing.assert_allclose(P[0,4].numpy(),[35,0,0])
            torch.testing.assert_close(tree.project_free(tree.pack_q(q)),q,atol=1e-12,rtol=0)
            torch.testing.assert_close(tree.aperture_cm(P,R),torch.tensor([4.,6.1],dtype=torch.float64),atol=1e-12,rtol=0)
            with self.assertRaises(ValueError):tree.project_free(tree.pack_q(q)[...,:6])
            with self.assertRaises(ValueError):tree.verified_adapter(p)
    def test_gradient_and_no_gt_projection_slide(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'synthetic.json';metadata(p);tree=MixedJointTree(p)
            q=torch.tensor([[.15,.01,.03]],dtype=torch.float64,requires_grad=True);P,R=tree.decode_q(q)
            torch.autograd.gradcheck(lambda x:tree.decode_q(x)[0],(q,),eps=1e-6,atol=1e-5)
            state=tree.pack_q(q.detach());state[:,2,6]=1.7
            self.assertAlmostEqual(float(tree.project_free(state)[0,1]),.017)
    def test_synthetic_saved_array_validator_stale_receipt_and_soft_limits(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);p=d/'synthetic.json';doc=metadata(p);tree=MixedJointTree(p)
            q=torch.tensor([[0.,.01,.03],[.1,.03,.041],[.2,.02,.02],[-.1,.01,.01]],dtype=torch.float64);P,R=tree.decode_q(q)
            z=d/'synthetic.npz';np.savez_compressed(z,recorded_robot_qpos=q.numpy(),position_native=P.numpy()/100.,global_rot6=rot6_from_matrix(R).numpy(),aperture_signed_m=tree.aperture_cm(P,R).numpy()/100.)
            doc['payload_sha256']=file_sha(z);p.write_text(json.dumps(doc));receipt=d/'synthetic_validation.json'
            v=validate_saved_qpos(p,z,receipt);self.assertEqual(v['status'],'passed');self.assertEqual(v['preserved_reference_limit_violations_per_coordinate'],[0,0,1])
            a=MixedJointTree(p).verified_adapter(receipt);self.assertEqual(a.coordinate_units,('radian','metre','metre'))
            doc['scope']='tamper';p.write_text(json.dumps(doc))
            with self.assertRaises(ValueError):MixedJointTree(p).verified_adapter(receipt)


class Runner(unittest.TestCase):
    def test_model_group_and_checkpoint_discovery(self):
        self.assertEqual(run.model_variant({'group':'A'}),'A');self.assertEqual(run.model_variant({'group':'B'}),'B')
        with self.assertRaises(ValueError):run.model_variant({})
        with tempfile.TemporaryDirectory() as d:
            d=Path(d)
            for n in (10,100,20):(d/f'checkpoint_step{n}.pt').touch()
            self.assertEqual(run.discover_checkpoints(d)[0].name,'checkpoint_step100.pt')
            (d/'checkpoint_final.pt').touch();self.assertEqual(run.discover_checkpoints(d)[0].name,'checkpoint_step100.pt')
    def test_fixed_episodes_true_subclip_and_hidden_model_input(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'synthetic.json';metadata(p);tree=MixedJointTree(p);body=FakeBody(tree)
            selected=run.select_episode_rows(body,clips=3,max_frames=4);self.assertEqual([s['group_index'] for s in selected],[1,2,3])
            case,c,_,_=run.make_case_and_condition(body,selected[0],torch.device('cpu'))
            self.assertFalse(case.complete_clip);self.assertEqual(case.source_frame_count,6)
            model=FakeModel();state,pred,timing,aux=run.sparse_model_initialization(model,body,c,2)
            self.assertEqual(state.shape,(4,5,7));self.assertEqual(model.calls,[(2,5,9),(2,5,9)])
            self.assertLess(pred.positions.max(),999.);self.assertEqual(float(aux.max()),999.)
            with self.assertRaises(ValueError):run.select_episode_rows(body,clips=2,max_frames=4)
            body.arrays['frame_index'][8]=100
            with self.assertRaises(ValueError):run.select_episode_rows(body,clips=3,max_frames=4)
    def test_cpu_three_stage_default_optimizer_io_and_bootstrap(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);p=d/'synthetic.json';metadata(p);tree=MixedJointTree(p);body=FakeBody(tree);model=FakeModel()
            cases,results=run.evaluate_body_episodes(model,body,adapter(tree,p),output=d/'out',clips=3,max_frames=4,batch_size=2,
                config=RefinementConfig(steps=2,learning_rate=.005),mixed_tree=tree)
            self.assertEqual(len(cases),3);self.assertFalse(any(r.get('status')=='failed_case_retained' for r in results),repr(results[0].get('error')))
            summary=run.summarize_checkpoint(cases,results,bootstrap_samples=10)
            self.assertEqual(summary['real_source_clip_count'],3);self.assertFalse(summary['all_cases_complete_original_clip'])
            for r in results:
                self.assertGreater(r['no_refine_initial_model']['selected_SE3']['position_mean_cm'],0)
                out=d/'out'/r['case_id'];self.assertGreater((out/'SE3_position_vs_steps_and_time.png').stat().st_size,1000)
                with np.load(out/'case_arrays.npz') as z:self.assertEqual(z['initial_model_slide_cm'].shape,(4,5))
                for stage in ('raw_free_posture','dof_projected','constrained_refinement','default_valid_q'):
                    value=json.loads((out/stage/'refinement_curve.json').read_text());self.assertEqual([x['step'] for x in value['records']],[0,1,2])
            self.assertIn('selected_SE3_position_cm',summary['stages']['no_refine'])
            self.assertIsNone(model.anchor.grad)
    def test_limit_control_checks_all_native_frames_and_keeps_invalid_starts(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);p=d/'synthetic.json';metadata(p);tree=MixedJointTree(p);body=FakeBody(tree)
            q=body.arrays['recorded_robot_qpos'].copy();q[6:8,1]=.041;q[12:18,1]=.041
            # Native violation at an omitted stride-2 frame must break the interval.
            q[19,1]=.041
            native=d/'synthetic_native.npz';np.savez_compressed(native,recorded_robot_qpos=q,
                source_group_index=body.arrays['group'],frame_index=body.arrays['frame_index'],split=body.arrays['split'])
            body.metadata['payload_sha256']=file_sha(native)
            selected,audit=run.select_reference_controls(body,tree,native,max_frames=4)
            self.assertEqual(len(selected),5);self.assertEqual(audit['strict_limit_valid_independent_clips'],2)
            self.assertEqual(audit['strict_control_CI_status'],'unavailable_fewer_than_three_independent_clips')
            self.assertIn('has_soft_limit_violations',selected[0]['reference_cohort'])
            self.assertEqual(int(body.arrays['frame_index'][selected[3]['rows'][0]]),2)
            self.assertTrue(all(s['native_reference_limit_qc']['all_native_q_within_hard_limits'] for s in selected if s['reference_cohort']=='strict_limit_valid_control'))
            case_rows=[];results=[]
            for selection in selected:
                case,_,_,_=run.make_case_and_condition(body,selection,torch.device('cpu'));case_rows.append(case)
                fail=run.failure_result(case,'synthetic retained failure')
                results.append({'case_id':case.case_id,'no_refine_initial_model':fail,'stages':{n:{'final_observation':fail} for n in ('raw_free_posture','dof_projected','constrained_refinement','default_valid_q')}})
            summary=run.summarize_checkpoint(case_rows,results,bootstrap_samples=5)
            self.assertFalse(summary['metrics_pooled_across_reference_cohorts']);self.assertEqual(len(summary['reference_cohorts']),2)
            # Keep only even sampled frames, while audit still sees original frame1.
            selected_rows=body.arrays['frame_index']%2==0
            body.arrays={k:v[selected_rows] for k,v in body.arrays.items()};body.metadata['training_stride']=2
            selected,audit=run.select_reference_controls(body,tree,native,max_frames=3)
            strict=[s for s in selected if s['reference_cohort']=='strict_limit_valid_control' and s['group_index']==3][0]
            self.assertEqual(body.arrays['frame_index'][strict['rows']].tolist(),[2,4])
            self.assertEqual(strict['native_reference_limit_qc']['native_frames_checked'],3)

    def test_real_A_B_factory_and_pure_sparse_mixed_forward_with_synthetic_body(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'synthetic.json';metadata(p);tree=MixedJointTree(p);body=FakeBody(tree)
            native_geom=body.geom
            def descriptive(B):
                g=native_geom(B);J=len(body.names);off=tree.offsets.float()[None].expand(B,-1,-1)
                depth=torch.tensor([0,1,2,2,2])[None].expand(B,-1);hop=torch.ones(B,J,J,dtype=torch.long);hop[:,torch.arange(J),torch.arange(J)]=0
                g.update(rest_off=off,bonelen=off.norm(dim=-1)/body.reach,depth=depth,hop=hop,
                    sem=torch.zeros(B,J,1024),dof=torch.zeros(B,J,19),metadata_missing=torch.zeros(B,J,2),pose_scale=torch.full((B,),body.reach))
                return g
            body.geom=descriptive
            selection=run.select_episode_rows(body,max_frames=2)[0]
            _,cond,_,_=run.make_case_and_condition(body,selection,torch.device('cpu'))
            cfg={'model':dict(d_model=32,n_heads=4,enc_layers=1,dec_layers=1,n_latent=8,n_latent_p=4,fsq_stages=1,fsq_levels=[32]*13,sem_dim=1024,n_judges=12)}
            for variant in ('A','B'):
                model=run.model_factory(cfg,variant).eval()
                state,pred,_,_=run.sparse_model_initialization(model,body,cond,2)
                self.assertEqual(state.shape,(2,5,7));self.assertTrue(torch.isfinite(state).all())
                self.assertTrue(bool(((state[:,2:4,6]>0)&(state[:,2:4,6]<4)).all()))
                self.assertTrue(np.isfinite(pred.positions).all());self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_failed_case_retained_in_all_stage_denominators(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'synthetic.json';metadata(p);tree=MixedJointTree(p);body=FakeBody(tree);model=FakeModel()
            model.forward_kinematic=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('synthetic injected failure'))
            cases,results=run.evaluate_body_episodes(model,body,adapter(tree,p),output=Path(d)/'out',clips=3,max_frames=4,config=RefinementConfig(steps=0))
            self.assertEqual(len(results),3)
            report=run.summarize_checkpoint(cases,results,bootstrap_samples=5)
            self.assertEqual(report['stages']['no_refine']['selected_SE3_position_cm'][0]['failed_or_unavailable_case_ids'],[c.case_id for c in cases])


if __name__=='__main__':unittest.main(verbosity=2)
