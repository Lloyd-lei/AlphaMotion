"""Targeted CPU regression for loader-native slides and restricted checkpoint CLI."""
from pathlib import Path
import copy,json,tempfile,unittest
from unittest.mock import patch
import numpy as np
import torch
import run_scaling_eval as run
from hinge_dof_adapter import file_sha
from mixed_dof_adapter import MixedJointTree
from test_run_scaling_eval import FakeBody,FakeModel,metadata,adapter
from trajectory_refinement import RefinementConfig


def body_fixture(folder):
    path=Path(folder)/'synthetic.json';doc=metadata(path);doc['payload_sha256']='a'*64;path.write_text(json.dumps(doc))
    tree=MixedJointTree(path);body=FakeBody(tree);body.metric=True
    body.metadata.update(payload_sha256=doc['payload_sha256'],metadata_sha256=file_sha(path),length_unit='cm',source_native_unit='m')
    body.arrays['slide_native']=body.arrays.pop('slide_position_m')*100.
    return body,tree


def checkpoint(path,step,**extra):
    torch.save(dict(model={'weight':torch.ones(2)},step=step,config={'group':'A','model':{}},**extra),path)


class LoaderBoundary(unittest.TestCase):
    def test_only_native_cm_is_needed_and_soft_limit_reference_is_not_clipped(self):
        with tempfile.TemporaryDirectory() as d:
            body,tree=body_fixture(d);body.arrays['slide_native'][0,2]=4.1
            self.assertNotIn('slide_position_m',body.arrays)
            values=run.reference_slides_cm(body,np.array([0]),body.geom(1),'cpu')
            self.assertAlmostEqual(float(values[0,2]),4.1,places=5);self.assertLess(float(values.max()),5.)
            self.assertEqual(run.validate_loader_adapter(body,tree)['status'],'matched')
    def test_surface_case_uses_loader_native_field_through_runner(self):
        with tempfile.TemporaryDirectory() as d:
            body,tree=body_fixture(d);original=body.geom
            def with_surface(B):
                g=original(B);g['surface']={'asset_sha256':'b'*64};return g
            body.geom=with_surface;selection=run.select_episode_rows(body,max_frames=2)[0];captured=[]
            def metric(pred,ref,g,**kw):
                captured.append(kw['reference_slide_native'].detach().clone());return {'status':'synthetic_integration_stub'}
            with patch.object(run,'evaluate_fixed_surface',side_effect=metric):
                cases,results=run.evaluate_body_episodes(FakeModel(),body,adapter(tree,tree.metadata_path),output=Path(d)/'out',
                    selected=[selection],config=RefinementConfig(steps=0),mixed_tree=tree)
            self.assertEqual(len(results),1);self.assertNotEqual(results[0].get('status'),'failed_case_retained',results[0].get('error'))
            self.assertEqual(len(captured),1)
            torch.testing.assert_close(captured[0],torch.tensor(body.arrays['slide_native'][selection['rows']]))

    def test_missing_native_wrong_units_nonfinite_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            body,tree=body_fixture(d);native=body.arrays.pop('slide_native')
            body.arrays['slide_normalized']=native/4
            with self.assertRaisesRegex(ValueError,'slide_native'):run.reference_slides_cm(body,np.array([0]),body.geom(1),'cpu')
            body.arrays['slide_native']=native;body.metadata['length_unit']='m'
            with self.assertRaisesRegex(ValueError,'centimetre'):run.reference_slides_cm(body,np.array([0]),body.geom(1),'cpu')
            body.metadata['length_unit']='cm';native[0,2]=np.nan
            with self.assertRaisesRegex(ValueError,'finite'):run.reference_slides_cm(body,np.array([0]),body.geom(1),'cpu')
    def test_adapter_identity_parents_metadata_and_typed_geometry(self):
        with tempfile.TemporaryDirectory() as d:
            body,tree=body_fixture(d);run.validate_loader_adapter(body,tree)
            body.parents=body.parents.copy();body.parents[4]=0
            with self.assertRaisesRegex(ValueError,'parents'):run.validate_loader_adapter(body,tree)
            body.parents=tree.parents;body.metadata['metadata_sha256']='0'*64
            with self.assertRaisesRegex(ValueError,'SHA'):run.validate_loader_adapter(body,tree)
            body.metadata['metadata_sha256']=file_sha(tree.metadata_path)
            original=body.geom
            for key,bad in [('root_fixed',False),('slide_upper',torch.tensor([0.,0.,.04,.04,0.])),('prismatic_axis',torch.zeros(5,3)),('joint_type',torch.tensor([4,2,2,2,1]))]:
                def modified(B,key=key,bad=bad):
                    g=original(B);g[key]=bad;return g
                body.geom=modified
                with self.assertRaises(ValueError,msg=key):run.validate_loader_adapter(body,tree)
            body.geom=original
            self.assertEqual(run.validate_loader_adapter(body,tree)['q_coordinates'],3)


class CheckpointBoundary(unittest.TestCase):
    def test_all_prefers_weight_nodes_skips_final_and_ignores_incomplete_names(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d)
            for step in (500,1000,3000):checkpoint(d/f'checkpoint_step{step:04d}.pt',step)
            checkpoint(d/'checkpoint.pt',3000,numpy_rng=np.random.get_state())
            (d/'checkpoint_step1500.tmp.pt').write_bytes(b'incomplete')
            (d/'checkpoint_step_bad.pt').write_bytes(b'bad')
            self.assertEqual([p.name for p in run.discover_checkpoints(d,'all')],['checkpoint_step0500.pt','checkpoint_step1000.pt','checkpoint_step3000.pt'])
            self.assertEqual(run.discover_checkpoints(d,'final')[0].name,'checkpoint_step3000.pt')
            for p in run.discover_checkpoints(d,'all'):self.assertIsNotNone(run.load_evaluation_checkpoint(p)['model'])
    def test_same_bytes_same_step_dedup_and_conflicting_step_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);canonical=d/'checkpoint_step0500.pt';checkpoint(canonical,500)
            duplicate=d/'checkpoint_step500.pt';duplicate.write_bytes(canonical.read_bytes())
            self.assertEqual(run.discover_checkpoints(d,'all'),[canonical])
            checkpoint(duplicate,500,extra='changed')
            with self.assertRaisesRegex(ValueError,'conflicting'):run.discover_checkpoints(d,'all')
    def test_tensor_only_final_fallback_discards_optimizer_rng(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);p=d/'checkpoint.pt';checkpoint(p,3000,optimizer={'state':{}},torch_rng=torch.get_rng_state())
            self.assertEqual(run.discover_checkpoints(d,'all'),[p]);data=run.load_evaluation_checkpoint(p)
            self.assertNotIn('optimizer',data);self.assertNotIn('torch_rng',data);self.assertEqual(data['step'],3000)
    def test_numpy_final_rejected_without_unsafe_retry(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'checkpoint.pt';checkpoint(p,3000,numpy_rng=np.random.get_state())
            original=torch.load
            with patch.object(run.torch,'load',wraps=original) as load:
                with self.assertRaisesRegex(RuntimeError,'No unsafe pickle fallback'):run.load_evaluation_checkpoint(p)
                self.assertEqual(load.call_count,1);self.assertTrue(load.call_args.kwargs['weights_only'])
    def test_filename_step_and_state_schema_must_match(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'checkpoint_step0500.pt';checkpoint(p,3000)
            with self.assertRaisesRegex(ValueError,'filename'):run.load_evaluation_checkpoint(p)
            torch.save(dict(model={'weight':'not a tensor'},step=500),p)
            with self.assertRaisesRegex(ValueError,'named tensors'):run.load_evaluation_checkpoint(p)


class TimingAccounting(unittest.TestCase):
    def test_initial_projection_included_once_and_default_excludes_model(self):
        protocol={'comparison':{'learned_projected_initialization':{'seconds_to_threshold':1.5},'default_initialization':{'seconds_to_threshold':1.7}},
            'learned':{'raw_free_posture':{'protocol':{}},'dof_projected':{'protocol':{},'records':[
                {'step':0,'projection_seconds':.25},{'step':1,'projection_seconds':10.}]}}}
        control=run.add_timing_accounting(protocol,{'wall_seconds':2.})
        self.assertEqual(control['initial_DOF_projection_seconds'],.25)
        self.assertEqual(control['learned_inference_plus_constrained_seconds_to_threshold'],3.75)
        self.assertEqual(control['default_seconds_to_threshold'],1.7)
        self.assertFalse(control['default_requires_model_inference_or_initial_DOF_projection'])
        self.assertIn('observer',protocol['learned']['raw_free_posture']['protocol']['elapsed_seconds_scope'])
        self.assertIn('raw optimization',protocol['learned']['dof_projected']['protocol']['elapsed_seconds_scope'])
        protocol['learned']['dof_projected']['records']=[]
        control=run.add_timing_accounting(protocol,{'wall_seconds':2.})
        self.assertIsNone(control['initial_DOF_projection_seconds'])
        self.assertIsNone(control['learned_inference_plus_constrained_seconds_to_threshold'])


if __name__=='__main__':unittest.main(verbosity=2)
