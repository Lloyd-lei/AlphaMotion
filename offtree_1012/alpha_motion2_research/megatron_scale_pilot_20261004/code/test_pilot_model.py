#!/usr/bin/env python3
"""CPU numerical checks using synthetic data and genuine local legacy source.

No training/optimizer, real-data batch, checkpoint load, download, SSH or GPU use.
The optional schema checks use a randomly initialized383-key legacy state_dict.
"""
from __future__ import annotations
import ast, datetime, hashlib, importlib.util, io, json, os, sys, time, unittest
from pathlib import Path
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True
import torch
import torch.nn.functional as F
HERE = Path(__file__).resolve().parent
LEGACY = Path(os.environ.get("PILOT_LEGACY_SOURCE", str(HERE / "legacy")))
sys.path.insert(0, str(LEGACY))
sys.path.insert(0, str(HERE))
SOURCE_HASHES_AT_IMPORT = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/"pilot_model.py", LEGACY/"common.py", LEGACY/"model.py", LEGACY/"model_dual.py"]}
from model_dual import DualTranslator
from pilot_model import ConditionalMegatron, initialize_added_identity_blocks, load_reference_weights

torch.set_num_threads(1)
METRICS = {}

def record(name, **values):
    METRICS[name] = values

def small_model(**extra):
    kw = dict(d_model=32, n_heads=4, enc_layers=5, dec_layers=5, n_latent=4,
              n_latent_p=2, fsq_stages=2, fsq_levels=(8,8,8,5,5,5), use_sem=True,
              sem_dim=1024, use_dof=True, contact_chan=True, n_judges=16,
              single_codebook=False, rot_only=False, rot_head="free",
              vis_residual=False, root_head=False)
    kw.update(extra)
    return ConditionalMegatron(**kw).cpu().eval()

def geometry(batch, joints):
    rest = torch.randn(batch, joints, 3)
    depth = torch.arange(joints)[None].expand(batch, -1)
    return dict(rest_off=rest, bonelen=rest.norm(dim=-1), depth=depth,
                hop=(depth[:,:,None]-depth[:,None,:]).abs(),
                sem=F.normalize(torch.randn(batch,joints,1024), dim=-1),
                dof=torch.zeros(batch,joints,19), metadata_missing=torch.zeros(batch,joints,2))

def condition(batch, count, joints, valid=True):
    ix=torch.arange(count)[None].expand(batch,-1).clone() % joints
    # Synthetic identity rotation6D + synthetic translation, never a measured pose.
    values=torch.zeros(batch,count,9)
    values[...,:6]=torch.tensor([1.,0.,0.,0.,1.,0.])
    values[...,6:]=torch.randn(batch,count,3)
    return dict(joint_indices=ix, values=values, valid=torch.full((batch,count),valid,dtype=torch.bool))

def tensor_outputs(out):
    return [x for x in out if torch.is_tensor(x)]

def delta(a,b):
    return max(float((u-v).abs().max().detach()) for u,v in zip(tensor_outputs(a),tensor_outputs(b)))

class PilotNumerics(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1729)
        self.model=small_model();self.B=2;self.Js=5;self.Jt=7
        self.sg=geometry(self.B,self.Js);self.tg=geometry(self.B,self.Jt)
        self.x=torch.randn(self.B,self.Js,9)
        self.hidden=torch.zeros(self.B,self.Js,dtype=torch.bool);self.hidden[:,[1,3]]=True

    def forward(self, x=None, **kw):
        return self.model(self.x if x is None else x,self.sg,self.tg,**kw)

    def assertFinite(self,out):
        self.assertTrue(all(bool(torch.isfinite(x).all()) for x in tensor_outputs(out)))

    def backward_and_check(self,out):
        loss=sum(x.square().mean() for x in tensor_outputs(out));loss.backward()
        bad=[n for n,p in self.model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
        self.assertEqual(bad,[])
        return float(loss.detach())

    def test_forward_zero_one_two_conditions(self):
        with torch.no_grad():
            for c in [0,1,2]:
                out=self.forward(condition=condition(self.B,c,self.Jt))
                self.assertEqual(tuple(out[0].shape),(self.B,self.Jt,6))
                self.assertEqual(tuple(out[1].shape),(self.B,self.Jt,3))
                self.assertEqual(tuple(out[2].shape),(self.B,self.Jt))
                self.assertFinite(out)
        record("condition_sizes", accepted=[0,1,2], source_joints=self.Js,target_joints=self.Jt)

    def test_reject_unsupported_inherited_modes(self):
        unsupported=dict(use_sem=False,use_dof=False,n_judges=1,single_codebook=True,
                         rot_only=True,rot_head="angles",vis_residual=True,
                         root_head=True,contact_chan=False)
        for key,value in unsupported.items():
            with self.subTest(flag=key),self.assertRaises(ValueError):small_model(**{key:value})
        record("fixed_pilot_flags",rejected=list(unsupported),non16_judge_rejected=True)

    def test_padding_mask_is_per_batch_member(self):
        c=condition(self.B,2,self.Jt);c['valid'][0]=False;c['joint_indices'][0]=-1;c['values'][0]=float('nan')
        with torch.no_grad():
            base=self.forward();out=self.forward(condition=c)
        for a,b in zip(tensor_outputs(base),tensor_outputs(out)):
            torch.testing.assert_close(a[0],b[0],atol=2e-6,rtol=2e-6)
        valid_delta=max(float((a[1]-b[1]).abs().max()) for a,b in zip(tensor_outputs(base),tensor_outputs(out)))
        self.assertGreater(valid_delta,1e-7)
        record("per_batch_condition_mask",all_invalid_member_max_abs=max(float((a[0]-b[0]).abs().max()) for a,b in zip(tensor_outputs(base),tensor_outputs(out))),valid_member_dependency_delta=valid_delta)

    def test_reject_more_than_two_conditions(self):
        with self.assertRaises(ValueError):self.forward(condition=condition(self.B,3,self.Jt))

    def test_reject_invalid_contracts(self):
        cases=[]
        c=condition(self.B,2,self.Jt);c['valid']=c['valid'].float();cases.append(c)
        c=condition(self.B,2,self.Jt);c['values']=c['values'][...,:8];cases.append(c)
        c=condition(self.B,2,self.Jt);c['joint_indices']=c['joint_indices'].int();cases.append(c)
        c=condition(self.B,2,self.Jt);c['joint_indices'][0,0]=-1;cases.append(c)
        c=condition(self.B,2,self.Jt);c['joint_indices'][0,0]=self.Jt;cases.append(c)
        c=condition(self.B,2,self.Jt);c['values'][0,0,0]=float('nan');cases.append(c)
        c=condition(self.B,2,self.Jt);c['values'][0,0,0]=float('inf');cases.append(c)
        c=condition(self.B+1,2,self.Jt);cases.append(c)
        for i,c in enumerate(cases):
            with self.subTest(case=i),self.assertRaises((ValueError,TypeError)):self.forward(condition=c)

    def test_no_condition_matches_zero_and_all_invalid(self):
        with torch.no_grad():
            baseline=self.forward()
            zero=self.forward(condition=condition(self.B,0,self.Jt))
            c=condition(self.B,2,self.Jt,valid=False)
            c['joint_indices'][:]=torch.tensor([-1,99999])
            c['values'][:,0]=float('nan');c['values'][:,1]=float('inf')
            invalid=self.forward(condition=c)
        for out in [zero,invalid]:
            self.assertFinite(out)
            for a,b in zip(tensor_outputs(baseline),tensor_outputs(out)):
                torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-6)
        record("no_condition_equivalence",zero_max_abs=delta(baseline,zero),invalid_max_abs=delta(baseline,invalid))

    def test_hidden_source_gt_perturbation_does_not_leak(self):
        other=self.x.clone();other[self.hidden]=torch.randn_like(other[self.hidden])*10000
        c=condition(self.B,2,self.Jt)
        with torch.no_grad():
            za=self.model.encode(self.x,self.sg,joint_mask=self.hidden)
            zb=self.model.encode(other,self.sg,joint_mask=self.hidden)
            a=self.forward(source_hidden=self.hidden,condition=c)
            b=self.forward(x=other,source_hidden=self.hidden,condition=c)
        self.assertTrue(all(torch.equal(u,v) for u,v in zip(za,zb)))
        self.assertEqual(delta(a,b),0.)
        record("hidden_source_gt_perturbation",latent_max_abs=max(float((u-v).abs().max()) for u,v in zip(za,zb)),output_max_abs=delta(a,b))

    def test_hidden_source_nan_inf_has_finite_zero_hidden_gradients(self):
        x=self.x.clone();x[:,1]=float('nan');x[:,3]=float('inf');x.requires_grad_(True)
        out=self.forward(x=x,source_hidden=self.hidden)
        self.assertFinite(out);loss=self.backward_and_check(out)
        self.assertTrue(bool(torch.isfinite(x.grad).all()));self.assertEqual(float(x.grad[self.hidden].abs().max()),0.)
        record("hidden_nonfinite_gradients",loss=loss,all_parameter_gradients_finite=True,hidden_input_gradient_max_abs=0.)

    def test_visible_nonfinite_rejected(self):
        for v in [float('nan'),float('inf')]:
            x=self.x.clone();x[:,0,0]=v
            with self.assertRaises(ValueError):self.forward(x=x,source_hidden=self.hidden)

    def test_invalid_sparse_padding_nan_inf_has_finite_zero_gradients(self):
        c=condition(self.B,2,self.Jt);c['valid'][:,1]=False;c['joint_indices'][:,1]=-1
        c['values'][0,1]=float('nan');c['values'][1,1]=float('inf');c['values'].requires_grad_(True)
        out=self.forward(condition=c);self.assertFinite(out);loss=self.backward_and_check(out)
        self.assertTrue(bool(torch.isfinite(c['values'].grad).all()))
        self.assertEqual(float(c['values'].grad[:,1].abs().max()),0.)
        record("invalid_sparse_padding_gradients",loss=loss,all_parameter_gradients_finite=True,invalid_value_gradient_max_abs=0.)

    def test_pure_condition_all_hidden_contact_none_no_gt_leak(self):
        hidden=torch.ones(self.B,self.Js,dtype=torch.bool)
        c=condition(self.B,2,self.Jt);nan_x=torch.full_like(self.x,float('nan'))
        with torch.no_grad():
            a=self.forward(source_hidden=hidden,source_contact=None,condition=c)
            b=self.forward(x=nan_x,source_hidden=hidden,source_contact=None,condition=c)
            blank=self.forward(x=nan_x,source_hidden=hidden,source_contact=None,condition=None)
            changed={k:v.clone() for k,v in c.items()};changed['values'][...,6:]+=10
            d=self.forward(x=nan_x,source_hidden=hidden,source_contact=None,condition=changed)
        self.assertFinite(a);self.assertFinite(b);self.assertFinite(blank);self.assertFinite(d)
        self.assertEqual(delta(a,b),0.)
        self.assertGreater(delta(a,d),1e-7)
        record("pure_condition",all_source_hidden=True,source_contact=None,source_gt_max_abs_delta=delta(a,b),changing_declared_condition_max_abs_delta=delta(a,d),no_condition_finite=True,note="Numerical dependency test on random weights; does not claim learned constraint satisfaction.")

    def test_unknown_contact_nan_and_inf_equals_none_after_nonzero_weights(self):
        with torch.no_grad():
            self.model.con_emb.weight.fill_(2.);self.model.con_emb.bias.fill_(.5)
            baseline=self.model._trunk(self.x,self.sg,self.hidden,None)
            for value in [float('nan'),float('inf'),-float('inf')]:
                con=torch.full((self.B,self.Js),value)
                out=self.model._trunk(self.x,self.sg,self.hidden,con)
                self.assertTrue(bool(torch.isfinite(out).all()));torch.testing.assert_close(out,baseline,atol=0,rtol=0)
        con=torch.full((self.B,self.Js),float('inf'),requires_grad=True)
        out=self.forward(source_hidden=self.hidden,source_contact=con);self.backward_and_check(out)
        self.assertTrue(bool(torch.isfinite(con.grad).all()));self.assertEqual(float(con.grad.abs().max()),0.)
        record("unknown_contacts",tested=["NaN","+Inf","-Inf"],none_equivalence_max_abs=0.,invalid_contact_gradient_max_abs=0.)

    def test_explicit_contact_remains_a_separate_observation(self):
        hidden=torch.ones(self.B,self.Js,dtype=torch.bool)
        with torch.no_grad():
            self.model.con_emb.weight.copy_(torch.linspace(-1,1,32)[:,None])
            a=self.model._trunk(self.x,self.sg,hidden,torch.zeros(self.B,self.Js))
            b=self.model._trunk(self.x,self.sg,hidden,torch.ones(self.B,self.Js))
        self.assertGreater(float((a-b).abs().max()),1e-6)
        record("contact_observation_boundary",hidden_pose_does_not_mask_explicit_contact=True,note="Trainer must explicitly pass None for pure-condition mode; hidden GT contact labels must not be supplied accidentally.")

    def test_metadata_missing_bits_are_shared_and_effective(self):
        marked={k:v.clone() for k,v in self.tg.items()};marked['metadata_missing'][...,0]=1
        with torch.no_grad():
            a=self.model.descriptor(self.tg);b=self.model.descriptor(marked)
        self.assertGreater(float((a-b).abs().max()),0.)
        self.assertEqual(tuple(self.model.metadata_missing_projection.weight.shape),(32,2))
        self.assertIsNotNone(self.model.judge);self.assertFalse(self.model.jf.use_index)
        self.assertIsNot(self.model.fsqp,self.model.fsqr)
        record("shared_topology_parameters",auxiliary_judge_present=True,pose_decoder_shared=True,index_embedding_absent=True,distinct_position_rotation_fsq=True)

    def test_same_model_handles_other_joint_counts_without_new_parameters(self):
        before={n:(id(p),tuple(p.shape)) for n,p in self.model.named_parameters()}
        with torch.no_grad():out=self.model(torch.randn(1,3,9),geometry(1,3),geometry(1,4))
        self.assertEqual(tuple(out[0].shape),(1,4,6));self.assertFinite(out)
        self.assertEqual(before,{n:(id(p),tuple(p.shape)) for n,p in self.model.named_parameters()})

    def test_added_blocks_start_as_exact_identity(self):
        initialize_added_identity_blocks(self.model,first_new=4);errors={}
        with torch.no_grad():
            for family in ['enc','selfp','selfr','decp','decr']:
                block=getattr(self.model,family)[4];q=torch.randn(self.B,self.Js,32)
                out=block(q,self.sg['hop'],None) if family in ['enc','selfp','selfr'] else block(q,torch.randn(self.B,4,32),None)
                errors[family]=float((out-q).abs().max());self.assertEqual(errors[family],0.)
        record("identity_added_blocks",max_abs_by_family=errors)

    def test_identity_added_blocks_receive_output_projection_gradients(self):
        initialize_added_identity_blocks(self.model,first_new=4);gradients={}
        for family in ['enc','selfp','selfr','decp','decr']:
            self.model.zero_grad(set_to_none=True)
            block=getattr(self.model,family)[4];q=torch.randn(self.B,self.Js,32)
            out=block(q,self.sg['hop'],None) if family in ['enc','selfp','selfr'] else block(q,torch.randn(self.B,4,32),None)
            out.square().mean().backward()
            projection=block.a.o if hasattr(block,'a') else block.mha.out_proj
            norms=[float(projection.weight.grad.norm()),float(block.ff[2].weight.grad.norm())]
            self.assertTrue(all(n>0 and n<float('inf') for n in norms));gradients[family]=norms
        record("identity_block_gradient_entry",output_projection_and_ff_norms=gradients,optimizer_step_executed=False)

    def test_auxiliary_hpos_is_returned_raw_not_claimed_xyz(self):
        captured=[];hook=self.model.hpos.register_forward_hook(lambda module,args,out:captured.append(out))
        try:
            with torch.no_grad():out=self.forward()
        finally:hook.remove()
        self.assertTrue(torch.equal(out[1],captured[-1]));self.assertIs(self.model.last_pos,out[1])
        record("auxiliary_head_contract",raw_hpos_returned=True,cartesian_semantics_validated=False,note="Only an interface check; root-height/auxiliary loss vs FK Cartesian loss remains trainer responsibility.")

class SyntheticLegacySchema(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(991)
        cls.kw=dict(d_model=256,n_heads=8,enc_layers=4,dec_layers=4,n_latent=128,n_latent_p=4,fsq_stages=1,fsq_levels=(32,)*13,use_sem=True,use_dof=True,n_judges=16,contact_chan=True,sem_dim=768)
        cls.reference=DualTranslator(**cls.kw).cpu().state_dict()
        cls.reference_hash=hashlib.sha256(''.join(k+str(tuple(v.shape)) for k,v in cls.reference.items()).encode()).hexdigest()

    def target(self):
        return ConditionalMegatron(**dict(self.kw,enc_layers=5,dec_layers=5,n_judges=16,sem_dim=1024)).cpu().eval()

    def test_complete_synthetic_383_key_schema(self):
        self.assertEqual(len(self.reference),383)
        model=self.target();new_sem={k:v.clone() for k,v in model.state_dict().items() if k.startswith('jf.sem_proj.')}
        report=load_reference_weights(model,self.reference);state=model.state_dict()
        self.assertEqual(len(report['loaded_reference_keys'])+len(report['skipped_reference_keys']),383)
        self.assertTrue(all(torch.equal(state[k],self.reference[k]) for k in report['loaded_reference_keys']))
        self.assertTrue(all(torch.equal(state[k],v) for k,v in new_sem.items()))
        self.assertTrue(all(k.startswith(('judge.','jf.sem_proj.')) for k in report['skipped_reference_keys']))
        record("synthetic_reference_schema",key_count=383,loaded_keys=len(report['loaded_reference_keys']),skipped_keys=len(report['skipped_reference_keys']),all_accepted_tensors_exact=True,qwen_projection_preserved=True,real_checkpoint_loaded=False,schema_keys_shapes_sha256=self.reference_hash)

    def test_empty_missing_extra_and_bad_shape_references_rejected(self):
        model=self.target()
        bad=[{}, {k:v for k,v in self.reference.items() if k!='inp.weight'},dict(self.reference,unexplained_extra=torch.zeros(1))]
        wrong=dict(self.reference);wrong['inp.weight']=torch.zeros(1);bad.append(wrong)
        for i,reference in enumerate(bad):
            with self.subTest(case=i),self.assertRaises((ValueError,KeyError)):load_reference_weights(model,reference)
        record("reference_rejections",empty=True,missing_required=True,unexpected_key=True,wrong_shape=True,synthetic_schema_only=True)

class RecordingResult(unittest.TextTestResult):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.records=[];self.started={}
    def startTest(self,test):
        self.started[str(test)]=time.monotonic();super().startTest(test)
    def _record(self,test,status,detail=None):
        self.records.append(dict(test=str(test),status=status,seconds=time.monotonic()-self.started[str(test)],detail=detail))
    def addSuccess(self,test):self._record(test,'passed');super().addSuccess(test)
    def addFailure(self,test,err):self._record(test,'failed',self._exc_info_to_string(err,test));super().addFailure(test,err)
    def addError(self,test,err):self._record(test,'error',self._exc_info_to_string(err,test));super().addError(test,err)
    def addSkip(self,test,reason):self._record(test,'skipped',reason);super().addSkip(test,reason)

def main():
    start=time.monotonic();started=datetime.datetime.now(datetime.timezone.utc).isoformat()
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(PilotNumerics),unittest.defaultTestLoader.loadTestsFromTestCase(SyntheticLegacySchema)])
    result=unittest.TextTestRunner(verbosity=2,resultclass=RecordingResult).run(suite)
    paths=[HERE/'pilot_model.py',Path(__file__),LEGACY/'common.py',LEGACY/'model.py',LEGACY/'model_dual.py']
    report=dict(schema='pilot-model-cpu-numerical-review-v1',started_at_utc=started,finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),wall_seconds=time.monotonic()-start,success=result.wasSuccessful(),tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),skips=len(result.skipped),test_records=result.records,metrics=METRICS,evidence_scope={'device':'CPU','data':'synthetic random tensors and synthetic independent joint graphs','weights':'random initialization only, including the383-key legacy-schema state_dict','real_data_batch_executed':False,'real_checkpoint_loaded':False,'optimizer_created':False,'training_step_executed':False,'diagnostic_backward_executed':True,'gpu_used':False,'network_used':False},runtime={'python':sys.executable,'torch':torch.__version__,'torch_threads':torch.get_num_threads()},source_files=[{'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'sha256_at_import':SOURCE_HASHES_AT_IMPORT.get(str(p))} for p in paths],not_validated=['Real checkpoint numerical compatibility','Real motion-data FK/units/joint-order correctness','Actual trainer loss masks, source-contact observation policy and sparse-SE3 coordinate contract','Real learned constraint following or retargeting quality','Remote model upload/state after host outage'])
    path=HERE/'model_numerical_review.json';path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'report':str(path),'success':report['success'],'tests_run':result.testsRun,'wall_seconds':report['wall_seconds']}))
    return 0 if result.wasSuccessful() else 1
if __name__=='__main__':raise SystemExit(main())
