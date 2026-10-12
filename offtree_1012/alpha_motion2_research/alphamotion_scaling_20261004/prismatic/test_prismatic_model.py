"""Actual Franka MJCF-derived states/mesh; CPU diagnostic backward, no training."""
from pathlib import Path
import sys,json,unittest,hashlib
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
for p in (HERE,ROOT/'surface',ROOT/'base',ROOT/'base'/'legacy'):sys.path.insert(0,str(p))
import torch
from common import matrix_to_rot6d,rot6d_to_matrix
from kinematics import mixed_forward_kinematics,decode_slide,derive_joint_rotations
from pilot_model import ConditionalMegatron,model_parameter_counts
from surface_model import SurfaceMegatron,surface_parameter_counts
from surface_contract import load_surface_asset,expand_surface,scale_surface_units
from surface_geometry import predict_bound_surface,surface_losses,surface_metrics
torch.set_num_threads(1)
ASSET=ROOT/'surface'/'patches'/'relay_franka_full_v2.json'
METRICS={}

def fixture(batch=2,surface=True):
    raw=json.loads(ASSET.read_text());k=raw['kinematic_adapter'];j=len(k['parents'])
    off=torch.tensor(k['kinematic_rest_off'],dtype=torch.float32)[None].expand(batch,-1,-1).clone()
    depth=torch.zeros(j,dtype=torch.long);hop=torch.full((j,j),1000,dtype=torch.long);hop.fill_diagonal_(0)
    for n,p in enumerate(k['parents']):
        if p>=0:depth[n]=depth[p]+1;hop[n,p]=hop[p,n]=1
    for n in range(j):hop=torch.minimum(hop,hop[:,n,None]+hop[None,n,:])
    g=dict(rest_off=off*100,kinematic_rest_off=off,bonelen=off.norm(dim=-1)/raw['pose_scale'],depth=depth[None].expand(batch,-1),
      hop=hop[None].expand(batch,-1,-1),sem=torch.zeros(batch,j,1024),dof=torch.zeros(batch,j,19),metadata_missing=torch.zeros(batch,j,2),
      parents=k['parents'],pose_scale=torch.full((batch,),raw['pose_scale']),root_fixed=k['root_fixed'],joint_motion_kind='mixed_tree')
    for key in ('joint_type','prismatic_axis','slide_lower','slide_upper','slide_reference','kinematic_fixed_rotation'):
        value=torch.tensor(k[key],dtype=torch.long if key=='joint_type' else torch.float32);g[key]=value[None].expand(batch,*value.shape)
    if surface:g['surface']=expand_surface(load_surface_asset(ASSET,{},j,allow_unencoded=True),batch)
    states=raw['verification_states'][1:1+batch]
    rot=matrix_to_rot6d(torch.tensor([s['global_rotation'] for s in states],dtype=torch.float32))
    pos=torch.tensor([s['position_native'] for s in states],dtype=torch.float32)
    native=torch.tensor([s['slide_native'] for s in states],dtype=torch.float32)
    valid=g['joint_type']==3;normalized=torch.where(valid,native/.04,torch.zeros_like(native))
    x=torch.cat([rot,pos/g['pose_scale'][:,None,None]],-1)
    return raw,g,x,normalized,native

def model(cls):return cls(d_model=32,n_heads=4,enc_layers=2,dec_layers=2,n_latent=128,n_latent_p=4,fsq_stages=1,fsq_levels=(32,)*13,sem_dim=1024,n_judges=12).eval()

class PrismaticTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(20261004);self.raw,self.g,self.x,self.normalized,self.native=fixture()

    def test_actual_mujoco_eight_states_match_mixed_fk_and_derived_rotations(self):
        errors=[];rotation=[]
        g={k:(v[:1] if isinstance(v,torch.Tensor) else v) for k,v in self.g.items() if k!='surface'}
        for state in self.raw['verification_states']:
            r=torch.tensor(state['global_rotation'],dtype=torch.float32)[None]
            result=mixed_forward_kinematics(matrix_to_rot6d(r),g,torch.tensor(state['slide_native'])[None])
            errors.append(float((result['joint_position']-torch.tensor(state['position_native'])[None]).abs().max()))
            rotation.append(float((result['joint_rotation']-r).abs().max()))
        self.assertLess(max(errors),3e-7);self.assertLess(max(rotation),4e-7)
        METRICS['actual_mujoco_fk_max_m']=max(errors);METRICS['actual_mujoco_rotation_max_abs']=max(rotation)

    def test_slide_logits_obey_actual_limits_and_gt_is_not_clipped(self):
        logits=torch.full((2,11),100.);logits[0]=-100
        normalized,native,valid=decode_slide(logits,self.g)
        self.assertTrue((native[valid]>=0).all());self.assertTrue((native[valid]<=.04).all())
        self.assertTrue((native[~valid]==0).all());self.assertEqual(float(normalized[1,8]),1.)
        q=self.native.clone();q[:,8]=.041
        a=mixed_forward_kinematics(self.x[...,:6],self.g,q)['joint_position'];q[:,8]=.04
        b=mixed_forward_kinematics(self.x[...,:6],self.g,q)['joint_position']
        self.assertTrue(torch.allclose((a[:,8]-b[:,8]).norm(dim=-1),torch.full((2,),.001),atol=2e-7))

    def test_fixed_root_and_slide_rotations_have_no_free_rotation_gradient(self):
        raw=torch.randn(2,11,6,requires_grad=True);out=derive_joint_rotations(raw,self.g)
        r=rot6d_to_matrix(out)
        self.assertTrue(torch.allclose(r[:,0],torch.eye(3).expand(2,3,3)))
        for j in (8,9,10):self.assertTrue(torch.allclose(r[:,j],r[:,7],atol=1e-6))
        out.square().sum().backward();self.assertEqual(float(raw.grad[:,[0,8,9,10]].abs().max()),0.)

    def test_source_visibility_masks_slide_values_and_gradients_A_B(self):
        for cls in (ConditionalMegatron,SurfaceMegatron):
            m=model(cls);hidden=torch.zeros(2,11,dtype=torch.bool);hidden[:,8:10]=True
            changed=self.normalized.clone();changed[:,8:10]=float('nan')
            a=m.forward_kinematic(self.x,self.g,self.g,source_hidden=hidden,source_slide_normalized=self.normalized)
            b=m.forward_kinematic(self.x,self.g,self.g,source_hidden=hidden,source_slide_normalized=changed)
            for key in ('rot6','slide_native'):self.assertTrue(torch.allclose(a[key],b[key],atol=1e-7))
            values=self.normalized.clone().requires_grad_();r=m.forward_kinematic(self.x,self.g,self.g,source_hidden=hidden,source_slide_normalized=values)
            (r['slide_normalized'].sum()+r['joint_position'].square().mean()).backward()
            self.assertEqual(float(values.grad[:,8:10].abs().max()),0.)
            with self.assertRaisesRegex(ValueError,'visible prismatic'):m.forward_kinematic(self.x,self.g,self.g)

    def test_shared_heads_have_q_position_and_surface_gradients(self):
        report={}
        for cls in (ConditionalMegatron,SurfaceMegatron):
            m=model(cls);out=m.forward_kinematic(self.x,self.g,self.g,source_slide_normalized=self.normalized)
            valid=out['slide_valid'];loss=(out['slide_normalized'][valid]-self.normalized[valid]).abs().mean()
            loss=loss+(out['joint_position']-self.x[...,6:]*self.g['pose_scale'][:,None,None]).norm(dim=-1).mean()
            if cls is SurfaceMegatron:loss=loss+surface_losses(out['rot6'],self.x[...,:6],self.g,pred_slide_native=out['slide_native'],target_slide_native=self.native)['total']
            loss.backward();names=['slide_head','source_slide_projection','slide_geometry_projection','joint_type_embedding']
            if cls is SurfaceMegatron:names+=['patch_geometry','patch_motion','target_surface_attention']
            norms={n:sum(float(p.grad.abs().sum()) for p in getattr(m,n).parameters() if p.grad is not None) for n in names}
            self.assertTrue(all(v>0 and v<float('inf') for v in norms.values()),norms);report[cls.__name__]=norms
            first=out['slide_native'].detach().clone();m.forward_kinematic(self.x,self.g,self.g,source_slide_normalized=self.normalized*.8)
            self.assertTrue(torch.equal(out['slide_native'].detach(),first))
        METRICS['actual_asset_backward_gradient_l1']=report

    def test_aperture_independent_fingers_and_dynamic_mesh_translation(self):
        q=torch.zeros_like(self.native);q[:,8]=.01;q[:,9]=.025
        result=predict_bound_surface(self.x[...,:6],self.g,q)
        left_axis=torch.einsum('bij,bj->bi',result['joint_rotation'][:,7],self.g['prismatic_axis'][:,8])
        aperture=((result['joint_position'][:,8]-result['joint_position'][:,9])*left_axis).sum(-1)
        self.assertTrue(torch.allclose(aperture,q[:,8]+q[:,9],atol=2e-7))
        q2=q.clone();q2[:,8]+=.012
        moved=predict_bound_surface(self.x[...,:6],self.g,q2)
        self.assertTrue(torch.equal(result['joint_position'][:,9],moved['joint_position'][:,9]))
        anchor=self.g['surface']['binding_weights'].argmax(-1)
        left=anchor==8;right=anchor==9;self.assertTrue(left.any() and right.any())
        delta=(moved['surface']['position']-result['surface']['position']).norm(dim=-1)
        self.assertTrue(torch.allclose(delta[left],torch.full_like(delta[left],.012),atol=2e-7));self.assertTrue((delta[right]==0).all())
        self.assertTrue(torch.allclose(moved['surface']['normal'],result['surface']['normal'],atol=1e-7))
        METRICS['finger_patch_counts']={'left':int(left[0].sum()),'right':int(right[0].sum())};METRICS['aperture_max_error_m']=float((aperture-q[:,8]-q[:,9]).abs().max())

    def test_surface_q_gradient_required_explicit_target_and_common_units(self):
        q=self.native.clone().requires_grad_();target=self.native*.7
        loss=surface_losses(self.x[...,:6],self.x[...,:6],self.g,pred_slide_native=q,target_slide_native=target)
        loss['total'].backward();self.assertGreater(float(q.grad[:,8:10].abs().sum()),0.)
        with self.assertRaisesRegex(ValueError,'explicit slide_native'):surface_losses(self.x[...,:6],self.x[...,:6],self.g,pred_slide_native=q)
        native=surface_metrics(self.x[...,:6],self.x[...,:6],self.g,pred_slide_native=q.detach(),target_slide_native=target)
        g=dict(self.g);g['surface']=scale_surface_units(g['surface'],100,destination_unit='cm')
        for name in ('kinematic_rest_off','pose_scale','slide_lower','slide_upper','slide_reference'):g[name]=g[name]*100
        cm=surface_metrics(self.x[...,:6],self.x[...,:6],g,pred_slide_native=q.detach()*100,target_slide_native=target*100)
        self.assertTrue(torch.allclose(native['position_m_per_frame'],cm['position_m_per_frame'],atol=1e-7))

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PrismaticTests))
    report={'passed':result.wasSuccessful(),'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
      'asset_sha256':hashlib.sha256(ASSET.read_bytes()).hexdigest(),'metrics':METRICS,
      'evidence':'Real source MJCF-derived neutral/random q states and meshes. CPU small random-weight A/B backward; no optimizer, real Qwen cache, motion training or task-success test.'}
    (HERE/'numerical_review.json').write_text(json.dumps(report,indent=2)+'\n');sys.exit(not result.wasSuccessful())
