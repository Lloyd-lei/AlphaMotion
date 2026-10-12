"""Run on CPU now; root runs --device cuda --batch-size 128 before training.

Actual Franka MJCF-derived states and meshes, explicit unencoded text fixture.
Checks FP32 geometry parity within BF16 AMP plus actual A/B model backward.
No optimizer/update or stateful service is created.
"""
from pathlib import Path
import sys,json,argparse,hashlib
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import torch
from test_prismatic_model import fixture,model,ConditionalMegatron,SurfaceMegatron,ASSET
from kinematics import joint_metadata,derive_joint_rotations,mixed_forward_kinematics
from surface_geometry import bound_surface,predict_bound_surface,surface_losses,surface_metrics,fk_global_positions


def run(device,batch):
    device=torch.device(device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable; no CPU fallback')
    raw,g,x,q01,q=fixture(batch=2)
    ix=torch.arange(batch)%2
    def move(value):
        if isinstance(value,dict):return {k:move(v) for k,v in value.items()}
        if isinstance(value,torch.Tensor):
            if value.ndim and value.shape[0]==2:value=value[ix]
            return value.to(device)
        return value
    g=move(g);x=move(x);q01=move(q01);q=move(q)
    # Same FP32 tensor values must produce exactly the same FP32 geometric
    # values/gradients inside and outside AMP. This detects silent bf16 matmul,
    # not only the original dtype mismatch exception.
    def geometric(amp):
        rot=x[...,:6].detach().clone().requires_grad_();slide=q.detach().clone().requires_grad_()
        with torch.autocast(device.type,dtype=torch.bfloat16,enabled=amp):
            meta=joint_metadata(g,batch,11,device)
            assert meta['fixed_rotation'].dtype==torch.float32
            fk=mixed_forward_kinematics(rot,g,slide)
            bound=bound_surface(fk['rot6'],fk['joint_position'],g['surface'])
            metric=surface_metrics(rot,rot,g,pred_slide_native=slide,target_slide_native=q*.7)
            loss=surface_losses(rot,rot,g,pred_slide_native=slide,target_slide_native=q*.7)
            total=loss['total']+fk['joint_position'].square().mean()+bound['normal'].square().mean()
        total.backward()
        values={'rotation':fk['joint_rotation'],'position':fk['joint_position'],'bound_position':bound['position'],
          'bound_normal':bound['normal'],'loss':loss['total'],'metric':metric['position_m_per_frame'],
          'rotation_gradient':rot.grad,'slide_gradient':slide.grad}
        assert all(v.dtype==torch.float32 and torch.isfinite(v).all() for v in values.values())
        return {k:v.detach() for k,v in values.items()}
    ref=geometric(False);amp=geometric(True)
    parity={k:float((ref[k]-amp[k]).abs().max()) for k in ref}
    assert max(parity.values())<=1e-7,parity
    # Incoming bf16 heads are promoted before geometry, even with no derived R.
    with torch.autocast(device.type,dtype=torch.bfloat16):
        assert derive_joint_rotations(x[...,:6].bfloat16(),{}).dtype==torch.float32
        assert fk_global_positions(x[...,:6].bfloat16(),g['parents'],g['kinematic_rest_off']).dtype==torch.float32
        incoming=predict_bound_surface(x[...,:6].bfloat16(),g,q.bfloat16())
        assert incoming['joint_position'].dtype==torch.float32 and incoming['surface']['normal'].dtype==torch.float32
    neural={}
    for cls in (ConditionalMegatron,SurfaceMegatron):
        torch.manual_seed(20261004);m=model(cls).to(device);observed={}
        def hook(name):
            def record(module,args,out):observed[name]=str(out.dtype)
            return record
        handles=[m.inp.register_forward_hook(hook('source_projection')),m.hrot.register_forward_hook(hook('rotation_head')),m.slide_head.register_forward_hook(hook('slide_head'))]
        with torch.autocast(device.type,dtype=torch.bfloat16):
            prediction=m.forward_kinematic(x,g,g,source_slide_normalized=q01)
            valid=prediction['slide_valid']
            loss=(prediction['slide_normalized'][valid]-q01[valid]).abs().mean()
            loss=loss+(prediction['joint_position']-x[...,6:]*g['pose_scale'][:,None,None]).square().mean()
            surf=surface_losses(prediction['rot6'],x[...,:6],g,pred_slide_native=prediction['slide_native'],target_slide_native=q)
            loss=loss+surf['total']
        loss.backward()
        for h in handles:h.remove()
        assert all(dtype=='torch.bfloat16' for dtype in observed.values()),observed
        for field in ('rot6','slide_native','slide_normalized','joint_position','joint_rotation'):
            assert prediction[field].dtype==torch.float32 and torch.isfinite(prediction[field]).all(),field
        names=['slide_head','source_slide_projection']+(['patch_geometry','patch_motion'] if cls is SurfaceMegatron else [])
        gradients={name:sum(float(p.grad.abs().sum()) for p in getattr(m,name).parameters() if p.grad is not None) for name in names}
        assert all(v>0 and v<float('inf') for v in gradients.values()),gradients
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
        neural[cls.__name__]={'neural_projection_dtypes':observed,'geometry_dtype':'torch.float32','loss':float(loss.detach()),'gradient_l1':gradients}
        del m
    return {'passed':True,'device':str(device),'torch_version':torch.__version__,'batch_size':batch,'autocast_dtype':'torch.bfloat16',
      'geometry_parity_max_abs':parity,'models':neural,'asset_sha256':hashlib.sha256(ASSET.read_bytes()).hexdigest(),
      'evidence':'Real Franka MJCF-derived states/mesh, small random-weight A/B models, unencoded text fixture. Diagnostic backward only; no optimizer/update.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cpu');parser.add_argument('--batch-size',type=int,default=2);parser.add_argument('--output')
    args=parser.parse_args();torch.set_num_threads(2)
    result=run(args.device,args.batch_size)
    dest=Path(args.output) if args.output else HERE/f'autocast_geometry_{torch.device(args.device).type}_review.json'
    dest.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
