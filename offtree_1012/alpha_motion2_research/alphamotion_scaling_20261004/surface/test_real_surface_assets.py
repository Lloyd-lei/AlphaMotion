"""CPU diagnostic with actual meshes and neutral geometry, no motion training.

Qwen vectors are explicitly unavailable in this test; no synthetic vectors are
presented as real encoded semantics. Text-gradient numerics live in fixture tests.
"""
from pathlib import Path
import sys,json,hashlib
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import torch
from surface_contract import load_surface_asset,expand_surface,scale_surface_units
from surface_geometry import predict_bound_surface,surface_losses,surface_metrics
from surface_model import SurfaceMegatron
torch.set_num_threads(1)

def geometry(path):
    raw=json.loads(path.read_text());k=raw['kinematic_adapter'];j=len(k['parents'])
    s=load_surface_asset(path,{},j,allow_unencoded=True)
    off=torch.tensor(k['kinematic_rest_off'],dtype=torch.float32)[None]
    desc=off*(100. if raw['length_unit']=='m' else 1.)
    depth=torch.zeros(j,dtype=torch.long);hop=torch.full((j,j),1000,dtype=torch.long);hop.fill_diagonal_(0)
    for n,p in enumerate(k['parents']):
        if p>=0:depth[n]=depth[p]+1;hop[n,p]=hop[p,n]=1
    for n in range(j):hop=torch.minimum(hop,hop[:,n,None]+hop[None,n,:])
    g=dict(rest_off=desc,kinematic_rest_off=off,bonelen=desc.norm(dim=-1)/float(raw['pose_scale']*(100. if raw['length_unit']=='m' else 1.)),
        depth=depth[None],hop=hop[None],sem=torch.zeros(1,j,1024),dof=torch.zeros(1,j,19),
        metadata_missing=torch.tensor([1.,0.]).expand(1,j,2),parents=k['parents'],pose_scale=torch.tensor([raw['pose_scale']]),surface=s)
    for name in ('joint_type','prismatic_axis','slide_lower','slide_upper','slide_reference','kinematic_fixed_rotation'):
        if name in k:g[name]=torch.tensor(k[name],dtype=torch.long if name=='joint_type' else torch.float32)[None]
    if 'root_fixed' in k:g['root_fixed']=k['root_fixed']
    r=torch.tensor(k['rest_joint_rotation'],dtype=torch.float32)[None]
    rot=r[...,:,:2].transpose(-1,-2).reshape(1,j,6)
    return raw,g,rot

def main():
    report={'scope':'Actual 9 mesh assets and geometry-neutral poses; CPU random-weight small model; no optimizer/motion rows/Qwen encoding/grasp success.',
      'qwen_evidence':'unencoded, all semantic_valid=false; never usable as B training cache','bodies':{}}
    for path in sorted((HERE/'patches').glob('*.json')):
        if path.stem=='relay_franka_arm':continue  # Archived v1 is replaced, never double-weighted.
        torch.manual_seed(20261004);raw,g,rot=geometry(path)
        q=g.get('slide_reference');kwargs={'source_slide_normalized':None if q is None else torch.zeros_like(q)}
        pred=predict_bound_surface(rot,g,q);s=g['surface']
        position_error=float((pred['surface']['position']-s['rest_xyz']).abs().max())
        normal_error=float((pred['surface']['normal']-s['rest_normal']).abs().max())
        assert position_error<max(2e-6,float(raw['pose_scale'])*2e-6),(path,position_error)
        assert normal_error<2e-6,(path,normal_error)
        x=torch.cat([rot,pred['joint_position']/g['pose_scale'][:,None,None]],-1)
        model=SurfaceMegatron(d_model=32,n_heads=4,enc_layers=2,dec_layers=2,n_latent=128,n_latent_p=4,
          fsq_stages=1,fsq_levels=(32,)*13,sem_dim=1024,n_judges=12).eval()
        decoded=model.forward_kinematic(x,g,g,**kwargs);out=decoded['rot6']
        loss=surface_losses(out,rot,g,pred_slide_native=decoded['slide_native'],target_slide_native=q);loss['total'].backward()
        grads={name:sum(float(p.grad.abs().sum()) for p in getattr(model,name).parameters() if p.grad is not None)
          for name in ('patch_geometry','patch_motion','target_surface_attention','hrot')}
        assert all(v>0 for v in grads.values()),grads
        with torch.no_grad():
            ablations={}
            for mode in ('drop_surface','static_only','no_attachment_bias'):
                with model.ablate_surface(mode):altered=model.forward_kinematic(x,g,g,**kwargs)['rot6']
                ablations[mode]=float((out-altered).abs().max());assert ablations[mode]>1e-6
            # An occluded source joint cannot influence its bound dynamic patches.
            hidden=torch.zeros(x.shape[:2],dtype=torch.bool);hidden[:,1::2]=True
            masked=x.clone();masked[hidden]=float('nan')
            a=model.forward_kinematic(x,g,g,source_hidden=hidden,**kwargs)['rot6'];b=model.forward_kinematic(masked,g,g,source_hidden=hidden,**kwargs)['rot6']
            occlusion=float((a-b).abs().max());assert occlusion<1e-6
        metrics=surface_metrics(out.detach(),rot,g,pred_slide_native=decoded['slide_native'].detach(),target_slide_native=q)
        if raw['length_unit']=='m':
            cg=dict(g);cg['surface']=scale_surface_units(s,100.,destination_unit='cm');cg['kinematic_rest_off']=g['kinematic_rest_off']*100;cg['pose_scale']=g['pose_scale']*100
            for key in ('slide_lower','slide_upper','slide_reference'):
                if key in cg:cg[key]=cg[key]*100
            cm=surface_metrics(out.detach(),rot,cg,pred_slide_native=decoded['slide_native'].detach()*100,target_slide_native=None if q is None else q*100)
            assert torch.allclose(metrics['position_m_per_frame'],cm['position_m_per_frame'],atol=2e-6)
            assert torch.allclose(metrics['position_normalized_per_frame'],cm['position_normalized_per_frame'],atol=2e-6)
        report['bodies'][path.stem]={'asset_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'joints':len(raw['kinematic_adapter']['parents']),
          'patches':len(raw['patch_ids']),'rest_position_max_abs_native':position_error,'rest_normal_max_abs':normal_error,
          'surface_loss_gradient_l1':grads,'ablation_output_max_abs':ablations,'hidden_source_nan_output_delta':occlusion,
          'cost':model.surface_cost(g,g,2),'common_metric_valid_patches':metrics['active_patches'],
          'source_audit':raw['binding_audit'],'passed':True}
    report['passed']=len(report['bodies'])==9
    (HERE/'real_asset_review.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'passed':report['passed'],'bodies':list(report['bodies'])}))
if __name__=='__main__':main()
