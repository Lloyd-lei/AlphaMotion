"""Read-only CPU metamorphic audit. Writes reports only; no optimizer/GPU/SSH.

Default: real mesh/MJCF geometry fixtures, synthetic contact labels clearly marked.
--actual-root PATH additionally tests actual XWBC payloads via the sealed loader.
--checkpoint PATH optionally audits an actual B checkpoint, on CPU only.
"""
from pathlib import Path
import os,sys,json,argparse,copy,hashlib
os.environ['CUDA_VISIBLE_DEVICES']='';sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT,ROOT/'base',ROOT/'base'/'legacy',ROOT/'surface',ROOT/'prismatic',ROOT/'orchestration'):sys.path.insert(0,str(p))
import numpy as np
import torch
from pilot_data import make_inputs
from recipe_training import make_reference_inputs
from surface_model import SurfaceMegatron
from test_prismatic_model import fixture
from test_real_surface_assets import geometry
from surface_geometry import predict_bound_surface
torch.set_num_threads(2)


class FixtureBody:
    def __init__(self,name,g,pose,slide=None):
        self.name=name;self.domain='xwbc';self.parents=np.asarray(g['parents']);off=g['kinematic_rest_off'];self.offsets=(off[0] if off.ndim==3 else off).numpy();self.names=[f'joint_{j}' for j in range(pose.shape[1])]
        self.arrays={'pose9':pose.clone(),'contact_sdist':torch.full(pose.shape[:2],.013),
          'contact_valid':torch.ones(pose.shape[:2],dtype=torch.bool),'contact_soft':torch.full(pose.shape[:2],.8),
          'contact_hard':torch.ones(pose.shape[:2],dtype=torch.bool)}
        if slide is not None:self.arrays['slide_normalized']=slide.clone()
        self.endpoint={'indices':[pose.shape[1]-2,pose.shape[1]-1]};self.g=g
    def tensor(self,key,rows,device):return torch.as_tensor(self.arrays[key])[rows].to(device)
    def ground_batch(self,rows,device):return {k:self.tensor(k,rows,device) for k in ('contact_valid','contact_soft','contact_hard')}
    def geom(self,batch):return self.g


def changed_body(body):
    out=copy.copy(body);out.arrays=dict(body.arrays);return out


def captured(model,inputs,source_g,target_g):
    captures={};handles=[]
    for name in ('inp','con_emb','enc.0','source_slide_projection','patch_motion','patch_geometry','patch_semantic','target_surface_attention','condition_projection'):
        def hook(module,args,name=name):
            captures.setdefault(name,[]).append(tuple(v.detach().clone() for v in args if isinstance(v,torch.Tensor)))
        handles.append(model.get_submodule(name).register_forward_pre_hook(hook))
    with torch.no_grad():output=model.forward_kinematic(**inputs,source_geom=source_g,target_geom=target_g)
    for h in handles:h.remove()
    return {k:v.detach().clone() for k,v in output.items() if isinstance(v,torch.Tensor)},captures


def compare(left,right):
    result={}
    for key in left:
        a,b=left[key],right[key]
        if a.dtype==torch.bool:result[key]=int((a!=b).sum())
        else:result[key]=float((a-b).abs().max())
    return result


def compare_hooks(left,right):
    out={}
    assert left.keys()==right.keys()
    for key in left:
        assert len(left[key])==len(right[key])
        vals=[]
        for a,b in zip(left[key],right[key]):
            for x,y in zip(a,b):vals.append(int((x!=y).sum()) if x.dtype==torch.bool else float((x-y).abs().max()))
        out[key]=max(vals,default=0.)
    return out


def pure_and_masked(model,actual_body=None):
    if actual_body is None:
        raw,g,x,q01,q=fixture();src=FixtureBody('Franka_actual_geometry_contact_stress_fixture',g,x,q01)
    else:
        rr=actual_body.eligible(True)[:2];g=actual_body.geom(2);x=actual_body.tensor('pose9',rr,torch.device('cpu'));q01=actual_body.tensor('slide_normalized',rr,torch.device('cpu'))
        src=FixtureBody(actual_body.name,g,x,q01)
        for k in ('contact_sdist','contact_valid','contact_soft','contact_hard'):src.arrays[k]=actual_body.tensor(k,rr,torch.device('cpu'))
    tgt=changed_body(src)
    result={};device=torch.device('cpu');rows=np.arange(2)
    for mode,count in (('sparse_only',2),('masked',2),('full',0),('full',2)):
        seed=193;inp,_=make_inputs(src,tgt,rows,np.random.default_rng(seed),device,mode,count,condition_dropout=0.,contact_dropout=0.)
        hidden=inp['source_hidden'];changed_src=changed_body(src);changed_tgt=changed_body(tgt)
        px=x.clone();px[hidden]=float('nan');changed_src.arrays['pose9']=px
        sq=q01.clone();sq[hidden]=float('nan');changed_src.arrays['slide_normalized']=sq
        contact=src.arrays['contact_sdist'].clone();contact[hidden]+=9.;changed_src.arrays['contact_sdist']=contact
        target=x.clone();visible=torch.zeros(x.shape[:2],dtype=torch.bool)
        for i in range(2):visible[i,inp['condition']['joint_indices'][i,inp['condition']['valid'][i]]]=True
        target[~visible]=float('nan');changed_tgt.arrays['pose9']=target
        changed_tgt.arrays['contact_sdist']=torch.full_like(contact,float('nan'));changed_tgt.arrays['slide_normalized']=torch.full_like(q01,float('nan'))
        altered,_=make_inputs(changed_src,changed_tgt,rows,np.random.default_rng(seed),device,mode,count,condition_dropout=0.,contact_dropout=0.)
        for key in ('joint_indices','valid','values'):torch.testing.assert_close(inp['condition'][key],altered['condition'][key],equal_nan=True)
        assert torch.equal(inp['source_hidden'],altered['source_hidden'])
        if mode=='sparse_only':assert inp['source_contact'] is None and altered['source_contact'] is None
        else:torch.testing.assert_close(inp['source_contact'],altered['source_contact'],equal_nan=True)
        # Poison unsupported dynamic fields to confirm query reads only allowlisted
        # static fields. Actual Body.geom also constructs only static fields.
        poisoned=dict(g);poisoned['surface']=dict(g['surface'])
        for key in ('dynamic_position','dynamic_normal','contact','pressure','slide_native','cached_pose'):
            poisoned[key]=torch.full((2,11,3),float('nan'));poisoned['surface'][key]=torch.full((2,96,3),float('nan'))
        before,h1=captured(model,inp,g,g);after,h2=captured(model,altered,g,poisoned)
        delta=compare(before,after);hooks=compare_hooks(h1,h2)
        assert max(delta.values())==0. and max(hooks.values())==0.,(mode,delta,hooks)
        # A preceding unrelated forward/legacy last_pos does not change this query.
        other=dict(inp);other['condition']={**inp['condition'],'values':inp['condition']['values']*.37}
        captured(model,other,g,g);model.last_pos=torch.full((2,11,3),float('nan'))
        again,_=captured(model,inp,g,g);assert max(compare(before,again).values())==0.
        result[f'{mode}_condition{count}']={'passed':True,'hidden_joint_frames':int(hidden.sum()),'all_output_max_abs':delta,
          'effective_input_hook_max_abs':hooks,'dynamic_cache_interleaving_invariant':True,
          'raw_source_payload_invariant':bool(torch.equal(inp['source'],altered['source'])),
          'raw_source_note':'raw masked pose/q may contain hidden GT; masking makes effective inputs invariant; full-source observations held fixed'}
    return result


def reference_contact_probe(model,body,config,rows):
    device=torch.device('cpu');result={};chosen=None
    for seed in range(32):
        inp,_=make_reference_inputs(body,body,rows,np.random.default_rng(seed),device,config)
        eligible=inp['source_hidden']&torch.isfinite(inp['source_contact'])
        if bool(eligible.any()):chosen=(seed,inp,eligible);break
    if chosen is None:return {'tested':False,'reason':'no hidden valid contact in first32 deterministic seeds'}
    seed,inp,eligible=chosen;altered_body=changed_body(body)
    a=body.arrays['contact_sdist'];a=a.clone() if isinstance(a,torch.Tensor) else np.array(a,copy=True)
    for i,row in enumerate(rows):
        mask=eligible[i].cpu().numpy();a[row,mask]=a[row,mask]+.05
    altered_body.arrays['contact_sdist']=a
    changed,_=make_reference_inputs(altered_body,body,rows,np.random.default_rng(seed),device,config)
    assert torch.equal(inp['source_hidden'],changed['source_hidden'])
    assert torch.equal(inp['source'],changed['source'])
    for key in ('joint_indices','valid','values'):torch.testing.assert_close(inp['condition'][key],changed['condition'][key],equal_nan=True)
    g=body.geom(len(rows));before,h1=captured(model,inp,g,g);after,h2=captured(model,changed,g,g)
    result={'tested':True,'seed':seed,'rows':len(rows),'changed_hidden_contact_joint_frames':int(eligible.sum()),
      'visible_pose_condition_mask_static_geometry_unchanged':True,'contact_sdist_delta':.05,
      'all_output_max_abs':compare(before,after),'effective_input_hook_max_abs':compare_hooks(h1,h2)}
    result['status']='observed_contact_extra_task'
    result['contact_input_dependency_detected']=result['effective_input_hook_max_abs'].get('con_emb',0.)>0.
    result['encoded_token_dependency_detected']=result['effective_input_hook_max_abs'].get('enc.0',0.)>0.
    result['contact_output_dependency_detected']=max(result['all_output_max_abs'].values())>0.
    result['contact_embedding_weight_l1']=float(model.con_emb.weight.detach().abs().sum())
    result['output_caveat']='Zero-initialized contact projection or FSQ plateaus can hide output sensitivity; contact pre-hook and first encoder inputs are the direct dependency evidence.'
    result['interpretation']='Declared additional observed-contact task, shared A/B. This is not the pure-SE3 protocol and is not treated as its failure.'
    # Frequency is a deterministic sampler probe, not live training log telemetry.
    hidden_rows=hidden_joints=hidden_valid=sampled=0
    for s in range(32):
        i,_=make_reference_inputs(body,body,rows,np.random.default_rng(s),device,config)
        h=i['source_hidden'];sampled+=len(rows);hidden_rows+=int(h.any(1).sum());hidden_joints+=int(h.sum());hidden_valid+=int((h&torch.isfinite(i['source_contact'])).sum())
    result['sampler_probe']={'rows':sampled,'hidden_rows':hidden_rows,'hidden_joint_frames':hidden_joints,'retained_hidden_contact_joint_frames':hidden_valid,'live_training_frequency':False}
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',default=str(ROOT/'orchestration/configs/B_S_seed0.json'));ap.add_argument('--actual-root');ap.add_argument('--checkpoint');ap.add_argument('--output',default=str(Path(__file__).with_suffix('.json')));a=ap.parse_args()
    config=json.loads(Path(a.config).read_text());torch.manual_seed(20261005)
    if a.checkpoint:
        checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=True);config=checkpoint['config']
        model=SurfaceMegatron(**config['model'],max_patches=config['surface']['max_patches']);model.load_state_dict(checkpoint['model'],strict=True)
    else:model=SurfaceMegatron(d_model=32,n_heads=4,enc_layers=2,dec_layers=2,n_latent=128,n_latent_p=4,fsq_stages=1,fsq_levels=(32,)*13,sem_dim=1024,n_judges=12)
    model.eval();result={}
    if a.actual_root:
        from scaling_data import load_scaling_bodies
        bodies,_=load_scaling_bodies(Path(a.actual_root),config,torch.device('cpu'),include_surface=True)
        result['pure_and_generic_masked']=pure_and_masked(model,next(b for b in bodies if b.name=='relay_franka_full_v2'))
        result['strict_protocol_provenance']='Actual sealed Franka full_v2 rows, Qwen and static mesh; target GT and hidden source mutations are synthetic audit interventions.'
        body=next(b for b in bodies if b.name=='unitree_g1_29dof');rows=body.eligible(True)[:16];provenance='actual sealed XWBC G1 rows, qwen and geometry via sealed loader'
    else:
        result['pure_and_generic_masked']=pure_and_masked(model)
        result['strict_protocol_provenance']='Actual Franka MJCF geometry/state fixture and unencoded text; contact labels are synthetic stress controls.'
        raw,g,rot=geometry(ROOT/'surface/patches/unitree_g1_29dof.json');g={k:(v.expand(16,*v.shape[1:]) if isinstance(v,torch.Tensor) and v.ndim and v.shape[0]==1 else v) for k,v in g.items()}
        from surface_contract import expand_surface
        g['surface']=expand_surface(g['surface'],16);rot=rot.expand(16,-1,-1)
        x=torch.cat([rot,predict_bound_surface(rot,g)['joint_position']/g['pose_scale'][:,None,None]],-1)
        body=FixtureBody('unitree_g1_29dof',g,x);body.names=raw['kinematic_adapter']['joint_names'];rows=np.arange(16);provenance='actual G1 geometry, neutral pose and synthetic all-valid contact stress labels; not recorded contact evidence'
    result['reference_recipe_contact']=reference_contact_probe(model,body,config,rows)
    result.update(scope='CPU inference/read-only audit, no optimizer/GPU/SSH; frozen training sources unchanged',contact_fixture_provenance=provenance,
      model='actual supplied checkpoint' if a.checkpoint else 'small random-weight B; structural dependency probe',
      configured_frequency={'xwbc_step_probability':config['mixture']['xwbc'],'directions_per_xwbc_step':config['recipe']['combo_sample'],'mask_row_gate_probability':config['recipe']['mask_prob'],'mask_mix':config['recipe']['mask_mix'],'contact_dropout_bypassed_for_hidden_rows':True,'live_training_frequency':'not measured; requires existing run direction logs'},
      criterion='pure/generic masked/full-source target-GT mutation must pass; reference observed-contact extra task is permitted separately and never labeled SE3-only',
      code_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'base/pilot_data.py',ROOT/'base/recipe_training.py',ROOT/'base/pilot_model.py',ROOT/'surface/surface_model.py',ROOT/'prismatic/kinematics.py',ROOT/'orchestration/scaling_data.py',ROOT/'orchestration/train_scaling.py']})
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
