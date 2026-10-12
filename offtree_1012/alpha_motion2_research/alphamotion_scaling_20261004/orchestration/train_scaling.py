"""Six from-scratch controlled runs, one Linux flock owner, evidence-bound gates."""
from pathlib import Path
import argparse,datetime,fcntl,gc,hashlib,json,os,random,subprocess,sys,time,traceback
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'base'));sys.path.insert(0,str(ROOT))
from pilot_data import make_inputs,sample_rows,fk,angle_degrees,read_json,write_json,sha256
from pilot_model import ConditionalMegatron
from surface.surface_model import SurfaceMegatron
from surface.surface_geometry import surface_losses,surface_metrics
from train_pilot import autocast,heldout_rows,array_stats,process_identity
from recipe_training import backward_training_step
from remote_state import collect,capacity
from scaling_data import load_scaling_bodies
from compute_accounting import ComputeCounter,effective_tokens

def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def digest(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()

def source_hashes():
    return {str(p.relative_to(ROOT)):sha256(p) for folder in ['base','surface','prismatic','orchestration'] for p in sorted((ROOT/folder).rglob('*.py')) if '__pycache__' not in p.parts}

def forward(model,src,tgt,inputs):
    return model.forward_kinematic(**inputs,source_geom=src.geom(len(inputs['source'])),target_geom=tgt.geom(len(inputs['source'])))

def gpu_snapshot():
    return {k:subprocess.check_output(['nvidia-smi',q,'--format=csv,noheader'],text=True).strip() for k,q in [
        ('gpu','--query-gpu=name,memory.used,memory.free,utilization.gpu'),('processes','--query-compute-apps=pid,process_name,used_gpu_memory')]}

def guard():
    cap=capacity();required=int(cap['remaining_bytes'])+31*1024**3
    if int(cap['available_bytes'])<required:raise RuntimeError(f'combined capacity guard: free {cap["available_bytes"]}, required {required}')
    return {'utc':utc(),'free_bytes':cap['available_bytes'],'remaining_fixed_download_bytes':cap['remaining_bytes'],
        'reserved_bytes':16*1024**3,'experiment_allocation_bytes':15*1024**3,'required_bytes':required}

def build_model(config,device):
    torch.manual_seed(config['seed']);np.random.seed(config['seed']);random.seed(config['seed'])
    cls=SurfaceMegatron if config['group']=='B' else ConditionalMegatron
    kw={'max_patches':config['surface']['max_patches']} if config['group']=='B' else {}
    model=cls(**config['model'],**kw).to(device)
    model.set_judge_body_order(config['judge_body_order']);return model

def loss_function(config):
    def objective(outputs,target,tgt,rows,condition,weights):
        rot=outputs['rot6'].float();aux=outputs['raw_hpos'].float();device=rot.device;a=tgt.arrays
        predicted=outputs['joint_position']/tgt.reach
        pe=(predicted-target[...,6:]).norm(dim=-1);re=angle_degrees(rot,target[...,:6],training=True)
        terms={'rotation_deg':re.mean(),'position_over_reach':pe.mean()}
        loss=terms['rotation_deg']/30+weights['position']*terms['position_over_reach']
        ix=condition['joint_indices'];v=condition['valid'];batch=torch.arange(len(rows),device=device)[:,None].expand_as(ix)
        if bool(v.any()):
            cp=pe[batch[v],ix[v]].mean();cr=re[batch[v],ix[v]].mean()
            terms.update(condition_position_over_reach=cp,condition_rotation_deg=cr)
            loss=loss+weights['condition_position']*cp+weights['condition_rotation']*cr/30
        rv=tgt.tensor('root_height_valid',rows,device).bool();height=tgt.tensor('root_height_m',rows,device).float()
        rv=rv&torch.isfinite(height)
        if bool(rv.any()):
            rh=torch.nn.functional.smooth_l1_loss(aux[rv,0,0],height[rv]);terms['root_height_smooth_l1_m']=rh
            loss=loss+weights['root_height']*rh
        from recipe_ground import ground_losses
        ground=ground_losses(rot,predicted*tgt.reach,aux[:,0,0],tgt.ground_batch(rows,device),
            w_pen=weights.get('penetration',0),w_cgeo=weights.get('contact_geometry',0),w_rootpos=0.)
        loss=loss+ground['total'];terms.update(ground['terms'])
        sv=outputs['slide_valid']
        if bool(sv.any()):
            slide_target=tgt.tensor('slide_normalized',rows,device).float()
            sl=(outputs['slide_normalized'][sv]-slide_target[sv]).abs().mean()
            terms['slide_normalized_l1']=sl;loss=loss+weights['slide_normalized']*sl
        counts={'rotation_joint_frames':len(rows)*len(tgt.parents),'position_joint_frames':len(rows)*len(tgt.parents),
            'condition_joint_frames':int(condition['valid'].sum()),'root_height_frames':int(a['root_height_valid'][rows].sum()),
            'contact_joint_frames':int(a['contact_valid'][rows].sum()),'ground_frames':int(a['ground_valid'][rows].sum()),
            'surface_patch_frames':0,'surface_ground_frames':0,'surface_contact_points':0,'slide_joint_frames':int(sv.sum())}
        if config['group']=='B' and tgt._surface_static is not None:
            w=config['surface']['weights']
            # Valid world ground requires original ground datum AND a trusted root height.
            gv=tgt.tensor('ground_valid',rows,device).bool() & tgt.tensor('root_height_valid',rows,device).bool()
            s=surface_losses(rot.float(),target[...,:6].float(),tgt.geom(len(rows)),
                pred_slide_native=outputs['slide_native'],target_slide_native=tgt.tensor('slide_native',rows,device) if 'slide_native' in a else None,
                root_height_m=aux.float()[:,0,0],root_height_valid=gv,
                w_position=w['position'],w_normal=w['normal'],w_penetration=w['penetration'],w_contact=w['contact'])
            loss=loss+s['total']
            terms.update({'surface_'+k:s[k] for k in ['position','normal','penetration','contact']})
            counts.update(surface_patch_frames=s['diagnostics']['active_patches'],surface_ground_frames=s['diagnostics']['ground_frames'],surface_contact_points=s['diagnostics']['contact_points'])
        terms.update({'count_'+k:rot.new_tensor(v,dtype=torch.float32) for k,v in counts.items()})
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite combined loss')
        return loss,terms
    return objective

def finite_backward(model):
    grads=[p.grad for p in model.parameters() if p.grad is not None]
    if not grads or not all(bool(torch.isfinite(g).all()) for g in grads):raise FloatingPointError('invalid full-recipe gradient')
    return float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True))

def validate_model(model,bodies,config,device):
    reports=[];obj=loss_function(config)
    for domain in ['xwbc','unimate']:
        model.zero_grad(set_to_none=True);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.monotonic()
        report=backward_training_step(model,bodies,config,np.random.default_rng(901),np.random.default_rng(902),np.random.default_rng(903),
            device=device,domain=domain,forward_fn=forward,objective_fn=obj,autocast_fn=autocast,force_all_losses=True)
        grad=finite_backward(model);torch.cuda.synchronize()
        reports.append({'kind':'full_recipe','domain':domain,'seconds':time.monotonic()-start,'peak_cuda_bytes':torch.cuda.max_memory_allocated(),
            'gradient_norm':grad,'report':report,'optimizer_updates':0})
    # Actual largest topology, hand, arm and bound mesh; source-hidden invariance.
    selected={max((b for b in bodies if len(b.eligible(True))),key=lambda b:len(b.parents)).name,
              'adroit_pen','relay_franka_full_v2','unitree_g1_29dof'}
    for b in bodies:
        if b.name not in selected:continue
        rng=np.random.default_rng(811);rows=sample_rows(b,rng,config['batch_size'])
        if b.name=='relay_franka_full_v2':
            visible,visible_target=make_inputs(b,b,rows,rng,device,'full',0)
            model.zero_grad(set_to_none=True)
            with autocast(device):visible_out=forward(model,b,b,visible)
            visible_loss,_=obj(visible_out,visible_target,b,rows,visible['condition'],config['loss']);visible_loss.backward();visible_grad=finite_backward(model)
            slide_grads={prefix:sum(float(p.grad.abs().sum()) for n,p in model.named_parameters() if n.startswith(prefix) and p.grad is not None) for prefix in ['source_slide_projection.','slide_geometry_projection.','slide_head.']}
            if not all(v>0 and np.isfinite(v) for v in slide_grads.values()):raise RuntimeError('actual prismatic gradient path inactive')
            reports.append({'kind':'visible_source_prismatic','body':b.name,'batch':len(rows),'loss':float(visible_loss.detach()),
                'gradient_norm':visible_grad,'slide_gradient_sums':slide_grads,'optimizer_updates':0})
        inp,target=make_inputs(b,b,rows,rng,device,'sparse_only',min(2,len(b.endpoint['indices'])))
        model.zero_grad(set_to_none=True)
        with autocast(device):out=forward(model,b,b,inp)
        loss,terms=obj(out,target,b,rows,inp['condition'],config['loss']);loss.backward();grad=finite_backward(model)
        surface_grad={n:float(p.grad.abs().sum()) for n,p in model.named_parameters() if p.grad is not None and any(x in n for x in ['surface_encoder','surface_decoder','patch','attachment'])}
        model.eval()
        with torch.no_grad(),autocast(device):
            y=forward(model,b,b,inp)['joint_position'];inp['source']=torch.full_like(inp['source'],float('nan'))
            if 'source_slide_normalized' in inp:inp['source_slide_normalized']=torch.full_like(inp['source_slide_normalized'],float('nan'))
            z=forward(model,b,b,inp)['joint_position']
        if not torch.equal(y,z):raise RuntimeError('hidden answer changed sparse-only output')
        model.train()
        reports.append({'kind':'real_body_batch','body':b.name,'batch':len(rows),'loss':float(loss.detach()),'gradient_norm':grad,
            'surface_gradient_sums':surface_grad,'hidden_source_nan_invariant':True,'optimizer_updates':0})
    model.zero_grad(set_to_none=True)
    if config['group']=='B' and not any(any(x>0 for x in r.get('surface_gradient_sums',{}).values()) for r in reports):raise RuntimeError('surface gradient path inactive')
    return {'status':'passed','optimizer_updates':0,'profiles':reports}

def evaluate(model,bodies,config,folder,label,quick=False):
    """Deterministic common-frame/no-SE3/SE3 evaluation; GT enters only metrics."""
    folder.mkdir(exist_ok=True);device=next(model.parameters()).device;was_training=model.training;model.eval();cases=[]
    human=next(b for b in bodies if b.name=='human_smpl')
    names={'unitree_g1_29dof','kuavo_s45','e457402d4b2e4681b5a5f88b747aea4a','79fed889082e4a25ad92be75b57f641c','adroit_pen','relay_franka_full_v2'}
    candidates=[b for b in bodies if len(b.eligible(False)) and (not quick or b.name in names)]
    with torch.inference_mode():
        for bi,tgt in enumerate(candidates):
            rows=heldout_rows(tgt,64 if quick else config['eval']['frames_per_body'],731)
            for mode in ['full','sparse_only']:
                if mode=='sparse_only' and not tgt.endpoint['indices']:continue
                src=human if tgt.domain=='xwbc' and mode=='full' else tgt
                counts=sorted(set([0,min(2,len(tgt.endpoint['indices']))]))
                for count in counts:
                    pes=[];res=[];selecteds=[];surfs=[];surfns=[];preds=[];slide_errors=[];aperture_errors=[]
                    for start in range(0,len(rows),config['eval']['batch_size']):
                        rr=rows[start:start+config['eval']['batch_size']]
                        inp,target=make_inputs(src,tgt,rr,np.random.default_rng(912+start),device,mode,count,contact_dropout=1.)
                        with autocast(device):out=forward(model,src,tgt,inp)
                        rot=out['rot6'];aux=out['raw_hpos']
                        pe=(out['joint_position']/tgt.reach-target[...,6:]).norm(dim=-1)
                        re=angle_degrees(rot,target[...,:6]);selected=torch.zeros_like(pe,dtype=torch.bool)
                        for i in range(len(rr)):selected[i,inp['condition']['joint_indices'][i,inp['condition']['valid'][i]]]=True
                        pes.append(pe.cpu().numpy());res.append(re.cpu().numpy());selecteds.append(selected.cpu().numpy())
                        gt_slide=tgt.tensor('slide_native',rr,device) if 'slide_native' in tgt.arrays else None
                        sm=surface_metrics(rot.float(),target[...,:6].float(),tgt.geom(len(rr)),pred_slide_native=out['slide_native'],target_slide_native=gt_slide,include_per_patch=False)
                        if gt_slide is not None:
                            slide_errors.append(((out['slide_native']-gt_slide).abs()[out['slide_valid']]*10).cpu().numpy())
                            # Actual source model finger-origin gap, no target q fills predicted FK.
                            if tgt.name=='relay_franka_full_v2':
                                gap=(out['joint_position'][:,8]-out['joint_position'][:,9]).norm(dim=-1)
                                gtgap=(target[:,8,6:]-target[:,9,6:]).norm(dim=-1)*tgt.reach
                                aperture_errors.append(((gap-gtgap).abs()*10).cpu().numpy())
                        if sm['supported']:
                            surfs.append(sm['position_normalized_per_frame'].cpu().numpy());surfns.append(sm['normal_angle_rad_per_frame'].cpu().numpy()*180/np.pi)
                        if start==0:preds.append(rot[:6].float().cpu().numpy())
                    pe=np.concatenate(pes);re=np.concatenate(res);sel=np.concatenate(selecteds);ee=tgt.endpoint['indices']
                    metrics={'all_position_over_reach':array_stats(pe),'all_rotation_deg':array_stats(re),
                        'selected_position_over_reach':array_stats(pe[sel]),'selected_rotation_deg':array_stats(re[sel]),
                        'unconditioned_position_over_reach':array_stats(pe[~sel]),'unconditioned_rotation_deg':array_stats(re[~sel]),
                        'endpoint_position_over_reach':array_stats(pe[:,ee]),'endpoint_rotation_deg':array_stats(re[:,ee]),
                        'surface_position_over_reach':array_stats(np.concatenate(surfs)) if surfs else None,
                        'surface_normal_deg':array_stats(np.concatenate(surfns)) if surfns else None}
                    metrics.update(slide_error_mm=array_stats(np.concatenate(slide_errors)) if slide_errors else None,
                        finger_origin_aperture_error_mm=array_stats(np.concatenate(aperture_errors)) if aperture_errors else None)
                    if tgt.metric:
                        metrics['all_position_mm']=array_stats(pe*tgt.reach*10)
                        metrics['endpoint_position_mm']=array_stats(pe[:,ee]*tgt.reach*10)
                        metrics['selected_success_10mm_5deg']=array_stats(((pe[sel]*tgt.reach*10<10)&(re[sel]<5)).astype(float))
                    filename=f'{label}_{tgt.name}_{mode}_c{count}.npz'
                    np.savez_compressed(folder/filename,rows=rows,groups=tgt.arrays['group'][rows],
                        frame_index=tgt.arrays['frame_index'][rows],position_over_reach=pe,rotation_deg=re,selected=sel,
                        example_pred_rot6=np.concatenate(preds),surface_position_over_reach=np.concatenate(surfs) if surfs else np.array([]))
                    clip_metrics=[]
                    for group in np.unique(tgt.arrays['group'][rows]):
                        ix=tgt.arrays['group'][rows]==group
                        clip_metrics.append({'clip':int(group),'frames':int(ix.sum()),'position_over_reach':float(pe[ix].mean()),'rotation_deg':float(re[ix].mean())})
                    case={'source':src.name,'target':tgt.name,'dataset':tgt.metadata['actual_dataset'],'mode':mode,'condition_count':count,
                        'task':'paired_cross_body_mapping' if src.name!=tgt.name else 'same_body_reconstruction_or_completion',
                        'body_exposure':'training_body' if len(tgt.eligible(True)) else 'entire_body_holdout',
                        'training_initialization':'from_scratch_no_old_motion_checkpoint','frames':len(rows),'error_file':filename,
                        'metrics':metrics,'clip_metrics':clip_metrics,'surface_asset_sha256':None if tgt._surface_static is None else tgt._surface_static['asset_sha256']}
                    if tgt.domain=='xwbc':
                        ss=tgt.arrays['source_clip_split'][rows]
                        case['source_clip_exposure']={str(s):{'frames':int((ss==s).sum()),'position_over_reach':array_stats(pe[ss==s]),'rotation_deg':array_stats(re[ss==s])} for s in np.unique(ss)}
                    cases.append(case)
    result={'label':label,'cases':cases,'frame':'root-relative canonical Y-up, retained heading, no per-frame alignment',
        'surface_comparison':'common fixed mesh points bound to each predicted/GT FK; A and B use same points',
        'no_world_slip_claim':'sampled individual rows; complete trajectory evidence evaluated separately',
        'fixed_eval_set_sha256':digest([{k:c[k] for k in ['source','target','mode','condition_count','frames']} for c in cases]),
        'refinement_steps':0,'physical_validation':False}
    write_json(folder/f'evaluation_{label}.json',result);model.train(was_training);return result

def checkpoint(model,opt,step,run,config,audit_sha,final=False):
    path=run/('checkpoint.pt' if final else f'checkpoint_step{step:04d}.pt');tmp=path.with_suffix('.tmp.pt')
    obj={'model':model.state_dict(),'step':step,'config':config,'data_manifest_sha256':audit_sha}
    if final:obj.update(optimizer=opt.state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
    torch.save(obj,tmp);tmp.replace(path)
    return {'path':str(path),'bytes':path.stat().st_size,'sha256':sha256(path),'step':step,'optimizer_included':final}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--action',choices=['validate','train'],required=True);args=ap.parse_args()
    lock=(ROOT/'gpu_queue.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if not torch.cuda.is_available():raise RuntimeError('remote CUDA required; no fallback')
    torch.set_num_threads(4);device=torch.device('cuda');guard()
    logs=ROOT/'logs';logs.mkdir(exist_ok=True);write_json(logs/'queue_identity.json',process_identity())
    ordered=['A_S_seed0','B_S_seed0','A_M_seed0','B_M_seed0','A_L_seed0','B_L_seed0']
    configs={name:read_json(ROOT/'orchestration/configs'/f'{name}.json') for name in ordered}
    bodies,audit=load_scaling_bodies(ROOT,configs[ordered[0]],device)
    audit_sha=digest(audit);write_json(ROOT/'data_manifest.json',audit)
    for name,cfg in configs.items():
        cfg['judge_body_order']=audit['judge_body_order'];cfg['data_manifest_sha256']=audit_sha
        cfg['train_allowed']='only_with_matching_validation_receipt'
        cfg['readiness']='exact source/data/config CUDA receipt is mandatory'
        if args.action=='validate':write_json(ROOT/'orchestration/configs'/f'{name}.json',cfg)
    code=source_hashes();config_hashes={n:digest(c) for n,c in configs.items()}
    gate=ROOT/'validation_receipt.json'
    if args.action=='validate':
        records={}
        for name in ordered:
            cfg=configs[name];model=build_model(cfg,device)
            records[name]=validate_model(model,bodies,cfg,device)
            write_json(logs/f'{name}_cuda_validation.json',records[name]);del model;gc.collect();torch.cuda.empty_cache()
        if source_hashes()!=code:raise RuntimeError('source changed during validation; no receipt issued')
        write_json(gate,{'status':'passed','utc':utc(),'optimizer_updates':0,'data_manifest_sha256':audit_sha,
            'source_sha256':code,'config_sha256':config_hashes,'gpu':gpu_snapshot(),'records':records})
        print(json.dumps({'status':'six_models_cuda_validated','receipt':str(gate)}),flush=True);return
    verified=read_json(gate)
    if verified.get('status')!='passed' or verified['data_manifest_sha256']!=audit_sha or verified['source_sha256']!=code or verified['config_sha256']!=config_hashes:raise RuntimeError('stale/missing exact-code-data-config CUDA receipt')
    for name in ordered:
        run=ROOT/'runs'/name
        if run.exists():
            if (run/'result.json').exists() and read_json(run/'result.json').get('status')=='completed':continue
            raise RuntimeError('unfinished existing run requires identity/exit review; no implicit restart: '+name)
        run.mkdir(parents=True);cfg=configs[name];model=build_model(cfg,device)
        write_json(run/'config.json',cfg);write_json(run/'source_snapshot.json',code);write_json(run/'process_identity.json',process_identity())
        write_json(run/'capacity_start.json',guard());write_json(run/'gpu_start.json',gpu_snapshot())
        write_json(logs/'queue_state.json',{'utc':utc(),'status':'running','current':name,'identity':process_identity(),'ordered_runs':ordered})
        obj=loss_function(cfg);counter=ComputeCounter(model);body_map={b.name:b for b in bodies}
        rng=np.random.default_rng(cfg['seed']);combo=np.random.default_rng(4);reenc=np.random.default_rng(cfg['seed']*1000+77)
        opt=torch.optim.AdamW(model.parameters(),lr=cfg['optimizer']['lr'],weight_decay=cfg['optimizer']['weight_decay'])
        schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,cfg['steps'])
        step=0;phase='initial_eval';status='failed';started=time.monotonic();train_seconds=0.;forward_flops=0;tokens={};checkpoint_records=[]
        try:
            evaluate(model,bodies,cfg,run/'eval','initial',quick=True)
            phase='training'
            for step in range(1,cfg['steps']+1):
                if train_seconds>cfg['max_training_seconds']:raise RuntimeError('bounded run time exhausted before fixed exposure')
                if step==1 or step%100==0:guard()
                if step==1 or step%1000==0:
                    if source_hashes()!=code:raise RuntimeError('run source changed after validation; no silent continuation')
                if step%1000==0:write_json(run/f'gpu_step{step}.json',gpu_snapshot())
                # A/B/S/M/L independent streams consume identical data decisions.
                domain='xwbc' if rng.random()<cfg['mixture']['xwbc'] else 'unimate'
                torch.manual_seed(cfg['seed']*1000000+step);opt.zero_grad(set_to_none=True);counter.reset()
                torch.cuda.synchronize();tick=time.monotonic()
                report=backward_training_step(model,bodies,cfg,rng,combo,reenc,device=device,domain=domain,
                    forward_fn=forward,objective_fn=obj,autocast_fn=autocast)
                grad=finite_backward(model);opt.step();schedule.step();torch.cuda.synchronize();elapsed=time.monotonic()-tick
                train_seconds+=elapsed;compute=counter.report();forward_flops+=compute['estimated_dense_training_flops']
                actual_tokens=effective_tokens(report,body_map)
                if cfg['group']=='A':
                    for key in list(actual_tokens):
                        if 'surface' in key:actual_tokens[key]=0
                for key,value in actual_tokens.items():
                    if isinstance(value,int):tokens[key]=tokens.get(key,0)+value
                sampling_sha=digest([{k:d[k] for k in ['source','target','mode','selected_row_sha256','sampling_sha256']} for d in report['directions']]+[report['auxiliary']])
                record={'step':step,**report,'sampling_sha256':sampling_sha,'gradient_norm_before_clip':grad,'lr':opt.param_groups[0]['lr'],
                    'step_train_seconds':elapsed,'cumulative_train_seconds':train_seconds,'cumulative_wall_seconds':time.monotonic()-started,
                    'compute':compute,'cumulative_estimated_training_flops':forward_flops,'effective_tokens':actual_tokens,'cumulative_tokens':dict(tokens)}
                with (run/'train.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
                if step==1 or step%100==0:print(json.dumps({'run':name,'step':step,'total':report['terms']['total'],'train_seconds':train_seconds}),flush=True)
                if step in cfg['checkpoint_steps']:
                    checkpoint_records.append(checkpoint(model,opt,step,run,cfg,audit_sha))
                    evaluate(model,bodies,cfg,run/'eval',f'step{step}',quick=True)
            phase='final_checkpoint';final=checkpoint(model,opt,step,run,cfg,audit_sha,True)
            phase='final_eval';evaluate(model,bodies,cfg,run/'eval','final',quick=False)
            status='completed';write_json(run/'result.json',{'status':status,'steps':step,'train_seconds':train_seconds,
                'total_wall_seconds':time.monotonic()-started,'checkpoint':final,'intermediate_checkpoints':checkpoint_records,
                'cumulative_estimated_training_flops':forward_flops,'cumulative_tokens':tokens,'data_manifest_sha256':audit_sha,
                'timing_scope':'synchronized full optimizer steps only; Qwen preprocessing, data loading, checkpoint I/O and eval excluded from train_seconds',
                'parameter_counts':{'main':sum(p.numel() for n,p in model.named_parameters() if not n.startswith('judge.')),
                    'auxiliary':sum(p.numel() for n,p in model.named_parameters() if n.startswith('judge.')),'frozen_qwen':cfg['frozen_qwen_parameters']},
                'interpretation':'fixed recipe and budget comparison; A/B is joint architecture+surface information+loss package, not isolated surface causality',
                'physical_robot_or_dynamics_validation':False})
        except Exception as exc:
            write_json(run/'failure.json',{'phase':phase,'step':step,'error':repr(exc),'traceback':traceback.format_exc()});raise
        finally:
            counter.close();write_json(run/'exit_status.json',{'utc':utc(),'status':status,'phase':phase,'step':step,'identity':process_identity()})
            del model,opt,schedule;gc.collect();torch.cuda.empty_cache()
    write_json(logs/'queue_state.json',{'utc':utc(),'status':'six_runs_completed','identity':process_identity(),'ordered_runs':ordered})

if __name__=='__main__':
    action=sys.argv[sys.argv.index('--action')+1] if '--action' in sys.argv else 'unknown'
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');status='completed';failure=None
    try:main()
    except Exception as exc:
        status='failed';failure={'error':repr(exc),'traceback':traceback.format_exc()};raise
    finally:
        (ROOT/'logs').mkdir(exist_ok=True)
        write_json(ROOT/'logs'/f'{action}_attempt_{stamp}_exit.json',{'utc':utc(),'action':action,'status':status,'identity':process_identity(),
            'optimizer_updates':0 if action=='validate' else 'see per-run train.jsonl and exit_status.json','failure':failure})
