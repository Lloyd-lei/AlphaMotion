"""Prepared continuation runner. No default plan and no implicit restart.

A plan can be frozen only after the initial six runs and all formal evaluations
have completed. This file never edits their source, configuration or artifacts.
"""
from pathlib import Path
import argparse,copy,datetime,fcntl,gc,hashlib,json,os,random,sys,time,traceback
import numpy as np
import torch

PHASE=Path(__file__).resolve().parent
ROOT=PHASE.parent
sys.path.insert(0,str(ROOT/'orchestration'))
import train_scaling as original
import stable_slide_gradient
stable_slide_gradient.install()

ORDER=['A_S_seed0','B_S_seed0','A_M_seed0','B_M_seed0','A_L_seed0','B_L_seed0']
GiB=1024**3

def read(p):return json.loads(Path(p).read_text())
def write(p,v):original.write_json(Path(p),v)
def sha(p):return original.sha256(Path(p))
def now():return datetime.datetime.now(datetime.timezone.utc)
def seal():
    out=original.source_hashes()
    out.update({str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'evaluation').glob('*.py'))})
    out[str(Path(__file__).relative_to(ROOT))]=sha(__file__)
    out[str((PHASE/'stable_slide_gradient.py').relative_to(ROOT))]=sha(PHASE/'stable_slide_gradient.py')
    return out

def prerequisite(plan):
    if plan.get('schema')!='common_continuation_v3' or plan.get('decision')!='frozen_after_initial_formal_review':
        raise RuntimeError('Explicit reviewed common-phase decision required; candidate budget is not a launch plan')
    if plan['run_order']!=ORDER or plan['initial_steps']!=3000:raise ValueError('exact six-run lineage required')
    if plan['total_steps']<=3000 or not plan['checkpoint_steps'] or plan['checkpoint_steps'][-1]!=plan['total_steps']:
        raise ValueError('common final checkpoint missing')
    if sorted(set(plan['checkpoint_steps']))!=plan['checkpoint_steps'] or min(plan['checkpoint_steps'])<=3000:
        raise ValueError('common checkpoints must be unique ascending global steps')
    if plan['future_eval_allocation_bytes']<5*GiB:raise ValueError('evaluation headroom below declared budget')
    if read(ROOT/'logs/queue_state.json').get('status')!='six_runs_completed':raise RuntimeError('initial training not complete')
    evaluation=read(ROOT/'logs/evaluation_queue/state.json')
    if evaluation.get('status')!='completed' or len(evaluation.get('jobs',[]))!=18 or any(j['status']!='completed' for j in evaluation['jobs']):
        raise RuntimeError('initial fixed evaluation queue not complete')
    if sha(ROOT/'logs/evaluation_queue/state.json')!=plan['initial_evaluation_state_sha256']:
        raise RuntimeError('reviewed formal evaluation receipt changed')
    p1=read(ROOT/'evaluation_p1/logs/state.json')
    if p1.get('status')!='completed' or len(p1.get('runs',[]))!=6 or read(ROOT/'evaluation_p1/logs/exit_status.json').get('returncode')!=0:
        raise RuntimeError('initial supplemental P1 evaluation not complete')
    for relative,expected_sha in plan['review_evidence_sha256'].items():
        if sha(ROOT/relative)!=expected_sha:raise RuntimeError('review evidence changed: '+relative)
    numeric=read(PHASE/'numeric_repair_receipt.json')
    if numeric['status']!='passed' or numeric['optimizer_updates']!=0 or numeric['kernel_sha256']!=sha(PHASE/'stable_slide_gradient.py'):
        raise RuntimeError('Exact numeric repair validation required')
    expected=plan['initial_runs'];results={};configs={}
    for name in ORDER:
        run=ROOT/'runs'/name;r=read(run/'result.json');e=read(run/'exit_status.json')
        if r['status']!='completed' or r['steps']!=3000 or e['status']!='completed':raise RuntimeError('incomplete initial run '+name)
        for filename,key in [('config.json','config_sha256'),('result.json','result_sha256'),('checkpoint.pt','checkpoint_sha256')]:
            if sha(run/filename)!=expected[name][key]:raise RuntimeError('initial lineage differs: '+name+'/'+filename)
        if r['data_manifest_sha256']!=plan['data_manifest_sha256']:raise RuntimeError('initial data mismatch')
        results[name]=r;configs[name]=read(run/'config.json')
    if seal()!=plan['source_sha256']:raise RuntimeError('frozen source changed')
    return results,configs

def guard(plan):
    cap=original.capacity()
    if cap['expected_unique_bytes']!=200505865539:raise RuntimeError('authorized download plan changed')
    allocated=sum(p.stat().st_size for p in (PHASE/'runs').rglob('*') if p.is_file())
    if allocated>plan['future_output_allocation_bytes']:raise RuntimeError('continuation output allocation exceeded')
    remaining=max(0,plan['future_output_allocation_bytes']-allocated)
    required=cap['remaining_bytes']+16*GiB+remaining+plan['future_eval_allocation_bytes']
    if cap['available_bytes']<required:raise RuntimeError('downloads plus16GiB reserve and remaining continuation/evaluation outputs do not fit')
    if now()>=datetime.datetime.fromisoformat(plan['latest_training_finish_utc']):
        raise RuntimeError('common phase training deadline reached; retain evidence, no silent shorter run')
    return dict(utc=now().isoformat(),available_bytes=cap['available_bytes'],remaining_download_bytes=cap['remaining_bytes'],
        phase_allocated_bytes=allocated,phase_future_reserved_bytes=remaining,required_available_bytes=required,
        atomic_full_checkpoint_scratch_included_in_output_budget=True)

def configuration(base,plan,name):
    cfg=copy.deepcopy(base)
    cfg.update(steps=plan['total_steps'],checkpoint_steps=plan['checkpoint_steps'],
        max_training_seconds=plan['max_phase_training_seconds_per_run'])
    cfg['continuation']={'version':'common_continuation_v3','initial_global_step':3000,
        'numerical_repair':'Forward identical sigmoid, cancellation-free analytic derivative exp(-abs(x))/(1+exp(-abs(x)))**2, shared by all six',
        'initial_checkpoint_sha256':plan['initial_runs'][name]['checkpoint_sha256'],
        'optimizer_state':'retain each own Adam moments from its 3000-step checkpoint',
        'scheduler':'same explicit cosine restart for all six; not an uninterrupted original schedule',
        'learning_rate_restart':base['optimizer']['lr'],
        'rng_policy':'explicit fresh shared NumPy phase streams; original NumPy states were not saved',
        'rng_seeds':plan['shared_numpy_seeds'],'motion_model_lineage':'own from-scratch initial run; no old pilot weights'}
    cfg['train_allowed']='only_with_matching_common_phase_validation_receipt'
    return cfg

def load_run(name,cfg,device):
    payload=torch.load(ROOT/'runs'/name/'checkpoint.pt',map_location='cpu',weights_only=True)
    if payload.get('step')!=3000 or payload['data_manifest_sha256']!=cfg['data_manifest_sha256']:
        raise RuntimeError('checkpoint state has wrong lineage')
    model=original.build_model(cfg,device);model.load_state_dict(payload['model'],strict=True)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['optimizer']['lr'],weight_decay=cfg['optimizer']['weight_decay'])
    opt.load_state_dict(payload['optimizer'])
    for state in opt.state.values():
        for value in state.values():
            if torch.is_tensor(value) and not torch.isfinite(value).all():raise RuntimeError('nonfinite saved Adam state')
    for group in opt.param_groups:
        group['lr']=cfg['optimizer']['lr'];group['initial_lr']=cfg['optimizer']['lr']
    del payload
    return model,opt

def save_resume(model,opt,scheduler,streams,step,run,cfg,audit_sha,stats):
    path=run/'checkpoint_resume.pt';temp=path.with_suffix('.tmp.pt')
    value={'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),
        'numpy_rng':{k:v.bit_generator.state for k,v in streams.items()},'python_rng':random.getstate(),
        'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
        'step':step,'config':cfg,'data_manifest_sha256':audit_sha,'phase_statistics':stats}
    torch.save(value,temp);temp.replace(path)
    return {'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path),'step':step,'optimizer_included':True,
        'numpy_rng_included':True,'scheduler_included':True,'scope':'task-owned rolling phase checkpoint; frozen initial checkpoints preserved'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--action',choices=['validate','train'],required=True);args=p.parse_args()
    plan=read(args.plan);plan_sha=sha(args.plan)
    lock=(ROOT/'gpu_queue.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    initial,bases=prerequisite(plan);guard(plan)
    if not torch.cuda.is_available():raise RuntimeError('real CUDA required; no fallback')
    torch.set_num_threads(4);device=torch.device('cuda');logs=PHASE/'logs';logs.mkdir(parents=True,exist_ok=True)
    configs={n:configuration(bases[n],plan,n) for n in ORDER}
    bodies,audit=original.load_scaling_bodies(ROOT,configs[ORDER[0]],device);audit_sha=original.digest(audit)
    if audit_sha!=plan['data_manifest_sha256']:raise RuntimeError('reloaded real data differs')
    code=seal();config_hashes={n:original.digest(c) for n,c in configs.items()};gate=PHASE/'validation_receipt.json'
    if args.action=='validate':
        records={}
        for name in ORDER:
            model,opt=load_run(name,configs[name],device)
            records[name]=original.validate_model(model,bodies,configs[name],device)
            write(logs/(name+'_cuda_validation.json'),records[name])
            del model,opt;gc.collect();torch.cuda.empty_cache()
        if seal()!=code:raise RuntimeError('source changed during validation')
        write(gate,dict(status='passed',utc=now().isoformat(),optimizer_updates=0,plan_sha256=plan_sha,
            source_sha256=code,config_sha256=config_hashes,data_manifest_sha256=audit_sha,gpu=original.gpu_snapshot(),records=records))
        print(json.dumps({'status':'six_continuation_models_cuda_validated','optimizer_updates':0}));return
    receipt=read(gate)
    if any(receipt.get(k)!=v for k,v in dict(status='passed',plan_sha256=plan_sha,source_sha256=code,config_sha256=config_hashes,data_manifest_sha256=audit_sha).items()):
        raise RuntimeError('missing or stale exact continuation CUDA receipt')
    reference_sampling=None
    for name in ORDER:
        run=PHASE/'runs'/name
        if run.exists():raise RuntimeError('existing phase run requires identity/exit review; no implicit duplicate or restart')
        run.mkdir(parents=True);cfg=configs[name];model,opt=load_run(name,cfg,device)
        for filename,value in [('config.json',cfg),('source_snapshot.json',code),('process_identity.json',original.process_identity()),('capacity_start.json',guard(plan))]:write(run/filename,value)
        write(logs/'queue_state.json',dict(utc=now().isoformat(),status='running',current=name,identity=original.process_identity(),ordered_runs=ORDER,plan_sha256=plan_sha))
        schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,plan['total_steps']-3000)
        streams={k:np.random.default_rng(v) for k,v in plan['shared_numpy_seeds'].items()}
        rng,combo,reenc=(streams[k] for k in ['data','combo','reenc'])
        objective=original.loss_function(cfg);counter=original.ComputeCounter(model);body_map={b.name:b for b in bodies}
        started=time.monotonic();train_seconds=0.;flops=0;tokens={};checkpoints=[];sampling=[];step=3000;status='failed';phase='start'
        try:
            phase='training'
            for step in range(3001,plan['total_steps']+1):
                local_step=step-3000
                if train_seconds>cfg['max_training_seconds']:raise RuntimeError('bounded phase training time exhausted')
                if local_step==1 or step%100==0:guard(plan)
                if local_step==1 or step%1000==0:
                    if seal()!=code:raise RuntimeError('source changed during continuation')
                domain='xwbc' if rng.random()<cfg['mixture']['xwbc'] else 'unimate'
                torch.manual_seed(cfg['seed']*1000000+step);opt.zero_grad(set_to_none=True);counter.reset()
                torch.cuda.synchronize();tick=time.monotonic()
                report=original.backward_training_step(model,bodies,cfg,rng,combo,reenc,device=device,domain=domain,
                    forward_fn=original.forward,objective_fn=objective,autocast_fn=original.autocast)
                torch.cuda.synchronize();forward_backward_seconds=time.monotonic()-tick
                check_tick=time.monotonic()
                sample=original.digest([{k:d[k] for k in ['source','target','mode','selected_row_sha256','sampling_sha256']} for d in report['directions']]+[report['auxiliary']])
                if reference_sampling is not None and sample!=reference_sampling[local_step-1]:raise RuntimeError('actual common phase sampling differs before optimizer update')
                sampling_check_seconds=time.monotonic()-check_tick
                torch.cuda.synchronize();tick=time.monotonic()
                grad=original.finite_backward(model);opt.step();schedule.step();torch.cuda.synchronize();elapsed=forward_backward_seconds+time.monotonic()-tick
                train_seconds+=elapsed;compute=counter.report();flops+=compute['estimated_dense_training_flops'];sampling.append(sample)
                actual=original.effective_tokens(report,body_map)
                if cfg['group']=='A':
                    for k in actual:
                        if 'surface' in k:actual[k]=0
                for k,v in actual.items():
                    if isinstance(v,int):tokens[k]=tokens.get(k,0)+v
                cumulative_tokens={k:initial[name]['cumulative_tokens'].get(k,0)+v for k,v in tokens.items()}
                record={'step':step,'phase_step':local_step,**report,'sampling_sha256':sample,'gradient_norm_before_clip':grad,
                    'lr':opt.param_groups[0]['lr'],'step_train_seconds':elapsed,'sampling_check_seconds':sampling_check_seconds,'phase_train_seconds':train_seconds,
                    'cumulative_train_seconds':initial[name]['train_seconds']+train_seconds,
                    'cumulative_wall_seconds':initial[name]['total_wall_seconds']+time.monotonic()-started,
                    'compute':compute,'phase_estimated_training_flops':flops,
                    'cumulative_estimated_training_flops':initial[name]['cumulative_estimated_training_flops']+flops,
                    'effective_tokens':actual,'phase_tokens':dict(tokens),'cumulative_tokens':cumulative_tokens}
                with (run/'train.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
                if local_step==1 or step%100==0:print(json.dumps({'run':name,'global_step':step,'phase_train_seconds':train_seconds}),flush=True)
                if step in plan['checkpoint_steps']:
                    guard(plan);checkpoints.append(original.checkpoint(model,opt,step,run,cfg,audit_sha))
                    resume=save_resume(model,opt,schedule,streams,step,run,cfg,audit_sha,dict(train_seconds=train_seconds,flops=flops,tokens=tokens))
                    write(run/'latest_resume_receipt.json',resume)
                    original.evaluate(model,bodies,cfg,run/'eval',f'step{step}',quick=True)
            phase='final_eval';original.evaluate(model,bodies,cfg,run/'eval','final',quick=False)
            final=run/'checkpoint.pt';(run/'checkpoint_resume.pt').replace(final)
            resume.update(path=str(final),scope='completed phase checkpoint; full Adam, scheduler and all RNG states')
            write(run/'latest_resume_receipt.json',resume)
            status='completed'
            write(run/'result.json',dict(status=status,steps=step,initial_steps=3000,additional_steps=step-3000,
                phase_train_seconds=train_seconds,train_seconds=initial[name]['train_seconds']+train_seconds,
                total_wall_seconds=initial[name]['total_wall_seconds']+time.monotonic()-started,
                checkpoint=dict(path=str(final),sha256=sha(final),bytes=final.stat().st_size,step=step,optimizer_included=True,numpy_rng_included=True),
                intermediate_checkpoints=checkpoints,phase_sampling_sha256=hashlib.sha256('\n'.join(sampling).encode()).hexdigest(),
                data_manifest_sha256=audit_sha,initial_checkpoint_sha256=plan['initial_runs'][name]['checkpoint_sha256'],
                phase_cumulative_tokens=tokens,cumulative_tokens=cumulative_tokens,
                cumulative_estimated_training_flops=initial[name]['cumulative_estimated_training_flops']+flops,
                parameter_counts=initial[name]['parameter_counts'],plan_sha256=plan_sha,
                timing_scope='synchronized forward/backward and optimizer; cumulative includes own initial phase; extra sampling comparison, I/O and evaluation excluded',
                physical_robot_or_dynamics_validation=False))
            if reference_sampling is None:reference_sampling=sampling
        except Exception as exc:
            write(run/'failure.json',dict(phase=phase,step=step,error=repr(exc),traceback=traceback.format_exc()));raise
        finally:
            counter.close();write(run/'exit_status.json',dict(utc=now().isoformat(),status=status,phase=phase,step=step,identity=original.process_identity()))
            del model,opt,schedule;gc.collect();torch.cuda.empty_cache()
    write(logs/'queue_state.json',dict(utc=now().isoformat(),status='six_runs_completed',identity=original.process_identity(),ordered_runs=ORDER,plan_sha256=plan_sha))

if __name__=='__main__':
    status='failed';error=None
    try:main();status='completed'
    except Exception as exc:error={'error':repr(exc),'traceback':traceback.format_exc()};raise
    finally:
        (PHASE/'logs').mkdir(parents=True,exist_ok=True)
        stamp=now().strftime('%Y%m%dT%H%M%SZ')
        write(PHASE/'logs'/('attempt_'+stamp+'_exit.json'),dict(utc=now().isoformat(),status=status,error=error,identity=original.process_identity(),action=sys.argv))
