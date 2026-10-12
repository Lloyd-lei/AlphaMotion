"""Bounded GPU pilot. Offline --describe is safe; real data and process checks gate execution."""
from __future__ import annotations
import argparse,contextlib,datetime,fcntl,json,math,os,platform,random,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from pilot_data import load_bodies,make_inputs,sample_rows,fk,angle_degrees,read_json,write_json,sha256
from pilot_model import ConditionalMegatron,load_reference_weights
from recipe_gate import assert_training_recipe_ready,training_recipe_status,write_validation_receipt,REFERENCE_CONFIG_SHA256
from recipe_training import backward_training_step


def autocast(device):return torch.autocast('cuda',dtype=torch.bfloat16) if device.type=='cuda' else contextlib.nullcontext()


def forward(model,src,tgt,inputs):
 return model(**inputs,source_geom=src.geom(len(inputs['source'])),target_geom=tgt.geom(len(inputs['source'])))


def objective(outputs,target,tgt,rows,condition,weights):
 rot,aux,_=outputs;rot=rot.float();aux=aux.float();target=target.float()
 predicted=fk(rot,tgt.parents,tgt.offsets)/tgt.reach
 ground_truth_fk=fk(target[...,:6],tgt.parents,tgt.offsets)/tgt.reach
 pe=(predicted-ground_truth_fk).norm(dim=-1);re=angle_degrees(rot,target[...,:6],training=True)
 terms={'rotation_deg':re.mean(),'position_over_reach':pe.mean()}
 loss=terms['rotation_deg']/30+weights['position']*terms['position_over_reach']
 ix=condition['joint_indices'];v=condition['valid'];batch=torch.arange(len(rows),device=rot.device)[:,None].expand_as(ix)
 if bool(v.any()):
  # Explicit valid selection before arithmetic; no NaN padding reaches loss.
  cp=pe[batch[v],ix[v]].mean();cr=re[batch[v],ix[v]].mean()
  terms.update(condition_position_over_reach=cp,condition_rotation_deg=cr)
  loss=loss+weights['condition_position']*cp+weights['condition_rotation']*cr/30
 rv=tgt.tensor('root_height_valid',rows,rot.device).bool()
 height=tgt.tensor('root_height_m',rows,rot.device).float();rv=rv & torch.isfinite(height)
 if bool(rv.any()):
  rh=F.smooth_l1_loss(aux[rv,0,0],height[rv]);terms['root_height_smooth_l1_m']=rh
  loss=loss+weights['root_height']*rh
 if weights.get('penetration',0) or weights.get('contact_geometry',0):
  from recipe_ground import ground_losses
  ground=ground_losses(rot,predicted*tgt.reach,aux[:,0,0],tgt.ground_batch(rows,rot.device),w_pen=weights.get('penetration',0),w_cgeo=weights.get('contact_geometry',0),w_rootpos=0.)
  loss=loss+ground['total'];terms.update(ground['terms'])
 if not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinite loss')
 return loss,terms


def array_stats(x):
 x=np.asarray(x,dtype='float64').reshape(-1)
 if not len(x):return None
 if not np.isfinite(x).all():raise FloatingPointError('nonfinite eval metric')
 return {'count':len(x),'mean':float(x.mean()),'p50':float(np.median(x)),'p95':float(np.quantile(x,.95))}


def heldout_rows(body,limit,seed):
 rng=np.random.default_rng(seed);available=body.eligible(False);groups=np.unique(body.arrays['group'][available]);rng.shuffle(groups)
 pools=[rng.permutation(available[body.arrays['group'][available]==g]).tolist() for g in groups]
 out=[]
 while len(out)<min(limit,len(available)):
  for pool in pools:
   if pool and len(out)<limit:out.append(pool.pop())
 return np.asarray(out,dtype=np.int64)


def evaluate(model,bodies,config,output,label,quick=False,condition_counts=(0,1,2)):
 """Same deterministic rows/masks across no/single/dual-SE3; no optimization or refinement."""
 device=next(model.parameters()).device;was_training=model.training;model.eval();cases=[]
 human=next(b for b in bodies if b.name=='human_smpl')
 candidates=[b for b in bodies if len(b.eligible(False))]
 if quick:
  old=next(b for b in candidates if b.name=='unitree_g1_29dof')
  seen=next(b for b in candidates if b.domain=='unimate' and len(b.eligible(True)))
  held=next((b for b in candidates if b.domain=='unimate' and not len(b.eligible(True))),None)
  candidates=[old,seen]+([held] if held else [])
 with torch.inference_mode():
  for bi,tgt in enumerate(candidates):
   rows=heldout_rows(tgt,config['eval']['quick_frames'] if quick else config['eval']['frames_per_body'],config['seed']+731)
   for mode in (['full','sparse_only'] if quick else ['full','masked','sparse_only']):
    if mode=='sparse_only' and not tgt.endpoint['indices']:continue
    src=human if tgt.domain=='xwbc' and mode=='full' else tgt
    for count in condition_counts:
     if count>len(tgt.endpoint['indices']):continue
     stats={};row_errors=[];row_angles=[];row_selected=[]
     for start in range(0,len(rows),config['eval']['batch_size']):
      rr=rows[start:start+config['eval']['batch_size']]
      # Source/mask RNG and condition RNG have fixed per-case state across condition counts.
      rng=np.random.default_rng(config['seed']+bi*10000+start+912)
      inputs,target=make_inputs(src,tgt,rr,rng,device,mode,count,contact_dropout=1.)
      with autocast(device):rot,aux,_=forward(model,src,tgt,inputs)
      pred=fk(rot,tgt.parents,tgt.offsets)/tgt.reach
      pe=(pred-target[...,6:]).norm(dim=-1).float();re=angle_degrees(rot,target[...,:6])
      selected=torch.zeros_like(pe,dtype=torch.bool);ci=inputs['condition']['joint_indices'];cv=inputs['condition']['valid']
      for i in range(len(rr)):selected[i,ci[i,cv[i]]]=True
      row_errors.append(pe.cpu().numpy());row_angles.append(re.cpu().numpy());row_selected.append(selected.cpu().numpy())
      ee=tgt.endpoint['indices'];metric_unit='mm' if tgt.metric else 'canonical_body_units'
      variants={'all_position_over_reach':pe,'all_rotation_deg':re,'unconditioned_position_over_reach':pe[~selected],
        'unconditioned_rotation_deg':re[~selected],'selected_position_over_reach':pe[selected],'selected_rotation_deg':re[selected],
        'endpoint_position_over_reach':pe[:,ee],'endpoint_rotation_deg':re[:,ee],
        'endpoint_position_'+metric_unit:pe[:,ee]*tgt.reach*(10 if tgt.metric else 1)}
      if tgt.metric:
       variants['selected_joint_success_10mm_5deg']=((pe[selected]*tgt.reach*10<10)&(re[selected]<5)).float()
      for key,value in variants.items():stats.setdefault(key,[]).append(value.cpu().numpy().reshape(-1))
      # Keep source exposure groups separately, never call LAFAN pretrain-seen a new motion holdout.
      if tgt.domain=='xwbc':
       for exposure,mask in [('old_pretrain_seen',tgt.arrays['pretrain_seen'][rr]),('old_excluded_motion',~tgt.arrays['pretrain_seen'][rr])]:
        for key,value in [('position_over_reach',pe),('rotation_deg',re)]:
         stats.setdefault(exposure+'_'+key,[]).append(value[torch.as_tensor(mask,device=device)].cpu().numpy().reshape(-1))
      valid_height=tgt.tensor('root_height_valid',rr,device).bool()
      height=tgt.tensor('root_height_m',rr,device)
      valid_height=valid_height & torch.isfinite(height)
      if bool(valid_height.any()):
       stats.setdefault('root_height_abs_error_m',[]).append((aux.float()[valid_height,0,0]-height[valid_height]).abs().cpu().numpy())
      if start==0 and not quick:
       np.savez_compressed(output/f'{label}_{tgt.name}_{mode}_c{count}_examples.npz',
        rows=rr[:6],parents=tgt.parents,joint_names=np.asarray(tgt.names),reference_positions=(target[:6,:,6:]*tgt.reach).cpu().numpy(),
        predicted_fk_positions=(pred[:6]*tgt.reach).cpu().numpy(),predicted_rot6=rot[:6].float().cpu().numpy(),reference_rot6=target[:6,:,:6].cpu().numpy(),
        condition_indices=ci[:6].cpu().numpy(),condition_valid=cv[:6].cpu().numpy(),condition_values=inputs['condition']['values'][:6].cpu().numpy(),
        reach=np.array(tgt.reach),metric_length_valid=np.array(tgt.metric))
     error_file=f'{label}_{tgt.name}_{mode}_c{count}_errors.npz'
     np.savez_compressed(output/error_file,rows=rows,position_over_reach=np.concatenate(row_errors),rotation_deg=np.concatenate(row_angles),selected=np.concatenate(row_selected))
     cases.append({'source':src.name,'target':tgt.name,'domain':tgt.domain,'mode':mode,'condition_count':count,'error_file':error_file,
       'body_exposure':'pilot_seen_body' if len(tgt.eligible(True)) else 'pilot_entire_body_holdout',
       'old_pretraining_exposure':'split_into_seen_and_old_excluded_metrics' if tgt.domain=='xwbc' else 'unknown_source_ancestry; new_body_not_old_XWBC_list',
       'rows':rows.tolist(),'groups':sorted(set(tgt.arrays['group'][rows].tolist())),
       'endpoint_policy':tgt.endpoint,'metrics':{k:array_stats(np.concatenate(v)) for k,v in stats.items()}})
 report={'label':label,'cases':cases,'positions_from':'predicted own-joint global rotations -> fixed-offset FK',
   'frame':'root-centered canonical Y-up; original canonical orientation retained',
   'world_or_object_anchor':None,'refinement_steps':0,'dynamics_simulation_or_hardware_validation':False,
   'sparse_only_note':'Underdetermined full-body completion; selected endpoints and remaining joints reported separately.',
   'conditioning_note':'One or two lawful GT-derived same-current-frame target joints; all other target pose values remain supervised outputs.'}
 write_json(output/f'evaluation_{label}.json',report);model.train(was_training);return report


def evaluate_reference(bodies,config,output):
 """Exact old model + existing frozen SigLIP vectors; no new download or target SE3 interface."""
 from model_dual import DualTranslator
 class ReferenceAdapter(torch.nn.Module):
  def __init__(self):
   super().__init__();self.core=DualTranslator(d_model=256,n_heads=8,enc_layers=4,dec_layers=4,n_latent=128,n_latent_p=4,
    fsq_stages=1,fsq_levels=(32,)*13,use_sem=True,use_dof=True,n_judges=16,contact_chan=True,sem_dim=768)
  def forward(self,source,source_geom,target_geom,source_hidden=None,source_contact=None,condition=None):
   if condition is not None and bool(condition['valid'].any()):raise ValueError('reference has no new target SE3 interface')
   zp,zr=self.core.encode(source,source_geom,joint_mask=source_hidden,con=source_contact)
   return self.core.decode(zp,zr,target_geom,ret_con=True)
 old=[b for b in bodies if b.domain=='xwbc'];device=next(iter(old[0].static.values())).device
 path=Path(config['reference_siglip_cache']);lut=torch.load(path,map_location='cpu',weights_only=True)
 saved={b.name:b.static['sem'] for b in old}
 try:
  for b in old:
   if any(n not in lut for n in b.names):raise ValueError('missing reference SigLIP name; no regeneration allowed')
   sem=torch.stack([torch.as_tensor(lut[n],dtype=torch.float32) for n in b.names]).to(device)
   if sem.shape!=(len(b.names),768) or not bool(torch.isfinite(sem).all()):raise ValueError('reference name shape')
   b.static['sem']=sem
  model=ReferenceAdapter().to(device)
  model.core.load_state_dict(torch.load(config['reference_checkpoint'],map_location='cpu',weights_only=True),strict=True)
  report=evaluate(model,old,config,output,'reference',condition_counts=(0,))
  write_json(output/'reference_provenance.json',{'checkpoint_sha256':sha256(config['reference_checkpoint']),
   'siglip_cache_path':str(path),'siglip_cache_sha256':sha256(path),'scope':'XWBC only; exact old architecture and frozen name vectors; matched new split/float32 payload/FK evaluation',
   'unsupported':'New target SE3 conditioning; no old UniMate baseline because extra frozen SigLIP names were not prepared.'})
  del model;return report
 finally:
  for b in old:b.static['sem']=saved[b.name]


def preflight(model,bodies,config,output):
 device=next(model.parameters()).device;reports=[];batch_profiles=[]
 # Actual supplied data, both domains, distinct mode/condition patterns. No optimizer update here.
 for domain in ['xwbc','unimate']:
  selected=next(b for b in bodies if b.domain==domain and len(b.eligible(True)) and len(b.endpoint['indices']))
  for count,mode in [(0,'full'),(1,'sparse_only'),(min(2,len(selected.endpoint['indices'])),'masked')]:
   rng=np.random.default_rng(123);rows=sample_rows(selected,rng,4)
   inputs,target=make_inputs(selected,selected,rows,rng,device,mode,count)
   model.zero_grad(set_to_none=True)
   with autocast(device):out=forward(model,selected,selected,inputs)
   loss,terms=objective(out,target,selected,rows,inputs['condition'],config['loss']);loss.backward()
   grads=[p.grad for p in model.parameters() if p.grad is not None]
   if not grads or not all(bool(torch.isfinite(g).all()) for g in grads):raise FloatingPointError('nonfinite real-batch gradients')
   reports.append({'domain':domain,'body':selected.name,'mode':mode,'condition_count':count,'loss':float(loss.detach()),'finite_gradient_tensors':len(grads)})
  # Values of hidden target/source answer channels cannot affect pure sparse inference.
  inputs,target=make_inputs(selected,selected,rows,rng,device,'sparse_only',1)
  model.eval()
  with torch.no_grad():
   first=forward(model,selected,selected,inputs)[0]
   inputs['source']=torch.full_like(inputs['source'],float('nan'))
   second=forward(model,selected,selected,inputs)[0]
  if not torch.allclose(first,second,atol=1e-6,rtol=1e-6):raise ValueError('hidden answer leakage')
  model.train()
 # Validate the proposed batch on the largest available topology in each source domain.
 for domain in ['xwbc','unimate']:
  selected=max((b for b in bodies if b.domain==domain and len(b.eligible(True))),key=lambda b:len(b.parents))
  rng=np.random.default_rng(2718);rows=sample_rows(selected,rng,config['batch_size'])
  inputs,target=make_inputs(selected,selected,rows,rng,device,'full',min(2,len(selected.endpoint['indices'])))
  model.zero_grad(set_to_none=True)
  if device.type=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
  start=time.monotonic()
  with autocast(device):out=forward(model,selected,selected,inputs)
  loss,_=objective(out,target,selected,rows,inputs['condition'],config['loss']);loss.backward()
  norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config['optimizer']['clip_norm'],error_if_nonfinite=True)
  if device.type=='cuda':torch.cuda.synchronize()
  batch_profiles.append({'domain':domain,'body':selected.name,'joints':len(selected.parents),'batch_size':len(rows),
   'forward_backward_seconds':time.monotonic()-start,'gradient_norm':float(norm),
   'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated() if device.type=='cuda' else None,'optimizer_steps':0})
 recipe_profiles=[]
 if 'recipe' in config and 'judge_body_order' in config:
  for domain in ['xwbc','unimate']:
   model.zero_grad(set_to_none=True)
   if device.type=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
   start=time.monotonic()
   profile=backward_training_step(model,bodies,config,np.random.default_rng(901),np.random.default_rng(902),np.random.default_rng(903),device=device,domain=domain,forward_fn=forward,objective_fn=objective,autocast_fn=autocast,force_all_losses=True)
   norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config['optimizer']['clip_norm'],error_if_nonfinite=True)
   grads=[p.grad for p in model.parameters() if p.grad is not None]
   if not grads or not all(bool(torch.isfinite(g).all()) for g in grads):raise FloatingPointError('nonfinite full-recipe gradients')
   if device.type=='cuda':torch.cuda.synchronize()
   recipe_profiles.append({**profile,'gradient_norm':float(norm),'forward_backward_seconds':time.monotonic()-start,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated() if device.type=='cuda' else None,'optimizer_steps':0})
 model.zero_grad(set_to_none=True)
 report={'status':'passed','real_payload_batch_checks':reports,'hidden_source_perturbation_invariant':True,
         'proposed_batch_profiles':batch_profiles,
         'full_reference_recipe_verified':len(recipe_profiles)==2 and device.type=='cuda','full_recipe_profiles':recipe_profiles,'training_recipe_gate':training_recipe_status(config),
         'optimizer_steps':0,'gpu':str(device),'does_not_establish':'training convergence or heldout performance'}
 write_json(output/'real_batch_preflight.json',report);return report


def capacity_guard(receipt,require_fresh=False):
 r=read_json(receipt);stamp=datetime.datetime.fromisoformat(r['checked_at_utc'])
 age=(datetime.datetime.now(datetime.timezone.utc)-stamp).total_seconds()
 if require_fresh and not 0<=age<300:raise RuntimeError('requires a live combined-download capacity receipt <5 minutes old')
 # Count actual complete/partial cache bytes again; do not mistake simultaneous download growth for pilot disk use.
 from remote_state import capacity
 live=capacity();remaining=int(live['remaining_bytes']);s=os.statvfs('/home');available=s.f_bavail*s.f_frsize
 required=remaining+16*1024**3+2*1024**3 #15GiB reserve+1GiB download overhead+2GiB pilot allocation
 if available<required:raise RuntimeError(f'capacity guard: free {available}, conservative required {required}')
 return {'available_bytes':available,'remaining_download_bytes_upper_bound':remaining,'required_bytes':required,'receipt_age_seconds':age}


def process_identity():
 pid=os.getpid();p=Path('/proc')/str(pid)
 if not p.exists():raise RuntimeError('real pilot requires Linux /proc identity')
 s=(p/'stat').read_text().split(') ',1)[1].split()
 return {'pid':pid,'start_ticks':s[19],'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
         'command_line':(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(),'cwd':str(Path.cwd()),'hostname':platform.node()}


def save_checkpoint(model,optimizer,step,run,config,audit):
 path=run/'checkpoint.pt';tmp=run/'checkpoint.tmp.pt'
 torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'step':step,'config':config,
             'dataset_audit':audit,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},tmp);tmp.replace(path)
 return {'path':str(path),'sha256':sha256(path),'bytes':path.stat().st_size,'step':step}


def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--root',type=Path)
 p.add_argument('--action',choices=['describe','validate','train'],default='describe');p.add_argument('--capacity-receipt',type=Path)
 p.add_argument('--recipe-validation-run',type=Path,help='completed full-loss real CUDA validate run; required by train')
 args=p.parse_args();config=read_json(args.config)
 if args.action=='describe':
  model=ConditionalMegatron(**config['model']);print(json.dumps({'parameters':sum(p.numel() for p in model.parameters()),'state_keys':len(model.state_dict()),'parameter_count_scope':'total includes separately reported training-only auxiliary judge' ,'parameter_counts':{'shared_model':sum(p.numel() for n,p in model.named_parameters() if not n.startswith('judge.')),'auxiliary_judge':sum(p.numel() for n,p in model.named_parameters() if n.startswith('judge.'))},'training_recipe_gate':training_recipe_status(config),'config':config},indent=2));return
 if args.action=='train':assert_training_recipe_ready(config,args.recipe_validation_run,args.root)
 if not args.root or not args.capacity_receipt:raise ValueError('root and fresh capacity receipt required')
 if not torch.cuda.is_available():raise RuntimeError('real pilot requires remote CUDA; no local fallback')
 device=torch.device('cuda');torch.set_num_threads(4)
 torch.manual_seed(config['seed']);random.seed(config['seed']);np.random.seed(config['seed']);rng=np.random.default_rng(config['seed'])
 root=args.root.resolve();lock=(root/'pilot_train.lock').open('a+')
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:raise RuntimeError('this pilot already has a live owner')
 cap=capacity_guard(args.capacity_receipt,True);identity=process_identity()
 stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');run=root/'runs'/f'{stamp}_{args.action}_seed{config["seed"]}'
 run.mkdir(parents=True,exist_ok=False);write_json(run/'config.json',config);write_json(run/'process_identity.json',identity);write_json(run/'capacity_start.json',cap)
 code={str(f.relative_to(Path(__file__).parent)):sha256(f) for f in Path(__file__).parent.rglob('*.py')}
 write_json(run/'source_snapshot.json',{'files_sha256':code,'torch_version':torch.__version__,'gpu':torch.cuda.get_device_name(0)})
 step=0;phase='loading';status='failed';started=time.monotonic()
 try:
  ref_path=Path(config['reference_checkpoint']);qpath=root/config['qwen_cache']
  if sha256(config['reference_config'])!=REFERENCE_CONFIG_SHA256:raise ValueError('live original run config checksum mismatch')
  if sha256(ref_path)!=config['reference_sha256']:raise ValueError('reference checkpoint checksum mismatch')
  reference=torch.load(ref_path,map_location='cpu',weights_only=True)
  model=ConditionalMegatron(**config['model']);load_report=load_reference_weights(model,reference,judge_body_order=read_json(root/'data_xwbc/chosen_split.json')['body_names'],judge_init_seed=config['seed']);del reference
  model.to(device);write_json(run/'weight_transfer.json',load_report)
  bodies,audit=load_bodies(root,qpath,config['qwen_sha256'],device);write_json(run/'dataset_audit.json',audit)
  config['judge_body_order']=read_json(root/'data_xwbc/chosen_split.json')['body_names']
  if len(set(config['judge_body_order']))!=16:raise ValueError('verified current judge label map must contain16 bodies')
  write_json(run/'judge_body_order.json',{'body_order':config['judge_body_order'],'old_head_warmstarted':False,'source':'verified chosen_split.json order; current auxiliary categories'})
  write_json(run/'config.json',config)
  if not any(b.domain=='unimate' and len(b.eligible(False)) and not len(b.eligible(True)) for b in bodies):
   raise ValueError('pilot needs at least one real FK-valid entire new-body holdout')
  phase='real_batch_preflight';preflight_report=preflight(model,bodies,config,run)
  if args.action=='validate':
   write_validation_receipt(config,root,run,preflight_report)
   status='recipe_validated_no_training';return
  phase='reference_eval';evaluate_reference(bodies,config,run)
  phase='initial_eval';initial=evaluate(model,bodies,config,run,'initial',quick=False)
  optimizer=torch.optim.AdamW(model.parameters(),lr=config['optimizer']['lr'],weight_decay=config['optimizer']['weight_decay'])
  scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,config['steps'])
  combo_rng=np.random.default_rng(4);reenc_rng=np.random.default_rng(config['seed']*1000+77)
  training_start=time.monotonic();phase='training';history=run/'train.jsonl';stopped_reason='planned_steps'
  for step in range(1,config['steps']+1):
   if time.monotonic()-training_start>config['max_training_seconds']:
    step-=1;stopped_reason='time_budget';break
   if step==1 or step%100==0:capacity_guard(args.capacity_receipt)
   domain='xwbc' if rng.random()<config['mixture']['xwbc'] else 'unimate'
   optimizer.zero_grad(set_to_none=True)
   step_report=backward_training_step(model,bodies,config,rng,combo_rng,reenc_rng,device=device,domain=domain,forward_fn=forward,objective_fn=objective,autocast_fn=autocast)
   grad=torch.nn.utils.clip_grad_norm_(model.parameters(),config['optimizer']['clip_norm'],error_if_nonfinite=True)
   optimizer.step();scheduler.step()
   log={'step':step,**step_report,'gradient_norm_before_clip':float(grad),'lr':optimizer.param_groups[0]['lr'],'training_elapsed_seconds':time.monotonic()-training_start}
   with history.open('a') as f:f.write(json.dumps(log,allow_nan=False)+'\n')
   if step==1 or step%25==0:print(json.dumps(log),flush=True)
   if step%config['checkpoint_every']==0:
    save_checkpoint(model,optimizer,step,run,config,audit);evaluate(model,bodies,config,run,f'step{step}',quick=True)
  phase='final_checkpoint';checkpoint=save_checkpoint(model,optimizer,step,run,config,audit)
  phase='final_eval';final=evaluate(model,bodies,config,run,'final',quick=False)
  write_json(run/'checkpoint_receipt.json',checkpoint)
  status='pilot_completed' if step>=config['minimum_useful_steps'] else 'stopped_before_minimum_steps'
  write_json(run/'result.json',{'status':status,'steps':step,'stop_reason':stopped_reason,'checkpoint':checkpoint,
    'initial_eval':'evaluation_initial.json','final_eval':'evaluation_final.json','baseline_status':'evaluation_reference.json; matched XWBC FK evaluation',
    'claimed_scope':'native single-frame reconstruction/paired kinematic mapping and sparse completion only',
    'causal_attribution':'Not isolated: text descriptors, depth, sparse conditions and data mixture differ together; original applicable XWBC single-frame terms restored.'})
  phase='paired_analysis'
  from analyze_pilot import paired_summary,plot_outputs
  paired_summary(run);plot_outputs(run)
 except Exception as exc:
  status='failed'
  write_json(run/'failure.json',{'phase':phase,'step':step,'error':repr(exc),'traceback':traceback.format_exc()});raise
 finally:
  write_json(run/'exit_status.json',{'status':status,'phase':phase,'step':step,'elapsed_seconds':time.monotonic()-started,'identity':identity})
  fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
  print(json.dumps({'run':str(run),'status':status,'step':step}),flush=True)

if __name__=='__main__':main()
