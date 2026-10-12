"""Original XWBC single-frame sampling, with explicitly separate new sparse tasks."""
from __future__ import annotations
import itertools
import hashlib
import numpy as np
import torch
from pilot_data import make_inputs,sample_rows
from common import SkeletonSpec
from masking import parse_mix,sample_joint_mask


def make_reference_inputs(src,tgt,rows,rng,device,config):
 if src.domain!='xwbc' or tgt.domain!='xwbc':raise ValueError('reference paired inputs require XWBC')
 # Generate only declared sparse target conditions with the established helper;
 # source masks/contact below replace its generic task policy completely.
 inputs,target=make_inputs(src,tgt,rows,rng,device,'full',condition_dropout=config['condition_dropout'],contact_dropout=0.)
 B=len(rows);recipe=config['recipe'];g=src.ground_batch(rows,device)
 hard=g['contact_hard'];soft=g['contact_soft'];valid=g['contact_valid']
 if hard.shape!=valid.shape or soft.shape!=valid.shape:raise ValueError('contact mask shape')
 soft=torch.where(valid,soft,torch.zeros_like(soft));hard=hard & valid
 spec=SkeletonSpec(src.name,src.parents,src.offsets,src.names)
 # Model initialization/dropout must never advance the data-mask stream.
 mask_seed=int(rng.integers(0,2**31-1))
 devices=[device.index if device.index is not None else torch.cuda.current_device()] if device.type=='cuda' else []
 with torch.random.fork_rng(devices=devices):
  torch.manual_seed(mask_seed)
  hidden=sample_joint_mask(spec,B,parse_mix(recipe['mask_mix']),device=device,contact=soft,contact_bin=hard)
  hidden &= torch.rand(B,1,device=device)<recipe['mask_prob']
  drop=(torch.rand(B,1,device=device)<config['contact_dropout']) & ~hidden.any(1,keepdim=True)
 con=src.tensor('contact_sdist',rows,device).float()
 cv=src.tensor('contact_valid',rows,device).bool()
 # Contact is an independent declared sensor/constraint, not hidden pose leakage.
 # Old code bypasses contact dropout for ANY row with a hidden joint.
 inputs['source_hidden']=hidden
 inputs['source_contact']=torch.where(cv & ~drop,con,torch.full_like(con,float('nan')))
 return inputs,target


def xwbc_combos(body_order,rng,count=8):
 combos=list(itertools.product(body_order,body_order))
 if len(set(body_order))!=len(body_order) or count>len(combos):raise ValueError('invalid direction universe')
 return [combos[int(i)] for i in rng.choice(len(combos),count,replace=False)]


def backward_training_step(model,bodies,config,rng,combo_rng,reenc_rng,*,device,domain,forward_fn,objective_fn,autocast_fn,force_all_losses=False):
 """Accumulate one step; no clipping, optimizer step, file writes or CUDA setup.
 Main directions are backpropagated one by one with the original 1/N reduction.
 Auxiliary losses use their own original weights and are not divided by N.
 """
 pool=[b for b in bodies if b.domain==domain and len(b.eligible(True))]
 if not pool:raise ValueError('empty training domain')
 metrics={};directions=[]
 def run(src,tgt,rows,inputs,target,scale,mode):
  with autocast_fn(device):out=forward_fn(model,src,tgt,inputs)
  loss,terms=objective_fn(out,target,tgt,rows,inputs['condition'],config['loss'])
  (scale*loss).backward()
  metrics['total']=metrics.get('total',0.)+float(loss.detach())*scale
  for k,v in terms.items():
   if not k.startswith('count_'):metrics[k]=metrics.get(k,0.)+float(v.detach())*scale
  sampling=hashlib.sha256(np.asarray(rows,dtype=np.int64).tobytes())
  for v in [inputs['source_hidden'],inputs['condition']['joint_indices'],inputs['condition']['valid']]:sampling.update(v.detach().cpu().numpy().tobytes())
  if inputs.get('source_contact') is not None:sampling.update(torch.isfinite(inputs['source_contact']).cpu().numpy().tobytes())
  directions.append({'source':src.name,'target':tgt.name,'mode':mode,'rows':len(rows),'valid_conditions':int(inputs['condition']['valid'].sum()),'weight':scale,
    'actual_source_dataset':src.metadata.get('actual_dataset',src.domain),'actual_target_dataset':tgt.metadata.get('actual_dataset',tgt.domain),
    'selected_rows':np.asarray(rows,dtype=np.int64).tolist(),'sampling_sha256':sampling.hexdigest(),
    'valid_loss_samples':{k.removeprefix('count_'):int(v.detach()) for k,v in terms.items() if k.startswith('count_')},
    'selected_row_sha256':hashlib.sha256(np.asarray(rows,dtype=np.int64).tobytes()).hexdigest(),
    'source_pose_tokens':len(rows)*len(src.parents),'target_joint_tokens':len(rows)*len(tgt.parents),
    'source_surface_tokens':len(rows)*int(src.metadata.get('surface_patch_count',0)),
    'target_surface_tokens':len(rows)*int(tgt.metadata.get('surface_patch_count',0))})
 if domain=='xwbc':
  from recipe_auxiliary import auxiliary_losses
  by={b.name:b for b in pool}
  order=config['judge_body_order']
  if set(order)!=set(by) or len(order)<6:raise ValueError('verified training-only body map with at least six bodies required')
  rows=sample_rows(by[order[0]],rng,config['batch_size'])
  dirs=xwbc_combos(order,combo_rng,config['recipe']['combo_sample'])
  for sn,tn in dirs:
   src,tgt=by[sn],by[tn]
   inputs,target=make_reference_inputs(src,tgt,rows,rng,device,config)
   run(src,tgt,rows,inputs,target,1./len(dirs),'reference_mask_mix')
  if force_all_losses:
   class ForcedReenc:
    def random(self):return 0.
    def integers(self,*a,**kw):return reenc_rng.integers(*a,**kw)
   active_reenc=ForcedReenc()
  else:active_reenc=reenc_rng
  with autocast_fn(device):
   aux=auxiliary_losses(model,by,rows,device=device,judge_body_order=order,combo_rng=combo_rng,reenc_rng=active_reenc,recipe=config['recipe'])
  aux['total'].backward()
  metrics['total']+=float(aux['total'].detach())
  for key in ['contrast','judge','reenc']:
   if aux.get(key) is not None:metrics[key]=float(aux[key].detach())
  extra=config['xwbc_sparse_extra']
  if force_all_losses or rng.random()<extra['probability']:
   selected=[b for b in pool if len(b.endpoint['indices'])]
   if not selected:raise ValueError('no lawful sparse endpoint')
   tgt=selected[int(rng.integers(len(selected)))];rr=sample_rows(tgt,rng,config['batch_size'])
   inputs,target=make_inputs(tgt,tgt,rr,rng,device,'sparse_only',condition_dropout=config['condition_dropout'],contact_dropout=1.)
   run(tgt,tgt,rr,inputs,target,extra['weight'],'sparse_only_extra')
  diagnostics=aux['diagnostics']
 elif domain=='unimate':
  tgt=pool[int(rng.integers(len(pool)))];mode=str(rng.choice(['full','masked','sparse_only'],p=config['mixture']['modes']))
  if mode=='sparse_only' and not tgt.endpoint['indices']:mode='masked'
  rows=sample_rows(tgt,rng,config['batch_size']);inputs,target=make_inputs(tgt,tgt,rows,rng,device,mode,condition_dropout=config['condition_dropout'],contact_dropout=1.)
  run(tgt,tgt,rows,inputs,target,1.,mode)
  diagnostics={'cross_body_auxiliary_applied':False,'reason':'No paired cross-body labels, judge category or metric ground supervision for UniMate'}
 else:raise ValueError(domain)
 return {'domain':domain,'terms':metrics,'directions':directions,'auxiliary':diagnostics}
