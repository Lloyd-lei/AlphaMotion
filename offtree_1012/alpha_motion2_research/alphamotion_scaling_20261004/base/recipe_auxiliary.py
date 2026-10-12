"""Original L_M20k_0903 auxiliary losses on verified XWBC source-row pairs only.

Truth: train_dual.py:502-513 (raw-hpos cycle),609-644 (InfoNCE/judge),
reference_recipe/L_M20k_0903.config.json. This module never steps an optimizer.
The caller owns autocast/backward. The judge is freshly initialized because old
class order is unverified; persist current validated chosen_split.body_names as its
class order. No old judge-class meaning is assumed.
"""
from __future__ import annotations
from dataclasses import dataclass,fields
from itertools import combinations,product
from collections.abc import Mapping
import numpy as np
import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class AuxiliaryRecipe:
    w_contrast:float=.5
    tau:float=.1
    contrast_sample:int=6
    judge_mode:str='grad'
    w_judge:float=.1
    reenc_consistency:float=1.
    reenc_prob:float=.3

    @classmethod
    def from_mapping(cls,value=None):
        if isinstance(value,cls):return value
        value={} if value is None else value
        out=cls(**{f.name:value[f.name] for f in fields(cls) if f.name in value})
        out.validate();return out

    def validate(self):
        if not all(np.isfinite(v) and v>=0 for v in [self.w_contrast,self.w_judge,self.reenc_consistency]):raise ValueError('auxiliary weights')
        if not np.isfinite(self.tau) or self.tau<=0:raise ValueError('contrast temperature')
        if type(self.contrast_sample) is not int or self.contrast_sample<0:raise ValueError('contrast sample count')
        if self.judge_mode not in {'off','observe','grad'}:raise ValueError('judge mode')
        if not 0<=self.reenc_prob<=1:raise ValueError('reenc probability')


def _stable_unique_indices(row_ids,device):
    ids=np.asarray(row_ids)
    if ids.ndim!=1 or ids.dtype.kind not in 'iu':raise ValueError('source row_ids must be an integer vector')
    _,first=np.unique(ids,return_index=True)
    return torch.as_tensor(np.sort(first),device=device,dtype=torch.long)


def contrast_loss(zs,row_ids,tau=.1):
    """Legacy symmetric InfoNCE, averaged over unordered body pairs.

    Only InfoNCE deduplicates repeated source rows. Judges retain the complete B
    examples and original weighting. Unique source rows reproduce the old CE loss.
    Codes are decoder-ready FSQ outputs; flatten position then rotation, normalize.
    """
    if not zs:raise ValueError('no contrast encodings')
    if tau<=0 or not np.isfinite(tau):raise ValueError('contrast temperature')
    first=next(iter(zs.values()))[0];batch=len(first)
    ix=_stable_unique_indices(row_ids,first.device)
    if len(row_ids)!=batch:raise ValueError('contrast row count')
    vectors={}
    for name,(zp,zr) in zs.items():
        if zp.ndim!=3 or zr.ndim!=3 or len(zp)!=batch or len(zr)!=batch:raise ValueError('latent shape')
        v=torch.cat([zp.reshape(batch,-1),zr.reshape(batch,-1)],-1).float()
        vectors[name]=F.normalize(v.index_select(0,ix),dim=-1)
    pairs=list(combinations(vectors,2))
    zero=sum(v.sum()*0 for v in vectors.values())
    if len(ix)<2 or not pairs:return zero,{'pairs':len(pairs),'unique_rows':len(ix),'duplicate_rows':batch-len(ix),'active':False}
    labels=torch.arange(len(ix),device=first.device)
    losses=[]
    for a,b in pairs:
        logits=(vectors[a]@vectors[b].t())/tau
        losses.append(.5*(F.cross_entropy(logits,labels)+F.cross_entropy(logits.t(),labels)))
    return torch.stack(losses).mean(),{'pairs':len(pairs),'unique_rows':len(ix),'duplicate_rows':batch-len(ix),'active':True}


def judge_loss(model,zs,judge_body_order,mode='grad'):
    """Legacy16-way independent sigmoid BCE, one-hot target, mean over bodies."""
    names=tuple(judge_body_order)
    if len(names)<2 or len(set(names))!=len(names):raise ValueError('judge requires validated training-body class order')
    if mode not in {'off','observe','grad'}:raise ValueError('judge mode')
    if not zs:raise ValueError('no judge encodings')
    zero=next(iter(zs.values()))[0].sum()*0
    if mode=='off':return zero
    model.set_judge_body_order(names)
    losses=[]
    for name,(zp,zr) in zs.items():
        if name not in names:raise ValueError('body has no current judge label: '+name)
        logits=model.judge_logits(zp,zr,mode=mode)
        if logits.shape!=(len(zp),len(names)):raise ValueError('judge output shape')
        target=torch.zeros_like(logits);target[:,names.index(name)]=1.
        losses.append(F.binary_cross_entropy_with_logits(logits.float(),target.float()))
    return torch.stack(losses).mean()


def cycle_consistency(model,source,source_geom,target_geom):
    """Unconditional encode→decode→encode; both latent operands retain gradients.

    Input is full9D, contact=None, no source mask or target SE3. Crucially the
    decoder's raw hpos is re-encoded, not FK positions or target ground truth.
    """
    if source.ndim!=3 or source.shape[-1]!=9:raise ValueError('cycle requires full pose9 only')
    zp,zr=model.encode(source,source_geom)
    rot,raw_hpos=model.decode(zp,zr,target_geom)
    zp_re,zr_re=model.encode(torch.cat([rot,raw_hpos],-1),target_geom)
    return (zp_re-zp).square().mean()+(zr_re-zr).square().mean()


def _paired_bodies(bodies,rows,judge_body_order):
    if not isinstance(bodies,Mapping) or not bodies:raise ValueError('body mapping required')
    order=tuple(judge_body_order)
    if len(order)<6 or len(set(order))!=len(order):raise ValueError('at least six verified training bodies required for original contrast recipe')
    if any(name not in order for name in bodies):raise ValueError('auxiliary body absent from current judge order')
    names=[n for n in order if n in bodies]
    rows=np.asarray(rows)
    if rows.ndim!=1 or rows.dtype.kind not in 'iu' or len(rows)==0:raise ValueError('nonempty integer selected rows required')
    ref=None
    for name in names:
        body=bodies[name]
        if body.domain!='xwbc':raise ValueError('auxiliary pairing is XWBC-only; UniMate rows are not positive pairs')
        if body.name!=name:raise ValueError('body identity mismatch')
        if np.any(rows<0) or np.any(rows>=len(body.arrays['pose9'])):raise ValueError('selected row out of range')
        if np.any(body.arrays['split'][rows]!=0):raise ValueError('auxiliary losses require training rows')
        selected={}
        for key in ['row_ids','source_cache_index','source_row_index','group_index','frame_index','timestamp_seconds']:
            if key not in body.arrays:raise ValueError('missing verified pairing field '+key)
            selected[key]=body.arrays[key][rows]
            if ref is not None and not np.array_equal(selected[key],ref[key]):raise ValueError('cross-body source pairing mismatch '+key)
        if ref is None:ref=selected
    return names,rows,ref['row_ids']


def auxiliary_losses(model,bodies_by_name,rows,*,device,judge_body_order,
                     combo_rng,reenc_rng,recipe=None,combos=None):
    """Return graph-valued total/contrast/judge/reenc and JSON-safe diagnostics.

    Call once per XWBC paired main batch, then total.backward() once. `combos`
    means the FULL allowed direction list, not the sampled8 main combinations.
    Independent default legacy RNG streams are caller-owned: default_rng(4) for
    combo/contrast sampling; default_rng(seed*1000+77) for cycle probability/direction.
    Do not call for UniMate; its unpaired samples are not eligible for this recipe.
    """
    cfg=AuxiliaryRecipe.from_mapping(recipe);cfg.validate()
    names,rows,row_ids=_paired_bodies(bodies_by_name,rows,judge_body_order)
    model.set_judge_body_order(judge_body_order)
    model.judge_mode=cfg.judge_mode
    zero=next(model.parameters()).sum()*0
    reenc=zero;contrast=zero;judge=zero
    diagnostics={'domain':'xwbc','batch_rows':len(rows),'judge_body_order':list(judge_body_order),
      'judge_mode':cfg.judge_mode,'judge_label_order_provenance':'fresh judge uses current validated chosen_split.body_names; historical class order not reused',
      'reenc_active':False,'reenc_frames':0,'reenc_direction':None,'contrast_bodies':[],
      'dedup_scope':'InfoNCE only; full batch retained for judge and original prefix retained for cycle'}
    if cfg.reenc_consistency>0 and float(reenc_rng.random())<cfg.reenc_prob:
        directions=list(product(names,repeat=2)) if combos is None else list(combos)
        if not directions or any(a not in names or b not in names for a,b in directions):raise ValueError('cycle directions must be XWBC train-body pairs')
        a,b=directions[int(reenc_rng.integers(len(directions)))]
        # Original max(8,bs//2); cap only protects explicit tiny synthetic smoke batches.
        count=min(len(rows),max(8,len(rows)//2));sub=rows[:count]
        source=bodies_by_name[a];target=bodies_by_name[b]
        reenc=cycle_consistency(model,source.tensor('pose9',sub,device).float(),source.geom(count),target.geom(count))
        diagnostics.update(reenc_active=True,reenc_frames=count,reenc_direction=[a,b])
    if cfg.w_contrast>0 or (cfg.w_judge>0 and cfg.judge_mode!='off'):
        cnames=names
        if cfg.contrast_sample and cfg.contrast_sample<len(names):
            pick=combo_rng.choice(len(names),cfg.contrast_sample,replace=False)
            cnames=[names[int(i)] for i in pick]
        zs={n:model.encode(bodies_by_name[n].tensor('pose9',rows,device).float(),bodies_by_name[n].geom(len(rows))) for n in cnames}
        diagnostics['contrast_bodies']=cnames
        if cfg.w_contrast>0:contrast,diag=contrast_loss(zs,row_ids,cfg.tau);diagnostics['contrast']=diag
        if cfg.w_judge>0 and cfg.judge_mode!='off':judge=judge_loss(model,zs,judge_body_order,cfg.judge_mode)
    total=cfg.w_contrast*contrast+cfg.w_judge*judge+cfg.reenc_consistency*reenc
    return {'total':total,'contrast':contrast,'judge':judge,'reenc':reenc,'diagnostics':diagnostics}
