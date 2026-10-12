"""Auditable dense arithmetic estimate; observed tokens/time remain separate."""
from __future__ import annotations
import math
import torch
from torch import nn


class ComputeCounter:
    """Counts actual module invocations, including repeated auxiliary encodings.

    FLOPs use multiply+add=2, Linear plus attention QK/AV arithmetic. Backward
    is estimated as 2x forward. Norms, activations, softmax, FK, loss, optimizer
    and memory movement are excluded: this is an estimate, never measured FLOPs.
    """
    def __init__(self,model):
        self.handles=[];self.reset()
        for module in model.modules():
            if isinstance(module,nn.Linear):self.handles.append(module.register_forward_hook(self._linear))
            elif isinstance(module,nn.MultiheadAttention):self.handles.append(module.register_forward_hook(self._mha))
            elif type(module).__name__=='GraphBiasAttention':self.handles.append(module.register_forward_hook(self._graph))
            elif type(module).__name__=='AttachmentCrossAttention':self.handles.append(module.register_forward_hook(self._attachment))
    def reset(self):self.forward_flops=0;self.linear_calls=0;self.attention_calls=0
    def _linear(self,m,inputs,output):
        vectors=inputs[0].numel()//m.in_features
        self.forward_flops+=2*vectors*m.in_features*m.out_features;self.linear_calls+=1
    def _mha(self,m,inputs,output):
        q,k,v=inputs[:3];batch=q.shape[0] if m.batch_first else q.shape[1]
        lq=q.shape[1] if m.batch_first else q.shape[0];lk=k.shape[1] if m.batch_first else k.shape[0]
        d=m.embed_dim
        self.forward_flops+=2*batch*(lq*q.shape[-1]*d+lk*k.shape[-1]*d+lk*v.shape[-1]*d+lq*d*d)+4*batch*lq*lk*d
        self.attention_calls+=1
    def _graph(self,m,inputs,output):
        batch,length,d=inputs[0].shape;self.forward_flops+=4*batch*length*length*d;self.attention_calls+=1
    def _attachment(self,m,inputs,output):
        q,k=inputs[:2];batch,lq,d=q.shape;self.forward_flops+=4*batch*lq*k.shape[1]*d;self.attention_calls+=1
    def report(self,backward=True):
        return {'estimated_dense_forward_flops':self.forward_flops,
                'estimated_dense_training_flops':self.forward_flops*(3 if backward else 1),
                'linear_invocations':self.linear_calls,'attention_invocations':self.attention_calls,
                'method':'multiply-add=2; observed Linear/MHA/graph-attention shapes; backward=2x forward',
                'excludes':'norm/activation/softmax/FK/loss/optimizer/memory movement; not measured hardware FLOPs'}
    def close(self):
        for h in self.handles:h.remove()


def effective_tokens(step_report,body_map):
    directions=step_report['directions']
    totals={k:sum(d.get(k,0) for d in directions) for k in ['source_pose_tokens','target_joint_tokens','source_surface_tokens','target_surface_tokens','valid_conditions']}
    aux=step_report.get('auxiliary',{});batch=int(aux.get('batch_rows',0))
    totals['auxiliary_contrast_joint_tokens']=sum(batch*len(body_map[n].parents) for n in aux.get('contrast_bodies',[]))
    totals['auxiliary_contrast_surface_tokens']=sum(batch*int(body_map[n].metadata.get('surface_patch_count',0)) for n in aux.get('contrast_bodies',[]))
    cycle=aux.get('reenc_direction');n=int(aux.get('reenc_frames',0))
    totals['auxiliary_cycle_joint_tokens']=0 if not cycle else n*(len(body_map[cycle[0]].parents)+2*len(body_map[cycle[1]].parents))
    totals['auxiliary_cycle_surface_tokens']=0 if not cycle else n*(int(body_map[cycle[0]].metadata.get('surface_patch_count',0))+2*int(body_map[cycle[1]].metadata.get('surface_patch_count',0)))
    totals['padding_tokens']=sum(2*d['rows']-d['valid_conditions'] for d in directions)
    totals['definition']='Logical source joints (including mask tokens), target joint queries, valid patches and supplied sparse conditions; two condition slots allocate separately counted padding. Auxiliary passes are not independent new examples. Internal FSQ slots excluded from data-token count but included in FLOPs.'
    return totals
