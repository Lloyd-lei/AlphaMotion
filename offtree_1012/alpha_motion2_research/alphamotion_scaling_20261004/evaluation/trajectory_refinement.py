"""One fixed optimization trajectory, evaluated at each integer step0..100.

The optimizer receives declared visible SE3 targets and optional physically known
limit/contact residuals ONLY. Full-body GT is not part of this API. Ground-truth
metrics may be computed by an observer of detached outputs, never in the loss.
"""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass,field
import math
import time
from typing import Callable
import torch
from torch import nn


@dataclass
class VisibleSE3:
    joint_indices: torch.Tensor             # [T,C] int64, only actual named joints
    positions: torch.Tensor                 # [T,C,3], same canonical frame/unit as FK
    rotations: torch.Tensor                 # [T,C,3,3]
    position_valid: torch.Tensor            # [T,C] bool
    rotation_valid: torch.Tensor            # [T,C] bool
    joint_names: tuple[str,...]              # complete target body names, verify indices
    source: str                             # actual observation/feasible trajectory provenance
    units: str                              # cm | canonical_body_units
    frame: str = 'canonical_root_relative'

    def validate(self,T,J):
        ix=self.joint_indices;shape=ix.shape
        if ix.ndim!=2 or shape[0]!=T or ix.dtype!=torch.long:raise ValueError('condition joint indices [T,C] int64')
        if len(self.joint_names)!=J or len(set(self.joint_names))!=J or not self.source:raise ValueError('actual named-joint/provenance contract')
        if self.units not in {'cm','canonical_body_units'} or self.frame!='canonical_root_relative':raise ValueError('explicit compatible condition frame/unit required')
        if self.positions.shape!=(*shape,3) or self.rotations.shape!=(*shape,3,3):raise ValueError('SE3 value shape')
        for valid,value in [(self.position_valid,self.positions),(self.rotation_valid,self.rotations)]:
            if valid.shape!=shape or valid.dtype!=torch.bool:raise ValueError('condition validity shape/type')
            if not bool(torch.isfinite(value[valid]).all()):raise ValueError('nonfinite valid SE3 value')
        v=self.position_valid|self.rotation_valid
        if bool(((ix[v]<0)|(ix[v]>=J)).any()):raise ValueError('visible condition joint absent from actual body')
        for i in range(T):
            ids=ix[i,v[i]]
            if len(torch.unique(ids))!=len(ids):raise ValueError('duplicate conditions on one joint/frame')
        R=self.rotations[self.rotation_valid]
        if len(R) and (not torch.allclose(R.transpose(-1,-2)@R,torch.eye(3,device=R.device,dtype=R.dtype),atol=2e-3,rtol=2e-3) or not torch.allclose(torch.linalg.det(R),torch.ones(len(R),device=R.device,dtype=R.dtype),atol=2e-3,rtol=2e-3)):
            raise ValueError('valid target rotations must be SO3')
        if not bool(v.any()):raise ValueError('at least one actual visible SE3 condition required')


@dataclass
class LegalConstraint:
    name: str
    kind: str                               # known_joint_limits | available_contact
    residual: Callable                      # residual(state,position,rotation) -> nonnegative tensor
    source: str                             # verified model limits / declared contact source
    weight: float = 1.
    unit: str = 'dimensionless'
    validity_description: str = ''

    def validate(self):
        if self.kind not in {'known_joint_limits','available_contact'}:raise ValueError('only known limits or available contact may supplement SE3 loss')
        if not self.name or not self.source or not self.validity_description:raise ValueError('constraint provenance and validity required')
        if not math.isfinite(self.weight) or self.weight<0:raise ValueError('constraint weight')


@dataclass(frozen=True)
class RefinementConfig:
    steps: int = 100
    optimizer: str = 'adam'
    learning_rate: float = .01
    position_scale: float = 1.
    position_weight: float = 1.
    rotation_weight: float = 1.
    position_tolerance: float = 1.            # cm only if units=cm
    rotation_tolerance_deg: float = 5.
    max_conditions: int = 2
    state_representation: str = 'caller_declared_continuous_state'
    evidence: str = 'kinematic refinement only; no physical execution or dynamics claim'

    def validate(self):
        if type(self.steps) is not int or not 0<=self.steps<=100:raise ValueError('steps must be an integer0..100')
        if self.optimizer not in {'adam','sgd'}:raise ValueError('fixed supported optimizer required')
        for v in [self.learning_rate,self.position_scale]:
            if not math.isfinite(v) or v<=0:raise ValueError('positive lr/position scale')
        for v in [self.position_weight,self.rotation_weight,self.position_tolerance,self.rotation_tolerance_deg]:
            if not math.isfinite(v) or v<0:raise ValueError('nonnegative loss/tolerance')
        if type(self.max_conditions) is not int or self.max_conditions<1:raise ValueError('explicit maximum observed joints')


@contextmanager
def frozen_modules(modules):
    states=[]
    try:
        for module in modules:
            states.append((module,module.training,[(p,p.requires_grad) for p in module.parameters()]))
            module.eval()
            for p in module.parameters():p.requires_grad_(False)
        yield
    finally:
        for module,training,params in states:
            module.train(training)
            for p,requires in params:p.requires_grad_(requires)


def _sync(tensor):
    if tensor.device.type=='cuda':torch.cuda.synchronize(tensor.device)


def _angles(A,B):
    Q=A.transpose(-1,-2)@B;c=(Q.diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2
    skew=torch.stack([Q[...,2,1]-Q[...,1,2],Q[...,0,2]-Q[...,2,0],Q[...,1,0]-Q[...,0,1]],-1)
    return torch.atan2(.5*skew.norm(dim=-1),c.clamp(-1,1))*(180/math.pi)


def _targets(positions,rotations,condition,config):
    T,J=positions.shape[:2];condition.validate(T,J)
    if condition.joint_indices.shape[1]>config.max_conditions:raise ValueError('more joints than declared legal condition budget')
    ix=condition.joint_indices;pv=condition.position_valid;rv=condition.rotation_valid
    batch=torch.arange(T,device=positions.device)[:,None].expand_as(ix)
    zero=positions.reshape(-1)[:0].sum()+rotations.reshape(-1)[:0].sum()
    loss=zero;metrics={}
    if bool(pv.any()):
        err=(positions[batch[pv],ix[pv]]-condition.positions[pv]).norm(dim=-1)
        loss=loss+config.position_weight*(err/config.position_scale).square().mean()
        metrics['selected_position_mean']=float(err.detach().mean());metrics['selected_position_max']=float(err.detach().max())
    if bool(rv.any()):
        pred=rotations[batch[rv],ix[rv]];target=condition.rotations[rv]
        # Smooth chordal rotation loss gives finite gradients at identity and pi.
        loss=loss+config.rotation_weight*((pred-target).square().sum((-1,-2))/2).mean()
        angles=_angles(pred,target);metrics['selected_rotation_mean_deg']=float(angles.detach().mean());metrics['selected_rotation_max_deg']=float(angles.detach().max())
    metrics['position_unit']=condition.units
    metrics['valid_position_targets']=int(pv.sum());metrics['valid_rotation_targets']=int(rv.sum())
    return loss,metrics


def refine_trajectory(initial_state:torch.Tensor,decode_fn:Callable,condition:VisibleSE3,*,
                      config=RefinementConfig(),constraints=(),frozen=(),observer=None,state_projector=None,projection_source=None,state_observer=None):
    """decode_fn(state)->(root_relative_positions[T,J,3], global_R[T,J,3,3]).

    One optimizer instance, fixed step size, no restart/best-step selection. Every
    integer step is returned including unchanged failed-tail records if numerical
    optimization stops. Observer gets detached tensors and cannot influence loss.
    The caller declares state representation; global-R6 optimization alone cannot
    establish robot joint-limit feasibility without a real q/limit adapter.
    """
    config.validate()
    if state_projector is not None and not projection_source:raise ValueError('state projection needs actual known-DOF provenance')
    if not torch.is_floating_point(initial_state) or not bool(torch.isfinite(initial_state).all()):raise ValueError('finite continuous initial state required')
    for c in constraints:c.validate()
    if len({c.name for c in constraints})!=len(constraints):raise ValueError('unique constraint names')
    state=nn.Parameter(initial_state.detach().clone())
    optimizer=(torch.optim.Adam if config.optimizer=='adam' else torch.optim.SGD)([state],lr=config.learning_rate)
    records=[];failure=None;executed=0;last_prediction=None
    _sync(state);start=time.perf_counter()
    with frozen_modules(frozen):
        for step in range(config.steps+1):
            if failure is not None:
                records.append({'step':step,'status':'not_executed_after_failure','failure_reason':failure,
                                'elapsed_seconds':time.perf_counter()-start,'executed_optimizer_steps':executed})
                continue
            _sync(state);tick=time.perf_counter();optimizer.zero_grad(set_to_none=True)
            p,r=decode_fn(state)
            if p.ndim!=3 or p.shape[-1]!=3 or r.shape!=(*p.shape[:2],3,3):raise ValueError('decoder FK output contract')
            if len(p)!=len(condition.joint_indices):raise ValueError('decode must preserve trajectory frames')
            if not bool(torch.isfinite(p).all() and torch.isfinite(r).all()):
                failure='nonfinite decoder output';records.append({'step':step,'status':'numerical_failure','failure_reason':failure,
                    'elapsed_seconds':time.perf_counter()-start,'executed_optimizer_steps':executed});continue
            loss,metrics=_targets(p,r,condition,config);constraint_metrics={}
            for c in constraints:
                residual=c.residual(state,p,r)
                if not torch.is_tensor(residual) or not bool(torch.isfinite(residual).all()) or bool((residual<0).any()):raise ValueError('legal residual must be finite and nonnegative: '+c.name)
                term=residual.square().mean() if residual.numel() else loss.reshape(-1)[:0].sum()
                loss=loss+c.weight*term
                constraint_metrics[c.name]={'kind':c.kind,'mean_residual':float(residual.detach().mean()) if residual.numel() else None,
                    'max_residual':float(residual.detach().max()) if residual.numel() else None,'violations':int((residual>0).sum()),'values':residual.numel(),'violation_fraction':float((residual>0).float().mean()) if residual.numel() else None,'unit':c.unit,'source':c.source}
            if not bool(torch.isfinite(loss)):
                failure='nonfinite legal objective';records.append({'step':step,'status':'numerical_failure','failure_reason':failure,'elapsed_seconds':time.perf_counter()-start,'executed_optimizer_steps':executed});continue
            _sync(state);prediction=(p.detach().clone(),r.detach().clone());last_prediction=prediction
            observed=observer(step,*prediction) if observer is not None else None
            state_observed=state_observer(step,state.detach().clone(),*prediction) if state_observer is not None else None
            _sync(state)
            target_success=(metrics.get('selected_position_max',0.)<=config.position_tolerance and metrics.get('selected_rotation_max_deg',0.)<=config.rotation_tolerance_deg)
            legal_success=all(v['violations']==0 for v in constraint_metrics.values())
            records.append({'step':step,'status':'measured','loss':float(loss.detach()),'metrics':metrics,'constraints':constraint_metrics,
                'target_tolerances_met':target_success,'known_constraints_met':legal_success,
                'elapsed_seconds':time.perf_counter()-start,'cumulative_seconds_per_frame':(time.perf_counter()-start)/len(p),'evaluation_seconds':time.perf_counter()-tick,
                'executed_optimizer_steps':executed,'observation':observed,'state_observation':state_observed})
            if step==config.steps:break
            if not bool(torch.isfinite(loss)):
                failure='nonfinite legal objective';continue
            if not loss.requires_grad:
                failure='no differentiable path from state to legal targets';continue
            loss.backward()
            if state.grad is None or not bool(torch.isfinite(state.grad).all()):
                failure='missing or nonfinite refinement-state gradient';continue
            optimizer.step()
            if state_projector is not None:
                with torch.no_grad():
                    projected=state_projector(state)
                    if projected.shape!=state.shape or not bool(torch.isfinite(projected).all()):raise ValueError('DOF projection shape/finite')
                    state.copy_(projected)
            executed+=1
    _sync(state)
    final=records[-1]
    final_status='numerical_failure' if failure else ('tolerances_met' if final['target_tolerances_met'] and final['known_constraints_met'] else 'target_tolerances_not_met')
    return {'schema':'trajectory-refinement-v1','records':records,'status':final_status,'failure_reason':failure,
        'executed_optimizer_steps':executed,'requested_steps':config.steps,'wall_seconds':time.perf_counter()-start,
        'final_state':state.detach().clone(),'final_prediction':last_prediction,
        'protocol':{'optimizer':config.optimizer,'learning_rate':config.learning_rate,'state_representation':config.state_representation,
            'position_scale':config.position_scale,'position_weight':config.position_weight,'rotation_weight':config.rotation_weight,
            'position_tolerance':config.position_tolerance,'rotation_tolerance_deg':config.rotation_tolerance_deg,
            'feasibility_level':'constrained_joint_coordinates' if state_projector is not None else 'raw_free_posture',
            'projection_source':projection_source,'condition_sha256':visible_se3_digest(condition),'units':condition.units,'condition_source':condition.source,'frame':condition.frame,
            'fixed_trajectory':True,'optimizer_instances':1,'no_refine_step':0,'ground_truth_fullbody_loss':False,
            'allowed_objectives':['visible_SE3']+[c.kind for c in constraints],
            'limits_status':'evaluated with declared adapter' if any(c.kind=='known_joint_limits' for c in constraints) else 'unknown; global rotation representation does not prove q limits',
            'penetration_status':'only available if an explicit validated geometry/contact residual supplies penetration; unknown is not zero',
            'contact_status':'evaluated from declared available contact' if any(c.kind=='available_contact' for c in constraints) else 'unavailable',
            'feasibility_claim':'target_tolerances_not_met does not mathematically prove unreachable; failed and difficult cases remain included',
            'evidence':config.evidence}}


def save_refinement(directory,result):
    """Persist JSON curve plus derived terminal state/FK arrays, never source weights."""
    import hashlib,json
    import numpy as np
    from pathlib import Path
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    arrays={'final_state':result['final_state'].detach().cpu().numpy()}
    if result['final_prediction'] is not None:
        arrays.update(final_positions=result['final_prediction'][0].cpu().numpy(),final_rotations=result['final_prediction'][1].cpu().numpy())
    path=directory/'refinement_arrays.npz';np.savez_compressed(path,**arrays)
    report={k:v for k,v in result.items() if k not in {'final_state','final_prediction'}}
    report['arrays']={'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'keys':list(arrays)}
    (directory/'refinement_curve.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def visible_se3_digest(condition):
    """Bind exact actual visible targets and masks, ignoring meaningless padding."""
    import hashlib,json
    h=hashlib.sha256(json.dumps({'names':condition.joint_names,'source':condition.source,'units':condition.units,'frame':condition.frame},sort_keys=True).encode())
    pv=condition.position_valid;rv=condition.rotation_valid;v=pv|rv
    values=[torch.where(v,condition.joint_indices,torch.zeros_like(condition.joint_indices)),
            torch.where(pv[...,None],condition.positions,torch.zeros_like(condition.positions)),
            torch.where(rv[...,None,None],condition.rotations,torch.zeros_like(condition.rotations)),pv,rv]
    for x in values:
        a=x.detach().cpu().contiguous().numpy();h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()
