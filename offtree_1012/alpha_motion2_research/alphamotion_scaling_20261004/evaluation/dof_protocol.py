"""Separate raw free posture, true DOF projection, and constrained refinement.

Adapters must come from actual verified robot model/frame/joint mappings. This
module never invents a global-R-to-q map. Without that evidence, projection and
constrained manipulation are explicitly unsupported, even with low free EE error.
"""
from dataclasses import dataclass,replace
from pathlib import Path
from typing import Callable
import hashlib
import torch
from trajectory_refinement import RefinementConfig,LegalConstraint,refine_trajectory


@dataclass
class VerifiedDOFAdapter:
    joint_names: tuple[str,...]
    coordinate_units: tuple[str,...]          # radian | metre | dimensionless for actual model q
    lower: torch.Tensor
    upper: torch.Tensor
    project_free: Callable                  # raw_state -> actual q[T,Q]; model-specific verified map
    decode_q: Callable                      # q -> root-relative FK[T,J,3], globalR[T,J,3,3]
    source_files_sha256: dict[str,str]        # actual axes/origin/limits and mapping source files
    model_mapping_description: str
    projection_method: str
    default_q: torch.Tensor | None = None
    default_source: str | None = None

    def validate(self):
        Q=len(self.joint_names)
        if not Q or len(set(self.joint_names))!=Q or len(self.coordinate_units)!=Q:raise ValueError('actual DOF order required')
        if any(u not in {'radian','metre','dimensionless'} for u in self.coordinate_units):raise ValueError('DOF coordinate units')
        if self.lower.shape!=(Q,) or self.upper.shape!=(Q,) or bool(torch.isnan(self.lower).any() or torch.isnan(self.upper).any()) or bool((self.lower>self.upper).any()):raise ValueError('actual DOF limits')
        if not self.source_files_sha256 or not self.model_mapping_description or not self.projection_method:raise ValueError('missing actual model/mapping/limit provenance')
        if self.default_q is not None:
            if self.default_q.shape!=(Q,) or not bool(torch.isfinite(self.default_q).all()) or bool(((self.default_q<self.lower)|(self.default_q>self.upper)).any()) or not self.default_source:raise ValueError('declared default q must be within actual known limits')
        for name,digest in self.source_files_sha256.items():
            path=Path(name)
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('DOF source hash mismatch: '+name)
        return self

    def clamp(self,q):
        if q.shape[-1]!=len(self.joint_names):raise ValueError('q width disagrees with verified model')
        return torch.maximum(torch.minimum(q,self.upper.to(q)),self.lower.to(q))

    def violations(self,q,*unused):
        return torch.relu(q-self.upper.to(q))+torch.relu(self.lower.to(q)-q)


def three_stage_refinement(initial_free_state,raw_decode,condition,*,adapter=None,
                           config=RefinementConfig(),constraints=(),frozen=(),observer=None):
    """Two one-pass optimizer trajectories plus instantaneous projection diagnostics.

    Raw free posture is sampled0..N from one optimizer. At each raw sample, actual
    q projection is evaluated without optimizing or using GT. Constrained q runs
    once0..N from the projected INITIAL raw posture. Wall times of both runs are
    distinct; neither restarts to fabricate a separate result for each step count.
    """
    from trajectory_refinement import _targets
    import time
    if adapter is None:
        raw=refine_trajectory(initial_free_state,raw_decode,condition,config=config,frozen=frozen,observer=observer)
        raw['protocol']['feasibility_level']='raw_free_posture'
        unsupported={'status':'unsupported','reason':'no verified actual DOF mapping, axes/origin/limits adapter'}
        return {'schema':'three-stage-manipulation-v1','raw_free_posture':raw,
                'dof_projected':unsupported.copy(),'constrained_refinement':unsupported.copy(),
                'claim':'free posture SE3 residual is a kinematic diagnostic, not executable joint motion'}
    adapter.validate();evidence='; '.join(sorted(adapter.source_files_sha256))
    known=LegalConstraint('verified_joint_limits','known_joint_limits',adapter.violations,evidence,
                          validity_description='all actual model DOF; explicit continuous limits allowed',unit='per-coordinate declared q unit')
    projected_records=[];initial_q=[];projected_start=time.perf_counter()
    def project_snapshot(step,state,unused_p,unused_r):
        tick=time.perf_counter()
        with torch.no_grad():
            q=adapter.clamp(adapter.project_free(state))
            if q.ndim!=2 or q.shape[0]!=len(state) or not bool(torch.isfinite(q).all()):raise ValueError('projected actual q trajectory shape/finite')
            if step==0:initial_q.append(q.detach().clone())
            p,r=adapter.decode_q(q);_,metrics=_targets(p,r,condition,config)
            violations={}
            for c in (known,*constraints):
                c.validate();v=c.residual(q,p,r)
                if not bool(torch.isfinite(v).all()) or bool((v<0).any()):raise ValueError('invalid projected constraint residual')
                violations[c.name]={'violations':int((v>0).sum()),'values':v.numel(),'max_residual':float(v.max()) if v.numel() else None,'unit':c.unit}
            observed=observer(step,p.detach(),r.detach()) if observer is not None else None
        record={'step':step,'status':'measured','metrics':metrics,'constraints':violations,
                'elapsed_seconds':time.perf_counter()-projected_start,'projection_seconds':time.perf_counter()-tick,
                'executed_optimizer_steps':0,'source_free_optimizer_steps':step,'observation':observed}
        projected_records.append(record)
        return {'dof_projection_measured':True}
    raw=refine_trajectory(initial_free_state,raw_decode,condition,config=config,frozen=frozen,observer=observer,state_observer=project_snapshot)
    raw['protocol']['feasibility_level']='raw_free_posture'
    if not initial_q:raise ValueError('initial free posture could not be projected')
    for record in raw['records'][len(projected_records):]:
        projected_records.append({'step':record['step'],'status':'not_executed_after_raw_failure','elapsed_seconds':record['elapsed_seconds'],'executed_optimizer_steps':0})
    projected={'records':projected_records,'requested_steps':config.steps,'status':'measured' if len(projected_records)==config.steps+1 and not raw['failure_reason'] else 'incomplete',
               'protocol':{'feasibility_level':'actual_DOF_projected','units':condition.units,'optimizer_instances':0,
                           'sampling':'instantaneous actual DOF projection after each raw free-posture refinement sample'}}
    cfg=replace(config,state_representation='actual named joint coordinates '+repr(adapter.coordinate_units))
    refined=refine_trajectory(initial_q[0],adapter.decode_q,condition,config=cfg,constraints=(known,*constraints),frozen=frozen,observer=observer,
                              state_projector=adapter.clamp,projection_source=evidence)
    refined['protocol']['actual_DOF_evidence']={'joint_names':list(adapter.joint_names),'coordinate_units':list(adapter.coordinate_units),
       'source_files_sha256':adapter.source_files_sha256,'mapping':adapter.model_mapping_description,'projection_method':adapter.projection_method}
    return {'schema':'three-stage-manipulation-v1','raw_free_posture':raw,'dof_projected':projected,'constrained_refinement':refined,
            'claim':'joint-coordinate limits evaluated; contact/penetration only if an actual adapter supplied; no dynamics or physical-execution claim'}


def refinement_outcome(result):
    measured=[r for r in result['records'] if r.get('status')=='measured']
    reached=[r for r in measured if r.get('target_tolerances_met') and r.get('known_constraints_met')]
    first=reached[0] if reached else None;final=measured[-1] if measured else None
    return {'step0':measured[0] if measured and measured[0]['step']==0 else None,
            'first_step_meeting_threshold':first['step'] if first else None,
            'seconds_to_threshold':first['elapsed_seconds'] if first else None,
            'final_targets_and_known_constraints_met':bool(result['status']=='tolerances_met'),
            'final':final,'requested_steps':result['requested_steps'],'executed_steps':result['executed_optimizer_steps'],
            'wall_seconds':result['wall_seconds'],'unknown_constraints':'Unknown contact/collision fields are not certified by a successful known-limit result.'}


def compare_learned_vs_default(initial_free_state,raw_decode,condition,*,adapter=None,
                               config=RefinementConfig(),constraints=(),frozen=(),observer=None):
    """One extra evaluation baseline, no seventh training run and no full-IK claim."""
    learned=three_stage_refinement(initial_free_state,raw_decode,condition,adapter=adapter,config=config,
                                   constraints=constraints,frozen=frozen,observer=observer)
    if adapter is None:return {'learned':learned,'default_valid_q_baseline':{'status':'unsupported','reason':'requires verified actual DOF adapter'}}
    adapter.validate()
    if adapter.default_q is None or not adapter.default_source:raise ValueError('source-defined default q required; no invented nominal robot pose')
    q=adapter.default_q.to(initial_free_state).expand(len(initial_free_state),-1).clone()
    evidence='; '.join(sorted(adapter.source_files_sha256))
    known=LegalConstraint('verified_joint_limits','known_joint_limits',adapter.violations,evidence,
                         validity_description='all actual model DOF',unit='per-coordinate declared q unit')
    cfg=replace(config,state_representation='actual named joint coordinates '+repr(adapter.coordinate_units))
    default=refine_trajectory(q,adapter.decode_q,condition,config=cfg,constraints=(known,*constraints),frozen=frozen,observer=observer,
                             state_projector=adapter.clamp,projection_source=evidence)
    default['protocol']['initialization']='source-defined constant default q for every true trajectory frame'
    default['protocol']['default_q_source']=adapter.default_source
    return {'schema':'learned-vs-default-refinement-v1','learned':learned,'default_valid_q_baseline':default,
            'comparison':{'learned_projected_initialization':refinement_outcome(learned['constrained_refinement']),
                          'default_initialization':refinement_outcome(default),
                          'same_conditions':True,'same_optimizer_step_size_steps_and_constraints':True,
                          'claim':'Initial-pose comparison under one matched restricted refinement protocol; not a comparison against a complete IK solver.'}}
