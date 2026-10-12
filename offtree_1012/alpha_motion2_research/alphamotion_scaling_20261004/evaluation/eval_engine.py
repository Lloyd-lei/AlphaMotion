"""Clip-first kinematic evaluation, independent of model architecture.

Positions are physical centimetres only when metric=True, otherwise native body
units. Input rotation matrices are own-joint GLOBAL orientations. Raw hpos is
never Cartesian XYZ. No model training, GPU allocation, downloads, or refinement.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
import json
import math
import time
import numpy as np


@dataclass
class PosePrediction:
    positions: np.ndarray                   # [T,J,3], root-relative canonical
    rotations: np.ndarray                   # [T,J,3,3], global, same canonical frame
    world_positions: np.ndarray | None = None
    world_rotations: np.ndarray | None = None
    sole_world_positions: np.ndarray | None = None  # [T,F,3], only measured/declared world anchor
    failure_reason: str | None = None


@dataclass
class TrajectoryCase:
    case_id: str
    clip_id: str                             # shared physical source identity across bodies
    source_body: str
    target_body: str
    task_kind: str                           # paired_mapping | reconstruction | completion | constraints_only
    body_exposure: str                      # seen_body | heldout_body (entire body excluded from training)
    clip_exposure: str                      # train_clip | heldout_clip
    frame_ids: np.ndarray
    timestamps_seconds: np.ndarray
    joint_names: list[str]
    metric: bool
    reach: float
    reference_positions: np.ndarray | None
    reference_rotations: np.ndarray | None
    root_index: int = 0
    complete_clip: bool = True
    source_frame_count: int | None = None
    reference_kind: str = 'paired_target'    # paired_target | same_body | constraints_only
    role_endpoints: dict[str, list[int]] = field(default_factory=dict)
    reference_world_positions: np.ndarray | None = None
    reference_world_rotations: np.ndarray | None = None
    contact: np.ndarray | None = None        # [T,F], source-supported boolean foot labels
    contact_valid: np.ndarray | None = None
    provenance: dict = field(default_factory=dict)
    condition_set_sha256: str | None = None
    target_morphology: str = 'unspecified'  # actual metadata: e.g. humanoid, quadruped, articulated_object

    def validate(self):
        T = len(self.frame_ids); J = len(self.joint_names)
        if not T or not J or len(set(self.joint_names)) != J: raise ValueError('empty or duplicate joint identity')
        if np.asarray(self.frame_ids).shape != (T,) or np.asarray(self.frame_ids).dtype.kind not in 'iu': raise ValueError('integer frame IDs required')
        ts = np.asarray(self.timestamps_seconds)
        if ts.shape != (T,) or not np.isfinite(ts).all() or (T > 1 and not np.all(np.diff(ts) > 0)): raise ValueError('strictly increasing true timestamps required')
        if self.complete_clip and (self.source_frame_count != T or not np.array_equal(self.frame_ids, np.arange(T))):
            raise ValueError('complete clip requires every original frame0..F-1 and source_frame_count')
        if not 0 <= self.root_index < J or not math.isfinite(self.reach) or self.reach <= 0: raise ValueError('root/reach')
        if self.body_exposure not in {'seen_body', 'heldout_body'} or self.clip_exposure not in {'train_clip', 'heldout_clip'}: raise ValueError('explicit exposure required')
        if self.task_kind not in {'paired_mapping', 'reconstruction', 'completion', 'constraints_only'}: raise ValueError('task kind')
        if self.reference_kind not in {'paired_target', 'same_body', 'constraints_only'}: raise ValueError('reference kind')
        if self.task_kind == 'paired_mapping' and self.reference_kind != 'paired_target': raise ValueError('mapping metrics need paired target reference')
        if self.reference_kind == 'same_body' and self.source_body != self.target_body: raise ValueError('same-body GT cannot prove cross-body mapping')
        if self.reference_kind == 'constraints_only':
            if self.reference_positions is not None or self.reference_rotations is not None or self.task_kind != 'constraints_only': raise ValueError('constraint-only case must not invent fullbody GT')
        else:
            _finite_shape(self.reference_positions, (T,J,3), 'reference positions')
            _rotation_shape(self.reference_rotations, (T,J,3,3), 'reference rotations')
            if not np.allclose(np.asarray(self.reference_positions)[:,self.root_index], 0., atol=1e-5): raise ValueError('reference positions must be root relative')
        for role, indices in self.role_endpoints.items():
            if not role or len(set(indices)) != len(indices) or any(type(i) is not int or not 0 <= i < J for i in indices): raise ValueError('actual endpoint indices required')
        for p, r, prefix in [(self.reference_world_positions, self.reference_world_rotations, 'world reference')]:
            if p is not None: _finite_shape(p, (T,J,3), prefix)
            if r is not None: _rotation_shape(r, (T,J,3,3), prefix)
        if self.contact is not None:
            c = np.asarray(self.contact); v = np.asarray(self.contact_valid)
            if c.ndim != 2 or c.shape[0] != T or v.shape != c.shape or c.dtype != bool or v.dtype != bool: raise ValueError('contact labels and validity must be [T,F] bool')
        return self


def _finite_shape(value, shape, label):
    a = np.asarray(value)
    if a.shape != shape or not np.isfinite(a).all(): raise ValueError(label + ' shape/finite')


def _rotation_shape(value, shape, label):
    a = np.asarray(value); _finite_shape(a, shape, label)
    if not np.allclose(a.swapaxes(-1,-2) @ a, np.eye(3), atol=2e-3) or not np.allclose(np.linalg.det(a), 1., atol=2e-3):
        raise ValueError(label + ' is not SO(3)')


def rotation_error_degrees(prediction, reference):
    q = np.swapaxes(np.asarray(prediction), -1, -2) @ np.asarray(reference)
    c = (np.trace(q, axis1=-2, axis2=-1) - 1.) / 2.
    skew = np.stack([q[...,2,1]-q[...,1,2],q[...,0,2]-q[...,2,0],q[...,1,0]-q[...,0,1]], -1)
    return np.degrees(np.arctan2(np.linalg.norm(skew, axis=-1) / 2., np.clip(c,-1.,1.)))


def metric_summary(values, unit):
    a = np.asarray(values,dtype=float).reshape(-1); good = np.isfinite(a); finite = a[good]
    return {'mean': float(a.mean()) if len(a) and good.all() else None,
            'finite_only_mean': float(finite.mean()) if len(finite) else None,
            'p95': float(np.quantile(a,.95)) if len(a) and good.all() else None,
            'finite_values': int(good.sum()), 'total_values': len(a), 'failed_values': int((~good).sum()), 'unit': unit,
            'aggregation': 'joint/time micro; unconditional mean is null if any prediction failed'}


def _yaw(r):
    # Canonical Y-up: yaw determined from root +Z axis projected into XZ.
    forward = np.asarray(r)[:,2]
    if np.linalg.norm(forward[[0,2]]) < 1e-8: raise ValueError('first-frame heading undefined')
    return math.atan2(forward[0],forward[2])


def _yaw_matrix(yaw):
    c,s=math.cos(yaw),math.sin(yaw)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])


def _prediction_arrays(case, prediction, alignment):
    T,J=len(case.frame_ids),len(case.joint_names)
    p=np.asarray(prediction.positions,dtype=float);r=np.asarray(prediction.rotations,dtype=float)
    if p.shape!=(T,J,3) or r.shape!=(T,J,3,3): raise ValueError('prediction shape must preserve every case frame and joint')
    invalid=~np.isfinite(p).all(-1)|~np.isfinite(r).all((-1,-2))
    with np.errstate(invalid='ignore'):
        invalid |= np.max(np.abs(r.swapaxes(-1,-2)@r-np.eye(3)),axis=(-1,-2))>2e-3
        invalid |= np.abs(np.linalg.det(r)-1.)>2e-3
    p=p.copy();r=r.copy();p[invalid]=np.nan;r[invalid]=np.nan
    if alignment == 'canonical_root_relative':
        p=p-p[:,case.root_index:case.root_index+1]
        return p,r,case.reference_positions,case.reference_rotations,invalid,'root translation removed; original canonical heading retained'
    if alignment == 'first_frame_yaw_aligned':
        if case.reference_rotations is None: raise ValueError('yaw alignment needs reference root orientation')
        A=_yaw_matrix(_yaw(case.reference_rotations[0,case.root_index])-_yaw(r[0,case.root_index]))
        p=np.einsum('ij,tkj->tki',A,p-p[:,case.root_index:case.root_index+1]);r=A@r
        return p,r,case.reference_positions,case.reference_rotations,invalid,'one reference-derived root yaw at frame0; constant transform for whole clip; no per-frame alignment'
    if alignment == 'world':
        if prediction.world_positions is None or case.reference_world_positions is None: raise ValueError('world metric needs separately declared predicted and reference world positions')
        wp=np.asarray(prediction.world_positions,dtype=float)
        if wp.shape!=(T,J,3): raise ValueError('predicted world shape')
        wr=prediction.world_rotations
        if wr is None or case.reference_world_rotations is None: raise ValueError('world rotations require both explicit orientation anchors')
        wr=np.asarray(wr,dtype=float)
        if wr.shape!=(T,J,3,3): raise ValueError('predicted world rotation shape')
        winvalid=~np.isfinite(wp).all(-1)|~np.isfinite(wr).all((-1,-2))
        with np.errstate(invalid='ignore'):
            winvalid|=np.max(np.abs(wr.swapaxes(-1,-2)@wr-np.eye(3)),axis=(-1,-2))>2e-3
            winvalid|=np.abs(np.linalg.det(wr)-1.)>2e-3
        wp=wp.copy();wr=wr.copy();wp[winvalid]=np.nan;wr[winvalid]=np.nan
        return wp,wr,case.reference_world_positions,case.reference_world_rotations,invalid|winvalid,'explicit world anchors; no reference-fit alignment'
    raise ValueError('unsupported alignment')


def temporal_metrics(positions, timestamps, unit):
    p=np.asarray(positions,dtype=float);t=np.asarray(timestamps,dtype=float)
    if len(t)<2:return {'status':'unavailable','reason':'fewer than two true frames'}
    v=np.diff(p,axis=0)/np.diff(t)[:,None,None]
    out={'status':'measured','speed':metric_summary(np.linalg.norm(v,axis=-1),unit+'/s')}
    if len(t)>=3:
        vt=(t[1:]+t[:-1])/2;acc=np.diff(v,axis=0)/np.diff(vt)[:,None,None]
        out['acceleration']=metric_summary(np.linalg.norm(acc,axis=-1),unit+'/s^2')
        if len(t)>=4:
            at=(vt[1:]+vt[:-1])/2;jerk=np.diff(acc,axis=0)/np.diff(at)[:,None,None]
            out['jerk']=metric_summary(np.linalg.norm(jerk,axis=-1),unit+'/s^3')
    out['interpretation']='kinematic temporal variation; low jerk alone does not establish task accuracy'
    return out


def foot_slip(case, prediction):
    if not case.complete_clip or len(case.frame_ids)<2 or not np.all(np.diff(case.frame_ids)==1):
        return {'status':'unsupported','reason':'requires original consecutive true clip frames'}
    if not case.metric or prediction.sole_world_positions is None or case.contact is None:
        return {'status':'unsupported','reason':'requires metric world-anchored sole positions and valid actual foot contact labels'}
    p=np.asarray(prediction.sole_world_positions,dtype=float);contact=np.asarray(case.contact);valid=np.asarray(case.contact_valid)
    if p.shape!=(*contact.shape,3):raise ValueError('sole/contact shape mismatch')
    active=contact[1:]&contact[:-1]&valid[1:]&valid[:-1]
    if not active.any():return {'status':'no_valid_contact_pairs','contact_pairs':0}
    speed=np.linalg.norm(np.diff(p,axis=0)[...,[0,2]],axis=-1)/np.diff(case.timestamps_seconds)[:,None]
    return {'status':'measured','contact_pairs':int(active.sum()),'horizontal_speed':metric_summary(speed[active],'cm/s'),
            'reference':'world-space sole horizontal displacement / actual timestamp delta; both endpoint contact labels valid and active'}


def evaluate_case(case:TrajectoryCase,prediction:PosePrediction,*,alignment='canonical_root_relative',timing=None):
    case.validate();p,r,gtp,gtr,invalid,note=_prediction_arrays(case,prediction,alignment)
    unit='cm' if case.metric else 'canonical_body_units'
    result={'schema':'alphamotion-case-evaluation-v1','case_id':case.case_id,'clip_id':case.clip_id,
            'source_body':case.source_body,'target_body':case.target_body,'task_kind':case.task_kind,
            'condition_set_sha256':case.condition_set_sha256,'target_morphology':case.target_morphology,'reference_kind':case.reference_kind,'body_exposure':case.body_exposure,'clip_exposure':case.clip_exposure,
            'alignment':alignment,'alignment_policy':note,'metric_length_valid':case.metric,
            'frame_count':len(case.frame_ids),'joint_count':len(case.joint_names),'complete_clip':case.complete_clip,
            'failed_frames':int(invalid.any(1).sum()),'failure_reason':prediction.failure_reason,
            'timing':timing,'metrics':{},'endpoints':{},'provenance':case.provenance}
    if prediction.failure_reason:
        result['failed_frames']=len(case.frame_ids);p=np.full_like(p,np.nan);r=np.full_like(r,np.nan)
    if gtp is not None:
        pe=np.linalg.norm(p-np.asarray(gtp),axis=-1);re=rotation_error_degrees(r,gtr)
        if prediction.failure_reason:pe[:]=np.nan;re[:]=np.nan
        result['metrics']={'position':metric_summary(pe,unit),'position_over_reach':metric_summary(pe/case.reach,'reach_fraction'),
                           'rotation':metric_summary(re,'degree')}
        for role,indices in case.role_endpoints.items():
            if indices:result['endpoints'][role]={'indices':indices,'joint_names':[case.joint_names[i] for i in indices],
                'position':metric_summary(pe[:,indices],unit),'position_over_reach':metric_summary(pe[:,indices]/case.reach,'reach_fraction'),
                'rotation':metric_summary(re[:,indices],'degree')}
    else:result['fullbody_reference_status']='not_available; only separately declared SE3/geometry constraints may be evaluated'
    result['temporal']=temporal_metrics(p,case.timestamps_seconds,unit) if case.complete_clip else {'status':'unsupported','reason':'incomplete clip'}
    result['foot_slip']=foot_slip(case,prediction)
    result['status']='failed_prediction' if result['failed_frames'] else 'measured'
    return result


def map_single_frames(frame_count:int,predict_frame:Callable,*,synchronize:Callable=lambda:None):
    """Call an existing frozen model one actual frame at a time; no temporal model claim.

    predict_frame(index)->(positions[J,3], rotations[J,3,3]). Timing includes the
    callback and its synchronized device work. Root/world anchoring remains explicit.
    """
    if frame_count<1:raise ValueError('frame count')
    positions=[];rotations=[];latency=[];synchronize();start=time.perf_counter()
    for i in range(frame_count):
        synchronize();tick=time.perf_counter();p,r=predict_frame(i);synchronize()
        latency.append(time.perf_counter()-tick);positions.append(np.asarray(p));rotations.append(np.asarray(r))
    return PosePrediction(np.stack(positions),np.stack(rotations)),{'wall_seconds':time.perf_counter()-start,
        'frame_latency_seconds':latency,'mean_latency_seconds':float(np.mean(latency)),
        'p95_latency_seconds':float(np.quantile(latency,.95)),'frames':frame_count,'execution':'single-frame callback; synchronized wall-clock'}


def write_result(path,result):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


def validate_exposure(cases,*,training_body_ids,training_clip_ids):
    """Bind labels to a particular run's complete-body and physical-clip split."""
    bodies=set(training_body_ids);clips=set(training_clip_ids)
    for case in cases:
        case.validate()
        expected_body='seen_body' if case.target_body in bodies else 'heldout_body'
        expected_clip='train_clip' if case.clip_id in clips else 'heldout_clip'
        if case.body_exposure!=expected_body or case.clip_exposure!=expected_clip:
            raise ValueError('case exposure contradicts actual run training manifest: '+case.case_id)
    return {'cases':len(cases),'training_bodies':sorted(bodies),'training_clip_count':len(clips),
            'policy':'body means any training appearance; clip means any training source exposure, regardless of target body'}


def map_independent_frame_batches(frame_count:int,batch_size:int,predict_batch:Callable,*,synchronize:Callable=lambda:None):
    """Batch independent single-frame model calls; no temporal context or smoothing.

    predict_batch(integer_frame_indices) returns [B,J,3] and [B,J,3,3]. Wall
    measurements are batch latency and amortized seconds/frame, not serial latency.
    """
    if type(frame_count) is not int or frame_count<1 or type(batch_size) is not int or batch_size<1:raise ValueError('positive frame/batch counts')
    pp=[];rr=[];batches=[];synchronize();start=time.perf_counter()
    for begin in range(0,frame_count,batch_size):
        ids=np.arange(begin,min(frame_count,begin+batch_size));synchronize();tick=time.perf_counter()
        p,r=predict_batch(ids);synchronize();elapsed=time.perf_counter()-tick;p=np.asarray(p);r=np.asarray(r)
        if len(p)!=len(ids) or len(r)!=len(ids):raise ValueError('batch callback dropped frames')
        pp.append(p);rr.append(r);batches.append({'start_frame':begin,'frames':len(ids),'wall_seconds':elapsed,'amortized_seconds_per_frame':elapsed/len(ids)})
    wall=time.perf_counter()-start
    return PosePrediction(np.concatenate(pp),np.concatenate(rr)),{'wall_seconds':wall,'frames':frame_count,'batch_size':batch_size,
        'batches':batches,'amortized_seconds_per_frame':wall/frame_count,'execution':'batched independent single-frame calls; no temporal model; amortized latency'}
