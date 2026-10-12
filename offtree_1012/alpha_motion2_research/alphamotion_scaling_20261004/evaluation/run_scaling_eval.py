#!/usr/bin/env python3
"""Root-owned execution entrypoint for actual six-run manipulation evaluation.

Uses hash-validated real heldout episodes, sparse-only model initialization,
three kinematic/DOF stages and one matched default-q evaluation baseline. No
training or model updates. This script never starts itself in the background.
"""
from __future__ import annotations
import argparse
import dataclasses
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parent
for path in [PACKAGE,PACKAGE/'orchestration',PACKAGE/'base',PACKAGE/'base/legacy',HERE]:
    if str(path) not in sys.path:sys.path.insert(0,str(path))
from eval_engine import (TrajectoryCase,PosePrediction,evaluate_case,temporal_metrics,
                        map_independent_frame_batches,rotation_error_degrees,write_result,metric_summary)
from trajectory_refinement import (VisibleSE3,RefinementConfig,save_refinement,visible_se3_digest)
from dof_protocol import compare_learned_vs_default
from hinge_dof_adapter import SingleHingeTree,matrix_from_rot6,rot6_from_matrix,file_sha
from mixed_dof_adapter import MixedJointTree
from fk_adapter import global_rot6_decoder
from curve_statistics import summarize_cases,bootstrap_clips,evaluation_set_digest,refinement_curve
from surface_metrics import evaluate_fixed_surface
from plot_curves import plot_refinement

BODIES=('adroit_pen','relay_franka_full_v2')


def json_write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(path)


def select_episode_rows(body,*,clips=3,max_frames=64):
    """Fixed heldout group ordering, no quality-based filtering or frame resampling."""
    if type(clips) is not int or clips<3:raise ValueError('at least3 independent heldout source episodes per body required')
    if type(max_frames) is not int or not 1<=max_frames<=64:raise ValueError('declared subclip frame budget1..64')
    a=body.arrays;groups=np.asarray(a['group']);split=np.asarray(a['split']);frame=np.asarray(a['frame_index'])
    candidates=[]
    for gid in sorted(np.unique(groups).tolist()):
        rows=np.flatnonzero(groups==gid)
        if len(np.unique(split[rows]))!=1:raise ValueError('physical episode crosses split')
        if int(split[rows[0]]) not in (1,2):continue
        rows=rows[np.argsort(frame[rows],kind='stable')]
        if len(np.unique(frame[rows]))!=len(rows):raise ValueError('duplicate frame in episode')
        stride=int(body.metadata.get('training_stride',1))
        if len(rows)>1 and not np.all(np.diff(frame[rows])==stride):raise ValueError('loader episode is not a contiguous sequence at declared sampling stride')
        selected=rows[:max_frames]
        candidates.append({'group_index':int(gid),'rows':selected,'loaded_episode_frames':len(rows),
                           'selected_frames':len(selected),'sampling_stride':stride})
    if len(candidates)<clips:raise ValueError(f'{body.name} has only{len(candidates)} heldout episodes; requested{clips}')
    return candidates[:clips]


def select_reference_controls(body,tree,native_payload,*,clips=3,max_frames=64):
    """Keep original starts; add legal controls checked across ALL covered native q.

    This is explicit source-feasibility stratification, never output-based filtering.
    Original difficult starts remain in their own reported cohort.
    """
    originals=select_episode_rows(body,clips=clips,max_frames=max_frames)
    if file_sha(native_payload)!=body.metadata['payload_sha256']:raise ValueError('native q limit-audit payload hash mismatch')
    with np.load(native_payload,allow_pickle=False) as z:
        q=np.asarray(z['recorded_robot_qpos'],dtype=np.float64)[:,tree.q_addresses]
        groups=z['source_group_index'];frames=z['frame_index'];splits=z['split']
    lower=tree.lower.numpy();upper=tree.upper.numpy()
    violation=np.maximum(np.maximum(q-upper,lower-q),0.)
    if not np.isfinite(q).all():raise ValueError('native reference q must be finite')
    coordinate_units=getattr(tree,'units',('radian',)*len(tree.q_names))
    by={}
    for gid in sorted(np.unique(groups).tolist()):
        idx=np.flatnonzero(groups==gid);idx=idx[np.argsort(frames[idx],kind='stable')]
        if not np.array_equal(frames[idx],np.arange(len(idx))) or len(np.unique(splits[idx]))!=1:raise ValueError('native physical episode frame/split integrity')
        by[int(gid)]=idx
    def annotate(selection,cohort):
        loaded_frames=body.arrays['frame_index'][selection['rows']]
        native=by[selection['group_index']];covered=native[(frames[native]>=loaded_frames[0])&(frames[native]<=loaded_frames[-1])]
        v=violation[covered]
        qc={'native_frames_checked':len(covered),'sampled_frames':len(loaded_frames),'all_native_q_within_hard_limits':not bool((v>0).any()),
            'violating_native_frames':int((v>0).any(1).sum()),'per_coordinate_violating_native_frames':(v>0).sum(0).tolist(),
            'per_coordinate_max_overshoot':v.max(0).tolist(),'coordinate_names':list(tree.q_names),'coordinate_units':list(coordinate_units),
            'tolerance':0.,'reference_q_clipped':False,'scope':'Every original native frame between first and last sampled frame; no claim of collision/contact/dynamics feasibility'}
        if cohort=='original_reference_start':cohort+='__within_limits' if qc['all_native_q_within_hard_limits'] else '__has_soft_limit_violations'
        return {**selection,'reference_cohort':cohort,'native_reference_limit_qc':qc}
    selected=[annotate(v,'original_reference_start') for v in originals];controls=[];unavailable=[]
    for gid,native in by.items():
        if int(splits[native[0]]) not in (1,2):continue
        loaded=np.flatnonzero(body.arrays['group']==gid);loaded=loaded[np.argsort(body.arrays['frame_index'][loaded],kind='stable')]
        if not len(loaded):continue
        good=~(violation[native]>0).any(1);starts=np.flatnonzero(good&~np.r_[False,good[:-1]]);ends=np.flatnonzero(good&~np.r_[good[1:],False])
        candidates=[]
        for first,last in zip(starts,ends):
            ids=loaded[(body.arrays['frame_index'][loaded]>=first)&(body.arrays['frame_index'][loaded]<=last)]
            if len(ids)>=2:candidates.append(ids[:max_frames])
        if not candidates:unavailable.append({'group_index':gid,'reason':'no at least two sampled frames connected by an all-native-q legal interval'});continue
        # Prefer a full declared budget; otherwise longest legal interval, earliest tie.
        candidates.sort(key=lambda ids:(-len(ids),int(body.arrays['frame_index'][ids[0]])))
        pick={'group_index':gid,'rows':candidates[0],'loaded_episode_frames':len(loaded),'selected_frames':len(candidates[0]),
            'sampling_stride':int(body.metadata.get('training_stride',1)),
            'selection_reason':'longest up to declared budget, earliest-tied source-only legal interval; all native q between sampled frames checked'}
        controls.append(annotate(pick,'strict_limit_valid_control'))
    controls=controls[:clips];selected.extend(controls)
    report={'original_start_cases_retained':len(originals),'strict_limit_valid_independent_clips':len(controls),'requested_strict_clips':clips,
        'strict_control_CI_status':'eligible' if len(controls)>=3 else 'unavailable_fewer_than_three_independent_clips',
        'unavailable_groups':unavailable,'reference_limit_source':str(native_payload),'payload_sha256':body.metadata['payload_sha256'],
        'selection_uses_model_predictions':False,'q_ground_truth_optimizer_loss':False,'GT_clipped':False,
        'strict_definition':'Known per-coordinate hard limits only; every covered native frame. Collision, contact, velocity/torque and physical execution not certified.'}
    return selected,report


def endpoint_indices(body,count=2):
    if count not in (1,2):raise ValueError('only one/two actual declared EE conditions')
    legal=[int(i) for i in body.endpoint['indices']]
    if len(set(legal))!=len(legal) or not legal or any(i<0 or i>=len(body.names) for i in legal):raise ValueError('real endpoint metadata missing/invalid')
    # Actual source names, accepted only if present in the source endpoint whitelist.
    preferred=[body.names.index(n) for n in ('thdistal','ffdistal','panda0_leftfinger','panda0_rightfinger','end_effector_site') if n in body.names and body.names.index(n) in legal]
    ordered=preferred+[i for i in legal if i not in preferred]
    return ordered[:count]


def make_case_and_condition(body,selection,device,condition_count=2):
    rows=selection['rows'];a=body.arrays;T=len(rows);J=len(body.names)
    gt6=torch.as_tensor(a['global_rot6'][rows],device=device,dtype=torch.float32)
    gtR=matrix_from_rot6(gt6)
    p=torch.as_tensor(a['pose9'][rows,:,6:],device=device,dtype=torch.float32)*body.reach
    ids=endpoint_indices(body,condition_count);ix=torch.tensor(ids,device=device,dtype=torch.long)[None].expand(T,-1)
    c=VisibleSE3(ix,p[:,ids],gtR[:,ids],torch.ones_like(ix,dtype=torch.bool),torch.ones_like(ix,dtype=torch.bool),
        tuple(body.names),'actual heldout recorded simulation FK; selected named endpoints only; payload SHA '+body.metadata['payload_sha256'],'cm')
    c.validate(T,J)
    episodes=body.metadata.get('episodes',[])
    groups=sorted({e['source_group'] for e in episodes})
    gid=selection['group_index']
    group_name=groups[gid] if gid<len(groups) else body.name+':source_group_index:'+str(gid)
    ep=next((e for e in episodes if e['source_group']==group_name),None)
    native_frames=int(ep['frames']) if ep is not None else selection['loaded_episode_frames']*selection['sampling_stride']
    frame=np.asarray(a['frame_index'][rows],dtype=np.int64);timestamps=np.asarray(a['timestamp_seconds'][rows],dtype=np.float64)
    complete=selection['sampling_stride']==1 and len(rows)==native_frames and np.array_equal(frame,np.arange(native_frames))
    cohort=selection.get('reference_cohort','original_reference_start_unchecked')
    case_id=body.name+'__'+hashlib.sha256(group_name.encode()).hexdigest()[:16]+'__'+cohort+'__f'+str(int(frame[0]))
    case=TrajectoryCase(case_id,group_name,body.name,body.name,'completion',
        'seen_body' if len(body.eligible(True)) else 'heldout_body','heldout_clip',frame,timestamps,list(body.names),True,float(body.reach),
        p.detach().cpu().numpy(),gtR.detach().cpu().numpy(),complete_clip=complete,source_frame_count=native_frames,
        reference_kind='same_body',role_endpoints={'declared_SE3_endpoints':ids},
        provenance={'dataset':body.metadata.get('actual_dataset',body.metadata.get('source_dataset')),
            'payload_sha256':body.metadata['payload_sha256'],'metadata_sha256':body.metadata.get('metadata_sha256'),
            'group_index':gid,'native_source_frames':native_frames,'loaded_episode_frames':selection['loaded_episode_frames'],
            'evaluation_is_subclip':not complete,'selection':selection.get('selection_reason','first consecutive loader frames, no motion-quality selection'),
            'reference_cohort':cohort,'native_reference_limit_qc':selection.get('native_reference_limit_qc',{'status':'not_checked'}),
            'native_frame_stride':selection['sampling_stride'],'time_source':body.metadata.get('time_source'),
            'actual_length_unit':'cm after explicit native metre*100 conversion',
            'reference_scope':'same-body recorded simulation target; not paired cross-body mapping or unique fullbody IK solution',
            'world_metrics':'not evaluated; canonical root relative only, original heading retained',
            'physical_robot_execution':False,'object_pose_contact_or_grasp_success':'not available/not inferred',
            'source_soft_limit_violations':body.metadata.get('source_soft_limit_violations',{'status':'not supplied'})},
        condition_set_sha256=visible_se3_digest(c),target_morphology='dexterous_hand' if body.name=='adroit_pen' else 'robot_arm')
    case.validate();return case,c,gt6,rows


def sparse_model_initialization(model,body,condition,batch_size):
    """Only named visible SE3 is supplied; no fullbody pose or recorded q input."""
    T=len(condition.joint_indices);device=next(model.parameters()).device;J=len(body.names)
    packed=[];raw_aux=[]
    def predict(ids):
        ix=torch.as_tensor(ids,device=device,dtype=torch.long);B=len(ids)
        observed=condition.joint_indices[ix];valid=condition.position_valid[ix]&condition.rotation_valid[ix]
        values=torch.cat([rot6_from_matrix(condition.rotations[ix]),condition.positions[ix]/body.reach],-1)
        with torch.inference_mode():
            output=model.forward_kinematic(torch.zeros((B,J,9),device=device),body.geom(B),body.geom(B),
                source_hidden=torch.ones(B,J,device=device,dtype=torch.bool),source_contact=None,
                source_slide_normalized=None,condition={'joint_indices':observed,'values':values,'valid':valid})
            if not isinstance(output,dict) or not all(k in output for k in ('rot6','slide_native','joint_position','joint_rotation','raw_hpos')):
                raise ValueError('explicit A/B mixed-kinematic output contract required')
            rotation=output['rot6'].float();slide=output['slide_native'].float()
            if rotation.shape!=(B,J,6) or slide.shape!=(B,J):raise ValueError('predicted mixed posture shape')
            packed.append(torch.cat([rotation,slide[...,None]],-1).detach());raw_aux.append(output['raw_hpos'].float().detach())
        return output['joint_position'].float().cpu().numpy(),output['joint_rotation'].float().cpu().numpy()
    sync=(lambda:torch.cuda.synchronize(device)) if device.type=='cuda' else (lambda:None)
    pred,timing=map_independent_frame_batches(T,batch_size,predict,synchronize=sync)
    timing.update(source_pose='constant zeros; all joints hidden',source_contact=None,source_slide_normalized=None,
        observed_condition_sha256=visible_se3_digest(condition),temporal_context=False,
        slide_initialization='actual model prediction only; no recorded q or full GT supplied')
    return torch.cat(packed),pred,timing,torch.cat(raw_aux)


def target_frame_metrics(p,R,condition,config):
    p=p.detach();R=R.detach();T=len(p);batch=torch.arange(T,device=p.device)[:,None]
    ix=condition.joint_indices
    pe=(p[batch,ix]-condition.positions).norm(dim=-1)
    re=torch.as_tensor(rotation_error_degrees(R[batch,ix].cpu().numpy(),condition.rotations.cpu().numpy()),device=p.device)
    goodpos=(pe<=config.position_tolerance)|~condition.position_valid
    goodrot=(re<=config.rotation_tolerance_deg)|~condition.rotation_valid
    known=(condition.position_valid|condition.rotation_valid).any(1)
    met=goodpos.all(1)&goodrot.all(1)&known
    return {'position':metric_summary(pe[condition.position_valid].cpu().numpy(),'cm'),
            'rotation':metric_summary(re[condition.rotation_valid].cpu().numpy(),'degree'),
            'target_frame_fraction':float(met.float().mean()),'target_frames':T,'frames_with_all_visible_SE3_tolerances_met':int(met.sum()),
            'position_mean_cm':float(pe[condition.position_valid].mean()) if bool(condition.position_valid.any()) else None,
            'rotation_mean_deg':float(re[condition.rotation_valid].mean()) if bool(condition.rotation_valid.any()) else None,
            'position_tolerance_cm':config.position_tolerance,'rotation_tolerance_deg':config.rotation_tolerance_deg,
            'note':'SE3 tolerances only; executable-q and collision validity reported separately'}


def make_observer(case,body,condition,config,mixed_tree=None):
    """Only detached evaluation reads fullbody GT; never an optimizer objective."""
    def observe(step,p,R):
        result=evaluate_case(case,PosePrediction(p.cpu().numpy(),R.cpu().numpy()))
        # A sampled contiguous subclip still has true timestamp-based variation;
        # explicitly report its sampled rate, never alias this to native500Hz jitter.
        result['sampled_subclip_temporal']=temporal_metrics(p.cpu().numpy(),case.timestamps_seconds,'cm')
        result['sampled_subclip_temporal']['frame_selection']=case.provenance['selection']
        result['sampled_subclip_temporal']['source_stride']=case.provenance['native_frame_stride']
        result['selected_SE3']=target_frame_metrics(p,R,condition,config)
        if mixed_tree is not None and mixed_tree.slide_bodies:
            from eval_engine import metric_summary
            predicted=mixed_tree.aperture_cm(p,R)
            truth=mixed_tree.aperture_cm(torch.as_tensor(case.reference_positions,device=p.device,dtype=p.dtype),
                torch.as_tensor(case.reference_rotations,device=p.device,dtype=R.dtype))
            result['aperture']={'absolute_error':metric_summary((predicted-truth).abs().cpu().numpy(),'cm'),
                'definition':mixed_tree.document['aperture_definition'],'contact_pad_clearance':False,
                'predicted_origin_opening_cm':predicted.cpu().tolist(),'reference_origin_opening_cm':truth.cpu().tolist()}
        return result
    return observe


def _curve_npz(path,result):
    rows=result['records'];keys={'step':[r['step'] for r in rows],'elapsed_seconds':[r['elapsed_seconds'] for r in rows],
        'selected_position_cm':[r.get('metrics',{}).get('selected_position_mean',np.nan) for r in rows],
        'selected_rotation_degrees':[r.get('metrics',{}).get('selected_rotation_mean_deg',np.nan) for r in rows],
        'target_frame_fraction':[r.get('observation',{}).get('selected_SE3',{}).get('target_frame_fraction',np.nan) if isinstance(r.get('observation'),dict) else np.nan for r in rows],
        'fullbody_position_cm':[r.get('observation',{}).get('metrics',{}).get('position',{}).get('mean',np.nan) if isinstance(r.get('observation'),dict) else np.nan for r in rows]}
    np.savez_compressed(path,**{k:np.asarray(v,dtype=float) for k,v in keys.items()})


def final_observation(result):
    if result.get('failure_reason'):return None
    records=result.get('records',[])
    return records[-1].get('observation') if records and records[-1].get('status')=='measured' else None


def add_timing_accounting(protocol,timing):
    """Include required initial free-to-valid-q projection, with its instrumentation."""
    control=protocol.setdefault('comparison',{})
    projected=protocol['learned']['dof_projected'];records=projected.get('records',[])
    initial=records[0].get('projection_seconds') if records and records[0].get('step')==0 else None
    if not isinstance(initial,(int,float)) or not math.isfinite(initial) or initial<0:initial=None
    learned_threshold=control.get('learned_projected_initialization',{}).get('seconds_to_threshold')
    control['measured_inference_wall_seconds']=timing['wall_seconds']
    control['initial_DOF_projection_seconds']=initial
    control['learned_inference_plus_constrained_seconds_to_threshold']=(None if learned_threshold is None or initial is None
        else timing['wall_seconds']+initial+learned_threshold)
    control['default_seconds_to_threshold']=control.get('default_initialization',{}).get('seconds_to_threshold')
    control['default_requires_model_inference_or_initial_DOF_projection']=False
    control['timing_scope']='Learned total = synchronized model inference + initial free-to-valid-q projection_seconds + constrained elapsed to threshold. The initial projection measurement includes FK, feasibility checks and its diagnostic observer, so this is conservative instrumented overhead. Later raw/projection samples are diagnostics and are excluded from this learned-path total. Default starts from valid q and needs neither model inference nor initial free-to-q projection.'
    raw=protocol['learned']['raw_free_posture']
    raw.setdefault('protocol',{})['elapsed_seconds_scope']='Instrumented wall time includes free-state optimization, evaluation diagnostics, and every DOF-projection observer; not isolated optimizer/kernel time.'
    projected.setdefault('protocol',{})['elapsed_seconds_scope']='Instrumented wall time since the raw/projection diagnostic pass began; includes raw optimization, raw diagnostics, projection and projection observers. No independent projected optimizer or pure projection-kernel latency is claimed.'
    projected['protocol']['projection_seconds_scope']='This sample projection, FK, constraint metrics and detached diagnostic observer; instrumented overhead.'
    return control


def save_case(directory,case,condition,initial_rot6,gt6,rows,protocol,timing,initial_eval,surface_eval):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(directory/'case_arrays.npz',frame_ids=case.frame_ids,timestamps_seconds=case.timestamps_seconds,
        loader_rows=rows,initial_model_global_rot6=initial_rot6[...,:6].detach().cpu().numpy(),initial_model_slide_cm=initial_rot6[...,6].detach().cpu().numpy(),reference_global_rot6=gt6.detach().cpu().numpy(),
        reference_position_cm=case.reference_positions,condition_joint_indices=condition.joint_indices.cpu().numpy(),
        condition_position_cm=condition.positions.cpu().numpy(),condition_global_rotation=condition.rotations.cpu().numpy(),
        condition_position_valid=condition.position_valid.cpu().numpy(),condition_rotation_valid=condition.rotation_valid.cpu().numpy())
    add_timing_accounting(protocol,timing)
    learned=protocol['learned'];stages={'raw_free_posture':learned['raw_free_posture'],
        'dof_projected':learned['dof_projected'],'constrained_refinement':learned['constrained_refinement'],
        'default_valid_q':protocol['default_valid_q_baseline']}
    curves={};summary={}
    for name,result in stages.items():
        stage=directory/name;stage.mkdir(exist_ok=True)
        if result.get('status')=='unsupported':json_write(stage/'status.json',result);summary[name]=result;continue
        if 'final_state' in result:save_refinement(stage,result)
        else:json_write(stage/'refinement_curve.json',result)
        _curve_npz(stage/'curve_metrics.npz',result);curves[name]=result
        summary[name]={'status':result.get('status'),'steps':result.get('requested_steps'),
            'final_observation':final_observation(result),'protocol':result.get('protocol'),
            'final_constraints':result['records'][-1].get('constraints') if result.get('records') else None}
    if curves:
        plot_refinement(curves,directory/'SE3_position_vs_steps_and_time.png','selected_position_mean')
        plot_refinement(curves,directory/'SE3_rotation_vs_steps_and_time.png','selected_rotation_mean_deg')
    metadata={'case_id':case.case_id,'clip_id':case.clip_id,'body':case.target_body,'task_kind':case.task_kind,
        'body_exposure':case.body_exposure,'clip_exposure':case.clip_exposure,'target_morphology':case.target_morphology,
        'complete_clip':case.complete_clip,'frames':len(case.frame_ids),'source_frame_count':case.source_frame_count,
        'provenance':case.provenance,'condition_sha256':visible_se3_digest(condition),
        'declared_joint_indices':condition.joint_indices[0].cpu().tolist(),
        'declared_joint_names':[case.joint_names[i] for i in condition.joint_indices[0].cpu().tolist()],
        'initial_inference':timing,'no_refine_initial_model':initial_eval,'fixed_patch_surface':surface_eval,
        'stages':summary,'default_control_comparison':protocol.get('comparison'),
        'claim_boundary':'Same-body simulation sparse completion. Free posture is not executable. Known q limits are tested only under validated actual adapter. No world/physical/contact/grasp-success claim.'}
    json_write(directory/'case_result.json',metadata);return metadata


def failure_result(case,message):
    shape=(len(case.frame_ids),len(case.joint_names))
    result=evaluate_case(case,PosePrediction(np.full((*shape,3),np.nan),np.full((*shape,3,3),np.nan),failure_reason=message))
    result['selected_SE3']={'target_frame_fraction':0.,'target_frames':len(case.frame_ids),'failure_reason':message,
        'position':metric_summary(np.full(len(case.frame_ids),np.nan),'cm'),'rotation':metric_summary(np.full(len(case.frame_ids),np.nan),'degree')}
    return result


def reference_slides_cm(body,rows,geom,device):
    """Loader-native GT slides for detached surface evaluation only; never clamp."""
    T=len(rows);J=len(body.names)
    typ=torch.as_tensor(geom.get('joint_type',np.zeros(J,dtype=np.int64)),device=device)
    if typ.shape==(J,):typ=typ[None].expand(T,-1)
    if typ.shape!=(T,J):raise ValueError('typed geometry shape for reference slides')
    active=typ==3
    if not bool(active.any()):return torch.zeros(T,J,device=device,dtype=torch.float32)
    if body.metadata.get('length_unit')!='cm':raise ValueError('reference slide_native needs explicit loader centimetre units')
    if 'slide_native' not in body.arrays:raise ValueError('loader slide_native(cm) required for surface observation; no GT field/unit inference')
    values=torch.as_tensor(body.arrays['slide_native'][rows],device=device,dtype=torch.float32)
    if values.shape!=(T,J) or not bool(torch.isfinite(values[active]).all()):raise ValueError('finite active reference slide_native shape')
    return torch.where(active,values,torch.zeros_like(values))


def validate_loader_adapter(body,tree):
    """Check the loader/validated q adapter boundary before any model evaluation."""
    if tuple(body.names)!=tree.names or not np.array_equal(body.parents,tree.parents):raise ValueError('loader/DOF adapter body names or parents differ')
    if body.metadata.get('payload_sha256')!=tree.document.get('payload_sha256') or body.metadata.get('metadata_sha256')!=file_sha(tree.metadata_path):raise ValueError('loader/DOF adapter payload or metadata SHA differs')
    if not body.metric or body.metadata.get('length_unit')!='cm' or body.metadata.get('source_native_unit')!='m':raise ValueError('external loader must explicitly convert native metres to centimetres')
    if not np.isfinite(body.reach) or body.reach<=0 or np.asarray(body.offsets).shape!=tuple(tree.offsets.shape) or not np.allclose(body.offsets,tree.offsets.numpy(),atol=1e-4,rtol=1e-6):raise ValueError('loader/DOF adapter physical offsets or reach differ')
    geom=body.geom(1);root=torch.as_tensor(geom.get('root_fixed',False),dtype=torch.bool)
    if root.numel()!=1 or not bool(root.all()):raise ValueError('source-verified fixed root must be explicit in loader geometry')
    J=len(body.names)
    def unbatch(key,shape):
        if key not in geom:raise ValueError('missing typed loader geometry '+key)
        value=torch.as_tensor(geom[key]).detach().cpu()
        if value.shape==(1,*shape):value=value[0]
        if tuple(value.shape)!=tuple(shape):raise ValueError('typed loader geometry shape '+key)
        return value
    off=unbatch('kinematic_rest_off',(J,3))
    if not torch.allclose(off.double(),tree.offsets,atol=1e-4,rtol=1e-6):raise ValueError('loader FK offsets differ from validated q tree')
    slide_bodies=getattr(tree,'slide_bodies',())
    if slide_bodies:
        typ=unbatch('joint_type',(J,)).long()
        expected=torch.ones(J,dtype=torch.long);expected[0]=4
        for j,slot in tree.body_slot.items():expected[j]=3 if tree.kinds[slot]=='slide' else 2
        if not torch.equal(typ,expected):raise ValueError('typed loader joint kind differs from actual fixed/hinge/slide map')
        axis=unbatch('prismatic_axis',(J,3)).double()
        lower=unbatch('slide_lower',(J,)).double();upper=unbatch('slide_upper',(J,)).double();reference=unbatch('slide_reference',(J,)).double()
        rotation_key='kinematic_fixed_rotation' if 'kinematic_fixed_rotation' in geom else 'prismatic_fixed_rotation'
        fixed=unbatch(rotation_key,(J,3,3)).double()
        for j in slide_bodies:
            slot=tree.body_slot[j]
            if not torch.allclose(axis[j],tree.axis[slot],atol=2e-6,rtol=1e-6):raise ValueError('loader prismatic rest axis differs from validated q tree')
            if not torch.allclose(torch.stack([lower[j],upper[j],reference[j]]),torch.stack([tree.lower[slot],tree.upper[slot],tree.q0[slot]])*100.,atol=2e-6,rtol=1e-6):raise ValueError('loader prismatic limits/reference must be actual metres converted to cm')
        for j in set(tree.fixed)|set(slide_bodies):
            if not torch.allclose(fixed[j],torch.eye(3,dtype=torch.float64),atol=1e-6,rtol=0):raise ValueError('loader fixed/prismatic rebase differs from validated identity inheritance')
        reference_slides_cm(body,np.array([0]),geom,torch.device('cpu'))
    elif 'joint_type' in geom and bool((torch.as_tensor(geom['joint_type'])==3).any()):raise ValueError('loader slide has no validated actual q mapping')
    return {'status':'matched','metadata_sha256':body.metadata['metadata_sha256'],'payload_sha256':body.metadata['payload_sha256'],
        'length_unit':'cm','source_native_unit':'m','root_fixed':True,'joint_count':J,'q_coordinates':len(tree.q_names),'slide_body_indices':list(slide_bodies)}


def evaluate_body_episodes(model,body,adapter,*,output,clips=3,max_frames=64,condition_count=2,batch_size=32,
                           config=RefinementConfig(),selected=None,mixed_tree=None):
    device=next(model.parameters()).device
    selected=select_episode_rows(body,clips=clips,max_frames=max_frames) if selected is None else selected
    model.eval();results=[];cases=[]
    from prismatic.kinematics import mixed_forward_kinematics
    def raw_decode(state):
        value=mixed_forward_kinematics(state[...,:6],body.geom(len(state)),state[...,6])
        return value['joint_position'],value['joint_rotation']
    for selection in selected:
        case,condition,gt6,rows=make_case_and_condition(body,selection,device,condition_count);cases.append(case)
        directory=Path(output)/case.case_id
        try:
            initial,pred,timing,raw_aux=sparse_model_initialization(model,body,condition,batch_size)
            if not torch.isfinite(initial).all():raise FloatingPointError('nonfinite model initialization')
            observer=make_observer(case,body,condition,config,mixed_tree=mixed_tree)
            no_refine=observer(0,torch.as_tensor(pred.positions,device=device),torch.as_tensor(pred.rotations,device=device))
            no_refine['timing']=timing
            protocol=compare_learned_vs_default(initial,raw_decode,condition,adapter=adapter,config=config,observer=observer)
            surface={'status':'NA','reason':'no verified mesh/binding'}
            g=body.geom(len(rows))
            if g.get('surface') is not None:
                from surface.surface_geometry import predict_bound_surface
                reference_slide=reference_slides_cm(body,rows,g,device)
                surface=evaluate_fixed_surface(initial[...,:6],gt6,g,bind_fn=predict_bound_surface,
                    expected_patch_sha256=g['surface']['asset_sha256'],predicted_slide_native=initial[...,6],reference_slide_native=reference_slide)
            saved=save_case(directory,case,condition,initial,gt6,rows,protocol,timing,no_refine,surface)
            results.append(saved)
        except Exception as exc:
            directory.mkdir(parents=True,exist_ok=True)
            fail=failure_result(case,repr(exc))
            value={'case_id':case.case_id,'clip_id':case.clip_id,'body':body.name,'status':'failed_case_retained',
                'error':repr(exc),'traceback':traceback.format_exc(),'no_refine_initial_model':fail,
                'stages':{name:{'status':'failed_case_retained','final_observation':fail} for name in ['raw_free_posture','dof_projected','constrained_refinement','default_valid_q']}}
            json_write(directory/'case_result.json',value);results.append(value)
    return cases,results


def summarize_checkpoint(cases,results,*,bootstrap_samples=1000):
    cohorts=sorted({c.provenance.get('reference_cohort','unspecified') for c in cases})
    if len(cohorts)>1:
        subsets={}
        for cohort in cohorts:
            subset=[c for c in cases if c.provenance.get('reference_cohort','unspecified')==cohort];ids={c.case_id for c in subset}
            subsets[cohort]=summarize_checkpoint(subset,[r for r in results if r['case_id'] in ids],bootstrap_samples=bootstrap_samples)
        return {'evaluation_set_sha256':evaluation_set_digest(cases),'case_count':len(cases),'real_source_clip_count':len({c.clip_id for c in cases}),
            'all_cases_complete_original_clip':all(c.complete_clip for c in cases),'reference_cohorts':subsets,'metrics_pooled_across_reference_cohorts':False,
            'claim_boundary':'Original starts with reference soft-limit violations and strictly limit-valid controls are separate. Same physical episode may appear in multiple cohorts; they are not independent samples across cohorts.'}
    stages={name:[] for name in ['no_refine','raw_free_posture','dof_projected','constrained_refinement','default_valid_q']}
    for r in results:
        stages['no_refine'].append(r['no_refine_initial_model'])
        for name in list(stages)[1:]:
            obs=r['stages'][name].get('final_observation')
            if obs is None:
                case=next(c for c in cases if c.case_id==r['case_id'])
                obs=failure_result(case,'stage unavailable or failed; no case dropped')
            stages[name].append(obs)
    report={'evaluation_set_sha256':evaluation_set_digest(cases),'case_count':len(cases),
            'real_source_clip_count':len({c.clip_id for c in cases}),'all_cases_complete_original_clip':all(c.complete_clip for c in cases),
            'sampling':'fixed actual source episode subclips; bootstrap cluster is physical episode, not frame','reference_cohort':cohorts[0],'stages':{}}
    for name,rows in stages.items():
        strata={}
        for r in rows:
            key=(r['task_kind'],r['target_morphology'],r['body_exposure'],r['clip_exposure'],r['alignment'])
            strata.setdefault(key,[]).append(r)
        limit_known=name in ('dof_projected','constrained_refinement','default_valid_q')
        report['stages'][name]={'feasibility_scope':('actual joint-coordinate limits clamped; selected SE3 tolerances evaluated; collision/contact unknown' if limit_known else 'free posture; true joint feasibility not certified'),
            'per_body_known_limits_plus_target_frame_fraction':({b:float(np.mean([r.get('selected_SE3',{}).get('target_frame_fraction',0.) for r in rows if r['target_body']==b])) for b in sorted({r['target_body'] for r in rows})} if limit_known else {'status':'unsupported'}),
            'penetration_rate':{'status':'unsupported','reason':'no validated collision geometry/environment query supplied'},
            'selected_SE3_position_cm':summarize_cases(rows,'selected_SE3.position'),
            'selected_SE3_rotation_degrees':summarize_cases(rows,'selected_SE3.rotation'),
            'selected_SE3_bootstrap':[{'stratum':list(k),'position':bootstrap_clips(v,'selected_SE3.position',samples=bootstrap_samples,minimum_clips=3),
                'rotation':bootstrap_clips(v,'selected_SE3.rotation',samples=bootstrap_samples,minimum_clips=3)} for k,v in strata.items()],
            'aperture_error_cm':summarize_cases([r for r in rows if 'aperture' in r],'aperture.absolute_error') if any('aperture' in r for r in rows) else {'status':'NA'},
            'fullbody_position_diagnostic':summarize_cases(rows,'metrics.position'),
            'fullbody_rotation_diagnostic':summarize_cases(rows,'metrics.rotation'),
            'fullbody_normalized_position_diagnostic':summarize_cases(rows),
            'bootstrap_by_task_morphology_exposure':[{'stratum':list(k),**bootstrap_clips(v,samples=bootstrap_samples,minimum_clips=3)} for k,v in strata.items()],
            'per_body_selected_target_frame_fraction':{b:float(np.mean([r.get('selected_SE3',{}).get('target_frame_fraction',0.) for r in rows if r['target_body']==b])) for b in sorted({r['target_body'] for r in rows})}}
    report['claim_boundary']='Fullbody recorded-posture error is a same-body diagnostic, not uniqueness of IK. Primary manipulation evidence is legal selected-SE3 residual and declared q/contact feasibility. No physical or complete-IK superiority claim.'
    return report


def model_variant(config,override=None):
    if override:
        if config.get('group') is not None and config['group']!=override:raise ValueError('explicit variant disagrees with checkpoint group')
        return override
    value=config.get('group',config.get('variant',config.get('architecture',config.get('model_family'))))
    if value in ('A','B'):return value
    raise ValueError('checkpoint needs explicit A/B variant or --variant; do not infer from filenames')


def model_factory(config,variant):
    if variant=='A':
        from pilot_model import ConditionalMegatron
        return ConditionalMegatron(**config['model'])
    if variant=='B':
        from surface.surface_model import SurfaceMegatron
        return SurfaceMegatron(max_patches=96,**config['model'])
    raise ValueError('A/B factory')


def checkpoint_step_from_name(path):
    import re
    match=re.fullmatch(r'checkpoint_step(\d+)\.pt',Path(path).name)
    return int(match.group(1)) if match else None


def discover_checkpoints(run,selection='final'):
    """Prefer immutable weight nodes; never append an optimizer/RNG final duplicate."""
    if selection not in ('final','all'):raise ValueError('checkpoint selection must be final/all')
    run=Path(run);by_step={}
    for path in sorted(run.glob('checkpoint_step*.pt'),key=lambda p:p.name):
        step=checkpoint_step_from_name(path)
        if step is None or not path.is_file():continue
        if step in by_step:
            if file_sha(path)!=file_sha(by_step[step]):raise ValueError(f'conflicting checkpoint files for step {step}: {by_step[step].name}, {path.name}')
            # Same exact bytes, one evaluation. Prefer the writer canonical format.
            if path.name==f'checkpoint_step{step:04d}.pt':by_step[step]=path
        else:by_step[step]=path
    candidates=[by_step[k] for k in sorted(by_step)]
    if candidates:return candidates if selection=='all' else [candidates[-1]]
    finals=[p for p in (run/'checkpoint_final.pt',run/'final_checkpoint.pt',run/'checkpoint.pt') if p.is_file()]
    if finals:return [finals[0]]
    raise FileNotFoundError('no completed checkpoint_stepNNNN.pt or final checkpoint in run')


def load_evaluation_checkpoint(path):
    """Use PyTorch restricted loading exclusively; never unpickle arbitrary RNG objects."""
    path=Path(path)
    try:payload=torch.load(path,map_location='cpu',weights_only=True)
    except Exception as exc:
        raise RuntimeError(f'Restricted weights-only checkpoint loading failed for {path.name}. Use an existing checkpoint_stepNNNN.pt weights node, or export a model/config/step-only checkpoint in the trusted training environment. No unsafe pickle fallback was attempted.') from exc
    if not isinstance(payload,dict) or not isinstance(payload.get('model'),dict):raise ValueError('checkpoint requires a model state dictionary')
    step=payload.get('step')
    if type(step) is not int or step<0:raise ValueError('checkpoint requires explicit nonnegative integer step')
    named=checkpoint_step_from_name(path)
    if named is not None and named!=step:raise ValueError('checkpoint filename step disagrees with payload step')
    if not all(isinstance(k,str) and isinstance(v,torch.Tensor) for k,v in payload['model'].items()):raise ValueError('model state must contain only named tensors')
    config=payload.get('config')
    if config is not None and (not isinstance(config,dict) or not isinstance(config.get('model'),dict)):raise ValueError('checkpoint model configuration contract')
    # Drop optimizer/RNG immediately; these are not used or restored for evaluation.
    return {'model':payload['model'],'step':step,'config':config,'checkpoint_load_policy':'torch.load weights_only=True; optimizer/RNG discarded'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,action='append');p.add_argument('--checkpoints',choices=['final','all'],default='final')
    p.add_argument('--variant',choices=['A','B']);p.add_argument('--device',default='cuda')
    p.add_argument('--adroit-validation',type=Path,required=True);p.add_argument('--franka-validation',type=Path,required=True)
    p.add_argument('--clips-per-body',type=int,default=3);p.add_argument('--max-frames',type=int,default=64)
    p.add_argument('--condition-count',type=int,choices=[1,2],default=2);p.add_argument('--inference-batch-size',type=int,default=32)
    p.add_argument('--refinement-steps',type=int,default=100);p.add_argument('--refinement-lr',type=float,default=.01)
    p.add_argument('--position-tolerance-cm',type=float,default=1.);p.add_argument('--rotation-tolerance-deg',type=float,default=5.)
    p.add_argument('--bootstrap-samples',type=int,default=1000);args=p.parse_args()
    torch.set_num_threads(1);device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('requested CUDA unavailable; no silent fallback')
    refs={'adroit_pen':args.adroit_validation,'relay_franka_full_v2':args.franka_validation}
    adroit_tree=SingleHingeTree(args.root/'data/payloads/adroit_pen.json')
    adroit=adroit_tree.verified_adapter(args.adroit_validation)
    original_project=adroit.project_free
    adroit=dataclasses.replace(adroit,project_free=lambda state:original_project(state[...,:6]),
        projection_method=adroit.projection_method+'; shared packed state discards nonprismatic last coordinate')
    mixed=MixedJointTree(args.root/'data/payloads/relay_franka_full_v2.json')
    adapters={'adroit_pen':adroit,'relay_franka_full_v2':mixed.verified_adapter(args.franka_validation)}
    checkpoints=args.checkpoint or discover_checkpoints(args.run,args.checkpoints)
    rcfg=RefinementConfig(steps=args.refinement_steps,learning_rate=args.refinement_lr,position_tolerance=args.position_tolerance_cm,
        rotation_tolerance_deg=args.rotation_tolerance_deg,state_representation='packed own-joint global rot6 plus predicted slide cm through shared typed FK')
    rcfg.validate();args.output.mkdir(parents=True,exist_ok=True);summaries=[];fixed_hash=None;seen_steps={}
    for path in checkpoints:
        payload=load_evaluation_checkpoint(path)
        if payload['step'] in seen_steps:
            if file_sha(path)==seen_steps[payload['step']]:continue
            raise ValueError('explicit checkpoint list has conflicting duplicate steps')
        seen_steps[payload['step']]=file_sha(path)
        config=payload.get('config') or json.loads((args.run/'config.json').read_text());variant=model_variant(config,args.variant)
        model=model_factory(config,variant);model.load_state_dict(payload['model'],strict=True);model.to(device).eval()
        from scaling_data import load_scaling_bodies
        bodies,audit=load_scaling_bodies(args.root,config,device);by={b.name:b for b in bodies}
        missing=set(BODIES)-set(by)
        if missing:raise ValueError('missing actual manipulation bodies '+repr(missing))
        out=args.output/path.stem;out.mkdir(parents=True,exist_ok=True);all_cases=[];all_results=[];selection_audits={};adapter_audits={};start=time.perf_counter()
        for name in BODIES:
            tree=mixed if name=='relay_franka_full_v2' else adroit_tree
            body=by[name]
            adapter_audits[name]=validate_loader_adapter(body,tree)
            selected,selection_audits[name]=select_reference_controls(body,tree,args.root/'data/payloads'/f'{name}.npz',clips=args.clips_per_body,max_frames=args.max_frames)
            cases,results=evaluate_body_episodes(model,by[name],adapters[name],output=out/name,clips=args.clips_per_body,
                max_frames=args.max_frames,condition_count=args.condition_count,batch_size=args.inference_batch_size,config=rcfg,mixed_tree=mixed if name=='relay_franka_full_v2' else None,selected=selected)
            all_cases.extend(cases);all_results.extend(results)
        report=summarize_checkpoint(all_cases,all_results,bootstrap_samples=args.bootstrap_samples)
        if fixed_hash is not None and fixed_hash!=report['evaluation_set_sha256']:raise ValueError('checkpoint evaluation set changed')
        fixed_hash=report['evaluation_set_sha256']
        report.update(checkpoint=str(path.resolve()),checkpoint_sha256=file_sha(path),checkpoint_step=payload.get('step'),variant=variant,
            wall_seconds=time.perf_counter()-start,refinement_config=dataclasses.asdict(rcfg),device=str(device),
            validation_receipts={name:{'path':str(refs[name]),'sha256':file_sha(refs[name])} for name in BODIES},
            failed_case_ids=[r['case_id'] for r in all_results if r.get('status')=='failed_case_retained'],
            data_audit=audit,loader_adapter_audits=adapter_audits,checkpoint_load_policy=payload['checkpoint_load_policy'],reference_selection_audits=selection_audits,training_updates=0,fullbody_GT_used_by_optimizer=False,
            evaluation_source_sha256={p.name:file_sha(p) for p in sorted(HERE.glob('*.py'))},
            raw_output_auxiliary_head_used_as_XYZ=False,reference_soft_limit_excursions='preserved; may create nonzero error floor under hard-bound predictions')
        json_write(out/'evaluation_summary.json',report);summaries.append({'path':str(out/'evaluation_summary.json'),'checkpoint':path.name,'failed_cases':len(report['failed_case_ids'])})
        del model,bodies,payload
        if device.type=='cuda':torch.cuda.empty_cache()
    json_write(args.output/'evaluation_index.json',{'evaluations':summaries,'fixed_evaluation_set_sha256':fixed_hash,'actual_execution':True,'training_updates':0})
    print(json.dumps({'output':str(args.output),'evaluations':summaries}))
    if any(s['failed_cases'] for s in summaries):raise SystemExit(1)


if __name__=='__main__':main()
