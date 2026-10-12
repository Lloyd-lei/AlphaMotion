"""Per-body macro statistics and source-clip cluster bootstrap for fixed eval sets.

Three short runs are empirical budget/quality comparisons, never a scaling law.
Failed cases are retained; a missing unconditional metric makes its aggregate
unavailable instead of silently averaging successful cases only.
"""
from __future__ import annotations
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import numpy as np


def nested(record,path):
    value=record
    for key in path.split('.'):
        if not isinstance(value,dict) or key not in value:return None
        value=value[key]
    return value


def training_curve(path):
    """Read actual current nested terms schema, with explicitly marked legacy fallback."""
    records=[]
    with Path(path).open() as handle:
        for lineno,line in enumerate(handle,1):
            if not line.strip():continue
            try:r=json.loads(line)
            except json.JSONDecodeError as exc:raise ValueError(f'invalid complete JSONL line {lineno}') from exc
            terms=r.get('terms')
            if isinstance(terms,dict):
                loss=terms.get('total');schema='terms.total'
            else:loss=r.get('loss');schema='legacy.loss'
            if loss is None:raise ValueError(f'no actual loss in line{lineno}')
            if not isinstance(loss,(int,float)) or not math.isfinite(loss):raise ValueError(f'nonfinite training loss line{lineno}')
            records.append({'step':r['step'],'loss':loss,'rotation_deg':terms.get('rotation_deg') if isinstance(terms,dict) else None,
                'position_over_reach':terms.get('position_over_reach') if isinstance(terms,dict) else None,
                'training_elapsed_seconds':r.get('training_elapsed_seconds'),'domain':r.get('domain'),'schema':schema})
    return records


def evaluation_set_digest(cases):
    rows=[]
    for c in cases:
        rows.append({'case_id':c.case_id,'clip_id':c.clip_id,'source_body':c.source_body,'target_body':c.target_body,
                     'task_kind':c.task_kind,'body_exposure':c.body_exposure,'clip_exposure':c.clip_exposure,
                     'frame_ids':np.asarray(c.frame_ids).tolist(),'timestamps_seconds':np.asarray(c.timestamps_seconds).tolist(),
                     'condition_set_sha256':c.condition_set_sha256,'joint_names':c.joint_names,'metric':c.metric,'target_morphology':c.target_morphology,'provenance':c.provenance})
    if len({r['case_id'] for r in rows})!=len(rows):raise ValueError('duplicate eval case_id')
    raw=json.dumps(sorted(rows,key=lambda r:r['case_id']),sort_keys=True,separators=(',',':'),allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def _stratum(result,metric_path):
    metric=nested(result,metric_path)
    return (result['task_kind'],result['body_exposure'],result['clip_exposure'],result['alignment'],metric.get('unit') if isinstance(metric,dict) else None,result.get('target_morphology','unspecified'))


def _aggregate(records,metric_path):
    by=defaultdict(list)
    for r in records:by[r['target_body']].append(r)
    per_body={};micro_num=0.;micro_den=0;missing=[]
    for name,rows in sorted(by.items()):
        means=[];finite=[];count=0
        for r in rows:
            m=nested(r,metric_path)
            value=m.get('mean') if isinstance(m,dict) else None
            if value is None or not math.isfinite(value):missing.append(r['case_id'])
            else:
                means.append(value);finite.append(value)
                n=m['total_values'];micro_num+=value*n;micro_den+=n
            count+=1
        per_body[name]={'mean':float(np.mean(means)) if len(means)==count and count else None,
                        'finite_clip_mean':float(np.mean(finite)) if finite else None,
                        'clips':count,'measured_clips':len(means),'failed_or_unavailable_clips':count-len(means)}
    macro=[v['mean'] for v in per_body.values()]
    return {'body_macro':float(np.mean(macro)) if macro and all(v is not None for v in macro) else None,
            'joint_time_micro':micro_num/micro_den if micro_den and not missing else None,
            'per_body':per_body,'cases':len(records),'bodies':len(by),'source_clips':len({r['clip_id'] for r in records}),
            'failed_or_unavailable_case_ids':missing,
            'failure_frame_fraction':sum(r['failed_frames'] for r in records)/sum(r['frame_count'] for r in records),
            'aggregation':'equal clip means within each body then equal-body macro; micro separately weights joint/time count; failed cases retained'}


def summarize_cases(records,metric_path='metrics.position_over_reach'):
    if len({r['case_id'] for r in records})!=len(records):raise ValueError('duplicate case records; aggregate one run/checkpoint at a time')
    grouped=defaultdict(list)
    for r in records:grouped[_stratum(r,metric_path)].append(r)
    return [{'task_kind':k[0],'body_exposure':k[1],'clip_exposure':k[2],'alignment':k[3],'unit':k[4],
             'target_morphology':k[5],'metric_path':metric_path,**_aggregate(v,metric_path)} for k,v in sorted(grouped.items(),key=lambda x:repr(x[0]))]


def bootstrap_clips(records,metric_path='metrics.position_over_reach',*,samples=2000,seed=0,confidence=.95,minimum_clips=3):
    """Resample physical source clip clusters, jointly retaining every target body.

    Call separately for each task/exposure/alignment/unit stratum. Same physical
    clip mapped to several bodies is one cluster, not independent observations.
    """
    if not records:raise ValueError('no cases')
    if len({r['case_id'] for r in records})!=len(records):raise ValueError('duplicate cases are not independent clip observations')
    if type(samples) is not int or samples<1 or not 0<confidence<1:raise ValueError('bootstrap settings')
    if len({_stratum(r,metric_path) for r in records})!=1:raise ValueError('bootstrap strata must not mix tasks, exposure, alignment or units')
    groups=defaultdict(list)
    for r in records:groups[r['clip_id']].append(r)
    base=_aggregate(records,metric_path)
    if type(minimum_clips) is not int or minimum_clips<2:raise ValueError('CI requires a configured minimum of at least2 independent clips')
    if len(groups)<minimum_clips:return {'status':'insufficient_independent_clips','estimate':base['body_macro'],'source_clips':len(groups),'minimum_clips':minimum_clips,'confidence_interval':None,'aggregate':base}
    if base['body_macro'] is None:return {'status':'unavailable_including_failed_cases','aggregate':base,'source_clips':len(groups),'seed':seed,'samples':samples}
    perbody=defaultdict(set)
    for r in records:perbody[r['target_body']].add(r['clip_id'])
    if any(len(v)<minimum_clips for v in perbody.values()):return {'status':'insufficient_independent_clips_per_body','estimate':base['body_macro'],'clips_per_body':{k:len(v) for k,v in perbody.items()},'minimum_clips':minimum_clips,'confidence_interval':None}
    keys=sorted(groups);rng=np.random.default_rng(seed);draws=[];failures=[]
    for _ in range(samples):
        for attempt in range(1000):
            selected=rng.choice(keys,len(keys),replace=True)
            rows=[r for k in selected for r in groups[k]]
            if {r['target_body'] for r in rows}==set(perbody):break
        else:return {'status':'insufficient_cluster_coverage_for_macro','confidence_interval':None,'source_clips':len(keys)}
        a=_aggregate(rows,metric_path);draws.append(a['body_macro']);failures.append(a['failure_frame_fraction'])
    alpha=(1-confidence)/2
    return {'status':'measured','estimate':base['body_macro'],'confidence_interval':np.quantile(draws,[alpha,1-alpha]).tolist(),
            'confidence':confidence,'source_clips':len(keys),'samples':samples,'seed':seed,'unit':next(iter({_stratum(r,metric_path)[4] for r in records})),
            'resampling_unit':'physical source clip cluster; all body outcomes retained together; redraw if a target body has no observations','body_aggregation':'equal-body macro of equal-clip means',
            'few_clusters_warning':len(keys)<10}


def paired_clip_difference(left,right,metric_path='metrics.position_over_reach',*,samples=2000,seed=0,confidence=.95):
    """Bootstrap paired per-case differences; negative means left is better."""
    L={r['case_id']:r for r in left};R={r['case_id']:r for r in right}
    if len(L)!=len(left) or len(R)!=len(right) or set(L)!=set(R):raise ValueError('paired comparison requires identical case set, no drops')
    rows=[]
    for key,a in L.items():
        b=R[key]
        if _stratum(a,metric_path)!=_stratum(b,metric_path) or (a['clip_id'],a['target_body'],a['frame_count'])!=(b['clip_id'],b['target_body'],b['frame_count']):raise ValueError('paired case contract mismatch')
        ma,mb=nested(a,metric_path),nested(b,metric_path)
        if not isinstance(ma,dict) or not isinstance(mb,dict):raise ValueError('paired metric absent')
        value=ma['mean']-mb['mean'] if ma['mean'] is not None and mb['mean'] is not None else None
        row=dict(a,metrics={'difference':{'mean':value,'total_values':ma['total_values'],'unit':ma['unit']}})
        row['failed_frames']=max(a['failed_frames'],b['failed_frames']);rows.append(row)
    result=bootstrap_clips(rows,'metrics.difference',samples=samples,seed=seed,confidence=confidence)
    result['sign']='left minus right; negative is better for an error metric';return result


def budget_comparison(checkpoints,*,axis,budget,evaluation_set_sha256,relative_tolerance=.01):
    """Select each run's closest earlier point under an explicit fixed budget.

    Required keys: run_id, checkpoint_id, examples_seen, training_compute_seconds,
    hardware_id, evaluation_set_sha256. Compute is measured training device wall
    time times device count, excluding evaluation; incompatible hardware rejected.
    Exact/effectively equal budgets are distinguished from under-budget snapshots.
    """
    if axis not in {'examples_seen','training_compute_seconds'}:raise ValueError('use separate exposure and measured-compute axes')
    if not math.isfinite(budget) or budget<=0:raise ValueError('positive budget')
    grouped=defaultdict(list)
    for c in checkpoints:
        if c.get('evaluation_set_sha256')!=evaluation_set_sha256:raise ValueError('fixed eval set changed')
        value=c.get(axis)
        if not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:raise ValueError('missing actual budget measurement')
        grouped[c['run_id']].append(c)
    if axis=='training_compute_seconds' and len({c.get('hardware_id') for c in checkpoints})!=1:raise ValueError('fixed compute needs one declared hardware class')
    if axis=='training_compute_seconds' and any(not c.get('hardware_id') for c in checkpoints):raise ValueError('compute hardware identity absent')
    selected=[]
    for run,points in sorted(grouped.items()):
        candidates=[c for c in points if c[axis]<=budget]
        if not candidates:
            selected.append({'run_id':run,'status':'no_checkpoint_under_budget'});continue
        chosen=max(candidates,key=lambda c:c[axis]);gap=(budget-chosen[axis])/budget
        selected.append({'run_id':run,'status':'matched' if gap<=relative_tolerance else 'under_budget_not_equal',
                         'requested_budget':budget,'actual_budget':chosen[axis],'relative_shortfall':gap,'checkpoint':chosen})
    return {'axis':axis,'budget':budget,'selected':selected,'evaluation_set_sha256':evaluation_set_sha256,
            'claim':'empirical finite-budget comparison; three model sizes/short runs do not establish a scaling law'}


def refinement_curve(refinement_results,metric='selected_position_mean'):
    """Retain every case and each requested integer step, including failed tails."""
    out=[]
    for case_id,result in refinement_results.items():
        expected=list(range(result['requested_steps']+1));actual=[r['step'] for r in result['records']]
        if actual!=expected:raise ValueError('refinement curve dropped/repeated a step')
        for row in result['records']:
            out.append({'case_id':case_id,'step':row['step'],'value':row.get('metrics',{}).get(metric),
                        'status':row['status'],'wall_seconds':row['elapsed_seconds'],
                        'optimizer_steps':row['executed_optimizer_steps'],'unit':result['protocol']['units'] if 'position' in metric else 'degree'})
    return out


def select_fixed_eval_cases(cases,*,max_source_clips_per_stratum=None,seed=0):
    """Select whole source clips before any model result exists; retain body pairs."""
    grouped=defaultdict(list)
    for c in cases:
        c.validate()
        if not c.complete_clip:raise ValueError('fixed trajectory eval set requires complete clips')
        grouped[(c.task_kind,c.body_exposure,c.clip_exposure,c.target_morphology)].append(c)
    rng=np.random.default_rng(seed);selected=[]
    for key,group in sorted(grouped.items()):
        ids=sorted({c.clip_id for c in group});rng.shuffle(ids)
        if max_source_clips_per_stratum is not None:
            if type(max_source_clips_per_stratum) is not int or max_source_clips_per_stratum<1:raise ValueError('positive source clip budget')
            ids=ids[:max_source_clips_per_stratum]
        chosen=set(ids);selected.extend(c for c in group if c.clip_id in chosen)
    selected=sorted(selected,key=lambda c:c.case_id)
    return selected,{'seed':seed,'max_source_clips_per_stratum':max_source_clips_per_stratum,
        'cases':len(selected),'evaluation_set_sha256':evaluation_set_digest(selected),
        'case_ids':[c.case_id for c in selected],'policy':'whole source clip selection independent of model outcome; no hard-case filtering'}
