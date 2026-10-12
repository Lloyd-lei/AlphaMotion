"""Aggregate existing manipulation evidence; no model, optimizer, or dataset writes."""
from pathlib import Path
import argparse, collections, hashlib, json
import numpy as np

RUNS = [f'{variant}_{size}_seed0' for size in 'SML' for variant in 'AB']
STAGES = ['raw_free_posture', 'dof_projected', 'constrained_refinement', 'default_valid_q']
METRICS = ['selected_position_cm', 'selected_rotation_degrees', 'target_frame_fraction']
REFERENCE_KEYS = ['frame_ids', 'timestamps_seconds', 'reference_position_cm',
    'reference_global_rot6', 'condition_joint_indices', 'condition_position_cm',
    'condition_global_rotation', 'condition_position_valid', 'condition_rotation_valid']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_case(path, steps):
    doc = json.loads(path.read_text())
    if doc.get('status') == 'failed_case_retained':
        return dict(document=doc, failed=True, path=str(path), sha256=sha(path))
    h = hashlib.sha256()
    with np.load(path.parent/'case_arrays.npz', allow_pickle=False) as arrays:
        for key in REFERENCE_KEYS:
            value = np.ascontiguousarray(arrays[key])
            h.update(key.encode()); h.update(str(value.dtype).encode())
            h.update(str(value.shape).encode()); h.update(value.tobytes())
    curves = {}
    for stage in STAGES:
        source = path.parent/stage/'curve_metrics.npz'
        if not source.exists():
            curves[stage] = None
            continue
        with np.load(source, allow_pickle=False) as arrays:
            if not np.array_equal(arrays['step'], np.arange(steps+1)):
                raise ValueError('Missing or unexpected refinement steps: '+str(source))
            curves[stage] = {key: arrays[key].copy() for key in METRICS+['elapsed_seconds']}
    return dict(document=doc, failed=False, reference_sha256=h.hexdigest(), curves=curves,
                path=str(path), sha256=sha(path))


def cluster_summary(values, clips, draws=1000):
    """Equal subclip means, with physical episodes as bootstrap clusters."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or not len(values):
        raise ValueError('Expected case by step matrix')
    if not np.isfinite(values).all():
        return dict(status='unavailable_including_failed_cases', mean=None,
                    confidence_interval=None, cases=len(values), physical_clips=len(set(clips)))
    mean = values.mean(0)
    keys = sorted(set(clips))
    result = dict(status='measured', mean=mean.tolist(), confidence_interval=None,
                  cases=len(values), physical_clips=len(keys), minimum_clips_for_CI=3)
    if len(keys) < 3:
        result['CI_status'] = 'insufficient_independent_clips'
        return result
    groups = {key: np.flatnonzero(np.asarray(clips) == key) for key in keys}
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(draws):
        selected = rng.choice(keys, len(keys), replace=True)
        boot.append(values[np.concatenate([groups[key] for key in selected])].mean(0))
    result.update(CI_status='measured', confidence_interval=np.quantile(boot,[.025,.975],axis=0).tolist(),
                  confidence=.95, bootstrap_samples=draws, bootstrap_seed=0,
                  resampling_unit='physical episode cluster; all selected subclips retained',
                  few_clusters_warning=len(keys)<10)
    return result


def summarize(inputs, steps, draws):
    all_cases = {}
    reports = {}
    for run, folder in inputs.items():
        report_path = folder/'evaluation_summary.json'
        reports[run] = json.loads(report_path.read_text())
        if reports[run]['refinement_config']['steps'] != steps:
            raise ValueError('Requested refinement budget differs from evidence: '+run)
        case_paths = sorted(folder.rglob('case_result.json'))
        all_cases[run] = {p.parent.name: read_case(p, steps) for p in case_paths}
        if len(all_cases[run]) != len(case_paths):
            raise ValueError('Duplicate case IDs: '+run)
        if len(all_cases[run]) != reports[run]['case_count']:
            raise ValueError('Incomplete case set: '+run)
    first = next(iter(all_cases))
    expected = set(all_cases[first])
    if any(set(cases) != expected for cases in all_cases.values()):
        raise ValueError('Different case sets; do not silently select an intersection')
    groups = collections.defaultdict(list)
    for case_id in sorted(expected):
        records = [cases[case_id] for cases in all_cases.values()]
        valid = [r for r in records if not r['failed']]
        base = (valid[0] if valid else records[0])['document']
        # Failure records omit provenance. Recover the declared cohort from a
        # matched successful case, or the runner's bound case-ID format.
        cohort = base.get('provenance',{}).get('reference_cohort') or '__'.join(case_id.split('__')[2:-1])
        if cohort not in {'strict_limit_valid_control','original_reference_start__has_soft_limit_violations','original_reference_start__within_limits'}:
            raise ValueError('Cannot establish reference cohort for '+case_id)
        if any(r['document']['provenance']['reference_cohort']!=cohort for r in valid):
            raise ValueError('Inconsistent declared cohort')
        groups[(base['body'],cohort)].append(case_id)
        if len({r['reference_sha256'] for r in valid}) > 1:
            raise ValueError('A/B reference or visible conditions differ: '+case_id)
        if any((r['document']['clip_id'],r['document']['body']) != (base['clip_id'],base['body']) for r in records):
            raise ValueError('Case identity mismatch')
    output = dict(schema='saved_manipulation_comparison_v1', steps=list(range(steps+1)),
        runs=list(inputs), all_six_available=set(inputs)==set(RUNS), groups=[],
        report_sources={run:{'path':str(inputs[run]/'evaluation_summary.json'),
            'sha256':sha(inputs[run]/'evaluation_summary.json'),
            'checkpoint_sha256':reports[run]['checkpoint_sha256'],
            'evaluation_set_sha256':reports[run]['evaluation_set_sha256']} for run in inputs},
        claim_boundary='Same-body sparse SE3 completion, seed 0. Reference cohorts stay separate. '
            'Clip intervals do not measure training-seed uncertainty. Known q limits do not establish '
            'collision, contact, grasp, dynamics, or physical-robot success. All failed cases retained.')
    for (body,cohort), ids in groups.items():
        clips=[all_cases[first][key]['document']['clip_id'] for key in ids]
        group=dict(body=body, reference_cohort=cohort, case_ids=ids, physical_clips=len(set(clips)),
                   curves={}, paired_B_minus_A=[], default_comparison=[], input_failures=[])
        matrices={}
        for run, cases in all_cases.items():
            group['curves'][run]={}
            for key in ids:
                if cases[key]['failed']: group['input_failures'].append({'run':run,'case_id':key})
            for stage in STAGES:
                metrics={}
                for metric in METRICS:
                    matrix=np.asarray([cases[key]['curves'][stage][metric] if not cases[key]['failed'] and cases[key]['curves'].get(stage) is not None else np.full(steps+1,np.nan) for key in ids])
                    matrices[(run,stage,metric)]=matrix
                    metrics[metric]=cluster_summary(matrix,clips,draws)
                group['curves'][run][stage]=metrics
            for key in ids:
                if cases[key]['failed']: continue
                doc=cases[key]['document']; timing=doc['default_control_comparison']
                group['default_comparison'].append(dict(run=run,case_id=key,clip_id=doc['clip_id'],
                    learned_step=timing['learned_projected_initialization']['first_step_meeting_threshold'],
                    default_step=timing['default_initialization']['first_step_meeting_threshold'],
                    learned_inference_projection_refinement_seconds=timing.get('learned_inference_plus_constrained_seconds_to_threshold'),
                    default_refinement_seconds=timing['default_initialization']['seconds_to_threshold'],
                    timing_scope=timing.get('timing_scope','legacy diagnostic: complete timing accounting unavailable')))
        for size in 'SML':
            a,b=f'A_{size}_seed0',f'B_{size}_seed0'
            if a not in inputs or b not in inputs: continue
            for stage in STAGES:
                for metric in METRICS:
                    difference=matrices[(b,stage,metric)]-matrices[(a,stage,metric)]
                    group['paired_B_minus_A'].append(dict(size=size,stage=stage,metric=metric,
                        sign='B minus A; negative favors B for error, positive favors B for target fraction',
                        **cluster_summary(difference,clips,draws)))
        output['groups'].append(group)
    return output


def plot(report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'S':'#008f83','M':'#4169bd','L':'#b04475'}
    for group in report['groups']:
        fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
        for run,stages in group['curves'].items():
            for ax,key,multiplier,label in zip(axes,METRICS,[10,1,100],['Selected endpoint error (mm)','Selected orientation error (degrees)','Frames meeting all visible SE3 tolerances (%)']):
                stat=stages['constrained_refinement'][key]
                if stat['mean'] is not None:
                    y=np.array(stat['mean'])*multiplier
                    ax.plot(report['steps'],y,label=run.replace('_seed0',''),color=colors[run[2]],linestyle='-' if run[0]=='A' else '--')
                ax.set(xlabel='Constrained refinement updates',ylabel=label);ax.grid(alpha=.2)
        # Plot each run's separately executed baseline, avoiding claims of exact timing identity.
        for i,(run,stages) in enumerate(group['curves'].items()):
            for ax,key,multiplier in zip(axes,METRICS,[10,1,100]):
                stat=stages['default_valid_q'][key]
                if stat['mean'] is not None:ax.plot(report['steps'],np.array(stat['mean'])*multiplier,color='#666666',linestyle=':',alpha=.65,label='Default valid q' if i==0 else None)
        for ax in axes:ax.relim();ax.autoscale(axis='y');ax.set_ylim(bottom=0)
        axes[2].set_ylim(0,102);axes[0].legend(fontsize=8,ncols=2)
        labels={'adroit_pen':'Adroit pen hand','relay_franka_full_v2':'Franka arm and fingers',
            'strict_limit_valid_control':'Reference within all known joint bounds',
            'original_reference_start__has_soft_limit_violations':'Original reference with soft-limit excursions',
            'original_reference_start__within_limits':'Original reference within known joint bounds'}
        scope='DIAGNOSTIC INTEGRATION ONLY | ' if report['execution_scope'].startswith('diagnostic') else report.get('protocol_label','P0')+' | '
        fig.suptitle(f"{scope}{labels.get(group['body'],group['body'])} | {labels.get(group['reference_cohort'],group['reference_cohort'])}\n{group['physical_clips']} source episodes; equal subclip means; same observed targets and optimizer; seed 0",fontsize=12)
        path=output/f"{group['body']}__{group['reference_cohort']}__constrained.png"
        fig.savefig(path,dpi=160);plt.close(fig)


def plot_all_stages(report, output):
    """Show where an unconstrained improvement survives real-DOF projection."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'S':'#008f83','M':'#4169bd','L':'#b04475'}
    titles=['Free posture\n(no joint-limit guarantee)',
            'Projection of each free step\n(not its own optimization)',
            'Native-q refinement\n(from projected learned pose)',
            'Native-q refinement\n(from default valid q)']
    names={'adroit_pen':'Adroit pen hand','relay_franka_full_v2':'Franka arm and fingers'}
    cohorts={'strict_limit_valid_control':'Reference within all known joint bounds',
        'original_reference_start__has_soft_limit_violations':'Original reference with soft-limit excursions',
        'original_reference_start__within_limits':'Original reference within known joint bounds'}
    for group in report['groups']:
        fig,axes=plt.subplots(3,4,figsize=(17,10),sharex=True,sharey='row',layout='constrained')
        for column,stage in enumerate(STAGES):
            axes[0,column].set_title(titles[column],fontsize=10)
            for run,stages in group['curves'].items():
                for row,(metric,scale) in enumerate(zip(METRICS,[10,1,100])):
                    stat=stages[stage][metric]
                    if stat['mean'] is None:continue
                    color='#777777' if stage=='default_valid_q' else colors[run[2]]
                    axes[row,column].plot(report['steps'],np.asarray(stat['mean'])*scale,
                        label=run.replace('_seed0',''),color=color,
                        linestyle='-' if run[0]=='A' else '--',alpha=.8)
            axes[2,column].set_xlabel('Refinement updates')
        for row,label in enumerate(['Selected endpoint error (mm)',
                'Selected orientation error (degrees)',
                'Frames within all tolerances (%)']):
            axes[row,0].set_ylabel(label)
            for ax in axes[row]:ax.grid(alpha=.2);ax.set_ylim(bottom=0)
        axes[2,0].set_ylim(0,102)
        axes[0,0].legend(fontsize=8,ncols=2)
        scope='DIAGNOSTIC INTEGRATION ONLY | ' if report['execution_scope'].startswith('diagnostic') else report.get('protocol_label','P0')+' | '
        fig.suptitle(f"{scope}{names.get(group['body'],group['body'])} | {cohorts[group['reference_cohort']]}\n"
            f"{group['physical_clips']} source episodes; means over the same subclips; common vertical scales within each row; seed 0",fontsize=12)
        path=output/f"{group['body']}__{group['reference_cohort']}__all_stages.png"
        fig.savefig(path,dpi=160);plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--step',type=int,default=3000)
    parser.add_argument('--run',action='append');parser.add_argument('--refinement-steps',type=int,default=100)
    parser.add_argument('--bootstrap-samples',type=int,default=1000);parser.add_argument('--diagnostic-input',type=Path)
    parser.add_argument('--input-root',type=Path);parser.add_argument('--protocol-label',default='P0')
    args=parser.parse_args();runs=args.run or RUNS
    if args.diagnostic_input and len(runs)!=1:parser.error('Diagnostic input requires one explicitly named run')
    base=args.input_root or args.root/'evaluation/results/manipulation'
    inputs={run:args.diagnostic_input or base/run/f'checkpoint_step{args.step:04d}' for run in runs}
    result=summarize(inputs,args.refinement_steps,args.bootstrap_samples)
    result['execution_scope']='diagnostic integration only' if args.diagnostic_input else 'formal saved evaluation'
    result['protocol_label']=args.protocol_label
    args.output.mkdir(parents=True,exist_ok=True)
    target=args.output/'manipulation_comparison.json';target.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    plot(result,args.output)
    plot_all_stages(result,args.output)
    print(json.dumps({'report':str(target),'scope':result['execution_scope'],'runs':runs,'groups':len(result['groups']),
        'failed_cases':sum(len(g['input_failures']) for g in result['groups'])}))


if __name__=='__main__':main()
