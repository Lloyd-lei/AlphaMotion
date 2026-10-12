"""Matched 3000/6000 and P0/P1 summaries from saved evidence only."""
from pathlib import Path
import argparse, collections, csv, json
import numpy as np
import summarize_manipulation as base

FOLDERS={
    (3000,'P0'):'evaluation/results/manipulation',
    (3000,'P1'):'evaluation_p1/results/manipulation',
    (6000,'P0'):'evaluation_phase2_v2/p0/manipulation',
    (6000,'P1'):'evaluation_phase2_v2/p1/results/manipulation',
}

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    all_cases={};identities={};sources={};summaries={};table=[]
    for (step,protocol),relative in FOLDERS.items():
        folders={run:a.root/relative/run/f'checkpoint_step{step:04d}' for run in base.RUNS}
        report=base.summarize(folders,100,1000)
        report.update(execution_scope='formal_all_six_fixed_cases',protocol_label=protocol,training_step=step)
        folder=a.output/f'{protocol}_step{step}';folder.mkdir(exist_ok=True)
        (folder/'manipulation_comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        if step==6000:
            base.plot(report,folder);base.plot_all_stages(report,folder)
        summaries[f'{protocol}_{step}']=dict(path=str(folder/'manipulation_comparison.json'),sha256=base.sha(folder/'manipulation_comparison.json'))
        for run,folder in folders.items():
            records={x.parent.name:base.read_case(x,100) for x in sorted(folder.rglob('case_result.json'))}
            all_cases[(step,protocol,run)]=records
            for cid,r in records.items():
                sources[r['path']]=r['sha256']
                if r['failed']:continue
                identity=dict(reference_sha256=r['reference_sha256'],body=r['document']['body'],clip_id=r['document']['clip_id'],cohort=r['document']['provenance']['reference_cohort'])
                if cid in identities and identity!=identities[cid]:raise ValueError('Cross-phase/protocol reference mismatch: '+cid)
                identities[cid]=identity
        for group in report['groups']:
            for run,stages in group['curves'].items():
                for stage,metrics in stages.items():
                    for metric,stat in metrics.items():
                        scale=10 if metric=='selected_position_cm' else 100 if metric=='target_frame_fraction' else 1
                        table.append(dict(training_step=step,protocol=protocol,run=run,body=group['body'],cohort=group['reference_cohort'],stage=stage,
                            metric={'selected_position_cm':'selected_position_mm','target_frame_fraction':'target_frame_percent'}.get(metric,metric),
                            mean_initial=None if stat['mean'] is None else stat['mean'][0]*scale,
                            mean_final=None if stat['mean'] is None else stat['mean'][-1]*scale,
                            final_ci_lower=None if stat['confidence_interval'] is None else stat['confidence_interval'][0][-1]*scale,
                            final_ci_upper=None if stat['confidence_interval'] is None else stat['confidence_interval'][1][-1]*scale,
                            status=stat['status'],physical_clips=stat['physical_clips'],cases=stat['cases']))
    expected=set(next(iter(all_cases.values())))
    if any(set(rs)!=expected for rs in all_cases.values()):raise ValueError('Incomplete matched case coverage')
    if set(identities)!=expected:raise ValueError('At least one case has no recoverable successful reference identity')
    groups=collections.defaultdict(list)
    for cid,identity in sorted(identities.items()):groups[(identity['body'],identity['cohort'])].append(cid)
    changes=[];aperture=[]
    comparisons=[('phase6000_minus3000', (3000,p),(6000,p)) for p in ['P0','P1']]+[('solverP1_minusP0',(s,'P0'),(s,'P1')) for s in [3000,6000]]
    for label,left,right in comparisons:
        for run in base.RUNS:
            l=all_cases[(*left,run)];r=all_cases[(*right,run)]
            for (body,cohort),ids in groups.items():
                clips=[identities[cid]['clip_id'] for cid in ids]
                for stage in base.STAGES:
                    for metric in base.METRICS:
                        differences=[]
                        for cid in ids:
                            x,y=l[cid],r[cid]
                            valid=not(x['failed'] or y['failed']) and x['curves'].get(stage) is not None and y['curves'].get(stage) is not None
                            differences.append(y['curves'][stage][metric]-x['curves'][stage][metric] if valid else np.full(101,np.nan))
                        changes.append(dict(comparison=label,left=list(left),right=list(right),run=run,body=body,cohort=cohort,stage=stage,metric=metric,
                            **base.cluster_summary(differences,clips)))
    for (step,protocol,run),records in all_cases.items():
        for (body,cohort),ids in groups.items():
            if body!='relay_franka_full_v2':continue
            clips=[identities[cid]['clip_id'] for cid in ids]
            for stage in base.STAGES:
                values=[]
                for cid in ids:
                    rec=records[cid]
                    obs=None if rec['failed'] else rec['document']['stages'][stage].get('final_observation')
                    value=None if obs is None else obs.get('aperture',{}).get('absolute_error',{}).get('mean')
                    values.append([np.nan if value is None else value*10])
                aperture.append(dict(training_step=step,protocol=protocol,run=run,body=body,cohort=cohort,stage=stage,unit='mm',
                    scope='Finger-origin signed opening error after saved final stage; not contact-pad clearance',**base.cluster_summary(values,clips)))
    with (a.output/'endpoint_results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    out=dict(schema='two_phase_manipulation_comparison_v1',all_six_available=True,case_count_per_run=len(expected),
        complete_cross_phase_protocol_reference_identity_match=True,summaries=summaries,endpoint_rows=table,paired_changes=changes,final_opening=aperture,
        case_sources_sha256=sources,
        phase_caveat='6000 minus3000 includes the common LR restart, fresh shared RNG and stable sigmoid backward correction, as well as extra updates. Not a pure training-step effect.',
        solver_caveat='P1 changes rotation weighting and native-q normalization together; same fixed100 updates. Its timing uses reused P0 inference plus newly measured solver time, a composite estimate.',
        uncertainty='Three source episodes per cohort, clustered bootstrap1000, single training seed; no training-seed uncertainty.',
        failure_case_records=sum(r['failed'] for rs in all_cases.values() for r in rs.values()),
        claim_boundary='Same-body visible sparse SE3, native-q limits where verified; no cross-end-effector transfer, contact, grasp, dynamics or physical execution.')
    (a.output/'two_phase_manipulation.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(a.output),case_sources=len(sources),endpoint_rows=len(table),paired_changes=len(changes),failure_case_records=out['failure_case_records'])))

if __name__=='__main__':main()
