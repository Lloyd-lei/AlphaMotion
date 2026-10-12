"""Compact human-review facts from completed, paired two-phase reports."""
from pathlib import Path
import argparse,json

def read(p):return json.loads(p.read_text())
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root;out=root/'evaluation_phase2_v2/reports'
    training=read(root/'reports/two_phase_results.json');loco=read(out/'locomotion/locomotion_comparison.json');manip=read(out/'manipulation/two_phase_manipulation.json');slide=read(out/'slide/slide_saturation_comparison.json')
    if not training['all_six_complete6000']:raise RuntimeError('Not all six6000 complete')
    result=dict(training={n:dict(steps=r['steps'],initial_training_seconds=r['initial_phase']['training_seconds'],phase2_training_seconds=r['phase2_training_seconds'],total_training_seconds=r['training_seconds'],
        estimated_dense_flops=r['estimated_dense_flops'],parameter_counts=r['phase2_result']['parameter_counts'],
        valid_loss_sample_totals=r['valid_loss_sample_totals'],target_dataset_direction_rows=r['target_dataset_direction_rows'],tokens=r['tokens']) for n,r in training['runs'].items()},
        sampling={k:v for k,v in training['comparisons'].items() if 'sampling' in k or 'match' in k},
        cost_comparisons={k:training['comparisons'][k] for k in ['training_seconds','estimated_dense_flops']})
    table=[]
    for r in loco['rows']:
        if r['step'] not in [3000,4500,6000] or r['clip_exposure']!='heldout_clip':continue
        if r['body_exposure']!='heldout_body' and r['body']!='unitree_g1_29dof':continue
        metrics={k:{x:v.get(x) for x in ['status','mean','confidence_interval','CI_status']} for k,v in r['metrics'].items() if k in ['position','position_over_reach','rotation_degrees','velocity_error','acceleration_error','jerk_error']}
        table.append({**{k:r[k] for k in ['run','step','body','dataset','body_exposure','input_mode','physical_clips','position_unit','failed_case_ids']},'metrics':metrics})
    result['locomotion_selected_whole_body_holdouts_and_G1']=table
    result['locomotion_all_body_count']=len({r['body'] for r in loco['rows']})
    result['locomotion_failed_records']=sum(len(r['failed_case_ids']) for r in loco['rows'])
    comparisons=[]
    for step in [3000,6000]:
        for mode in ['full_source','pure_se3','masked_source']:
            for size in 'SML':
                for dataset in ['XWBC','UniMate']:
                    rows=[r for r in loco['paired_B_minus_A'] if r['step']==step and r['input_mode']==mode and r['size']==size and r['clip_exposure']=='heldout_clip' and r['metric']=='position_over_reach']
                    held={r['body'] for r in loco['rows'] if r['body_exposure']=='heldout_body' and r['dataset']==dataset};rows=[r for r in rows if r['body'] in held]
                    comparisons.append(dict(step=step,dataset=dataset,input_mode=mode,size=size,bodies=len(rows),B_lower_error=sum(r['mean'] is not None and r['mean']<0 for r in rows),A_lower_error=sum(r['mean'] is not None and r['mean']>0 for r in rows),
                        B_negative_episode_CI=sum(r['confidence_interval'] is not None and r['confidence_interval'][1]<0 for r in rows),
                        A_positive_episode_CI=sum(r['confidence_interval'] is not None and r['confidence_interval'][0]>0 for r in rows),
                        task='paired source-to-target mapping' if dataset=='XWBC' and mode=='full_source' else 'same-body reconstruction' if mode=='full_source' else 'conditioned posture diagnostic',
                        caveat='Descriptive error-direction counts, not a retargeting success rate. Bodies/source clips are not independent training seeds; no multiplicity-adjusted superiority claim.'))
    result['heldout_body_error_direction_counts_by_dataset']=comparisons
    result['manipulation_strict_endpoint_rows']=[r for r in manip['endpoint_rows'] if r['cohort']=='strict_limit_valid_control']
    result['manipulation_failed_execution_cases']=manip['failure_case_records']
    result['manipulation_strict_opening']=[r for r in manip['final_opening'] if r['cohort']=='strict_limit_valid_control']
    result['slide_strict']=[r for r in slide['rows'] if r['cohort']=='strict_limit_valid_control']
    path=out/'final_review_inputs.json';path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(path)

if __name__=='__main__':main()
