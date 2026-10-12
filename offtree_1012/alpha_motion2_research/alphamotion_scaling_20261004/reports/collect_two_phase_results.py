"""Merge immutable phase1 and phase2 optimizer evidence without changing either."""
from pathlib import Path
import argparse,collections,hashlib,json
import collect_live_results as base

ORDER=base.ORDER
def read(p):return json.loads(Path(p).read_text())

def collect(root):
    report=base.collect(root);report['scope']='Six seed0 runs with common second phase; LR restart and stable sigmoid derivative are declared interventions, not a pure extra-step experiment.'
    report['phase2_plan']=read(root/'continuation_v3/execution_plan.json')
    report['phase2_plan_sha256']=base.sha(root/'continuation_v3/execution_plan.json')
    report['initial_phase_comparisons']=dict(report['comparisons'])
    samples={};nodes=[]
    for name in ORDER:
        r=report['runs'][name];run=root/'continuation_v3/runs'/name
        r['initial_phase']=dict(status=r['status'],steps=r['steps'],training_seconds=r.get('training_seconds'),result=r.get('result'))
        rows=list(base.read_logs(run/'train.jsonl'));samples[name]=[x['sampling_sha256'] for x in rows]
        if not rows:r['status']='phase2_not_started';continue
        if [x['step'] for x in rows]!=list(range(3001,rows[-1]['step']+1)):raise ValueError('nonconsecutive phase2 optimizer records: '+name)
        last=rows[-1];r.update(status='phase2_running',steps=last['step'],additional_steps=len(rows),training_seconds=last['cumulative_train_seconds'],phase2_training_seconds=last['phase_train_seconds'],wall_seconds=last['cumulative_wall_seconds'],estimated_dense_flops=last['cumulative_estimated_training_flops'],tokens=last['cumulative_tokens'])
        r['curve']+=base.curves(rows)
        r['domain_updates']=dict(collections.Counter(r['domain_updates'])+collections.Counter(x['domain'] for x in rows))
        r['phase2_sampling_sha256']=hashlib.sha256('\n'.join(samples[name]).encode()).hexdigest()
        for key,field in [('valid_loss_sample_totals','valid_loss_samples'),('target_dataset_direction_rows',None),('mode_direction_rows',None)]:
            totals=collections.Counter(r[key])
            for item in rows:
                for direction in item['directions']:
                    if field:totals.update(direction[field])
                    else:totals[direction['actual_target_dataset'] if key=='target_dataset_direction_rows' else direction['mode']]+=direction['rows']
            r[key]=dict(totals)
        for f,k in [('result.json','phase2_result'),('exit_status.json','phase2_exit'),('failure.json','phase2_failure')]:
            if (run/f).exists():r[k]=read(run/f)
        if r.get('phase2_result',{}).get('status')=='completed':r['status']='completed6000'
        elif 'phase2_failure' in r:r['status']='phase2_failed'
        cfg=read(run/'config.json')
        for p in sorted((run/'eval').glob('evaluation_*.json')):
            d=read(p);label=d['label'];step=cfg['steps'] if label=='final' else int(label.removeprefix('step'))
            sample=rows[step-3001];signature=base.exact_eval_signature(p.parent,d)
            point=dict(run=name,label='final6000' if label=='final' else label,step=step,phase=2,evaluation_set_sha256=signature,
                training_seconds=sample['cumulative_train_seconds'],estimated_dense_flops=sample['cumulative_estimated_training_flops'],cases=d['cases'],file=str(p),sha256=base.sha(p))
            r['evaluations'].append(point);nodes.append(point)
        r['phase2_serialized_files_bytes']=sum(p.stat().st_size for p in run.rglob('*') if p.is_file())
    started=[n for n in ORDER if samples[n]];common=min((len(samples[n]) for n in started),default=0)
    differences=[]
    for name in started[1:]:
        mismatch=next((i+3001 for i,(a,b) in enumerate(zip(samples[started[0]][:common],samples[name][:common])) if a!=b),None)
        if mismatch is not None:differences.append(dict(run=name,first_global_step=mismatch))
    report['comparisons']['phase2_actual_shared_sampling']=dict(runs=started,shared_additional_steps=common,all_six_available=len(started)==6,mismatches=differences)
    report['comparisons']['phase2_evaluation_nodes']=[]
    for label in sorted({p['label'] for p in nodes}):
        group=[p for p in nodes if p['label']==label]
        report['comparisons']['phase2_evaluation_nodes'].append(dict(label=label,runs=[p['run'] for p in group],all_six_available=len(group)==6,same_exact_evaluation_rows_masks=len({p['evaluation_set_sha256'] for p in group})==1,evaluation_set_sha256=group[0]['evaluation_set_sha256']))
    report['comparisons']['cross_phase_quick_evaluation_sets_match']=len({p['evaluation_set_sha256'] for r in report['runs'].values() for p in r['evaluations'] if p['label'].startswith('step')})==1
    report['all_six_complete6000']=all(r['status']=='completed6000' for r in report['runs'].values())
    # A common cost cut may select nodes from different phases. Report it as an
    # observed resource comparison, never an isolated architecture/step effect.
    for axis in ['training_seconds','estimated_dense_flops']:
        available={n:[p for p in report['runs'][n]['evaluations'] if p['label'].startswith('step')] for n in ORDER}
        if all(available.values()):
            budget=min(max(p[axis] for p in v) for v in available.values());selected=[]
            for n,v in available.items():
                earlier=[p for p in v if p[axis]<=budget]
                if not earlier:selected.append(dict(run=n,status='no_saved_point_under_budget'));continue
                p=max(earlier,key=lambda x:x[axis]);gap=(budget-p[axis])/budget
                selected.append(dict(run=n,label=p['label'],step=p['step'],phase=p.get('phase',1),actual_budget=p[axis],relative_shortfall=gap,
                    status='matched_within_1pct' if gap<=.01 else 'earlier_under_budget_not_equal',evaluation_set_sha256=p['evaluation_set_sha256']))
            report['comparisons'][axis]=dict(common_covered_budget=budget,selected=selected,estimated_not_measured_flops=axis=='estimated_dense_flops',
                scope='Observed saved nodes under a shared cumulative budget; may mix phases; LR restart and derivative correction confound a pure step or architecture attribution.')
    if differences:raise ValueError('Cross-model phase2 sampling mismatch')
    return report

def plot(report,out):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'S':'#008f83','M':'#4169bd','L':'#b04475'}
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    for name,r in report['runs'].items():
        points=[p for p in r['curve'] if p['domain']=='xwbc']
        for row,(key,yl) in enumerate([('position_over_reach','Joint position error / static reach'),('rotation_deg','Joint rotation error (degrees)')]):
            for col,(axis,label,factor) in enumerate([('mean_step','Global optimizer updates',1),('mean_training_seconds','Observed cumulative training minutes',1/60),('mean_estimated_dense_flops','Estimated dense training TFLOPs',1e-12)]):
                for phase in [1,2]:
                    selected=[p for p in points if (p['mean_step']<=3000)==(phase==1) and key in p['terms']]
                    axes[row,col].plot([p[axis]*factor for p in selected],[p['terms'][key] for p in selected],color=colors[name[2]],linestyle='-' if name[0]=='A' else '--',alpha=1 if phase==1 else .65,label=name if phase==1 else None)
                axes[row,col].set(xlabel=label,ylabel=yl);axes[row,col].grid(alpha=.2)
                if col==0:axes[row,col].axvline(3000,color='#555555',linewidth=.6)
    axes[0,0].legend(ncols=2,fontsize=8)
    fig.suptitle('Actual shared XWBC training terms; six seed0 runs\nPhase2: common LR restart + sigmoid derivative correction; cross-phase changes are not solely due to step count',fontsize=12)
    fig.savefig(out,dpi=160,bbox_inches='tight');plt.close(fig)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--plot',action='store_true');a=p.parse_args()
    report=collect(a.root);base.write(a.output,report)
    if a.plot:plot(report,a.output.with_suffix('.png'))
    print(json.dumps(dict(all_six_complete6000=report['all_six_complete6000'],sampling=report['comparisons']['phase2_actual_shared_sampling'],runs={n:{k:v.get(k) for k in ['status','steps','phase2_training_seconds']} for n,v in report['runs'].items()})))
if __name__=='__main__':main()
