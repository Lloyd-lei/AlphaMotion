"""Aggregate all saved formal continuous windows, preserving units and failures."""
from pathlib import Path
import argparse,collections,csv,hashlib,json
import numpy as np

RUNS=[f'{v}_{s}_seed0' for s in 'SML' for v in 'AB']
STEPS=[100,250,500,1000,1500,2000,2500,3000]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def scalar(record,path):
    value=record
    for key in path.split('.'):
        if not isinstance(value,dict):return np.nan
        value=value.get(key)
    return np.nan if value is None else float(value)


def statistic(values,clips):
    values=np.asarray(values,dtype=float);keys=sorted(set(clips))
    d=dict(status='measured',mean=None,confidence_interval=None,cases=len(values),physical_clips=len(keys))
    if not np.isfinite(values).all():
        d['status']='unavailable_including_failed_cases';return d
    d['mean']=float(values.mean())
    if len(keys)<3:
        d['CI_status']='insufficient_independent_clips';return d
    groups=[np.flatnonzero(np.asarray(clips)==key) for key in keys]
    counts=np.asarray([len(g) for g in groups]);sums=np.asarray([values[g].sum() for g in groups])
    indices=np.random.default_rng(0).integers(0,len(keys),size=(1000,len(keys)))
    boot=sums[indices].sum(1)/counts[indices].sum(1)
    d.update(CI_status='measured',confidence_interval=np.quantile(boot,[.025,.975]).tolist(),confidence=.95,
        bootstrap_samples=1000,bootstrap_seed=0,resampling_unit='physical episode cluster; every subclip retained',few_clusters_warning=len(keys)<10)
    return d


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    live=json.loads((args.root/'reports/live_results.json').read_text());reports={};sources={};rows=[];comparison=[]
    metric_paths={'position':'metrics.position.mean','position_over_reach':'metrics.position_over_reach.mean',
        'rotation_degrees':'metrics.rotation.mean','endpoint_position':'endpoints.declared_endpoint_policy.position.mean',
        'endpoint_rotation_degrees':'endpoints.declared_endpoint_policy.rotation.mean',
        'velocity_error':'error_window_temporal.speed.mean','acceleration_error':'error_window_temporal.acceleration.mean',
        'jerk_error':'error_window_temporal.jerk.mean'}
    expected=None;identities={};matrices={};clipsets={}
    for step in STEPS:
        for run in RUNS:
            path=args.root/'evaluation/results/locomotion'/run/f'checkpoint_step{step:04d}'/'evaluation_summary.json'
            d=json.loads(path.read_text());reports[(run,step)]=d;sources[str(path)]=sha(path)
            records={r['case_id']:r for r in d['records']}
            if expected is None:expected=set(records)
            if set(records)!=expected:raise ValueError('Formal case coverage differs')
            for cid,r in records.items():
                ident={k:r[k] for k in ['source_body','target_body','clip_id','input_mode','body_exposure','clip_exposure','condition_set_sha256','frame_count']}
                ident['reference_window_sha256']=r['provenance']['reference_window_numeric_sha256']
                if cid in identities and ident!=identities[cid]:raise ValueError('Unmatched formal reference/input '+cid)
                identities[cid]=ident
            grouped=collections.defaultdict(list)
            for r in records.values():grouped[(r['target_body'],r['input_mode'],r['clip_exposure'])].append(r)
            node=next(n for n in live['runs'][run]['evaluations'] if n['label']==f'step{step}')
            for group,rs in sorted(grouped.items()):
                rs.sort(key=lambda r:r['case_id']);clips=[r['clip_id'] for r in rs];first=rs[0]
                meta=dict(run=run,step=step,body=group[0],input_mode=group[1],clip_exposure=group[2],
                    body_exposure=first['body_exposure'],dataset=first['dataset'],morphology=first['target_morphology'],
                    position_unit=first['metrics']['position']['unit'],cases=len(rs),physical_clips=len(set(clips)),
                    training_seconds=node['training_seconds'],estimated_dense_flops=node['estimated_dense_flops'],
                    frame_count=sum(r['frame_count'] for r in rs),failed_case_ids=[r['case_id'] for r in rs if r['status']!='measured'])
                metrics={}
                for key,mp in metric_paths.items():
                    values=np.array([scalar(r,mp) if r['status']=='measured' else np.nan for r in rs])
                    metrics[key]=statistic(values,clips);matrices[(run,step,group,key)]=values
                clipsets[group]=clips;rows.append(dict(**meta,metrics=metrics))
        for group in clipsets:
            for size in 'SML':
                for key in metric_paths:
                    a=matrices[(f'A_{size}_seed0',step,group,key)];b=matrices[(f'B_{size}_seed0',step,group,key)]
                    comparison.append(dict(step=step,size=size,body=group[0],input_mode=group[1],clip_exposure=group[2],metric=key,
                        sign='B minus A; negative favors B for errors',**statistic(b-a,clipsets[group])))
    output=dict(schema='formal_locomotion_comparison_v1',runs=RUNS,steps=STEPS,case_count_per_node=len(expected),
        report_sources_sha256=sources,rows=rows,paired_B_minus_A=comparison,
        complete_reference_identity_match=True,scope='Continuous root-relative subclips, original heading/timestamps. No world-root recovery or physical execution.',
        uncertainty='Source episode cluster intervals, minimum3; a single training seed. No training-seed uncertainty or scaling law.',
        pure_se3_fullbody_role='Posture diagnostic: sparse targets do not uniquely specify all joints.',
        units='XWBC positions cm, plots may explicitly multiply10 to mm. UniMate canonical units or reach normalization, never physical mm.',
        source_cost_caveat='B-L initial timing includes documented late CPU probe concurrency; fixed-budget saved nodes are generally under budget, not exact equal cost.',
        saved_under_budget_comparisons={k:live['comparisons'][k] for k in ['training_seconds','estimated_dense_flops']})
    args.output.mkdir(parents=True,exist_ok=True);target=args.output/'locomotion_comparison.json';target.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    final=[r for r in rows if r['step']==3000]
    with (args.output/'final_by_body.csv').open('w',newline='') as f:
        fields=['run','body','dataset','body_exposure','clip_exposure','input_mode','position_unit','cases','physical_clips','frame_count']
        keys=list(metric_paths);writer=csv.DictWriter(f,fieldnames=fields+keys+['CI_available']);writer.writeheader()
        for r in final:writer.writerow({**{k:r[k] for k in fields},**{k:r['metrics'][k]['mean'] for k in keys},'CI_available':r['physical_clips']>=3})
    plot(output,args.output)
    print(json.dumps({'output':str(target),'reports':len(reports),'cases_per_node':len(expected),'heldout_bodies':sorted(set(r['body'] for r in rows if r['body_exposure']=='heldout_body')),'failed_case_records':sum(len(r['failed_case_ids']) for r in rows)}))


def plot(report,out):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'S':'#008f83','M':'#4169bd','L':'#b04475'}
    held=sorted(set(r['body'] for r in report['rows'] if r['body_exposure']=='heldout_body'))
    for mode in ['full_source','pure_se3','masked_source']:
        available=[b for b in held if any(r['body']==b and r['input_mode']==mode and r['clip_exposure']=='heldout_clip' for r in report['rows'])]
        if not available:continue
        for metric in ['position_over_reach','rotation_degrees']:
            fig,axes=plt.subplots(2,4,figsize=(17,8),layout='constrained');axes=axes.ravel()
            for ax,body in zip(axes,available):
                body_rows=[r for r in report['rows'] if r['body']==body and r['input_mode']==mode and r['clip_exposure']=='heldout_clip']
                label=body if body_rows[0]['dataset']=='XWBC' else 'UniMate '+body_rows[0]['morphology']+' '+body[:6]
                for run in RUNS:
                    series=sorted([r for r in body_rows if r['run']==run],key=lambda r:r['step'])
                    ax.plot([r['step'] for r in series],[np.nan if r['metrics'][metric]['mean'] is None else r['metrics'][metric]['mean'] for r in series],
                        color=colors[run[2]],linestyle='-' if run[0]=='A' else '--',marker='o',markersize=3,label=run.replace('_seed0',''))
                ax.set(title=label+' | '+str(body_rows[0]['physical_clips'])+' source clips',xlabel='Optimizer steps',
                    ylabel='Position error / static reach' if metric=='position_over_reach' else 'Joint rotation error (degrees)');ax.grid(alpha=.2);ax.set_ylim(bottom=0)
            for ax in axes[len(available):]:ax.set_visible(False)
            axes[0].legend(ncols=2,fontsize=8)
            fig.suptitle('All available whole-body-heldout results | '+mode+' | heldout motion windows\n'
                'A solid / B dashed; exact paired source frames and targets; sparse full-body error is a posture diagnostic; seed 0',fontsize=12)
            fig.savefig(out/f'heldout_bodies__{mode}__{metric}.png',dpi=150,bbox_inches='tight',pad_inches=.15);plt.close(fig)


if __name__=='__main__':main()
