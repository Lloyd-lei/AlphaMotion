"""Aggregate the fixed pure-SE3 Franka diagnosis without inferring recovery."""
from pathlib import Path
import argparse,collections,hashlib,json
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    source=a.root/'evaluation_phase2_v2/slide_diagnostic/summary.json';d=json.loads(source.read_text());groups=collections.defaultdict(list);ids={}
    for r in d['records']:
        key=(r['run'],r['step'],r['cohort']);groups[key].append(r)
        identity=(r['condition_sha256'],r['reference_case_sha256'])
        # Original case_result files have model-dependent outputs; compare the
        # condition hash here and the actual saved frame/opening arrays below.
        cid=r['case_id'];artifact=Path(r['artifact'])
        if hashlib.sha256(artifact.read_bytes()).hexdigest()!=r['artifact_sha256']:raise ValueError('Diagnostic artifact changed')
        with np.load(artifact,allow_pickle=False) as z:
            signature=(identity[0],z['frame_ids'].tobytes(),z['timestamps_seconds'].tobytes(),z['reference_opening_cm'].tobytes())
            if cid in ids and signature!=ids[cid]:raise ValueError('Unmatched diagnostic input '+cid)
            ids[cid]=signature
    if len(d['records'])!=108 or len(groups)!=36:raise ValueError('Incomplete six-run by three-node by two-cohort diagnostic')
    rows=[]
    for (run,step,cohort),records in sorted(groups.items()):
        if len(records)!=3:raise ValueError('Incomplete fixed cohort')
        logits=[];errors=[]
        for r in records:
            with np.load(r['artifact'],allow_pickle=False) as z:logits.append(z['logits'].reshape(-1));errors.append(z['absolute_opening_error_mm'])
        l=np.concatenate(logits);e=np.concatenate(errors)
        rows.append(dict(run=run,step=step,cohort=cohort,cases=len(records),frames=len(e),
            saturation_count=sum(r['exact_saturation_count'] for r in records),logit_count=sum(r['logit_count'] for r in records),
            exact_saturation_fraction=sum(r['exact_saturation_count'] for r in records)/sum(r['logit_count'] for r in records),
            logit_min=float(l.min()),logit_p05=float(np.quantile(l,.05)),logit_median=float(np.median(l)),logit_p95=float(np.quantile(l,.95)),logit_max=float(l.max()),
            mean_case_opening_error_mm=float(np.mean([r['opening_error_mm']['mean'] for r in records])),
            frame_weighted_opening_error_mm=float(e.mean()),frame_weighted_opening_p95_mm=float(np.quantile(e,.95)),
            scope='Unrefined pure-SE3 network typed FK. Equal-case mean and frame-weighted tail are explicit. Actual-q final performance is reported separately.'))
    out=dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),rows=rows,matched_fixed_case_inputs=True,
        interpretation='A nonzero stable analytic gradient, lower exact saturation or better opening alone does not establish manipulation recovery. Compare final native-q SE3 and opening metrics under the same solver.')
    a.output.mkdir(parents=True,exist_ok=True);(a.output/'slide_saturation_comparison.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4.2),layout='constrained');colors={'S':'#008f83','M':'#4169bd','L':'#b04475'}
    for run in sorted({r['run'] for r in rows}):
        series=sorted([r for r in rows if r['run']==run and r['cohort']=='strict_limit_valid_control'],key=lambda r:r['step'])
        for ax,key,label,mult in zip(axes,['logit_median','exact_saturation_fraction','mean_case_opening_error_mm'],['Median finger logit','Exactly saturated CUDA FP32 sigmoid (%)','Unrefined opening error (mm)'],[1,100,1]):
            ax.plot([r['step'] for r in series],[r[key]*mult for r in series],color=colors[run[2]],linestyle='-' if run[0]=='A' else '--',marker='o',label=run.replace('_seed0',''))
            ax.set(xlabel='Training updates',ylabel=label);ax.axvline(3000,color='#555555',linewidth=.6);ax.grid(alpha=.2)
    axes[1].set_ylim(-1,101);axes[2].set_ylim(bottom=0);axes[0].legend(ncols=2,fontsize=8)
    fig.suptitle('Franka fixed pure-SE3 cases | Strictly valid reference cohort\nAt step3000: common LR restart + stable sigmoid backward; opening is finger-origin distance, not grasp success',fontsize=11)
    fig.savefig(a.output/'slide_saturation_comparison.png',dpi=150,bbox_inches='tight');plt.close(fig)
    print(json.dumps(dict(rows=len(rows),output=str(a.output))))

if __name__=='__main__':main()
