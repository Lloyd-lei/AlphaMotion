"""Reuse original protocols/case selection and measure a conservative I/O bound."""
from pathlib import Path
import copy,math,zipfile
from common import ROOT,PHASE,TRAIN,ORDER,read,sha,write,utc,capacity

def bounded_bytes(folder):
    total=0;files=0
    for p in folder.rglob('*'):
        if not p.is_file():continue
        files+=1
        if p.suffix=='.npz':
            with zipfile.ZipFile(p) as z:total+=sum(x.file_size+4096 for x in z.infolist())+4096
        else:total+=max(4096,p.stat().st_size*2)
    return total,files
def main():
    if (PHASE/'execution_plan.json').exists():raise RuntimeError('No rewrite of frozen evaluation plan')
    train=read(TRAIN/'execution_plan.json');assert train['total_steps']==6000
    costs={};p0_manip=sum(bounded_bytes(ROOT/'evaluation/results/manipulation'/n/'checkpoint_step3000')[0] for n in ORDER)
    p1_manip=sum(bounded_bytes(ROOT/'evaluation_p1/results/manipulation'/n/'checkpoint_step3000')[0] for n in ORDER)
    loco=sum(max(bounded_bytes(ROOT/'evaluation/results/locomotion'/n/f'checkpoint_step{s:04d}')[0] for s in [100,250,500,1000,1500,2000,2500,3000]) for n in ORDER)*2
    refs=bounded_bytes(ROOT/'evaluation/results/locomotion/_references')[0]
    # Additional spare for small indexes, 18 checkpoint diagnostics, copied reference archives, figures and final report/PDF.
    extra=768*1024**2;total=p0_manip+p1_manip+loco+refs+extra
    bound=dict(utc=utc(),status='passed' if total<=5*1024**3 else 'failed',conservative_bound_bytes=total,budget_bytes=5*1024**3,
        components=dict(P0_manipulation=p0_manip,P1_manipulation=p1_manip,two_locomotion_nodes_all_six=loco,shared_references=refs,diagnostics_figures_and_reports=extra),
        method='Uncompressed NPZ payloads plus per-entry headers; at least2x actual other-file sizes; per-run maximum of all eight initial locomotion nodes times2; all references once; extra768MiB. No old artifact deletion.')
    write(PHASE/'capacity_bound.json',bound)
    if bound['status']!='passed':raise RuntimeError('Actual conservative storage bound exceeds5GiB')
    baseline=read(ROOT/'evaluation/execution_plan_io_v2.json');jobs=[]
    for job in baseline['jobs'][:12]:
        job=copy.deepcopy(job);name=job['run'];manip='manipulation' in job['id']
        job['id']=name+('_manipulation6000' if manip else '_locomotion4500_6000')
        for a in job['arguments']:
            if isinstance(a,dict):
                path=a['root_relative']
                if path.startswith('runs/'):path='continuation_v3/'+path
                path=path.replace('evaluation/results/','evaluation_phase2/p0/').replace('checkpoint_step3000','checkpoint_step6000');a['root_relative']=path
        if not manip:job['arguments']+=['--checkpoint',{'root_relative':f'continuation_v3/runs/{name}/checkpoint_step4500.pt'},'--selection',{'root_relative':'evaluation/results/locomotion/fixed_selection.json'}]
        job['completion_file']=job['completion_file'].replace('evaluation/results/','evaluation_phase2/p0/')
        for key in ['step','checkpoint','checkpoints']:job.pop(key,None)
        jobs.append(job)
    jobs.append(dict(id='slide_saturation_all_nodes',runner='evaluation_phase2/slide_diagnostic.py',arguments=[],completion_file='evaluation_phase2/slide_diagnostic/summary.json'))
    jobs.append(dict(id='P1_manipulation6000_all_six',runner='evaluation_phase2/run_p1.py',arguments=[],completion_file='evaluation_phase2/p1/logs/state.json',owns_gpu_lease=True))
    p1=read(ROOT/'evaluation_p1/execution_plan.json');p1['output_total_budget_bytes']=5*1024**3
    p1['status']='frozen_after_synthetic_probe';p1['continuation_scope']='Reuse unchanged original P1 configuration and probe; phase2 P0 learned initializations only'
    pp=PHASE/'p1';(pp/'logs').mkdir(parents=True,exist_ok=True);(pp/'preregistration.json').write_bytes((ROOT/'evaluation_p1/preregistration.json').read_bytes())
    sources=dict(baseline['source_sha256']);sources.update(p1['source_sha256']);sources.update({str(p.relative_to(ROOT)):sha(p) for p in PHASE.glob('*.py')})
    p1['source_sha256']=sources;write(pp/'execution_plan.json',p1)
    inputs=['continuation_v3/execution_plan.json','continuation_v3/validation_receipt.json','evaluation_phase2/capacity_bound.json','evaluation_phase2/p1/execution_plan.json','evaluation_phase2/p1/preregistration.json','evaluation/results/locomotion/fixed_selection.json']
    plan=dict(schema='common_second_phase_evaluation_v1',utc=utc(),run_order=ORDER,common_steps=6000,common_nodes=[4500,6000],
        future_eval_allocation_bytes=5*1024**3,data_manifest_sha256=train['data_manifest_sha256'],jobs=jobs,source_sha256=sources,
        frozen_inputs_sha256={f:sha(ROOT/f) for f in inputs},capacity=capacity(),
        comparison='Same P0 and frozen P1, all original cases/cohorts/frames; phase2 has common LR restart and declared sigmoid backward numerical repair. Cross-phase differences are not solely a step-count effect.',
        process_policy='Wait for existing shared GPU lease; all six training exits required; no signals or restarts')
    write(PHASE/'execution_plan.json',plan);print(dict(plan_sha256=sha(PHASE/'execution_plan.json'),storage=bound,jobs=len(jobs)))
if __name__=='__main__':main()
