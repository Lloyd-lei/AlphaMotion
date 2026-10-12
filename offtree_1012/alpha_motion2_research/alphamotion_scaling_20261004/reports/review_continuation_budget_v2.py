"""Read-only continuation budget after completed P0/P1; never launches training."""
from pathlib import Path
import argparse,datetime,hashlib,importlib.util,json,math,os

ORDER=[f'{v}_{s}_seed0' for s in 'SML' for v in 'AB'];GiB=1024**3
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=a.root
    old=read(r/'logs/evaluation_queue/state.json');p1=read(r/'evaluation_p1/logs/state.json');ex=read(r/'evaluation_p1/logs/exit_status.json')
    if old['status']!='completed' or len(old['jobs'])!=18 or p1['status']!='completed' or len(p1['runs'])!=6 or ex['returncode']!=0:
        raise RuntimeError('Initial P0 and supplemental P1 must finish')
    base=r.parents[1]/'pretraining_foundation';spec=importlib.util.spec_from_file_location('budget_capacity',base/'acquire.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    plan,repos=m.load_plan(base/'configs/download_plan.json');f,other=m.load_plan(base/'configs/download_followup_plan.json');assert repos==other
    plan['stages']+=f['stages'];v=os.statvfs('/home');cap=m.capacity(plan,repos,available=v.f_bavail*v.f_frsize)
    assert cap['expected_unique_bytes']==200505865539
    costs=[]
    for name in ORDER:
        d=r/'runs'/name;result=read(d/'result.json');assert result['status']=='completed' and result['steps']==3000
        costs.append(dict(run=name,weight_bytes=max(p.stat().st_size for p in d.glob('checkpoint_step*.pt')),
            full_state_bytes=(d/'checkpoint.pt').stat().st_size,log_bytes_per_step=(d/'train.jsonl').stat().st_size/3000,
            eval_bytes=sum(p.stat().st_size for p in (d/'eval').rglob('*') if p.is_file()),
            training_seconds_per_step=result['train_seconds']/3000,
            training_time_caveat='Observed B-L includes documented CPU probe concurrency' if name=='B_L_seed0' else 'observed phase1 optimizer timing'))
    now=datetime.datetime.now(datetime.timezone.utc);deadline=datetime.datetime(2026,10,5,12,tzinfo=datetime.timezone.utc)
    candidates=[]
    for total,nodes in [(6000,[4500,6000]),(12000,[6000,9000,12000]),(12000,[7500,12000]),(20000,[10000,20000])]:
        extra=total-3000
        serialized=sum(c['weight_bytes']*len(nodes)+c['full_state_bytes']+c['log_bytes_per_step']*extra*1.25+c['eval_bytes'] for c in costs)
        scratch=max(c['full_state_bytes'] for c in costs)+256*1024**2
        output=math.ceil(serialized*1.15+scratch)
        evaluation=5*GiB
        training=sum(c['training_seconds_per_step'] for c in costs)*extra
        seconds=1.5*training+3*3600
        required=cap['required_available_bytes']+output+evaluation
        candidates.append(dict(total_steps=total,additional_steps=extra,common_checkpoint_steps=nodes,
            future_output_allocation_bytes=output,future_eval_allocation_bytes=evaluation,
            serialized_model_Adam_log_eval_bytes=math.ceil(serialized),atomic_full_state_scratch_and_metadata_bytes=scratch,
            required_available_bytes=required,headroom_after_all_reserved_bytes=cap['available_bytes']-required,
            capacity_fits=cap['available_bytes']>=required,measured_training_rate_seconds=training,
            seconds_with_50pct_training_margin_and_3h_eval_QA=seconds,
            seconds_to_delivery_deadline=(deadline-now).total_seconds(),time_fits=seconds<(deadline-now).total_seconds()))
    out=dict(schema='post_P0_P1_common_phase_candidate_budget_v2',utc=now.isoformat(),status='candidate_only_no_launch',capacity=cap,
        preservation='All existing data, downloads, original pilot, six phase1 checkpoints, P0 and P1 results stay on disk. Current free space already excludes them; no deletion/reclamation.',
        future_budget_policy='Finished P0/P1 unused allocations are not permanent reservations. A new combined 5GiB future-evaluation/report budget covers phase2 continuous nodes, final P0/P1 refinement, remaining phase1 videos/reports and atomic scratch. Exact bound must be sealed before launching its follower.',
        costs=costs,candidates=candidates,inputs_sha256={str(p):sha(p) for p in [r/'logs/evaluation_queue/state.json',r/'evaluation_p1/logs/state.json',r/'evaluation_p1/logs/exit_status.json']},
        common_recipe='All six same phase, same checkpoint nodes, retain own Adam/model, explicit new shared NumPy streams and cosine restart. No selective continuation.')
    a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'output':str(a.output),'candidates':candidates}))


if __name__=='__main__':main()
