"""Read-only budget evidence for a possible common continuation, never a launcher."""
from pathlib import Path
import argparse,datetime,hashlib,json,math,os,importlib.util

ORDER=['A_S_seed0','B_S_seed0','A_M_seed0','B_M_seed0','A_L_seed0','B_L_seed0']
GiB=1024**3

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 now=datetime.datetime.now(datetime.timezone.utc);deadline=datetime.datetime(2026,10,5,12,tzinfo=datetime.timezone.utc)
 live=a.root/'reports/live_results.json';s=json.loads(live.read_text())
 base=a.root.parents[1]/'pretraining_foundation'
 spec=importlib.util.spec_from_file_location('continuation_capacity',base/'acquire.py');ac=importlib.util.module_from_spec(spec);spec.loader.exec_module(ac)
 plan,repos=ac.load_plan(base/'configs/download_plan.json');follow,fr=ac.load_plan(base/'configs/download_followup_plan.json')
 assert repos==fr;plan['stages']+=follow['stages'];v=os.statvfs('/home');cap=ac.capacity(plan,repos,available=v.f_bavail*v.f_frsize)
 assert cap['expected_unique_bytes']==200505865539
 rows=[];ready=True
 for name in ORDER:
  r=s['runs'][name];run=a.root/'runs'/name
  ready &= r.get('status')=='completed' and r.get('exit',{}).get('status')=='completed' and r.get('steps')==3000
  cfg=json.loads((run/'config.json').read_text()) if (run/'config.json').exists() else None
  weights=list(run.glob('checkpoint_step*.pt'))
  weight_size=max((x.stat().st_size for x in weights),default=0)
  final=r.get('result',{}).get('checkpoint',{}).get('bytes',3*weight_size)
  step_cost=r.get('training_seconds',0)/max(r.get('steps',0),1)
  log_cost=(run/'train.jsonl').stat().st_size/max(r.get('steps',0),1) if (run/'train.jsonl').exists() else 0
  rows.append(dict(run=name,status=r['status'],observed_steps=r.get('steps'),optimizer_seconds_per_step=step_cost,
    weights_node_bytes=weight_size,full_checkpoint_bytes=final,log_bytes_per_step=log_cost,
    final_checkpoint_size_is_estimate='result' not in r,learning_rate_at_phase_boundary=0 if r.get('status')=='completed' else None))
 evalstate=json.loads((a.root/'logs/evaluation_queue/state.json').read_text())
 cases=[]
 for total,nodes in [(6000,[4500,6000]),(12000,[6000,9000,12000]),(20000,[6000,10000,15000,20000])]:
  added=total-3000
  training=sum(r['optimizer_seconds_per_step'] for r in rows)*added
  files=sum(r['weights_node_bytes']*len(nodes)+r['full_checkpoint_bytes']+r['log_bytes_per_step']*added for r in rows)
  storage=math.ceil(files*1.10)+GiB
  future_eval=5*GiB
  protected=cap['required_available_bytes']
  # Conservative reservation: current training's remaining output is not yet included.
  pending_initial=sum((r['weights_node_bytes']*8+r['full_checkpoint_bytes']) for r in rows if r['status']!='completed')
  projected=training*1.5+3*3600
  cases.append(dict(total_global_optimizer_steps=total,additional_steps_per_run=added,common_checkpoint_steps=nodes,
    measured_rate_projection_seconds=training,projection_with_50pct_training_margin_and_3h_eval_qa_seconds=projected,
    seconds_until_delivery_deadline=(deadline-now).total_seconds(),time_fits=projected<(deadline-now).total_seconds(),
    future_continuation_allocation_bytes=storage,future_eval_allocation_bytes=future_eval,
    conservative_pending_initial_output_bytes=pending_initial,
    required_available_bytes=protected+storage+future_eval+pending_initial,
    capacity_fits=cap['available_bytes']>=protected+storage+future_eval+pending_initial))
 out=dict(checked_at_utc=now.isoformat(),status='candidate_budget_only_no_launch',all_six_initial_training_completed=ready,
    formal_evaluation_queue_status=evalstate['status'],formal_job_count=len(evalstate.get('jobs',[])),
    decision_ready=ready and evalstate['status']=='completed',live_results_sha256=digest(live),live_results_timestamp=s['utc'],
    capacity=cap,run_costs=rows,candidates=cases,
    continuation_contract=['All six or none under a separately versioned common schedule; retain the initial milestone',
      'Load each own 3000-step model and Adam state; never use the earlier AlphaMotion warmstart pilot',
      'Original final checkpoint lacks NumPy RNG states; any fresh shared phase streams must be explicit, not called exact stochastic resume',
      'If restarting cosine learning rate, declare the same restart and duration for all six; do not claim uninterrupted original cosine',
      'Keep data/splits/losses/model source fixed and verify manifest, config, checkpoint and source hashes',
      'Record global and phase steps, phase sampling hashes, valid loss denominators, optimizer time and estimated FLOPs',
      'Use heldout evaluations as repeatedly observed diagnostic/validation data, not an untouched final test',
      'No source edit, process action, model load, optimizer update or training launch performed by this budget report'])
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
 print(json.dumps({k:out[k] for k in ['status','all_six_initial_training_completed','formal_evaluation_queue_status','decision_ready','candidates']}))
if __name__=='__main__':main()
