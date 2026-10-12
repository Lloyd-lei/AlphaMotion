"""Freeze the reviewed common 6000-step phase, with fresh combined capacity."""
from pathlib import Path
import datetime,hashlib,json,subprocess,sys
import train_common_phase as runner

ROOT=Path(__file__).resolve().parents[1];PHASE=ROOT/'continuation_v2'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())

def main():
    target=PHASE/'execution_plan.json'
    if target.exists() or (PHASE/'runs').exists():raise RuntimeError('No overwrite of a frozen or started phase')
    budget_path=PHASE/'capacity_at_freeze.json'
    subprocess.run([sys.executable,str(ROOT/'reports/review_continuation_budget_v2.py'),'--root',str(ROOT),'--output',str(budget_path)],check=True)
    budget=read(budget_path);candidate=next(c for c in budget['candidates'] if c['total_steps']==6000)
    if not candidate['capacity_fits'] or not candidate['time_fits']:raise RuntimeError('Common 6000 stage does not fit capacity/deadline')
    files=['logs/queue_state.json','logs/evaluation_queue/state.json','evaluation_p1/logs/state.json',
        'evaluation_p1/logs/exit_status.json','evaluation_p1/execution_plan.json','evaluation_p1/preregistration.json',
        'reports/native_q_oracle_diagnostic_v1.json','reports/training_timing_context.json',
        'reports/actual_B_S_checkpoint_hidden_observations.json','continuation_v2/capacity_at_freeze.json']
    initial={name:{k:sha(ROOT/'runs'/name/f) for f,k in [('config.json','config_sha256'),('result.json','result_sha256'),('checkpoint.pt','checkpoint_sha256')]} for name in runner.ORDER}
    manifest=read(ROOT/'runs'/runner.ORDER[0]/'result.json')['data_manifest_sha256']
    plan=dict(schema='common_continuation_v2',decision='frozen_after_initial_formal_review',
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),run_order=runner.ORDER,initial_steps=3000,total_steps=6000,
        checkpoint_steps=[4500,6000],future_output_allocation_bytes=candidate['future_output_allocation_bytes'],
        future_eval_allocation_bytes=candidate['future_eval_allocation_bytes'],
        initial_evaluation_state_sha256=sha(ROOT/'logs/evaluation_queue/state.json'),initial_runs=initial,
        data_manifest_sha256=manifest,source_sha256=runner.seal(),
        review_evidence_sha256={f:sha(ROOT/f) for f in files},shared_numpy_seeds=dict(data=1000003,combo=1000007,reenc=1000033),
        latest_training_finish_utc='2026-10-05T04:00:00+00:00',max_phase_training_seconds_per_run=3600,
        decision_basis='All six phase1 runs and all P0/P1 cases complete. Training loss still decreases; heldout curves do not support universal size or B advantage. Common bounded second phase tests further training, without selecting a favorable model.',
        reviewed_findings=dict(adroit='P1 native-q improvement varies with size; no monotonic scaling',franka='All six P0/P1 native-q fail joint 1cm/5deg threshold',oracle='Native q FK and gradient diagnostics pass; retain failed learned-initialization results',locomotion='Eight whole-body heldouts evaluated; three UniMate bodies have one source clip and no reliable episode CI'),
        model_data_training_loss_changes=False,restore='Each own step3000 model and Adam moments',
        scheduler='One common 3000-update cosine LR restart from the original 0.0001 learning rate',
        rng='Shared fresh NumPy streams; no claim of exact uninterrupted continuation because initial NumPy state was not saved',
        evaluation_commitment='Both common nodes use original continuous locomotion cases; final6000 uses original P0 plus frozen P1 manipulation protocols, same cohorts and references, matched visual evidence. No training or evaluation case selection.',
        preservation='Initial3000 weights, data, source and P0/P1 artifacts remain unchanged; no existing download/service process action',
        freezer_sha256=sha(__file__))
    runner.prerequisite(plan);capacity=runner.guard(plan);plan['capacity_before_freeze']=capacity
    target.write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(plan=str(target),sha256=sha(target),capacity=capacity)))

if __name__=='__main__':main()
