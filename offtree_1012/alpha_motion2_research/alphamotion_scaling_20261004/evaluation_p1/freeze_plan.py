"""Seal supplemental P1 after preregistered probes and all original jobs finish."""
from pathlib import Path
import hashlib,json,math,os,zipfile
from protocol import ROOT
from run_p1 import ORDER,capacity


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    phase=ROOT/'evaluation_p1';target=phase/'execution_plan.json'
    if target.exists():raise RuntimeError('P1 execution plan already frozen')
    state=json.loads((ROOT/'logs/evaluation_queue/state.json').read_text())
    if state['status']!='completed' or len(state['jobs'])!=18 or any(j['status']!='completed' for j in state['jobs']):
        raise RuntimeError('Original evaluations still incomplete')
    probe=json.loads((phase/'synthetic_probe_receipt.json').read_text())
    if probe['status']!='passed':raise RuntimeError('P1 synthetic probe failed')
    original=json.loads((ROOT/'evaluation/execution_plan_io_v2.json').read_text())
    code=dict(original['source_sha256'])
    for p in sorted(phase.glob('*.py')):code[str(p.relative_to(ROOT))]=sha(p)
    inputs={}
    for p in [ROOT/'logs/evaluation_queue/state.json',phase/'preregistration.json',phase/'synthetic_probe_receipt.json',
              ROOT/'reports/native_q_oracle_diagnostic_v1.json',ROOT/'data_manifest.json']:
        inputs[str(p.relative_to(ROOT))]=sha(p)
    payload=ROOT/'evaluation/results/manipulation';counts={};plain=0;expanded=0;other=0;ref=0
    for name in ORDER:
        folder=payload/name/'checkpoint_step3000';summary=json.loads((folder/'evaluation_summary.json').read_text())
        if summary['failed_case_ids'] or summary['case_count']!=12:raise RuntimeError('P0 input initialization coverage needs review')
        files=[folder/'evaluation_summary.json']+sorted(folder.rglob('case_result.json'))+sorted(folder.rglob('case_arrays.npz'))
        for p in files:inputs[str(p.relative_to(ROOT))]=sha(p)
        counts[name]=summary['case_count']
        for p in folder.rglob('*'):
            if not p.is_file():continue
            if p.name=='case_arrays.npz':ref+=p.stat().st_size
            elif p.suffix=='.npz':
                with zipfile.ZipFile(p) as z:expanded+=sum(v.file_size for v in z.infolist())+4096
            elif p.suffix in ('.json','.csv'):plain+=p.stat().st_size
            else:other+=p.stat().st_size
    bound=math.ceil(plain*2+expanded*1.25+other*2)+256*1024**2
    budget=int(1.5*1024**3)
    if bound>budget:raise RuntimeError('P1 conservative output estimate exceeds reservation')
    plan=dict(schema='P1_six_model_supplement_execution_v1',status='frozen_after_synthetic_probe',run_order=ORDER,
        output_total_budget_bytes=budget,source_sha256=code,input_sha256=inputs,case_counts=counts,
        adapter_receipts={'adroit_pen':'adroit_dof_validation.json','relay_franka_full_v2':'relay_franka_full_v2_dof_validation.json'},
        storage_evidence=dict(measured_P0_json_csv_bytes=plain,measured_P0_terminal_arrays_expanded_bytes=expanded,
            measured_P0_other_bytes=other,shared_P0_reference_and_initialization_bytes=ref,
            conservative_P1_bound_bytes=bound,extra_reports_metadata_and_atomic_allowance_bytes=256*1024**2,
            formula='2x P0 JSON/CSV,1.25x expanded terminal NPZ+headers,2x P0 PNG/other,256MiB reports/atomic scratch. Identical case_arrays hardlink P0.'),
        decision='Supplemental protocol, not bug correction; exact original P0 retained, no model/optimizer weight updates')
    plan['capacity_at_freeze']=capacity(plan)
    target.write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps({'plan':str(target),'sha256':sha(target),'cases':counts,'storage':plan['storage_evidence'],
        'capacity':plan['capacity_at_freeze']}))


if __name__=='__main__':main()
