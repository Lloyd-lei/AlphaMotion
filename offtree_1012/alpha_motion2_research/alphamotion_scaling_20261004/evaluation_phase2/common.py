"""Shared storage/lineage guard for the bounded second-phase evaluation."""
from pathlib import Path
import datetime,hashlib,importlib.util,json,os
ROOT=Path(__file__).resolve().parents[1];PHASE=ROOT/'evaluation_phase2';TRAIN=ROOT/'continuation_v3'
ORDER=[f'{v}_{s}_seed0' for s in 'SML' for v in 'AB']
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(p,v):
    p=Path(p);p.parent.mkdir(exist_ok=True,parents=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)
def capacity(budget=5*1024**3):
    base=ROOT.parents[1]/'pretraining_foundation';spec=importlib.util.spec_from_file_location('phase2_capacity',base/'acquire.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    p,repos=m.load_plan(base/'configs/download_plan.json');f,other=m.load_plan(base/'configs/download_followup_plan.json');assert repos==other;p['stages']+=f['stages']
    v=os.statvfs('/home');cap=m.capacity(p,repos,available=v.f_bavail*v.f_frsize)
    if cap['expected_unique_bytes']!=200505865539:raise RuntimeError('Fixed download plan changed')
    allocated=sum(p.stat().st_size for p in PHASE.rglob('*') if p.is_file())
    if allocated>budget:raise RuntimeError('phase2 evaluation/report budget exceeded')
    remaining=budget-allocated;required=cap['required_available_bytes']+remaining
    if cap['available_bytes']<required:raise RuntimeError('Insufficient room for downloads+16GiB+remaining phase2 evaluation')
    return dict(**cap,phase2_evaluation_allocated_bytes=allocated,phase2_evaluation_remaining_bytes=remaining,required_with_eval_bytes=required)
def checkpoint_receipt(name,step):
    result=read(TRAIN/'runs'/name/'result.json');ex=read(TRAIN/'runs'/name/'exit_status.json')
    if result['status']!='completed' or result['steps']!=6000 or ex['status']!='completed':raise RuntimeError('Phase2 run did not complete '+name)
    receipt=next(r for r in result['intermediate_checkpoints'] if r['step']==step)
    path=TRAIN/'runs'/name/f'checkpoint_step{step:04d}.pt'
    if sha(path)!=receipt['sha256']:raise RuntimeError('Phase2 checkpoint changed '+str(path))
    return receipt
