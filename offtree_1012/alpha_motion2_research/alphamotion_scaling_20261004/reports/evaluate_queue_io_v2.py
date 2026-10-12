"""Version2 evaluation follower with measured shared-reference storage budget.

It never starts or resumes training, signals another process, changes a model,
or modifies sealed data/configuration. The root freezes execution_plan.json
after real CPU integration checks before launching this follower.
"""
from pathlib import Path
import datetime,fcntl,hashlib,importlib.util,json,os,subprocess,sys,time,traceback

ROOT=Path(__file__).resolve().parents[1]
LOG=ROOT/'logs/evaluation_queue'

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)

def identity(pid=None):
    pid=os.getpid() if pid is None else pid;p=Path('/proc')/str(pid)
    try:
        stat=(p/'stat').read_text().rsplit(')',1)[1].split()
        return {'pid':pid,'start_ticks':stat[19],'state':stat[0],'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'argv':[v.decode() for v in (p/'cmdline').read_bytes().split(b'\0') if v]}
    except FileNotFoundError:return None

def capacity(reserve_extra):
    base=ROOT.parents[1]/'pretraining_foundation';spec=importlib.util.spec_from_file_location('eval_capacity',base/'acquire.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    p,repos=m.load_plan(base/'configs/download_plan.json');f,_=m.load_plan(base/'configs/download_followup_plan.json');p['stages']=p['stages']+f['stages']
    v=os.statvfs('/home');out=m.capacity(p,repos,available=v.f_bavail*v.f_frsize)
    if out['expected_unique_bytes']!=200505865539:raise RuntimeError('fixed authorized download plan changed')
    allocated=sum(p.stat().st_size for p in (ROOT/'evaluation/results').rglob('*') if p.is_file())
    if allocated>reserve_extra:raise RuntimeError('evaluation allocation exceeded measured total budget')
    remaining=max(0,reserve_extra-allocated)
    out['evaluation_total_allocation_bytes']=reserve_extra;out['evaluation_allocated_bytes']=allocated
    out['evaluation_future_headroom_bytes']=remaining;out['required_with_eval_bytes']=out['required_available_bytes']+remaining
    if out['available_bytes']<out['required_with_eval_bytes']:raise RuntimeError('insufficient capacity for downloads,16GiB protection and evaluation headroom')
    return out

def verify_sources(plan):
    evidence=plan['storage_capacity_receipt']
    if sha(ROOT/evidence['root_relative'])!=evidence['sha256']:raise RuntimeError('reviewed storage capacity receipt changed')
    receipt=json.loads((ROOT/evidence['root_relative']).read_text())
    if receipt['status']!='passed' or receipt['conservative_total_bound_bytes']>plan['future_eval_allocation_bytes']:raise RuntimeError('storage capacity not validated')
    for rel,expected in plan['source_sha256'].items():
        if sha(ROOT/rel)!=expected:raise RuntimeError('evaluation/training source seal changed: '+rel)

def check_training_complete(plan):
    queue=json.loads((ROOT/'logs/queue_state.json').read_text())
    if queue.get('status')!='six_runs_completed':raise RuntimeError('GPU lease available but six training runs did not complete; inspect exit state, no implicit retry')
    evidence={}
    for run in plan['run_order']:
        result=json.loads((ROOT/'runs'/run/'result.json').read_text());exit_status=json.loads((ROOT/'runs'/run/'exit_status.json').read_text())
        if result.get('status')!='completed' or result['steps']!=plan['common_steps'] or exit_status['status']!='completed':raise RuntimeError('incomplete training '+run)
        if result['data_manifest_sha256']!=plan['data_manifest_sha256']:raise RuntimeError('training data changed '+run)
        evidence[run]=result
    return evidence

def main():
    LOG.mkdir(parents=True,exist_ok=True)
    follower=(LOG/'follower.lock').open('a+');fcntl.flock(follower,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan_path=ROOT/'evaluation/execution_plan_io_v2.json';plan=json.loads(plan_path.read_text());plan_sha=sha(plan_path)
    if (LOG/'state.json').exists():
        old=json.loads((LOG/'state.json').read_text())
        if old.get('status')!='completed':raise RuntimeError('prior evaluation queue state needs explicit identity/exit review; no implicit restart')
        if old.get('plan_sha256')==plan_sha:raise RuntimeError('this evaluation plan already completed')
    state={'started_at_utc':now(),'identity':identity(),'plan_sha256':plan_sha,'status':'waiting_for_training_gpu_lease','jobs':[],'process_signals':[],'training_updates':0}
    write(LOG/'state.json',state);gpu=(ROOT/'gpu_queue.lock').open('a+');status='failed'
    try:
        while True:
            try:fcntl.flock(gpu,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:
                state['updated_at_utc']=now();state['capacity']=capacity(plan['future_eval_allocation_bytes']);write(LOG/'state.json',state);time.sleep(30)
        verify_sources(plan);training=check_training_complete(plan);state['status']='evaluating';write(LOG/'state.json',state)
        for index,job in enumerate(plan['jobs']):
            verify_sources(plan);cap=capacity(plan['future_eval_allocation_bytes'])
            run=job['run'];checkpoint_receipts=[]
            for cp in job.get('checkpoints',[{'path':job.get('checkpoint'),'step':job.get('step')}]):
                checkpoint=ROOT/cp['path'];step=cp['step']
                declared=next((r for r in training[run]['intermediate_checkpoints'] if r['step']==step),None)
                if declared is None or sha(checkpoint)!=declared['sha256']:raise RuntimeError('checkpoint identity differs from completed training receipt')
                checkpoint_receipts.append({'path':str(checkpoint),'step':step,'sha256':declared['sha256']})
            log=LOG/f'{index:03d}_{job["id"]}.log';command=[sys.executable,'-u',str(ROOT/job['runner'])]+[str(ROOT/x['root_relative']) if isinstance(x,dict) else str(x) for x in job['arguments']]
            record={'id':job['id'],'status':'running','started_at_utc':now(),'command':command,'checkpoints':checkpoint_receipts,'capacity':cap,'log':str(log)}
            state['jobs'].append(record);state['current_job']=job['id'];write(LOG/'state.json',state)
            with log.open('w') as out:
                child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT)
                record['identity']=identity(child.pid);write(LOG/'state.json',state);rc=child.wait()
            record.update(returncode=rc,finished_at_utc=now(),status='completed' if rc==0 else 'failed')
            completed=ROOT/job['completion_file']
            if rc==0 and not completed.is_file():record.update(status='failed',failure_reason='runner returned zero without required completion receipt')
            if completed.is_file():
                snapshot=LOG/f'{index:03d}_{job["id"]}_completion.json';snapshot.write_bytes(completed.read_bytes())
                record['completion_receipt']={'path':str(snapshot),'source':str(completed),'sha256':sha(snapshot)}
            write(LOG/'state.json',state)
        status='completed' if all(r['status']=='completed' for r in state['jobs']) else 'completed_with_failures'
        state.update(status=status,finished_at_utc=now());write(LOG/'state.json',state)
    except Exception as e:
        state.update(status='failed',error=repr(e),traceback=traceback.format_exc(),finished_at_utc=now());write(LOG/'state.json',state);raise
    finally:
        write(LOG/'exit_status.json',{'utc':now(),'status':status,'identity':identity(),'plan_sha256':plan_sha,'training_updates':0});gpu.close()
    if status!='completed':raise SystemExit(1)

if __name__=='__main__':main()
