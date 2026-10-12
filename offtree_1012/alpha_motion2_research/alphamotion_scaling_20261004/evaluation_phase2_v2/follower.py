"""Wait for the shared GPU lease, then evaluate one sealed common phase once."""
from pathlib import Path
import datetime,fcntl,json,os,subprocess,sys,time,traceback
from phase_common import ROOT,PHASE,TRAIN,ORDER,read,sha,write,utc,capacity,checkpoint_receipt
LOG=PHASE/'logs'
def identity(pid=None):
    pid=os.getpid() if pid is None else pid;p=Path('/proc')/str(pid);s=(p/'stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=pid,start_ticks=s[19],boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),argv=[x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x])
def verify(plan):
    for relative,h in plan['source_sha256'].items():
        if sha(ROOT/relative)!=h:raise RuntimeError('Evaluation source changed '+relative)
    for relative,h in plan['frozen_inputs_sha256'].items():
        if sha(ROOT/relative)!=h:raise RuntimeError('Evaluation input changed '+relative)
    bound=read(PHASE/'capacity_bound.json')
    if bound['conservative_bound_bytes']>plan['future_eval_allocation_bytes']:raise RuntimeError('Declared evaluation storage bound does not fit')
def coverage():
    records=[]
    for name in ORDER:
        for family,steps,source in [('manipulation',[6000],'evaluation/results/manipulation'),('locomotion',[4500,6000],'evaluation/results/locomotion'),('p1',[6000],'evaluation_p1/results/manipulation')]:
            original=read(ROOT/source/name/'checkpoint_step3000/evaluation_summary.json')
            base=PHASE/'p1/results/manipulation' if family=='p1' else PHASE/'p0'/family
            for step in steps:
                folder=base/name/f'checkpoint_step{step:04d}';d=read(folder/'evaluation_summary.json')
                if d['evaluation_set_sha256']!=original['evaluation_set_sha256'] or d['case_count']!=original['case_count']:raise RuntimeError('Original case coverage changed '+str(folder))
                failures=[x['case_id'] for x in d.get('records',[]) if x.get('status')!='measured'] if family=='locomotion' else [read(p)['case_id'] for p in folder.rglob('case_result.json') if read(p).get('status')=='failed_case_retained']
                records.append(dict(run=name,family=family,step=step,case_count=d['case_count'],evaluation_set_sha256=d['evaluation_set_sha256'],failed_case_ids=failures,summary_sha256=sha(folder/'evaluation_summary.json')))
    return dict(records=records,all_original_case_sets_match=True,failed_execution_cases=sum(len(r['failed_case_ids']) for r in records))
def main():
    LOG.mkdir(parents=True,exist_ok=True);lease=(LOG/'follower.lock').open('a+');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (LOG/'state.json').exists():raise RuntimeError('Already attempted: identity and exit review required; no duplicate')
    plan=read(PHASE/'execution_plan.json');verify(plan);state=dict(status='waiting_for_training_gpu_lease',identity=identity(),started_at_utc=utc(),plan_sha256=sha(PHASE/'execution_plan.json'),jobs=[],training_updates=0,process_signals=[])
    write(LOG/'state.json',state);gpu=(ROOT/'gpu_queue.lock').open('a+');status='failed'
    try:
        while True:
            try:fcntl.flock(gpu,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:
                state.update(updated_at_utc=utc(),capacity=capacity());write(LOG/'state.json',state)
                if datetime.datetime.now(datetime.timezone.utc)>=datetime.datetime(2026,10,5,8,tzinfo=datetime.timezone.utc):raise RuntimeError('Evaluation wait deadline reached')
                time.sleep(30)
        if read(TRAIN/'logs/queue_state.json')['status']!='six_runs_completed':raise RuntimeError('Training lease released before all six phase2 runs completed')
        for name in ORDER:
            for step in [4500,6000]:checkpoint_receipt(name,step)
        state['status']='evaluating';write(LOG/'state.json',state)
        for index,job in enumerate(plan['jobs']):
            verify(plan);cap=capacity();command=[sys.executable,'-u',str(ROOT/job['runner'])]+[str(ROOT/x['root_relative']) if isinstance(x,dict) else str(x) for x in job.get('arguments',[])]
            if job.get('owns_gpu_lease'):fcntl.flock(gpu,fcntl.LOCK_UN)
            record=dict(id=job['id'],status='running',started_at_utc=utc(),command=command,capacity=cap);state['jobs'].append(record);state['current_job']=job['id'];write(LOG/'state.json',state)
            with (LOG/f'{index:03d}_{job["id"]}.log').open('x') as stream:
                child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
                record['identity']=identity(child.pid);write(LOG/'state.json',state);rc=child.wait()
            if job.get('owns_gpu_lease'):fcntl.flock(gpu,fcntl.LOCK_EX|fcntl.LOCK_NB)
            record.update(returncode=rc,finished_at_utc=utc(),status='completed' if rc==0 else 'failed');receipt=ROOT/job['completion_file']
            if rc==0 and receipt.exists():record['completion_sha256']=sha(receipt)
            else:record['status']='failed'
            write(LOG/'state.json',state)
            if record['status']!='completed':raise RuntimeError('Evaluation job failed; retained output requires review: '+job['id'])
        result=coverage();write(PHASE/'coverage_receipt.json',result)
        status='completed' if result['failed_execution_cases']==0 else 'completed_with_case_failures'
        state.update(status=status,finished_at_utc=utc(),coverage=result);write(LOG/'state.json',state)
    except Exception as exc:
        state.update(status='failed',error=repr(exc),traceback=traceback.format_exc(),finished_at_utc=utc());write(LOG/'state.json',state);raise
    finally:
        write(LOG/'exit_status.json',dict(status=status,utc=utc(),identity=identity(),training_updates=0));gpu.close()
if __name__=='__main__':main()
