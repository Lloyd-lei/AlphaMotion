"""Finite report queue after successful fixed evaluation; never performs training."""
from pathlib import Path
import datetime,fcntl,hashlib,json,os,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[1];PHASE=ROOT/'evaluation_phase2_v2'
sys.path.insert(0,str(PHASE))
from phase_common import capacity,write,read,utc

def identity():
    p=Path('/proc')/str(os.getpid());s=(p/'stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=os.getpid(),start_ticks=s[19],boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),argv=[x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x])

def main():
    out=PHASE/'reports';logs=out/'logs';logs.mkdir(parents=True,exist_ok=True)
    lock=(logs/'postprocess.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (logs/'state.json').exists():raise RuntimeError('Existing report attempt; inspect its identity/exit before any retry')
    source_names=['postprocess_phase2.py','collect_two_phase_results.py','collect_live_results.py','summarize_two_phase_locomotion.py','summarize_two_phase_manipulation.py','summarize_manipulation.py','summarize_slide_saturation.py','render_locomotion_pairs.py','render_manipulation_pairs.py']
    sources={name:hashlib.sha256((ROOT/'reports'/name).read_bytes()).hexdigest() for name in source_names}
    state=dict(status='waiting_for_fixed_evaluation',started_at_utc=utc(),identity=identity(),jobs=[],training_updates=0,model_inference_calls=0,source_sha256=sources)
    write(logs/'state.json',state)
    try:
        while True:
            p=PHASE/'logs/exit_status.json'
            if p.exists():
                d=read(p)
                if d['status'] not in ['completed','completed_with_case_failures']:raise RuntimeError('Evaluation failed; automatic reporting stopped for review')
                break
            state['last_checked_at_utc']=utc();write(logs/'state.json',state)
            if datetime.datetime.now(datetime.timezone.utc)>=datetime.datetime(2026,10,5,9,tzinfo=datetime.timezone.utc):raise RuntimeError('Finite report wait deadline reached')
            time.sleep(30)
        summary=read(PHASE/'logs/state.json')
        if len(summary['jobs'])!=14 or any(x['status']!='completed' for x in summary['jobs']):raise RuntimeError('Incomplete fixed evaluation jobs')
        coverage=read(PHASE/'coverage_receipt.json')
        if not coverage['all_original_case_sets_match']:raise RuntimeError('Fixed case identities did not match')
        base=[sys.executable,'-u'];jobs=[
            ('training_records','collect_two_phase_results.py',['--root',str(ROOT),'--output',str(ROOT/'reports/two_phase_results.json'),'--plot']),
            ('locomotion','summarize_two_phase_locomotion.py',['--root',str(ROOT),'--output',str(out/'locomotion')]),
            ('manipulation','summarize_two_phase_manipulation.py',['--root',str(ROOT),'--output',str(out/'manipulation')]),
            ('slide_diagnostic','summarize_slide_saturation.py',['--root',str(ROOT),'--output',str(out/'slide')]),
            ('locomotion_visuals','render_locomotion_pairs.py',['--root',str(PHASE/'p0/locomotion'),'--output',str(out/'locomotion_visuals'),'--step','6000','--video',
                '--target','unitree_g1_29dof','--target','kuavo_s45','--target','e457402d4b2e4681b5a5f88b747aea4a','--target','79fed889082e4a25ad92be75b57f641c']),
        ]
        for protocol,folder,other in [('P0','p0/manipulation','p1/results/manipulation'),('P1','p1/results/manipulation','p0/manipulation')]:
            for detail in [False,True]:
                args=['--root',str(ROOT),'--results-root',str(PHASE/folder),'--comparison-results-root',str(PHASE/other),
                    '--output',str(out/f'manipulation_visuals_{protocol}'),'--step','6000','--protocol-label',protocol,
                    '--body','adroit_pen','--body','relay_franka_full_v2']
                if detail:args.append('--detail')
                jobs.append((protocol+('_detail' if detail else '_full'),'render_manipulation_pairs.py',args))
        state.update(status='processing',evaluation_coverage=coverage);write(logs/'state.json',state)
        env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        for name,script,args in jobs:
            if any(hashlib.sha256((ROOT/'reports'/n).read_bytes()).hexdigest()!=h for n,h in sources.items()):raise RuntimeError('Frozen report source changed during wait or processing')
            cap=capacity();record=dict(id=name,status='running',started_at_utc=utc(),capacity=cap);state['jobs'].append(record);write(logs/'state.json',state)
            with (logs/(name+'.log')).open('x') as stream:
                command=base+[str(ROOT/'reports'/script)]+args;record['command']=command;write(logs/'state.json',state)
                result=subprocess.run(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,timeout=1800)
            record.update(status='completed' if result.returncode==0 else 'failed',returncode=result.returncode,finished_at_utc=utc());write(logs/'state.json',state)
            if result.returncode:raise RuntimeError('Report job failed: '+name)
        state.update(status='artifacts_ready_visual_review_pending',finished_at_utc=utc(),capacity=capacity());write(logs/'state.json',state)
    except Exception as exc:
        state.update(status='failed',error=repr(exc),traceback=traceback.format_exc());write(logs/'state.json',state);raise
    finally:write(logs/'exit_status.json',dict(status=state['status'],identity=identity(),finished_at_utc=utc(),training_updates=0))

if __name__=='__main__':main()
