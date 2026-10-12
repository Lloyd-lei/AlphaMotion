"""Single launcher for the frozen, CUDA-validated common continuation phase."""
from pathlib import Path
import datetime,fcntl,hashlib,json,os,subprocess,sys

PHASE=Path(__file__).resolve().parent;ROOT=PHASE.parent;LOG=PHASE/'logs'
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ident(pid):
    p=Path('/proc')/str(pid);s=(p/'stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=pid,start_ticks=s[19],boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),argv=[x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x])
def write(p,v):
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(v,indent=2)+'\n');temp.replace(p)
def main():
    LOG.mkdir(exist_ok=True);lock=(LOG/'launcher.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (LOG/'launch_receipt.json').exists() or (PHASE/'runs').exists():raise RuntimeError('Already launched or phase runs exist; no duplicate/retry')
    plan=PHASE/'execution_plan.json';receipt=json.loads((PHASE/'validation_receipt.json').read_text())
    if receipt['status']!='passed' or receipt['plan_sha256']!=sha(plan):raise RuntimeError('Matching real CUDA validation required')
    gpu_lock=(ROOT/'gpu_queue.lock').open('a+');fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(gpu_lock,fcntl.LOCK_UN);gpu_lock.close()
    command=[sys.executable,'-u',str(PHASE/'train_common_phase.py'),'--plan',str(plan),'--action','train']
    with (LOG/'runner.log').open('x') as output:
        child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT)
        write(LOG/'launch_receipt.json',dict(utc=utc(),launcher=ident(os.getpid()),runner=ident(child.pid),plan_sha256=sha(plan),validation_sha256=sha(PHASE/'validation_receipt.json'),launcher_sha256=sha(__file__)))
        code=child.wait()
    write(LOG/'launcher_exit_status.json',dict(utc=utc(),returncode=code,status='completed' if code==0 else 'failed'))
    return code
if __name__=='__main__':sys.exit(main())
