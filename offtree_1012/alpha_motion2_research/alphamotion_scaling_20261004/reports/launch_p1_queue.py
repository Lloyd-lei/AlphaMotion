"""Single-launch receipt wrapper for the authorized frozen P1 evaluation."""
from pathlib import Path
import datetime,fcntl,hashlib,json,os,subprocess,sys,time

ROOT=Path(__file__).resolve().parents[1];LOG=ROOT/'evaluation_p1/logs'
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def ident(pid):
    p=Path('/proc')/str(pid);s=(p/'stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=pid,start_ticks=s[19],boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        argv=[x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x])
def write(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)
def main():
    LOG.mkdir(exist_ok=True);lock=(LOG/'launcher.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (LOG/'launch_receipt.json').exists() or (LOG/'state.json').exists():raise RuntimeError('P1 already launched; no duplicate')
    command=[sys.executable,'-u',str(ROOT/'evaluation_p1/run_p1.py')]
    with (LOG/'runner.log').open('w') as stream:
        child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
        write(LOG/'launch_receipt.json',dict(utc=utc(),launcher=ident(os.getpid()),runner=ident(child.pid),
            launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            plan_sha256=hashlib.sha256((ROOT/'evaluation_p1/execution_plan.json').read_bytes()).hexdigest()))
        code=child.wait()
    write(LOG/'exit_status.json',dict(utc=utc(),returncode=code,status='completed' if code==0 else 'failed'))
    return code
if __name__=='__main__':sys.exit(main())
