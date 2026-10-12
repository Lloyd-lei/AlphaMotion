"""Download a fixed, bounded motion subset using HF per-blob locks, two workers."""
import os
os.environ.setdefault('HF_XET_NUM_CONCURRENT_RANGE_GETS','2')
from pathlib import Path
import json,sys,time,concurrent.futures
from huggingface_hub import hf_hub_download
ROOT=Path('/home/arenalabs/Desktop/"be water, robot"/alpha_motion2')
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'pretraining_foundation'))
import acquire as A
plan=json.loads((HERE/'pilot_selection.json').read_text())
assert plan['motion_bytes']+plan['conditioning_bytes']<1024**3
assert plan['revision']=='053e8b4b06b6a8ba717a030e675dd5b7fd5be92e'
cache='/home/arenalabs/.cache/huggingface/hub'
main_state=json.loads((ROOT/'pretraining_foundation/logs/acquire/download_state.json').read_text())
records=[]
def fetch(item):
    p=hf_hub_download(repo_id=plan['repo_id'],repo_type='dataset',revision=plan['revision'],
        filename=item['rfilename'],cache_dir=cache)
    result=dict(**item,path=p,verification=A.verify_file(Path(p),item))
    return result
def save(status):
    state=dict(status=status,revision=plan['revision'],max_workers=2,expected_files=len(plan['files']),
        complete_files=len(records),verified_bytes=sum(x['size'] for x in records),
        primary_stage_at_start=main_state.get('current_stage'),files=records)
    A.atomic_json(HERE/'subset_download_report.json',state)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    futures=[pool.submit(fetch,item) for item in plan['files']]
    for f in concurrent.futures.as_completed(futures):
        records.append(f.result());save('downloading')
        if len(records)%20==0:print(json.dumps(dict(verified=len(records),total=len(plan['files']))),flush=True)
save('verified_complete')
print(json.dumps(dict(status='verified_complete',files=len(records),bytes=sum(x['size'] for x in records))),flush=True)
