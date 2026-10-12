"""Remote read-only process/capacity inspection; only writes its own requested receipt.
Never starts, stops, signals, retries or changes a downloader, extractor or training job.
"""
import argparse,datetime,importlib.util,json,os
from pathlib import Path
from pilot_data import write_json
FOUNDATION=Path('/home/arenalabs/Desktop/"be water, robot"/alpha_motion2/pretraining_foundation')
PILOT=Path('/home/arenalabs/Desktop/"be water, robot"/alpha_motion2/research/megatron_scale_pilot_20261004')


def identity(pid):
 p=Path('/proc')/str(pid);out={'pid':int(pid),'exists':p.exists()}
 try:
  stat=(p/'stat').read_text().rsplit(')',1)[1].split();raw=(p/'cmdline').read_bytes()
  out.update(state=stat[0],start_ticks=stat[19],cmdline=raw.replace(b'\0',b' ').decode(),argv=[x.decode() for x in raw.split(b'\0') if x],boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
 except (FileNotFoundError,PermissionError,ProcessLookupError):out['race_or_unreadable']=True
 return out


def capacity():
 spec=importlib.util.spec_from_file_location('pilot_capacity_readonly',FOUNDATION/'acquire.py')
 a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
 primary,repos=a.load_plan(FOUNDATION/'configs/download_plan.json');follow,_=a.load_plan(FOUNDATION/'configs/download_followup_plan.json')
 combined=dict(primary);combined['stages']=primary['stages']+follow['stages'];cap=a.capacity(combined,repos)
 if cap['expected_unique_bytes']!=200505865539:raise ValueError('fixed combined plan bytes changed; inspect before training')
 return cap


def collect():
 cap=capacity();ids=[]
 for pid,ticks,script in [(3258020,'172790670','acquire.py'),(3272054,'172846036','queue_followup.py')]:
  entry=identity(pid);entry['matches_initial_identity']=entry.get('start_ticks')==ticks and str(FOUNDATION/script) in entry.get('argv',[]);ids.append(entry)
 own=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:args=(p/'cmdline').read_bytes().split(b'\0')
  except (FileNotFoundError,PermissionError,ProcessLookupError):continue
  # Exact script path/name predicates, not broad substring process-group assumptions.
  if any(x.decode(errors='replace').startswith(str(PILOT)+'/') and x.endswith((b'extract_xwbc_pilot.py',b'pilot_adapter.py',b'train_pilot.py')) for x in args):own.append(identity(int(p.name)))
 now=datetime.datetime.now(datetime.timezone.utc)
 return {'checked_at_utc':now.isoformat(),'remaining_combined_bytes':cap['remaining_bytes'],'complete_blob_bytes':cap['complete_blob_bytes'],
  'partial_blob_bytes':cap['partial_blob_bytes'],'free_home_bytes':cap['available_bytes'],'required_available_bytes':cap['required_available_bytes'],
  'fits_with_reserve':cap['fits'],'combined_expected_unique_bytes':cap['expected_unique_bytes'],'initial_download_identities':ids,'pilot_processes':own,
  'original_pids_are_not_authority':'Discover current task-owned identities and recorded exit state before any action.',
  'process_actions':[],'archive_extraction':'not included; separately budget before extraction'}

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
 data=collect();args.output.parent.mkdir(parents=True,exist_ok=True);write_json(args.output,data);print(json.dumps(data,indent=2))
