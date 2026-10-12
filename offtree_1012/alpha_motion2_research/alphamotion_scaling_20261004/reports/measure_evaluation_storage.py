"""Measured storage bound for unchanged six-run evaluation, not a performance report."""
from pathlib import Path
import json,zipfile,hashlib,datetime,os,importlib.util,math
R=Path(__file__).resolve().parents[1]
G=1024**3
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def filesizes(root):
 files=[p for p in root.rglob('*') if p.is_file()]
 suffix={s:sum(p.stat().st_size for p in files if p.suffix==s) for s in {p.suffix for p in files}}
 return dict(total=sum(suffix.values()),suffix=suffix,npz_expanded=sum(sum(x.file_size for x in zipfile.ZipFile(p).infolist()) for p in files if p.suffix=='.npz'),npz_count=sum(p.suffix=='.npz' for p in files),largest=max(p.stat().st_size for p in files))
def main():
 paths={k:R/'evaluation'/('capacity_probe_cpu_'+k+'_A_S_step3000_v2') for k in ['manipulation','locomotion']}
 for p in paths.values():
  assert read(p/'exit_status.json')['returncode']==0
 roundtrip=R/'evaluation/storage_roundtrip_probe_v2/numeric_roundtrip_receipt.json';rt=read(roundtrip)
 assert rt['status']=='passed' and rt['cases']==246
 measured={k:filesizes(p) for k,p in paths.items()}
 cases={}
 for k,p in paths.items():
  rows=[read(q) for q in p.rglob('case_result.json')]
  frames=[r.get('frames',r.get('frame_count')) for r in rows]
  assert all(isinstance(n,int) and 0<n<=64 for n in frames)
  cases[k]={'cases':len(rows),'minimum_frames':min(frames),'maximum_frames':max(frames),'total_frame_occurrences':sum(frames)}
 assert cases['manipulation']['cases']==12 and cases['locomotion']['cases']==246
 for p in paths['manipulation'].rglob('refinement_curve.json'):
  v=read(p);assert [x['step'] for x in v['records']]==list(range(101))
 m,l=measured['manipulation'],measured['locomotion']
 manip_bound=2*m['suffix']['.json']+m['npz_expanded']+4096*m['npz_count']+2*m['suffix']['.png']
 loco_bound=rt['dynamic']['expanded_bytes']+4096*rt['dynamic']['files']+2*sum(l['suffix'].get(s,0) for s in ['.json','.png','.csv'])
 references=rt['shared_references']['expanded_bytes']+4096*rt['shared_references']['files']
 extra={'completion_receipt_copies':256*1024**2,'atomic_temporary_files':64*1024**2,'plots_videos_and_document_exports':G,
    'possible_failure_tracebacks':16*1024*cases['locomotion']['cases']*48}
 bound=6*manip_bound+48*loco_bound+references+sum(extra.values())
 observed_v1=6*m['total']+48*l['total']
 observed_v2=6*m['total']+48*(rt['dynamic']['compressed_bytes']+l['total']-l['suffix']['.npz'])+rt['shared_references']['compressed_bytes']
 source=R.parents[1]/'pretraining_foundation';spec=importlib.util.spec_from_file_location('capacity',source/'acquire.py');ac=importlib.util.module_from_spec(spec);spec.loader.exec_module(ac)
 p,repos=ac.load_plan(source/'configs/download_plan.json');f,_=ac.load_plan(source/'configs/download_followup_plan.json');p['stages']+=f['stages']
 v=os.statvfs('/home');cap=ac.capacity(p,repos,available=v.f_bavail*v.f_frsize);assert cap['expected_unique_bytes']==200505865539
 budget=7*G
 out={'schema':'measured_eval_io_capacity_v2','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'passed' if bound<=budget and cap['available_bytes']>=cap['required_available_bytes']+budget else 'failed',
  'all_formal_jobs_started':0,'probe_actual_cases_and_frames':cases,'probe_refinement_points':101,'probe_refinement_stages':4,
  'probe_scope':'Real A-S checkpoint3000 on CPU; all fixed trajectory frames and refinement points. Bootstrap100 changes interval sampling, not stored arrays; formal bootstrap remains1000.',
  'measured':measured,'node_counts':{'manipulation':6,'locomotion':48},'observed_v1_projection_bytes':observed_v1,'observed_v2_projection_bytes':observed_v2,
  'bound_basis':'2x measured JSON/CSV/PNG, expanded NPZ bytes plus4096B/file, each reference array set once, explicit copy/temp/presentation/failure allowances. Fixed shapes and dtype established from full246-case numeric roundtrip.',
  'manipulation_node_bound_bytes':manip_bound,'locomotion_dynamic_node_bound_bytes':loco_bound,'shared_reference_bound_bytes':references,
  'additional_allowances_bytes':extra,'conservative_total_bound_bytes':bound,'future_eval_total_budget_bytes':budget,'download_capacity':cap,
  'headroom_after_download_reserve_and_eval_budget_bytes':cap['available_bytes']-cap['required_available_bytes']-budget,
  'roundtrip_receipt':{'path':str(roundtrip),'sha256':sha(roundtrip),'arrays_equal':True},
  'prediction_optimization_changes':False,'state_trace_scope':'Every0..100 metric sample retained; existing manipulation terminal state/FK arrays remain NPZ. No claim that original evaluator saved every intermediate full q tensor.',
  'training_updates':0}
 target=R/'reports/evaluation_storage_capacity_v2.json';target.write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps({k:v for k,v in out.items() if k not in ['measured','roundtrip_receipt','state_trace_scope']}))
 assert out['status']=='passed'
if __name__=='__main__':main()
