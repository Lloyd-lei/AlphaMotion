"""Hash-bound real-validation gate. User config readiness flags never waive it."""
from pathlib import Path
import hashlib,json,platform,sys,math

REFERENCE_CONFIG_SHA256='0e61ed5b012420d5fa186bc09537e9cac32ed14fa4533182c6d44805f5e7ee58'
REFERENCE_CONFIG=Path(__file__).with_name('reference_recipe')/'L_M20k_0903.config.json'
IMPLEMENTATION_REVISION='restored_single_frame_recipe_v2'
ONLINE_REQUIREMENTS=['Hash-check actual original run config/checkpoint and pinned text vectors','Validate completed XWBC pairing, source-revalidated sole/contact/ground seal and all-frame FK','Validate UniMate payload and valid-only modality masks','Run all restored losses forward/backward on real CUDA payloads without optimizer updates','Persist current verified 16-body auxiliary-judge label map; head reinitialized, old labels not guessed']


def sha256(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()


def reference_config():
 if sha256(REFERENCE_CONFIG)!=REFERENCE_CONFIG_SHA256:raise RuntimeError('audited original run config snapshot hash mismatch')
 return json.loads(REFERENCE_CONFIG.read_text())


def configuration_digest(config):
 # Derived map is separately bound to chosen_split bytes and receipt contents.
 c={k:v for k,v in config.items() if k!='judge_body_order'}
 return hashlib.sha256(json.dumps(c,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def code_hashes():
 p=Path(__file__).resolve().parent
 return {str(f.relative_to(p)):sha256(f) for f in sorted(p.rglob('*.py'))}


def config_blockers(config):
 original=reference_config();out=[]
 if config.get('reference_config_sha256')!=REFERENCE_CONFIG_SHA256:out.append('candidate must pin actual original config hash')
 for k in ['w_contrast','tau','contrast_sample','reenc_consistency','reenc_prob','n_judges','judge_mode','w_judge','w_pen','w_cgeo','mask_prob','mask_mix','combo_sample']:
  if config.get('recipe',{}).get(k)!=original[k]:out.append('reference recipe mismatch: '+k)
 for key,value in {'position':5.,'root_height':3.,'penetration':200.,'contact_geometry':1.}.items():
  if config.get('loss',{}).get(key)!=value:out.append('effective objective weight mismatch: '+key)
 if config.get('contact_dropout')!=.3:out.append('original contact dropout required')
 if config.get('optimizer',{}).get('clip_norm')!=1.:out.append('original gradient clip required')
 if config.get('model',{}).get('n_judges')!=16:out.append('original 16-class auxiliary head required')
 if config.get('optimizer',{}).get('lr')!=original['lr'] or config.get('optimizer',{}).get('weight_decay')!=1e-4:out.append('original AdamW rate/decay required')
 return out


REQUIRED_ARTIFACTS={'dataset_audit.json','weight_transfer.json','real_batch_preflight.json','judge_body_order.json'}


def runtime_identity():
 import torch
 cuda=torch.cuda.is_available()
 props=torch.cuda.get_device_properties(0) if cuda else None
 return {'python':sys.version,'torch':torch.__version__,'torch_cuda':torch.version.cuda,'hostname':platform.node(),
  'cuda_available':cuda,'gpu_name':props.name if props else None,'gpu_uuid':str(getattr(props,'uuid','unavailable')) if props else None}


def live_evidence_paths(config,root):
 root=Path(root).resolve()
 paths=[Path(config['reference_config']),Path(config['reference_checkpoint']),root/config['qwen_cache']]
 for directory in ['data_xwbc','data_audit']:
  paths.extend(p for p in (root/directory).rglob('*') if p.is_file() and p.suffix in {'.json','.npz'})
 required=[root/'data_xwbc'/n for n in ['chosen_split.json','manifest.json','checks.json','row_table.npz']]+[root/'data_audit/payload_manifest.json']
 if any(not p.is_file() for p in required):raise ValueError('missing live dataset completion artifacts')
 return sorted(set(p.resolve() for p in paths))


def check_full_report(report):
 if report.get('full_reference_recipe_verified') is not True or report.get('optimizer_steps')!=0 or not report.get('gpu','').startswith('cuda'):raise ValueError('full real CUDA recipe preflight missing')
 profiles=report.get('full_recipe_profiles',[])
 if len(profiles)!=2 or {p['domain'] for p in profiles}!={'xwbc','unimate'}:raise ValueError('both domain full backward profiles required')
 for p in profiles:
  if p.get('optimizer_steps')!=0 or not isinstance(p.get('gradient_norm'),(int,float)) or not math.isfinite(p['gradient_norm']) or p['gradient_norm']<0:raise ValueError('no-update finite backward evidence required')
  if not p.get('terms') or any(not isinstance(v,(int,float)) or not math.isfinite(v) for v in p['terms'].values()):raise ValueError('nonfinite or missing full-recipe terms')
  if p['domain']=='xwbc':
   if not {'pen','cgeo','root_height_smooth_l1_m','rotation_deg','position_over_reach','contrast','judge','reenc'}<=set(p['terms']):raise ValueError('missing active original loss in full-recipe profile')
   primary=[d for d in p['directions'] if d['mode']=='reference_mask_mix']
   if len(primary)!=8 or len({(d['source'],d['target']) for d in primary})!=8 or any(d.get('weight')!=.125 for d in primary):raise ValueError('eight unique primary directions with weight1/8 required')
   extra=[d for d in p['directions'] if d['mode']=='sparse_only_extra']
   if len(extra)!=1 or extra[0].get('weight')!=1.:raise ValueError('pure SE3 extra branch not tested at weight1')
   if not p['auxiliary'].get('reenc_active') or len(p['auxiliary'].get('contrast_bodies',[]))!=6 or len(set(p['auxiliary'].get('contrast_bodies',[])))!=6:raise ValueError('cycle and six-body contrast not tested')
   if p['auxiliary'].get('contrast',{}).get('active') is not True:raise ValueError('contrast needs distinct physical rows')
  elif p['auxiliary'].get('cross_body_auxiliary_applied') is not False:raise ValueError('UniMate must not claim paired auxiliary labels')
 return True


def training_recipe_status(config,validation_run=None,root=None):
 original=reference_config();blockers=config_blockers(config)
 if validation_run is None or root is None:blockers.append('No matching real CUDA recipe-validation run; offline tests do not authorize data readiness')
 else:
  try:
   vr=Path(validation_run).resolve();receipt=json.loads((vr/'recipe_validation_receipt.json').read_text())
   if receipt['status']!='passed' or receipt['implementation_revision']!=IMPLEMENTATION_REVISION:raise ValueError('receipt revision/status')
   if receipt['configuration_sha256']!=configuration_digest(config):raise ValueError('candidate config changed')
   if receipt['code_sha256']!=code_hashes():raise ValueError('implementation changed since validation')
   if receipt['root']!=str(Path(root).resolve()):raise ValueError('dataset root mismatch')
   exit_status=json.loads((vr/'exit_status.json').read_text())
   if exit_status['status']!='recipe_validated_no_training':raise ValueError('validation has not cleanly exited')
   if set(receipt['validation_artifacts'])!=REQUIRED_ARTIFACTS:raise ValueError('validation artifact set incomplete')
   if receipt['runtime']!=runtime_identity() or not receipt['runtime']['cuda_available']:raise ValueError('runtime/device changed since validation')
   for relative,digest in receipt['validation_artifacts'].items():
    if sha256(vr/relative)!=digest:raise ValueError('validation artifact changed: '+relative)
   report=json.loads((vr/'real_batch_preflight.json').read_text())
   check_full_report(report)
   if set(receipt['live_files_sha256'])!={str(p) for p in live_evidence_paths(config,root)}:raise ValueError('live source/data file evidence set incomplete')
   if sha256(config['reference_config'])!=REFERENCE_CONFIG_SHA256 or sha256(config['reference_checkpoint'])!=config['reference_sha256']:raise ValueError('reference source hash mismatch')
   if sha256(Path(root)/config['qwen_cache'])!=config['qwen_sha256']:raise ValueError('pinned Qwen hash mismatch')
   for name,digest in receipt['live_files_sha256'].items():
    if sha256(name)!=digest:raise ValueError('source/data changed: '+name)
   split=json.loads((Path(root)/'data_xwbc/chosen_split.json').read_text())
   if receipt['judge_body_order']!=split['body_names'] or len(set(receipt['judge_body_order']))!=16:raise ValueError('current auxiliary label map mismatch')
  except (OSError,KeyError,ValueError,TypeError) as exc:blockers.append('Real recipe validation unavailable or stale: '+str(exc))
 return {'train_allowed':not blockers,'implementation_revision':IMPLEMENTATION_REVISION,'reference_run':'L_M20k_0903','reference_config_sha256':REFERENCE_CONFIG_SHA256,
  'source':'hash-matched actual run config, not CLI defaults','verified_active_reference_flags':{k:original[k] for k in ['w_contrast','tau','contrast_sample','reenc_consistency','reenc_prob','judge_mode','n_judges','w_judge','w_pen','w_cgeo','slip_pairs','w_slip_w','w_hpair']},
  'blockers':blockers,'temporal_exceptions':['World-slip and hpair disabled: sparse rows lack complete adjacent world-root/ground/time supervision.'],
  'online_requirements':ONLINE_REQUIREMENTS,'no_config_override':True,'does_not_require_new_user_permission':True}


def assert_training_recipe_ready(config,validation_run=None,root=None):
 status=training_recipe_status(config,validation_run,root)
 if not status['train_allowed']:raise RuntimeError('TRAINING_RECIPE_BLOCKED: '+'; '.join(status['blockers']))
 return status


def write_validation_receipt(config,root,run,report):
 """Only called after production strict loader + all-loss CUDA backward complete."""
 if config_blockers(config):raise RuntimeError('invalid reference recipe settings')
 check_full_report(report)
 if not runtime_identity()['cuda_available']:raise RuntimeError('real CUDA full-recipe preflight required')
 root=Path(root).resolve();run=Path(run)
 if sha256(config['reference_config'])!=REFERENCE_CONFIG_SHA256:raise ValueError('live original config hash mismatch')
 if sha256(config['reference_checkpoint'])!=config['reference_sha256']:raise ValueError('live original checkpoint mismatch')
 paths=live_evidence_paths(config,root)
 value={'status':'passed','implementation_revision':IMPLEMENTATION_REVISION,'root':str(root),'configuration_sha256':configuration_digest(config),'code_sha256':code_hashes(),'runtime':runtime_identity(),
  'judge_body_order':config['judge_body_order'],'judge_warmstart':'none; entire auxiliary head reinitialized under verified current map',
  'live_files_sha256':{str(p.resolve()):sha256(p) for p in paths},'validation_artifacts':{n:sha256(run/n) for n in ['dataset_audit.json','weight_transfer.json','real_batch_preflight.json','judge_body_order.json']}}
 p=run/'recipe_validation_receipt.json';p.write_text(json.dumps(value,indent=2)+'\n');return value
