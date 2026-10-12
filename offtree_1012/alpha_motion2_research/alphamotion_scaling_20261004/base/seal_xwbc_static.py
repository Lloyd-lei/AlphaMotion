#!/usr/bin/env python3
"""Seal a completed old extraction by comparing each static file to live original sources.

No re-extraction or remote connection. Supply the previously verified original cache
manifest digest explicitly. Newly observed spec/DOF hashes are recorded as observations
at seal time, never represented as trusted historical hashes. The recorded descriptor
source_spec_sha256 additionally anchors each spec to the extraction-time observation.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
from pathlib import Path
import json
from xwbc_contract import (validate_xwbc,validate_descriptor,load_npz,load_dof,load_legacy_spec,read_json,
                           sha256,require)


def seal_static(data_root,source_cache,source_manifest_sha256):
 root=Path(data_root);source=Path(source_cache).resolve()
 m,checks,split,row,entries,artifacts=validate_xwbc(root,require_static_seal=False,require_ground_seal=False)
 observed=sha256(source/'manifest.json')
 require(observed==source_manifest_sha256,'original cache manifest differs from supplied verified digest')
 require(len(split.get('source_sequence_files',[]))==2,'source sequence provenance missing')
 require(split['source_sequence_files'][0]['manifest_sha256']==observed,'original cache differs from extraction provenance')
 # Verify all bodies before modifying the manifest. No payload or original source is written.
 for name,e in entries.items():
  files={}
  for kind in ['spec','dof']:
   original=source/f'{name}_{kind}.npz';relative=f'static/{name}_{kind}.npz';copied=root/relative
   old_hash=sha256(original);copy_hash=sha256(copied)
   require(old_hash==copy_hash,'copied '+kind+' differs from original source: '+name)
   files[kind]={'path':relative,'sha256':copy_hash,'source_path':str(original),'source_sha256':old_hash}
  relative=f'static/{name}_descriptor.json';desc=read_json(root/relative)
  require(desc.get('source_spec_sha256')==files['spec']['sha256'],'spec changed since extraction: '+name)
  validate_descriptor(name,load_legacy_spec(root/files['spec']['path'],files['spec']['sha256']),load_dof(root/files['dof']['path']),desc,e['reach_cm'],e['J'])
  files['descriptor']={'path':relative,'sha256':sha256(root/relative)}
  e['static_files']=files
 prior_manifest_sha256=sha256(root/'manifest.json')
 m['static_seal']={'schema_version':1,'mode':'source_revalidated_at_seal_time',
  'created_at_utc':datetime.now(timezone.utc).isoformat(),'source_cache':str(source),
  'source_cache_manifest_sha256':observed,'prior_manifest_sha256':prior_manifest_sha256,
  'note':'All copied spec/DOF files compared byte-for-byte to the verified original cache at seal time. New DOF and descriptor hashes are current observations, not historical trusted hashes. Descriptor values independently rederived; spec also matches its extraction-time source_spec_sha256.'}
 target=root/'manifest.json';tmp=target.with_suffix('.seal.tmp')
 tmp.write_text(json.dumps(m,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
 tmp.replace(target)
 validate_xwbc(root,require_ground_seal=bool(m.get("ground_recipe_seal")))
 return {'manifest':str(target),'prior_sha256':prior_manifest_sha256,'sha256':sha256(target),'bodies':len(entries),'mode':m['static_seal']['mode']}



def read_contact_rows(name,spec,caches,row):
 """Restore actual native hard/soft/trust/root gap, not thresholded input SD."""
 import numpy as np
 n=len(row['row_ids']);j=len(spec['parents']);root_index=int(np.flatnonzero(spec['parents']<0)[0])
 out={'root_height_m':np.full(n,np.nan,np.float32),'root_height_valid':np.zeros(n,bool),
      'contact_sdist':np.full((n,j),np.nan,np.float32),'contact_soft':np.full((n,j),np.nan,np.float32),
      'contact_valid':np.zeros((n,j),bool),'contact_hard':np.zeros((n,j),bool),'contact_trust':np.zeros(n,bool)}
 records=[];all_root_gaps=[]
 for ci,cache in enumerate(caches):
  path=Path(cache)/f'{name}_contact.npz'
  require(path.is_file(),'original contact source required to supplement extraction: '+str(path))
  where=np.flatnonzero(row['cache_index']==ci);sel=row['source_row_index'][where]
  with np.load(path,allow_pickle=False) as z:
   h=json.loads(str(z['header']));require(h['J']==j and h['joint_names']==spec['joint_names'].astype(str).tolist(),'contact skeleton identity '+name)
   gap_source=z['gap'];require(int(h['N'])==len(gap_source),'contact row count '+name)
   all_root_gaps.append(gap_source[:,root_index].astype(np.float32))
   require(not len(sel) or (sel.min()>=0 and sel.max()<int(h['N'])),'contact selected-row bounds '+name)
   require(abs(float(h.get('h0_m',0)))<.30,'original floor sanity '+name)
   thr=float(h['thr_m']);D=float(h['D_used_m']);require(np.isfinite(thr) and thr>0 and np.isfinite(D) and D>0,'contact scale '+name)
   gap=gap_source[sel].astype(np.float32);del gap_source
   soft=z['contact_soft'][sel].astype(np.float32);hard=z['contact'][sel].astype(bool)
   fallback='trust' not in z.files
   tr=z['trust'][sel].astype(bool) if not fallback else np.ones(len(sel),bool)
  # Original dual_common.load_contact explicitly uses all True for old files
  # with no trust field; retain and disclose that exact compatibility policy.
  pen=gap<0;soft=np.maximum(soft,pen.astype(np.float32));hard=hard|pen
  score=thr-gap;score=(np.sign(score)*np.log1p(np.abs(score)/thr)/np.log1p(D/thr)).astype(np.float16).astype(np.float32)
  valid=tr[:,None]&np.isfinite(gap)&np.isfinite(soft)&np.isfinite(score)
  hv=tr&np.isfinite(gap[:,root_index])
  out['contact_sdist'][where]=np.where(valid,score,np.nan);out['contact_soft'][where]=np.where(valid,soft,np.nan)
  out['contact_valid'][where]=valid;out['contact_hard'][where]=hard&valid;out['contact_trust'][where]=tr
  out['root_height_m'][where]=np.where(hv,gap[:,root_index],np.nan);out['root_height_valid'][where]=hv
  records.append({'path':str(path.resolve()),'exists':True,'sha256':sha256(path),'bytes':path.stat().st_size,'header':h,
                  'selected_rows':len(where),'selected_trusted_rows':int(tr.sum()),'trust_legacy_all_true_fallback':fallback})
 root_median=float(np.median(np.concatenate(all_root_gaps)))
 require(-.30<root_median<2.,'original combined-cache root-gap median sanity '+name)
 for record in records:record['combined_root_gap_median_m']=root_median
 out['ground_valid']=out['contact_trust']&out['root_height_valid']
 return out,records


def build_ground_static(name,spec,spec_sha256,records,sole_json,legacy_source_dir):
 """Execute only the specified original sole_offsets function, never the source module."""
 import ast,os,types
 import numpy as np
 from xwbc_contract import validate_ground_static
 source=Path(legacy_source_dir)/'contact_label.py';sole=Path(sole_json)
 require(source.is_file() and sole.is_file(),'original contact_label.py and sole_offsets3.json required; missing extraction fields are not evidence that original sole data is absent')
 if os.environ.get('AM_SOLE','1')=='0':raise ValueError('AM_SOLE=0 would disable the required original sole mapping')
 tree=ast.parse(source.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='sole_offsets')
 alias=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SOLE_ALIAS' for t in n.targets))
 namespace={'np':np,'os':os,'json':json,'USE_SOLE':True,'SOLE_JSON':str(sole),'SOLE_ALIAS':alias}
 exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),namespace)
 mapped=namespace['sole_offsets'](name,types.SimpleNamespace(joint_names=spec['joint_names'].astype(str).tolist()))
 js=sorted(mapped);off=np.asarray([mapped[k] for k in js],np.float32).reshape(-1,3)
 thr=float(records[-1]['header']['thr_m'])
 ground={'schema_version':1,'body':name,'J':len(spec['parents']),'source_spec_sha256':spec_sha256,
  'ground_geometry_verified':True,'sole_j':js,'sole_off_m':off.tolist(),'sole_joint_names':[str(spec['joint_names'][k]) for k in js],
  'mapping_status':'verified_nonempty_legacy_sole_mapping' if js else 'verified_empty_legacy_mapping',
  'empty_mapping_semantics':'An explicitly observed empty result of the original function retains legacy joint-centre semantics; no missing file is substituted by this result.' if not js else None,
  'sole_source':{'path':str(sole.resolve()),'sha256':sha256(sole)},'sole_function_source':{'path':str(source.resolve()),'sha256':sha256(source),'function':'sole_offsets','USE_SOLE':True,'aliases':alias},
  'contact_threshold_m':thr,'contact_tau_m':thr/3.,'threshold_source_cache_index':len(records)-1,
  'threshold_policy':'exact old load_contact: geometry uses final configured cache header; signed-distance inputs use each source cache own thr/D',
  'contact_header_thresholds_m':[float(r['header']['thr_m']) for r in records],
  'coordinate_contract':'Y-up; FK positions cm divided by100; sole vectors in local ardy basis metres; root_height_m is target GT root gap',
  'source_contacts':[{'path':r['path'],'sha256':r['sha256'],'header':{k:r['header'].get(k) for k in ['N','J','h0_m','thr_m','D_used_m','sole_applied','sole_joint_names','sole_drop_cm']}} for r in records]}
 validate_ground_static(name,ground,spec,records)
 return ground


def _finish_ground_commit(root,journal):
 """Resume a prepared derived-file commit after interruption; original data untouched."""
 for record in journal['files']:
  target=root/record['target'];staged=root/record['staged']
  if target.is_file() and sha256(target)==record['sha256']:continue
  require(staged.is_file() and sha256(staged)==record['sha256'],'ground backfill staging changed '+record['staged'])
  staged.replace(target)
 for key,filename in [('checks','checks.json'),('manifest','manifest.json')]:
  target=root/filename;tmp=target.with_suffix('.ground.tmp');tmp.write_text(json.dumps(journal[key],indent=2,allow_nan=False)+'\n');tmp.replace(target)
 (root/'ground_backfill_pending.json').replace(root/'ground_backfill_completed.json')
 return journal['manifest']


def backfill_ground(data_root,source_caches,legacy_source_dir,sole_json,*,contract=None):
 """Supplement a COMPLETE extraction without scanning/moving full pose caches.

 Uses selected row table against original contact NPZ, restores missing static sole
 metadata, seals all derived hashes. Stages every derived payload before commit;
 an interrupted commit resumes from its journal. Run remotely only after recovery.
 """
 import numpy as np
 from xwbc_contract import validate_ground_payload
 root=Path(data_root);pending=root/'ground_backfill_pending.json'
 if pending.exists():
  m=_finish_ground_commit(root,read_json(pending))
  validate_xwbc(root,contract=contract,require_static_seal=bool(m.get('static_seal')))
  return {'status':'resumed_and_complete','manifest_sha256':sha256(root/'manifest.json')}
 caches=[Path(p).resolve() for p in source_caches];require(len(caches)==2,'two original ordered source caches required')
 m,checks,split,row,entries,_=validate_xwbc(root,contract=contract,require_static_seal=False,require_ground_seal=False)
 for i,cache in enumerate(caches):
  require(sha256(cache/'manifest.json')==split['source_sequence_files'][i]['manifest_sha256'],'original cache manifest differs from extraction provenance')
 stage=root/'.ground-backfill';stage.mkdir(exist_ok=True);files=[];old_manifest=sha256(root/'manifest.json')
 for name,e in entries.items():
  specfile=root/f'static/{name}_spec.npz';specsha=sha256(specfile)
  require(sha256(caches[0]/f'{name}_spec.npz')==specsha,'source spec changed '+name)
  spec=load_legacy_spec(specfile,specsha)
  fields,records=read_contact_rows(name,spec,caches,row)
  ground=build_ground_static(name,spec,specsha,records,sole_json,legacy_source_dir)
  arrays=load_npz(root/e['payload']);arrays.update(fields);validate_ground_payload(name,arrays)
  staged=stage/(name+'.npz');np.savez_compressed(staged,**arrays)
  files.append({'target':e['payload'],'staged':str(staged.relative_to(root)),'sha256':sha256(staged)})
  e.update(payload_sha256=sha256(staged),payload_bytes=staged.stat().st_size,source_contacts=records)
  gpath=stage/(name+'_ground.json');gpath.write_text(json.dumps(ground,indent=2,allow_nan=False)+'\n')
  relative=f'static/{name}_ground.json';e['ground_static']={'path':relative,'sha256':sha256(gpath)}
  files.append({'target':relative,'staged':str(gpath.relative_to(root)),'sha256':sha256(gpath)})
  for collection in [m['checks'],checks['checks']]:
   for check in collection:
    if check['body']==name:check.update(root_height_valid_count=int(fields['root_height_valid'].sum()),contact_valid_fraction=float(fields['contact_valid'].mean()),ground_valid_count=int(fields['ground_valid'].sum()),ground_geometry_verified=True)
 m['ground_recipe_seal']={'schema_version':1,'mode':'source_revalidated_backfill','prior_manifest_sha256':old_manifest,
  'sole_json_sha256':sha256(sole_json),'source_contact_label_sha256':sha256(Path(legacy_source_dir)/'contact_label.py'),
  'created_at_utc':datetime.now(timezone.utc).isoformat(),'note':'Previously omitted extraction geometry restored from original sources; no claim those source fields were missing.'}
 journal={'files':files,'manifest':m,'checks':checks,'prior_manifest_sha256':old_manifest}
 pending.write_text(json.dumps(journal,indent=2,allow_nan=False)+'\n');_finish_ground_commit(root,journal)
 validate_xwbc(root,contract=contract,require_static_seal=bool(m.get('static_seal')))
 return {'status':'complete','manifest_sha256':sha256(root/'manifest.json'),'bodies':len(entries),'original_sources_written':False}


def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--data-root',type=Path,required=True)
 parser.add_argument('--source-cache',type=Path)
 parser.add_argument('--source-manifest-sha256')
 parser.add_argument('--backfill-ground',action='store_true')
 parser.add_argument('--source-caches',type=Path,nargs=2)
 parser.add_argument('--legacy-source-dir',type=Path)
 parser.add_argument('--sole-json',type=Path,default=Path('/home/arenalabs/tmp/sole_offsets3.json'))
 args=parser.parse_args()
 if args.backfill_ground:
  if not args.source_caches or not args.legacy_source_dir:parser.error('--backfill-ground requires --source-caches and --legacy-source-dir')
  result=backfill_ground(args.data_root,args.source_caches,args.legacy_source_dir,args.sole_json)
 else:
  if not args.source_cache or not args.source_manifest_sha256:parser.error('static sealing requires --source-cache and --source-manifest-sha256')
  result=seal_static(args.data_root,args.source_cache,args.source_manifest_sha256)
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
