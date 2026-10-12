"""Fail-closed acceptance of the bounded XWBC pilot extraction (no remote access)."""
from __future__ import annotations
import hashlib
import io
import json
from pathlib import Path
import numpy as np

BODY_JOINTS = {
 'human_smpl':22, 'booster_k1':21, 'booster_t1':24, 'booster_t1_29dof':32,
 'fourier_gr3':32, 'fourier_n1':27, 'hightorque_hi':26, 'kuavo_s45':20,
 'pnd_adam_lite':24, 'tienkung':22, 'unitree_g1_23dof':36,
 'unitree_g1_29dof':36, 'unitree_h1':20, 'unitree_h1_2':26,
 'unitree_h2':32, 'unitree_r1':38,
}
DEFAULT_CONTRACT = dict(body_joints=BODY_JOINTS, rows=30322, train_rows=24744,
 cache_rows=(496593,4733703), fps=30., max_frames=96, min_stride=3,
 group_counts={(0,0):69,(0,1):8,(1,0):256,(1,1):64})
SPLIT_CHECKS = dict(train_eval_group_intersection=0, train_eval_row_intersection=0,
 amass_train_in_old_exclude_set=0, amass_eval_all_in_old_exclude_set=True,
 same_row_table_for_all_bodies=True)
ROW_FIELDS = {'source_cache_index':'cache_index','source_row_index':'source_row_index',
 'row_ids':'row_ids','group_index':'group_index','frame_index':'frame_index',
 'timestamp_seconds':'timestamp_seconds','split':'split','pretrain_seen':'pretrain_seen'}

def sha256(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
 return h.hexdigest()

def read_json(path):return json.loads(Path(path).read_text())
def load_npz(path):
 with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def load_dof(path):
 # Original DOF NPZ also has object joint_names/slot_joints; read only numeric input.
 with np.load(path,allow_pickle=False) as z:return {'dof':z['dof']}

def load_legacy_spec(path,expected_sha256):
 """The old writer stores joint_names as object; only read after byte-hash verification.

 Motion payloads still use allow_pickle=False. One immutable verified byte buffer is
 used for the narrowly scoped legacy metadata read; do not use this for other files.
 """
 data=Path(path).read_bytes()
 require(hashlib.sha256(data).hexdigest()==expected_sha256,'legacy spec SHA256 mismatch')
 with np.load(io.BytesIO(data),allow_pickle=False) as z:
  out={k:z[k] for k in ['parents','rest_offsets','name']}
 with np.load(io.BytesIO(data),allow_pickle=True) as z:
  names=z['joint_names']
 require(names.ndim==1 and all(isinstance(n,str) for n in names),'legacy names must be a string array')
 out['joint_names']=np.asarray(names,dtype=str)
 return out

def require(ok,message):
 if not ok:raise ValueError('XWBC: '+message)
def local_path(root,relative):
 require(isinstance(relative,str) and relative and not Path(relative).is_absolute(),'expected relative artifact path')
 root=Path(root).resolve();p=(root/relative).resolve()
 require(p.is_relative_to(root),'artifact escapes data directory')
 require(p.is_file(),'missing artifact '+relative)
 return p

def verify_file(root,relative,digest,artifacts):
 p=local_path(root,relative)
 require(isinstance(digest,str) and len(digest)==64,'missing SHA256 for '+relative)
 actual=sha256(p);require(actual==digest,'SHA256 mismatch '+relative)
 artifacts[relative]=actual
 return p

def body_map(records,expected,label):
 require(isinstance(records,list),label+' must be a list')
 names=[r.get('body') for r in records]
 require(len(names)==len(expected) and len(set(names))==len(names) and set(names)==set(expected),label+' missing, duplicate or unexpected body')
 return {r['body']:r for r in records}

def _check_terminal_rows(records,expected,n,row_hash,label):
 by=body_map(records,expected,label)
 for name,r in by.items():
  require(r.get('rows')==n and r.get('J')==expected[name],label+' count '+name)
  for k in ['pose9_all_finite','invalid_contact_is_nan','invalid_root_height_is_nan','contact_valid_values_finite']:
   require(r.get(k) is True,label+' failed '+name+'/'+k)
  for k in ['position_root_max_abs_cm','pose9_position_formula_max_abs_error']:
   require(r.get(k)==0,label+' failed '+name+'/'+k)
  require(r.get('row_content_sha256')==row_hash,label+' row digest '+name)
 return by

def validate_descriptor(name,s,d,desc,reach,joints):
 """Compare derived JSON to original static NPZ, including the nonzero root offset."""
 for k in ['parents','rest_offsets','joint_names']:require(k in s,'spec missing '+k)
 par=s['parents'];off=s['rest_offsets'];names=s['joint_names'].astype(str).tolist()
 require(par.shape==(joints,) and par.dtype.kind in 'iu','parents shape/type '+name)
 require(joints>0 and par[0]==-1 and all(0<=int(p)<j for j,p in enumerate(par[1:],1)),'invalid parent-first tree '+name)
 require(off.shape==(joints,3) and np.isfinite(off).all(),'offset shape/finiteness '+name)
 require(len(names)==joints and len(set(names))==joints,'joint names '+name)
 if 'name' in s:require(str(s['name'])==name,'spec body identity '+name)
 require('dof' in d and d['dof'].shape==(joints,19) and np.isfinite(d['dof']).all(),'DOF shape/finiteness '+name)
 require(desc.get('body')==name and desc.get('J')==joints and desc.get('root_index')==0,'descriptor identity '+name)
 require(desc.get('joint_names')==names and desc.get('parents')==par.tolist(),'descriptor names/tree '+name)
 require(np.array_equal(np.asarray(desc.get('rest_offsets_cm')),off),'descriptor offsets '+name)
 require(desc.get('dof_shape')==[joints,19] and desc.get('reach_cm')==reach,'descriptor DOF/reach '+name)
 off32=np.asarray(off,dtype=np.float32);bl=np.linalg.norm(off32,axis=-1);bl=bl/(float(bl.sum()) or 1.)
 depth=np.zeros(joints,dtype=np.int64)
 for j in range(1,joints):depth[j]=depth[par[j]]+1
 hop=np.full((joints,joints),joints+1,dtype=np.int64);np.fill_diagonal(hop,0)
 for j,p in enumerate(par):
  if p>=0:hop[j,p]=hop[p,j]=1
 for k in range(joints):hop=np.minimum(hop,hop[:,k,None]+hop[None,k,:])
 require(np.allclose(np.asarray(desc.get('norm_bonelen')),bl,atol=1e-7,rtol=1e-6),'descriptor bone lengths '+name)
 require(np.array_equal(desc.get('depth'),depth) and np.array_equal(desc.get('hop'),hop),'descriptor depth/hop '+name)
 return s,d,desc

def validate_xwbc(root,*,contract=None,require_static_seal=True,require_ground_seal=True):
 """Return manifest, checks, split, row table, entries and verified file hashes.

 The optional contract is solely for small synthetic fixtures. load_bodies always uses
 the pinned production contract. Validation never treats progress.json as completion.
 """
 c=DEFAULT_CONTRACT if contract is None else contract;root=Path(root)
 expected=c['body_joints'];n=c['rows'];artifacts={}
 m=read_json(root/'manifest.json');checks=read_json(root/'checks.json');split=read_json(root/'chosen_split.json')
 for p in ['manifest.json','checks.json','chosen_split.json']:artifacts[p]=sha256(root/p)
 require(m.get('status')=='complete' and m.get('rows')==n and m.get('bodies')==len(expected),'terminal manifest incomplete')
 require(m.get('fps')==c['fps'] and split.get('fps')==c['fps'],'fps changed')
 require(checks.get('all_pass') is True,'terminal checks did not pass')
 require(checks.get('split_checks')==SPLIT_CHECKS and split.get('no_leakage_checks')==SPLIT_CHECKS,'split checks failed/missing')
 require(split.get('rows')==n,'split row count')
 names=split.get('body_names',[])
 require(len(names)==len(expected) and len(set(names))==len(names) and set(names)==set(expected),'chosen body names mismatch')
 require(split.get('max_frames_per_clip')==c['max_frames'] and split.get('minimum_source_frame_stride')==c['min_stride'],'sampling policy changed')
 verify_file(root,'chosen_split.json',m.get('chosen_split_sha256'),artifacts)
 rp=verify_file(root,'row_table.npz',m.get('row_table_sha256'),artifacts)
 require(split.get('row_table_sha256')==artifacts['row_table.npz'],'split row-table checksum mismatch')
 row=load_npz(rp)
 for key in ROW_FIELDS.values():
  require(key in row and row[key].shape==(n,),'row table field shape '+key)
  if key=='timestamp_seconds':require(row[key].dtype==np.float64,'timestamp dtype')
  elif key=='pretrain_seen':require(row[key].dtype==np.bool_,'pretrain_seen dtype')
  else:require(row[key].dtype.kind in 'iu','integer row field '+key)
 groups=split.get('groups',[]);reconstructed=[];seen={};counts={};previous=None
 for gi,g in enumerate(groups):
  ci=g.get('cache_index');sp=g.get('split');clip=g.get('clip');start=g.get('start');length=g.get('length');frames=g.get('frame_indices')
  require(type(ci) is int and ci in (0,1) and type(sp) is int and sp in (0,1),'group cache/split')
  require(isinstance(clip,str) and clip,'group clip')
  key=(ci,clip)
  require(key not in seen,'source group duplicated/cross-split: '+str(key));seen[key]=sp
  require(g.get('group_id')==('lafan::' if ci==0 else 'amass::')+clip,'group identity')
  require(type(start) is int and type(length) is int and start>=0 and length>0 and start+length<=c['cache_rows'][ci],'source group bounds')
  require(previous is None or previous<(ci,start),'group table not strictly source ordered');previous=(ci,start)
  require(isinstance(frames,list) and 0<len(frames)<=c['max_frames'] and all(type(x) is int for x in frames),'group frame list')
  require(g.get('chosen_frame_count')==len(frames) and frames[0]>=0 and frames[-1]<length and all(b-a>=c['min_stride'] for a,b in zip(frames,frames[1:])),'group frame stride/count')
  k=min(c['max_frames'],(length-1)//c['min_stride']+1)
  chosen=np.floor(np.linspace(0,length-1,k)).astype(np.int64).tolist() if k>1 else [0]
  require(frames==chosen,'group frame selection differs from uniform bounded recipe')
  seen_before=not(ci==1 and sp==1)
  require(g.get('pretrain_seen') is seen_before and g.get('pretrain_excluded') is (not seen_before),'pretraining exposure labels')
  counts[(ci,sp)]=counts.get((ci,sp),0)+1
  for fr in frames:reconstructed.append([ci,start+fr,(0 if ci==0 else c['cache_rows'][0])+start+fr,gi,fr,sp,int(seen_before)])
 require(counts==c['group_counts'],'source-group counts changed')
 tab=np.asarray(reconstructed,dtype=np.int64)
 require(tab.shape==(n,7),'reconstructed row count')
 for i,key in enumerate(['cache_index','source_row_index','row_ids','group_index','frame_index','split','pretrain_seen']):
  require(np.array_equal(row[key],tab[:,i]),'row table disagrees with groups: '+key)
 require(np.unique(row['row_ids']).size==n,'duplicate source rows')
 require(np.array_equal(row['timestamp_seconds'],row['frame_index'].astype(np.float64)/c['fps']),'timestamps disagree with source frame')
 digest=hashlib.sha256(tab.tobytes()).hexdigest()
 require(split.get('row_table_content_sha256')==digest,'row content SHA256')
 want_counts={'amass_train_groups':counts.get((1,0),0),'amass_eval_groups':counts.get((1,1),0),'lafan_train_groups':counts.get((0,0),0),'lafan_eval_groups':counts.get((0,1),0),'train_rows':c['train_rows'],'eval_rows':n-c['train_rows']}
 require(split.get('counts')==want_counts and int((row['split']==0).sum())==c['train_rows'],'split counts')
 provenance=split.get('pretrain_heldout_provenance',[])
 require(len(provenance)==2 and all(p.get('exclude_set') and len(p.get('config_sha256',''))==64 and len(p.get('launch_args_sha256',''))==64 for p in provenance),'old exclusion provenance missing')
 require(len(split.get('exclude_set_sha256',''))==64,'exclude-set checksum missing')
 _check_terminal_rows(m.get('checks'),expected,n,digest,'manifest checks')
 checked=_check_terminal_rows(checks.get('checks'),expected,n,digest,'checks.json')
 require(m['checks']==checks['checks'],'terminal per-body checks disagree')
 entries=body_map(m.get('body_payloads'),expected,'payload manifest')
 require({p.name for p in (root/'payloads').glob('*.npz')}=={name+'.npz' for name in expected},'payload directory missing/extra body')
 if require_static_seal:
  seal=m.get('static_seal',{})
  require(seal.get('schema_version')==1 and seal.get('mode') in {'verified_during_extraction','source_revalidated_at_seal_time'},'static seal missing; verify original sources before sealing')
  require(len(seal.get('source_cache_manifest_sha256',''))==64,'static seal source-manifest checksum missing')
  source_sequence_files=split.get('source_sequence_files',[])
  require(len(source_sequence_files)==2 and seal['source_cache_manifest_sha256']==source_sequence_files[0].get('manifest_sha256'),'static seal source manifest mismatch')
 for name,e in entries.items():
  require(e.get('J')==expected[name] and e.get('payload')==f'payloads/{name}.npz','payload identity '+name)
  p=verify_file(root,e['payload'],e.get('payload_sha256'),artifacts)
  require(e.get('payload_bytes')==p.stat().st_size,'payload byte count '+name)
  reach=e.get('reach_cm');require(isinstance(reach,(int,float)) and np.isfinite(reach) and reach>0,'reach '+name)
  require(e.get('static_descriptor')==f'static/{name}_descriptor.json','descriptor path '+name)
  rotations=e.get('source_rotations',[])
  require(len(rotations)==2 and all(r.get('body')==name and r.get('rows')==c['cache_rows'][ci] and r.get('shape')==[c['cache_rows'][ci],expected[name],6] for ci,r in enumerate(rotations)),'normalizer source shapes '+name)
  require(all(isinstance(r.get('reach_cm'),(int,float)) and np.isfinite(r['reach_cm']) and r['reach_cm']>0 for r in rotations),'invalid normalizer source reach '+name)
  require(reach==max(r['reach_cm'] for r in rotations),'normalizer maxima '+name)
  if require_static_seal:
   fs=e.get('static_files',{});require(set(fs)=={'spec','dof','descriptor'},'static files seal missing '+name)
   for kind,ext in [('spec','npz'),('dof','npz'),('descriptor','json')]:
    rec=fs[kind];rel=f'static/{name}_{kind}.{ext}'
    require(rec.get('path')==rel,'static path '+name+'/'+kind)
    verify_file(root,rel,rec.get('sha256'),artifacts)
    if kind!='descriptor':require(rec.get('source_sha256')==rec['sha256'] and rec.get('source_path'),'static original-source comparison missing '+name+'/'+kind)
   desc=read_json(root/fs['descriptor']['path'])
   require(desc.get('source_spec_sha256')==fs['spec']['sha256'],'descriptor original spec digest '+name)
   validate_descriptor(name,load_legacy_spec(root/fs['spec']['path'],fs['spec']['sha256']),load_dof(root/fs['dof']['path']),desc,reach,expected[name])
  if require_ground_seal:
   rec=e.get('ground_static',{})
   require(rec.get('path')==f'static/{name}_ground.json','ground static missing; run seal_xwbc_static.py --backfill-ground '+name)
   gp=verify_file(root,rec['path'],rec.get('sha256'),artifacts)
   require(m.get('ground_recipe_seal',{}).get('schema_version')==1,'ground recipe seal missing')
   specpath=root/f'static/{name}_spec.npz'
   specsha=sha256(specpath)
   g=validate_ground_static(name,read_json(gp),load_legacy_spec(specpath,specsha),e.get('source_contacts',[]))
   require(g['source_spec_sha256']==specsha,'ground original spec mismatch '+name)
   seal=m['ground_recipe_seal']
   require(g['sole_source']['sha256']==seal.get('sole_json_sha256') and g['sole_function_source']['sha256']==seal.get('source_contact_label_sha256'),'ground source geometry seal mismatch '+name)
   validate_ground_payload(name,load_npz(p))
 return m,checks,split,row,entries,artifacts

def validate_payload(name,a,row,entry):
 n=len(row['row_ids']);j=entry['J']
 for field,key in ROW_FIELDS.items():require(field in a and np.array_equal(a[field],row[key]),'payload row pairing '+name+'/'+field)
 for key,shape in [('pose9',(n,j,9)),('global_rot6',(n,j,6)),('position_cm',(n,j,3))]:
  require(key in a and a[key].shape==shape and a[key].dtype==np.float32 and np.isfinite(a[key]).all(),'payload pose dtype/shape/finiteness '+name+'/'+key)
 require(np.array_equal(a['pose9'][...,:6],a['global_rot6']),'rotation slots '+name)
 require(np.asarray(a.get('reach_cm')).shape==() and float(a['reach_cm'])==entry['reach_cm'],'payload reach '+name)
 require(np.array_equal(a['pose9'][...,6:],a['position_cm']/float(a['reach_cm'])),'position/reach formula '+name)
 require(np.all(a['position_cm'][:,0]==0),'root-relative position '+name)
 for key,shape in [('root_height_valid',(n,)),('contact_valid',(n,j)),('contact_hard',(n,j)),('contact_trust',(n,))]:
  require(key in a and a[key].shape==shape and a[key].dtype==np.bool_,'validity dtype/shape '+name+'/'+key)
 for key,mask,shape in [('root_height_m','root_height_valid',(n,)),('contact_sdist','contact_valid',(n,j)),('contact_soft','contact_valid',(n,j))]:
  require(key in a and a[key].shape==shape and a[key].dtype==np.float32,'measurement dtype/shape '+name+'/'+key)
  require(np.isfinite(a[key][a[mask]]).all() and np.isnan(a[key][~a[mask]]).all(),'valid finite / unknown NaN '+name+'/'+key)
 require(not np.any(a['root_height_valid'] & ~a['contact_trust']),'root height trust '+name)
 require(not np.any(a['contact_valid'] & ~a['contact_trust'][:,None]),'contact trust '+name)
 require(not np.any(a['contact_hard'] & ~a['contact_valid']),'invalid hard contact '+name)


def validate_ground_static(name, ground, spec, source_contacts):
 """No missing sole table is ever silently treated as a joint-centre mapping."""
 require(ground.get('schema_version')==1 and ground.get('body')==name,'ground identity '+name)
 require(ground.get('ground_geometry_verified') is True,'ground geometry not verified '+name)
 require(ground.get('source_spec_sha256') and len(ground['source_spec_sha256'])==64,'ground spec provenance '+name)
 for key in ['sole_source','sole_function_source']:
  require(isinstance(ground.get(key),dict) and len(ground[key].get('sha256',''))==64 and ground[key].get('path'),'ground missing original '+key+' '+name)
 j=len(spec['parents']);names=spec['joint_names'].astype(str).tolist()
 raw_js=np.asarray(ground.get('sole_j'))
 require(raw_js.ndim==1 and (not len(raw_js) or raw_js.dtype.kind in 'iu'),'sole index integer type '+name)
 js=raw_js.astype(np.int64);off=np.asarray(ground.get('sole_off_m'),dtype=np.float32).reshape(-1,3)
 require(js.ndim==1 and off.shape==(len(js),3) and np.isfinite(off).all(),'sole mapping shape/finiteness '+name)
 require(js.tolist()==sorted(set(js.tolist())) and all(0<=v<j for v in js),'sole indices '+name)
 require(ground.get('sole_joint_names')==[names[k] for k in js],'sole joint names '+name)
 require(ground.get('mapping_status')==('verified_nonempty_legacy_sole_mapping' if len(js) else 'verified_empty_legacy_mapping'),'sole mapping status '+name)
 require(len(source_contacts)==2 and all(v.get('exists') is True for v in source_contacts),'original contact source missing '+name)
 source_refs=ground.get('source_contacts',[])
 require(len(source_refs)==2,'ground source-contact lineage '+name)
 for ref,record in zip(source_refs,source_contacts):
  require(ref.get('path')==record.get('path') and ref.get('sha256')==record.get('sha256') and len(ref.get('sha256',''))==64,'ground source-contact hash binding '+name)
 for record in source_contacts:
  h=record['header']
  require(h.get('sole_applied')==js.tolist() and h.get('sole_joint_names')==[names[k] for k in js],'contact header/sole table disagree '+name)
  require(abs(float(h.get('h0_m',0)))<.30,'contact floor sanity '+name)
  require(float(h['thr_m'])>0 and float(h['D_used_m'])>0,'contact scale '+name)
 require(ground.get('threshold_source_cache_index')==1 and ground.get('contact_threshold_m')==float(source_contacts[-1]['header']['thr_m']),'legacy final-cache geometry threshold '+name)
 require(ground.get('contact_tau_m')==ground['contact_threshold_m']/3.,'contact tau '+name)
 return ground


def validate_ground_payload(name,a):
 n,j=a['contact_valid'].shape
 require('ground_valid' in a and a['ground_valid'].shape==(n,) and a['ground_valid'].dtype==np.bool_,'ground valid shape '+name)
 require(np.array_equal(a['ground_valid'],a['contact_trust'] & a['root_height_valid']),'ground validity differs from trusted root gap '+name)
 require(not np.any(a['ground_valid'] & ~np.isfinite(a['root_height_m'])),'ground root finite '+name)
 require(np.all((a['contact_soft'][a['contact_valid']]>=0)&(a['contact_soft'][a['contact_valid']]<=1)),'contact soft range '+name)
