"""Pinned UniML3D native arrays -> own-joint global pose9; no 12D inversion.
Only writes this audit directory. CPU NumPy; never imports/trains a model.
"""
import argparse, hashlib, json, pickle, re
from pathlib import Path
import numpy as np

REV='053e8b4b06b6a8ba717a030e675dd5b7fd5be92e'
HOLDOUT={'54e7e37365ac4063ba2ba65151fd9f5d','669a76c557bbceb71e7106c6_fbx','79fed889082e4a25ad92be75b57f641c','669a750957bbceb71e6ec4a5_fbx'}
FK_TOL=1e-5

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def write_json(path,obj):
 path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n');tmp.replace(path)

class NumpyOnlyUnpickler(pickle.Unpickler):
 def find_class(self,module,name):
  if module in {'numpy.core.multiarray','numpy._core.multiarray'} and name in {'_reconstruct','scalar'}:
   return getattr(np.core.multiarray,name)
  if module=='numpy' and name in {'ndarray','dtype'}:return getattr(np,name)
  raise pickle.UnpicklingError(f'unsupported {module}.{name}')

def read_cond(path, expected):
 path=Path(path)
 if path.stat().st_size > 128*1024*1024:raise ValueError('cond oversized')
 if sha(path)!=expected:raise ValueError('cond checksum mismatch')
 with path.open('rb') as f:
  v=np.lib.format.read_magic(f)
  if v==(1,0):shape,order,dtype=np.lib.format.read_array_header_1_0(f)
  elif v==(2,0):shape,order,dtype=np.lib.format.read_array_header_2_0(f)
  else:raise ValueError('NPY version')
  if shape!=() or dtype!=np.dtype('O'):raise ValueError('cond format')
  obj=NumpyOnlyUnpickler(f).load().item()
 if not isinstance(obj,dict):raise ValueError('cond mapping')
 return obj

def qmat(q):
 q=np.asarray(q,dtype=np.float64);norm=np.linalg.norm(q,axis=-1)
 if not np.all(np.isfinite(q)) or np.any(norm<1e-8):raise ValueError('invalid quaternion')
 if np.max(np.abs(norm-1))>1e-3:raise ValueError('quaternion norm error')
 w,x,y,z=np.moveaxis(q/norm[...,None],-1,0)
 return np.stack([1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x),2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)],axis=-1).reshape(q.shape[:-1]+(3,3))

def globalize(local,parents):
 out=np.empty_like(local);out[...,0,:,:]=local[...,0,:,:]
 for j,p in enumerate(parents[1:],1):out[...,j,:,:]=out[...,p,:,:]@local[...,j,:,:]
 return out

def fk(global_R,offsets,parents,root=None):
 out=np.zeros(global_R.shape[:-2]+(3,),dtype=np.float64)
 if root is not None:out[...,0,:]=root
 for j,p in enumerate(parents[1:],1):out[...,j,:]=out[...,p,:]+np.einsum('...ab,b->...a',global_R[...,p,:,:],offsets[j])
 return out

def rot6(R):return np.concatenate([R[..., :,0],R[..., :,1]],axis=-1)

def err(a,b):
 e=np.linalg.norm(a-b,axis=-1)
 return {'max':float(e.max()),'mean':float(e.mean()),'p99':float(np.quantile(e,.99))}

def prepare_spec(c):
 parents=np.asarray(c['parents'],dtype=np.int64);J=len(parents);names=[str(n) for n in c['joint_names']]
 if len(names)!=J:raise ValueError('joint_names length')
 if parents[0]!=-1 or any(not 0<=p<j for j,p in enumerate(parents[1:],1)):raise ValueError('topology not root0 parent-first')
 tpos=np.asarray(c['tpos_first_frame'],dtype=np.float64).reshape(J,3)
 tl=qmat(np.asarray(c['tpos_local_rotations']).reshape(J,4));tg=globalize(tl,parents)
 offsets=np.zeros((J,3),dtype=np.float64)
 for j,p in enumerate(parents[1:],1):offsets[j]=tpos[j]-tpos[p]
 rawoff=np.asarray(c['offsets'],dtype=np.float64).reshape(J,3)
 tpose_fk=fk(tg[None],rawoff,parents,tpos[None,0])[0]
 if err(tpose_fk,tpos)['max']>FK_TOL:raise ValueError('Tpose local-offset FK inconsistent')
 pathlength=np.zeros(J)
 for j,p in enumerate(parents[1:],1):pathlength[j]=pathlength[p]+np.linalg.norm(offsets[j])
 reach=float(pathlength.max())
 if reach<=0:raise ValueError('zero reach')
 payload={'parents':parents,'rest_offsets':offsets.astype('float32'),'rest_offsets_descriptor':(100*offsets).astype('float32'),'joint_names':np.asarray(names,dtype='U'),'reach':np.array(reach,dtype='float32'),'reach_descriptor':np.array(reach*100,dtype='float32'),'ee_indices':np.array([j for j in range(1,J) if j not in parents],dtype=np.int64),'dof_features':np.zeros((J,19),dtype='float32'),'dof_valid':np.zeros(J,dtype=bool),'metric_length_valid':np.array(False),'contact_valid':np.zeros(J,dtype=bool),'ground_valid':np.array(False),'root_height_valid':np.array(False)}
 topo_hash=hashlib.sha256(parents.tobytes()+np.round(offsets,10).tobytes()+json.dumps(names).encode()).hexdigest()
 return payload, {'parents':parents,'offsets':offsets,'raw_offsets':rawoff,'tpose_local_R':tl,'tpose_global_R':tg,'reach':reach,'topology_hash':topo_hash,'tpose_fk_error':err(tpose_fk,tpos)}

def adapt(path,s):
 with np.load(path,allow_pickle=False) as z:
  dtype={k:str(z[k].dtype) for k in z.files};p=np.asarray(z['global_positions'],dtype=np.float64);q=np.asarray(z['local_rotations'],dtype=np.float64);fps=float(z['fps'])
 if p.ndim!=3 or p.shape[1:]!=(len(s['parents']),3) or q.shape!=p.shape[:2]+(4,):raise ValueError('native shapes')
 if not np.isfinite(p).all():raise ValueError('nonfinite positions')
 if not np.isfinite(fps) or fps<=0:raise ValueError('invalid frame rate')
 local=qmat(q);M=globalize(local,s['parents']);T=s['tpose_global_R'];G=M@T.swapaxes(-1,-2)
 # Literal matrix equivalent of official compute_rots_from_tpos lines 59-73.
 rebase=np.empty_like(local);rebase[:,0]=local[:,0]@s['tpose_local_R'][0].T
 for j,pa in enumerate(s['parents'][1:],1):rebase[:,j]=T[pa]@local[:,j]@s['tpose_local_R'][j].T@T[pa].T
 rebase_global=globalize(rebase,s['parents'])
 formula=float(np.max(np.abs(G-rebase_global)))
 relative=p-p[:,:1];raw_fk=fk(M,s['raw_offsets'],s['parents'],p[:,0]);canonical_fk=fk(G,s['offsets'],s['parents'])
 r6=rot6(G);pose9=np.concatenate([r6,relative/s['reach']],axis=-1)
 raw_error=err(raw_fk,p);canonical_error=err(canonical_fk,relative)
 # If local positions vary, rotation-only fixed rest skeleton is insufficient.
 local_offset_errors=[]
 for j,pa in enumerate(s['parents'][1:],1):
  actual=np.einsum('tba,tb->ta',M[:,pa],p[:,j]-p[:,pa]);local_offset_errors.append(float(np.linalg.norm(actual-s['raw_offsets'][j],axis=-1).max()))
 qc={'frames':len(p),'joints':p.shape[1],'fps':fps,'raw_dtypes':dtype,'raw_fk_error':raw_error,'canonical_fk_error':canonical_error,'max_fk_error_over_reach':canonical_error['max']/s['reach'],'official_rebase_formula_max_abs_error':formula,'max_native_local_offset_residual':max(local_offset_errors,default=0),'quaternion_norm_max_error':float(np.max(np.abs(np.linalg.norm(q,axis=-1)-1))),'threshold_canonical_body_units':FK_TOL,'passed':bool(raw_error['max']<FK_TOL and canonical_error['max']<FK_TOL and formula<1e-10)}
 arrays={'global_rot6':r6.astype('float32'),'position_canonical':relative.astype('float32'),'pose9':pose9.astype('float32'),'fps':np.full(len(p),fps,dtype='float32')}
 # Hash native numeric values independent of ZIP representation; detect exact aliases.
 h=hashlib.sha256()
 for x in (p,q):h.update(str(x.shape).encode());h.update(np.ascontiguousarray(x).tobytes())
 return arrays,qc,h.hexdigest()

def verify_payload(path, is_holdout):
 with np.load(path,allow_pickle=False) as z:
  r6=z['global_rot6'];pos=z['position_canonical'];parents=z['parents'];off=z['rest_offsets'];pose9=z['pose9']
  N,J=pos.shape[:2]
  if r6.shape!=(N,J,6) or pose9.shape!=(N,J,9):raise AssertionError('payload shapes')
  if not all(np.isfinite(z[k]).all() for k in ['global_rot6','position_canonical','pose9','rest_offsets','reach']):raise AssertionError('nonfinite serialized payload')
  a=r6[...,:3].astype('float64');b=r6[...,3:6].astype('float64');a/=np.linalg.norm(a,axis=-1,keepdims=True);b-=np.sum(a*b,axis=-1,keepdims=True)*a;b/=np.linalg.norm(b,axis=-1,keepdims=True)
  R=np.stack([a,b,np.cross(a,b)],axis=-1);error=err(fk(R,off,parents),pos)
  if error['max']>=FK_TOL:raise AssertionError('serialized float32 FK error')
  if not np.allclose(pose9[...,:6],r6,rtol=0,atol=0) or not np.allclose(pose9[...,6:],pos/float(z['reach']),rtol=1e-6,atol=2e-7):raise AssertionError('pose9 normalization')
  rows=np.stack([z['file_index'],z['frame_index']],axis=-1)
  if len(np.unique(rows,axis=0))!=N:raise AssertionError('overlapping native rows')
  for group in np.unique(z['source_group_index']):
   if len(np.unique(z['split'][z['source_group_index']==group]))!=1:raise AssertionError('serialized group leakage')
  if is_holdout and np.any(z['split']==0):raise AssertionError('serialized holdout train rows')
  for mask in ['dof_valid','metric_length_valid','contact_valid','ground_valid','root_height_valid']:
   if np.any(z[mask]):raise AssertionError('fabricated validity')
  return {'passed':True,'serialized_float32_fk_error':error,'native_row_overlap_count':0,'group_split_conflict_count':0,'frames':N,'joints':J}

class DSU:
 def __init__(self,n):self.p=list(range(n))
 def root(self,i):
  while self.p[i]!=i:self.p[i]=self.p[self.p[i]];i=self.p[i]
  return i
 def union(self,a,b):self.p[self.root(a)]=self.root(b)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parent);args=ap.parse_args();root=args.root
 selected=json.loads((root/'selected_joint_names.json').read_text());download=json.loads((root/'subset_download_report.json').read_text())
 if download['status']!='verified_complete' or download['revision']!=REV:raise ValueError('download incomplete/revision')
 bodies={b['object_type']:b for b in selected['bodies']};cond_cache={};specs={};static={};files=sorted(download['files'],key=lambda x:x['rfilename'])
 for b,meta in bodies.items():
  cp=meta['conditioning_path']
  if cp not in cond_cache:cond_cache[cp]=read_cond(cp,meta['conditioning_sha256'])
  try:
   static[b],specs[b]=prepare_spec(cond_cache[cp][b])
  except Exception as e:
   specs[b]={'error':str(e),'topology_hash':hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest(),'reach':None,'tpose_fk_error':None}
 print(json.dumps({'stage':'conditioning_ready','bodies':len(bodies)}),flush=True)
 records=[];array_cache={};union=DSU(len(files));groupkeys={};blobkeys={};numerickeys={}
 for i,f in enumerate(files):
  b=f['object_type'];source=re.sub(r'-\d{3,}$','',Path(f['rfilename']).stem);source_key=f"UniML3D:{f['dataset']}:{source}"
  actual_sha=sha(f['path'])
  if actual_sha!=f['verification']['hash']:raise ValueError('motion checksum mismatch')
  rec={**f,'file_index':i,'source_key':source_key,'topology_hash':specs[b]['topology_hash'],'source_sha256':actual_sha}
  try:
   if 'error' in specs[b]:raise ValueError('conditioning_spec: '+specs[b]['error'])
   arrays,qc,numeric_sha=adapt(f['path'],specs[b]);rec.update(qc=qc,numeric_sha256=numeric_sha,status='passed' if qc['passed'] else 'quarantined_fixed_skeleton_fk_mismatch')
   if qc['passed']:array_cache[i]=arrays
  except Exception as e:rec.update(status='quarantined_parse_or_numeric_error',error=str(e));numeric_sha=None
  for key,seen in [(source_key,groupkeys),(actual_sha,blobkeys),(numeric_sha,numerickeys)]:
   if key is None:continue
   if key in seen:union.union(i,seen[key])
   else:seen[key]=i
  records.append(rec)
  if i%30==0:print(json.dumps({'stage':'qc','files':i+1,'total':len(files),'last_status':rec['status']}),flush=True)
 clusters={}
 for i in range(len(files)):clusters.setdefault(union.root(i),[]).append(i)
 clusters=sorted(clusters.values(),key=lambda ids:min(records[i]['source_key'] for i in ids));group_records=[]
 for gi,ids in enumerate(clusters):
  gid='uniml3d_source_'+hashlib.sha256('\n'.join(sorted({records[i]['source_key'] for i in ids})).encode()).hexdigest()[:20]
  bs=sorted({records[i]['object_type'] for i in ids});group_records.append({'source_group_index':gi,'source_group_id':gid,'source_keys':sorted({records[i]['source_key'] for i in ids}),'file_indices':ids,'bodies':bs,'is_body_holdout':any(b in HOLDOUT for b in bs)})
  for i in ids:records[i].update(source_group_index=gi,source_group_id=gid)
 # All body holdout group aliases inherit holdout; other groups deterministic per-body 80/20.
 group_split={g['source_group_index']:2 for g in group_records if g['is_body_holdout']}
 for b in sorted(bodies):
  gs=sorted({r['source_group_index'] for r in records if r['object_type']==b and r['status']=='passed' and r['source_group_index'] not in group_split},key=lambda g:hashlib.sha256(f'pilot0:{group_records[g]["source_group_id"]}'.encode()).hexdigest())
  if not gs:continue
  neval=max(1,int(round(.2*len(gs)))) if len(gs)>=2 else 0
  for g in gs:group_split[g]=1 if g in gs[:neval] else 0
 # Exact same body+numeric motion alias appears once. Record all provenance aliases.
 dedup={}
 for r in records:
  r['split']=group_split.get(r['source_group_index'],2 if r['object_type'] in HOLDOUT else -1)
  if r['status']!='passed':continue
  key=(r['object_type'],r['numeric_sha256'])
  if key in dedup:r['status']='duplicate_alias_excluded';r['duplicate_of_file_index']=dedup[key]
  else:dedup[key]=r['file_index']
 outdir=root/'payloads';outdir.mkdir(exist_ok=True);body_reports=[]
 # Send first useful new quadruped body early while full manifest writes complete.
 ordering=sorted(bodies,key=lambda b:(b!='e457402d4b2e4681b5a5f88b747aea4a',b))
 for b in ordering:
  rs=[r for r in records if r['object_type']==b and r['status']=='passed'];allrs=[r for r in records if r['object_type']==b]
  body_info={'object_type':b,'upstream_category':bodies[b]['upstream_category'],'joint_names':bodies[b]['joint_names'],'topology_hash':specs[b]['topology_hash'],'conditioning_path':bodies[b]['conditioning_path'],'conditioning_sha256':bodies[b]['conditioning_sha256'],'source_files':allrs,'whole_body_holdout':b in HOLDOUT,'native_files':len(allrs),'included_files':len(rs),'included_unique_groups':len({r['source_group_index'] for r in rs}),'frames':sum(r['qc']['frames'] for r in rs),'tpose_fk_error':specs[b]['tpose_fk_error'],'reach':specs[b]['reach'],'units':'canonical_body_units_not_metric','coordinate_conventions':{'up_axis':'Y','handedness':'right','root_translation':'removed in all 3 axes','orientation':'native canonical motion orientation retained; no additional facing rotation'},'descriptor_scale':100,'rotation_layout':'own_joint_global_tpose_relative_column0_then_column1','position_layout':'native_global_positions_minus_native_root_position','normalization':'pose9 = concat(global_rot6, position_canonical / reach); reach=max_root_to_joint_path_rest_length','split_codes':{'train':0,'seen_body_unseen_motion_eval':1,'whole_body_holdout':2},'validity':{'contact':False,'ground':False,'root_height':False,'metric_length':False,'joint_dof_type_limits_axis':False},'exposure':{'old_XWBC_pretraining_body':'not_in_old_XWBC_body_list','target_geometry_seen':'unknown','target_motion_seen':'unknown','per_body_training':'unknown','per_body_finetuning':'unknown','inference_optimization':'unknown'},'evaluations_dimensionless_over_reach_only':True,'surface_association':{'mesh_available':False,'skin_binding_available':False,'local_region_semantics_available':False,'association_status':'native_feature_skeleton_only; raw assets separately published, not bound here'},'conditioning_spec_error':specs[b].get('error')}
  if rs:
   keys=['global_rot6','position_canonical','pose9','fps'];arrays={k:np.concatenate([array_cache[r['file_index']][k] for r in rs]) for k in keys};arrays.update(static[b]);N=len(arrays['pose9'])
   for k,dtype in [('source_group_index','int64'),('file_index','int64'),('split','int8')]:arrays[k]=np.concatenate([np.full(r['qc']['frames'],r[k],dtype=dtype) for r in rs])
   arrays['frame_index']=np.concatenate([np.arange(r['qc']['frames'],dtype='int64') for r in rs])
   arrays['position_valid']=np.ones((N,len(static[b]['parents'])),dtype=bool);arrays['rotation_valid']=np.ones_like(arrays['position_valid'])
   cursor=0
   for r in rs:r['payload_row_range']=[cursor,cursor+r['qc']['frames']];cursor+=r['qc']['frames']
   path=outdir/f'{b}.npz';np.savez_compressed(path,**arrays)
   body_info['serialized_payload_qc']=verify_payload(path,b in HOLDOUT)
   body_info.update(payload_path=str(path),payload_sha256=sha(path),payload_shapes={k:list(v.shape) for k,v in arrays.items()},frame_counts_by_split={str(s):int((arrays['split']==s).sum()) for s in [0,1,2]},group_counts_by_split={str(s):len(set(arrays['source_group_index'][arrays['split']==s].tolist())) for s in [0,1,2]},qc_max_fk_error=max(r['qc']['canonical_fk_error']['max'] for r in rs))
  else:body_info.update(payload_path=None,status='no_FK_valid_motion_quarantined')
  write_json(outdir/f'{b}.json',body_info);body_reports.append(body_info)
  print(json.dumps({'stage':'body_ready','body':b,'payload_path':body_info['payload_path'],'frames':body_info['frames'],'included_groups':body_info['included_unique_groups']}),flush=True)
 for g in group_records:g['split']=group_split.get(g['source_group_index'],-1)
 report={'schema_version':'uniml3d_canonical_payload_v1','revision':REV,'source_repository_commit':'2c5b384715aa63d8639b1ed7eb74bfe614570c7a','adapter_path':str(Path(__file__).resolve()),'adapter_sha256':sha(__file__),'status':'complete','body_count_selected':len(bodies),'body_count_with_payload':sum(b['payload_path'] is not None for b in body_reports),'native_files':len(files),'included_files':sum(r['status']=='passed' for r in records),'duplicate_alias_files':sum(r['status']=='duplicate_alias_excluded' for r in records),'quarantined_files':sum(r['status'].startswith('quarantined') for r in records),'frames_included':sum(b['frames'] for b in body_reports),'source_groups':group_records,'bodies':body_reports,'validation':{'checksum_level':'every_selected_native_NPZ_full_SHA256_and_all_numeric_frames_FK; complete_conditioning_SHA256','split_leakage_checked':True,'group_overlap_count':0,'holdout_training_row_count':0,'raw_source_mutated':False,'native_NPZ_not_UniMate_12D':True,'known_identity_guarantee':'native source basename minus final window index, content SHA, exact native numeric-array SHA union across bodies; unavailable source ancestry cannot be certified','not_claimed':['metric units','ground/contact validity','paired cross-body targets','old checkpoint never saw underlying source motions','UMR optimization','dynamics or physical robot validation']}}
 # Verify group labels and every output native row exactly once before declaring complete.
 assignments={}
 for r in records:
  if r['status']!='passed':continue
  g=r['source_group_index'];assignments.setdefault(g,set()).add(r['split'])
  if r['object_type'] in HOLDOUT and r['split']==0:raise AssertionError('holdout leakage')
 if any(len(x)>1 for x in assignments.values()):raise AssertionError('group leakage')
 write_json(root/'payload_manifest.json',report)
 write_json(root/'all_sequence_qc.json',records)
 print(json.dumps({k:report[k] for k in ['status','body_count_with_payload','native_files','included_files','duplicate_alias_files','quarantined_files','frames_included']}),flush=True)

if __name__=='__main__':main()
