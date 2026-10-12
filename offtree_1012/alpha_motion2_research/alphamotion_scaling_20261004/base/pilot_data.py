"""Strict single-frame data contract; no download, fabricated pair, or unit inference."""
from __future__ import annotations
import hashlib,json,sys
from dataclasses import dataclass,field
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent/'legacy'))
from common import SkeletonSpec,rot6d_to_matrix
from model import hop_distance
from endpoint_policy import resolve_endpoints
from xwbc_contract import validate_xwbc, validate_payload, load_legacy_spec, load_dof


def sha256(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()


def read_json(path):return json.loads(Path(path).read_text())


def write_json(path,value):
 path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
 tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(path)


def fk(rot6,parents,offsets):
 """Own-joint GLOBAL rotations -> root-centered positions; offsets retain body units."""
 R=rot6d_to_matrix(rot6.float());off=torch.as_tensor(offsets,device=R.device,dtype=R.dtype);out=[]
 for j,p in enumerate(parents):
  out.append(R.new_zeros(R.shape[0],3) if p<0 else out[p]+torch.einsum('bij,j->bi',R[:,p],off[j]))
 return torch.stack(out,1)


def angle_degrees(a,b,training=False):
 A=rot6d_to_matrix(a.float());B=rot6d_to_matrix(b.float());Q=A.transpose(-1,-2)@B
 cos=(Q.diagonal(dim1=-2,dim2=-1).sum(-1)-1)*.5
 if training:return torch.acos(cos.clamp(-1+1e-6,1-1e-6))* (180/np.pi)
 skew=torch.stack([Q[...,2,1]-Q[...,1,2],Q[...,0,2]-Q[...,2,0],Q[...,1,0]-Q[...,0,1]],-1)
 return torch.atan2(.5*skew.norm(dim=-1),cos.clamp(-1,1))*(180/np.pi)



@dataclass
class Body:
 name:str
 domain:str
 arrays:dict
 parents:np.ndarray
 offsets:np.ndarray
 descriptor_offsets:np.ndarray
 names:list
 reach:float
 dof:np.ndarray
 dof_valid:np.ndarray
 metric:bool
 metadata:dict
 endpoint:dict=field(init=False)
 static:dict=field(default_factory=dict,init=False)

 def __post_init__(self):
  J=len(self.parents);a=self.arrays;N=len(a['pose9'])
  if not J or not N or self.parents.shape!=(J,) or self.parents.dtype.kind not in 'iu':raise ValueError('empty body or invalid parents')
  if self.parents[0]!=-1 or any(not 0<=p<j for j,p in enumerate(self.parents[1:],1)):
   raise ValueError(self.name+': requires one parent-first tree rooted at 0')
  if a['pose9'].shape!=(N,J,9) or a['global_rot6'].shape!=(N,J,6):raise ValueError('pose shapes')
  if not np.isfinite(a['pose9']).all() or not np.isfinite(a['global_rot6']).all():raise ValueError('nonfinite pose')
  if not np.array_equal(a['pose9'][...,:6],a['global_rot6']):raise ValueError('rotation slot mismatch')
  if not np.isfinite(self.reach) or self.reach<=0:raise ValueError('reach')
  if len(self.names)!=J or len(set(self.names))!=J or self.offsets.shape!=(J,3) or self.descriptor_offsets.shape!=(J,3) or self.dof.shape!=(J,19) or self.dof_valid.shape!=(J,):raise ValueError('static shapes/names')
  if self.dof_valid.dtype!=np.bool_ or not all(np.isfinite(v).all() for v in [self.offsets,self.descriptor_offsets,self.dof]):raise ValueError('nonfinite static or invalid DOF mask')
  for key in ['split','group','pretrain_seen','root_height_m','root_height_valid']:
   if key not in a or a[key].shape!=(N,):raise ValueError('row vector shape '+key)
  if a['split'].dtype.kind not in 'iu' or a['group'].dtype.kind not in 'iu':raise ValueError('split/group dtype')
  for key,mask,shape in [('root_height_m','root_height_valid',(N,)),('contact_sdist','contact_valid',(N,J))]:
   if key not in a or mask not in a or a[key].shape!=shape or a[mask].shape!=shape or a[mask].dtype!=np.bool_:raise ValueError('measurement/mask shape '+key)
   if not np.isfinite(a[key][a[mask]]).all() or not np.isnan(a[key][~a[mask]]).all():raise ValueError('valid measurement must be finite; unknown must be NaN: '+key)
  for name in ['rotation_valid','position_valid']:
   a.setdefault(name,np.ones((N,J),dtype=bool))
   if a[name].shape!=(N,J) or not a[name].all():raise ValueError('v1 rejects partial pose validity; isolate before loader')
  if not set(a['split'].tolist())<={0,1,2}:raise ValueError('unknown split')
  for group in np.unique(a['group']):
   if len(np.unique(a['split'][a['group']==group]))!=1:raise ValueError('group train/eval leakage')
  if np.any(a['split']==2) and np.any(a['split']==0):raise ValueError('body-holdout has train rows')
  self.endpoint=resolve_endpoints(self.name,self.domain,self.names,self.parents,self.offsets)
  # Do not guess a unit conversion from descriptor magnitude.
  self.metadata.update(metric_length_valid=self.metric,endpoint_policy=self.endpoint)

 def setup(self,names_cache,device):
  if any(n not in names_cache for n in self.names):raise ValueError('missing Qwen names '+self.name)
  sem=torch.stack([torch.as_tensor(names_cache[n],dtype=torch.float32) for n in self.names])
  if sem.shape!=(len(self.parents),1024) or not torch.isfinite(sem).all():raise ValueError('Qwen vectors')
  if not torch.allclose(sem.norm(dim=-1),torch.ones(len(self.parents)),atol=2e-5):raise ValueError('Qwen normalization')
  spec=SkeletonSpec(self.name,self.parents,self.descriptor_offsets,self.names);hop=hop_distance(self.parents)
  values={'rest_off':self.descriptor_offsets,'bonelen':spec.norm_bonelen,'depth':hop[0],
          'hop':hop,'dof':self.dof,'sem':sem,
          'metadata_missing':np.stack([~self.dof_valid,np.full(len(self.parents),not self.metric)],-1).astype('float32')}
  self.static={k:torch.as_tensor(v,device=device,dtype=torch.long if k=='hop' else torch.float32) for k,v in values.items()}

 def geom(self,B):return {k:v[None].expand(B,*v.shape) for k,v in self.static.items()}
 def tensor(self,key,rows,device):return torch.as_tensor(self.arrays[key][rows],device=device)
 def eligible(self,train):return np.flatnonzero(self.arrays['split']==0 if train else self.arrays['split']>0)

 def ground_batch(self,rows,device):
  """Actual old labels plus sealed sole geometry; no inferred soft/hard targets."""
  a=self.arrays;N,J=a['pose9'].shape[:2];B=len(rows);g=self.metadata.get('ground_static',{})
  def take(key,default,dtype):
   value=a[key][rows] if key in a else default
   return torch.as_tensor(value,device=device,dtype=dtype)
  soft=take('contact_soft',np.full((B,J),np.nan,np.float32),torch.float32)
  label_valid=take('contact_valid',np.zeros((B,J),bool),torch.bool)&torch.isfinite(soft)
  hard=take('contact_hard',np.zeros((B,J),bool),torch.bool)
  trust=take('contact_trust',np.zeros(B,bool),torch.bool)
  ground_valid=take('ground_valid',np.zeros(B,bool),torch.bool)
  if bool(ground_valid.any()) and not g.get('ground_geometry_verified',False):raise ValueError('valid ground rows lack sealed original sole geometry: '+self.name)
  if not self.metric and (bool(ground_valid.any()) or bool(label_valid.any()) or bool(hard.any())):raise ValueError('UniMate/nonmetric floor supervision is unavailable')
  return {'root_height_m':self.tensor('root_height_m',rows,device).float(),
   'root_height_valid':self.tensor('root_height_valid',rows,device).bool(),
   'contact_soft':soft,'contact_hard':hard,'contact_valid':label_valid,'contact_trust':trust,'ground_valid':ground_valid,
   'contact_threshold_m':torch.as_tensor(g.get('contact_threshold_m',float('nan')),device=device,dtype=torch.float32),
   'sole_j':torch.as_tensor(g.get('sole_j',[]),device=device,dtype=torch.long),
   'sole_off_m':torch.as_tensor(g.get('sole_off_m',[]),device=device,dtype=torch.float32).reshape(-1,3),
   'metric_length_valid':self.metric,'ground_geometry_verified':g.get('ground_geometry_verified',False),
   'mapping_status':g.get('mapping_status','unknown_unavailable')}

 def audit_fk(self,chunk=2048):
  maximum=0.;N=len(self.arrays['pose9'])
  for start in range(0,N,chunk):
   r=torch.from_numpy(self.arrays['global_rot6'][start:start+chunk]);p=fk(r,self.parents,self.offsets).numpy()/self.reach
   e=np.linalg.norm(p-self.arrays['pose9'][start:start+chunk,:,6:],axis=-1)
   if not np.isfinite(e).all():raise ValueError(self.name+': nonfinite FK error')
   maximum=max(maximum,float(e.max()))
  if maximum>2e-5:raise ValueError(f'{self.name}: fixed-skeleton FK mismatch {maximum}')
  return maximum


def load_payload(path):
 with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def load_bodies(root,qwen_path,qwen_sha,device):
 root=Path(root);xd=root/'data_xwbc';ud=root/'data_audit';bodies=[];artifacts={}
 xm,checks,split,row_table,entries,xhashes=validate_xwbc(xd)
 artifacts.update({'data_xwbc/'+k:v for k,v in xhashes.items()})
 um=read_json(ud/'payload_manifest.json');artifacts['data_audit/payload_manifest.json']=sha256(ud/'payload_manifest.json')
 if um['status']!='complete' or um['revision']!='053e8b4b06b6a8ba717a030e675dd5b7fd5be92e':raise ValueError('UniMate audit incomplete')
 if sha256(qwen_path)!=qwen_sha:raise ValueError('pinned Qwen cache checksum')
 qwen=torch.load(qwen_path,map_location='cpu',weights_only=True)
 if not isinstance(qwen,dict) or len(qwen)!=584:raise ValueError('expected pinned 584-name cache')
 for name,e in sorted(entries.items()):
  path=xd/e['payload'];a=load_payload(path)
  validate_payload(name,a,row_table,e)
  s=load_legacy_spec(xd/e['static_files']['spec']['path'],e['static_files']['spec']['sha256']);d=load_dof(xd/e['static_files']['dof']['path'])
  a['group']=a['group_index'];J=e['J']
  body=Body(name,'xwbc',a,s['parents'],s['rest_offsets'],s['rest_offsets'],list(s['joint_names'].astype(str)),float(a['reach_cm']),d['dof'],np.ones(J,dtype=bool),True,{'payload_path':str(path),'payload_sha256':e['payload_sha256'],'length_unit':'cm','source_pairing':'shared verified row table; source_cache_index + source_row_index','static_files':e['static_files'],'static_seal':xm['static_seal'],'ground_static':read_json(xd/e['ground_static']['path'])})
  bodies.append(body)
 for meta in um['bodies']:
  if not meta['payload_path']:continue
  if meta.get('serialized_payload_qc',{}).get('passed') is not True:raise ValueError('UniMate serialized FK audit not passed')
  path=ud/'payloads'/Path(meta['payload_path']).name
  if sha256(path)!=meta['payload_sha256']:raise ValueError('UniMate payload changed')
  a=load_payload(path);a['group']=a['source_group_index'];N,J=a['pose9'].shape[:2]
  if any(np.any(a[k]) for k in ['metric_length_valid','dof_valid','ground_valid','root_height_valid','contact_valid']):raise ValueError('unexpected claimed UniMate mechanics/units')
  a['root_height_valid']=np.zeros(N,dtype=bool);a['root_height_m']=np.full(N,np.nan,dtype='float32')
  a['contact_valid']=np.zeros((N,J),dtype=bool);a['contact_sdist']=np.full((N,J),np.nan,dtype='float32')
  a['contact_soft']=np.full((N,J),np.nan,dtype='float32');a['contact_hard']=np.zeros((N,J),bool)
  a['contact_trust']=np.zeros(N,bool);a['ground_valid']=np.zeros(N,bool)
  a['pretrain_seen']=np.zeros(N,dtype=bool) # use domain to label old-pretraining exposure UNKNOWN, never infer unseen from this sentinel.
  bodies.append(Body(meta['object_type'],'unimate',a,a['parents'],a['rest_offsets'],a['rest_offsets_descriptor'],list(a['joint_names'].astype(str)),float(a['reach']),a['dof_features'],a['dof_valid'],False,meta))
 if not any(b.domain=='unimate' and len(b.eligible(True)) for b in bodies):raise ValueError('no trainable UniMate body')
 # Global group keys within each domain must not cross splits, even across bodies.
 assignments={}
 for b in bodies:
  for group in np.unique(b.arrays['group']):
   split=int(b.arrays['split'][b.arrays['group']==group][0]);key=(b.domain,int(group))
   if key in assignments and assignments[key]!=split:raise ValueError('cross-body source group leakage')
   assignments[key]=split
 audit=[]
 for b in bodies:
  b.setup(qwen,device);fk_error=b.audit_fk()
  audit.append({'body':b.name,'domain':b.domain,'upstream_category':b.metadata.get('upstream_category','human' if b.name=='human_smpl' else 'humanoid_robot'),
   'frames':len(b.arrays['pose9']),'joints':len(b.parents),'unique_groups':len(np.unique(b.arrays['group'])),
   'frames_by_split':{str(s):int((b.arrays['split']==s).sum()) for s in [0,1,2]},'metric':b.metric,'fk_max_error_over_reach':fk_error,
   'endpoint':b.endpoint,'payload_sha256':b.metadata['payload_sha256'],
   'motion_representation':'global rotations plus fixed rest offsets; root translation removed',
   'mechanical_axes_limits_known':bool(b.dof_valid.all()),'prismatic_or_variable_bone_output_supported':False,
   'mechanical_joint_type':'rotational descriptor only' if b.domain=='xwbc' else 'unknown; acceptance only establishes observed fixed-offset FK compatibility',
   'mimic_or_gripper_constraint_enforcement':False,'dexterous_hand_or_gripper_mapping_validated':False})
 quarantine=[{'body':m['object_type'],'category':m['upstream_category'],'included_files':m['included_files'],
  'quarantined_files':sum(f['status'].startswith('quarantined') for f in m['source_files']),
  'quarantine_reasons':[{'file':f['rfilename'],'status':f['status'],'qc':f.get('qc'),'error':f.get('error')} for f in m['source_files'] if f['status'].startswith('quarantined')],
  'unsupported_motion_policy':'Rotation-only fixed-offset FK mismatch is excluded; a rigid-arm category is not proof of prismatic/mimic or hand support.'} for m in um['bodies']]
 return bodies,{'body_audit':audit,'input_manifest_hashes':artifacts,'qwen_sha256':qwen_sha,
  'verified_within_domain_group_overlap':0,'cross_dataset_source_ancestry':'unknown; not certified by filenames/captions/frame numbers',
  'unimate_file_qc':quarantine,'unimate_file_counts':{k:um[k] for k in ['native_files','included_files','duplicate_alias_files','quarantined_files','frames_included']},
  'xwbc_terminal_checks':checks,'old_manifest_schema_keys':sorted(xm)}


def sample_rows(body,rng,batch,train=True):
 eligible=body.eligible(train);groups=np.unique(body.arrays['group'][eligible])
 if not len(groups):raise ValueError('empty split '+body.name)
 chosen=rng.choice(groups,size=batch);return np.asarray([rng.choice(eligible[body.arrays['group'][eligible]==g]) for g in chosen],dtype=np.int64)


def make_inputs(src,tgt,rows,rng,device,mode='full',condition_count=None,condition_dropout=.35,contact_dropout=.3):
 B=len(rows);x=src.tensor('pose9',rows,device).float();target=tgt.tensor('pose9',rows,device).float()
 hidden=torch.zeros(x.shape[:2],dtype=torch.bool,device=device)
 if mode=='masked':hidden=torch.as_tensor(rng.random(x.shape[:2])<.6,device=device)
 elif mode=='sparse_only':hidden[:]=True
 elif mode!='full':raise ValueError(mode)
 con=src.tensor('contact_sdist',rows,device).float();cv=src.tensor('contact_valid',rows,device).bool()
 # Hidden pose never carries per-joint contact side information in this pilot.
 drop=torch.as_tensor(rng.random((B,1))<contact_dropout,device=device)
 con=torch.where(cv & ~hidden & ~drop,con,torch.full_like(con,float('nan')))
 if mode=='sparse_only':con=None
 indices=np.full((B,2),-1,dtype=np.int64);valid=np.zeros((B,2),dtype=bool)
 endpoints=tgt.endpoint['indices']
 for i in range(B):
  count=min(len(endpoints),int(condition_count) if condition_count is not None else int(rng.integers(1,3)))
  if condition_count is None and rng.random()<condition_dropout:count=0
  # Always consume the full permutation: explicit one/two-condition evals share a prefix.
  permutation=rng.permutation(endpoints)
  if count:indices[i,:count]=permutation[:count];valid[i,:count]=True
 ix=torch.as_tensor(indices,device=device);v=torch.as_tensor(valid,device=device)
 values=torch.full((B,2,9),float('nan'),device=device)
 for i in range(B):
  if bool(v[i].any()):values[i,v[i]]=target[i,ix[i,v[i]]]
 # Only these selected sparse values are passed to model.forward; full target stays in loss/eval code.
 condition={'joint_indices':ix,'valid':v,'values':values}
 inputs={'source':x,'source_hidden':hidden,'source_contact':con,'condition':condition}
 if 'slide_normalized' in src.arrays:
  inputs['source_slide_normalized']=src.tensor('slide_normalized',rows,device).float()
 return inputs,target
