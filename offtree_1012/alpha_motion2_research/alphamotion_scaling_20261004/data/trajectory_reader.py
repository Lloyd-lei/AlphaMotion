#!/usr/bin/env python3
"""Read full source clips lazily; original caches remain read-only.
XWBC arrays returned in canonical Y-up cm, metric=True. UniMate units stay native.
Contact NPZ reads only contiguous row bytes; compressed members may require sequential
inflation to reach a row but never allocate the full array. No full tensor scan.
"""
from pathlib import Path
import argparse,json,hashlib,zipfile
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def static_reach(parents,off):
 d=np.zeros(len(parents))
 for j,p in enumerate(parents):
  if p>=j:raise ValueError('parents not topological')
  if p>=0:d[j]=d[p]+np.linalg.norm(off[j])
 if d.max()<=0:raise ValueError('zero geometry reach')
 return float(d.max())
def npz_rows(path,key,start,length):
 """Bounded contiguous C-order numeric NPY slice inside NPZ; reject objects/Fortran."""
 with zipfile.ZipFile(path) as z,z.open(key+'.npy') as f:
  version=np.lib.format.read_magic(f)
  reader=np.lib.format.read_array_header_1_0 if version==(1,0) else np.lib.format.read_array_header_2_0
  shape,fortran,dtype=reader(f)
  if dtype.hasobject or fortran or len(shape)<1:raise ValueError('requires C-order numeric row array')
  if start<0 or length<0 or start+length>shape[0]:raise IndexError('row bounds')
  stride=int(np.prod(shape[1:],dtype=np.int64))*dtype.itemsize
  f.seek(start*stride,1);raw=f.read(length*stride)
  if len(raw)!=length*stride:raise ValueError('short array member')
  return np.frombuffer(raw,dtype).reshape((length,)+shape[1:]).copy()
def rot6matrix(x):
 a=x[...,:3];b=x[...,3:];a=a/np.linalg.norm(a,axis=-1,keepdims=True);b=b-(a*b).sum(-1,keepdims=True)*a;b/=np.linalg.norm(b,axis=-1,keepdims=True)
 return np.stack([a,b,np.cross(a,b)],axis=-1)
def local_fk(local6,parents,off):
 r=rot6matrix(local6);g=np.empty_like(r);p=np.zeros(r.shape[:-2]+(3,),r.dtype)
 for j,pa in enumerate(parents):
  g[:,j]=r[:,j] if pa<0 else g[:,pa]@r[:,j]
  if pa>=0:p[:,j]=p[:,pa]+np.einsum('nab,b->na',g[:,pa],off[j])
 return g,p

def load_xwbc_trajectory(manifest,group_id,body,with_contact=True):
 import torch
 manifest=json.loads(Path(manifest).read_text()) if isinstance(manifest,(str,Path)) else manifest
 x=manifest['xwbc'];b=next(b for b in x['bodies'] if b['body']==body);g=next(g for g in x['groups'] if g['group_id']==group_id)
 spec=Path(b['static_spec'])
 if sha(spec)!=b['static_spec_sha256']:raise ValueError('trusted source spec hash changed')
 with np.load(spec,allow_pickle=True) as s:
  parents=np.asarray(s['parents'],np.int64);off=np.asarray(s['rest_offsets'],np.float32);names=s['joint_names'].astype(str)
 cache=b['caches'][g['cache_index']];start=g['start'];length=g['length'];sl=slice(start,start+length)
 t=torch.load(cache['local_rot6_tensor'],weights_only=True,mmap=True,map_location='cpu');local=t[sl].float().numpy().copy();del t
 R,P=local_fk(local,parents,off)
 rw=torch.load(cache['root_world_cm_tensor'],weights_only=True,mmap=True,map_location='cpu')[sl].float().numpy().copy()
 assert rw.shape==(length,3) and np.isfinite(rw).all()
 reach=static_reach(parents,off);gr6=np.concatenate([R[..., :,0],R[..., :,1]],axis=-1);frames=np.arange(length)
 result={'pose9':np.concatenate([gr6,P/reach],axis=-1),'global_rot6':gr6,'global_rotation_matrices':R,'position_native':P,'world_root':rw,'world_positions':P+rw[:,None,:],'parents':parents,'rest_offsets':off,'joint_names':names,'reach':np.array(reach),'frame_index':frames,'source_row_index':frames+start,'timestamp_seconds':frames/float(x['fps']),'fps':np.array(x['fps']), 'metric_length_valid':np.array(True)}
 ground=json.loads(Path(b['static_ground']).read_text());result['sole_j']=np.array(ground['sole_j'],np.int64);result['sole_off_m']=np.array(ground['sole_off_m'],np.float32).reshape(-1,3)
 if with_contact:
  path=cache['contact_npz']
  with np.load(path,allow_pickle=False) as f:header=json.loads(str(f['header']));hastrust='trust' in f.files
  if header['joint_names']!=names.tolist():raise ValueError('contact/spec identity mismatch')
  gap=npz_rows(path,'gap',start,length).astype('float32');soft=npz_rows(path,'contact_soft',start,length).astype('float32');hard=npz_rows(path,'contact',start,length).astype(bool)
  trust=npz_rows(path,'trust',start,length).astype(bool) if hastrust else np.ones(length,bool)
  valid=trust[:,None]&np.isfinite(gap)&np.isfinite(soft);hard=(hard|(gap<0))&valid;soft=np.maximum(soft,(gap<0).astype('float32'))
  root=int(np.flatnonzero(parents<0)[0]);hv=trust&np.isfinite(gap[:,root])
  result.update(contact_hard=hard,contact_soft=np.where(valid,soft,np.nan),contact_valid=valid,contact_gap_m=gap,contact_trust=trust,root_height_m=np.where(hv,gap[:,root],np.nan),root_height_valid=hv,ground_valid=hv&trust)
  js=result['sole_j'];result['world_sole_positions_cm']=result['world_positions'][:,js]+np.einsum('nsab,sb->nsa',R[:,js],result['sole_off_m']*100)
 result['metadata']={'body':body,'group_id':group_id,'clip_id':g['clip'],'body_exposure':'heldout_body' if b['heldout_body'] else 'seen_body','clip_exposure':'train_clip' if g['split']==0 else 'heldout_clip','task_kind':'paired_mapping','reference_kind':'paired_target','target_morphology':'humanoid','length_unit':'cm','up_axis':'Y','position_frame':'root_relative; world arrays separately named','world_orientation':'original cache global rotation; no heading canonicalization','source_frame_count':length,'frame_semantics':'every original contiguous row start..start+length-1','contact_source':cache['contact_npz'] if with_contact else None,'sole_source':b['static_ground'],'full_source_numeric_checksum':False}
 return result

def load_unpaired_trajectory(manifest,body,group_index):
 """Complete stored own-body sequence. Explicit unit conversion; never cross-body GT."""
 manifest=json.loads(Path(manifest).read_text()) if isinstance(manifest,(str,Path)) else manifest
 ub=next((b for b in manifest['unimate']['bodies'] if b['object_type']==body),None)
 if ub is not None:
  path=Path(ub['payload_path']);records=[r for r in ub['source_sequences'] if r['status']=='passed' and r['source_group_index']==group_index]
  if len(records)!=1:raise ValueError('expected one unique included complete source sequence')
  r=records[0];lo,hi=r['payload_row_range'];metric=False;scale=1.;pkey='position_canonical';gid=r['source_group_id'];source=ub['object_type'];fps=r['fps'];sp=r['split'];worldroot=False
 else:
  ext=manifest.get('external',manifest.get('external_frozen_manifest'));b=next(b for b in ext['bodies'] if b['body_id']==body)
  md=json.loads(Path(b['manifest']).read_text());path=Path(b['payload']);episodes=md['episodes'];groups=sorted({e['source_group'] for e in episodes});gid=groups[group_index];r=next(e for e in episodes if e['source_group']==gid);lo,hi=r['row_range'];metric=True;scale=100.;pkey='position_native';source=md['source_dataset'];fps=100 if body=='adroit_pen' else 500;sp=r['split'];worldroot=True
 n=hi-lo
 with np.load(path,allow_pickle=False) as z:parents=z['parents'];off=z['rest_offsets']*scale;names=z['joint_names'];reach=float(z['reach'])*scale;ee=z['ee_indices']
 G=npz_rows(path,'global_rot6',lo,n);P=npz_rows(path,pkey,lo,n)*scale;R=rot6matrix(G.astype('float64'));fr=npz_rows(path,'frame_index',lo,n)
 if not np.array_equal(fr,np.arange(n)):raise ValueError('source sequence frame order/gaps')
 ts=npz_rows(path,'timestamp_seconds',lo,n) if worldroot else fr/float(fps)
 out={'global_rot6':G,'global_rotation_matrices':R,'position_native':P,'pose9':np.concatenate([G,P/reach],axis=-1),'parents':parents,'rest_offsets':off,'joint_names':names,'reach':np.array(reach),'frame_index':fr,'timestamp_seconds':ts,'fps':np.array(fps),'ee_indices':ee,'metric_length_valid':np.array(metric),'contact_valid':np.zeros((n,len(names)),bool),'ground_valid':np.zeros(n,bool)}
 if worldroot:
  with np.load(path,allow_pickle=False) as z:
   if 'joint_type' in z.files:
    for k in ['joint_type','prismatic_axis','slide_mask','prismatic_fixed_rotation']:out[k]=z[k]
    for k in ['slide_lower','slide_upper','slide_reference']:out[k]=z[k]*scale
    out['slide_normalized']=npz_rows(path,'slide_normalized',lo,n)
    out['slide_valid']=npz_rows(path,'slide_valid',lo,n)
    out['slide_position']=npz_rows(path,'slide_position_m',lo,n)*scale
    out['aperture_cm']=npz_rows(path,'aperture_m',lo,n)*scale
  AX=np.array(md['model']['AX_zup_to_yup']);rw=npz_rows(path,'world_root_m_zup',lo,n)@AX.T*100;out['world_root']=rw;out['world_positions']=P+rw[:,None,:]
  out['world_rotation_matrices']=R@(AX@np.array(md['model']['rest_body_rotations_world_zup'])@AX.T)
 out['metadata']={'body':body,'source_dataset':source,'group_id':gid,'source_frame_count':n,'task_kind':'reconstruction','reference_kind':'same_body','paired_cross_body_gt':False,'body_exposure':'heldout_body' if sp==2 else 'seen_body','clip_exposure':'train_clip' if sp==0 else 'heldout_clip','length_unit':'cm' if metric else 'canonical_body_units_not_metric','up_axis':'Y','frame_semantics':'all consecutive saved native states of one complete group','metric':metric,'position_frame':'root relative; world available only where explicitly returned'}
 return out

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--group',required=True);p.add_argument('--body',required=True);p.add_argument('--out');p.add_argument('--without-contact',action='store_true');a=p.parse_args()
 d=load_xwbc_trajectory(a.manifest,a.group,a.body,not a.without_contact)
 if a.out:np.savez_compressed(a.out,**{k:v for k,v in d.items() if k!='metadata'});Path(a.out).with_suffix('.json').write_text(json.dumps(d['metadata'],indent=2)+'\n')
 print(json.dumps({'metadata':d['metadata'],'shapes':{k:list(v.shape) for k,v in d.items() if isinstance(v,np.ndarray)}},indent=2))
