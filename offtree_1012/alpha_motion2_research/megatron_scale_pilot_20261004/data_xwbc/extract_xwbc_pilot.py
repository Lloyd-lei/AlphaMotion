#!/usr/bin/env python3
"""Bounded CPU-only extraction; original XWBC sources/checkpoints remain read-only."""
from pathlib import Path
import os,sys,json,time,hashlib,collections,shutil
import numpy as np
import torch
BASE=Path('/home/arenalabs/Desktop')/'"be water, robot"'
SRC=BASE/'pose_translator_design/exp_topo'
sys.path.insert(0,str(SRC))
from common import SkeletonSpec,fk_global,matrix_to_rot6d
from train_ae import static_cond
torch.set_num_threads(2);torch.set_num_interop_threads(1)
OUT=Path(__file__).resolve().parent
CACHE=Path('/var/tmp/alphamotion/topo_exp')
CACHES=[CACHE/'cache_lafan_v3',CACHE/'cache_amass_lconv_tfix']
REF=CACHE/'runs/L_M20k_0903'
EX=CACHE/'cache_amass_heldout/sequences.json'
SEED=20261004;FPS=30.0;MAX_FRAMES=96;MIN_STRIDE=3
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for c in iter(lambda:f.read(8<<20),b''):h.update(c)
 return h.hexdigest()
def write(n,d):
 p=OUT/n;tmp=p.with_suffix(p.suffix+'.tmp')
 tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def stat(p):
 q=Path(p);s=q.stat()
 return {'path':str(q),'bytes':s.st_size,'mtime_ns':s.st_mtime_ns}
def pick(seq,n):
 by=collections.defaultdict(list)
 for q in seq:by[q['name'].split('__')[0]].append(q)
 h=lambda s:hashlib.sha256((str(SEED)+'|'+s).encode()).hexdigest()
 for k in by:by[k].sort(key=lambda q:h(q['name']))
 out=[];ks=sorted(by,key=h)
 while len(out)<n:
  for k in ks:
   if by[k] and len(out)<n:out.append(by[k].pop())
 return out
for n in ['payloads','static']:(OUT/n).mkdir(exist_ok=True)
man=[json.loads((c/'manifest.json').read_text()) for c in CACHES]
seq=[json.loads((c/'sequences.json').read_text()) for c in CACHES]
excluded={q['name'] for q in json.loads(EX.read_text())}
assert len(excluded)==600
names=[n for n in man[0]['embodiments'] if n not in ['pal_talos','engineai_pm01']]
assert len(names)==16 and man[0]['embodiments']==man[1]['embodiments']
for body in names:
 for c in CACHES:
  assert (c/f'{body}_paired_local6d.pt').is_file()
 assert (CACHES[0]/f'{body}_dof.npz').is_file()
old_provenance=[]
for run in [REF.parent/'L_M20k_0903_s1',REF]:
 cfg=json.loads((run/'config.json').read_text());args=(run/'launch_args.txt').read_text()
 assert cfg['exclude_set']==str(EX) and str(EX) in args
 old_provenance.append({'run':str(run),'config_sha256':sha(run/'config.json'),'launch_args_sha256':sha(run/'launch_args.txt'),'exclude_set':cfg['exclude_set'],'steps':cfg['steps']})
atr=pick([q for q in seq[1] if q['name'] not in excluded],256)
aev=pick([q for q in seq[1] if q['name'] in excluded],64)
laf=sorted(seq[0],key=lambda q:hashlib.sha256((str(SEED)+'|lafan|'+q['name']).encode()).hexdigest())
lev=laf[:8];ltr=laf[8:]
groups=[]
for ci,qs,split in [(0,ltr,0),(0,lev,1),(1,atr,0),(1,aev,1)]:
 for q in qs:
  group_id=('lafan::' if ci==0 else 'amass::')+q['name']
  pretrain_seen=not(ci==1 and q['name'] in excluded)
  k=min(MAX_FRAMES,(int(q['len'])-1)//MIN_STRIDE+1)
  frames=np.floor(np.linspace(0,q['len']-1,k)).astype(np.int64) if k>1 else np.array([0],np.int64)
  assert len(np.unique(frames))==k and (k<2 or np.min(np.diff(frames))>=MIN_STRIDE)
  groups.append({'group_id':group_id,'clip':q['name'],'source_dataset':q['name'].split('__')[0] if ci==1 else 'LAFAN',
   'cache_index':ci,'start':int(q['start']),'length':int(q['len']),'split':split,
   'pretrain_seen':pretrain_seen,'pretrain_excluded':not pretrain_seen,'frame_indices':frames.tolist(),'chosen_frame_count':k})
groups.sort(key=lambda q:(q['cache_index'],q['start']))
assert len(groups)==397
train_ids={g['group_id'] for g in groups if g['split']==0};eval_ids={g['group_id'] for g in groups if g['split']==1}
assert train_ids.isdisjoint(eval_ids)
assert not ({g['clip'] for g in groups if g['split']==0 and g['cache_index']==1}&excluded)
rows=[];offsets=[0,man[0]['rows']]
for gi,g in enumerate(groups):
 for fr in g['frame_indices']:
  rows.append((g['cache_index'],g['start']+fr,offsets[g['cache_index']]+g['start']+fr,gi,fr,g['split'],g['pretrain_seen']))
tab=np.asarray(rows,dtype=np.int64);N=len(tab);assert N==30322
ci=tab[:,0];local_rows=tab[:,1];row_ids=tab[:,2];group_index=tab[:,3];frame_index=tab[:,4]
split=tab[:,5].astype(np.int8);pretrain_seen=tab[:,6].astype(bool);timestamps=frame_index.astype(np.float64)/FPS
assert np.unique(row_ids).size==N
row_hash=hashlib.sha256(tab.tobytes()).hexdigest()
np.savez_compressed(OUT/'row_table.npz',cache_index=ci.astype(np.int8),source_row_index=local_rows,row_ids=row_ids,group_index=group_index,frame_index=frame_index,timestamp_seconds=timestamps,split=split,pretrain_seen=pretrain_seen)
split_meta={'seed':SEED,'selection':'AMASS source-dataset round robin, deterministic SHA256 order within each source; full source clip is the split unit.',
 'groups':groups,'body_names':names,'row_table_sha256':sha(OUT/'row_table.npz'),'row_table_content_sha256':row_hash,'rows':N,
 'counts':{'amass_train_groups':256,'amass_eval_groups':64,'lafan_train_groups':69,'lafan_eval_groups':8,'train_rows':int((split==0).sum()),'eval_rows':int((split==1).sum())},
 'fps':FPS,'max_frames_per_clip':MAX_FRAMES,'minimum_source_frame_stride':MIN_STRIDE,
 'pretrain_seen_definition':'Clip was eligible in the old pretraining corpus; not a claim that every selected frame was sampled by its random mini-batches.',
 'pretrain_heldout_provenance':old_provenance,'exclude_set_sha256':sha(EX),
 'source_sequence_files':[{'path':str(c/'sequences.json'),'sha256':sha(c/'sequences.json'),'manifest_sha256':sha(c/'manifest.json')} for c in CACHES],
 'no_leakage_checks':{'train_eval_group_intersection':0,'train_eval_row_intersection':0,'amass_train_in_old_exclude_set':0,'amass_eval_all_in_old_exclude_set':True,'same_row_table_for_all_bodies':True}}
write('chosen_split.json',split_meta)
write('progress.json',{'status':'running','rows':N,'completed':[]})
print(json.dumps({'stage':'selected','rows':N,**split_meta['counts']}),flush=True)
checks=[];body_manifest=[];reach_records=[];start_all=time.monotonic()
source_code=[{'path':str(SRC/n),'sha256':sha(SRC/n)} for n in ['common.py','train_global.py','dual_common.py','train_ae.py','ladder.sh','train_dual.py']]
for bi,body in enumerate(names):
 t0=time.monotonic()
 zs=np.load(CACHES[0]/f'{body}_spec.npz',allow_pickle=True)
 spec=SkeletonSpec(str(zs['name']),zs['parents'],zs['rest_offsets'],list(zs['joint_names']))
 J=spec.J;root=int(np.where(spec.parents<0)[0][0])
 parts=[torch.load(c/f'{body}_paired_local6d.pt',map_location='cpu',mmap=True,weights_only=True) for c in CACHES]
 assert all(tuple(v.shape)==(man[i]['rows'],J,6) for i,v in enumerate(parts))
 maxima=[]
 # Inherit the old recipe's global normalization domain, including excluded groups.
 for i,(c,x) in enumerate(zip(CACHES,parts)):
  scan_t=time.monotonic();mx=0.
  with torch.no_grad():
   for k in range(0,len(x),8192):
    _,gp=fk_global(x[k:k+8192].float(),spec)
    mx=max(mx,float(torch.linalg.vector_norm(gp,dim=-1).max()))
  rec={**stat(c/f'{body}_paired_local6d.pt'),'body':body,'rows':len(x),'shape':list(x.shape),'reach_cm':max(mx,1e-6),
   'method':'all source rows, CPU float32 FK, chunk8192, torch threads2, first-cache reference spec','elapsed_seconds':time.monotonic()-scan_t}
  maxima.append(rec);reach_records.append(rec)
 reach=max(v['reach_cm'] for v in maxima)
 gr6=np.empty((N,J,6),np.float32);pos=np.empty((N,J,3),np.float32)
 for i,x in enumerate(parts):
  where=np.flatnonzero(ci==i);sel=local_rows[where]
  with torch.no_grad():
   for k in range(0,len(where),4096):
    wr=where[k:k+4096];lr=sel[k:k+4096]
    R,P=fk_global(x[torch.from_numpy(lr)].float(),spec)
    gr6[wr]=matrix_to_rot6d(R).numpy();pos[wr]=P.numpy()
 pose9=np.concatenate([gr6,pos/reach],axis=-1)
 root_h=np.full(N,np.nan,np.float32);rh_valid=np.zeros(N,bool)
 sd=np.full((N,J),np.nan,np.float32);cv=np.zeros((N,J),bool)
 cs=np.full((N,J),np.nan,np.float32);ch=np.zeros((N,J),bool);trust=np.zeros(N,bool)
 contact_sources=[]
 for i,c in enumerate(CACHES):
  f=c/f'{body}_contact.npz';wh=np.flatnonzero(ci==i);sel=local_rows[wh]
  if not f.exists():
   contact_sources.append({'path':str(f),'exists':False});continue
  with np.load(f) as z:
   hdr=json.loads(str(z['header']))
   assert hdr['N']==man[i]['rows'] and hdr['J']==J and hdr['joint_names']==list(spec.joint_names)
   assert abs(float(hdr.get('h0_m',0)))<.30
   gap=z['gap'][sel].astype(np.float32)
   tr=z['trust'][sel].astype(bool) if 'trust' in z.files else np.zeros(len(sel),bool)
   soft=z['contact_soft'][sel].astype(np.float32);hard=z['contact'][sel].astype(bool)
   valid=tr[:,None]&np.isfinite(gap)&np.isfinite(soft)
   thr=float(hdr['thr_m']);D=float(hdr['D_used_m'])
   score=thr-gap;score=np.sign(score)*np.log1p(np.abs(score)/thr)/np.log1p(D/thr)
   # Match training ContactStore storage precision: signed-distance source is fp16.
   score=score.astype(np.float16).astype(np.float32)
   pen=gap<0;soft=np.maximum(soft,pen.astype(np.float32));hard=hard|pen
   sd[wh]=np.where(valid,score,np.nan);cv[wh]=valid;trust[wh]=tr
   cs[wh]=np.where(valid,soft,np.nan);ch[wh]=hard&valid
   hv=tr&np.isfinite(gap[:,root]);root_h[wh]=np.where(hv,gap[:,root],np.nan);rh_valid[wh]=hv
  contact_sources.append({**stat(f),'header':hdr,'exists':True,'selected_trusted_rows':int(tr.sum()),'selected_rows':len(wh)})
 payload={'pose9':pose9,'global_rot6':gr6,'position_cm':pos,'root_height_m':root_h,'root_height_valid':rh_valid,
  'contact_sdist':sd,'contact_valid':cv,'contact_soft':cs,'contact_hard':ch,'contact_trust':trust,
  'row_ids':row_ids,'source_cache_index':ci.astype(np.int8),'source_row_index':local_rows,
  'group_index':group_index,'frame_index':frame_index,'timestamp_seconds':timestamps,
  'split':split,'pretrain_seen':pretrain_seen,'reach_cm':np.asarray(reach,np.float64)}
 f=OUT/'payloads'/f'{body}.npz';np.savez_compressed(f,**payload)
 for kind in ['spec','dof']:
  origin=CACHES[0]/f'{body}_{kind}.npz';dest=OUT/'static'/origin.name;shutil.copy2(origin,dest)
  assert sha(origin)==sha(dest)
 st=static_cond(spec,'cpu')
 desc={'body':body,'J':J,'joint_names':list(spec.joint_names),'parents':spec.parents.tolist(),'rest_offsets_cm':spec.rest_offsets.tolist(),
  'norm_bonelen':st['bonelen'].numpy().tolist(),'depth':st['depth'].numpy().tolist(),'hop':st['hop'].numpy().tolist(),
  'dof_shape':list(np.load(OUT/'static'/f'{body}_dof.npz')['dof'].shape),'reach_cm':reach,'root_index':root,
  'spec_policy':'Independent byte-identical copy of first-cache LAFAN spec/dof, exactly as the reference loader.',
  'source_spec_sha256':sha(CACHES[0]/f'{body}_spec.npz'),'legacy_h2_warning':'Anonymous body00..body31 tree; not official H2 J36.' if body=='unitree_h2' else None}
 write('static/'+body+'_descriptor.json',desc)
 check={'body':body,'rows':N,'J':J,'pose9_all_finite':bool(np.isfinite(pose9).all()),'position_root_max_abs_cm':float(np.abs(pos[:,root]).max()),
  'pose9_position_formula_max_abs_error':float(np.max(np.abs(pose9[...,6:]-pos/reach))),
  'invalid_contact_is_nan':bool(np.isnan(sd[~cv]).all()),'invalid_root_height_is_nan':bool(np.isnan(root_h[~rh_valid]).all()),
  'contact_valid_values_finite':bool(np.isfinite(sd[cv]).all()),'root_height_valid_count':int(rh_valid.sum()),
  'contact_valid_fraction':float(cv.mean()),'row_content_sha256':row_hash,'elapsed_seconds':time.monotonic()-t0}
 assert check['pose9_all_finite'] and check['position_root_max_abs_cm']==0 and check['pose9_position_formula_max_abs_error']==0
 checks.append(check)
 body_manifest.append({'body':body,'J':J,'payload':str(f.relative_to(OUT)),'payload_sha256':sha(f),'payload_bytes':f.stat().st_size,
  'reach_cm':reach,'source_rotations':maxima,'source_contacts':contact_sources,'static_descriptor':f'static/{body}_descriptor.json'})
 write('reach_scan.json',reach_records);write('progress.json',{'status':'running','rows':N,'completed':names[:bi+1],'last_check':check})
 print(json.dumps({'stage':'body_complete','body':body,'index':bi+1,'of':len(names),'reach_cm':reach,'MiB':round(f.stat().st_size/2**20,2),'seconds':round(time.monotonic()-t0,2),'trusted_rows':int(rh_valid.sum())}),flush=True)
 del parts,pose9,gr6,pos,sd,cv,cs,ch,payload
total=sum(f.stat().st_size for f in OUT.rglob('*') if f.is_file())
assert total<2**30,(total,total/2**30)
final={'status':'complete','rows':N,'bodies':len(names),'fps':FPS,'total_bytes_before_final_manifest':total,'body_payloads':body_manifest,
 'normalizer_policy':'Full original configured training-cache maxima using first-cache spec. Includes previously excluded eval clips only in inherited normalization statistics, matching the old checkpoint preprocessing; never refit from pilot.',
 'contract':{'pose9':'float32[N,J,9] = global rotation6 + root-relative position_cm/reach_cm','global_rot6':'float32[N,J,6], first two matrix columns in repository convention','position_cm':'float32[N,J,3], Y-up, target/body root-relative','root_height_m':'float32[N], actual contact gap of root; unknown/untrusted is NaN','root_height_valid':'bool[N], trusted finite metric gap only','contact_sdist':'float32[N,J], exact legacy signed log distance transformed then fp16-rounded; unknown/untrusted is NaN','contact_valid':'bool[N,J]','contact_soft':'float32[N,J], max(actual soft contact, gap<0), invalid NaN','contact_hard':'bool[N,J], actual contact OR gap<0, invalid false with explicit validity mask','row_ids':'int64[N], original LAFAN+AMASS concatenated row id','split':'int8[N], train0/eval1','reach_cm':'float64 scalar; original full-cache normalization inherited'},
 'chosen_split_sha256':sha(OUT/'chosen_split.json'),'row_table_sha256':sha(OUT/'row_table.npz'),
 'source_code':source_code,'checks':checks,
 'scope_notes':['No training or model forward executed. Only source FK and metadata extraction.','AMASS eval64 are excluded from both old training stages; LAFAN eval8 are only pilot-heldout and explicitly pretrain_seen.',
 'Same source group and timestamp map to every body, and no selected group crosses pilot splits.',
 'Sparse sampling at minimum 3 original frames means these rows are not adjacent-frame training pairs. Use timestamp deltas if any derivative loss is later enabled.',
 'No original arrays or extracted payloads were copied to the local workstation.',
 'Only row-level provenance established; retargeting quality and physical execution are not validated.',
 'Freeform action captions are not supplied here; source names must not silently become gold text labels.']}
write('manifest.json',final);write('checks.json',{'all_pass':True,'checks':checks,'split_checks':split_meta['no_leakage_checks']})
write('progress.json',{'status':'complete','rows':N,'completed':names,'elapsed_seconds':time.monotonic()-start_all,'total_bytes':sum(f.stat().st_size for f in OUT.rglob('*') if f.is_file())})
print(json.dumps({'stage':'complete','rows':N,'bodies':16,'total_GiB':sum(f.stat().st_size for f in OUT.rglob('*') if f.is_file())/2**30,'seconds':time.monotonic()-start_all,'manifest_sha256':sha(OUT/'manifest.json')}),flush=True)
