#!/usr/bin/env python3
"""Pinned, offline, frozen Qwen embedding preparation; never writes shared caches."""
import argparse,ast,datetime,hashlib,importlib.metadata,json,os,re,sys,time,gc,subprocess
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
import numpy as np
import torch
from transformers import AutoTokenizer,AutoModel
HERE=Path(__file__).resolve().parent
REV='97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3'
SNAP=Path('/var/tmp/alphamotion/hf_cache/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots')/REV
LEGACY=Path('/var/tmp/alphamotion/topo_exp/qwen3_names.pt')
SOURCE=Path('/home/arenalabs/Desktop/"be water, robot"/pose_translator_design/exp_topo/semantics.py')
CACHE=HERE/'qwen3_names_pinned.pt'
MANIFEST=HERE/'qwen3_names_pinned.manifest.json'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def hbytes(b):return hashlib.sha256(b).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def gpu():
 def q(args):
  p=subprocess.run(['nvidia-smi']+args,capture_output=True,text=True)
  return {'returncode':p.returncode,'stdout':p.stdout.strip(),'stderr':p.stderr.strip()}
 return {'time_utc':utc(),'gpu':q(['--query-gpu=index,name,memory.used,memory.free,utilization.gpu','--format=csv,noheader']),'processes':q(['--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'])}
def atomic_json(p,obj):
 tmp=p.with_name(p.name+'.tmp-'+str(os.getpid()))
 tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');os.replace(tmp,p)
def vecsha(t):return hbytes(t.detach().cpu().to(torch.float32).contiguous().numpy().astype('<f4',copy=False).tobytes())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--names-json');ap.add_argument('--batch-size',type=int,default=16);args=ap.parse_args()
 assert 1<=args.batch_size<=32
 started=utc();t0=time.monotonic();legacy_sha=sha(LEGACY)
 if args.names_json:
  sourcep=Path(args.names_json);obj=json.loads(sourcep.read_text())
  names=obj if isinstance(obj,list) else obj.get('names',obj.get('joint_names'))
  assert isinstance(names,list),'names JSON must be a string list or dict names/joint_names'
  source_info={'kind':'confirmed_additional_joint_names','path':str(sourcep),'sha256':sha(sourcep)}
 else:
  legacy=torch.load(LEGACY,map_location='cpu',weights_only=True)
  names=list(legacy.keys());assert len(names)==380,len(names)
  del legacy
  source_info={'kind':'legacy_keys_only_reencoded_no_legacy_vectors_reused','path':str(LEGACY),'sha256':legacy_sha}
 assert all(isinstance(n,str) for n in names)
 names=list(dict.fromkeys(names))
 lookup={};entries={};runs=[]
 if CACHE.exists() or MANIFEST.exists():
  assert CACHE.exists() and MANIFEST.exists(),'partial previous cache state requires inspection'
  prior=json.loads(MANIFEST.read_text());assert sha(CACHE)==prior['outputs']['torch_cache']['sha256']
  assert prior['model']['revision']==REV
  lookup=torch.load(CACHE,map_location='cpu',weights_only=True)
  entries={e['name']:e for e in prior['names']};runs=prior['runs']
  assert set(lookup)==set(entries)
  assert all(vecsha(v)==entries[n]['embedding_float32_le_sha256'] for n,v in lookup.items())
 need=[n for n in names if n not in lookup]
 print(json.dumps({'status':'starting','pid':os.getpid(),'requested':len(names),'existing_pinned':len(lookup),'need_encode':len(need)}),flush=True)
 if not need:
  print(json.dumps({'status':'all_requested_names_already_pinned','cache':str(CACHE),'entries':len(lookup)}),flush=True);return
 assert sha(SNAP/'model.safetensors')=='0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd'
 semantics_sha=sha(SOURCE)
 nodes=[n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ['joint_prompt','normalize_joint_name']]
 assert len(nodes)==2
 env={'re':re};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(SOURCE),'exec'),env)
 prompts=[env['joint_prompt'](n) for n in need]
 before=gpu()
 assert torch.cuda.is_available(),'GPU requested, CUDA unavailable'
 free,total=torch.cuda.mem_get_info(0);assert free>8*1024**3,('insufficient free GPU memory',free)
 torch.set_num_threads(4)
 tok=AutoTokenizer.from_pretrained(str(SNAP),local_files_only=True,trust_remote_code=False,padding_side='left')
 full_token_ids=tok(prompts,padding=False,truncation=False)['input_ids']
 model=AutoModel.from_pretrained(str(SNAP),local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,attn_implementation='sdpa').eval()
 model.requires_grad_(False);model.to('cuda:0')
 assert not model.training and not any(p.requires_grad for p in model.parameters())
 torch.cuda.reset_peak_memory_stats(0)
 loaded=gpu();model_class=type(model).__name__;batch_records=[]
 with torch.inference_mode():
  for start in range(0,len(need),args.batch_size):
   batch_names=need[start:start+args.batch_size];batch_prompts=prompts[start:start+args.batch_size]
   inputs=tok(batch_prompts,padding=True,truncation=True,max_length=64,return_tensors='pt')
   ids=inputs['input_ids'];am=inputs['attention_mask'];assert bool((am[:,-1]==1).all())
   device_inputs={k:v.to('cuda:0') for k,v in inputs.items()}
   hidden=model(**device_inputs,use_cache=False).last_hidden_state
   assert hidden.shape[-1]==1024
   # Position -1 is the final valid token because left padding was asserted above.
   pooled=hidden[:,-1,:].float()
   feats=torch.nn.functional.normalize(pooled,p=2,dim=-1).cpu().contiguous()
   assert feats.shape==(len(batch_names),1024) and bool(torch.isfinite(feats).all())
   for j,name in enumerate(batch_names):
    v=feats[j].clone();norm=float(v.norm())
    assert abs(norm-1)<1e-5
    lookup[name]=v
    prompt=batch_prompts[j];actual_ids=ids[j][am[j].bool()].tolist()
    entries[name]={'name':name,'name_utf8_sha256':hbytes(name.encode()),'normalized_name':env['normalize_joint_name'](name),'actual_input_text':prompt,'prompt_utf8_sha256':hbytes(prompt.encode()),'token_ids_no_padding':actual_ids,'token_ids_no_padding_json_sha256':hbytes(json.dumps(actual_ids,separators=(',',':')).encode()),'padded_input_ids':ids[j].tolist(),'attention_mask':am[j].tolist(),'valid_token_count':len(actual_ids),'full_token_count':len(full_token_ids[start+j]),'truncated':len(full_token_ids[start+j])>64,'pooling_index_in_padded_sequence':int(ids.shape[1]-1),'embedding_shape':[1024],'embedding_dtype':'float32','embedding_float32_le_sha256':vecsha(v),'finite':True,'l2_norm':norm,'run_index':len(runs)}
   batch_records.append({'first_input_index':start,'batch_size':len(batch_names),'padded_token_length':int(ids.shape[1])})
   print(json.dumps({'status':'encoded_batch','done':min(start+len(batch_names),len(need)),'total':len(need)}),flush=True)
   del hidden,pooled,feats,device_inputs
 torch.cuda.synchronize(0)
 peak={'allocated_bytes':torch.cuda.max_memory_allocated(0),'reserved_bytes':torch.cuda.max_memory_reserved(0)}
 del model,tok;gc.collect();torch.cuda.empty_cache();after_release=gpu()
 assert sha(LEGACY)==legacy_sha,'shared legacy cache changed during run; worker never writes it'
 matrix=torch.stack([lookup[n].float() for n in lookup]);assert bool(torch.isfinite(matrix).all())
 tmp=CACHE.with_name(CACHE.name+'.tmp-'+str(os.getpid()));torch.save(lookup,tmp);os.replace(tmp,CACHE)
 npz=HERE/'qwen3_names_pinned.npz';temp_npz=npz.with_name(npz.name+'.tmp-'+str(os.getpid()))
 with temp_npz.open('wb') as f:np.savez_compressed(f,names=np.asarray(list(lookup)),embeddings=matrix.numpy())
 os.replace(temp_npz,npz)
 run={'started_at_utc':started,'finished_at_utc':utc(),'wall_seconds':time.monotonic()-t0,'pid':os.getpid(),'proc_start_ticks':Path('/proc/self/stat').read_text().split()[21],'proc_cmdline':Path('/proc/self/cmdline').read_bytes().replace(b'\0',b' ').decode(),'input_source':source_info,'requested_unique_count':len(names),'newly_encoded_count':len(need),'model_class':model_class,'model_training':False,'any_parameter_requires_grad':False,'inference_mode':True,'gpu_before':before,'gpu_loaded':loaded,'gpu_after_model_release':after_release,'cuda_peak':peak,'batches':batch_records,'legacy_cache_sha256_before':legacy_sha,'legacy_cache_sha256_after':sha(LEGACY)}
 runs.append(run)
 weight=SNAP/'model.safetensors'
 tokenfiles=[]
 for name in ['config.json','tokenizer_config.json','tokenizer.json','merges.txt','vocab.json','1_Pooling/config.json','modules.json']:
  p=SNAP/name;tokenfiles.append({'file':name,'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)})
 manifest={'schema':'pinned-qwen-joint-name-cache-v1','updated_at_utc':utc(),'model':{'repo_id':'Qwen/Qwen3-Embedding-0.6B','revision':REV,'snapshot':str(SNAP),'weights_sha256':sha(weight),'weight_dtype':'bfloat16','forward_dtype':'bfloat16','output_dtype':'float32','output_dimensions':1024,'local_files_only':True,'trust_remote_code':False,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','tokenizer_and_config_files':tokenfiles},'encoding':{'prompt_source':str(SOURCE),'prompt_source_sha256':semantics_sha,'executed_functions':['normalize_joint_name','joint_prompt'],'prompt_template':'the {normalize_joint_name(name)} joint of a body','padding_side':'left','padding':True,'max_length':64,'truncation':True,'add_special_tokens':'tokenizer default True; actual effective IDs recorded per name','packing':'independent sequences, dynamic left padding per batch; no concatenation of names','pooling':'last valid token; left padding and attention_mask[:,-1]==1 verified','l2_normalization':'torch.nn.functional.normalize on float32 pooled state','attention_implementation':'sdpa','use_cache':False,'legacy_embeddings_reused':False},'entry_count':len(lookup),'ordered_names':list(lookup),'names':[entries[n] for n in lookup],'runs':runs,'quality':{'all_finite':bool(torch.isfinite(matrix).all()),'shape':list(matrix.shape),'l2_norm_min':float(matrix.norm(dim=-1).min()),'l2_norm_max':float(matrix.norm(dim=-1).max()),'truncated_name_count':sum(e['truncated'] for e in entries.values())},'outputs':{'torch_cache':{'path':str(CACHE),'bytes':CACHE.stat().st_size,'sha256':sha(CACHE)},'numpy_cache':{'path':str(npz),'bytes':npz.stat().st_size,'sha256':sha(npz)},'worker_script':{'path':str(Path(__file__).resolve()),'sha256':sha(Path(__file__))}},'runtime':{'python':sys.executable,'packages':{n:importlib.metadata.version(n) for n in ['torch','transformers','tokenizers','safetensors']}}}
 atomic_json(MANIFEST,manifest)
 print(json.dumps({'status':'complete','cache':str(CACHE),'manifest':str(MANIFEST),'entry_count':len(lookup),'quality':manifest['quality'],'cuda_peak':peak,'wall_seconds':run['wall_seconds']}),flush=True)
if __name__=='__main__':main()
