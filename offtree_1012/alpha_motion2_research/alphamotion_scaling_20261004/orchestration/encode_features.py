"""One held-lock, offline frozen Qwen pass. Existing584 vectors remain byte-exact."""
from pathlib import Path
import os,sys,json,ast,re,fcntl,time,datetime,subprocess
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'base'))
from pilot_data import sha256,read_json,write_json
from train_pilot import process_identity
from remote_state import collect
import torch
from transformers import AutoModel,AutoTokenizer

def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--pilot-root',type=Path,required=True)
    ap.add_argument('--extend',type=Path);ap.add_argument('--names-file',type=Path);ap.add_argument('--surface-file',type=Path)
    ap.add_argument('--output-stem',default='frozen_features');args=ap.parse_args()
    folder=ROOT/'qwen';folder.mkdir(exist_ok=True)
    lock=(ROOT/'gpu_queue.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (folder/(args.output_stem+'.pt')).exists():raise RuntimeError('cache exists; verify/reuse rather than overwrite')
    cap=collect();write_json(folder/'capacity.json',cap)
    if cap['free_home_bytes']<cap['remaining_combined_bytes']+31*1024**3:raise RuntimeError('combined download +16GiB safety +15GiB experiment guard')
    write_json(folder/'process_identity.json',process_identity())
    start=time.monotonic();torch.set_num_threads(4)
    original=args.pilot_root/'qwen_audit/qwen3_names_pinned.pt'
    old_sha=sha256(original)
    if old_sha!='462b84e41da470acf811d024bdead2f0b41fb8d054ef713e2f5401ebf48bc3ec':raise ValueError('original584 cache hash')
    original_names=torch.load(original,map_location='cpu',weights_only=True)
    previous=torch.load(args.extend,map_location='cpu',weights_only=True) if args.extend else {'joint_names':original_names,'surface_texts':{}}
    names=previous['joint_names']
    if any(n not in names or not torch.equal(names[n],v) for n,v in original_names.items()):raise ValueError('original vectors changed in extension')
    name_file=args.names_file or ROOT/'data/all_required_qwen_names.json';surface_file=args.surface_file or ROOT/'surface/required_surface_texts.json'
    wanted=read_json(name_file)['deduplicated_joint_names'];patches=read_json(surface_file)
    # Source prompt functions are parsed; no repository module code executes.
    source=Path('/home/arenalabs/Desktop/"be water, robot"/pose_translator_design/exp_topo/semantics.py')
    nodes=[n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ['joint_prompt','normalize_joint_name']]
    if len(nodes)!=2:raise ValueError('joint prompt definitions')
    env={'re':re};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),env)
    entries=patches['texts']
    tasks=[('joint_names',n,env['joint_prompt'](n)) for n in wanted if n not in names]
    tasks.extend(('surface_texts',p['key'],p['prompt']) for p in entries if p['key'] not in previous['surface_texts'])
    snap=Path('/var/tmp/alphamotion/hf_cache/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3')
    if sha256(snap/'model.safetensors')!='0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd':raise ValueError('Qwen source weights hash')
    gpu_before=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True)
    tok=AutoTokenizer.from_pretrained(str(snap),local_files_only=True,trust_remote_code=False,padding_side='left')
    model=AutoModel.from_pretrained(str(snap),local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,attn_implementation='sdpa').eval().requires_grad_(False).to('cuda')
    result=previous;audit=[]
    with torch.inference_mode():
        for i in range(0,len(tasks),16):
            batch=tasks[i:i+16];inputs=tok([x[2] for x in batch],padding=True,truncation=False,return_tensors='pt')
            if inputs['input_ids'].shape[1]>128 or not inputs['attention_mask'][:,-1].all():raise ValueError('unexpected prompt length or padding')
            hidden=model(**{k:v.to('cuda') for k,v in inputs.items()},use_cache=False).last_hidden_state
            vectors=torch.nn.functional.normalize(hidden[:,-1].float(),dim=-1).cpu()
            if vectors.shape!=(len(batch),1024) or not torch.isfinite(vectors).all():raise ValueError('frozen feature shape/finite')
            for j,(kind,key,prompt) in enumerate(batch):
                result[kind][key]=vectors[j].clone();ids=inputs['input_ids'][j][inputs['attention_mask'][j].bool()].tolist()
                audit.append({'kind':kind,'key':key,'actual_prompt':prompt,'token_ids':ids,'valid_tokens':len(ids),'norm':float(vectors[j].norm())})
            print(json.dumps({'encoded':min(i+16,len(tasks)),'total':len(tasks)}),flush=True)
    if sha256(original)!=old_sha:raise ValueError('original cache changed')
    params=sum(p.numel() for p in model.parameters());del model;torch.cuda.empty_cache()
    path=folder/(args.output_stem+'.pt');tmp=folder/(args.output_stem+'.tmp.pt');torch.save(result,tmp);tmp.replace(path)
    manifest={'status':'completed','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'wall_seconds':time.monotonic()-start,
        'cache_sha256':sha256(path),'original584_sha256':old_sha,'original584_byte_exact':True,
        'joint_name_count':len(result['joint_names']),'surface_text_count':len(result['surface_texts']),
        'source_names_sha256':sha256(name_file),'source_surface_texts_sha256':sha256(surface_file),'source_prompt_sha256':sha256(source),
        'frozen_model_unique_parameters':params,'model_revision':snap.name,'model_weight_sha256':sha256(snap/'model.safetensors'),
        'encoding':'offline frozen bfloat16 Qwen, last valid token, float32 L2 normalization, no prompt truncation',
        'entries':audit,'gpu_before':gpu_before,'excluded_from_training_loop_timing':True,'optimizer_updates':0,
        'extension_source':str(args.extend) if args.extend else None,'extension_source_sha256':sha256(args.extend) if args.extend else None}
    write_json(folder/(args.output_stem+'.manifest.json'),manifest);print(json.dumps({k:v for k,v in manifest.items() if k not in ['entries','gpu_before']}),flush=True)

if __name__=='__main__':main()
