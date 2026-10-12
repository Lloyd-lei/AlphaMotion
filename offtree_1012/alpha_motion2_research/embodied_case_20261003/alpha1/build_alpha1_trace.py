"""Isolated, CPU-only actual checkpoint trace; no training or source modifications."""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from pathlib import Path
import sys, json, hashlib, time, datetime
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path('/home/arenalabs/Desktop/"be water, robot"')
SRC = ROOT / 'pose_translator_design/exp_topo'
OUT = ROOT / 'alpha_motion2/research/embodied_case_20261003/alpha1'
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(SRC))
from common import SkeletonSpec, fk_global, matrix_to_rot6d, rot6d_to_matrix
from dual_common import build_model, fk_pos_torch
from train_ae import static_cond
from semantics import joint_prompt, _encode_siglip
from eval_newbody import build_from_mjcf

torch.set_num_threads(2)
torch.manual_seed(0)
RUN = Path('/var/tmp/alphamotion/topo_exp/runs/L_M20k_0903')
CFG = json.loads((RUN/'config.json').read_text())
CACHES = [Path(p) for p in CFG['cache'].split(',')]
BODY = 'unitree_g1_29dof'
CLIP = 'fallAndGetUp3_subject1'
FRAMES = [1800, 2000, 2600]
XML = Path('/var/tmp/alphamotion/robot_models/unitree_ros/robots/h2_description/unitree_h2_mjcf.xml')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')

spec_file=CACHES[0]/f'{BODY}_spec.npz'
# Existing local spec is explicitly checksum-identified before object-name loading.
inventory=json.loads((ROOT/'alpha_motion2/pretraining_foundation/manifests/existing_assets.json').read_text())
allowed=next(b['spec']['sha256'] for c in inventory['recommended_caches'] if c['id']=='cache_lafan_v3' for b in c['bodies'] if b['body']==BODY)
assert sha(spec_file)==allowed
with np.load(spec_file, allow_pickle=True) as z:
    source_spec=SkeletonSpec(str(z['name']),z['parents'],z['rest_offsets'],list(z['joint_names']))
with np.load(CACHES[0]/f'{BODY}_dof.npz',allow_pickle=False) as z:
    source_dof=z['dof'].copy()
seq=next(s for s in json.loads((CACHES[0]/'sequences.json').read_text()) if s['name']==CLIP)
rows=np.array(FRAMES)+seq['start']
local=torch.load(CACHES[0]/f'{BODY}_paired_local6d.pt',mmap=True,weights_only=True)[rows].float()
target_spec,target_dof,target_rest,target_slots,_=build_from_mjcf(str(XML),'unitree_h2_official_query')

reach_file=OUT/'reach_scan.json'
reach_results=[]
if reach_file.exists():
    reach_results=json.loads(reach_file.read_text())
else:
    with torch.inference_mode():
        for cache in CACHES:
            tensor_path=cache/f'{BODY}_paired_local6d.pt'
            all_local=torch.load(tensor_path,mmap=True,weights_only=True)
            mx=0.;begin=time.monotonic()
            print('reach_scan_start',str(cache),all_local.shape,flush=True)
            for i in range(0,len(all_local),8192):
                _,positions=fk_global(all_local[i:i+8192].float(),source_spec)
                mx=max(mx,float(positions.norm(dim=-1).max()))
            reach_results.append({'cache':str(cache),'tensor_path':str(tensor_path),'rows':len(all_local),'tensor_size_bytes':tensor_path.stat().st_size,'reach_cm':mx,'method':'Full configured-cache row scan, FK chunk8192, CPU float32, two threads','elapsed_seconds':time.monotonic()-begin})
            print('reach_scan_complete',reach_results[-1],flush=True)
    dump(reach_file,reach_results)
reach=max(r['reach_cm'] for r in reach_results)

lut_path=Path('/var/tmp/alphamotion/topo_exp/siglip_names.pt')
lut=torch.load(lut_path,map_location='cpu',weights_only=True)
names=list(dict.fromkeys(source_spec.joint_names+target_spec.joint_names))
missing=[n for n in names if n not in lut]
new_sem={}
if missing:
    print('offline_siglip_new_names',missing,flush=True)
    embeddings=F.normalize(_encode_siglip([joint_prompt(n) for n in missing], 'cpu'), dim=-1)
    new_sem={n:embeddings[i] for i,n in enumerate(missing)}
    # Preserve original shared names cache; generated names remain isolated here.
    torch.save(new_sem,OUT/'additional_siglip_names.pt')
lut={**lut,**new_sem}
def make_geom(spec,dof):
    static=static_cond(spec,'cpu')
    g={k:static[k][None].expand(len(FRAMES),*static[k].shape) for k in ['rest_off','bonelen','depth','hop']}
    g['sem']=torch.stack([lut[n] for n in spec.joint_names])[None].expand(len(FRAMES),-1,-1)
    g['dof']=torch.as_tensor(dof).float()[None].expand(len(FRAMES),-1,-1)
    return g
gs,gt=make_geom(source_spec,source_dof),make_geom(target_spec,target_dof)
with torch.inference_mode():
    global_R,global_p=fk_global(local,source_spec)
    pose=torch.cat([matrix_to_rot6d(global_R),global_p/reach],-1)
contact_path=CACHES[0]/f'{BODY}_contact.npz'
with np.load(contact_path,allow_pickle=False) as z:
    header=json.loads(str(z['header']))
    gap=z['gap'][rows].astype(np.float32)
thr,D=float(header['thr_m']),float(header['D_used_m'])
signed=np.sign(thr-gap)*np.log1p(np.abs(thr-gap)/thr)/np.log1p(D/thr)
con=torch.from_numpy(signed)
mask=torch.zeros(pose.shape[:2],dtype=torch.bool)
model=build_model(CFG,'cpu')
state=torch.load(RUN/'model.pt',map_location='cpu',weights_only=True)
model.load_state_dict(state,strict=True)
model.eval()
arrays={}
def save_tensor(name,t): arrays[name]=t.detach().cpu().numpy()
parity={}
def basic_hook(name):
    def hook(mod,args,out):
        if isinstance(out,tuple):
            for i,t in enumerate(out):
                if torch.is_tensor(t):save_tensor(name+f'_{i}',t)
        elif torch.is_tensor(out):save_tensor(name,out)
    return hook
def graph_hook(name):
    def hook(m,args,out):
        x,hop,kpm=args;B,J,D=x.shape
        q=m.q(x).view(B,J,m.h,m.dk).transpose(1,2)
        k=m.k(x).view(B,J,m.h,m.dk).transpose(1,2)
        v=m.v(x).view(B,J,m.h,m.dk).transpose(1,2)
        bias=m.hop_emb(hop.clamp(max=m.max_hop+1)).permute(0,3,1,2)
        logits=q@k.transpose(-1,-2)/(m.dk**.5)+bias
        if kpm is not None:logits=logits.masked_fill(kpm[:,None,None,:],float('-inf'))
        att=logits.softmax(-1)
        recompute=m.o((att@v).transpose(1,2).reshape(B,J,D))
        parity[name]={'max_abs_output_error':float((out-recompute).abs().max()),'max_row_sum_error':float((att.sum(-1)-1).abs().max())}
        for n,t in [('normalized_input',x),('q',q),('k',k),('v',v),('hop_bias',bias),('attention',att)]:save_tensor(name+'_'+n,t)
    return hook
def cross_hook(name):
    def hook(m,args,out):
        qin,kin,vin=args;wq,wk,wv=m.in_proj_weight.chunk(3);bq,bk,bv=m.in_proj_bias.chunk(3)
        B,L,D=qin.shape;S=kin.shape[1];H=m.num_heads;hd=D//H
        q=F.linear(qin,wq,bq).view(B,L,H,hd).transpose(1,2)
        k=F.linear(kin,wk,bk).view(B,S,H,hd).transpose(1,2)
        v=F.linear(vin,wv,bv).view(B,S,H,hd).transpose(1,2)
        att=(q@k.transpose(-1,-2)/(hd**.5)).softmax(-1)
        computed=m.out_proj((att@v).transpose(1,2).reshape(B,L,D))
        parity[name]={'max_abs_output_error':float((computed-out[0]).abs().max()),'max_row_sum_error':float((att.sum(-1)-1).abs().max())}
        for n,t in [('normalized_q_input',qin),('normalized_kv_input',kin),('q',q),('k',k),('v',v),('attention',att)]:save_tensor(name+'_'+n,t)
    return hook
handles=[]
for label,module in [('source_pose_embedding',model.inp),('encoder_final',model.enc[-1]),('position_pool_tokens',model.poolp),('rotation_pool_tokens',model.poolr),('position_fsq_pre',model.fsqp.proj_in),('rotation_fsq_pre',model.fsqr.proj_in),('position_fsq',model.fsqp),('rotation_fsq',model.fsqr)]:
    handles.append(module.register_forward_hook(basic_hook(label)))
handles.append(model.enc[0].a.register_forward_hook(graph_hook('encoder_layer0')))
handles.append(model.poolr.mha.register_forward_hook(cross_hook('rotation_pool')))
handles.append(model.decr[0].mha.register_forward_hook(cross_hook('target_rotation_decoder_layer0')))
t0=time.monotonic()
with torch.inference_mode():
    source_descriptor=model.jf(gs['rest_off'],gs['bonelen'],gs['depth'],gs['sem'],gs['dof'])
    target_query=model.jf(gt['rest_off'],gt['bonelen'],gt['depth'],gt['sem'],gt['dof'])
    source_combined=model.inp(pose)+source_descriptor+model.con_emb(con[...,None])
    zp,zr,codes=model.encode(pose,gs,joint_mask=mask,return_codes=True,con=con)
    pred_rot,pred_head,pred_con_logits=model.decode(zp,zr,gt,ret_con=True)
    pred_R=rot6d_to_matrix(pred_rot)
    pred_fk_cm=fk_pos_torch(pred_rot,target_spec)
for h in handles:h.remove()
elapsed=time.monotonic()-t0
for name,t in {'source_local_rot6d':local,'source_global_rotation_matrices_y':global_R,'source_fk_positions_root_relative_cm_y':global_p,'source_pose9':pose,'source_contact_signed_distance':con,'source_joint_mask_hidden':mask,'source_semantics768':gs['sem'],'target_semantics768':gt['sem'],'source_descriptor_embedding256':source_descriptor,'source_combined_token256':source_combined,'target_descriptor_query256':target_query,'source_root_world_cm_y':torch.load(CACHES[0]/f'{BODY}_root_world.pt',mmap=True,weights_only=True)[rows],'predicted_global_rot6d_y':pred_rot,'predicted_global_rotation_matrices_y':pred_R,'predicted_fk_positions_root_relative_cm_y':pred_fk_cm,'predicted_position_head_raw':pred_head,'predicted_contact_logits':pred_con_logits}.items():save_tensor(name,t)
for prefix,g in [('source',gs),('target',gt)]:
    for key,value in g.items():save_tensor(prefix+'_geom_'+key,value)
for prefix,spec in [('source',source_spec),('target',target_spec)]:
    arrays[prefix+'_parents']=spec.parents
    arrays[prefix+'_rest_offsets_cm_y']=spec.rest_offsets
for prefix,mod in [('position',model.fsqp),('rotation',model.fsqr)]:
    f=torch.from_numpy(arrays[prefix+'_fsq_pre'])
    save_tensor(prefix+'_fsq_bounded_grid_units',torch.tanh(f)*15.5)
    save_tensor(prefix+'_fsq_integer_indices',mod.ints(f))
arrays['target_rest_rotations_y']=target_rest
arrays['source_gap_m']=gap
np.savez_compressed(OUT/'trace.npz',**arrays)
source_joint='left_elbow_link';si=source_spec.joint_names.index(source_joint)
target_joint='left_elbow_link';ti=target_spec.joint_names.index(target_joint)
joint_records=[]
for b,frame in enumerate(FRAMES):
    joint_records.append({'clip_frame':frame,'time_seconds':frame/30,'cache_row':int(rows[b]),'source_joint':source_joint,'source_joint_index':si,'source_pose9':pose[b,si].tolist(),'source_global_rot6d':pose[b,si,:6].tolist(),'source_root_relative_pos_cm':global_p[b,si].tolist(),'source_position_divided_by_reach':pose[b,si,6:].tolist(),'source_contact_signed_distance':float(con[b,si]),'source_mask_hidden':False,'target_joint':target_joint,'target_joint_index':ti,'predicted_target_rot6d':pred_rot[b,ti].tolist(),'predicted_target_fk_pos_cm':pred_fk_cm[b,ti].tolist(),'predicted_position_head_raw':pred_head[b,ti].tolist(),'root_head_output_raw':pred_head[b,0].tolist()})
trace={'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'case_id':'g1_fallAndGetUp3_subject1_to_official_h2','device':'cpu','torch_threads':2,'forward_seconds':elapsed,'training':False,'weights':{'path':str(RUN/'model.pt'),'sha256':sha(RUN/'model.pt'),'strict_load':True,'state_tensor_count':len(state),'config_path':str(RUN/'config.json'),'config_sha256':sha(RUN/'config.json')},'source':{'body':BODY,'clip':CLIP,'frames':FRAMES,'fps':30,'cache_rows':rows.tolist(),'joint_names':list(source_spec.joint_names),'joint_count':source_spec.J,'spec_path':str(spec_file),'spec_sha256':allowed,'reach_cm':reach,'reach_scope':'maximum FK radius across BOTH full caches in actual training config','pose9':'global rotation6 concatenated matrix columns 0 and1, then FK root-relative cm positions divided by full-training-cache reach_cm','contact':'signed geometric distance, not probability or tactile','contact_threshold_m':thr,'contact_D_m':D,'mask':'all source joints visible; bool True would hide pose embedding only'},'target':{'body':'Unitree H2','query':'new descriptor built from real H2 MJCF; not legacy anonymous body00..body31 cache','xml_path':str(XML),'xml_sha256':sha(XML),'joint_count':target_spec.J,'joint_names':list(target_spec.joint_names),'slot_joint_names':target_slots,'query_input':'34D geometry/mechanics + 768D cached or offline SigLIP joint-name text, projected/summed to256D','legacy_h2_cache_used_for_query':False,'zero_shot_claim':False},'semantics':{'model':'google/siglip-base-patch16-224','pooled_dim':768,'normalization':'L2 normalize pooled text tower output before cached lookup','cache_path':str(lut_path),'cache_sha256':sha(lut_path),'additional_names_computed_offline':missing,'source_selected_prompt':joint_prompt(source_joint),'target_selected_prompt':joint_prompt(target_joint),'mesh_text_encoder':False},'tensors':{n:{'shape':list(v.shape),'dtype':str(v.dtype)} for n,v in arrays.items()},'joint_records':joint_records,'attention':{'all':'Actual forward inputs captured using read-only hooks; per-head weights recomputed algebraically from actual trained projections; output parity checked','encoder_layer0':{'Q_K_V':'same pre-LayerNorm source tokens','heads':8,'head_dim':32,'normalization':'softmax over source key joints after QK/sqrt32+learned per-head hop bias','head_to_display':0},'rotation_pool':{'Q':'LayerNorm learned128 rotation queries','K_V':'LayerNorm encoder output,36 source tokens','normalization':'softmax over source keys; no hop bias','head_to_display':0},'target_rotation_decoder_layer0':{'Q':'LayerNorm official H2 descriptor query','K_V':'LayerNorm128 post-FSQ reconstructed rotation tokens','normalization':'softmax over latent keys; no hop bias','head_to_display':0},'parity':parity},'scope_limits':['Three independent single-frame forwards, not temporal generation.','Source clip is a training-domain demonstration, not held-out or zero-shot evidence.','Official H2 query differs from the legacy H2 anonymous training tree.','Predicted free global rotations and their FK positions are not hardware qpos and have not undergone control, IK, or dynamics validation.','Non-root hpos vectors and root-specific hpos vector have different supervision semantics; use FK for skeleton visualization.','Other robot cards are alternative embodiments, not concatenated simultaneous inputs.']}
dump(OUT/'trace.json',trace)
print('forward_complete',json.dumps({'frames':FRAMES,'source_J':source_spec.J,'target_J':target_spec.J,'reach_cm':reach,'seconds':elapsed,'attention_parity':parity,'trace_npz':str(OUT/'trace.npz')}),flush=True)
