#!/usr/bin/env python3
"""Plot saved AlphaMotion1 forward tensors; no model execution or source edit."""
from pathlib import Path
import hashlib,json,subprocess,datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap,TwoSlopeNorm
from matplotlib.patches import Rectangle
OUT=Path(__file__).resolve().parent; TRACE=OUT.parent/"alpha1"
BASE=Path('/home/arenalabs/Desktop')/'"be water, robot"'; SRC=BASE/"pose_translator_design/exp_topo"
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
original={n:sha(TRACE/n) for n in ["trace.json","trace.npz"]}
meta=json.loads((TRACE/"trace.json").read_text()); z=np.load(TRACE/"trace.npz",allow_pickle=False)
B=0; frame=int(meta["source"]["frames"][B]); slot=0
sn=meta["source"]["joint_names"]; tn=meta["target"]["joint_names"]
sj=sn.index("left_elbow_link"); tj=tn.index("left_elbow_link")
commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=BASE,text=True).strip()
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"text.color":"#222831","axes.labelcolor":"#434b55","xtick.color":"#5e6873","ytick.color":"#5e6873","axes.edgecolor":"#ccd3da","axes.spines.top":False,"axes.spines.right":False,"figure.facecolor":"white","axes.facecolor":"white","savefig.facecolor":"white","svg.fonttype":"none","svg.hashsalt":"alphamotion-case-20261003","axes.titlesize":17,"axes.titleweight":"bold"})
blue="#246ea4"; orange="#c8773b"; gray="#6b7480"
seq=LinearSegmentedColormap.from_list("evidence_blue",["#f7fafc","#c7dceb","#6da3c9","#1e5f91"])
div=LinearSegmentedColormap.from_list("signed_embedding",["#246ea4","#fcfcfb","#d98a4b"])
entries=[]; values={}
caption=f"fallAndGetUp3_subject1  ·  frame {frame} / 30 fps  ·  actual checkpoint forward"
def header(fig,title,subtitle,subtitle_y=.917):
 fig.text(.065,.963,title,fontsize=19,fontweight="bold",va="top")
 fig.text(.065,subtitle_y,subtitle,fontsize=10.5,color=gray,va="top")
def footer(fig,text): fig.text(.065,.018,text,fontsize=9,color=gray,va="bottom")
def save(fig,stem,e):
 files=[]
 for ext in ["png","svg"]:
  p=OUT/f"{stem}.{ext}"
  fig.savefig(p,dpi=170,metadata={"Creator":"AlphaMotion source evidence audit"} if ext=="svg" else {"Software":"AlphaMotion source evidence audit"})
  files.append({"file":p.name,"sha256":sha(p),"bytes":p.stat().st_size})
 plt.close(fig)
 e.update({"stem":stem,"files":files,"source_frame":frame,"source_clip":meta["source"]["clip"],"classification":"actual_forward_tensor_visualization","synthetic_values":False})
 entries.append(e)
keys=["source_pose_embedding","source_descriptor_embedding256","source_combined_token256"]
arr=np.stack([z[k][B,sj] for k in keys]); lim=float(np.max(np.abs(arr)))
fig=plt.figure(figsize=(12,4.3))
header(fig,"One real joint, 256 feature channels",caption+"  ·  G1 left_elbow_link [23]",subtitle_y=.88)
gs=fig.add_gridspec(3,1,left=.23,right=.88,bottom=.20,top=.79,hspace=.65)
for i,(v,label) in enumerate(zip(arr,["Pose9 projection","Body descriptor","Encoder input"])):
 ax=fig.add_subplot(gs[i,0]); im=ax.imshow(v[None],aspect="auto",interpolation="nearest",cmap=div,norm=TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim))
 ax.set_yticks([]); ax.set_ylabel(label,rotation=0,ha="right",va="center",labelpad=12,fontsize=11); ax.set_xticks([0,63,127,191,255])
 if i<2: ax.set_xticklabels([]); ax.tick_params(axis="x",length=0)
 else: ax.set_xlabel("Feature channel (0-based)",labelpad=6)
 for spine in ax.spines.values(): spine.set_visible(False)
cax=fig.add_axes([.905,.27,.014,.43]); cb=fig.colorbar(im,cax=cax); cb.set_label("Activation",fontsize=10); cb.ax.tick_params(labelsize=9)
footer(fig,"Encoder input = pose projection + signed-contact embedding + body descriptor. Shared zero-centered scale; no channel averaging.")
values["source_token256"]={k:z[k][B,sj].tolist() for k in keys}
save(fig,"01-source-token256",{"tensor_keys":keys,"selection":{"batch":B,"source_joint":sj,"joint_name":sn[sj]},"display_shape":[3,256],"color_scale":{"type":"linear_diverging","min":-lim,"max":lim,"center":0},"qkv_origin":"Encoder input is layer-normalized before first Q/K/V projections."})
def heat(stem,title,key,qlabel,klabel,qorigin,kvorigin,selected,label,size):
 a=z[key][B,0]; rows,cols=a.shape; fig=plt.figure(figsize=size)
 header(fig,title,caption+"  ·  head 0 / 8")
 ax=fig.add_axes([.10,.34,.76,.45]); im=ax.imshow(a,aspect="auto",interpolation="nearest",cmap=seq,vmin=0,vmax=1)
 yt=sorted(set([0,rows//4,rows//2,3*rows//4,rows-1,selected])); xt=sorted(set([0,cols//4,cols//2,3*cols//4,cols-1]))
 ax.set_yticks(yt); ax.set_xticks(xt); ax.tick_params(labelsize=9,length=2)
 ax.set_ylabel(qlabel,fontsize=10); ax.set_xlabel(klabel,fontsize=10,labelpad=6)
 ax.add_patch(Rectangle((-.5,selected-.5),cols,1,fill=False,ec=orange,lw=1))
 cax=fig.add_axes([.885,.34,.013,.45]); cb=fig.colorbar(im,cax=cax,ticks=[0,.25,.5,.75,1]); cb.set_label("Attention probability",fontsize=10); cb.ax.tick_params(labelsize=9)
 ax2=fig.add_axes([.10,.13,.76,.095]); ax2.bar(np.arange(cols),a[selected],color=blue,width=.82)
 ax2.set_ylim(0,max(float(a[selected].max())*1.12,.05)); ax2.set_xlim(-.5,cols-.5)
 ax2.set_xticks(xt); ax2.tick_params(labelsize=8,length=2); ax2.set_ylabel("P",rotation=0,labelpad=12,fontsize=9)
 ax2.set_title(f"Outlined query: {label} [{selected}] · row sum = {a[selected].sum():.7f}",fontsize=10,fontweight="normal",loc="left",pad=8)
 ax2.set_xlabel(klabel,fontsize=9,labelpad=3); ax2.set_yticks([0,round(float(a[selected].max()),3)])
 ax2.grid(axis="y",color="#e9edf0",linewidth=.6); ax2.set_axisbelow(True)
 fig.text(.10,.847,f"Q: {qorigin}\nK,V: {kvorigin}",fontsize=10,color="#434b55",va="top",linespacing=1.45)
 footer(fig,"Softmax across each row's keys. Full matrix: linear 0–1 scale; lower strip: actual selected row. All indices are 0-based.")
 values[stem]={"tensor_key":key,"batch":B,"head":0,"query_index":selected,"query_label":label,"weights":a[selected].tolist()}
 save(fig,stem,{"tensor_keys":[key],"selection":{"batch":B,"head":0},"display_shape":list(a.shape),"color_scale":{"type":"linear_sequential","min":0,"max":1},"query_origin":qorigin,"key_value_origin":kvorigin,"softmax_axis":"last dimension / each query row across keys","row_sum_max_abs_error":float(np.max(np.abs(a.sum(-1)-1))),"selected_row":selected,"global_min":float(a.min()),"global_max":float(a.max())})
heat("02-encoder-attention","Source encoder · layer 0 · 36 × 36","encoder_layer0_attention","G1 joint query index","G1 joint key index","projected LN(source encoder input)","projected same LN(source encoder input); logits include learned hop bias",sj,"G1 left_elbow_link",(10.6,7.7))
heat("03-rotation-pool-attention","Rotation pooling · 128 × 36","rotation_pool_attention","Learned rotation query slot","G1 joint key index","projected LN(128 learned rotation queries)","projected LN(final source encoder output)",slot,"rotation latent slot",(10.6,8.4))
heat("04-target-decoder-attention","H2 rotation decoder · layer 0 · 36 × 128","target_rotation_decoder_layer0_attention","Official H2 joint query index","Rotation latent key slot","projected LN(official H2 descriptor query)","projected LN(128 post-FSQ reconstructed rotation tokens)",tj,"H2 left_elbow_link",(11.8,7.4))
f=z["rotation_fsq_pre"][B,slot]; q=z["rotation_fsq_1"][B,slot]; bounded=np.tanh(f); half=15.5
maxerr=float(np.max(np.abs(np.round(bounded*half)/half-q)))
fig=plt.figure(figsize=(11.8,5.5))
header(fig,"Rotation FSQ · one real slot, 13 scalar channels",caption+"  ·  rotation slot 0 · quantization stage 0",subtitle_y=.888)
ax0=fig.add_axes([.09,.66,.84,.115]); fm=max(abs(float(f.min())),abs(float(f.max())))
ax0.imshow(f[None],aspect="auto",interpolation="nearest",cmap=div,norm=TwoSlopeNorm(vmin=-fm,vcenter=0,vmax=fm))
ax0.set_xticks([]); ax0.set_yticks([]); ax0.set_ylabel("Raw f",rotation=0,ha="right",va="center",labelpad=12)
for j,v in enumerate(f): ax0.text(j,0,f"{v:.2f}",ha="center",va="center",fontsize=10,color="white" if abs(v)>.62*fm else "#25303b")
for spine in ax0.spines.values(): spine.set_visible(False)
ax=fig.add_axes([.09,.22,.84,.34]); xx=np.arange(len(f))
for j in xx: ax.plot([j,j],[bounded[j],q[j]],color="#a7b0b9",lw=1.3,zorder=1)
ax.scatter(xx-.06,bounded,s=48,facecolors="white",edgecolors=blue,linewidths=1.5,label="Before rounding: tanh(f)",zorder=3)
ax.scatter(xx+.06,q,s=31,c=orange,marker="s",label="After rounding: q",zorder=4)
ax.set_xlim(-.5,12.5); ax.set_ylim(-1.15,1.15); ax.set_xticks(xx); ax.set_xlabel("Scalar channel (0-based)")
ax.set_yticks([-1,-.5,0,.5,1]); ax.set_ylabel("Normalized value",fontsize=10)
ax.axhline(0,color="#d4dae0",lw=.9); ax.grid(axis="y",color="#ebeff2",lw=.6); ax.set_axisbelow(True)
ax.legend(loc="upper center",bbox_to_anchor=(.5,-.22),ncol=2,frameon=False,fontsize=10)
fig.text(.09,.824,"Raw preprojection values above; bounded and quantized values below share the same normalized axis.",fontsize=10,color=gray)
footer(fig,"q = round(15.5 × tanh(f)) / 15.5  ·  levels = [32] × 13  ·  eval: no code noise  ·  projected back to 256 channels for decoding.")
values["rotation_fsq_slot0"]={"tensor_keys":["rotation_fsq_pre","rotation_fsq_1","rotation_fsq_integer_indices"],"batch":B,"slot":slot,"pre_f":f.tolist(),"tanh_f":bounded.tolist(),"post_q":q.tolist(),"integer_indices":z["rotation_fsq_integer_indices"][0,B,slot].tolist(),"numpy_reconstruction_max_abs_error":maxerr}
save(fig,"05-fsq13",{"tensor_keys":["rotation_fsq_pre","rotation_fsq_1"],"selection":{"batch":B,"rotation_slot":slot,"stage":0},"display_shape":[13],"pre_units":"unbounded projection activation","post_units":"normalized rounded scalar grid","formula":"q=round(15.5*tanh(f))/15.5","raw_strip_scale":{"min":-fm,"max":fm},"numpy_reconstruction_max_abs_error":maxerr})
def evidence(name,lo,hi,fact):
 p=SRC/name; lines=p.read_text().splitlines()
 return {"file":name,"path":str(p),"commit":commit,"sha256":sha(p),"line_start":lo,"line_end":hi,"fact":fact,"snippet":"\n".join(f"{i}: {lines[i-1]}" for i in range(lo,hi+1))}
ev=[
 evidence("model_dual.py",226,245,"rot_only=false returns raw hpos(qp); last_pos carries this tensor."),
 evidence("train_dual.py",438,452,"Only target-root channel 0 is directly supervised against root height. Computed world displacement and yaw do not feed this loss."),
 evidence("train_dual.py",552,561,"Adjacent-frame root channel 0 differences are additionally supervised at 30 fps."),
 evidence("dual_common.py",389,402,"fk_loss=true routes XYZ loss through predicted rotation FK/reach, bypassing raw hpos XYZ."),
 evidence("train_dual.py",502,513,"All decoded hpos values enter active re-encoding consistency."),
 evidence("model.py",58,77,"Actual FSQ bound, round, normalize and reconstruct."),
 evidence("model.py",213,264,"Self attention, hop bias, pre-LN blocks, cross attention Q/K/V."),
 evidence("model_dual.py",121,141,"Position and rotation use distinct query banks and FSQ branches."),
 evidence("model_dual.py",156,174,"Pose projection, masking, signed-contact embedding, descriptor sum, encoder.")]
cp=Path("/var/tmp/alphamotion/topo_exp/runs/L_M20k_0903/config.json"); config=json.loads(cp.read_text())
keep=["fk_loss","rot_only","rot_head","root_head","root_pos","w_rootpos","w_hpair","reenc_consistency","reenc_prob","vis_residual","var_floor","fsq_stages","fsq_levels","contact_chan","sem_encoder","init"]
audit={"schema_version":1,"status":"proposed supplemental metadata; original trace.json unchanged","case_id":meta["case_id"],"evidence_class":"source-audited checkpoint-forward interpretation; not physics or control validation","source_commit":commit,"trace_inputs":[{"path":str(TRACE/n),"sha256":v} for n,v in original.items()],"checkpoint":meta["weights"],"config":{"path":str(cp),"sha256":sha(cp),"values":{k:config.get(k) for k in keep}},
"hpos":{"tensor_key":"predicted_position_head_raw","shape":list(z["predicted_position_head_raw"].shape),"root_index":0,
"root_channel0":{"semantic":"predicted target root height above contact ground","units":"meters","direct_loss":"w_rootpos * smooth_l1(hpos[:, root, 0], target_gap_root)","temporal_loss":"w_hpair * mean((30 * (delta hpos_root0 - delta target_gap_root))**2)","temporal_difference_units":"meters/second","citation_indices":[1,2]},
"root_channels1_2":{"semantic":"raw auxiliary outputs; no direct world-horizontal displacement or velocity target in active loss","units":"no metric interpretation justified","training":"indirectly constrained by active decode-reencode latent consistency","citation_indices":[1,4]},
"nonroot_channels":{"semantic":"raw auxiliary position-head outputs; no direct XYZ supervision under fk_loss=true","units":"no metric interpretation justified","training":"indirectly constrained by active decode-reencode latent consistency","citation_indices":[3,4]},
"noteworthy_unused_computation":"train_dual.py computes rw_target-rw_source and yaw trigonometry at lines 443-445, but neither enters root loss at lines 451-452.",
"gradient_scope_caveat":"Auxiliary channels are not untrained: active re-encoding consistency supplies an indirect gradient path.",
"example_frame1800":{"raw_root3":z["predicted_position_head_raw"][B,0].tolist(),"raw_h2_left_elbow3":z["predicted_position_head_raw"][B,tj].tolist()}},
"rotation_fk":{"tensor_key":"predicted_fk_positions_root_relative_cm_y","shape":list(z["predicted_fk_positions_root_relative_cm_y"].shape),"semantic":"target-root-relative FK from predicted global rotations and fresh official H2 offsets","units":"centimeters","world_translation_included":False,"raw_hpos_used_for_nonroot_fk":False,"interpretation":"Metric kinematic visualization output, distinct from raw hpos."},
"attention":{"interpretation":"Per-head softmax recomputed from actual forward-captured inputs and trained projection weights; module-output parity checked.","heads":8,"head_dim":32,"probability_axis":"last dimension; each query row sums to 1","softmax_output_max_abs_error":meta["numeric_validation"]["attention_max_output_error"],"source":"Q,K,V = projections of block-pre-normalized source tokens; learned hop bias enters logits.","rotation_pool":"Q=projection of LN(learned queries); K,V=projections of LN(final encoder output).","target_decoder_layer0":"Q=projection of LN(official H2 descriptor); K,V=projections of LN(post-FSQ reconstructed rotation tokens).","normalization_note":"Pre-LN means normalization BEFORE Q/K/V projection; captured projection inputs are normalized."},
"scope_limits":meta["scope_limits"],"source_evidence":ev}
(OUT/"hpos-semantics.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2)+"\n")
(OUT/"selected-tensor-values.json").write_text(json.dumps(values,indent=2)+"\n")
manifest={"schema_version":1,"case_id":meta["case_id"],"created_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"classification":"actual checkpoint forward tensors and audited code semantics","source_commit":commit,"input_trace_hashes":original,"frame_selection":{"batch":B,"clip_frame":frame,"fps":30,"selection_rule":"First saved frame and rotation latent slot 0; no search for preferred results."},"matrices":{"encoder_layer0_attention":["G1 joint query","G1 joint key"],"rotation_pool_attention":["learned rotation query slot","G1 joint key"],"target_rotation_decoder_layer0_attention":["official H2 joint query","post-FSQ rotation latent key"]},"source_joint_names":sn,"target_joint_names":tn,"latent_slot_indices":list(range(128)),"plots":entries,"evidence_limits":["Single-frame kinematic forward, not temporal generation or control.","Attention probability is not causal contribution or semantic correspondence.","Latent slot indices have no assigned human semantic labels.","No claim of held-out generalization, hardware feasibility or physics validation.","Full heatmaps share linear [0,1]; row bars have explicitly independent y scales."],"supporting_files":[{"file":n,"sha256":sha(OUT/n),"bytes":(OUT/n).stat().st_size} for n in ["hpos-semantics.json","selected-tensor-values.json","render_evidence.py"]]}
(OUT/"plot-manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
readme="""# AlphaMotion1 actual-forward evidence assets

These five figures use saved tensors from strict-loaded L_M20k_0903: G1 fallAndGetUp3_subject1, frame 1800 (30 fps), to a fresh descriptor built from the official Unitree H2 MJCF. No model was run or modified by this plotting script.

- 01-source-token256: all 256 channels of the actual left-elbow pose projection, body descriptor and combined encoder input; common signed scale.
- 02-encoder-attention: layer 0 / head 0, 36 G1 queries by 36 G1 keys.
- 03-rotation-pool-attention: head 0, 128 learned rotation queries by 36 G1 keys.
- 04-target-decoder-attention: layer 0 / head 0, 36 official H2 queries by 128 rotation latents.
- 05-fsq13: actual rotation slot 0 before and after one-stage scalar quantization. Raw and normalized values use explicitly different axes.

Attention heatmaps retain every cell and share a linear probability scale from 0 to 1. Orange outlines select one query; lower bars show that row on a separately labeled probability axis. Sparse axes use zero-based indices; the manifest preserves all joint names. Stored attention was reconstructed from forward-captured inputs and checkpoint projections, with module-output parity below 7.2e-7. These weights do not establish causal importance or joint correspondence.

hpos-semantics.json supplements the original trace with source hashes, exact line snippets and checkpoint/config evidence. Target-root channel 0 has a direct ground-relative-height loss in meters and a temporal height-difference loss. Root channels 1-2 and nonroot raw hpos have no direct metric XYZ supervision under this configuration, but enter re-encoding consistency. Separate FK is target-root-relative centimeters from predicted rotations and actual target offsets. Raw hpos is not substituted for FK.

plot-manifest.json hashes every PNG/SVG and links source tensor keys. selected-tensor-values.json preserves plotted strips and attention rows. SVG text stays editable. render_evidence.py reproduces all outputs from the adjacent alpha1 trace.json and trace.npz with NumPy and Matplotlib.

Scope: three independent frames exist in the parent trace; these assets use its first one. This is a training-domain single-frame kinematic demonstration, not temporal generation, physical-robot control, joint-limit or physics validation, or held-out generalization. Official H2 descriptors do not bind the legacy anonymous cache tree to official meshes. No Google Doc or source/model/service state was changed.
"""
(OUT/"README.md").write_text(readme)
for n,h in original.items(): assert sha(TRACE/n)==h,f"Input changed: {n}"
print(json.dumps({"output_dir":str(OUT),"plots":len(entries),"files":sorted(p.name for p in OUT.iterdir() if p.is_file()),"fsq_reconstruction_max_abs_error":maxerr,"trace_unchanged":True},indent=2))
