from pathlib import Path
import json,hashlib
import pdfplumber
from PIL import Image
O=Path(__file__).resolve().parent;A=O.parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
refs={}
def source(k,rel,a,b,purpose):
 p=A/rel;assert 1<=a<=b<=len(p.read_text().splitlines());refs[k]={'path':str(p),'start':a,'end':b,'sha256':sha(p),'purpose':purpose}
source('config','verified-megatron-config.json',1,76,'Actual current megatron run settings, not constructor defaults.')
source('source_inputs','source-snapshot/AlphaMotion1/model_dual.py',65,88,'JointFeature, pose projection, mask token and source encoder construction.')
source('dual_latent','source-snapshot/AlphaMotion1/model_dual.py',88,102,'Independent query banks and FSQ modules for two streams.')
source('heads','source-snapshot/AlphaMotion1/model_dual.py',101,125,'Decoder stacks and separate position, rotation and contact heads.')
source('pool','source-snapshot/AlphaMotion1/model_dual.py',127,146,'CrossBlock learned-query pooling followed by stream-specific FSQ.')
source('trunk_order','source-snapshot/AlphaMotion1/model_dual.py',155,174,'Pose embedding -> pose mask replacement -> finite contact embedding -> morphology sum -> encoder.')
source('target_queries','source-snapshot/AlphaMotion1/model_dual.py',209,216,'Same JointFeature builds target queries; CrossBlock then EncBlock at every decoder depth.')
source('optional_contact','source-snapshot/AlphaMotion1/model_dual.py',242,251,'Contact logit returned only when ret_con=True; root head conditional.')
source('fsq','source-snapshot/AlphaMotion1/model.py',43,78,'Linear -> tanh-bound -> round with STE -> normalize -> Linear.')
source('joint_feature','source-snapshot/AlphaMotion1/model.py',175,208,'Geometry15 + DOF19 projection; SigLIP768 ->256 ->GELU ->256, added.')
source('hop_attention','source-snapshot/AlphaMotion1/model.py',213,236,'Independent Wq/Wk/Wv, per-head learned hop bias, softmax times V, concat and output projection.')
source('encblock','source-snapshot/AlphaMotion1/model.py',239,249,'Pre-norm attention residual and pre-norm 4D GELU FFN residual.')
source('crossblock','source-snapshot/AlphaMotion1/model.py',252,264,'Separate q/kv LayerNorms, standard MHA projections, attention residual and FFN residual.')
source('root_height','source-snapshot/AlphaMotion1/train_dual.py',438,452,'Only first position-head channel at root joint is supervised for root height.')
source('text','source-snapshot/AlphaMotion1/semantics.py',36,49,'Frozen cached SigLIP text inference; no_grad/eval.')
source('judge','source-snapshot/AlphaMotion1/model_dual.py',13,24,'Training auxiliary mean/max pooled two-stream JudgeHead, omitted from main inference path.')
tensors=json.loads((A/'verified-megatron-tensors.json').read_text())['tensors']
expected={'inp.weight':[256,9],'jf.proj.weight':[256,34],'jf.sem_proj.0.weight':[256,768],'jf.sem_proj.2.weight':[256,256],'qp':[4,256],'qr':[128,256],'fsqp.proj_in.weight':[13,256],'fsqr.proj_out.weight':[256,13],'hpos.1.weight':[3,256],'hrot.1.weight':[6,256],'hcon.1.weight':[1,256]}
for k,v in expected.items():assert tensors[k]['shape']==v,(k,tensors[k]['shape'],v)
assert not any(k.startswith('hroot.') for k in tensors)
operators={
 'overview':{'source_encoder':'4 distinct EncBlocks; each body uses the same stack','pool_position':'1 CrossBlock, Q learned[4,256], K/V H[B,J_s,256]','pool_rotation':'1 CrossBlock, Q learned[128,256], K/V same H','fsq':'separate modules per stream, one stage each; 256->13->256; configured levels[32]*13','decoder':'4 layers per stream; each layer = CrossBlock then hop EncBlock; no cross-stream weight tying','queries':'J_t target queries from the same JointFeature object used on source morphology','heads':'separate LayerNorm+Linear modules for pos3, global rot6, optional contact1'},
 'block_detail':{'encoder_attention':'Q=LN(x)Wq+bq, K=LN(x)Wk+bk, V=LN(x)Wv+bv; softmax(QK^T/sqrt(32)+hop_bias)V','cross_attention':'Q from LN_q(q); K,V from LN_kv(memory); independent MHA projections, no hop bias in cross-attention','residual':'x + attention(LN(x)), then x + FFN(LN(x)); CrossBlock residual carrier is q, never memory kv','ffn':'Linear256->1024, GELU, Linear1024->256; own FFN for each EncBlock and CrossBlock'} }
qa=[]
for name in ['alphamotion1_path','alphamotion1_blocks']:
 with pdfplumber.open(O/(name+'.pdf')) as d:
  assert len(d.pages)==1; p=d.pages[0];assert(p.width,p.height)==(510,640)
  chars=p.chars;assert min(c['size'] for c in chars)>=8
  assert not[c for c in chars if c['x0']<0 or c['x1']>510 or c['top']<0 or c['bottom']>640]
  image=Image.open(O/(name+'.png'));assert image.size==(1700,2134),image.size
  qa.append({'name':name,'pdf_pages':1,'size_pt':[510,640],'body_font_pt':9.1,'minimum_annotation_font_pt':min(c['size'] for c in chars),'png_dpi':240,'png_pixels':image.size,'bounds':'pass','rendered_visual_review':'pass','changes_verified':['correct pre-norm','independent Q/K/V projections','softmax output multiplied by V','contact after mask','body-shared decoder','separate memory per stream','no causal mask','FSQ purpose label fits']})
payload={'version':'alpha1-transformer-redraw-v1','status':'frozen_after_QA','sources':refs,'verified_tensor_shapes':expected,'operators':operators,'original_source_root':'/home/arenalabs/Desktop/be water, robot/pose_translator_design/exp_topo','source_run':'L_M20k_0903','deployed_model':'megatron','figures':qa,'body_weights_shared':True,'runtime':'single-frame','omissions':['training-only JudgeHead','disabled vis_residual and root_head','separate temporal/latent diffusion experiments'],'not_claimed':['centimeter-uniform FSQ quantization','causal Transformer','contact head BCE trained','per-body decoder parameters']}
(O/'source-map.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
lines=['# AlphaMotion-1 经典 Transformer 图：读图说明','', '冻结版本：alpha1-transformer-redraw-v1。按真实 megatron / L_M20k_0903 配置绘制。两页原生均为510×640 pt；算子正文至少9.1 pt，少量维度及说明注释最低8.5 pt。PDF为矢量，SVG可编辑；同名PNG为240 dpi，_preview为105 dpi。','',
 '## 第一页：从下向上读主路径','',
 '左下输入是一帧源身体姿态：每关节9维，包含全局6D旋转和归一化的root-relative位置。Linear映射到256维；需要隐藏的姿态随后替换成mask token。可用接触条件在mask之后加入，NaN接触贡献为零；最后加入源身体JointFeature，送入4层Hop-biased EncBlock。图中的两次圆形加号严格遵循这一顺序。','',
 '同一份源关节特征分别作为两个CrossBlock的K/V；两组学习到的Q分别有4和128个token，对应位置流和旋转流。它们各过独立的单阶段FSQ：256→13，tanh限制范围、round+STE、归一化，再映回256。levels配置为[32]×13；不是物理空间厘米均匀量化，也没有在图中宣称精确有效码字数。','',
 '右下目标身体只提供描述：几何15维加DOF19维，以及冻结SigLIP的768维关节名语义。源、目标调用同一个JointFeature对象：Linear(34→256)加上语义MLP(768→256→GELU→256)。目标关节数J_t决定动态queries数量。不会为新身体增加专属encoder或decoder参数。位置/旋转两流彼此有独立权重，但每一流的权重在所有身体之间共享。','',
 '每流解码器重复4次“CrossBlock读取本流FSQ memory→Hop EncBlock混合目标骨架关节”。每个CrossBlock和EncBlock都自带FFN及残差，因此一个解码层含两次attention和两次FFN。上方三个头实际分别是LayerNorm+Linear；图中旋转/接触的“2×”表示两套独立头，未合并LayerNorm参数。','',
 'position head输出3维；当前root_pos损失仅监督root关节输出第0通道的高度，不应把它画成独立root scalar head。contact head存在，但w_cbce=0；返回contact logit还需要ret_con=True。图中的3D position头不能解释为已独立验证的全部世界平移轨迹。','',
 '## 第二页：展开可复用神经模块','',
 '左列是Hop EncBlock。先LayerNorm，再用不同Wq/Wk/Wv投影生成Q、K、V；“3 projections of LN(x)”表示来源相同，不表示Q=K=V。每头维度256/8=32；学习到的hop-distance bias加在attention logits，softmax后乘V，再拼8个head并做输出Linear。其输出与原输入相加。','',
 '接着再次LayerNorm，经过256→1024→GELU→256的FFN，再做一次残差相加。黑色绕线表示未归一化的残差载体，不应改成2017图常见的post-norm。','',
 '右列CrossBlock对q和memory分别做LayerNorm，再作独立Q/K/V投影；memory是K/V来源，不能接进q的残差旁路。这里无hop bias，无因果mask。CrossBlock内部MHA采用PyTorch打包projection参数，图中分开画的是数学运算，不代表额外新增三层网络。','',
 '## 颜色与边界','',
 '粉色：embedding/投影；黄色：attention；蓝色：LayerNorm；绿色：FFN/输出头；紫色：身体或接触条件。紫色虚线传条件，黑灰箭头传特征。Nx括号表示有各自参数的重复层，不能误读为层内循环共享权重。训练用16路JudgeHead不属于主推理通路，故仅在source-map标注。tdec=0；未把独立时序解码或latent diffusion实验混入当前主模型。','',
 '## 精确源码映射','']
for k,r in refs.items():lines += [f"- **{k}**：[{Path(r['path']).name}:{r['start']}-{r['end']}](<{r['path']}:{r['start']}>) — {r['purpose']}"]
lines += ['', '## 复现','', '`build.py`生成两张PDF/SVG；`render.sh`生成240 dpi PNG与预览；`finalize.py`核对源码行号、真实权重shape、PDF文字边界及图片尺寸，并冻结source-map/manifest。模型、Docs和原图均未修改。']
(O/'中文读图说明.md').write_text('\n'.join(lines)+'\n')
files=[p for p in O.iterdir() if p.is_file() and p.name!='manifest.json']
manifest={'version':payload['version'],'status':payload['status'],'files':{p.name:sha(p) for p in sorted(files)}}
(O/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'frozen':payload['version'],'source_refs':len(refs),'shape_checks':len(expected),'figures':qa},ensure_ascii=False))
