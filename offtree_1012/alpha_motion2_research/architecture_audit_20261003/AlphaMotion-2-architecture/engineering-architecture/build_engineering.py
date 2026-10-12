from pathlib import Path
import json, hashlib, math, html, textwrap
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from reportlab.pdfbase.pdfmetrics import stringWidth

A=Path(__file__).resolve().parents[1]; OUT=A/'engineering-architecture'
SHAS={'UniMate':'2c5b384715aa63d8639b1ed7eb74bfe614570c7a','UMR':'d1cdaedaf67a827198da71c5fa831f48b2e51f91'}
REPO={'UniMate':'Friedrich-M/UniMate','UMR':'hanyang9/UMR'}
sources={}
def ref(key,tag,path,start,end):
    candidates=[OUT/'source'/tag/path,A/'source-snapshot'/tag/path,A/'tool-audit/audit-source'/tag/path]
    p=next(p for p in candidates if p.exists())
    assert 1<=start<=end<=len(p.read_text().splitlines()),(path,start,end)
    url=(f'https://github.com/{REPO[tag]}/blob/{SHAS[tag]}/{path}#L{start}-L{end}' if tag in REPO else None)
    sources[key]=dict(project=tag,path=path,start=start,end=end,local=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),url=url)
    return key

ref('U-config','UniMate','configs/uniml3d_60frames_graph_adaln.json',18,81)
ref('U-marker','UniMate','data_process/motion_export/export_general.py',92,180)
ref('R-default','UMR','humanoid_retarget_defaults.json',61,79)
ref('U-pre','UniMate','data_process/README.md',67,80)
ref('U-export','UniMate','data_process/scripts/run_export.sh',75,126)
ref('U-extract','UniMate','data_process/feature_extraction/extract_features.py',369,423)
ref('U-resume','UniMate','data_process/feature_extraction/extract_features.py',511,566)
ref('U-workers','UniMate','data_process/feature_extraction/extract_features.py',646,726)
ref('U-activity','UniMate','data_process/utils/skeleton.py',468,526)
ref('U-jump','UniMate','data_process/utils/motion_features.py',506,543)
ref('U-format','UniMate','data_process/utils/motion_features.py',563,590)
ref('U-text','UniMate','unimate/tools/precompute_text_emb.py',192,241)
ref('U-tcache','UniMate','unimate/utils/text_emb_cache.py',138,243)
ref('U-dataset','UniMate','unimate/dataset/factory.py',11,105)
ref('U-item','UniMate','unimate/dataset/mixture/dataset.py',827,920)
ref('U-splits','UniMate','unimate/dataset/mixture/dataset.py',1017,1148)
ref('U-collate','UniMate','unimate/dataset/mixture/collate.py',62,211)
ref('U-train','UniMate','unimate/training/train.py',79,142)
ref('U-loop','UniMate','unimate/training/train.py',187,267)
ref('U-checkpoint','UniMate','unimate/training/train.py',337,411)
ref('U-trainer','UniMate','unimate/training/trainer.py',143,170)
ref('U-viz','UniMate','unimate/training/trainer.py',176,253)
ref('U-infer','UniMate','unimate/inference/generate.py',78,125)
ref('U-output','UniMate','unimate/utils/visualization.py',30,176)
ref('U-lbs','UniMate','data_process/mesh_animation/animate_lbs.py',80,349)
ref('R-entry','UMR','scripts/humanoid_retarget_pipeline.py',1034,1068)
ref('R-compatible','UMR','scripts/humanoid_retarget_pipeline.py',390,511)
ref('R-build','UMR','scripts/humanoid_retarget_pipeline.py',698,885)
ref('R-train','UMR','scripts/humanoid_retarget_pipeline.py',890,942)
ref('R-loop','UMR','scripts/train_correspondence_template_residual_ae.py',644,746)
ref('R-save','UMR','scripts/train_correspondence_template_residual_ae.py',773,823)
ref('R-bind','UMR','scripts/retarget_smpl_to_humanoid_surface_vector.py',627,712)
ref('R-jobs','UMR','scripts/humanoid_retarget_pipeline.py',945,1009)
ref('R-solve','UMR','scripts/retarget_smpl_to_humanoid_surface_vector.py',2700,2754)
ref('R-results','UMR','scripts/retarget_smpl_to_humanoid_surface_vector.py',2826,2876)
ref('R-viewentry','UMR','scripts/humanoid_retarget_pipeline.py',1012,1031)
ref('R-view','UMR','scripts/visualize_robot_retarget_result.py',1742,2043)
ref('A-physics','AlphaMotion1','physeval.py',364,538)
ref('A-serv','AlphaMotion-release','src/alphamotion/service/app.py',111,151)
ref('A-jobs','AlphaMotion-release','src/alphamotion/service/app.py',661,710)
ref('F-schema','Foundation','dataset.py',25,101)
ref('F-cache','Foundation','cache.py',11,63)
ref('F-index','Foundation','index_existing.py',291,344)
ref('F-loader','Foundation','loader.py',24,109)

def module(id,name,entry,inputs,outputs,reuse,boundary,refs):
    return dict(id=id,name=name,entry=entry,inputs=inputs,outputs=outputs,reuse=reuse,boundary=boundary,sources=refs)
modules=[
module('U1','原始资产导出与并行worker','data_process/scripts/run_export.sh → motion_export/export_*.py',
       'Truebones FBX、Mixamo animation FBX、Objaverse GLB',
       'export motions NPZ、joint_names与summary、每asset完成marker',
       '复用worker分片、失败不合并summary、asset级完成marker；把原mesh及骨骼映射保留给surface adapter。',
       'Mixamo animation-only文件缺mesh，需要先把动作应用到人物rig；Blender版本与模型资产必须成为lineage。',['U-pre','U-export','U-marker']),
module('U2','标注、canonical化与前置清洗','render → vlm_caption / joint_annotation → feature_extraction.extract_features',
       '导出NPZ、多视图render、caption、clean/facing joints、人工patch列表',
       'motions NPZ、cond.npy、captions.json、filtered_clips.json、metadata.txt',
       '活动度/断裂统计、人工activity_keep、按object并行、阈值与metadata变更失效、失败对象隔离，可拆成可配置预处理步骤。',
       '当前canonical diameter scaling和grounding不是物理世界坐标；过滤在配置空间计算，指标不能直接标厘米。',['U-pre','U-extract','U-resume','U-workers','U-activity','U-jump','U-format']),
module('U3','文本embedding缓存','unimate.tools.precompute_text_emb + utils.text_emb_cache',
       'cond joint names、captions、encoder type/version',
       'joint_emb_cache.npz、caption_emb_cache.npz：pooled/tokens/mask',
       '复用按需加载encoder、补缺项、版本/维数核对、write-then-rename；生产key加入文本内容与encoder revision hash。',
       '当前缓存metadata记录encoder名称/格式，不是全链路输入内容寻址；已有同clip-key文字变更需显式失效，不能只检查key存在。',['U-text','U-tcache']),
module('U4','Dataset、采样、collate','dataset.factory → Mixture/MotionDataset → MixtureSampler → mixture_batch_collate',
       'feature目录、cond、text cache、split配置、保存的stats（推理）',
       'motion[B,J,12,T] + cond：joint/time/text masks、topology、mean/std、caption',
       '复用变长padding/masks、train/eval按配置分离、topology增删/合并增强、统计值随run保存。',
       '默认配置test_split_ratio=0，存在eval map不自动证明已做完整holdout benchmark；MixtureLoader载入策略不等同大规模流式数据仓库。',['U-config','U-dataset','U-item','U-splits','U-collate']),
module('U5','训练控制与模型工厂','training.train → models.factory + DiffusionTrainer',
       'JSON config、DataLoader、模型类、flow或DDPM',
       '梯度更新、EMA、TensorBoard loss/grad/lr、周期可视化',
       '复用Accelerate、分布式一致nonfinite skip、梯度累计/clip、EMA、配置落盘及保存归一化stats。',
       'training.diff_model选择flow/diffusion；这些是训练工程能力，不代表surface模型或物理验收已经存在。',['U-config','U-train','U-loop','U-trainer']),
module('U6','checkpoint与恢复','_save_checkpoint_and_visualize / _resume_from_checkpoint',
       'model、optimizer、LR scheduler、EMA、tracker',
       'checkpoint_step_*.pt + config.json + dataset_stats.npy',
       '复用明确run bundle与多状态恢复；生产将checkpoint提交与昂贵预览解耦。',
       '当前先visualize再save；保存字典未含RNG与DataLoader精确游标，不能称bitwise精确中断续训。',['U-checkpoint']),
module('U7','推理、诊断预览与mesh恢复','inference.sample / generate → visualization；data_process.mesh_animation',
       'checkpoint/config/stats、文本/Tpose、可选已知帧或编辑约束',
       'motion特征NPY、RIC与FK预览MP4；可另做GLB/FBX或逐帧LBS vertices',
       '复用统一generator调用、双重建对比、纯NumPy LBS和bind矩阵抽取。',
       '已审路径提供采样预览与数据过滤，未找到集成机器人动力学/sim/real benchmark runner；官网已审客户端播放已有animation而非在线模型API。',['U-infer','U-output','U-lbs','U-viz']),
module('R1','配置驱动stage编排','scripts/humanoid_retarget_pipeline.py',
       'defaults + robot/motion overrides、stage/force参数',
       'build → train → retarget → optional view；可单独运行',
       '复用阶段边界和兼容性检查；缓存存在但template/robot不符会重建。',
       '不是通用大数据预训练框架；缺slots时retarget阶段可以触发静态训练。',['R-entry','R-compatible']),
module('R2','canonical表面训练数据','build_correspondence_dataset / build_correspondence_ae_dataset.py',
       'SMPL-X/SOMA模板、robot MJCF/visual mesh、采样姿态/点数',
       '静态表面点云NPZ、mesh metadata、sample face ids、模板配置',
       '复用外表面采点、根坐标转换、robot visual geom提取与模板metadata。',
       '应保留visual/collision区别；单位与shape/betas信息属于资产版本，缺手指动作不能由静态mesh弥补。',['R-build']),
module('R3','静态对应训练与slots缓存','train_correspondence → train_correspondence_template_residual_ae.py',
       'canonical pointcloud/template + effective train config',
       'checkpoint-latest/final.pt；correspondence_slots_latest/final.npz',
       '复用模板残差基线、表面重建/排斥/边平滑损失、checkpoint及有序slots导出。',
       'Python有效默认500epochs/batch2；浏览器60另算。当前主模型无文本局部region锚定。',['R-default','R-train','R-loop','R-save']),
module('R4','slot绑定与重定向job','retarget_motion → surface_vector.py 或 batch.py',
       'slots、source motion、robot model、solver配置、optional stream chunk size',
       '每帧source目标、MuJoCo几何FK/Jacobian、Clarabel dq、最终qpos',
       '复用material-point绑定、接触overlay所需数据、约束solver作为baseline/pseudo-label生产器。',
       '有流式chunk脚本选择，不等于集群调度；动力学闭环控制不在该solver链内。',['R-bind','R-jobs','R-solve']),
module('R5','结果、检查与可视化','surface_vector.main → NPZ；visualize_result → GLFW/Viser',
       '优化qpos、frame ids/fps、来源信息',
       '最终qpos NPZ、cost日志、LQR前后temporal summary、surface/contact同步视图',
       '复用同步viewer、slot/normal/contact overlay、视频导出与结果兼容性检查。',
       '保存的final qpos和展示不是物理可执行性证明；solver cost不能替代统一QC metric suite。',['R-results','R-viewentry','R-view','R-compatible']),
module('A1','已有离线物理QC与产品服务','AlphaMotion1 physeval.evaluate；release service + LiveViewer',
       '带fps世界qpos + 每个robot MJCF；异步播放/编辑请求',
       '关节限制、滑移/穿透、jerk、平衡/逆动力学报告；SQLite job/结果与review UI',
       '优先把physeval封装为embodiment条件指标worker；服务与同步viewer作为审核壳。',
       'physeval是离线一致性，非闭环sim/real。已有服务重启把unfinished job标failed，不自动恢复执行。',['A-physics','A-serv','A-jobs']),
module('F1','本轮基础工程已实现','pretraining_foundation dataset/cache/index_existing/loader',
       '旧XWBC/Alpha1缓存、UniML3D样本、manifest与holdout配置',
       '带坐标/来源/missing mask样本；hash校验缓存；SQLite索引；mmap窗口batch',
       '作为统一schema、缓存、分组split和DataLoader起点；已有按输入fingerprint重用与atomic index替换。',
       '主任务报告31项基础测试通过；本附录没有重跑该测试。surface/binding/region仍未绑定，不是生产级清洗系统验收。',['F-schema','F-cache','F-index','F-loader'])
]

contracts=[
dict(interface='AssetRecord',fields='corpus/source sequence/embodiment、raw hash/revision/license、mesh与物理模型hash、joint map、派生链',purpose='同名文件内容变更会失效；保留原始动作及对应资产，避免错误复用与跨split泄漏。'),
dict(interface='MotionEnvelope',fields='timestamps/fps、length unit、up axis/handedness、world↔canonical transform、rotation convention、root trajectory、parents/DOFs、valid frame/joint masks',purpose='m与topology-normalized明确分支；仅在有尺度锚点时换算。物理指标始终消费world坐标，不读取viewer normalization。'),
dict(interface='SurfaceBinding',fields='mesh revision、faces/face id+barycentric、geom/link-local或LBS weights+inverse bind、joint map、sampling seed、point mask；region文本与membership另带来源',purpose='geometry每asset缓存，motion窗口只引用；固定material points才能比较接触与表面轨迹，缺失region不补零真值。'),
dict(interface='MetricResult',fields='metric name/version/unit、per-frame值+summary、valid/unknown mask、asset/clip/version、reason codes、参数/置信度、evidence tier',purpose='指标只测量；pass/review/reject由独立PolicyVersion决定。未知执行器范围是unknown，不是pass。'),
dict(interface='JobSpec / StageResult',fields='输入hash+adapter/model/metric revision+config hash+seed、资源需求、尝试次数、lease、阶段产物校验、committed marker',purpose='同spec幂等；临时产物校验后提交引用，读者只见committed。瞬态错误有限重试；schema/数据错误进入quarantine，不无限重跑。'),
dict(interface='DatasetRelease',fields='不可变sample manifest、source-sequence与body split、所有依赖hash、metric/policy版本、批准记录与exclusion reasons',purpose='train和eval读取同一版本化release但分区严格隔离；人工审核不覆盖原始指标，保留理由与变更记录。')]

production=[
dict(component='纯指标内核与分级worker',implementation='先把finite/shape、SO(3)、FK残差、骨长、速度/加速度、surface穿透/滑移写成无副作用、固定输入得固定输出的CPU函数；适合向量化者批处理。模型、MuJoCo/逆动力学、sim tracking各自独立worker进程，保留每robot资产。',acceptance='同golden clip重复运行及不同worker数结果在预先登记的数值容差内一致；逐帧reason定位可重现。'),
dict(component='内容寻址与原子发布',implementation='key包含raw bytes hash、代码/配置/资产revision、采样seed和metric version；临时目录写完，验证内容hash后原子提交manifest指针。目录存在或进程退出0不视为完成。',acceptance='在任意阶段中断不暴露半成品；相同job重放复用相同产物；改输入内容但不改文件名必须失效。'),
dict(component='调度、恢复与资源预算',implementation='每stage申请CPU/GPU/RAM/disk预算；有限队列长度和高水位暂停上游；每clip/job有lease与heartbeat；阶段级resume、退避重试、最大尝试次数、dead-letter/quarantine。下载、渲染、模型训练各用独立资源池。',acceptance='worker中断后lease可回收且不重复提交；磁盘不足时不会继续制造孤立产物；慢viewer不阻塞指标提交。'),
dict(component='阈值政策与人工审核',implementation='PolicyVersion按任务/本体和metric单位解释结果，输出accept/review/reject；界面显示坏帧、raw-vs-derived、surface/contact overlay与reason；人工覆盖附原因且保留旧政策。',acceptance='同指标换policy只重算决策、不重跑FK/模型；unknown和测量失败不会被视为成功；审核样本可以按来源与错误类型分层抽样。'),
dict(component='可观测性',implementation='结构化事件带job/clip/stage/attempt；记录吞吐、排队延迟、阶段耗时、缓存命中、失败/重试、各reason占比、GPU/RAM/disk；一条trace把输入、计算、审核和release相连。',acceptance='能由任一发布样本追到原文件、模型/指标版本、计算日志与审核理由；报警针对状态变化而不是轮询噪音。'),
dict(component='golden clips与失败集',implementation='保留人工核对的bind pose、跨单位/坐标、短序列、缺模态、左右手、遮挡contact、坏quaternion、旧缓存、被杀worker等样本。几何QC、离线动力学、闭环sim、真实机器人分别记录证据。',acceptance='每次adapter/metric/policy升级都跑固定回归集并给差异；性能、吞吐、故障恢复和sim/real各有实测记录，31个基础测试不能替代这些验收。')]

colors={'ink':'#172536','muted':'#536476','line':'#8b9aaa','data':'#e9f2fb','compute':'#e6f2ef','artifact':'#faf1df','review':'#f1ecfa','proposed':'#fff1df','light':'#f5f7fa'}
W,H=1640,1040
class Figure:
    def __init__(self,name,title,subtitle,status):
        self.name=name;self.nodes={};self.edges=[];self.svg=[];self.boxtexts=[]
        self.c=canvas.Canvas(str(OUT/(name+'.pdf')),pagesize=(W,H));self.c.setTitle(title)
        self.svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}"><title>{html.escape(title)}</title>')
        self.rect(0,0,W,H,'#ffffff',None)
        self.text(55,56,title,31,bold=True)
        self.text(55,90,subtitle,17,color=colors['muted'])
        self.text(55,118,status,14,color=colors['muted'])
    def text(self,x,y,s,size=17,bold=False,color=None):
        color=color or colors['ink'];font='Helvetica-Bold' if bold else 'Helvetica'
        self.c.setFillColor(HexColor(color));self.c.setFont(font,size);self.c.drawString(x,H-y,s)
        self.svg.append(f'<text x="{x}" y="{y}" font-family="Arial,Helvetica,sans-serif" font-size="{size}" font-weight="{700 if bold else 400}" fill="{color}">{html.escape(s)}</text>')
    def rect(self,x,y,w,h,fill,stroke='#d4dce5',radius=0):
        self.c.setFillColor(HexColor(fill));self.c.setStrokeColor(HexColor(stroke or fill));self.c.setLineWidth(1.2)
        self.c.roundRect(x,H-y-h,w,h,radius,stroke=bool(stroke),fill=1)
        self.svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke or fill}"/>')
    def lane(self,y,label):
        self.rect(32,y-36,W-64,188,colors['light'],None,14);self.text(52,y-12,label,15,True,colors['muted'])
    def node(self,id,x,y,title,lines,kind='compute',w=320,h=120):
        self.nodes[id]=dict(id=id,x=x,y=y,w=w,h=h,title=title,lines=lines,kind=kind)
        self.rect(x,y,w,h,colors[kind],'#c5d2dd',12)
        self.text(x+18,y+29,title,20,True)
        for i,line in enumerate(lines):
            assert stringWidth(line,'Helvetica',16)<w-34,(id,line)
            self.text(x+18,y+55+i*22,line,16)
        assert 55+(len(lines)-1)*22<=h-12,(id,lines)
    def point(self,id,side):
        n=self.nodes[id];x,y,w,h=n['x'],n['y'],n['w'],n['h']
        return {'left':(x,y+h/2),'right':(x+w,y+h/2),'top':(x+w/2,y),'bottom':(x+w/2,y+h)}[side]
    def arrow(self,a,b,side_a='right',side_b='left',via=None,label=None,label_xy=None,dashed=False):
        pts=[self.point(a,side_a)]+(via or [])+[self.point(b,side_b)]
        self.edges.append(dict(source=a,target=b,points=pts,label=label,dashed=dashed))
        self.c.setStrokeColor(HexColor(colors['line']));self.c.setFillColor(HexColor(colors['line']));self.c.setLineWidth(2)
        self.c.setDash(6,4) if dashed else self.c.setDash()
        p=self.c.beginPath();p.moveTo(pts[0][0],H-pts[0][1])
        for x,y in pts[1:]:p.lineTo(x,H-y)
        self.c.drawPath(p)
        x,y=pts[-1];px,py=pts[-2];angle=math.atan2(y-py,x-px)
        tip=[(x,y),(x-9*math.cos(angle)+4*math.sin(angle),y-9*math.sin(angle)-4*math.cos(angle)),(x-9*math.cos(angle)-4*math.sin(angle),y-9*math.sin(angle)+4*math.cos(angle))]
        q=self.c.beginPath();q.moveTo(tip[0][0],H-tip[0][1])
        for xx,yy in tip[1:]:q.lineTo(xx,H-yy)
        q.close();self.c.drawPath(q,stroke=0,fill=1);self.c.setDash()
        self.svg.append(f'<polyline points="'+ ' '.join(f'{x},{y}' for x,y in pts)+f'" fill="none" stroke="{colors["line"]}" stroke-width="2"'+(' stroke-dasharray="6 4"' if dashed else '')+'/>')
        self.svg.append(f'<polygon points="'+ ' '.join(f'{x},{y}' for x,y in tip)+f'" fill="{colors["line"]}"/>')
        if label:self.text(*(label_xy or (pts[0][0]+9,pts[0][1]-8)),label,13,color=colors['muted'])
    def foot(self,lines):
        self.rect(32,870,W-64,145,'#f5f7fa',None,12)
        for i,line in enumerate(lines):self.text(55,900+25*i,line,16,color=colors['muted'])
    def save(self):
        self.c.showPage();self.c.save();self.svg.append('</svg>');(OUT/(self.name+'.svg')).write_text('\n'.join(self.svg))
        return dict(name=self.name,nodes=list(self.nodes.values()),edges=self.edges)

X=[55,455,855,1255]
figs=[]
f=Figure('unimate_engineering','UniMate | code and artifact architecture',
 'Offline corpus preparation, text-conditioned training, sampling and inspection',
 'Existing implementation - arrows show calls or saved artifact flow; code locations are in the companion audit.')
for y,l in [(205,'1   CORPUS PREPARATION'),(455,'2   TRAINING'),(705,'3   INFERENCE / INSPECTION')]:f.lane(y,l)
nodes=[
('u1',0,205,'Raw rigged assets',['FBX / GLB; animation clips','Original mesh + skeleton','Truebones / Mixamo / Objaverse'],'data'),
('u2',1,205,'Export + render',['Blender workers; motion NPZ','Multi-view frames; joint names','Mixamo: attach a character rig'],'compute'),
('u3',2,205,'Annotate + extract',['Captions; names; facing','Canonicalize; trim; activity/jump QC','Per-object workers + resume'],'compute'),
('u4',3,205,'Feature corpus',['Clip NPZ + cond.npy + captions','Filter report + cached cond_parts','Mesh stays in source assets'],'artifact'),
('u5',0,455,'Text feature cache',['Joint vectors; caption tokens','Encoder / format validation','Atomic write; encode misses'],'data'),
('u6',1,455,'Mixture + collate',['Split / sampler / topology aug','Normalization; variable padding','Motion + condition + valid masks'],'compute'),
('u7',2,455,'Train + model factory',['JSON config; graph/full backbone','Accelerate; flow or DDPM; EMA','Global NaN guard; loss logging'],'compute'),
('u8',3,455,'Saved run bundle',['Model + optimizer + LR + EMA','Step / epoch; config + data stats','Restore path; sample previews'],'artifact'),
('u9',0,705,'Generation conditions',['Text + target topology / T-pose','Optional known frames / edits','Saved normalization contract'],'data'),
('u10',1,705,'Sampling entry',['Text / in-between / edit / expand','CFG + flow ODE or diffusion','Load matching run configuration'],'compute'),
('u11',2,705,'Reconstruct + inspect',['Denormalize; RIC and rotation FK','NPY motion + MP4 comparisons','Preview is not physics validation'],'review'),
('u12',3,705,'Optional mesh animation',['Generated motion + source rig','Blender or pure NumPy LBS','GLB / FBX / surface trajectories'],'artifact')]
for id,c,y,title,lines,kind in nodes:f.node(id,X[c],y,title,lines,kind)
for row in [('u1','u2','u3','u4'),('u5','u6','u7','u8'),('u9','u10','u11','u12')]:
 for a,b in zip(row,row[1:]):f.arrow(a,b)
f.arrow('u4','u6','bottom','top',[(1415,375),(615,375)],'clips + topology', (910,366))
f.arrow('u4','u5','bottom','top',[(1415,398),(215,398)],'names / captions',(275,390))
f.arrow('u8','u10','bottom','top',[(1415,630),(615,630)],'weights + config + stats',(900,622))
f.foot(['Existing reuse: per-object caching, metadata-aware invalidation, atomic text cache, padded masks and saved normalization.',
        'Boundary: numerical filtering and sample previews are present; no audited integrated robot dynamics / sim / real benchmark runner.',
        'Checkpoint restore does not save RNG / exact data cursor here. The audited public website plays prepared animation assets.',
        'Model-layer details are intentionally omitted; this diagram describes software stages and their artifacts.'])
figs.append(f.save())

f=Figure('umr_engineering','UMR | code and artifact architecture',
 'Canonical surface learning, constrained retargeting jobs and inspection tools',
 'Existing Python pipeline; the browser Studio is a separate execution path, not a neural-network layer.')
for y,l in [(205,'1   STATIC CORRESPONDENCE'),(455,'2   PER-MOTION RETARGETING'),(705,'3   TOOLING / ALTERNATE RUNTIME')]:f.lane(y,l)
nodes=[
('r1',0,205,'Config + geometry',['Defaults + robot / motion config','SMPL-X / SOMA; robot MJCF','Mesh, shape and sampling pose'],'data'),
('r2',1,205,'Build surface dataset',['Exterior surface sampling','Canonical frame + mesh metadata','Static point-cloud NPZ'],'compute'),
('r3',2,205,'Train correspondence',['PointNet template-residual AE','Reconstruction + regularizers','Pipeline default: 500 epochs'],'compute'),
('r4',3,205,'Correspondence artifacts',['Model / optimizer checkpoint','Ordered reconstructed slots NPZ','Template / robot reuse checks'],'artifact'),
('r5',0,455,'Source motion',['Sequence + frame range + fps','Body / object motion if present','Source model and shape'],'data'),
('r6',1,455,'Bind + update surfaces',['Source and robot slot binding','Source target points / normals','Robot FK + point Jacobians'],'compute'),
('r7',2,455,'Constrained solver',['Clarabel per-frame update','Contact / limits / smoothness','Optional chunks; LQR filtering'],'compute'),
('r8',3,455,'Retarget result',['Final qpos NPZ + frame metadata','Solver cost / temporal logs','Result compatibility checks'],'artifact')]
for id,c,y,title,lines,kind in nodes:f.node(id,X[c],y,title,lines,kind)
for row in [('r1','r2','r3','r4'),('r5','r6','r7','r8')]:
 for a,b in zip(row,row[1:]):f.arrow(a,b)
f.arrow('r4','r6','bottom','top',[(1415,390),(615,390)],'ordered slots + geometry references',(850,380))
f.node('r9',55,705,'GLFW / Viser inspection',['Synchronized source / robot; slots and contact overlays','Playback / video / human inspection; uses geometric FK','This path is not closed-loop dynamics simulation'],'review',w=720)
f.node('r10',855,705,'Browser Studio: separate runtime',['Client loads source pack + robot; WASM correspondence','Then MuJoCo Jacobians + Clarabel; browser default 60 epochs','Optional server API branch exists; backend availability unverified'],'review',w=720)
f.arrow('r8','r9','bottom','top',[(1415,620),(415,620)],'saved result + source references',(810,611))
f.foot(['Existing reuse: explicit build / train / retarget / view stages, compatible-slot reuse, robot mesh binding and contact overlays.',
        'The learned stage uses canonical static geometry. New or incompatible slots can trigger training before retargeting.',
        'Solver logs and viewer checks are not a versioned cross-robot acceptance suite. Python and browser training settings differ.',
        'Surface correspondence outputs are useful pseudo-labels; they are not manual ground-truth correspondences.'])
figs.append(f.save())

f=Figure('alphamotion2_cleaning_proposed','AlphaMotion-2 | proposed production cleaning pipeline',
 'One traceable sample contract for skeleton, surface, semantics and embodiment-specific evidence',
 'PROPOSED - combines reusable modules with new orchestration; this is an implementation target, not a readiness claim.')
for y,l in [(205,'1   INGEST / VALIDATE / SCHEDULE'),(455,'2   DERIVE / MEASURE / DECIDE'),(705,'3   RETRY / RELEASE / CONSUME / REVIEW')]:f.lane(y,l)
nodes=[
('p1',0,205,'Asset intake + lineage',['Raw bytes / revision / source hash','Clip group + body + asset links','Immutable input manifest'],'data'),
('p2',1,205,'Canonical adapters',['Units / frames / timestamps','Skeleton + root + valid masks','direct / derived / pseudo / missing'],'proposed'),
('p3',2,205,'Fast deterministic QC',['Finite / SO(3) / FK / time checks','CPU kernels; batched execution','Versioned metrics + reason codes'],'proposed'),
('p4',3,205,'Budgeted work queue',['Stage leases; idempotent job key','CPU / GPU / RAM / disk quotas','Backpressure; bounded retry'],'proposed'),
('p5',0,455,'Surface + region cache',['Rigid FK / LBS material points','Mesh binding + region text source','Asset hash + sample seed + version'],'proposed'),
('p6',1,455,'Model / retarget workers',['Pinned model + condition schema','Baseline UMR / new shared model','Motion and correspondence lineage'],'proposed'),
('p7',2,455,'Evidence metric workers',['Geometry QC; offline dynamics','Separate sim / real evidence jobs','Per-frame values + validity + units'],'proposed'),
('p8',3,455,'Policy + review gate',['Policy version; accept/review/reject','Unknown is distinct from pass','Reasons and overrides retained'],'review'),
('p9',0,705,'Quarantine + retry',['Invalid data: isolate with reason','Transient failure: bounded retry','Resume from committed stages'],'review'),
('p10',1,705,'Versioned dataset release',['Validated artifacts; atomic manifest','Input / metric / policy references','Source-group + embodiment split'],'artifact'),
('p11',2,705,'Train + benchmark readers',['Mmap / bounded streaming windows','Same schema; disjoint partitions','Golden clips + failure regression'],'data'),
('p12',3,705,'Review + operations UI',['Surface / skeleton / contact replay','Bad frames + trace to raw input','Queue / cache / failure telemetry'],'review')]
for id,c,y,title,lines,kind in nodes:f.node(id,X[c],y,title,lines,kind)
for row in [('p1','p2','p3','p4'),('p5','p6','p7','p8'),('p10','p11','p12')]:
 for a,b in zip(row,row[1:]):f.arrow(a,b)
f.arrow('p4','p5','bottom','top',[(1415,390),(215,390)],'validated jobs + bounded resource grants',(560,380))
f.arrow('p8','p10','bottom','top',[(1415,620),(615,620)],'accepted / reviewed',(1000,610))
f.arrow('p8','p9','bottom','top',[(1415,648),(215,648)],'rejected or failed stages',(525,638),dashed=True)
f.arrow('p8','p12','bottom','top',[(1595,575),(1595,681),(1415,681)],'review',(1530,610),dashed=True)
f.foot(['Reusable today: foundation schema/cache/index/loader, UniMate cleaning + LBS, UMR bindings/overlays, AlphaMotion physics QC/service shell.',
        'Still to build: dense surface + region binding, versioned metric/policy registry, robust stage scheduler and cross-stage commit contract.',
        'Acceptance evidence stays separate: geometry QC | offline dynamics | closed-loop simulation | real robot. Missing evidence stays unknown.',
        'Every release must pass golden-clip regression, crash/replay checks and measured resource limits; basic unit tests alone are insufficient.'])
figs.append(f.save())

payload=dict(title='训练、评测与工具链工程架构审计',status='source_audit_and_proposal',modules=modules,contracts=contracts,production_requirements=production,sources=sources,figures=figs,
limits=['未修改模型、启动训练或部署服务','diagram arrows是实际调用/产物流程；目标图全标Proposed','基础工程31项测试由主任务报告，本附录未重跑','UMR browser details继承已完成tool audit，Python入口独立核查','不将统一运动学表示等同统一动力学控制'])
(OUT/'engineering_architecture.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
def link(key):
 s=sources[key];label=f"{s['path']}:{s['start']}-{s['end']}";target=s['url'] or f"{s['local']}:{s['start']}"
 return f'[{label}](<{target}>)'
doc=['# 训练、评测与工具链工程架构','',
 '三张图分别回答：UniMate如何把原始资产变成训练与生成结果；UMR如何训练静态表面对应再批次重定向；AlphaMotion-2如何把这些现成部件变成可追溯、可恢复的数据清洗基础设施。箭头表示代码调用或落盘产物传递，不表示神经网络层。','',
 '最值得复用的组合是：UniMate的按资产清洗/缓存与LBS，UMR的表面绑定/约束重定向和contact可视化，AlphaMotion已有的本体条件离线物理QC，再以本轮schema/index/cache/loader为数据接口基础。不能只搬训练脚本或网页外观就称为production-ready。','',
 '## 三张工程结构图','', '优先使用文档简版：宽510 pt，正文字号9.5 pt；按468 pt宽插入后正文约8.72 pt。完整大图保留更多产物说明，适合单独放大。两版与逐条源码旁表配合阅读。','']
for name,title in [('unimate_engineering','UniMate现状'),('umr_engineering','UMR现状'),('alphamotion2_cleaning_proposed','AlphaMotion-2拟议清洗管道')]:
 doc += [f'### {title}','',f'![{title}（文档简版）]({OUT/name}_doc.png)','',f'[文档版SVG]({OUT/name}_doc.svg) · [文档版PDF]({OUT/name}_doc.pdf) · [完整大图SVG]({OUT/name}.svg) · [完整大图PDF]({OUT/name}.pdf)','']
doc += ['## 现有模块与复用边界','']
for m in modules:
 doc += [f"### {m['name']}",'',f"入口：`{m['entry']}`。",'',f"输入：{m['inputs']}。输出：{m['outputs']}。",'',f"可复用：{m['reuse']}",'',f"边界：{m['boundary']}",'','源码：'+'；'.join(link(k) for k in m['sources'])+'。','']
doc += ['## 两套仓库的工程比较','',
 '|维度|UniMate现有实现|UMR现有实现|AlphaMotion-2接入动作|',
 '|---|---|---|---|',
 '|任务边界|素材导出/标注/清洗、模型训练、推理预览|canonical对应训练、运动约束优化、viewer|统一StageResult，不要求两者拥有相同训练形态|',
 '|数据恢复|object级缓存、参数/metadata摘要、atomic缓存、失败对象重跑|dataset/slots/result兼容性检查；缺slots补建补训|统一输入内容hash、code/config版本与commit marker|',
 '|训练恢复|model/optimizer/LR/EMA/step恢复；未见RNG/精确数据游标|静态AE周期checkpoint；pipeline优先复用已完成slots|训练checkpoint与清洗job恢复分别定义，避免混为一谈|',
 '|评测入口|清洗统计、loss、RIC/FK采样预览|solver cost、temporal summary、contact overlay|复用算法后抽纯指标内核，补metric版本/validity/reason与policy|',
 '|surface资产|raw rig + LBS可派生；训练feature NPZ未携带dense绑定|canonical有序slots绑定并追踪表面|mesh/bind/region单独asset契约；每帧复用material point|',
 '|线上工具|已审官网为预生成animation播放|浏览器默认WASM训练+优化；可选server分支|AlphaMotion既有FastAPI/SQLite/LiveViewer更接近可复用服务壳|','',
 '## 拟议最小数据与模块接口','']
for c in contracts:doc += [f"**{c['interface']}**：{c['fields']}。{c['purpose']}",'']
doc += ['## 从基础工程到生产验收还差什么','']
for r in production:doc += [f"**{r['component']}**：{r['implementation']}",'',f"验收：{r['acceptance']}",'']
doc += ['## 建议实施顺序','',
 '1. 先将现有真实样本接入统一MotionEnvelope，保留world与canonical两条表示，固定source-group/body划分；把缺mesh、缺region、缺contact显式标unknown。',
 '2. 对少量golden assets完成robot rigid-link和人体/艺术rig LBS两种SurfaceBinding，验证bind pose、骨骼重排、单位、固定face+barycentric轨迹；加入局部区域文本来源。',
 '3. 把现有数值过滤与physeval封装成不依赖viewer的metric worker，产每帧值与reason；先形成清洗报告再逐步开启自动拒收。',
 '4. 在现有cache/index基础上补JobSpec、lease、预算、原子manifest提交、有限重试与quarantine；用故障注入证明恢复，再扩大数据量。',
 '5. 审核UI接surface/region/contact overlay与bad-frame跳转；稳定数据release后，训练与评测读取相同schema。动力学、sim与real证据保持独立，分别建立闭环。','',
 '本轮没有修改模型、运行训练、触发重定向或部署服务。两张现状图依据固定版本源码；目标图是设计提案。本附录只检查本轮基础工程源码，不以已有31项基础测试替代吞吐、容错、全量清洗或sim2real验收。','',
 '网站客户端与AlphaMotion发布工程的详细来源和许可边界见[工具与surface数据审计](<'+str(A/'tool-audit/tools_and_surface_data.md')+'>)。','']
(OUT/'工程审计说明.md').write_text('\n'.join(doc))
(OUT/'source-validation.json').write_text(json.dumps(dict(source_references=len(sources),line_ranges_valid=True,figure_count=len(figs),neural_model_changed=False,training_started=False),indent=2)+'\n')
print(json.dumps({'figures':[f['name'] for f in figs],'modules':len(modules),'sources':len(sources),'report_chars':len('\n'.join(doc))},ensure_ascii=False))
