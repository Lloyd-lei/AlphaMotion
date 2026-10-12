"""Editable SVG + vector PDF architecture figures; all coordinates in points.

Source facts are pinned in sources.md. No network or model execution is needed.
The same primitive list draws SVG and PDF, so both remain vector-native.
"""
from pathlib import Path
import math, html, json, re, subprocess
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor

HERE = Path(__file__).resolve().parent
HERE.mkdir(exist_ok=True)
pdfmetrics.registerFont(TTFont('Arial', '/System/Library/Fonts/Supplemental/Arial.ttf'))
pdfmetrics.registerFont(TTFont('ArialB', '/System/Library/Fonts/Supplemental/Arial Bold.ttf'))
INK='#172A3A'; MUTED='#536576'; LINE='#A7B4BE'; WHITE='#FFFFFF'
BLUE='#176B98'; BLUE_BG='#EDF5FA'; GREEN='#147D75'; GREEN_BG='#EAF6F3'
ORANGE='#AF6220'; ORANGE_BG='#FFF3E6'; PURPLE='#705BA2'; PURPLE_BG='#F3F0F9'
GRAY_BG='#F4F6F8'
ALL=[]

class Fig:
    def __init__(self, name, title, subtitle, w=1140, h=860):
        self.name=name; self.w=w; self.h=h; self.ops=[]; self.warnings=[]
        self.compact=w<=600
        self.rect(0,0,w,h,WHITE,WHITE)
        if self.compact:
            self.text(22,18,title,14.5,INK,True)
            for i,t in enumerate(self.wrap(subtitle,w-44,7.7)):
                self.text(22,41+i*10,t,7.7,MUTED)
            self.line([(22,67),(w-22,67)],LINE,.6)
        else:
            self.text(34,36,title,26,INK,True)
            self.text(34,70,subtitle,13.5,MUTED)
            self.line([(34,98),(w-34,98)],LINE,0.8)
    def rect(self,x,y,w,h,fill=WHITE,stroke=LINE,sw=1,dash=False,r=8):
        self.ops.append(('rect',x,y,w,h,fill,stroke,sw,dash,r))
    def line(self,pts,color=LINE,sw=1.4,dash=False,arrow=False):
        self.ops.append(('line',pts,color,sw,dash,arrow))
    def text(self,x,y,s,size=14,color=INK,bold=False,align='left'):
        self.ops.append(('text',x,y,s,size,color,bold,align))
    def wrap(self,s,width,size,bold=False):
        lines=[]
        for paragraph in s.split('\n'):
            line=''
            for word in paragraph.split(' '):
                trial=(line+' '+word).strip()
                if pdfmetrics.stringWidth(trial,'ArialB' if bold else 'Arial',size)>width and line:
                    lines.append(line);line=word
                else: line=trial
            lines.append(line)
        return lines
    def box(self,x,y,w,h,title,body='',fill=WHITE,color=BLUE,fs=14,ts=16,dash=False):
        self.rect(x,y,w,h,fill,color,1.2,dash)
        titlelines=self.wrap(title,w-28,ts,True)
        yy=y+15
        for t in titlelines:
            self.text(x+14,yy,t,ts,color,True);yy+=ts*1.2
        if body:
            yy+=8
            for t in self.wrap(body,w-28,fs):
                self.text(x+14,yy,t,fs,INK);yy+=fs*1.25
        if yy>y+h-5:self.warnings.append(f'{self.name}: overflow {title}: {yy-y:.1f}/{h}')
    def arrow(self,x1,y1,x2,y2,color=LINE,dash=False):
        self.line([(x1,y1),(x2,y2)],color,1.6,dash,True)
    def patharrow(self,pts,color=LINE,dash=False):self.line(pts,color,1.6,dash,True)
    def label(self,x,y,s,color=BLUE,size=13):self.text(x,y,s,size,color,True)
    def footer(self,s):
        if self.compact:
            self.line([(22,self.h-35),(self.w-22,self.h-35)],LINE,.6)
            for i,t in enumerate(self.wrap(s,self.w-44,7.1)):
                self.text(22,self.h-26+i*8.5,t,7.1,MUTED)
        else:
            self.line([(34,self.h-47),(self.w-34,self.h-47)],LINE,.7)
            self.text(34,self.h-30,s,11,MUTED)
    def save(self):
        c=canvas.Canvas(str(HERE/(self.name+'.pdf')),pagesize=(self.w,self.h))
        c.setTitle(self.name.replace('_',' '));c.setAuthor('ArenaLabs - source-audited architecture study')
        svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}pt" height="{self.h}pt" viewBox="0 0 {self.w} {self.h}">',
             '<title>'+html.escape(self.name)+'</title>',
             '<desc>Editable source-audited architecture diagram. Text remains editable. See sources.md.</desc>']
        for op in self.ops:
            if op[0]=='rect':
                _,x,y,w,h,fill,stroke,sw,dash,r=op
                c.setFillColor(HexColor(fill));c.setStrokeColor(HexColor(stroke));c.setLineWidth(sw);c.setDash([5,4] if dash else [])
                c.roundRect(x,self.h-y-h,w,h,r,stroke=1,fill=1)
                dd=' stroke-dasharray="5 4"' if dash else ''
                svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{dd}/>')
            elif op[0]=='text':
                _,x,y,s,size,col,bold,align=op
                baseline=y+size*.82
                font='ArialB' if bold else 'Arial';c.setFont(font,size);c.setFillColor(HexColor(col))
                if align=='center':c.drawCentredString(x,self.h-baseline,s)
                elif align=='right':c.drawRightString(x,self.h-baseline,s)
                else:c.drawString(x,self.h-baseline,s)
                anchor={'left':'start','center':'middle','right':'end'}[align]
                svg.append(f'<text x="{x}" y="{baseline:.3f}" font-family="Arial, sans-serif" font-size="{size}" font-weight="{700 if bold else 400}" fill="{col}" text-anchor="{anchor}">{html.escape(s)}</text>')
            elif op[0]=='line':
                _,pts,col,sw,dash,arr=op
                c.setLineWidth(sw);c.setStrokeColor(HexColor(col));c.setDash([5,4] if dash else [])
                p=c.beginPath();p.moveTo(pts[0][0],self.h-pts[0][1])
                for x,y in pts[1:]:p.lineTo(x,self.h-y)
                c.drawPath(p)
                xy=' '.join(f'{x},{y}' for x,y in pts);dd=' stroke-dasharray="5 4"' if dash else ''
                svg.append(f'<polyline points="{xy}" fill="none" stroke="{col}" stroke-width="{sw}"{dd}/>')
                if arr:
                    (ax,ay),(bx,by)=pts[-2:];theta=math.atan2(by-ay,bx-ax);n=7
                    tri=[(bx,by),(bx-n*math.cos(theta-.42),by-n*math.sin(theta-.42)),(bx-n*math.cos(theta+.42),by-n*math.sin(theta+.42))]
                    c.setFillColor(HexColor(col));p=c.beginPath();p.moveTo(tri[0][0],self.h-tri[0][1]);p.lineTo(tri[1][0],self.h-tri[1][1]);p.lineTo(tri[2][0],self.h-tri[2][1]);p.close();c.drawPath(p,fill=1,stroke=0)
                    svg.append('<polygon points="'+' '.join(f'{x:.3f},{y:.3f}' for x,y in tri)+f'" fill="{col}"/>')
        c.showPage();c.save();svg.append('</svg>');(HERE/(self.name+'.svg')).write_text('\n'.join(svg))
        ALL.append({'name':self.name,'width_pt':self.w,'height_pt':self.h,'min_font_pt':min(o[4] for o in self.ops if o[0]=='text'),'layout_warnings':self.warnings})

def overview():
    f=Fig('01_implemented_systems','Implemented systems: three different computational contracts',
          'Source-audited comparison | Architecture and inference mechanisms; no performance ranking',h=900)
    xs=[34,406,778];w=328
    names=[('A  AlphaMotion-1',BLUE,'megatron / L_M20k_0903'),('B  UniMate',GREEN,'Graph AdaLN / flow matching'),('C  UMR',ORANGE,'Surface correspondence + optimization')]
    for x,(t,col,sub) in zip(xs,names):
        f.text(x,118,t,23,col,True);f.text(x,148,sub,13,MUTED)
    rows=[180,320,460,600,735];hs=[110,110,110,110,97]
    content=[
       [('Source pose + morphology [A1]','Single frame: B x Js x 9\nJoint geometry 15 + DOF 19\nFrozen SigLIP joint names: 768',BLUE_BG),
        ('Shared skeleton encoder [A2]','Width 256; 4 hop-biased blocks\n8 attention heads\nMasked pose + contact conditioning',BLUE_BG),
        ('Dual FSQ bottleneck [A3]','Position: 4 slots; rotation: 128\n256 -> 13 -> scalar rounding -> 256\nOne stage; configured levels [32] x 13',BLUE_BG),
        ('Target-skeleton queries [A4]','Each stream: 4 cross + hop blocks\nPosition 3; rotation 6; contact 1\nTarget geometry, DOF and text names',BLUE_BG),
        ('Motion mapping','Discrete, frame-wise codec\nNo temporal denoising in this path',GRAY_BG)],
       [('Noisy sequence + T-pose [U1]','B x J x 12 x 60\nFrozen FLAN-T5-base: 768\nCaption and joint-name semantics',GREEN_BG),
        ('Motion / morphology tokens [U2]','Root / joint MLPs: 12 -> 512\nPrepend one T-pose frame\nDepth + name embeddings',GREEN_BG),
        ('10 spatiotemporal blocks [U3]','Spatial graph attention, temporal attention\n8 heads; 512 width; adaLN gates\nSwiGLU hidden width: 1365',GREEN_BG),
        ('Flow velocity + ODE [U4]','Root aggregates non-root joints\nRoot / joint heads: 512 -> 12\nIntegrate velocity from noise to motion',GREEN_BG),
        ('Motion generation / control','Continuous sequence model\nConditioning and sampling form the task',GRAY_BG)],
       [('Canonical surface point sets [R1]','Human + configured robot templates\nN = 4096 sampled surface points\nNo text encoder in this path',ORANGE_BG),
        ('Learn ordered surface slots [R2]','PointNet encoder + max pooling\nTemplate-residual MLP decoder\nCanonical correspondence training',ORANGE_BG),
        ('Bind slots to moving meshes [R3]','Human motion -> source slot trajectories\nRobot mesh -> FK-attached target slots\nPositions, normals and contacts',ORANGE_BG),
        ('Constrained retargeting [R4]','MuJoCo FK / geometric Jacobians\nClarabel constrained pose updates\nThis solver is not a neural layer',ORANGE_BG),
        ('Robot joint trajectory','Learning + geometric optimization\nMissing template slots trigger training',GRAY_BG)]
    ]
    for j,x in enumerate(xs):
        for i,(title,body,bg) in enumerate(content[j]):
            f.box(x,rows[i],w,hs[i],title,body,bg,names[j][1],fs=14,ts=16)
            if i<4:f.arrow(x+w/2,rows[i]+hs[i]+3,x+w/2,rows[i+1]-5,names[j][1])
    f.footer('A1-A4 / U1-U4 / R1-R4 resolve to pinned source locations in sources.md. Tensor layouts are shown explicitly per system.')
    f.save()

def alpha_detail():
    f=Fig('02_alphamotion1_layers','AlphaMotion-1: the deployed dual-stream FSQ computation',
          'megatron / L_M20k_0903 | d=256, h=8, encoder=4, decoder=4 per stream | Source: A1-A7',h=1110)
    f.box(34,122,320,148,'1  Source token construction [A1]',
          'Pose x: B x Js x 9 -> Linear(9,256)\nReplace hidden pose with mask token\nAdd Linear(1,256) contact after masking\nAdd joint descriptor E(g)',BLUE_BG,BLUE,14,16)
    f.box(390,122,340,148,'2  Encoder block x 4 [A2]',
          'z = z + HopAttn(LayerNorm(z))\nz = z + FFN(LayerNorm(z))\nFFN: 256 -> GELU(1024) -> 256\nOutput: B x Js x 256',BLUE_BG,BLUE,14,16)
    f.box(766,122,340,148,'E(g): shared joint feature [A1]',
          'Geometry 15 + DOF 19 -> Linear(34,256)\nSigLIP joint names: 768 -> 256\nGELU -> Linear(256,256); add branches\nUsed for both source and target bodies',GRAY_BG,BLUE,14,16)
    f.arrow(354,198,384,198,BLUE)
    f.label(34,312,'DUAL POSITION / ROTATION STREAMS',BLUE)
    for x,k,name in [(34,4,'Position'),(592,128,'Rotation')]:
        f.box(x,340,514,112,f'3  {name} query pooling [A3]',
              f'Learned query bank: B x {k} x 256\nCrossBlock(query, source tokens); 8 heads, FFN 256 -> 1024 -> 256\nOutput: B x {k} x 256',BLUE_BG,BLUE,14,16)
        f.box(x,484,514,131,f'4  {name} FSQ: one quantization stage [A3]',
              f'Linear(256,13): f in B x {k} x 13; a = (32 - 1) / 2 = 15.5\nu = a * tanh(f); q = (u + stopgrad(round(u) - u)) / a\nLinear(13,256): decoder memory B x {k} x 256\nNo residual quantizer stack in this checkpoint',BLUE_BG,BLUE,14,16)
        f.box(x,647,514,128,f'5  {name} target decoder x 4 [A4]',
              'q0 = E(target morphology + DOF + joint text)\nq = CrossBlock(q, quantized memory)\nq = EncBlock(q, target hop distances)\nEach CrossBlock and EncBlock has its own residual FFN',BLUE_BG,BLUE,14,16)
        f.arrow(x+257,452,x+257,478,BLUE);f.arrow(x+257,615,x+257,641,BLUE)
    f.patharrow([(560,270),(560,325),(291,325),(291,334)],BLUE)
    f.patharrow([(560,325),(849,325),(849,334)],BLUE)
    f.box(34,807,514,86,'6  Position head [A4]',
          'LayerNorm -> Linear(256,3): B x Jt x 3\nroot_pos supervises only root-slot coordinate 0: height (m)',BLUE_BG,BLUE,14,16)
    f.box(592,807,514,86,'6  Rotation + contact heads [A4]',
          'LayerNorm -> Linear(256,6): global rotation 6D\nContact module: LN -> Linear(256,1); contact BCE weight = 0',BLUE_BG,BLUE,14,16)
    f.arrow(291,775,291,801,BLUE);f.arrow(849,775,849,801,BLUE)
    f.box(34,925,514,121,'Attention operators [A2, A5]',
          'HopAttn = Wo softmax(QK^T / sqrt(32) + H[min(hop,17)]) V\nH is an 18 x 8 learned per-head hop-bias table.\nCrossBlock: MHA(LN(q), LN(kv), LN(kv)), residual;\nthen q += Linear(1024,256)(GELU(Linear(256,1024)(LN(q)))).',GRAY_BG,BLUE,12.7,15)
    f.box(592,925,514,121,'Training-only auxiliary branch [A6]',
          'JudgeHead: concatenate mean + max of both latent streams\n1024 -> Linear(1024,256) -> GELU -> Linear(256,16)\nNot part of the retargeting output path.\nFrozen text tower internals are external pretrained components.',GRAY_BG,MUTED,13.3,15)
    f.footer('Source configuration is checkpoint-specific: use_dof=true, contact_chan=true, root_head=false, vis_residual=false. Js/Jt may differ.')
    f.save()

def unimate_detail():
    f=Fig('03_unimate_layers','UniMate: Graph AdaLN flow-matching network',
          'Default shipped graph_adaln configuration | T=60, J<=60, d=512, h=8, L=10 | Source: U1-U9',h=1180)
    f.box(34,120,515,136,'1  InputLayer: root / joint / T-pose [U2]',
          'Motion x_t: B x J x 12 x 60 -> B x 60 x J x 512\nSeparate root and joint MLPs: 12 -> 512 -> 512, SiLU\nT-pose uses separate root, joint and parent MLPs.\nNon-root T-pose: concat(child,parent) 1024 -> 512 -> 512\nPrepend T-pose: B x 61 x J x 512',GREEN_BG,GREEN,14,16)
    f.box(591,120,515,136,'2  Add tokens; build condition c [U5]',
          'Token additions: joint depth + projected name embedding\nFrozen FLAN-T5-base: caption and joint names, width 768\nc = time embedding + projected pooled caption + TposPool\nTime: sinusoid 256 -> Linear(256,512) -> SiLU -> Linear\nTposPool: 4 learned queries, 4-head cross-attn, mean pool',GREEN_BG,GREEN,14,16)
    f.label(34,286,'3  REPEAT THE FOLLOWING BLOCK 10 TIMES',GREEN)
    f.patharrow([(291,256),(291,270),(19,270),(19,408),(164,408),(164,420)],GREEN)
    f.arrow(849,256,849,312,GREEN)
    f.box(34,318,1072,77,'Condition modulation [U3]',
          'SiLU(c) -> Linear(512,4608) -> 9 vectors: shift, scale, gate for spatial attention, temporal attention and FFN.',GRAY_BG,GREEN,15,16)
    f.box(34,426,328,228,'Spatial graph attention [U3, U6]',
          'B x 61 x J x 512\nRMSNorm + affine modulation\nLinear QKV: 512 -> 1536\n8 heads x 64; Q/K RMSNorm\nSpectral graph RoPE on Q/K\nsoftmax(QK^T / 8 + graph bias)V\nOutput projection 512 -> 512\nGated residual addition',GREEN_BG,GREEN,14.3,16)
    f.box(406,426,328,228,'Temporal attention [U3]',
          'Reshape: (B*J) x 61 x 512\nRMSNorm + affine modulation\nLinear QKV: 512 -> 1536\n8 heads x 64; Q/K RMSNorm\nTime RoPE on Q/K\nNoncausal attention over 61 frames\nOutput projection 512 -> 512\nGated residual addition',GREEN_BG,GREEN,14.3,16)
    f.box(778,426,328,228,'SwiGLU FFN [U3, U7]',
          'RMSNorm + affine modulation\nLinear: 512 -> 2*1365\nSplit a, b\nSiLU(a) * b\nLinear: 1365 -> 512\nGated residual addition\n1365 = int(2/3 * ff_size)\nConfigured ff_size = 2048',GREEN_BG,GREEN,14.3,16)
    f.arrow(362,540,400,540,GREEN);f.arrow(734,540,772,540,GREEN)
    for x in [198,570,942]:f.arrow(x,395,x,420,GREEN)
    f.box(34,684,515,143,'4  FinalLayer [U4]',
          'RMSNorm + c -> affine scale / shift (2*512)\nPer frame: root-query cross-attn over non-root joints, 4 heads\nRoot += MHA(RMS(root), RMS(joints), joints)\nSeparate root / joint MLP heads: 512 -> 512 -> 12, SiLU\nDiscard T-pose frame -> v_theta: B x J x 12 x 60',GREEN_BG,GREEN,14,16)
    f.box(591,684,515,143,'5  Flow objective and sampling [U8]',
          'Noise x0 ~ N(0,I), clean motion x1, t ~ Uniform(0,1)\nx_t = (1-t)*x0 + t*x1; target velocity u = x1 - x0\nMasked MSE(v_theta,u) + 0.5*L_geo + 0.1*L_smooth\nClean estimate for geometry: x_hat1 = x_t + (1-t)*v_theta\nInference: integrate learned velocity; optional CFG',GREEN_BG,GREEN,14,16)
    f.patharrow([(942,654),(942,667),(291,667),(291,678)],GREEN)
    f.arrow(549,756,585,756,GREEN)
    f.box(34,858,515,141,'Graph bias and spectral position [U6, U9]',
          'Hop / relation: two Embedding(6,128) -> Linear(128,8),\nscaled and added as per-head attention bias.\n8 Laplacian modes -> SignNet: phi(v)+phi(-v), phi 1->64->64\nConcat 8*64 -> rho 512->64->32 rotation angles per head.\nSignNet handles eigenvector sign, not arbitrary basis rotations.',GRAY_BG,GREEN,13.7,15)
    f.box(591,858,515,141,'Each gated residual has the same form [U3]',
          'a = (1 + scale) * RMSNorm(x) + shift\nx = x + gate * Operator(a)\nOperators execute in order: spatial, temporal, SwiGLU.\nRMSNorm epsilon = 1e-6; dropout = 0 in this configuration.\nCaption is an adaLN condition here, not per-block cross-attn.',GRAY_BG,GREEN,13.7,15)
    f.box(34,1028,1072,76,'12-channel motion representation [U1]',
          'RIFKE position 3 | T-pose-relative rotation 6D | local velocity 3. Root channels have dedicated semantics; padded joints and frames are masked.',GRAY_BG,MUTED,14,15)
    f.footer('This figure covers Graph AdaLN only. Full-attention / text-cross-attention variants exist and are not merged into this computation graph.')
    f.save()

def umr_detail():
    f=Fig('04_umr_layers','UMR: learned canonical correspondence, then constrained motion fitting',
          'PointNetTemplateResidualAE and the Python surface-vector retargeting path | Source: R1-R7',h=1100)
    f.label(34,124,'STAGE I  STATIC CANONICAL SURFACE LEARNING',ORANGE,16)
    f.box(34,160,232,111,'Sample / normalize [R1]',
          'Human / robot template meshes\nSurface samples B x 4096 x 3\nTranspose to B x 3 x 4096',ORANGE_BG,ORANGE,14,16)
    f.box(303,160,368,191,'PointNet encoder [R2]',
          'Conv1d(3,64,k=1) -> BN -> ReLU\nConv1d(64,128,k=1) -> BN -> ReLU\nConv1d(128,512,k=1) -> BN -> ReLU\nConv1d(512,1024,k=1)\nMax pool over 4096 points\nGlobal shape latent: B x 1024',ORANGE_BG,ORANGE,14.7,16)
    f.box(709,160,397,191,'Template-residual decoder [R2]',
          'Linear(1024,512) -> ReLU\nLinear(512,512) -> ReLU\nLinear(512,12288), zero-initialized\nReshape residual: B x 4096 x 3\nOrdered slots = canonical template + residual\nOutput slot index inherits template ordering',ORANGE_BG,ORANGE,14.7,16)
    f.arrow(266,219,297,219,ORANGE);f.arrow(671,254,703,254,ORANGE)
    f.box(34,305,232,147,'Effective Python recipe [R5]',
          '500 epochs; batch size 2\nAdamW: lr 1e-3, wd 1e-4\nHeight normalization\nGeodesic edge graph: k=32\nNo text encoder in this path',GRAY_BG,ORANGE,14,15)
    f.box(303,383,803,103,'Static correspondence objective [R2, R5]',
          'L = 1.0*Chamfer_L2 + 0.002*repulsion + 0.4*edge_smoothness + 0.0*residual_L2\nChamfer: symmetric nearest-neighbor squared distance. Repulsion: mean exp(-d_knn^2/r^2).\nEdge term: mean ||residual_i - residual_j||^2 along template geodesic-neighbor edges.',ORANGE_BG,ORANGE,14,16)
    f.arrow(908,351,908,377,ORANGE)
    f.line([(34,516),(1106,516)],LINE,.9)
    f.label(34,539,'STAGE II  MOTION RETARGETING: GEOMETRIC SOLVER, NOT A NEURAL NETWORK',ORANGE,16)
    f.box(34,580,310,123,'Bind learned slots [R3]',
          'Source mesh motion -> slot trajectories\nTarget slots attach to robot geometry\nSource / target slot positions + normals\nGround, self and object contact context',ORANGE_BG,ORANGE,14,16)
    f.box(382,580,346,123,'Linearize around pose q [R4]',
          'MuJoCo forward kinematics\nGeometric point / normal Jacobians\nStack weighted residual r and Jacobian J\nTemporal smoothness / acceleration terms',ORANGE_BG,ORANGE,14,16)
    f.box(766,580,340,123,'Constrained update [R6]',
          'P = J^T J + damping*I; b = J^T r\nmin_dq 0.5*dq^T P dq + b^T dq\nJoint / step bounds and contact inequalities\nClarabel; optional SOC step constraints',ORANGE_BG,ORANGE,14,16)
    f.arrow(344,643,376,643,ORANGE);f.arrow(728,643,760,643,ORANGE)
    f.box(766,735,340,109,'Integrate joint update [R4]',
          'mj_integratePos(q,dq); clamp limits\nConfigured per-frame iterations\nRobot joint trajectory q(t)',ORANGE_BG,ORANGE,14,16)
    f.arrow(936,703,936,729,ORANGE)
    f.patharrow([(766,793),(752,793),(752,721),(555,721),(555,708)],ORANGE)
    f.box(34,735,694,123,'Lifecycle and contact semantics [R4, R7]',
          'Compatible learned template slots can be reused; a missing/new robot template triggers correspondence training.\nThe solver preserves selected surface/contact relations with weighted residuals and inequalities.\nForward kinematics and trajectory playback do not establish dynamic feasibility.',GRAY_BG,ORANGE,13.8,15)
    f.box(34,877,1072,139,'Scope of this computation graph',
          'The learned module processes canonical point sets; it is not an end-to-end text-conditioned motion generator.\nIts ordered slots are bound to source and robot surfaces before constrained per-frame pose fitting.\nThe current browser demo runs its own correspondence training path (60 epochs); this figure uses the Python pipeline recipe.\nSelf-contact clearance is one-sided in the audited solver; ground terms are not equivalent to a full rigid-body dynamics rollout.',GRAY_BG,MUTED,14.4,16)
    f.footer('R1-R7 resolve to pinned source locations in sources.md. Training losses and inference objectives are deliberately separated.')
    f.save()

def proposed():
    f=Fig('05_alphamotion2_proposed','AlphaMotion-2: joint skeleton-surface representation',
          'PROPOSED / UNVALIDATED | Research architecture: mandatory learned mesh path; generator and bottleneck choices remain open',h=1120)
    f.box(34,124,328,142,'Skeleton and motion observations',
          'Topology, rest geometry, DOF constraints\nJoint poses / trajectories; known SE(3) goals\nAlphaMotion-1 + UniML3D motion assets\nVariable embodiment and observation masks',BLUE_BG,BLUE,14.5,16)
    f.box(406,124,328,142,'Mesh and local surface regions',
          'Points, normals, local shape and bindings\nPalm, finger pad, sole, gripper contact areas\nMesh provenance / availability masks\nSurface motion where labels are available',ORANGE_BG,ORANGE,14.5,16)
    f.box(778,124,328,142,'Shared language semantics',
          'Region functions + joint descriptions\nFrozen text encoder + learned adapters\nSigLIP / Qwen embedding candidates\nScene-level task text remains a distinct input',PURPLE_BG,PURPLE,14.5,16)
    f.box(34,317,328,130,'Skeleton encoder + L',
          'Topology- and geometry-aware joint tokens\nMotion / temporal context\nLearned embodiment-independent semantics\nDimensions and depth to be selected',BLUE_BG,BLUE,14.5,16)
    f.box(406,317,328,130,'Semantic surface encoder',
          'Local region geometry + semantic anchors\nShared point / patch correspondence network\nSurface-only representation path remains valid\nUMR-inspired canonical correspondence loss',ORANGE_BG,ORANGE,14.5,16)
    f.box(778,317,328,130,'Target embodiment descriptor',
          'New skeleton + mesh + regional descriptions\nTarget joint / patch query construction\nShared weights consume target descriptors\nNo embodiment-index output head assumed',PURPLE_BG,PURPLE,14.5,16)
    f.arrow(198,266,198,311,BLUE);f.arrow(570,266,570,311,ORANGE)
    f.patharrow([(850,266),(850,289),(660,289),(660,311)],PURPLE)
    f.patharrow([(1034,266),(1034,311)],PURPLE)
    f.label(792,238,'L: shared text port',PURPLE,11)
    f.box(34,504,700,141,'Joint skeleton-surface fusion and correspondence',
          'Bidirectional token exchange: skeletal kinematics <-> local surface geometry\nMotion semantics + contact / affordance relations across source and target bodies\nSurface tokens participate in pretraining and cross-body mapping\nSurface-only and skeleton-only paths are explicit ablations, not prerequisites for one another',GREEN_BG,GREEN,14.5,17)
    f.arrow(198,447,198,498,BLUE);f.arrow(570,447,570,498,ORANGE)
    f.patharrow([(850,447),(850,475),(700,475),(700,498)],PURPLE)
    f.box(778,504,328,141,'Condition / observation interface',
          'Text task; observed source motion\nSE(3) end-effector constraints\nKnown contacts and object geometry\nFuture tactile tokens + validity mask',PURPLE_BG,PURPLE,14.5,16,dash=True)
    f.box(34,697,700,139,'Shared motion latent and conditional generation',
          'Representation axis: continuous latent, FSQ, or a hybrid bottleneck\nGeneration axis: flow matching / diffusion candidates\nSupport source-motion mapping, task-conditioned generation, and constrained inpainting\nThese are separable design axes; the final backbone is not yet selected',GREEN_BG,GREEN,14.5,17)
    f.arrow(384,645,384,691,GREEN);f.patharrow([(942,645),(942,677),(696,677),(696,691)],PURPLE)
    f.box(778,697,328,139,'Target-conditioned joint + surface decoder',
          'Joint motion; patch motion / correspondence\nContact-aware reconstruction\nDifferentiable FK / skinning consistency\nVariable robot and end-effector geometry',BLUE_BG,BLUE,14.5,16)
    f.arrow(734,766,772,766,GREEN)
    f.patharrow([(1106,381),(1120,381),(1120,765),(1112,765)],PURPLE)
    f.box(34,884,700,145,'Joint pretraining objectives: candidate suite',
          'Motion reconstruction / flow objective; canonical surface correspondence\nSkeleton-surface agreement via FK / mesh bindings; local semantic alignment\nContact geometry and SE(3) task consistency; temporal regularity\nOnly apply mesh / contact / tactile losses when the corresponding assets and labels exist\nEmbodiment-specific dynamics and control validation remain separate downstream loops',GRAY_BG,GREEN,14.2,16)
    f.box(778,884,328,145,'Core research hypothesis',
          'Local language anchors + surface geometry may enable fixed-weight correspondence to unseen embodiments.\nThe full model adds skeleton-surface co-training; this claim remains to be tested.',PURPLE_BG,PURPLE,14.8,16)
    f.patharrow([(942,836),(942,861),(734,861),(734,878)],GREEN)
    f.footer('Proposed blocks are design hypotheses, not implemented results. No accuracy, zero-shot, contact or sim2real performance claim is implied.')
    f.save()

def cb(f,x,y,w,h,title,body,col=BLUE,bg=BLUE_BG,fs=8.5,ts=10,dash=False):
    # Compact paper boxes use smaller internal padding while preserving the
    # exact physical font sizes. The original poster helper remains unchanged.
    f.rect(x,y,w,h,bg,col,.8,dash,r=5)
    yy=y+9
    for t in f.wrap(title,w-18,ts,True):f.text(x+9,yy,t,ts,col,True);yy+=ts*1.15
    yy+=5
    for t in f.wrap(body,w-18,fs):f.text(x+9,yy,t,fs,INK);yy+=fs*1.23
    if yy>y+h-3:f.warnings.append(f'{f.name}: overflow {title}: {yy-y:.1f}/{h}')

def paper_overview():
    f=Fig('paper_01_comparison','Three implemented motion systems','A: deployed megatron. B: default Graph AdaLN. C: Python UMR pipeline. Source IDs map to sources.md.',510,500)
    xs=[22,182,342];cols=[BLUE,GREEN,ORANGE];bgs=[BLUE_BG,GREEN_BG,ORANGE_BG]
    for x,name,col in zip(xs,['A  AlphaMotion-1','B  UniMate','C  UMR'],cols):f.text(x,81,name,10.5,col,True)
    rows=[106,190,274,358];h=65
    data=[
      [('Pose + morphology [A1]','Single frame: Js x 9\nGeometry 15 + DOF 19\nSigLIP names: 768'),('Skeleton encoder [A2]','4 hop-biased blocks\nd=256; h=8\nMask + contact conditioning'),('Dual FSQ [A3]','4 position / 128 rotation slots\n256 -> 13 -> round -> 256\nSingle stage; levels=[32]x13'),('Target decoder [A4]','Each stream: 4 cross + hop blocks\nPosition 3 / rotation 6\nContact module: 1')],
      [('Noisy sequence [U1]','60 frames x J x 12\nT-pose + joint names\nFLAN-T5 caption: 768'),('Token embedding [U2]','12 -> 512; root / joint split\nT-pose prefix: 61 frames\nDepth + name features'),('Flow network [U3]','10 spatial + temporal blocks\nd=512; h=8; adaLN\nSwiGLU inner width: 1365'),('Velocity + ODE [U4]','Root aggregates joints\nSeparate 512 -> 12 heads\nNoise -> motion integration')],
      [('Canonical meshes [R1]','Human + configured robot\n4096 sampled surface points\nNo text encoder'),('Learned slots [R2]','PointNet + max pooling\nTemplate-residual MLP\nStatic correspondence loss'),('Surface binding [R3]','Attach ordered slots to meshes\nSource motion / robot FK\nPosition, normal and contact'),('Constrained fit [R4]','FK / geometric Jacobians\nClarabel pose increments\nRobot joint trajectory')]
    ]
    for j,x in enumerate(xs):
        for i,(t,b) in enumerate(data[j]):
            cb(f,x,rows[i],146,h,t,b,cols[j],bgs[j],8.15,9.1)
            if i<3:f.arrow(x+73,rows[i]+h+2,x+73,rows[i+1]-4,cols[j])
    f.text(22,439,'Frame-wise discrete mapping',8,BLUE,True)
    f.text(182,439,'Continuous sequence generation',8,GREEN,True)
    f.text(342,439,'Learning + pose optimization',8,ORANGE,True)
    f.footer('Comparison of computational mechanisms, not a performance ranking. UMR fitting is not a neural layer or a dynamics rollout.')
    f.save()

def paper_alpha():
    f=Fig('paper_02_alphamotion1','AlphaMotion-1: dual-stream FSQ','megatron / L_M20k_0903 | d=256, h=8, encoder=4, decoder=4 per stream | A1-A7',510,652)
    cb(f,22,83,224,71,'Source pose tokens [A1]','B x Js x 9 -> Linear(9,256)\nHidden pose -> learned mask token\nAdd contact Linear(1,256) after masking')
    cb(f,264,83,224,71,'Shared morphology E(g) [A1]','Geometry 15 + DOF 19 -> Linear(34,256)\nAdd SigLIP MLP: 768 -> 256 -> 256\nSource and target use the same feature module',MUTED,GRAY_BG)
    cb(f,22,181,466,73,'Encoder: 4 hop-biased residual blocks [A2]','x += HopAttn(LN(x)); x += FFN(LN(x))\nHopAttn: softmax(QK^T/sqrt(32) + H[min(hop,17)]) V, then Wo\nFFN: 256 -> GELU(1024) -> 256. Output: B x Js x 256')
    f.arrow(134,154,134,177,BLUE);f.arrow(376,154,376,177,BLUE)
    for x,k,title in [(22,4,'Position'),(264,128,'Rotation')]:
        cb(f,x,283,224,63,f'{title} pool [A3]',f'{k} learned queries -> CrossBlock(query, x)\n8-head MHA + residual FFN\nB x {k} x 256')
        cb(f,x,372,224,83,'Single-stage FSQ [A3]',f'Linear(256,13); a=15.5; u=a*tanh(f)\nq=(u+stopgrad(round(u)-u))/a\nLinear(13,256) -> B x {k} x 256\nConfigured levels: [32] x 13')
        cb(f,x,481,224,62,'Target decoder x 4 [A4]', 'q0 = E(target g)\nq = CrossBlock(q, quantized memory)\nq = HopEncBlock(q, target hop)')
        f.arrow(x+112,346,x+112,368,BLUE);f.arrow(x+112,455,x+112,477,BLUE)
    f.patharrow([(255,254),(255,269),(134,269),(134,279)],BLUE)
    f.patharrow([(255,269),(376,269),(376,279)],BLUE)
    cb(f,22,568,224,40,'Position head','LN -> Linear(256,3): B x Jt x 3',fs=8.5,ts=9.5)
    cb(f,264,568,224,40,'Rotation / contact modules','LN -> Linear(256,6); separate LN -> 1',fs=8.5,ts=9.5)
    f.arrow(134,543,134,564,BLUE);f.arrow(376,543,376,564,BLUE)
    f.footer('root_pos supervises root-slot coordinate 0 as height; no root-height head. Contact BCE weight=0. Auxiliary JudgeHead is detailed in the appendix.')
    f.save()

def paper_unimate():
    f=Fig('paper_03_unimate','UniMate: Graph AdaLN flow network','Default shipped graph_adaln configuration | T=60, J<=60, d=512, h=8, L=10 | U1-U9',510,651)
    cb(f,22,83,224,93,'Input and token embedding [U2]','x_t: B x J x 12 x 60\nRoot / joint MLPs: 12 -> 512 -> 512\nT-pose parent fusion: 1024 -> 512 -> 512\nPrepend T-pose -> B x 61 x J x 512\nAdd depth + FLAN-T5 joint-name features',GREEN,GREEN_BG)
    cb(f,264,83,224,93,'Condition c [U5]','Sinusoidal time: 256 -> 512 -> 512\n+ projected pooled FLAN-T5 caption\n+ TposPool: 4 queries, 4-head attention\nSiLU(c) -> Linear(512,4608)\n9 vectors: shift / scale / gate per branch',GREEN,GREEN_BG)
    cb(f,22,212,466,88,'Spatial graph attention [U3, U6]','RMSNorm + affine modulation -> QKV 512 -> 1536; heads 8 x 64\nQ/K RMSNorm + spectral RoPE; softmax(QK^T/8 + graph bias)V\nOutput Linear(512,512); gated residual\nGraph bias = hop and relation embeddings, projected to 8 heads',GREEN,GREEN_BG)
    cb(f,22,326,466,65,'Temporal attention [U3]','Reshape: (B*J) x 61 x 512; RMSNorm + affine modulation\n8-head QKV + Q/K RMSNorm + time RoPE; noncausal attention\nOutput Linear(512,512); gated residual',GREEN,GREEN_BG)
    cb(f,22,417,466,65,'SwiGLU FFN [U3, U7]','RMSNorm + affine modulation; Linear(512,2730), split a,b\nSiLU(a)*b -> Linear(1365,512); gated residual\nAll three branches: x += gate * Operator((1+scale)*RMS(x)+shift)',GREEN,GREEN_BG)
    f.arrow(134,176,134,208,GREEN)
    f.patharrow([(376,176),(376,192),(494,192),(494,462)],PURPLE)
    for y in [246,357,448]:f.arrow(494,y,490,y,PURPLE)
    f.text(485,185,'c',8,PURPLE,True)
    f.arrow(255,300,255,322,GREEN);f.arrow(255,391,255,413,GREEN)
    f.line([(17,212),(10,212),(10,482),(17,482)],GREEN,.8)
    f.text(230,196,'Repeat x 10 (distinct block parameters)',8,GREEN,True)
    cb(f,22,508,466,65,'Output head and flow objective [U4, U8]','RMS + affine -> root queries joints (4 heads) -> root / joint MLPs to 12\nRemove prefix -> v_theta: B x J x 12 x 60\nx_t=(1-t)x0+t*x1; target velocity=x1-x0; inference integrates v_theta',GREEN,GREEN_BG)
    f.arrow(255,482,255,504,GREEN)
    f.text(22,590,'Loss: masked velocity MSE + 0.5*L_geo + 0.1*L_smooth.',8.5,GREEN,True)
    f.footer('The three attention/FFN branches repeat as one block. Spectral SignNet and TposPool inner layers appear in the detailed layer appendix.')
    f.save()

def paper_umr():
    f=Fig('paper_04_umr','UMR: surface learning and constrained fitting','Python pipeline | N=4096, latent=1024; canonical training precedes geometric motion fitting | R1-R7',510,565)
    f.text(22,84,'I  Learn ordered canonical surface slots',10,ORANGE,True)
    cb(f,22,109,224,132,'PointNet encoder [R2]','B x 4096 x 3 -> transpose\nConv1D 3 -> 64 + BN + ReLU\nConv1D 64 -> 128 + BN + ReLU\nConv1D 128 -> 512 + BN + ReLU\nConv1D 512 -> 1024\nMax over points -> B x 1024',ORANGE,ORANGE_BG)
    cb(f,264,109,224,132,'Template-residual decoder [R2]','Linear 1024 -> 512 + ReLU\nLinear 512 -> 512 + ReLU\nLinear 512 -> 12288 (zero init)\nReshape: B x 4096 x 3\nAdd ordered canonical template\nShared slot index -> correspondence',ORANGE,ORANGE_BG)
    f.arrow(246,175,260,175,ORANGE)
    f.patharrow([(488,225),(499,225),(499,367),(95,367),(95,374)],ORANGE)
    cb(f,22,265,466,64,'Canonical shape loss [R2, R5]','Chamfer_L2 + 0.002*repulsion + 0.4*geodesic-edge smoothness\nResidual L2 weight=0. Python pipeline: 500 epochs, batch 2.\nNew / missing robot template slots invoke correspondence training.',ORANGE,ORANGE_BG)
    f.arrow(376,241,376,261,ORANGE)
    f.text(22,352,'II  Bind surface slots and fit motion',10,ORANGE,True)
    for x,t,b in [(22,'Binding [R3]','Source slot trajectories\nTarget FK-attached slots\nPositions / normals\nContact context'),(182,'Linearization [R4]','MuJoCo FK + Jacobians\nStack residual r and J\nTemporal terms\nP=J^T J + damping*I'),(342,'Solve / update [R6]','Clarabel constrained dq\nJoint + contact bounds\nIntegrate pose; repeat\nOutput trajectory q(t)')]:
        cb(f,x,378,146,94,t,b,ORANGE,ORANGE_BG,8.2,9.2)
    f.arrow(168,425,178,425,ORANGE);f.arrow(328,425,338,425,ORANGE)
    f.text(22,490,'min_dq 0.5*dq^T P dq + (J^T r)^T dq, with bounds and optional cone constraints',8.5,ORANGE,True)
    f.footer('Stage II is an optimization solver, not a neural layer. FK and trajectory playback are not dynamics validation. The browser training recipe differs.')
    f.save()

def paper_proposed():
    f=Fig('paper_05_alphamotion2_proposed','AlphaMotion-2: skeleton + semantic surface','PROPOSED / UNVALIDATED | Mesh participates in learning, correspondence and decoding.',510,580)
    cb(f,22,84,146,73,'Skeleton + motion','Topology / geometry / DOF\nJoint trajectories\nObserved motion masks',BLUE,BLUE_BG,8.5,9.5)
    cb(f,182,84,146,73,'Mesh + regions','Points / normals / bindings\nLocal patch motion\nAsset / label availability',ORANGE,ORANGE_BG,8.5,9.5)
    cb(f,342,84,146,73,'Language  [L]','Joint + region descriptions\nShared text embedding\nLocal semantic anchors',PURPLE,PURPLE_BG,8.5,9.5)
    cb(f,22,190,146,73,'Skeleton encoder + L','Morphology-aware tokens\nKinematics + motion\nShared semantic features',BLUE,BLUE_BG,8.5,9.5)
    cb(f,182,190,146,73,'Surface encoder + L','Local geometry + semantics\nShared patch correspondence\nSurface-only path is allowed',ORANGE,ORANGE_BG,8.5,9.5)
    cb(f,342,190,146,73,'Target body  [D]','New skeleton + mesh + L\nMorphology / patch queries\nShared target descriptor',PURPLE,PURPLE_BG,8.5,9.5)
    f.arrow(95,157,95,186,BLUE);f.arrow(255,157,255,186,ORANGE)
    f.text(350,167,'L: same text port',7.7,PURPLE,True)
    cb(f,22,298,306,71,'Skeleton-surface fusion + target D','Bidirectional joint / patch token exchange\nShared motion semantics and cross-body correspondence\nMask either branch for single-modality ablations',GREEN,GREEN_BG,8.5,10)
    f.arrow(95,263,95,294,BLUE);f.arrow(255,263,255,294,ORANGE)
    f.patharrow([(415,263),(415,280),(310,280),(310,294)],PURPLE)
    cb(f,342,298,146,71,'Task + future T','Text task + SE(3) goals\nKnown contacts / objects\nPartial observations',PURPLE,PURPLE_BG,8.5,9.5)
    cb(f,22,400,306,75,'Shared motion latent / generator','Representation: continuous / FSQ / hybrid candidates\nGeneration: flow matching / diffusion candidates\nMotion mapping, generation and constrained inpainting\nThe final architecture and dimensions remain open.',GREEN,GREEN_BG,8.5,10)
    cb(f,342,400,146,75,'Joint + surface decoder','Target D conditions outputs\nJoint / patch trajectories\nContact geometry\nFK / mesh-binding consistency',BLUE,BLUE_BG,8.2,9.2)
    f.arrow(175,369,175,396,GREEN);f.patharrow([(415,369),(415,385),(295,385),(295,396)],PURPLE)
    f.arrow(328,437,338,437,GREEN)
    cb(f,342,489,146,45,'T: future tactile port','Tokens + validity mask',PURPLE,PURPLE_BG,8.2,9.2,True)
    f.text(22,493,'Candidate losses: motion + surface correspondence;',8.1,GREEN)
    f.text(22,505,'semantic alignment + FK/mesh + contact + SE(3).',8.1,GREEN)
    f.text(22,517,'L, D and T denote shared named conditioning ports.',7.7,PURPLE)
    f.footer('Hypothesis: local language anchors + geometry may support surface-only unseen-body correspondence. Full-model co-training and gains remain unvalidated.')
    f.save()

if __name__=='__main__':
    overview();alpha_detail();unimate_detail();umr_detail();proposed()
    paper_overview();paper_alpha();paper_unimate();paper_umr();paper_proposed()
    (HERE/'layout_checks.json').write_text(json.dumps(ALL,indent=2))
    for a in ALL:
        subprocess.run(['pdftoppm','-png','-r','90','-singlefile',str(HERE/(a['name']+'.pdf')),str(HERE/(a['name']+'_preview'))],check=True)
        if a['name'].startswith('paper_'):
            subprocess.run(['pdftoppm','-png','-r','240','-singlefile',str(HERE/(a['name']+'.pdf')),str(HERE/a['name'])],check=True)
    print(json.dumps(ALL,indent=2))
