from pathlib import Path
import math,html,json,hashlib
from reportlab.pdfgen import canvas
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.colors import HexColor
O=Path(__file__).resolve().parent; A=O.parents[1]
W,H=510,640
C={'embedding':'#F3DCE5','attention':'#F9E7A5','norm':'#DCEBFA','ffn':'#DDEDDC','condition':'#E6DDF4','white':'#FFFFFF','ink':'#30343B','muted':'#606875','purple':'#79628C'}
class Fig:
 def __init__(s,name,title,sub):
  s.name=name;s.c=canvas.Canvas(str(O/(name+'.pdf')),pagesize=(W,H));s.c.setTitle(title);s.svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="510pt" height="640pt" viewBox="0 0 510 640"><title>{html.escape(title)}</title>'];s.text(20,614,title,17,True);s.text(20,596,sub,9.3,color=C['muted']);s.labels=[]
 def text(s,x,y,t,size=10,bold=False,color=None,anchor='start'):
  color=color or C['ink'];font='Helvetica-Bold' if bold else 'Helvetica';width=stringWidth(t,font,size);xx=x-(width/2 if anchor=='middle' else width if anchor=='end' else 0);assert xx>=0 and xx+width<=510,(t,xx,width);s.c.setFillColor(HexColor(color));s.c.setFont(font,size);s.c.drawString(xx,y,t);s.svg.append(f'<text x="{x}" y="{H-y}" font-family="Arial,Helvetica,sans-serif" font-size="{size}" font-weight="{700 if bold else 400}" text-anchor="{anchor}" fill="{color}">{html.escape(t)}</text>')
 def label(s,x,y,t,size=9,anchor='middle'):
  tw=stringWidth(t,'Helvetica',size);left=x-tw/2 if anchor=='middle' else x;s.c.setFillColor(HexColor('#FFFFFF'));s.c.rect(left-2,y-2,tw+4,size+4,stroke=0,fill=1);s.svg.append(f'<rect x="{left-2}" y="{H-y-size-2}" width="{tw+4}" height="{size+4}" fill="white"/>');s.text(x,y,t,size,anchor=anchor)
 def box(s,x,y,w,h,title,sub=None,kind='white',size=10.3):
  s.c.setFillColor(HexColor(C[kind]));s.c.setStrokeColor(HexColor('#69717B'));s.c.setLineWidth(.75);s.c.rect(x,y,w,h,stroke=1,fill=1);s.svg.append(f'<rect x="{x}" y="{H-y-h}" width="{w}" height="{h}" fill="{C[kind]}" stroke="#69717B" stroke-width="0.75"/>');cy=y+h/2
  if sub:s.text(x+w/2,cy+3,title,size,True,anchor='middle');s.text(x+w/2,cy-10,sub,9.1,anchor='middle')
  else:s.text(x+w/2,cy-3,title,size,True,anchor='middle')
 def line(s,pts,color=None,dash=False,arrow=False,lw=.9):
  color=color or C['ink'];s.c.setStrokeColor(HexColor(color));s.c.setFillColor(HexColor(color));s.c.setLineWidth(lw);s.c.setDash(3,2) if dash else s.c.setDash();p=s.c.beginPath();p.moveTo(*pts[0]);
  for q in pts[1:]:p.lineTo(*q)
  s.c.drawPath(p);s.c.setDash();s.svg.append('<polyline points="'+' '.join(f'{x},{H-y}' for x,y in pts)+f'" fill="none" stroke="{color}" stroke-width="{lw}"'+(' stroke-dasharray="3 2"' if dash else '')+'/>')
  if arrow:
   x,y=pts[-1];px,py=pts[-2];ang=math.atan2(y-py,x-px);t=[(x,y),(x-4.5*math.cos(ang)+2*math.sin(ang),y-4.5*math.sin(ang)-2*math.cos(ang)),(x-4.5*math.cos(ang)-2*math.sin(ang),y-4.5*math.sin(ang)+2*math.cos(ang))];p=s.c.beginPath();p.moveTo(*t[0]);p.lineTo(*t[1]);p.lineTo(*t[2]);p.close();s.c.drawPath(p,fill=1,stroke=0);s.svg.append('<polygon points="'+' '.join(f'{a},{H-b}' for a,b in t)+f'" fill="{color}"/>')
 def arrow(s,pts,**kw):s.line(pts,arrow=True,**kw)
 def add(s,x,y):
  s.c.setFillColor(HexColor('#FFFFFF'));s.c.setStrokeColor(HexColor('#202020'));s.c.circle(x,y,7,stroke=1,fill=1);s.svg.append(f'<circle cx="{x}" cy="{H-y}" r="7" fill="white" stroke="#202020"/>');s.text(x,y-3.2,'+',12,anchor='middle')
 def bracket(s,x,y1,y2,label,side='left'):
  d=5 if side=='left' else -5;s.line([(x+d,y1),(x,y1),(x,y2),(x+d,y2)],lw=1.1);s.text(x,y2+7,label,10.5,True,anchor='middle')
 def save(s):s.c.showPage();s.c.save();s.svg.append('</svg>');(O/(s.name+'.svg')).write_text('\n'.join(s.svg))

f=Fig('alphamotion1_path','AlphaMotion-1 | shared single-frame translator','One weight set for all bodies; target descriptions create a variable number of queries.')
f.text(128,584,'SOURCE ENCODER',10,True,anchor='middle');f.text(383,584,'TARGET DECODER',10,True,anchor='middle')
# bottom inputs and exact mask/contact/morphology order
f.box(25,65,90,28,'Pose 9D','global rot + rel pos',kind='white',size=10)
f.box(129,65,104,28,'Source body','geometry + names',kind='condition',size=10)
f.box(25,112,90,28,'Linear 9 -> 256',kind='embedding',size=9.5)
f.box(129,112,104,28,'JointFeature','34D + text 768',kind='embedding',size=10)
f.box(25,157,90,28,'Mask replace','hide pose only',kind='embedding',size=10)
f.arrow([(70,93),(70,112)]);f.arrow([(181,93),(181,112)],color=C['purple'],dash=True);f.arrow([(70,140),(70,157)])
f.add(70,217);f.arrow([(70,185),(70,210)])
f.box(129,197,104,32,'Contact embed','1 -> 256; if valid',kind='embedding',size=10)
f.text(181,177,'optional contact',9.1,anchor='middle');f.arrow([(181,184),(181,197)],color=C['purple'],dash=True)
f.arrow([(129,213),(77,217)],color=C['purple'],dash=True)
f.add(70,261);f.arrow([(70,224),(70,254)])
f.arrow([(233,126),(242,126),(242,261),(77,261)],color=C['purple'],dash=True)
f.box(25,297,208,43,'Hop-biased EncBlock','mix source joints; pre-norm',kind='attention',size=11)
f.arrow([(70,268),(70,280),(129,280),(129,297)]);f.bracket(14,297,340,'4x')
f.text(176,274,'J_s x 256',9,anchor='middle')
# independent query pooling and FSQ
for x,k,title in [(25,4,'Position stream'),(142,128,'Rotation stream')]:
 cx=x+45.5
 f.box(x,379,91,37,'CrossBlock',f'Q: learned {k}',kind='attention',size=10.3)
 f.arrow([(129,340),(129,359),(cx,359),(cx,379)])
 f.box(x,450,91,43,'FSQ', 'project-round-project',kind='embedding',size=11)
 f.arrow([(cx,416),(cx,450)]);f.label(cx,432,'256 -> 13 -> 256',8.8);f.text(cx,562,title,9.5,True,anchor='middle')
f.text(129,365,'K,V: source features',8.5,anchor='middle')
# target descriptor and two dynamically-sized query streams
f.box(280,65,202,32,'Target body descriptors','geometry / DOF + frozen joint-name semantics',kind='condition',size=11)
f.box(280,126,202,36,'Same JointFeature weights','Linear(34) + MLP(SigLIP 768)',kind='embedding',size=11)
f.arrow([(381,97),(381,126)],color=C['purple'],dash=True)
f.arrow([(381,162),(381,190),(320,190),(320,335)],color=C['purple'],dash=True)
f.arrow([(381,190),(442,190),(442,335)],color=C['purple'],dash=True)
f.label(381,175,'Q initial: J_t x 256',9.4)

for x in [272,394]:
 cx=x+48
 f.box(x,335,96,34,'CrossBlock','read latent memory',kind='attention',size=10.3)
 f.box(x,385,96,34,'Hop EncBlock','mix target joints',kind='attention',size=10.3)
 f.arrow([(cx,369),(cx,385)]);f.bracket(x-8,335,419,'4x')
# separate memory paths (no body-specific weights)
f.arrow([(70.5,493),(70.5,510),(252,510),(252,352),(272,352)])
f.text(120,515,'K,V: z_p [4,256]',8.7,anchor='middle')
f.arrow([(187.5,493),(187.5,544),(375,544),(375,352),(394,352)])
f.text(218,549,'K,V: z_r [128,256]',8.7,anchor='middle')
# heads
f.box(272,460,96,43,'LN -> Linear 3','position head',kind='ffn',size=10.5)
f.box(394,460,96,43,'2 x LN -> Linear','rot 6 | contact 1',kind='ffn',size=9.1)
f.arrow([(320,419),(320,460)]);f.arrow([(442,419),(442,460)])
f.box(272,554,96,25,'Position 3D',kind='white',size=10.5);f.box(394,554,96,25,'Rot6 + contact',kind='white',size=10.3)
f.arrow([(320,503),(320,554)]);f.arrow([(442,503),(442,554)])
f.text(20,40,'D = 256; 8 heads. Separate position/rotation branches; shared across bodies.',9.2)
f.text(20,25,'root_pos supervises root[0] height; no root head. Contact BCE weight = 0.',9.1)
f.text(20,11,'FSQ: one stage, levels = [32] x 13. Training-only Judge omitted. Blocks: next page.',8.5)
f.save()

f=Fig('alphamotion1_blocks','AlphaMotion-1 | pre-norm Transformer blocks','Bottom-to-top flow. Every black bypass is a residual; no causal attention mask.')
f.text(128,571,'HOP EncBlock',12,True,anchor='middle');f.text(382,571,'CrossBlock',12,True,anchor='middle')
f.text(128,555,'source encoder / target graph mixing',8.7,anchor='middle');f.text(382,555,'latent pooling / memory retrieval',8.7,anchor='middle')
# the two reusable operator ladders
for cx in [128,382]:
 qcx=cx if cx==128 else 337
 qw=122 if cx==128 else 94
 f.text(qcx,77,'x' if cx==128 else 'q',11,True,anchor='middle')
 f.box(qcx-qw/2,109,qw,28,'LayerNorm','normalize input',kind='norm',size=10.2)
 f.box(qcx-qw/2,158,qw,28,'Q / K / V Linear' if cx==128 else 'Q Linear','project Q, K and V' if cx==128 else 'project query',kind='embedding',size=9.8)
 f.box(cx-76,209,152,35,'Hop-biased attention' if cx==128 else 'Cross-attention','softmax(QK^T / sqrt(32) + b)V' if cx==128 else 'softmax(QK^T / sqrt(32))V',kind='attention',size=10.4)
 f.box(cx-61,260,122,28,'Concat -> Linear','merge 8 heads',kind='attention',size=10.2)
 f.add(cx,311)
 f.box(cx-61,338,122,26,'LayerNorm',kind='norm',size=10.3)
 f.box(cx-61,383,122,26,'Linear 256 -> 1024',kind='ffn',size=10)
 f.box(cx-61,421,122,24,'GELU',kind='ffn',size=10.5)
 f.box(cx-61,457,122,26,'Linear 1024 -> 256',kind='ffn',size=10)
 f.add(cx,507);f.text(cx,537,'output [*,256]',10,True,anchor='middle')
 for a,b in [(244,260),(288,304),(318,338),(364,383),(409,421),(445,457),(483,500),(514,527)]:f.arrow([(cx,a),(cx,b)])
 for a,b in [(82,109),(137,158),(186,209)]:f.arrow([(qcx,a),(qcx,b)])
 # original input and intermediate residual paths
 rx=cx-101
 f.arrow([(qcx,92),(rx,92),(rx,311),(cx-7,311)],color='#111111',lw=1.1)
 f.arrow([(cx,325),(rx,325),(rx,507),(cx-7,507)],color='#111111',lw=1.1)
f.text(30,225,'hop',9,color=C['purple']);f.arrow([(49,228),(52,228)],color=C['purple'],dash=True)
f.label(128,194,'3 projections of LN(x)',8.5)
# separate KV norm/projection in CrossBlock, shown as an explicit compact side branch
# overwrite right Q projection with a narrow query path and memory pathway for fidelity
# the cross attention memory pipeline is an inset to the right, routed into attention
f.box(423,94,74,27,'memory kv',kind='condition',size=9.4)
f.box(423,132,74,26,'LayerNorm',kind='norm',size=9.1)
f.box(423,171,74,26,'K / V Linear',kind='embedding',size=9.1)
f.arrow([(460,121),(460,132)],color=C['purple'],dash=True);f.arrow([(460,158),(460,171)],color=C['purple'],dash=True)
f.arrow([(460,197),(460,202),(430,202),(430,209)],color=C['purple'],dash=True)
f.label(382,194,'Q from q; K,V from kv',8.5)
f.text(20,49,'Each block: attention residual, then FFN residual. Separate LayerNorm parameters.',9.1)
f.text(20,34,'Pool: q = learned slots, kv = encoder output. Decoder: q = target joints, kv = FSQ.',8.8)
f.text(20,19,'EncBlock FFN and CrossBlock FFN are both 256 -> 1024 -> 256 with GELU.',9.1)
f.save()
print('created path and block pages')
