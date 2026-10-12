from pathlib import Path
import math,html,json,subprocess
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor

OUT=Path(__file__).resolve().parent
pdfmetrics.registerFont(TTFont('Arial','/System/Library/Fonts/Supplemental/Arial.ttf'))
pdfmetrics.registerFont(TTFont('ArialB','/System/Library/Fonts/Supplemental/Arial Bold.ttf'))
EMB='#F3DCE5';ATT='#F9E7A5';NORM='#DCEBFA';FFN='#DDEDDC';COND='#E6DDF4';GRAY='#ECEFF1'
INK='#252A2D';PURPLE='#76568E';WHITE='#FFFFFF'

class Drawing:
 def __init__(self,name,title,subtitle):
  self.name=name;self.w=510;self.h=640;self.ops=[]
  self.rect(0,0,510,640,WHITE,WHITE)
  self.text(20,16,title,15,bold=True)
  self.text(20,39,subtitle,8.4)
  self.line([(20,59),(490,59)],'#9BA4AA',.5)
 def rect(self,x,y,w,h,fill=WHITE,stroke=INK,sw=.8,dash=False,r=4):self.ops.append(('rect',x,y,w,h,fill,stroke,sw,dash,r))
 def text(self,x,y,t,size=9,color=INK,bold=False,align='left'):self.ops.append(('text',x,y,t,size,color,bold,align))
 def line(self,pts,col=INK,sw=.9,dash=False,arrow=False):self.ops.append(('line',pts,col,sw,dash,arrow))
 def arrow(self,pts,col=INK,dash=False):self.line(pts,col,1,dash,True)
 def op(self,x,y,w,h,lines,color,size=9.5):
  self.rect(x,y,w,h,color)
  if isinstance(lines,str):lines=[lines]
  step=size*1.16; yy=y+(h-step*len(lines))/2
  for s in lines:self.text(x+w/2,yy,s,size,bold=False,align='center');yy+=step
 def plus(self,x,y,r=8):
  self.ops.append(('circle',x,y,r,WHITE,INK))
  self.line([(x-3.5,y),(x+3.5,y)],INK,.9)
  self.line([(x,y-3.5),(x,y+3.5)],INK,.9)
 def dot(self,x,y):self.ops.append(('circle',x,y,1.7,INK,INK))
 def caption(self,lines):
  self.line([(20,616),(490,616)],'#9BA4AA',.5)
  for i,s in enumerate(lines):self.text(20,623+i*9,s,8)
 def save(self):
  c=canvas.Canvas(str(OUT/(self.name+'.pdf')),pagesize=(510,640));c.setTitle(self.name);c.setAuthor('ArenaLabs source-audited architecture study')
  svg=['<svg xmlns="http://www.w3.org/2000/svg" width="510pt" height="640pt" viewBox="0 0 510 640">']
  for z in self.ops:
   if z[0]=='rect':
    _,x,y,w,h,fill,stroke,sw,dash,r=z;c.setFillColor(HexColor(fill));c.setStrokeColor(HexColor(stroke));c.setLineWidth(sw);c.setDash([3,2] if dash else []);c.roundRect(x,640-y-h,w,h,r,stroke=1,fill=1)
    svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"'+(' stroke-dasharray="3 2"' if dash else '')+'/>')
   elif z[0]=='circle':
    _,x,y,r,fill,stroke=z;c.setDash([]);c.setFillColor(HexColor(fill));c.setStrokeColor(HexColor(stroke));c.setLineWidth(.9);c.circle(x,640-y,r,fill=1,stroke=1);svg.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width=".9"/>')
   elif z[0]=='text':
    _,x,y,t,size,col,bold,align=z;base=y+size*.82;c.setFont('ArialB' if bold else 'Arial',size);c.setFillColor(HexColor(col))
    if align=='center':c.drawCentredString(x,640-base,t)
    elif align=='right':c.drawRightString(x,640-base,t)
    else:c.drawString(x,640-base,t)
    anc={'left':'start','center':'middle','right':'end'}[align]
    svg.append(f'<text x="{x}" y="{base}" font-family="Arial, sans-serif" font-size="{size}" font-weight="{700 if bold else 400}" fill="{col}" text-anchor="{anc}">{html.escape(t)}</text>')
   elif z[0]=='line':
    _,pts,col,sw,dash,arr=z;c.setStrokeColor(HexColor(col));c.setLineWidth(sw);c.setDash([3,2] if dash else []);p=c.beginPath();p.moveTo(pts[0][0],640-pts[0][1])
    for x,y in pts[1:]:p.lineTo(x,640-y)
    c.drawPath(p);svg.append('<polyline points="'+' '.join(f'{x},{y}' for x,y in pts)+f'" fill="none" stroke="{col}" stroke-width="{sw}"'+(' stroke-dasharray="3 2"' if dash else '')+'/>')
    if arr:
     (ax,ay),(bx,by)=pts[-2:];th=math.atan2(by-ay,bx-ax);n=5;tr=[(bx,by),(bx-n*math.cos(th-.4),by-n*math.sin(th-.4)),(bx-n*math.cos(th+.4),by-n*math.sin(th+.4))]
     c.setFillColor(HexColor(col));p=c.beginPath();p.moveTo(tr[0][0],640-tr[0][1]);p.lineTo(tr[1][0],640-tr[1][1]);p.lineTo(tr[2][0],640-tr[2][1]);p.close();c.drawPath(p,fill=1,stroke=0);svg.append('<polygon points="'+' '.join(f'{x},{y}' for x,y in tr)+f'" fill="{col}"/>')
  c.showPage();c.save();svg.append('</svg>');(OUT/(self.name+'.svg')).write_text('\n'.join(svg))
  for dpi,suffix in [(90,'_preview'),(240,'')]:subprocess.run(['pdftoppm','-r',str(dpi),'-png','-singlefile',str(OUT/(self.name+'.pdf')),str(OUT/(self.name+suffix))],check=True)

def main_path():
 d=Drawing('unimate_01_main','UniMate: from current state to motion velocity','Default Graph AdaLN path. Read bottom to top. Dimensions shown on connecting paths.')
 # Outer generation loop, drawn first so all neural layers stay visually foreground.
 d.op(302,79,175,31,['ODE update of x_tau','outer generation loop'],GRAY,9)
 d.arrow([(150,145),(286,145),(286,94),(302,94)])
 d.arrow([(477,94),(498,94),(498,608),(250,608),(250,594),(150,594),(150,589)])
 d.text(485,578,'next state',8,align='right')
 d.text(150,116,'v_theta(x_tau, tau, c)',10,bold=True,align='center')
 d.text(150,132,'B x J x 12 x 60',8.5,align='center')
 d.op(77,154,146,28,'Concat + remove prefix',GRAY,9)
 d.arrow([(150,154),(150,145)])
 d.dot(150,145)
 # Head: root query aggregation plus two separate output MLPs.
 d.op(47,195,99,31,['Root MLP','512 -> 512 -> 12'],FFN,9)
 d.op(174,195,99,31,['Joint MLP','512 -> 512 -> 12'],FFN,9)
 d.arrow([(97,195),(97,188),(150,188),(150,182)])
 d.arrow([(224,195),(224,188),(150,188)])
 d.plus(97,244,7)
 d.op(47,263,99,29,['Root cross-attention','4 heads'],ATT,9)
 d.arrow([(97,263),(97,251)])
 d.arrow([(97,237),(97,226)])
 d.arrow([(150,327),(97,327),(97,292)])
 d.arrow([(150,327),(224,327),(224,226)])
 d.arrow([(224,305),(164,305),(164,278),(146,278)])
 d.dot(224,305)
 d.text(169,283,'K: RMS(joints)',8)
 d.text(169,294,'V: joints',8)
 d.text(61,308,'Q: RMS(root)',8)
 d.dot(97,316);d.arrow([(97,316),(31,316),(31,244),(90,244)])
 d.op(77,339,146,27,'RMS + affine(c)',NORM)
 d.arrow([(150,339),(150,327)])
 # Repeated block summary, expanded with pre-norm residuals on page two.
 d.rect(68,391,164,103,WHITE,'#78838A',.8,r=5)
 d.op(78,399,144,23,'SwiGLU residual branch',FFN,9.2)
 d.op(78,431,144,23,'Temporal attention branch',ATT,9.2)
 d.op(78,463,144,23,'Spatial attention branch',ATT,9.2)
 d.arrow([(150,431),(150,422)]);d.arrow([(150,463),(150,454)])
 d.arrow([(150,391),(150,366)])
 d.line([(60,391),(53,391),(53,494),(60,494)])
 d.text(44,436,'10x',10,bold=True,align='right')
 d.text(156,375,'B x 61 x J x 512',8.3)
 d.plus(150,513,7)
 d.arrow([(150,506),(150,494)])
 d.op(10,499,100,27,['Depth + joint-name','embeddings'],EMB,9)
 d.arrow([(110,513),(143,513)])
 d.op(77,531,146,26,'Prepend T-pose tokens',EMB,9)
 d.arrow([(150,531),(150,520)])
 d.op(77,564,146,25,['Root / joint motion MLPs','12 -> 512 -> 512'],EMB,9)
 d.arrow([(150,564),(150,557)])
 d.dot(150,594)
 d.text(150,603,'x_tau: noisy state [B,J,12,60]',8.7,bold=True,align='center')
 # T-pose branch: per-joint prefix and global pooling are explicit.
 d.op(317,531,167,31,['Root / joint T-pose MLPs','+ parent T-pose fusion'],EMB,9)
 d.text(400,574,'Target T-pose',9,bold=True,align='center')
 d.arrow([(400,573),(400,562)])
 d.dot(400,521)
 d.arrow([(400,531),(400,521),(239,521),(239,544),(223,544)])
 d.arrow([(400,521),(462,521),(462,421)])
 # Global c construction (independent from output-head feature flow).
 d.op(427,390,68,31,['TposPool','4 queries'],COND,9)
 d.op(335,390,79,31,['Mean + project','768 -> 512'],COND,9)
 d.op(335,427,79,29,['Frozen','FLAN-T5'],EMB,9)
 d.text(374,472,'Caption',9,align='center')
 d.arrow([(374,468),(374,456)]);d.arrow([(374,427),(374,421)])
 d.op(254,390,71,31,['Time MLP','256->512->512'],COND,9)
 d.op(254,432,71,29,['sin/cos','scalar -> 256'],EMB,9)
 d.text(289,475,'tau',9,bold=True,align='center')
 d.arrow([(289,473),(289,461)]);d.arrow([(289,432),(289,421)])
 d.plus(374,363,7)
 d.arrow([(287,390),(287,376),(367,376),(367,363)])
 d.arrow([(374,390),(374,370)])
 d.arrow([(462,390),(462,376),(381,376),(381,363)])
 d.text(388,353,'c: B x 512',9,PURPLE,bold=True)
 d.arrow([(374,356),(374,345),(246,345),(246,352),(223,352)],PURPLE,True)
 d.arrow([(246,345),(246,407),(232,407)],PURPLE,True)
 d.text(303,325,'global condition',8,PURPLE)
 d.caption(['10x denotes distinct network blocks; ODE updates are a separate outer loop.', 'tau is generation time, not a motion-frame index. Source map: U1-U8.'])
 d.save()

def block():
 d=Drawing('unimate_02_block','Inside one UniMate block','Pre-RMS residual branches. Read bottom to top. Repeat with distinct parameters 10 times.')
 xc=210
 # Identity bypasses first; the source token always bypasses normalization.
 for low,high in [(580,414),(405,253),(244,104)]:
  d.dot(xc,low);d.arrow([(xc,low),(83,low),(83,high),(202,high)])
 # Spatial branch.
 d.op(137,548,146,23,'RMSNorm',NORM)
 d.op(137,517,146,23,'Scale + shift',COND)
 d.op(112,460,196,37,['Spatial attention (8 heads)','QK-RMS + spectral RoPE'],ATT,9.3)
 d.op(137,435,146,20,'Gate',COND)
 d.plus(xc,414)
 d.arrow([(xc,594),(xc,571)]);d.arrow([(xc,548),(xc,540)])
 # Split modulated features into the Q,K,V input projections.
 d.line([(xc,517),(xc,510),(155,510)]);d.line([(xc,510),(265,510)])
 for x,lab in [(155,'Q'),(210,'K'),(265,'V')]:
  d.arrow([(x,510),(x,497)]);d.text(x-5,500,lab,8,align='right')
 d.arrow([(xc,460),(xc,455)]);d.arrow([(xc,435),(xc,422)])
 d.text(17,476,'joint <-> joint',9);d.text(17,489,'within one frame',8)
 d.text(319,468,'hop + relation',8);d.text(319,488,'attention bias',8)
 d.arrow([(353,482),(308,482)])
 # Temporal branch.
 d.op(137,377,146,23,'RMSNorm',NORM)
 d.op(137,349,146,23,'Scale + shift',COND)
 d.op(112,292,196,37,['Temporal attention (8 heads)','QK-RMS + time RoPE'],ATT,9.3)
 d.op(137,269,146,20,'Gate',COND)
 d.plus(xc,253)
 d.arrow([(xc,406),(xc,400)])
 d.arrow([(xc,377),(xc,372)])
 d.line([(xc,349),(xc,343),(155,343)]);d.line([(xc,343),(265,343)])
 for x,lab in [(155,'Q'),(210,'K'),(265,'V')]:d.arrow([(x,343),(x,329)]);d.text(x-5,333,lab,8,align='right')
 d.arrow([(xc,292),(xc,289)]);d.arrow([(xc,269),(xc,261)])
 d.text(17,313,'same joint',9);d.text(17,326,'across frames',8)
 d.text(319,314,'noncausal',8)
 # SwiGLU branch.
 d.op(137,216,146,23,'RMSNorm',NORM)
 d.op(137,188,146,23,'Scale + shift',COND)
 d.op(112,146,196,31,['SwiGLU FFN','512 -> 1365 -> 512'],FFN,9.3)
 d.op(137,117,146,20,'Gate',COND)
 d.plus(xc,104)
 d.arrow([(xc,245),(xc,239)])
 d.arrow([(xc,216),(xc,211)]);d.arrow([(xc,188),(xc,177)])
 d.arrow([(xc,146),(xc,137)]);d.arrow([(xc,117),(xc,112)])
 d.arrow([(xc,96),(xc,81)])
 d.text(xc,67,'Output: B x 61 x J x 512',9.5,bold=True,align='center')
 d.text(xc,596,'Input: B x 61 x J x 512',9.5,bold=True,align='center')
 d.text(24,554,'identity',8);d.text(24,388,'identity',8);d.text(24,227,'identity',8)
 # Nine per-block vectors; condition bus is distinct from black data arrows.
 d.text(420,590,'global c',10,PURPLE,bold=True,align='center')
 d.op(360,539,120,39,['SiLU -> Linear','512 -> 4608'],COND,9.5)
 d.arrow([(420,586),(420,578)],PURPLE,True)
 d.op(366,508,108,22,'Split into 9 vectors',COND,9)
 d.arrow([(420,539),(420,530)],PURPLE,True)
 d.line([(420,508),(420,125)],PURPLE,1,True)
 for y,name in [(528,'shift / scale S'),(445,'gate S'),(360,'shift / scale T'),(279,'gate T'),(199,'shift / scale F'),(127,'gate F')]:
  if y==528:
   d.arrow([(366,519),(331,519),(331,528),(283,528)],PURPLE,True)
   d.text(292,539,'shift / scale S',8,PURPLE)
  else:
   d.arrow([(420,y),(283,y)],PURPLE,True)
   d.text(333,y-13,name,8,PURPLE)
 d.text(321,93,'S: spatial   T: temporal',8,PURPLE)
 d.text(321,106,'F: feed-forward',8,PURPLE)
 d.caption(['Black bypasses add the unchanged branch input; purple dashed paths carry conditioning.', 'Q/K/V are projected from the modulated tokens. All attention is bidirectional. U3, U5-U7.'])
 d.save()

if __name__=='__main__':
 OUT.mkdir(parents=True,exist_ok=True)
 main_path();block()
