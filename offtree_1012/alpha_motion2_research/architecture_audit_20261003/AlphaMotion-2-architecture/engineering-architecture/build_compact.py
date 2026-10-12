from pathlib import Path
import math, html, json
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from reportlab.pdfbase.pdfmetrics import stringWidth
O=Path(__file__).resolve().parent
W,H=510,610
C={'ink':'#172536','muted':'#536476','line':'#8292a4','data':'#e9f2fb','compute':'#e6f2ef','artifact':'#faf1df','review':'#f1ecfa','proposed':'#fff1df','light':'#f5f7fa'}
class Fig:
 def __init__(s,name,title,subtitle):
  s.name=name;s.c=canvas.Canvas(str(O/(name+'.pdf')),pagesize=(W,H));s.c.setTitle(title);s.nodes={};s.svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="510pt" height="610pt" viewBox="0 0 510 610"><title>{html.escape(title)}</title>'];s.rect(0,0,W,H,'#ffffff',0);s.text(20,32,title,19,True);s.text(20,52,subtitle,10)
 def text(s,x,y,t,z=10,b=False,color=None):
  color=color or C['ink'];font='Helvetica-Bold' if b else 'Helvetica';assert x+stringWidth(t,font,z)<W-12,(s.name,t);s.c.setFillColor(HexColor(color));s.c.setFont(font,z);s.c.drawString(x,H-y,t);s.svg.append(f'<text x="{x}" y="{y}" font-family="Arial,Helvetica,sans-serif" font-size="{z}" font-weight="{700 if b else 400}" fill="{color}">{html.escape(t)}</text>')
 def rect(s,x,y,w,h,color,r=7,stroke=None):
  s.c.setFillColor(HexColor(color));s.c.setStrokeColor(HexColor(stroke or color));s.c.setLineWidth(.6);s.c.roundRect(x,H-y-h,w,h,r,stroke=bool(stroke),fill=1);s.svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{color}" stroke="{stroke or color}"/>')
 def stage(s,y,title):s.rect(12,y-15,486,145,C['light']);s.text(22,y,title,10.5,True,C['muted'])
 def node(s,i,x,y,title,detail,kind='compute',w=216):
  s.nodes[i]=(x,y,w,46);s.rect(x,y,w,46,C[kind],7,'#c5d2dd');s.text(x+10,y+18,title,12,True);s.text(x+10,y+34,detail,9.5)
 def point(s,i,k):
  x,y,w,h=s.nodes[i];return {'L':(x,y+h/2),'R':(x+w,y+h/2),'T':(x+w/2,y),'B':(x+w/2,y+h)}[k]
 def arr(s,a,b,sa='R',sb='L',via=(),dash=False):
  p=[s.point(a,sa),*via,s.point(b,sb)];s.c.setStrokeColor(HexColor(C['line']));s.c.setFillColor(HexColor(C['line']));s.c.setLineWidth(1);s.c.setDash(3,2) if dash else s.c.setDash();q=s.c.beginPath();q.moveTo(p[0][0],H-p[0][1]);
  for x,y in p[1:]:q.lineTo(x,H-y)
  s.c.drawPath(q);s.svg.append('<polyline points="'+' '.join(f'{x},{y}' for x,y in p)+'" fill="none" stroke="'+C['line']+'" stroke-width="1"'+(' stroke-dasharray="3 2"' if dash else '')+'/>');x,y=p[-1];xx,yy=p[-2];ang=math.atan2(y-yy,x-xx);t=[(x,y),(x-5*math.cos(ang)+2.5*math.sin(ang),y-5*math.sin(ang)-2.5*math.cos(ang)),(x-5*math.cos(ang)-2.5*math.sin(ang),y-5*math.sin(ang)+2.5*math.cos(ang))];q=s.c.beginPath();q.moveTo(t[0][0],H-t[0][1]);
  for xx,yy in t[1:]:q.lineTo(xx,H-yy)
  q.close();s.c.drawPath(q,fill=1,stroke=0);s.c.setDash();s.svg.append('<polygon points="'+' '.join(f'{x},{y}' for x,y in t)+'" fill="'+C['line']+'"/>')
 def snake(s,ids):
  a,b,c,d=ids;s.arr(a,b);s.arr(b,c,'B','T');s.arr(c,d,'L','R')
 def foot(s,lines):
  s.rect(12,547,486,50,C['light']);
  for j,t in enumerate(lines):s.text(22,565+j*14,t,9.5,color=C['muted'])
 def save(s):s.c.showPage();s.c.save();s.svg.append('</svg>');(O/(s.name+'.svg')).write_text('\n'.join(s.svg));return s.name
p=[(22,96),(272,96),(272,160),(22,160),(22,255),(272,255),(272,319),(22,319),(22,414),(272,414),(272,478),(22,478)]
f=Fig('unimate_engineering_doc','UniMate: engineering architecture','Existing pipeline: corpus preparation, training and sampling')
for y,t in [(80,'1  Prepare the corpus'),(239,'2  Train and save a run'),(398,'3  Generate and inspect')]:f.stage(y,t)
ns=[('Raw rigged assets','FBX / GLB + animation','data'),('Export + render','Blender workers; motion + views','compute'),('Annotate + extract','Captions; trim; activity / jump QC','compute'),('Feature corpus','Clip NPZ + topology + captions','artifact'),('Text feature cache','Joint vectors + caption tokens','data'),('Dataset + collate','Padded motion / text; valid masks','compute'),('Train + model factory','Flow or DDPM; Accelerate; EMA','compute'),('Saved run bundle','Weights + optimizer + config / stats','artifact'),('Generation conditions','Text + T-pose; optional known frames','data'),('Sampling entry','CFG + ODE / diffusion sampler','compute'),('Reconstruct + inspect','RIC / FK; motion NPY + video','review'),('Optional mesh output','Source rig + LBS; FBX / GLB','artifact')]
for i,((x,y),(t,d,k)) in enumerate(zip(p,ns),1):f.node(i,x,y,t,d,k)
f.snake((1,2,3,4));f.snake((5,6,7,8));f.arr(4,5,'L','L',[(8,183),(8,278)]);f.arr(4,6,'R','T',[(254,183),(254,229),(380,229)]);f.text(271,221,'clips + topology',9.5,color=C['muted']);f.arr(8,10,'B','T',[(130,388),(380,388)]);f.text(142,382,'weights / config / stats',9.5,color=C['muted']);f.arr(9,10);f.arr(10,11,'B','T');f.arr(11,12,'L','R')
f.foot(['Reuse: asset caches, text features, padded masks, normalization and LBS.','Sample previews are not physics validation; source details are in the audit.']);f.save()
f=Fig('umr_engineering_doc','UMR: engineering architecture','Existing pipeline: static correspondence, retargeting and inspection')
for y,t in [(80,'1  Learn canonical surface correspondence'),(239,'2  Retarget each motion'),(398,'3  Inspect results; alternate browser runtime')]:f.stage(y,t)
ns=[('Config + geometry','Source template + robot MJCF / mesh','data'),('Build surface dataset','Canonical static point-cloud NPZ','compute'),('Train correspondence','AE training; pipeline default 500 epochs','compute'),('Correspondence artifacts','Checkpoint + ordered slots NPZ','artifact'),('Source motion','Sequence, frame range and fps','data'),('Bind + update surfaces','Material points; FK and Jacobians','compute'),('Constrained solver','Clarabel; limits, contact, smoothness','compute'),('Retarget result','Final qpos NPZ + solver logs','artifact')]
for i,((x,y),(t,d,k)) in enumerate(zip(p,ns),1):f.node(i,x,y,t,d,k)
f.snake((1,2,3,4));f.arr(5,6);f.arr(6,7,'B','T');f.arr(7,8,'L','R');f.arr(4,6,'B','T',[(130,229),(380,229)]);f.text(155,221,'slots + geometry',9.5,color=C['muted']);f.node(9,22,414,'GLFW / Viser inspection','Surface / contact overlays; video','review',466);f.arr(8,9,'L','L',[(8,342),(8,437)]);f.node(10,22,478,'Browser Studio: separate execution path','WASM correspondence (60 epochs) -> Jacobians + Clarabel','review',466)
f.foot(['A new or incompatible template can trigger correspondence training.','Geometric FK is not a dynamics rollout; Python and browser paths differ.']);f.save()
f=Fig('alphamotion2_cleaning_proposed_doc','AlphaMotion-2: cleaning pipeline','PROPOSED architecture; reusable pieces exist, orchestration remains to build')
for y,t in [(80,'1  Ingest, validate and schedule'),(239,'2  Derive and measure'),(398,'3  Decide, publish and inspect')]:f.stage(y,t)
ns=[('Asset intake + lineage','Raw hash; clip / body / asset identity','data'),('Canonical adapters','Units, frames, time and valid masks','proposed'),('Fast deterministic QC','CPU batches; versioned reason codes','proposed'),('Budgeted work queue','Idempotent jobs; bounded retry / quotas','proposed'),('Surface + region cache','FK / LBS bindings + region text source','proposed'),('Model / retarget workers','Pinned baseline or new shared model','proposed'),('Evidence metrics','Geometry / offline dynamics / sim / real','proposed'),('Measured artifact bundle','Values, validity, units and lineage','artifact'),('Policy + review gate','Versioned accept / review / reject','review'),('Versioned data release','Atomic manifest; grouped holdout splits','artifact'),('Training + review readers','Golden clips; bad-frame replay; traces','data'),('Quarantine + retry','Invalid input or failed stages; reasons','review')]
for i,((x,y),(t,d,k)) in enumerate(zip(p,ns),1):f.node(i,x,y,t,d,k)
f.snake((1,2,3,4));f.arr(4,5,'L','L',[(8,183),(8,278)]);f.snake((5,6,7,8));f.arr(8,9,'L','L',[(8,342),(8,437)]);f.arr(9,10);f.arr(10,11,'B','T');f.arr(9,12,'B','T',dash=True);f.text(140,472,'rejected / failed',9.5,color=C['muted'])
f.foot(['Evidence stays separate: geometry | offline dynamics | simulation | real robot.','Readiness needs regression, recovery and throughput proof beyond basic tests.']);f.save()
(O/'compact-qa.json').write_text(json.dumps({'width_pt':W,'height_pt':H,'minimum_font_pt':9.5,'at_468pt_width_minimum_font_pt':9.5*468/W,'figures':['unimate_engineering_doc','umr_engineering_doc','alphamotion2_cleaning_proposed_doc'],'status':'generated; pending rendered visual QA'},indent=2))
print('Created 3 document editions; body 9.5 pt, final 468pt width = 8.72 pt.')
