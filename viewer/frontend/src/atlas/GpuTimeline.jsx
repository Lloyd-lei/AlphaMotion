import {memo,useEffect,useRef,useState} from 'react';
import {values} from '../api';
import {REGIONS,REGION_HEX,REGION_ZH,heatCSS} from './palette';
const RH=5,FH=13,LABEL=190;
function groupsFor(scene){
 const has=k=>!!values(scene,'timeline_'+k);
 const g=[['ground_contact','机器人 · 地面接触','#1fa365','bool'],['human_ground_contact','人体 · 地面接触（1 cm 尺）','#8fd3ae','bool'],['ground_slip','机器人 · 打滑','#9b5bd0','bool'],['pen','穿透 > 0.5 mm','#e0414f','pen'],['ground_clearance_mm','机器人 · 离地间隙 0–20 mm','','heat']];
 if(has('object_contact'))g.push(['object_contact','机器人 · 物体接触','#ef8a22','bool']);
 if(has('human_object_contact'))g.push(['human_object_contact','人体 · 物体接触','#f6c58a','bool']);
 return g;
}
export const GpuTimeline=memo(function GpuTimeline({scene,total,fps,frame,onSeek,flagBits}){
 const host=useRef(null),cv=useRef(null),[w,setW]=useState(900),[hover,setHover]=useState(null);
 useEffect(()=>{const ob=new ResizeObserver(e=>setW(Math.floor(e[0].contentRect.width)));if(host.current)ob.observe(host.current);return()=>ob.disconnect();},[]);
 const groups=scene?groupsFor(scene):[],R=13,gH=R*RH+6,bits=flagBits||[],rows=bits.length+1,H=groups.length*gH+8+rows*FH+4;
 useEffect(()=>{const c=cv.current;if(!c||!scene)return;const dpr=Math.min(devicePixelRatio,2);c.width=w*dpr;c.height=H*dpr;const x=c.getContext('2d');x.scale(dpr,dpr);x.clearRect(0,0,w,H);x.font='10px system-ui';x.textBaseline='middle';
  const W=w-LABEL;const cols=Math.max(1,Math.floor(W));
  const span=px=>{const a=Math.floor(px/cols*total),b=Math.max(a+1,Math.ceil((px+1)/cols*total));return [a,Math.min(b,total)];};
  const pg=values(scene,'timeline_ground_pen_mm'),po=values(scene,'timeline_object_pen_mm');
  groups.forEach(([k,title,color,kind],gi)=>{const y0=gi*gH;x.fillStyle='#3f5562';x.fillText(title,4,y0+gH/2-1);for(let r=0;r<R;r++){x.fillStyle=REGION_HEX[r];x.fillRect(LABEL-8,y0+r*RH,5,RH-1);}
   x.fillStyle='#f3f6f8';x.fillRect(LABEL,y0,W,R*RH);
   const data=kind==='pen'?null:values(scene,'timeline_'+k);
   for(let px=0;px<cols;px++){const [a,b]=span(px);for(let r=0;r<R;r++){let hit=false,best=Infinity;for(let f=a;f<b;f++){const i=f*R+r;if(kind==='bool'){if(data[i]){hit=true;break;}}else if(kind==='pen'){if((pg&&pg[i]>.5)||(po&&po[i]>.5)){hit=true;break;}}else{const v=data[i];if(v<best)best=v;}}
     if(kind==='heat'){if(best<=20){x.fillStyle=heatCSS(Math.max(best,0));x.fillRect(LABEL+px,y0+r*RH,1,RH-1);}else if(best<-.5){x.fillStyle='#ff2fb4';x.fillRect(LABEL+px,y0+r*RH,1,RH-1);}}
     else if(hit){x.fillStyle=color;x.fillRect(LABEL+px,y0+r*RH,1,RH-1);}}}
  });
  const fy=groups.length*gH+8,fl=values(scene,'timeline_flags'),rj=values(scene,'timeline_root_jump_cm');
  [...bits,{bit:0,zh:'根跳变 > 10 cm/帧（界面现算）'}].forEach((b,bi)=>{const y=fy+bi*FH;x.fillStyle='#3f5562';x.font='9.5px system-ui';x.fillText((b.bit?`flag ${b.bit} · `:'')+b.zh.replace(/（.*?）/,'').slice(0,18),4,y+FH/2);x.fillStyle=bi%2?'#f7f9fa':'#eef3f5';x.fillRect(LABEL,y,W,FH-1);
   x.fillStyle=b.bit?'#c2453f':'#7a3fb0';for(let px=0;px<cols;px++){const [a,c]=span(px);for(let f=a;f<c;f++){if(b.bit?(fl[f]&b.bit):(rj&&rj[f]>10)){x.fillRect(LABEL+px,y+1,1,FH-3);break;}}}});
 },[scene,w,H,total,groups.length,bits.length]);
 const frac=total>1?frame/(total-1):0;
 const locate=e=>{const r=e.currentTarget.getBoundingClientRect(),px=e.clientX-r.left,py=e.clientY-r.top;if(px<LABEL)return null;const f=Math.max(0,Math.min(total-1,Math.round((px-LABEL)/Math.max(w-LABEL,1)*(total-1))));const gi=Math.floor(py/gH);let what='';if(gi<groups.length){const rr=Math.floor((py-gi*gH)/RH);if(rr<13)what=groups[gi][1]+' · '+REGION_ZH[REGIONS[rr]];}else{const bi=Math.floor((py-groups.length*gH-8)/FH);const b=(flagBits||[])[bi];what=b?`flag ${b.bit} ${b.zh}`:bi===(flagBits||[]).length?'根跳变 > 10 cm/帧':'';}return {f,what};};
 return <section className="at-timeline" ref={host}><div className="at-section-head"><h3>逐帧时间轴</h3><span>13 区 × 每组一行一区（左侧色条 = 区域颜色）· 点击跳帧 · 悬停看区域</span></div>
  <div className="at-tl-wrap" style={{height:H}} onPointerDown={e=>{const l=locate(e);if(l)onSeek(l.f);}} onPointerMove={e=>setHover(locate(e))} onPointerLeave={()=>setHover(null)}><canvas ref={cv} style={{width:w,height:H}}/><div className="at-playhead" style={{left:LABEL+frac*(w-LABEL)}}/></div>
  <div className="at-tl-foot"><span>0 s</span><span>{hover?`${hover.what} · 帧 ${hover.f}（${(hover.f/fps).toFixed(2)} s）`:'—'}</span><span>{(total/Math.max(fps,1)).toFixed(2)} s</span></div></section>;
});
