import {REGIONS,REGION_HEX,REGION_ZH,heatCSS} from './palette';
export function ContactExplainer({compact=false}){
 return <details className="at-explain" open={!compact}><summary>接触是什么？（看图前先读这一段）</summary>
  <p><b>现在数据里存的是：</b>①逐点<b>二值接触</b>——机器人 4096 个表面点里，哪些点离地面 / 物体在 [-0.5, 1] mm 之内（文件字段 contact_idx，地面和物体分开）；②<b>2 cm 内的距离场</b>——离接触面不到 2 cm 的点和它们的距离（near_idx + near_dist_q，按 0.1 mm 量化）；③每个点的<b>13 区身体部位标签</b>，以及按区汇总的接触 / 打滑 / 穿透 / 离地间隙。</p>
  <p><b>它不是力，也不是压力。</b>"距离场"热力图看起来像压力图，但颜色只表示"离得多近"，不是"压得多重"；"法向箭头"是表面法线方向的几何示意（长度随距离变短），<b>是几何代理，不是真实接触力</b>。</p>
  <p>要得到真正的压力分布，需要把动作放进物理仿真（MuJoCo 等）里算接触力，再分到各接触点上——这是以后的工作，当前数据和本页面都没有做。</p>
 </details>;
}
export function ModeLegend({mode,hasObject=true,human=false}){
 if(mode==='region')return <div className="at-legend">{REGIONS.map((r,i)=><span key={r}><i style={{background:REGION_HEX[i]}}/>{REGION_ZH[r]}</span>)}<em>人体和机器人用同一套 13 区调色板</em></div>;
 if(mode==='contact')return <div className="at-legend"><span><i style={{background:'#1fb36b'}}/>地面接触点</span>{hasObject?<span><i style={{background:'#f2922a'}}/>物体接触点</span>:null}<span><i style={{background:'#ff2fb4'}}/>人体穿地 &gt; 0.5 mm（界面现算）</span><span><i style={{background:'#c3ccd3'}}/>其它点</span><em>机器人：文件 contact_idx（[-0.5,1] mm）；人体：界面按点高现算（[-0.5,1] mm 为接触）</em></div>;
 const stops=[0,2,5,10,15,20];
 return <div className="at-legend"><span className="at-heat">{stops.map(s=><b key={s} style={{background:heatCSS(s)}}>{s}</b>)}<small>mm（越近越亮）</small></span><span><i style={{background:'#ff2fb4'}}/>穿透（≤ −0.5 mm，量化下限）</span><span><i style={{background:'#3a4852'}}/>&gt; 2 cm</span>{mode==='normals'?<span className="at-proxy">箭头 = 表面法线 × 距离衰减长度，起点颜色绿=地面 / 橙=物体 —— 几何代理，不是真实接触力</span>:null}<em>{human?'人体距离为界面现算（点高）':'机器人距离 = near_dist_q × 0.1 − 0.5 mm（文件）；人体为界面现算点高'}</em></div>;
}
