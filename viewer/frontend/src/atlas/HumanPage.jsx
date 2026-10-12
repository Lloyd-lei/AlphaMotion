import {useEffect,useRef,useState} from 'react';
import {binary,json} from '../api';
import {Playback} from '../Playback';
import {PointCloudView} from './PointCloudView';
import {useChunkStream,usePlayClock} from './useChunkStream';
import {ContactExplainer,ModeLegend} from './Explain';
import {MODES,fmtH} from './palette';
const SRC=[['','全部来源'],['hiphi','HiPHI'],['omnicontact','OmniContact'],['grail','GRAIL（只列清单）']];
const VAR={cpu:'CPU 转换',gpu:'GPU 转换 90 Hz',gpu_30fps:'GPU 转换 30 Hz 变体'};
const CAMS=[['follow','跟随'],['orbit','环绕'],['front','正视'],['side','侧视'],['feet','脚部'],['under','仰视地面']];

function ResidualCurve({scene,total,frame,onSeek}){
 const cv=useRef(null),host=useRef(null),[w,setW]=useState(800);
  const r=scene?.arrays?.residuals?.data;
 useEffect(()=>{const ob=new ResizeObserver(e=>setW(Math.floor(e[0].contentRect.width)));if(host.current)ob.observe(host.current);return()=>ob.disconnect();},[!!r]);
 useEffect(()=>{const c=cv.current;if(!c||!r)return;const H=54,dpr=Math.min(devicePixelRatio,2);c.width=w*dpr;c.height=H*dpr;const x=c.getContext('2d');x.scale(dpr,dpr);x.clearRect(0,0,w,H);let mx=0;for(const v of r)if(Number.isFinite(v))mx=Math.max(mx,v);mx=mx||1;x.strokeStyle='#4a86c5';x.beginPath();for(let px=0;px<w;px++){const a=Math.floor(px/w*total),b=Math.max(a+1,Math.ceil((px+1)/w*total));let v=-1;for(let f=a;f<b&&f<total;f++)v=Math.max(v,r[f]);const y=H-4-(H-10)*v/mx;px?x.lineTo(px,y):x.moveTo(px,y);}x.stroke();x.fillStyle='#6b7f8b';x.font='10px system-ui';x.fillText(`峰值 ${mx.toFixed(4)}（文件 per_frame_residuals，单位以转换器为准）`,6,10);},[r,w,total]);
 if(!r)return <p className="muted">此片段没有逐帧拟合残差序列。</p>;
 return <div className="at-resid" ref={host} onPointerDown={e=>{const b=e.currentTarget.getBoundingClientRect();onSeek(Math.round((e.clientX-b.left)/b.width*(total-1)));}}><canvas ref={cv} style={{width:w,height:54}}/><div className="at-playhead" style={{left:(total>1?frame/(total-1):0)*w}}/></div>;
}

export function HumanPage(){
 const [filters,setFilters]=useState({source:'',variant:'',category:'',q:'',sort:'source'}),[list,setList]=useState(null),[err,setErr]=useState(null),[sel,setSel]=useState(null),[detail,setDetail]=useState(null),[scene,setScene]=useState(null),[loadErr,setLoadErr]=useState(null),[mode,setMode]=useState('contact'),[cam,setCam]=useState('follow'),[skel,setSkel]=useState(true),[limit,setLimit]=useState(200);
 useEffect(()=>{const t=setTimeout(()=>{const q=new URLSearchParams({...filters,limit:String(limit)});json('/api/human/catalog?'+q).then(r=>{setList(r);setErr(null);setSel(s=>s||r.items.find(x=>!x.viewer_unsupported)?.id||null);}).catch(e=>setErr(e.message));},200);return()=>clearTimeout(t);},[filters,limit]);
 useEffect(()=>{setDetail(null);setScene(null);setLoadErr(null);if(!sel)return;let live=true;json('/api/human/items/'+sel).then(d=>{if(!live)return;setDetail(d);return binary(`/api/human/items/${sel}/scene`).then(s=>live&&setScene(s));}).catch(e=>live&&setLoadErr(e.message));return()=>{live=false;};},[sel]);
 const total=detail?.frames||1,fps=detail?.fps||30;
 const clock=usePlayClock(total,fps);
 const stream=useChunkStream(scene&&sel?`/api/human/items/${sel}`:null,'',total,clock.frame,clock.playing);
 clock.readyRef.current=()=>stream.ready;
 useEffect(()=>{clock.setFrame(0);clock.setPlaying(false);},[sel]); // eslint-disable-line
 const facets=list?.facets||{},cats=filters.source&&facets[filters.source]?Object.entries(facets[filters.source].categories).sort((a,b)=>b[1]-a[1]):[];
 const set=(k,v)=>{setFilters(f=>({...f,[k]:v,...(k==='source'?{category:'',variant:''}:{})}));setLimit(200);};
 return <div className="at-page at-human">
  <aside className="at-rail"><h2>人体原始数据 <small>SMPL-X</small></h2>
   <div className="at-facets">{Object.entries(facets).map(([k,v])=><span key={k}>{k}：{v.clips} 段 · {fmtH(v.hours)}{v.variants?.gpu_30fps?`（另有 30 Hz 变体 ${v.variants.gpu_30fps} 段，不重复计）`:''}</span>)}</div>
   <div className="at-filters"><select value={filters.source} onChange={e=>set('source',e.target.value)}>{SRC.map(([v,t])=><option key={v} value={v}>{t}</option>)}</select>
    {filters.source==='hiphi'?<select value={filters.variant} onChange={e=>set('variant',e.target.value)}><option value="">全部转换版本</option>{Object.entries(VAR).map(([v,t])=><option key={v} value={v}>{t}</option>)}</select>:null}
    {cats.length?<select value={filters.category} onChange={e=>set('category',e.target.value)}><option value="">全部类别</option>{cats.map(([c,n])=><option key={c} value={c}>{c}（{n}）</option>)}</select>:null}
    <input placeholder="搜索片段名 / 类别…" value={filters.q} onChange={e=>set('q',e.target.value)}/>
    <select value={filters.sort} onChange={e=>set('sort',e.target.value)}><option value="source">按来源 / 类别</option><option value="duration">时长从长到短</option><option value="residual">拟合残差从大到小</option></select></div>
   {err?<p className="at-err">{err}</p>:null}
   <div className="at-count">{list?`${list.total} 段（显示 ${list.items.length}）`:'读取清单…'}</div>
   <div className="at-list">{list?.items.map(e=><button key={e.id} className={'at-item'+(e.id===sel?' sel':'')+(e.viewer_unsupported?' off':'')} onClick={()=>setSel(e.id)} title={e.viewer_unsupported||''}><strong>{e.clip}</strong><span>{e.source} · {e.category} · {VAR[e.variant]||e.variant}</span><span>{e.duration_s?`${e.duration_s.toFixed(1)} s`:'—'} · {e.fps?Number(e.fps).toFixed(e.fps%1?3:0):'—'} Hz · 残差 {e.residual_cm_mean!=null?`${e.residual_cm_mean} cm（p95 ${e.residual_cm_p95}）`:e.src_valid_frac!=null?`源有效 ${(e.src_valid_frac*100).toFixed(0)}%`:'—'}</span></button>)}
    {list&&list.items.length<list.total?<button className="at-more" onClick={()=>setLimit(l=>Math.min(l+300,500))} disabled={limit>=500}>{limit>=500?'最多显示 500 条，请用筛选缩小范围':'显示更多'}</button>:null}</div></aside>
  <main className="at-main">
   <div className="at-toolbar"><div className="at-seg">{MODES.map(([v,t])=><button key={v} className={mode===v?'active':''} onClick={()=>setMode(v)}>{t}</button>)}</div><div className="at-seg small">{CAMS.map(([v,t])=><button key={v} className={cam===v?'active':''} onClick={()=>setCam(v)}>{t}</button>)}</div><label><input type="checkbox" checked={skel} onChange={e=>setSkel(e.target.checked)}/>骨架（22 个身体关节）</label></div>
   <ModeLegend mode={mode} hasObject={false} human/>
   {loadErr?<div className="at-banner err">{loadErr}</div>:null}
   {sel&&!detail&&!loadErr?<div className="at-banner">正在后端 CPU 现算 SMPL-X（读取动作 + 地面校准）…</div>:null}
   <div className="at-single"><PointCloudView scene={scene} chunk={stream.chunk} localFrame={stream.localFrame} kind="human" mode={mode} showSkeleton={skel} cameraView={cam} title={detail?detail.clip:'人体'} subtitle={detail?`${detail.source} · ${detail.category} · 4096 槽位点 · 界面现算`:''}/></div>
   <div className="at-evidence">{stream.error?<span className="at-err">{stream.error.message}</span>:<span>显示帧 {stream.shownFrame??'—'}{stream.busy?` · 正在现算帧块 ${Math.floor(clock.frame/32)*32}…`:''}</span>}</div>
   <Playback frame={clock.frame} setFrame={clock.setFrame} total={total} fps={fps} playing={clock.playing} setPlaying={clock.setPlaying} speed={clock.speed} setSpeed={clock.setSpeed} range={clock.range} setRange={clock.setRange} loop={clock.loop} setLoop={clock.setLoop}/>
   <section className="at-card"><div className="at-section-head"><h3>拟合残差（逐帧）</h3><span>点击跳帧</span></div><ResidualCurve scene={scene} total={total} frame={clock.frame} onSeek={f=>{clock.setPlaying(false);clock.setFrame(f);}}/></section>
   {detail?<section className="at-card at-kv"><div><span>文件</span><code>{detail.path}</code></div><div><span>帧数 / 帧率</span>{detail.frames} 帧 · {Number(detail.fps).toFixed(4)} Hz · {(detail.frames/detail.fps).toFixed(1)} s</div><div><span>拟合残差（qa.json）</span>{detail.residual_cm_mean!=null?`均值 ${detail.residual_cm_mean} cm · p95 ${detail.residual_cm_p95} cm · 最大 ${detail.residual_cm_max} cm`:'—'}</div><div><span>地面（界面现算）</span>原始坐标 z = {Number(detail.floor_m_raw).toFixed(4)} m · {detail.floor_method} · 样本 {detail.floor_samples}（步长 {detail.floor_stride} 帧）</div><div><span>表面点</span>{detail.point_count} 个 UMR 人体槽位 · 绑定来源 {detail.slot_robot}</div><div><span>上方向</span>{detail.up==='y'?'源为 y-up，已按 UMR 的 Y→Z 矩阵转成 z-up':'z-up'} · 性别 {detail.gender}</div>{detail.objects?.length?<div><span>物体</span>{detail.objects.join('、')}（本页不画物体；物体网格见 GPU 输出页）</div>:null}{(detail.notes||[]).map((n,i)=><p key={i} className="hint">{n}</p>)}</section>:null}
   <ContactExplainer/>
  </main></div>;
}
