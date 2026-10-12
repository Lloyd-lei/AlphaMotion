import {useCallback,useEffect,useMemo,useRef,useState} from 'react';
import {binary,json} from '../api';
import {Playback} from '../Playback';
import {PointCloudView} from './PointCloudView';
import {useChunkStream,usePlayClock} from './useChunkStream';
import {GpuTimeline} from './GpuTimeline';
import {ContactExplainer,ModeLegend} from './Explain';
import {MODES,ROBOT_ZH,pct} from './palette';
const PAGE=6;
const CAMS=[['orbit','环绕'],['follow','跟随'],['front','正视'],['side','侧视'],['top','俯视'],['feet','脚部'],['under','仰视地面']];
const SRC_ZH={omnicontact:'OmniContact',hiphi:'HiPHI',lafan:'LAFAN',grail:'GRAIL',unknown:'未知来源'};
const PIPE_ZH={gpu_export:'评测导出',production_v1:'量产 v1'};

function Tile({kind,robotKey,sha,scene,frames,frame,playing,mode,showSkeleton,showMesh,cameraView,title,subtitle,onReady,human}){
 const base=robotKey?`/api/gpu/items/${robotKey}`:null;
 const s=useChunkStream(sha&&scene?base:null,`sha256=${sha}&human=${human?1:0}`,frames,frame,playing);
 useEffect(()=>{onReady(kind+robotKey,s.ready);},[s.ready,kind,robotKey,onReady]);
 return <div className="at-tile"><PointCloudView scene={scene} chunk={s.chunk} localFrame={s.localFrame} kind={kind} mode={mode} showSkeleton={showSkeleton} showMesh={showMesh} cameraView={cameraView} title={title} subtitle={subtitle}/>
  <div className="at-tile-foot">{s.error?<span className="at-err">{s.error.message}</span>:<span>显示帧 {s.shownFrame??'—'}{s.busy?` · 等待帧 ${frame}`:''}</span>}</div></div>;
}

function QcCard({robot,detail,selected,onSelect}){
 if(!detail)return <button className="at-qc" onClick={onSelect}><strong>{ROBOT_ZH(robot)}</strong><span>读取中…</span></button>;
 if(detail.error)return <div className="at-qc err"><strong>{ROBOT_ZH(robot)}</strong><span>{detail.error}</span></div>;
 const qc=detail.qc||{},T=detail.frames,rm=detail.root_motion||{};
 const hist=detail.flag_counts||{};
 return <button className={'at-qc'+(selected?' sel':'')} onClick={onSelect}><strong>{ROBOT_ZH(robot)}</strong>
  <dl><dt>被剔除帧</dt><dd className={qc.flagged/T>.1?'bad':''}>{pct((qc.flagged??detail.flagged_frames)/T)}<small> {qc.flagged??detail.flagged_frames}/{T}</small></dd>
  <dt>最大穿透</dt><dd>地 {Number(qc.ground_pen_max_mm??0).toFixed(1)} · 物 {Number(qc.object_pen_max_mm??0).toFixed(1)} mm</dd>
  <dt>打滑占比</dt><dd>{pct(qc.ground_slip_frac_of_contact)}<small> 接触帧中</small></dd>
  <dt>人机接触一致</dt><dd className={qc.human_contact_robot_solid_frac<.8?'bad':''}>{pct(qc.human_contact_robot_solid_frac)}</dd>
  <dt>根跳变</dt><dd className={rm.frames_over_10cm?'bad':''}>{rm.frames_over_10cm??'—'} 帧 &gt;10 cm<small> 最大 {Number(rm.max_cm_per_frame||0).toFixed(0)} cm</small></dd></dl>
  <div className="at-bits">{(detail.flag_bits||[]).map(b=><span key={b.bit} title={`flag ${b.bit}：${b.zh}`}><i style={{width:`${Math.min(100,100*(hist[b.bit]||0)/Math.max(T,1))}%`}}/><em>{b.bit}</em><b>{hist[b.bit]||0}</b></span>)}</div></button>;
}

export function GpuPage(){
 const [catalog,setCatalog]=useState(null),[err,setErr]=useState(null),[clipId,setClipId]=useState(null),[mode,setMode]=useState('region'),[cam,setCam]=useState('follow'),[skel,setSkel]=useState(true),[mesh,setMesh]=useState(true),[page,setPage]=useState(0),[sel,setSel]=useState(null);
 const [data,setData]=useState({}),[flt,setFlt]=useState({q:'',source:'',pipeline:''}),[limit,setLimit]=useState(200),[clipObj,setClipObj]=useState(null);
 useEffect(()=>{const t=setTimeout(()=>json('/api/gpu/catalog?'+new URLSearchParams({...flt,limit:String(limit)})).then(c=>{setCatalog(c);setErr(null);const pick=c.clips.find(x=>x.pipeline==='gpu_export'&&x.has_object)||c.clips[0];setClipId(id=>id||pick?.clip_key);setClipObj(o=>o||pick||null);}).catch(e=>setErr(e.message)),200);return()=>clearTimeout(t);},[flt,limit]);
 const clip=catalog?.clips.find(c=>c.clip_key===clipId)||(clipObj?.clip_key===clipId?clipObj:null);
 const robots=clip?.robots||[];const shown=robots.slice(page*PAGE,page*PAGE+PAGE);
 useEffect(()=>{setPage(0);setSel(null);const keep=new Set((clip?.robots||[]).map(r=>r.key));setData(x=>Object.fromEntries(Object.entries(x).map(([k,v])=>[k,keep.has(k)?v:{detail:v.detail}])));},[clipId]); // drop other clips' scenes (bounded browser memory)
 // details for every robot of the clip, scenes for the visible page only (bounded memory)
 useEffect(()=>{if(!clip)return;let live=true;(async()=>{for(const r of clip.robots){if(!live)return;if(data[r.key]?.detail)continue;try{const d=await json(`/api/gpu/items/${r.key}`);if(live)setData(x=>({...x,[r.key]:{...x[r.key],detail:d}}));}catch(e){if(live)setData(x=>({...x,[r.key]:{...x[r.key],detail:{error:e.message}}}));}}})();return()=>{live=false;};},[clipId]); // eslint-disable-line
 const shownKeys=shown.map(r=>r.key).join(',');
 useEffect(()=>{let live=true;(async()=>{for(const r of shown){const d=data[r.key]?.detail;if(!d||d.error||data[r.key]?.scene)continue;try{const s=await binary(`/api/gpu/items/${r.key}/scene?sha256=${d.sha256}`);if(live)setData(x=>({...x,[r.key]:{...x[r.key],scene:s}}));}catch(e){if(live)setData(x=>({...x,[r.key]:{...x[r.key],sceneError:e.message}}));}}})();return()=>{live=false;};},[shownKeys,Object.values(data).filter(v=>v.detail).length]); // eslint-disable-line
 const humanRobot=useMemo(()=>{const ok=shown.filter(r=>data[r.key]?.detail?.has_human);return ok.find(r=>r.robot==='unitree_g1_29dof')||ok[0];},[shownKeys,data]); // eslint-disable-line
 const first=data[shown[0]?.key]?.detail,total=first?.frames||clip?.frames||1,fps=first?.fps||clip?.fps||30;
 const clock=usePlayClock(total,fps);const ready=useRef({});
 const onReady=useCallback((k,v)=>{ready.current[k]=v;},[]);
 clock.readyRef.current=()=>Object.values(ready.current).every(Boolean);
 useEffect(()=>{ready.current={};clock.setFrame(0);clock.setPlaying(false);},[clipId,page]); // eslint-disable-line
 const selKey=sel||shown[0]?.key,selData=data[selKey];
 useEffect(()=>{const fn=e=>{if(/input|textarea|select/i.test(e.target.tagName))return;if(e.code==='Space'){e.preventDefault();clock.setPlaying(p=>!p);}if(e.key==='ArrowLeft'){clock.setPlaying(false);clock.setFrame(f=>Math.max(0,f-1));}if(e.key==='ArrowRight'){clock.setPlaying(false);clock.setFrame(f=>Math.min(total-1,f+1));}};window.addEventListener('keydown',fn);return()=>window.removeEventListener('keydown',fn);},[total]); // eslint-disable-line
 const warnings=[...new Set(shown.flatMap(r=>(data[r.key]?.detail?.warnings||[]).map(w=>w)))];
 const hd=humanRobot?data[humanRobot.key]:null;
 return <div className="at-page at-gpu">
  <aside className="at-rail"><h2>GPU 输出 <small>gpu_npz_v2</small><button className="at-refresh" title="重新扫描输出目录" onClick={()=>{setFlt(f=>({...f}));}}>↻</button></h2>{err?<p className="at-err">{err}</p>:null}{!catalog?<p className="muted">读取目录…</p>:null}
   <div className="at-filters"><input placeholder="搜索片段名…" value={flt.q} onChange={e=>{setFlt(f=>({...f,q:e.target.value}));setLimit(200);}}/>
    <select value={flt.pipeline+'|'+flt.source} onChange={e=>{const [pipeline,source]=e.target.value.split('|');setFlt(f=>({...f,pipeline,source}));setLimit(200);}}><option value="|">全部管线 / 来源</option>{(catalog?.facets||[]).map(f=><option key={f.pipeline+f.source} value={f.pipeline+'|'+f.source}>{PIPE_ZH[f.pipeline]||f.pipeline} · {SRC_ZH[f.source]||f.source}（{f.clips} 段 / {f.files} 文件）</option>)}</select></div>
   <div className="at-count">{catalog?`${catalog.total} 个片段（显示 ${catalog.clips.length}）`:''}</div>
   <div className="at-list">{catalog?.clips.map(c=><button key={c.clip_key} className={'at-clip'+(c.clip_key===clipId?' sel':'')} onClick={()=>{setClipId(c.clip_key);setClipObj(c);}}><strong title={c.clip_id}>{c.invalid?<em className="at-bad">已作废 </em>:null}{c.clip_id}</strong><span>{PIPE_ZH[c.pipeline]||c.pipeline} · {SRC_ZH[c.source]||c.source} · {c.frames} 帧 · {Number(c.fps).toFixed(2)} Hz{c.has_object?' · 含物体':''}</span><span>{c.robots.map(r=>ROBOT_ZH(r.robot)).join(' / ')}</span></button>)}
   {catalog&&catalog.clips.length<catalog.total?<button className="at-more" disabled={limit>=500} onClick={()=>setLimit(l=>Math.min(500,l+300))}>{limit>=500?'最多 500 条，请用搜索缩小':'显示更多'}</button>:null}</div>
   {catalog?<p className="hint">扫描：{catalog.export_dirs.join('、')}；量产目录 {catalog.production_root} {catalog.production_exists?'':'（尚不存在）'}</p>:null}</aside>
  <main className="at-main">
   <div className="at-toolbar"><div className="at-seg">{MODES.map(([v,t])=><button key={v} className={mode===v?'active':''} onClick={()=>setMode(v)}>{t}</button>)}</div>
    <div className="at-seg small">{CAMS.map(([v,t])=><button key={v} className={cam===v?'active':''} onClick={()=>setCam(v)}>{t}</button>)}</div>
    <label><input type="checkbox" checked={skel} onChange={e=>setSkel(e.target.checked)}/>骨架</label><label><input type="checkbox" checked={mesh} onChange={e=>setMesh(e.target.checked)}/>半透明网格</label>
    {robots.length>PAGE?<span className="at-pager"><button disabled={!page} onClick={()=>setPage(p=>p-1)}>上一组</button>{page+1}/{Math.ceil(robots.length/PAGE)}<button disabled={(page+1)*PAGE>=robots.length} onClick={()=>setPage(p=>p+1)}>下一组</button></span>:null}</div>
   <ModeLegend mode={mode} hasObject={!!clip?.has_object}/>
   <div className="at-grid" style={{gridTemplateColumns:`repeat(${shown.length+1<=3?shown.length+1:shown.length+1<=6?3:4},minmax(0,1fr))`}}>
    <Tile kind="human" robotKey={humanRobot?.key} sha={hd?.detail?.sha256} scene={hd?.scene} frames={total} frame={clock.frame} playing={clock.playing} mode={mode} showSkeleton={false} showMesh cameraView={cam} human title="人体源（UMR 槽位）" subtitle={humanRobot?`4096 点 · 来自 ${ROBOT_ZH(humanRobot.robot)} 捕获`:'无捕获'} onReady={onReady}/>
    {shown.map(r=>{const d=data[r.key]||{};return <Tile key={r.key} kind="robot" robotKey={r.key} sha={d.detail?.sha256} scene={d.scene} frames={total} frame={clock.frame} playing={clock.playing} mode={mode} showSkeleton={skel} showMesh={mesh} cameraView={cam} title={ROBOT_ZH(r.robot)} subtitle={d.sceneError||d.detail?.error||'MJCF · CPU 正向运动学'} onReady={onReady}/>;})}
   </div>
   <Playback frame={clock.frame} setFrame={clock.setFrame} total={total} fps={fps} playing={clock.playing} setPlaying={clock.setPlaying} speed={clock.speed} setSpeed={clock.setSpeed} range={clock.range} setRange={clock.setRange} loop={clock.loop} setLoop={clock.setLoop}/>
   <div className="at-qcs">{robots.map(r=><QcCard key={r.key} robot={r.robot} detail={data[r.key]?.detail} selected={r.key===selKey} onSelect={()=>setSel(r.key)}/>)}</div>
   {selData?.scene?<GpuTimeline scene={selData.scene} total={total} fps={fps} frame={clock.frame} onSeek={f=>{clock.setPlaying(false);clock.setFrame(f);}} flagBits={selData.detail?.flag_bits}/>:<p className="muted">时间轴：等待 {ROBOT_ZH(robots.find(r=>r.key===selKey)?.robot)} 场景…（点上面的质检卡切换机器人）</p>}
   <ContactExplainer compact/>
   <details className="at-notes" open><summary>数据说明与问题（{warnings.length}）</summary>
    <p>人体源 = UMR 捕获里喂给求解器的 4096 个人体表面槽位（problems/&lt;clip&gt;/&lt;robot&gt;.stock.cpu.npz 的 src，已按 smpl_scale 缩放，减去同一个 floor_offset_removed），与机器人在同一地面。物体网格取自同一捕获的 objpen 模型（已缩放），位姿为文件 object_pose。</p>
    {hd?.detail?.source_ref?<p>人体源（clip → SMPL-X）：{hd.detail.source_ref.resolved||`捕获时路径 ${hd.detail.source_ref.data} 已不存在`}；seq_key = {hd.detail.source_ref.seq_key}{hd.detail.source_ref.factory_matches?.length?<>；数据工厂对应：{hd.detail.source_ref.factory_matches.map(m=><code key={m.id}> {m.path}（人体原始页 id {m.id}）</code>)}</>:'；数据工厂 smplx 目录里没有同名片段'}。</p>:null}
    {warnings.map((w,i)=><p key={i} className="at-warn">{w}</p>)}</details>
  </main></div>;
}
