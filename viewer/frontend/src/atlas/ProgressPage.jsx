import {useEffect,useState} from 'react';
import {json} from '../api';
import {fmtGB,fmtH,ROBOT_ZH} from './palette';
const STATE_ZH={ready:'已下载并解包',foreign:'已在盘（他人下载）',extracted:'已解包',downloading:'下载中',queued:'排队',partial:'部分下载',unknown:'未知'};
const STATE_C={ready:'#1f9e6a',foreign:'#4b9fc4',extracted:'#1f9e6a',downloading:'#e7a52b',queued:'#d5dde2',partial:'#f0c87a',unknown:'#d5dde2'};
const ago=t=>{if(!t)return '—';const s=Date.now()/1000-t;return s<90?`${Math.round(s)} 秒前`:s<5400?`${Math.round(s/60)} 分钟前`:`${(s/3600).toFixed(1)} 小时前`;};

function Bar({h,known,color,label}){const w=known?Math.min(100,100*(h||0)/known):0;return <div className="at-bar" title={label}><i style={{width:w+'%',background:color}}/><span>{label}</span></div>;}

function Download({d,known}){
 if(!d||(!d.kind&&!d.error))return <span className="muted">{d?.note||'本机无下载任务'}</span>;
 if(d.error)return <span className="at-err">{d.error}</span>;
 const frac=d.total_bytes?d.done_bytes/d.total_bytes:(d.segments_total?d.segments_done/d.segments_total:null);
 return <div className="at-dl"><div className="at-bar"><i style={{width:(frac==null?0:Math.min(100,frac*100))+'%',background:'#3c8fc1'}}/><span>{d.total_bytes?`${fmtGB(d.done_bytes)} / ${fmtGB(d.total_bytes)}`:fmtGB(d.done_bytes)} · {d.segments_done??'—'}/{d.segments_total??'—'} {d.kind}{d.hours_done!=null?` · ${fmtH(d.hours_done)} / ${fmtH(d.hours_total)}`:''}</span></div>
  <div className="at-dl-meta">{d.current?<span>当前：{typeof d.current==='string'&&/^\d{4}$/.test(d.current)?`第 ${d.current} 卷`:d.current}</span>:null}{d.rate_MBps!=null?<span>速度 {Number(d.rate_MBps).toFixed(2)} MB/s{d.rate_MBps_10m!=null?`（10 分钟均 ${Number(d.rate_MBps_10m).toFixed(2)}）`:''}</span>:d.last_round_rate_MBps!=null?<span>上一轮均速 {d.last_round_rate_MBps} MB/s</span>:null}{d.eta_h!=null?<span>预计 {Number(d.eta_h).toFixed(1)} h 下完</span>:null}{d.net_up===false?<span className="at-err">网络断开</span>:null}{d.paused_reason?<span className="at-err">暂停：{d.paused_reason}</span>:null}{d.status_time?<span>状态 {d.status_time}</span>:null}</div>
  {d.parts?<div className="at-parts">{d.parts.map(p=><i key={p.part} title={`第 ${p.part} 卷 · ${STATE_ZH[p.state]||p.state} · ${fmtGB(p.bytes)} · ${p.hours} h（原片+镜像）${p.state==='downloading'?` · 已下 ${fmtGB(p.have_bytes)}`:''}`} style={{background:STATE_C[p.state]||'#d5dde2'}}>{p.state==='downloading'?<b style={{height:Math.round(100*p.have_bytes/p.bytes)+'%'}}/>:null}</i>)}</div>:null}
  {d.hold?.length?<div className="hint">另有暂停的部分下载：{d.hold.join('、')}</div>:null}</div>;
}

export function ProgressPage(){
 const [p,setP]=useState(null),[err,setErr]=useState(null);
 useEffect(()=>{let live=true,timer;const load=force=>json('/api/atlas/progress'+(force?'?force=true':'')).then(r=>{if(!live)return;setP(r);setErr(null);timer=setTimeout(()=>load(false),r.scanning||!r.result?3000:60000);}).catch(e=>{if(live){setErr(e.message);timer=setTimeout(()=>load(false),15000);}});load(true);return()=>{live=false;clearTimeout(timer);};},[]);
 const r=p?.result;
 return <div className="at-page at-progress"><main className="at-main wide">
  <div className="at-section-head"><h2>数据转换进度</h2><span>{r?`扫描于 ${ago(r.generated_at)}（用时 ${r.scan_seconds} s）· 后台最多每 5 分钟重扫一次，只读文件头`:'首次扫描中…'}{p?.scanning?' · 正在后台重扫':''}</span></div>
  {err?<div className="at-banner err">{err}</div>:null}{p?.error?<div className="at-banner err">上次扫描出错：{p.error}</div>:null}
  {r?<><table className="at-funnel"><thead><tr><th>来源</th><th>已知总时长</th><th>① 已下载</th><th>② 已转成 SMPL-X</th><th>③ 已重定向到机器人</th></tr></thead><tbody>{r.funnel.map(f=><tr key={f.id}><th><strong>{f.name}</strong><small>{f.note}</small></th>
    <td><b>{fmtH(f.known_h)}</b>{f.mirror_h?<small>原片 {fmtH(f.orig_h)} + 镜像 {fmtH(f.mirror_h)}</small>:null}</td>
    <td><Download d={f.download} known={f.known_h}/></td>
    <td><Bar h={f.smplx.hours} known={f.orig_h||f.known_h} color="#2aa37a" label={`${fmtH(f.smplx.hours)} · ${f.smplx.clips} 段`}/>{Object.keys(f.smplx.variants||{}).length?<small>{Object.entries(f.smplx.variants).map(([k,v])=>`${k} ${fmtH(v.hours)}/${v.clips}段`).join(' · ')}</small>:null}</td>
    <td><Bar h={f.retarget.hours} known={f.known_h} color="#c4782a" label={`${fmtH(f.retarget.hours)} · ${f.retarget.clip_robot_pairs} 个 片段×机器人 · ${f.retarget.robots} 台`}/></td></tr>)}</tbody></table>
   <p className="hint">条形长度 = 该阶段小时数 ÷ 已知总时长（SMPL-X 用原片时长作分母）。HiPHI 色块：每卷一个，绿 = 已下载，蓝 = 已在盘，黄 = 下载中（填充高度 = 进度），灰 = 排队；鼠标悬停看卷号、大小、时长。</p>
   {r.production?<section className="at-card at-prod"><div className="at-section-head"><h3>GPU 量产 v1 运行状态</h3><span>{r.production.root}</span></div><p>任务 {r.production.done}/{r.production.jobs} 完成{r.production.failed?` · 失败 ${r.production.failed}`:''}{r.production.stop_file?' · 已放置 _STOP（运行器停止）':''}{r.invalid_files?` · ${r.invalid_files} 个文件属于已作废引擎，未计入下表`:''}</p>{Object.entries(r.production.invalid_engines||{}).map(([e,why])=><p key={e} className="at-warn">作废引擎 {e}：{why}</p>)}{(r.production.last_log||[]).map((l,i)=><p key={i} className="hint">{l}</p>)}</section>:null}
   <div className="at-section-head"><h3>来源 × 机器人（{r.robots.length} 台）</h3><span>每格 = 已重定向时长 / 片段×机器人文件数；空 = 0；悬停看各管线</span></div>
   <div className="at-matrix-wrap"><table className="at-matrix"><thead><tr><th/>{[...r.robots,...r.extra_robots].map(rb=><th key={rb} title={rb}><span>{ROBOT_ZH(rb)}</span></th>)}</tr></thead><tbody>{r.funnel.map(f=><tr key={f.id}><th>{f.name}</th>{[...r.robots,...r.extra_robots].map(rb=>{const c=r.matrix[f.id]?.[rb];return <td key={rb} className={c?'on':''} title={c?Object.entries(c.by_pipeline).map(([k,v])=>`${r.pipelines[k]||k}：${(v.hours*60).toFixed(1)} min / ${v.clips} 个`).join('\n'):''}>{c?<><b>{c.hours*60>=60?fmtH(c.hours):(c.hours*60).toFixed(1)+'′'}</b><small>{c.clips}</small></>:''}</td>;})}</tr>)}</tbody></table></div>
   <details className="at-notes"><summary>口径说明</summary>{r.notes.map((n,i)=><p key={i}>{n}</p>)}<p>SMPL-X 索引共 {r.human_index.entries} 条（含 GRAIL 清单行），读头用时 {r.human_index.scan_seconds} s。</p></details></>:<p className="muted">首次扫描通常需要几秒…</p>}
 </main></div>;
}
