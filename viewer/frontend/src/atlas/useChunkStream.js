import {useEffect,useRef,useState} from 'react';
import {binary} from '../api';
// Aligned chunk loader: frame -> chunk [floor(f/size)*size, +size). Keeps the last
// real frame on screen while the next chunk loads (never interpolates), LRU of 4.
export function useChunkStream(base,query,frames,frame,playing,size=32){
 const cache=useRef(new Map()),pending=useRef(new Map()),epoch=useRef(0),shown=useRef(null);
 const [tick,setTick]=useState(0),[error,setError]=useState(null);
 const ident=base?base+'?'+query:null;
 useEffect(()=>{epoch.current++;for(const c of pending.current.values())c.abort();pending.current.clear();cache.current.clear();shown.current=null;setError(null);setTick(t=>t+1);},[ident]);
 const start=frames?Math.floor(Math.min(Math.max(frame,0),frames-1)/size)*size:0;
 useEffect(()=>{if(!ident||!frames)return;const ep=epoch.current;
  const get=at=>{if(at>=frames||cache.current.has(at)||pending.current.has(at))return;const ctl=new AbortController();pending.current.set(at,ctl);
   binary(`${base}/chunk?${query}${query?'&':''}start=${at}&count=${size}`,ctl.signal).then(d=>{if(ep!==epoch.current)return;cache.current.set(at,d);while(cache.current.size>4){const k=[...cache.current.keys()].find(k=>k!==start);cache.current.delete(k);}setTick(t=>t+1);}).catch(e=>{if(e.name!=='AbortError'&&ep===epoch.current)setError(e);}).finally(()=>{if(pending.current.get(at)===ctl)pending.current.delete(at);});};
  for(const [k,c] of pending.current)if(k!==start&&k!==start+size){c.abort();pending.current.delete(k);}
  get(start);if(playing)get(start+size);
 },[ident,start,frames,playing,size]);
 const hit=cache.current.get(start);
 if(hit){const lf=frame-hit.meta.start;if(lf>=0&&lf<hit.meta.count)shown.current={chunk:hit,localFrame:lf,frame};}
 return {chunk:shown.current?.chunk||null,localFrame:shown.current?.localFrame??-1,shownFrame:shown.current?.frame??null,ready:!!hit,busy:!hit,error,tick};
}
// Shared clock: advances only when every registered view has the current frame.
export function usePlayClock(total,fps){
 const [frame,setFrame]=useState(0),[playing,setPlaying]=useState(false),[speed,setSpeed]=useState(1),[loop,setLoop]=useState(true),[range,setRange]=useState(null);
 const readyRef=useRef(()=>true),state=useRef({});state.current={playing,speed,total,fps,loop,range};
 useEffect(()=>{let id,last=0,acc=0;const step=t=>{id=requestAnimationFrame(step);const s=state.current;if(!s.playing||!readyRef.current()){last=t;acc=0;return;}acc+=Math.min((t-last)/1000,.1)*s.fps*s.speed;last=t;const adv=Math.floor(acc);if(!adv)return;acc-=adv;setFrame(f=>{const lo=s.range?.[0]??0,hi=s.range?.[1]??s.total;const n=f+adv;if(n>=hi){if(s.loop)return lo+(n-lo)%Math.max(hi-lo,1);setPlaying(false);return hi-1;}return Math.max(n,lo);});};id=requestAnimationFrame(step);return()=>cancelAnimationFrame(id);},[]);
 return {frame,setFrame,playing,setPlaying,speed,setSpeed,loop,setLoop,range,setRange,readyRef};
}
