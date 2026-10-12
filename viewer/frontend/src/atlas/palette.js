// Shared colours for the atlas pages. Human and robot use the SAME 13-region palette.
export const REGIONS=['trunk','L_thigh','R_thigh','L_shin','R_shin','L_foot','R_foot','L_upper_arm','R_upper_arm','L_forearm','R_forearm','L_hand','R_hand'];
export const REGION_ZH={trunk:'躯干',L_thigh:'左大腿',R_thigh:'右大腿',L_shin:'左小腿',R_shin:'右小腿',L_foot:'左脚',R_foot:'右脚',L_upper_arm:'左上臂',R_upper_arm:'右上臂',L_forearm:'左前臂',R_forearm:'右前臂',L_hand:'左手',R_hand:'右手'};
// Left side: blue/green family, right side: red/yellow family; trunk neutral.
export const REGION_HEX=['#8d9aa6','#3d6fd6','#dd5338','#68aef0','#f08b5c','#1c3c96','#a12f27','#26a174','#c99d22','#79d4ae','#ead06b','#0d684a','#86640f'];
const hexRGB=h=>[parseInt(h.slice(1,3),16)/255,parseInt(h.slice(3,5),16)/255,parseInt(h.slice(5,7),16)/255];
export const REGION_RGB=REGION_HEX.map(hexRGB);
export const C={ground:hexRGB('#1fb36b'),object:hexRGB('#f2922a'),pen:hexRGB('#ff2fb4'),idle:hexRGB('#c3ccd3'),idleDark:hexRGB('#3a4852'),human:hexRGB('#8ea3b0')};
// Distance heat 0..20 mm: near = bright (white-yellow), far = dark purple ("pressure-map" look).
const STOPS=[[0,[1,.98,.78]],[2,[1,.86,.3]],[5,[1,.55,.13]],[10,[.86,.2,.17]],[15,[.55,.09,.3]],[20,[.25,.07,.33]]];
export function heatRGB(mm){
 if(!(mm<=20))return null;
 const d=Math.max(0,mm);
 for(let i=1;i<STOPS.length;i++){const [b,cb]=STOPS[i];if(d<=b){const [a,ca]=STOPS[i-1],t=(d-a)/(b-a);return [ca[0]+(cb[0]-ca[0])*t,ca[1]+(cb[1]-ca[1])*t,ca[2]+(cb[2]-ca[2])*t];}}
 return STOPS[STOPS.length-1][1];
}
export const heatCSS=mm=>{const c=heatRGB(mm);return `rgb(${Math.round(c[0]*255)},${Math.round(c[1]*255)},${Math.round(c[2]*255)})`;};
export const MODES=[['region','部位语义'],['contact','接触·二值'],['distance','接触·距离场'],['normals','法向箭头']];
export const regionName=r=>REGION_ZH[r]||r;
export const ROBOT_ZH=r=>String(r||'').replace('unitree_','').replace('fourier_','').replace('agibot_','').replace('booster_','').replace('_29dof',' 29').replace('_23dof',' 23').replace('_t3d0','').toUpperCase();
export const fmtH=h=>h==null?'—':h>=100?h.toFixed(0)+' h':h>=10?h.toFixed(1)+' h':h>=0.01?h.toFixed(2)+' h':h>0?(h*60).toFixed(1)+' min':'0';
export const fmtGB=b=>b==null?'—':b>=1e9?(b/1e9).toFixed(b>=1e11?0:1)+' GB':(b/1e6).toFixed(0)+' MB';
export const pct=x=>x==null||!Number.isFinite(x)?'—':(x*100).toFixed(x<0.1?2:1)+'%';
