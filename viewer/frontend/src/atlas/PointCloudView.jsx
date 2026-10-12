import {memo,useEffect,useRef,useState} from 'react';
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {row,values} from '../api';
import {poseElements} from '../geometryMath';
import {REGION_RGB,C,heatRGB} from './palette';

const HUMAN_PARENTS=[-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19];
function makeGeometry(desc,pack){let g;const s=desc.size||[1,1,1];if(desc.type==='mesh'){const v=values(pack,desc.mesh+'_vertices'),f=values(pack,desc.mesh+'_faces');if(!v||!f)return null;g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.BufferAttribute(v,3));g.setIndex(new THREE.BufferAttribute(f,1));g.computeVertexNormals();}else if(desc.type==='box')g=new THREE.BoxGeometry(2*s[0],2*s[1],2*s[2]);else if(desc.type==='sphere')g=new THREE.SphereGeometry(s[0],20,14);else if(desc.type==='ellipsoid'){g=new THREE.SphereGeometry(1,20,14);g.scale(...s);}else if(desc.type==='capsule'){g=new THREE.CapsuleGeometry(s[0],2*s[1],5,12);g.rotateX(Math.PI/2);}else if(desc.type==='cylinder'){g=new THREE.CylinderGeometry(s[0],s[0],2*s[1],20);g.rotateX(Math.PI/2);}return g;}
// Ground grid built from short segments: a single 12 m line that passes behind a low
// camera gets dropped whole by some WebGL line clippers, leaving only a far band.
function makeGrid(half=6,step=.2,piece=.2){const pos=[],col=[];const major=[.62,.7,.75],minor=[.78,.83,.86];const n=Math.round(half/step);
 for(let i=-n;i<=n;i++){const c=i%5===0?major:minor,u=i*step;for(let t=-half;t<half-1e-9;t+=piece){pos.push(u,t,0,u,t+piece,0,t,u,0,t+piece,u,0);col.push(...c,...c,...c,...c);}}
 const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(pos,3));g.setAttribute('color',new THREE.Float32BufferAttribute(col,3));
 const l=new THREE.LineSegments(g,new THREE.LineBasicMaterial({vertexColors:true,transparent:true,opacity:1}));l.position.z=-.0005;return l;}
const M=(p,r,op=0,or=0)=>new THREE.Matrix4().fromArray(poseElements(p,r,op,or));
function meshes(descs,pack,opacity){return (descs||[]).flatMap(d=>{const g=makeGeometry(d,pack);if(!g)return[];const c=d.rgba||[.6,.66,.7,1];const m=new THREE.MeshStandardMaterial({color:new THREE.Color(c[0],c[1],c[2]),roughness:.75,metalness:.05,transparent:opacity<1,opacity,depthWrite:opacity>=1,side:THREE.FrontSide});const mesh=new THREE.Mesh(g,m);mesh.renderOrder=1;mesh.matrixAutoUpdate=false;mesh.userData.desc=d;return[mesh];});}
function dynPoints(n,size){const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.BufferAttribute(new Float32Array(n*3),3).setUsage(THREE.DynamicDrawUsage));g.setAttribute('color',new THREE.BufferAttribute(new Float32Array(n*3),3).setUsage(THREE.DynamicDrawUsage));const p=new THREE.Points(g,new THREE.PointsMaterial({size,vertexColors:true,sizeAttenuation:true,transparent:true,opacity:1}));p.renderOrder=2;return p;}
function dynLines(n,opts={}){const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.BufferAttribute(new Float32Array(n*6),3).setUsage(THREE.DynamicDrawUsage));g.setAttribute('color',new THREE.BufferAttribute(new Float32Array(n*6),3).setUsage(THREE.DynamicDrawUsage));g.setDrawRange(0,0);return new THREE.LineSegments(g,new THREE.LineBasicMaterial({vertexColors:true,transparent:true,opacity:opts.opacity??1,depthTest:opts.depthTest??true}));}

// kind: 'robot' (surface points via geom FK) | 'human' (points supplied per frame).
// mode: region | contact | distance | normals. Everything coloured here is either a
// file label (robot contact/near) or explicitly UI-computed (human height ruler).
export const PointCloudView=memo(function PointCloudView({scene,chunk,localFrame,kind='robot',mode='region',showSkeleton=true,showMesh=true,cameraView='orbit',title,subtitle,regionKey,dimmed=false,onStats}){
 const host=useRef(null),st=useRef(null),latest=useRef({}),[fail,setFail]=useState(null);
 latest.current={chunk,localFrame,mode,showSkeleton,showMesh,cameraView,onStats};
 useEffect(()=>{if(!scene||!host.current)return;let renderer;try{renderer=new THREE.WebGLRenderer({antialias:true,powerPreference:'low-power',preserveDrawingBuffer:true});}catch(e){setFail('WebGL 不可用：'+e.message);return;}setFail(null);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio,1.75));renderer.outputColorSpace=THREE.SRGBColorSpace;host.current.appendChild(renderer.domElement);
  const world=new THREE.Scene(),camera=new THREE.PerspectiveCamera(36,1,.01,100);camera.up.set(0,0,1);camera.position.set(2.6,-3.2,1.8);
  const controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;controls.dampingFactor=.12;controls.target.set(0,0,.8);
  world.add(new THREE.HemisphereLight(0xffffff,0x7d8f9c,2.1));const sun=new THREE.DirectionalLight(0xffffff,2.2);sun.position.set(3,-4,7);world.add(sun);
  const grid=makeGrid();world.add(grid);
  const isRobot=kind==='robot';
  const robotMeshes=isRobot?meshes(scene.meta.geoms,scene,.16):[];const objectMeshes=meshes(scene.meta.object_geoms,scene,.55);for(const m of [...robotMeshes,...objectMeshes])world.add(m);
  const n=isRobot?(scene.meta.surface_point_count||4096):(scene.meta.human_point_count||scene.meta.point_count||4096);
  const pts=dynPoints(n,.016);world.add(pts);
  const bones=dynLines(isRobot?(scene.meta.nbody||64):24,{opacity:.95,depthTest:false});bones.renderOrder=3;world.add(bones);
  const joints=dynPoints(isRobot?(scene.meta.nbody||64):24,.022);joints.material.vertexColors=false;joints.material.color.set(0x0b7f86);joints.material.depthTest=false;joints.renderOrder=4;world.add(joints);
  const arrows=dynLines(n);world.add(arrows);
  // overlay: contact / near points drawn larger and on top so a handful of contact points stay visible
  const hl=dynPoints(n,.03);hl.material.depthTest=false;hl.renderOrder=5;world.add(hl);
  const ro=new ResizeObserver(()=>{if(!host.current)return;const w=host.current.clientWidth,h=host.current.clientHeight;renderer.setSize(w,h,false);camera.aspect=w/Math.max(h,1);camera.updateProjectionMatrix();});ro.observe(host.current);
  const ctx={renderer,world,camera,controls,robotMeshes,objectMeshes,pts,bones,joints,arrows,hl,grid,scene,kind,n,init:false,prevView:'',last:null};st.current=ctx;
  let raf;const loop=()=>{raf=requestAnimationFrame(loop);const p=latest.current;const sig=`${p.localFrame}|${p.mode}|${p.showSkeleton}|${p.showMesh}|${p.cameraView}`;if(p.chunk&&(ctx.last!==p.chunk||ctx.sig!==sig)){draw(ctx,p);ctx.last=p.chunk;ctx.sig=sig;}controls.update();renderer.render(world,camera);};loop();
  return()=>{cancelAnimationFrame(raf);ro.disconnect();controls.dispose();world.traverse(o=>{o.geometry?.dispose();if(o.material)(Array.isArray(o.material)?o.material:[o.material]).forEach(m=>m.dispose());});renderer.dispose();renderer.forceContextLoss();renderer.domElement.remove();st.current=null;};
 },[scene,kind]);
 return <div className={'at-view'+(mode==='distance'||mode==='normals'?' dark':'')+(dimmed?' dimmed':'')}><div className="at-view-title"><strong>{title}</strong><span>{subtitle}</span></div><div className="at-gl" ref={host}/>{!scene?<div className="at-view-empty">正在读取几何…</div>:null}{fail?<div className="at-view-empty err">{fail}</div>:null}<div className="at-view-caption">Z↑ 米 · 地面 z=0</div></div>;
});

function draw(ctx,p){
 const {scene,kind,n}=ctx,chunk=p.chunk,f=p.localFrame;if(f<0||f>=chunk.meta.count)return;
 const robot=kind==='robot',mode=p.mode;
 const pos=ctx.pts.geometry.attributes.position.array,col=ctx.pts.geometry.attributes.color.array;
 let gp,gr,nrmW=null;
 if(robot){gp=row(chunk,'geom_pos',f);gr=row(chunk,'geom_rot',f);const sp=values(scene,'surface_points'),sn=values(scene,'surface_normals'),gb=values(scene,'surface_geom');if(!gp||!sp)return;nrmW=new Float32Array(n*3);
  for(let i=0;i<n;i++){const o=i*3,b=gb[i],q=b*9,t=b*3,x=sp[o],y=sp[o+1],z=sp[o+2];pos[o]=gr[q]*x+gr[q+1]*y+gr[q+2]*z+gp[t];pos[o+1]=gr[q+3]*x+gr[q+4]*y+gr[q+5]*z+gp[t+1];pos[o+2]=gr[q+6]*x+gr[q+7]*y+gr[q+8]*z+gp[t+2];const a=sn[o],bb=sn[o+1],c=sn[o+2];nrmW[o]=gr[q]*a+gr[q+1]*bb+gr[q+2]*c;nrmW[o+1]=gr[q+3]*a+gr[q+4]*bb+gr[q+5]*c;nrmW[o+2]=gr[q+6]*a+gr[q+7]*bb+gr[q+8]*c;}}
 else{const hp=row(chunk,'human_points',f);if(!hp)return;pos.set(hp.subarray(0,n*3));}
 // per-point contact / distance (mm) / partner
 const contact=new Uint8Array(n),dist=new Float32Array(n).fill(NaN),partner=new Uint8Array(n);let nContact=0,nNear=0,nPen=0;
 if(robot){const cp=values(chunk,'contact_ptr'),ci=values(chunk,'contact_idx'),cpa=values(chunk,'contact_partner');if(cp)for(let j=cp[f];j<cp[f+1];j++){contact[ci[j]]=cpa[j]===0?1:2;nContact++;}
  const np=values(chunk,'near_ptr'),ni=values(chunk,'near_idx'),npa=values(chunk,'near_partner'),nq=values(chunk,'near_dist_q');if(np)for(let j=np[f];j<np[f+1];j++){const i=ni[j],d=nq[j]*.1-.5;if(!(dist[i]<=d)){dist[i]=d;partner[i]=npa[j];}nNear++;}}
 else for(let i=0;i<n;i++){const z=pos[i*3+2]*1000;if(z<20){dist[i]=z;nNear++;}if(z>=-.5&&z<=1){contact[i]=1;nContact++;}if(z<-.5)nPen++;}
 const regions=values(scene,robot?'surface_region':'human_region');
 for(let i=0;i<n;i++){const o=i*3;let c;
  if(mode==='region')c=REGION_RGB[regions?.[i]??0]||C.idle;
  else if(mode==='contact')c=contact[i]===1?C.ground:contact[i]===2?C.object:(!robot&&dist[i]<-.5)?C.pen:(robot?C.idle:C.human);
  else{const d=dist[i];c=Number.isFinite(d)?(d<=-.5?C.pen:heatRGB(d)):C.idleDark;}
  col[o]=c[0];col[o+1]=c[1];col[o+2]=c[2];}
 ctx.pts.geometry.attributes.position.needsUpdate=true;ctx.pts.geometry.attributes.color.needsUpdate=true;ctx.pts.geometry.computeBoundingSphere();
 ctx.pts.material.size=robot?.016:.02;
 {const hp=ctx.hl.geometry.attributes.position.array,hc=ctx.hl.geometry.attributes.color.array;let k=0;
  for(let i=0;i<n;i++){let c=null;if(mode==='contact'){c=contact[i]===1?C.ground:contact[i]===2?C.object:(!robot&&dist[i]<-.5)?C.pen:null;}else if(mode==='distance'||mode==='normals'){const d=dist[i];if(Number.isFinite(d))c=d<=-.5?C.pen:heatRGB(d);}
   if(!c)continue;const o=i*3;hp[k*3]=pos[o];hp[k*3+1]=pos[o+1];hp[k*3+2]=pos[o+2];hc[k*3]=c[0];hc[k*3+1]=c[1];hc[k*3+2]=c[2];k++;}
  ctx.hl.geometry.setDrawRange(0,k);ctx.hl.geometry.attributes.position.needsUpdate=true;ctx.hl.geometry.attributes.color.needsUpdate=true;ctx.hl.geometry.computeBoundingSphere();ctx.hl.visible=k>0;ctx.hl.material.size=mode==='contact'?.085:.05;}
 // normal arrows: geometric proxy (surface normal, length shrinking with distance)
 const aPos=ctx.arrows.geometry.attributes.position.array,aCol=ctx.arrows.geometry.attributes.color.array;let na=0;
 if(mode==='normals'){
  let hn=null;if(!robot){const nf=values(chunk,'near_frame'),npnt=values(chunk,'near_point'),nn=values(chunk,'near_normal');if(nf){hn=new Map();for(let j=0;j<nf.length;j++)if(nf[j]===f)hn.set(npnt[j],j);ctx._hn=nn;}}
  for(let i=0;i<n&&na<n;i++){const d=dist[i];if(!Number.isFinite(d)||d>20)continue;let nx,ny,nz;if(robot){nx=nrmW[i*3];ny=nrmW[i*3+1];nz=nrmW[i*3+2];}else{const j=hn?.get(i);if(j===undefined)continue;const nn=ctx._hn;nx=nn[j*3];ny=nn[j*3+1];nz=nn[j*3+2];}
   const L=.006+.044*(1-Math.min(Math.max(d,0),20)/20),o=i*3,k=na*6;aPos[k]=pos[o];aPos[k+1]=pos[o+1];aPos[k+2]=pos[o+2];aPos[k+3]=pos[o]+nx*L;aPos[k+4]=pos[o+1]+ny*L;aPos[k+5]=pos[o+2]+nz*L;
   const c=heatRGB(d)||C.idle;const pc=partner[i]===1?C.object:C.ground;aCol[k]=pc[0];aCol[k+1]=pc[1];aCol[k+2]=pc[2];aCol[k+3]=c[0];aCol[k+4]=c[1];aCol[k+5]=c[2];na++;}}
 ctx.arrows.geometry.setDrawRange(0,na*2);ctx.arrows.geometry.attributes.position.needsUpdate=true;ctx.arrows.geometry.attributes.color.needsUpdate=true;ctx.arrows.geometry.computeBoundingSphere();ctx.arrows.visible=na>0;
 // robot / object meshes
 for(const m of ctx.robotMeshes){const id=m.userData.desc.id;m.matrix.copy(M(gp,gr,id*3,id*9));m.matrixWorldNeedsUpdate=true;m.visible=p.showMesh;}
 const op=row(chunk,'object_pos',f),orr=row(chunk,'object_rot',f);for(const m of ctx.objectMeshes){m.visible=!!(op&&orr);if(m.visible){const d=m.userData.desc;m.matrix.copy(M(op,orr).multiply(M(d.local_pos||[0,0,0],d.local_rot||[1,0,0,0,1,0,0,0,1])));m.matrixWorldNeedsUpdate=true;}}
 // skeleton: robot = body origin -> parent body origin (world body skipped); human = SMPL-X 22 body joints
 const bp=ctx.bones.geometry.attributes.position.array,bc=ctx.bones.geometry.attributes.color.array,jp=ctx.joints.geometry.attributes.position.array;let nb=0,nj=0;
 if(p.showSkeleton){if(robot){const B=row(chunk,'body_pos',f),par=scene.meta.body_parents||[];if(B)for(let b=1;b<par.length;b++){jp[nj*3]=B[b*3];jp[nj*3+1]=B[b*3+1];jp[nj*3+2]=B[b*3+2];nj++;const q=par[b];if(q>=1){const k=nb*6;bp.set([B[b*3],B[b*3+1],B[b*3+2],B[q*3],B[q*3+1],B[q*3+2]],k);bc.set([0,.5,.53,0,.5,.53],k);nb++;}}}
  else{const J=row(chunk,'joints',f);if(J)for(let j=0;j<22;j++){jp[nj*3]=J[j*3];jp[nj*3+1]=J[j*3+1];jp[nj*3+2]=J[j*3+2];nj++;const q=HUMAN_PARENTS[j];if(q>=0){const k=nb*6;bp.set([J[j*3],J[j*3+1],J[j*3+2],J[q*3],J[q*3+1],J[q*3+2]],k);bc.set([0,.5,.53,0,.5,.53],k);nb++;}}}}
 ctx.bones.geometry.setDrawRange(0,nb*2);ctx.bones.geometry.attributes.position.needsUpdate=true;ctx.bones.geometry.attributes.color.needsUpdate=true;ctx.bones.geometry.computeBoundingSphere();ctx.bones.visible=nb>0;
 ctx.joints.geometry.setDrawRange(0,nj);ctx.joints.geometry.attributes.position.needsUpdate=true;ctx.joints.geometry.computeBoundingSphere();ctx.joints.visible=nj>0;
 // camera
 const dark=mode==='distance'||mode==='normals';ctx.renderer.setClearColor(dark?0x1b242a:0xeaf0f4);ctx.grid.material.opacity=dark?.35:1;ctx.grid.material.transparent=dark;
 let cx=0,cy=0,cz=.8;if(robot){const r=row(chunk,'root_pos',f);if(r){cx=r[0];cy=r[1];}}else{let sx=0,sy=0;for(let i=0;i<n;i++){sx+=pos[i*3];sy+=pos[i*3+1];}cx=sx/n;cy=sy/n;}
 const view=p.cameraView,{camera,controls}=ctx;
 if(!ctx.init||ctx.prevView!==view){controls.target.set(cx,cy,view==='feet'?.3:.8);camera.up.set(0,0,1);const D=3.0;if(view==='front')camera.position.set(cx,cy-D,1.1);else if(view==='side')camera.position.set(cx+D,cy,1.1);else if(view==='top'){camera.up.set(0,1,0);camera.position.set(cx,cy,.8+D);}else if(view==='feet')camera.position.set(cx+1.25,cy-1.55,.85);else if(view==='under')camera.position.set(cx+.01,cy-.6,-1.6);else camera.position.set(cx+2.2,cy-2.8,1.7);controls.update();ctx.init=true;ctx.prevView=view;ctx.prev=[cx,cy];}
 else if(view!=='orbit'){const dx=cx-ctx.prev[0],dy=cy-ctx.prev[1];camera.position.x+=dx;camera.position.y+=dy;controls.target.x+=dx;controls.target.y+=dy;ctx.prev=[cx,cy];}
 ctx.grid.position.x=Math.round(cx);ctx.grid.position.y=Math.round(cy);
 p.onStats?.({contact:nContact,near:nNear,pen:nPen});
}
