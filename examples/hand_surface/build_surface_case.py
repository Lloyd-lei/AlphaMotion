from surface_retargeting.config import CFG as SRCFG  # machine paths: configs/sr.toml
from pathlib import Path
import os
os.environ['MUJOCO_GL']='egl';os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2'
import json,hashlib,math
import numpy as np,mujoco,trimesh
from scipy.optimize import brentq
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(f'{SRCFG.research_root}/alpha_motion2/research/hand_surface_case_20261004')
OUT=ROOT/'derived';OUT.mkdir(exist_ok=True)
PANDA=Path(f'{SRCFG.research_root}/alpha_motion2/research/embodied_case_20261003/geometry/assets/franka_emika_panda/hand.xml')
RADIUS=.03;THRESHOLD=.002

def dump(path,obj):path.write_text(json.dumps(obj,indent=2)+'\n')
def visual_geoms(m):return [g for g in range(m.ngeom) if m.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH and m.geom_contype[g]==0 and m.geom_conaffinity[g]==0]
def mesh(m,g):
 k=int(m.geom_dataid[g]);v=m.mesh_vert[m.mesh_vertadr[k]:m.mesh_vertadr[k]+m.mesh_vertnum[k]].astype(float);f=m.mesh_face[m.mesh_faceadr[k]:m.mesh_faceadr[k]+m.mesh_facenum[k]].copy();return k,v,f

def closest_record(m,d,g,center):
 k,v,f=mesh(m,g);rg=d.geom_xmat[g].reshape(3,3);pg=d.geom_xpos[g];local=(center-pg)@rg
 point,dist,tri=trimesh.proximity.closest_point_naive(trimesh.Trimesh(v,f,process=False),local[None]);pt=point[0];idx=int(tri[0]);bc=trimesh.triangles.points_to_barycentric(v[f[idx]][None],pt[None])[0]
 return float(dist[0]-RADIUS),idx,bc,pt@rg.T+pg

def camera_meta(scene,w,h):
 a,b=scene.camera;pos=(a.pos.astype(float)+b.pos)/2;fw=(a.forward.astype(float)+b.forward)/2;fw/=np.linalg.norm(fw);up=(a.up.astype(float)+b.up)/2;up/=np.linalg.norm(up);right=np.cross(fw,up);right/=np.linalg.norm(right);up=np.cross(right,fw)
 return dict(position_object_m=pos.tolist(),forward=fw.tolist(),up=up.tolist(),right=right.tolist(),frustum_top=float(a.frustum_top),frustum_bottom=float(a.frustum_bottom),frustum_near=float(a.frustum_near),frustum_center=float(a.frustum_center),width=w,height=h,pixel_origin='top_left')
def project(p,c):
 p=np.asarray(p)-c['position_object_m'];x=p@np.array(c['right']);y=p@np.array(c['up']);z=p@np.array(c['forward']);hh=(c['frustum_top']-c['frustum_bottom'])/2;hw=hh*c['width']/c['height'];sx=x*c['frustum_near']/z;sy=y*c['frustum_near']/z
 return np.stack([(sx-c['frustum_center'])/hw*c['width']/2+c['width']/2,c['height']/2-(sy-(c['frustum_top']+c['frustum_bottom'])/2)/hh*c['height']/2],-1)
def arrow(dr,s,e,color,width=3):
 s=np.array(s);e=np.array(e);dr.line([tuple(s),tuple(e)],fill=color,width=width);v=e-s;n=np.linalg.norm(v)
 if n>2:
  v/=n;p=np.array([-v[1],v[0]]);dr.polygon([tuple(e),tuple(e-v*10+p*4),tuple(e-v*10-p*4)],fill=color)
def render(m,d,center,label,samples,selected):
 w=h=1100;m.vis.global_.offwidth=w;m.vis.global_.offheight=h;m.light_active[:]=0;m.vis.headlight.ambient[:]=.4;m.vis.headlight.diffuse[:]=.6;m.vis.headlight.specular[:]=.08
 gids=visual_geoms(m)
 for g in range(m.ngeom):m.geom_group[g]=1 if g in gids else 5
 opt=mujoco.MjvOption();opt.geomgroup[:]=0;opt.geomgroup[1]=1
 P=d.xpos[1:]-center;vertices=[]
 for g in gids:
  k,v,f=mesh(m,g);vertices.append(v@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]-center)
 bound=np.concatenate(vertices);lo=bound.min(0);hi=bound.max(0)
 cam=mujoco.MjvCamera();cam.lookat[:]=(lo+hi)/2;cam.distance=float(np.linalg.norm(hi-lo))*1.55;cam.azimuth=125 if label=='allegro' else 140;cam.elevation=-28
 ren=mujoco.Renderer(m,height=h,width=w);ren.update_scene(d,camera=cam,scene_option=opt)
 for i in range(ren.scene.ngeom):ren.scene.geoms[i].pos[:]-=center
 ren.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=0
 c=camera_meta(ren.scene,w,h)
 def capture(prefix):
  rgb=ren.render().copy();ren.enable_segmentation_rendering();sg=ren.render().copy();ren.disable_segmentation_rendering();mask=sg[:,:,0]>=0
  rgba=np.dstack([rgb,mask.astype(np.uint8)*255]);rgba[~mask,:3]=255
  Image.fromarray(rgba).save(OUT/(prefix+'_transparent.png'));white=rgba.copy();white[~mask]=255;Image.fromarray(white).convert('RGB').save(OUT/(prefix+'.png'));return Image.fromarray(rgba)
 body=capture(label+'_mesh')
 # Analytic object is independent of the robot source model, no dynamics step.
 idx=ren.scene.ngeom;ren.scene.ngeom+=1;geom=ren.scene.geoms[idx]
 mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_SPHERE,np.array([RADIUS,0,0]),np.zeros(3),np.eye(3).ravel(),np.array([.20,.60,.87,.42],np.float32));geom.objtype=int(mujoco.mjtObj.mjOBJ_UNKNOWN);geom.objid=999;geom.segid=idx
 scene=capture(label+'_sphere_mesh')
 xy=project(P,c);overlay=Image.new('RGBA',(w,h),(255,255,255,0));dr=ImageDraw.Draw(overlay)
 for b in range(1,m.nbody):
  pa=int(m.body_parentid[b]);pt=xy[b-1]
  if pa>0:dr.line([tuple(xy[pa-1]),tuple(pt)],fill=(28,133,166,225),width=4)
  dr.ellipse([pt[0]-4,pt[1]-4,pt[0]+4,pt[1]+4],fill=(24,116,153,255),outline='white')
 ax=project(np.vstack([np.zeros(3),np.eye(3)*.045]),c)
 for k,col in enumerate([(217,72,60,255),(48,149,78,255),(58,94,207,255)]):arrow(dr,ax[0],ax[k+1],col,4);dr.text(tuple(ax[k+1]+[3,3]),'XYZ'[k],fill=col,font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20))
 overlay.save(OUT/(label+'_skeleton_transparent.png'));Image.alpha_composite(Image.new('RGBA',(w,h),'white'),overlay).save(OUT/(label+'_skeleton.png'))
 comp=Image.alpha_composite(scene,overlay);comp.save(OUT/(label+'_mesh_skeleton_transparent.png'));Image.alpha_composite(Image.new('RGBA',(w,h),'white'),comp).convert('RGB').save(OUT/(label+'_mesh_skeleton.png'))
 pts=project(samples['position_object_m'],c);samples['camera_pixel']=pts
 pov=Image.new('RGBA',(w,h),(255,255,255,0));pd=ImageDraw.Draw(pov)
 # Displayed samples are transparent overlay markers, not occlusion-correct depth evidence.
 for i in range(0,len(pts),max(1,len(pts)//1000)):
  x,y=pts[i];pd.ellipse([x-2,y-2,x+2,y+2],fill=(35,132,153,145))
 for i in np.flatnonzero(samples['contact_geometry']):
  x,y=pts[i];pd.ellipse([x-3,y-3,x+3,y+3],fill=(232,124,39,240))
 px=[]
 for n,r in enumerate(selected):
  s=np.array(r['position_object_m']);o=np.array(r['object_nearest_point_m']);nrm=np.array(r['normal_object']);a=project(np.vstack([s,o,s+nrm*.012]),c);r.update(camera_pixel=a[0].tolist(),object_nearest_point_pixel=a[1].tolist(),normal_endpoint_pixel=a[2].tolist())
  x,y=a[0];pd.ellipse([x-7,y-7,x+7,y+7],fill=(248,113,44,255),outline='white',width=2);arrow(pd,a[0],a[2],(171,71,181,255),3)
  px.append(a[0])
  crop=[int(x)-130,int(y)-130,int(x)+130,int(y)+130];r['crop_source_box_xyxy']=crop;r['crop_anchor_pixel']=[2*(x-crop[0]),2*(y-crop[1])]
  cropped=Image.alpha_composite(Image.alpha_composite(Image.new('RGBA',(w,h),'white'),scene),pov).crop(crop).resize((520,520),Image.Resampling.LANCZOS);cropped.save(OUT/(label+'_contact_'+str(n)+'_crop.png'))
 points=Image.alpha_composite(scene,pov);points.save(OUT/(label+'_surface_samples_transparent.png'));Image.alpha_composite(Image.new('RGBA',(w,h),'white'),points).convert('RGB').save(OUT/(label+'_surface_samples.png'))
 joints=[]
 for j in range(m.njnt):
  anchor=d.xanchor[j]-center;axis=d.xaxis[j];xyj=project(np.vstack([anchor,anchor+axis*.02]),c)
  joints.append(dict(name=m.joint(j).name,body=m.body(int(m.jnt_bodyid[j])).name,type='hinge' if m.jnt_type[j]==mujoco.mjtJoint.mjJNT_HINGE else 'slide',dof=1,q=float(d.qpos[m.jnt_qposadr[j]]),q_unit='rad' if m.jnt_type[j]==mujoco.mjtJoint.mjJNT_HINGE else 'm',limits=m.jnt_range[j].tolist(),axis_link_local=m.jnt_axis[j].tolist(),axis_object=axis.tolist(),anchor_object_m=anchor.tolist(),camera_pixel=xyj[0].tolist(),axis_endpoint_pixel=xyj[1].tolist()))
 ren.close();return dict(camera=c,joints=joints,skeleton_nodes=[dict(name=m.body(b).name,parent=m.body(int(m.body_parentid[b])).name,position_object_m=P[b-1].tolist(),camera_pixel=xy[b-1].tolist()) for b in range(1,m.nbody)],object_frame_axis_pixels=ax.tolist(),selected_surface_points=selected)

def run(label,xml,q,center):
 m=mujoco.MjModel.from_xml_path(str(xml));d=mujoco.MjData(m);d.qpos[:]=q;mujoco.mj_forward(m,d)
 gids=visual_geoms(m);arrs={k:[] for k in ['position_object_m','normal_object','position_link_local_m','normal_link_local','geom_id','triangle_index','barycentric','body_id']};selected=[];exact=[];bindings=[]
 for g in gids:
  k,v,f=mesh(m,g);tri=v[f];n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);areas=np.linalg.norm(n,axis=1);valid=areas>1e-14;rng=np.random.default_rng(20261004+g);idx=rng.choice(len(f),size=1600,p=areas/areas.sum());uv=rng.random((1600,2));a=np.sqrt(uv[:,0]);bary=np.stack([1-a,a*(1-uv[:,1]),a*uv[:,1]],axis=-1);local=np.einsum('ni,nij->nj',bary,tri[idx]);norm=n[idx]/areas[idx,None];rg=d.geom_xmat[g].reshape(3,3);pg=d.geom_xpos[g];world=local@rg.T+pg;bodyid=int(m.geom_bodyid[g]);rb=d.xmat[bodyid].reshape(3,3);pb=d.xpos[bodyid];obj=world-center
  vals={'position_object_m':obj,'normal_object':norm@rg.T,'position_link_local_m':(world-pb)@rb,'normal_link_local':(norm@rg.T)@rb,'geom_id':np.full(1600,g),'triangle_index':idx,'barycentric':bary,'body_id':np.full(1600,bodyid)}
  for n2,x in vals.items():arrs[n2].append(x)
  gap,ti,bc,pt=closest_record(m,d,g,center);name=m.body(bodyid).name;exact.append(dict(geom_id=g,body=name,closest_triangle=ti,minimum_surface_gap_m=gap))
  bindings.append(dict(geom_id=g,body_id=bodyid,body_name=name,mesh_id=k,mesh_name=m.mesh(k).name,triangle_count=len(f),vertex_count=len(v),triangle_index_basis='compiled MuJoCo mesh local faces',seed=20261004+g,samples=1600))
 arrays={k:np.concatenate(v) for k,v in arrs.items()};norms=np.linalg.norm(arrays['position_object_m'],axis=1);objnear=arrays['position_object_m']/norms[:,None]*RADIUS;gap=norms-RADIUS;arrays.update(object_nearest_point_m=objnear,d_body_minus_object_m=arrays['position_object_m']-objnear,signed_gap_m=gap,distance_m=np.abs(gap),valid=np.isfinite(gap)&(norms>1e-12),contact_geometry=np.abs(gap)<=THRESHOLD,penetration=gap<0,penetration_depth_m=np.maximum(-gap,0))
 wanted=['ff_tip','mf_tip','rf_tip','th_tip'] if label=='allegro' else ['left_finger','right_finger']
 for name in wanted:
  bid=m.body(name).id;ii=np.flatnonzero(arrays['body_id']==bid);i=ii[np.argmin(abs(arrays['signed_gap_m'][ii]-.0009))];gid=int(arrays['geom_id'][i]);k,v,f=mesh(m,gid)
  r=dict(sample_index=int(i),body=name,geom_id=gid,mesh_name=m.mesh(k).name,triangle_index=int(arrays['triangle_index'][i]),triangle_vertex_indices=f[arrays['triangle_index'][i]].tolist(),barycentric=arrays['barycentric'][i].tolist(),classification='derived from constructed geometric illustration',distance_m=float(arrays['distance_m'][i]),signed_gap_m=float(gap[i]),contact_geometry=bool(arrays['contact_geometry'][i]),valid=bool(arrays['valid'][i]),penetration=bool(gap[i]<0),penetration_depth_m=float(max(-gap[i],0)),threshold_m=THRESHOLD)
  for key in ['position_object_m','normal_object','position_link_local_m','normal_link_local','object_nearest_point_m','d_body_minus_object_m']:r[key]=arrays[key][i].tolist()
  r['position_world_model_m']=(arrays['position_object_m'][i]+center).tolist();selected.append(r)
 binding_arrays={}
 for g in gids:
  mid,v,f=mesh(m,g);bid=int(m.geom_bodyid[g]);binding_arrays.update({f'g{g}_vertices':v,f'g{g}_faces':f,f'g{g}_rotation_to_world':d.geom_xmat[g].reshape(3,3).copy(),f'g{g}_translation_world':d.geom_xpos[g].copy(),f'b{bid}_rotation_to_world':d.xmat[bid].reshape(3,3).copy(),f'b{bid}_translation_world':d.xpos[bid].copy()})
 np.savez_compressed(OUT/(label+'_compiled_mesh_binding.npz'),**binding_arrays)
 trace=render(m,d,center,label,arrays,selected)
 np.savez_compressed(OUT/(label+'_surface_samples.npz'),**arrays)
 trace.update(model=label,source_xml=str(xml),source_xml_sha256=hashlib.sha256(xml.read_bytes()).hexdigest(),model_nq=m.nq,model_nv=m.nv,pose_qpos=np.asarray(q).tolist(),pose_classification='constructed geometric illustration',object={'primitive':'sphere','radius_m':RADIUS,'object_to_model_world_translation_m':np.asarray(center).tolist(),'object_to_model_world_rotation':np.eye(3).tolist(),'object_center_object_m':[0,0,0],'frame':'same right-handed XYZ object coordinates for both bodies'},sampling={'method':'deterministic area-weighted triangle sampling with barycentric coordinates','bindings':bindings,'total_samples':len(gap),'valid_count':int(arrays['valid'].sum()),'geometric_contact_count':int(arrays['contact_geometry'].sum()),'threshold_m':THRESHOLD,'binary_rule':'valid AND abs(norm(p_body_object)-radius)<=threshold','d_convention':'d = p_body_surface - p_object_nearest; vector points from object to body; not force','normal_provenance':'actual triangle winding, transformed from compiled mesh to link and object frame','overlay_visibility':'surface markers and skeleton are visible overlays, not occlusion tests'},exact_mesh_sphere_minima=exact,minimum_exact_surface_gap_m=min(x['minimum_surface_gap_m'] for x in exact),limits=['No training, model prediction, UMR optimization, dynamic simulation, grasp-success test or tactile data.','Geometric proximity labels are thresholded features, not ground-truth force/contact measurements.','Joint limits enforced; self-collision, force closure, torque feasibility and friction are not established.'])
 dump(OUT/(label+'_trace.json'),trace)
 print(label, json.dumps({k:trace[k] for k in ['model_nq','minimum_exact_surface_gap_m']}), 'contact samples',trace['sampling']['geometric_contact_count'])
 return trace
pose=json.loads((ROOT/'constructed_pose.json').read_text());run('allegro',ROOT/'source_assets/wonik_allegro/right_hand.xml',pose['qpos'],np.array(pose['center']))
m=mujoco.MjModel.from_xml_path(str(PANDA));d=mujoco.MjData(m);center=np.array([0,0,.103])
def panda_gap(q):
 d.qpos[:]=q;mujoco.mj_forward(m,d);return min(closest_record(m,d,g,center)[0] for g in visual_geoms(m) if m.body(int(m.geom_bodyid[g])).name=='left_finger')-.0006
q=brentq(panda_gap,.025,.04);run('panda',PANDA,[q,q],center)

# Clarify constrained coordinate and control counts from the actual Panda XML.
import xml.etree.ElementTree as ET
pj=json.loads((OUT/'panda_trace.json').read_text());pm=mujoco.MjModel.from_xml_path(str(PANDA));pd=mujoco.MjData(pm);pd.qpos[:]=pj['pose_qpos'];mujoco.mj_forward(pm,pd)
def ray_first(v,f,origin,direction):
 tri=v[f];e1=tri[:,1]-tri[:,0];e2=tri[:,2]-tri[:,0];h=np.cross(np.broadcast_to(direction,e2.shape),e2);a=np.einsum('ij,ij->i',e1,h);ok=abs(a)>1e-12;inv=np.zeros_like(a);inv[ok]=1/a[ok];s=origin-tri[:,0];u=inv*np.einsum('ij,ij->i',s,h);cross=np.cross(s,e1);vv=inv*(cross@direction);tt=inv*np.einsum('ij,ij->i',e2,cross);mask=ok&(u>=-1e-10)&(vv>=-1e-10)&(u+vv<=1+1e-10)&(tt>0);return float(tt[mask].min()) if mask.any() else None
pads={}
for g in visual_geoms(pm):
 mid,v,f=mesh(pm,g);name=pm.body(int(pm.geom_bodyid[g])).name
 if name in ['left_finger','right_finger'] and pm.mesh(mid).name=='finger_1':
  world=v@pd.geom_xmat[g].reshape(3,3).T+pd.geom_xpos[g];direction=np.array([0.,-1.,0.]) if name=='left_finger' else np.array([0.,1.,0.]);pads[name]=ray_first(world,f,center,direction)
e=ET.parse(PANDA).getroot();pj['kinematic_counts']={'J_scalar_joints':2,'D_independent_opening_dof':1,'nq_unconstrained':int(pm.nq),'nv_unconstrained':int(pm.nv),'actuator_control_count':int(pm.nu),'equality_constraint_count':int(pm.neq),'relation':'finger_joint1 = finger_joint2; both prismatic. One symmetric opening degree of freedom.'}
pj['coupling']={'joints':[{'name':pm.joint(j).name,'type':'prismatic','q_m':float(pd.qpos[pm.jnt_qposadr[j]]),'range_m':pm.jnt_range[j].tolist()} for j in range(pm.njnt)],'equality_xml':ET.tostring(e.find('equality'),encoding='unicode'),'equality_polynomial':pm.eq_data[0,:5].tolist(),'tendon_xml':ET.tostring(e.find('tendon'),encoding='unicode'),'actuator_xml':ET.tostring(e.find('actuator'),encoding='unicode'),'tendon_coordinate_m':float(pd.ten_length[0]),'xml_mimic_element':False,'mimic_note':'MJCF uses equality and fixed tendon, no URDF mimic element in this loaded asset.','joint_coordinate_opening_q1_plus_q2_m':sum(pj['pose_qpos']),'visual_inner_pad_aperture_at_sphere_center_m':sum(pads.values()),'visual_aperture_method':'Ray/triangle intersections from sphere center along +/- object Y, using actual finger_1 visual pad meshes.','inner_pad_half_apertures_m':pads,'actuator_control_range':pm.actuator_ctrlrange[0].tolist(),'actuator_command_applied':False,'force_or_tactile_measurement':False}
dump(OUT/'panda_trace.json',pj)
aj=json.loads((OUT/'allegro_trace.json').read_text());aj['kinematic_counts']={'J_scalar_joints':16,'D_independent_joint_dof':16,'actuator_control_count':16,'equality_constraint_count':0,'q_unit':'rad','fingers':4,'model':'Allegro Hand V3 right'};dump(OUT/'allegro_trace.json',aj)
# Full Panda context: provided arm home posture, constructed symmetric finger opening.
full=PANDA.with_name('panda.xml');fm=mujoco.MjModel.from_xml_path(str(full));fd=mujoco.MjData(fm);fd.qpos[:]=fm.key_qpos[0];fd.qpos[-2:]=q;mujoco.mj_forward(fm,fd);hand=fm.body('hand').id;fc=fd.xpos[hand]+fd.xmat[hand].reshape(3,3)@np.array([0,0,.103]);empty={'position_object_m':np.zeros((0,3)),'contact_geometry':np.zeros(0,dtype=bool)}
ctx=render(fm,fd,fc,'panda_arm_context',empty,[]);ctx.update(source_xml=str(full),pose_qpos=fd.qpos.tolist(),classification='reference arm home keyframe with constructed finger opening; kinematic illustration only',sphere_radius_m=RADIUS,object_to_world_translation_m=fc.tolist(),object_to_world_rotation=np.eye(3).tolist(),relationship='Same Panda hand model and finger q; arm context orientation differs from standalone contact comparison. No neural or physical execution.');dump(OUT/'panda_arm_context_trace.json',ctx)
print('panda coupling',json.dumps(pj['coupling']))
