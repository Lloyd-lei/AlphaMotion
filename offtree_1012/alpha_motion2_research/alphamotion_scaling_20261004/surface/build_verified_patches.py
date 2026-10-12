#!/usr/bin/env python3
"""Read-only remote asset compiler. Emits small derived patches as JSON on stdout.

No mesh download, source mutation, renderer, GPU, training or installation. Verifies
cached skeleton names/parents/offsets, random-q FK, and a rigid geometry-to-cached-
frame transform across random joint configurations. A name match alone is insufficient.
Run with existing ardy Python and -B. Patch source faces use compiled MuJoCo ordering
and record compiler version/compiled-mesh digest as well as original asset hashes.
"""
from pathlib import Path
import os,sys,json,hashlib,argparse,xml.etree.ElementTree as ET
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
sys.dont_write_bytecode=True
import numpy as np
import mujoco
AX=np.array([[0.,1,0],[0,0,1],[1,0,0]],np.float64)
BASE=Path('/home/arenalabs/Desktop')/'"be water, robot"'
SRC=BASE/'pose_translator_design/exp_topo'
CACHE=Path('/var/tmp/alphamotion/topo_exp/cache_lafan_v3')
sys.path.insert(0,str(SRC))

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest_arrays(*arrays):
 h=hashlib.sha256()
 for a in arrays:h.update(np.ascontiguousarray(a).tobytes())
 return h.hexdigest()
def jsonable(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,(np.integer,np.floating)):return x.item()
 raise TypeError(type(x).__name__)

def mesh_sources(xml):
 records={};seen=set();xmls=[];documents=[]
 def visit(path):
  path=Path(path).resolve()
  if path in seen:return
  seen.add(path);xmls.append({'path':str(path),'sha256':sha(path)})
  tree=ET.parse(path).getroot();documents.append((path,tree))
  for inc in tree.findall('.//include'):
   path2=path.parent/inc.attrib['file']
   if not path2.is_file():path2=Path(xml).parent/inc.attrib['file']
   visit(path2)
 visit(xml)
 # MuJoCo merges compiler declarations from included XMLs before loading assets.
 dirs=[(path,c.get('meshdir','')) for path,tree in documents for c in tree.findall('.//compiler') if c.get('meshdir')]
 for path,tree in documents:
  for mesh in tree.findall('.//asset/mesh'):
   f=mesh.get('file')
   if not f:continue
   candidates=[Path(f)] if Path(f).is_absolute() else []
   candidates += [base/d/f for cpath,d in reversed(dirs) for base in (Path(xml).parent,cpath.parent,path.parent)]
   candidates += [path.parent/f,Path(xml).parent/f]
   found=next((p.resolve() for p in candidates if p.is_file()),None)
   if found is None:raise ValueError('unresolved original mesh '+f)
   name=mesh.get('name',Path(f).stem)
   records[name]={'path':str(found),'sha256':sha(found),'mesh_declared_scale':mesh.get('scale','1 1 1')}
 return records,xmls

def random_states(model,count=7):
 rng=np.random.default_rng(20261004);states=[model.qpos0.copy()]
 for _ in range(count):
  q=model.qpos0.copy()
  for j in range(model.njnt):
   adr=int(model.jnt_qposadr[j]);typ=int(model.jnt_type[j])
   if typ==int(mujoco.mjtJoint.mjJNT_HINGE):
    if model.jnt_limited[j]:
     lo,hi=model.jnt_range[j];q[adr]=rng.uniform(lo+.08*(hi-lo),hi-.08*(hi-lo))
    else:q[adr]=rng.uniform(-.9,.9)
   elif typ==int(mujoco.mjtJoint.mjJNT_SLIDE):
    lo,hi=model.jnt_range[j] if model.jnt_limited[j] else (-.03,.03)
    q[adr]=rng.uniform(lo,hi)
   elif typ==int(mujoco.mjtJoint.mjJNT_FREE):
    q[adr:adr+3]+=rng.normal(0,.1,3);quat=rng.normal(size=4);q[adr+3:adr+7]=quat/np.linalg.norm(quat)
   elif typ==int(mujoco.mjtJoint.mjJNT_BALL):
    quat=rng.normal(size=4);q[adr:adr+4]=quat/np.linalg.norm(quat)
  states.append(q)
 return states

def role_text(body,name):
 plain=name.replace('_',' ').replace(':',' ')
 low=name.lower()
 if any(k in low for k in ['thumb','index','middle','ring','little','pinky','finger']) or any(low.startswith(k) for k in ['ff','mf','rf','lf','thbase','thproximal','thhub','thmiddle','thdistal']):role='a hand digit or articulated hand link when identified by the asset name; no tactile state is implied'
 elif any(k in low for k in ['wrist','hand','palm']):role='a wrist or terminal hand region of the articulated upper limb; surface contact is unobserved'
 elif any(k in low for k in ['elbow','forearm','shoulder','arm']):role='an upper-limb mechanical segment linking the torso and distal end-effector region'
 elif any(k in low for k in ['ankle','foot','toe','knee','hip','thigh','calf','leg']):role='an articulated leg or distal foot region; current ground contact is unobserved'
 elif any(k in low for k in ['pelvis','torso','waist','base']):role='a central body or support-link region'
 else:role='the named rigid mechanical link, with no additional functional label inferred'
 return f'Robot {body}. Surface on {plain}. Local function: {role}. This is a geometric patch rigidly attached to its verified joint frame, not a pressure measurement.'

def source_adapter(body,adroit_meta=None):
 if adroit_meta:
  meta=json.loads(Path(adroit_meta).read_text());mm=meta['model']
  xml=Path(mm['model_xml']);keep=np.asarray(mm['body_ids'],int);frame=keep.copy()
  model=mujoco.MjModel.from_xml_path(str(xml));rest=np.asarray(mm['rest_body_rotations_world_zup'],np.float64)
  jt=np.ones(len(keep),int);jt[0]=4;axis=np.zeros((len(keep),3));lo=np.zeros(len(keep));hi=lo.copy();ref=lo.copy();qadr=np.full(len(keep),-1,int)
  for i,bid in enumerate(keep):
   for jid in np.flatnonzero(model.jnt_bodyid==bid):
    typ=int(model.jnt_type[jid]);jt[i]=3 if typ==int(mujoco.mjtJoint.mjJNT_SLIDE) else 2
    if typ==int(mujoco.mjtJoint.mjJNT_SLIDE):
     if not model.jnt_limited[jid]:raise ValueError('finite true slide limits required')
     axis[i]=AX@rest[i]@model.jnt_axis[jid];lo[i],hi[i]=model.jnt_range[jid];qadr[i]=int(model.jnt_qposadr[jid]);ref[i]=model.qpos0[qadr[i]]
  ancestor=int(keep[0]);root_fixed=True
  while ancestor>0:
   root_fixed=root_fixed and int(model.body_jntnum[ancestor])==0;ancestor=int(model.body_parentid[ancestor])
  return {'body':body,'xml':xml,'model':model,'keep':keep,'frame':frame,
   'names':list(mm['joint_names']),'parents':np.asarray(mm['parents'],int),'offsets':np.asarray(mm['rest_offsets_m'],np.float64),
   'unit_factor':1.,'meters_per_unit':1.,'rest_rebase':np.asarray(mm['rest_body_rotations_world_zup'],np.float64),
   'spec_sha256':sha(adroit_meta),'source_meta':str(adroit_meta),'kinematic_unit':'m','kind':'external_rest_rebased',
   'reach':float(mm['reach_m']),'joint_type':jt,'prismatic_axis':axis,'slide_lower':lo,'slide_upper':hi,'slide_reference':ref,'slide_qpos_addresses':qadr,
   'kinematic_fixed_rotation':np.tile(np.eye(3),(len(keep),1,1)),'root_fixed':root_fixed}
 import prep_xwbc_cache as PX,prep_gmr_cache as PG
 xml=Path(PX.resolve_xml(body)).resolve();model=mujoco.MjModel.from_xml_path(str(xml));keep,frame=PG.build_merge(model)
 rebuilt=PG.build_spec(model,body,keep);specpath=CACHE/f'{body}_spec.npz'
 with np.load(specpath,allow_pickle=True) as z:
  names=[str(x) for x in z['joint_names']];parents=z['parents'];off=z['rest_offsets']
 if names!=list(rebuilt.joint_names) or not np.array_equal(parents,rebuilt.parents):raise ValueError('cached topology/name order differs from source model')
 err=float(np.max(np.abs(off[1:]-rebuilt.rest_offsets[1:])))
 if err>1e-4:raise ValueError(f'cached nonroot offsets mismatch {err}cm')
 path=np.zeros(len(parents))
 for j,p in enumerate(parents):
  if p>=0:path[j]=path[p]+np.linalg.norm(off[j])
 return dict(body=body,xml=xml,model=model,keep=np.asarray(keep,int),frame=np.asarray([frame[b] for b in keep],int),names=names,
  parents=parents,offsets=off,unit_factor=100.,meters_per_unit=.01,rest_rebase=None,spec_sha256=sha(specpath),source_meta=str(specpath),
  kinematic_unit='cm',kind='xwbc_merged_frame',reach=float(path.max()),rebuilt_nonroot_offset_max_error_cm=err)

def build_body(adapter,patches=96):
 model=adapter['model'];xml=adapter['xml'];body=adapter['body'];keep=adapter['keep'];frame=adapter['frame'];joints=len(keep)
 slides={int(model.jnt_bodyid[j]) for j in range(model.njnt) if model.jnt_type[j]==mujoco.mjtJoint.mjJNT_SLIDE}
 supported_slides=slides.intersection(set(keep.tolist()))
 if supported_slides and 'joint_type' not in adapter:raise ValueError('kept prismatic joint requires verified explicit axis/limits contract')
 excluded_slides=slides-supported_slides
 sources,xmls=mesh_sources(xml);data=mujoco.MjData(model);states=random_states(model);snapshots=[]
 for q in states:
  data.qpos[:]=q;mujoco.mj_forward(model,data)
  r=data.xmat[frame].reshape(-1,3,3).copy()
  if adapter['rest_rebase'] is not None:r=r@adapter['rest_rebase'].transpose(0,2,1)
  r=AX[None]@r@AX.T[None];p=(data.xpos[keep]-data.xpos[keep[0]])@AX.T*adapter['unit_factor']
  gp=(data.geom_xpos-data.xpos[keep[0]])@AX.T*adapter['unit_factor'];gr=AX[None]@data.geom_xmat.reshape(-1,3,3)@AX.T[None]
  fk=np.zeros_like(p)
  slideq=np.zeros(joints)
  if 'slide_qpos_addresses' in adapter:
   sm=adapter['slide_qpos_addresses']>=0;slideq[sm]=q[adapter['slide_qpos_addresses'][sm]]*adapter['unit_factor']
  for n,par in enumerate(adapter['parents']):
   if par>=0:
    translated=adapter['offsets'][n].copy()
    if n<len(slideq) and 'prismatic_axis' in adapter:translated=translated+adapter['prismatic_axis'][n]*(slideq[n]-adapter['slide_reference'][n])
    fk[n]=fk[par]+r[par]@translated
    if 'joint_type' in adapter and adapter['joint_type'][n] in (1,3) and np.max(np.abs(r[n]-r[par]@adapter['kinematic_fixed_rotation'][n]))>1e-7:raise ValueError('derived fixed/slide rotation mismatch')
  if adapter.get('root_fixed') and np.max(np.abs(r[0]-np.eye(3)))>1e-7:raise ValueError('fixed-root rebase mismatch')
  err=float(np.max(np.linalg.norm(fk-p,axis=-1)))
  snapshots.append({'r':r,'p':p,'gp':gp,'gr':gr,'fk_error':err,'slide_native':slideq,'qpos':q.copy()})
 maxfk=max(s['fk_error'] for s in snapshots)
 if maxfk>max(1e-6,adapter['unit_factor']*1e-5):raise ValueError(f'random-q cached-frame FK mismatch {maxfk} {adapter["kinematic_unit"]}')
 visual=[g for g in range(model.ngeom) if model.geom_bodyid[g]>0 and model.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH and model.geom_contype[g]==0 and model.geom_conaffinity[g]==0]
 mesh_scope='visual_mesh'
 if not visual:
  visual=[g for g in range(model.ngeom) if model.geom_bodyid[g]>0 and model.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH];mesh_scope='declared_mesh_geometry_no_visual_collision_split'
 eligible=[];omitted=[]
 for gid in visual:
  bid=int(model.geom_bodyid[gid])
  ancestry=[];cursor=bid
  while cursor>0:ancestry.append(cursor);cursor=int(model.body_parentid[cursor])
  if excluded_slides.intersection(ancestry):omitted.append({'geom':gid,'reason':'excluded_prismatic_body_or_descendant_unsupported'});continue
  # Excludes task object/target/world for external hand; never binds them to the robot.
  ancestor=bid
  while ancestor not in set(keep.tolist()) and ancestor>0:ancestor=int(model.body_parentid[ancestor])
  if ancestor==0:omitted.append({'geom':gid,'reason':'outside_robot_subtree'});continue
  candidates=[]
  for anchor in range(joints):
   mats=[s['r'][anchor].T@s['gr'][gid] for s in snapshots]
   trans=[s['r'][anchor].T@(s['gp'][gid]-s['p'][anchor]) for s in snapshots]
   er=max(float(np.max(np.abs(m-mats[0]))) for m in mats);ep=max(float(np.linalg.norm(t-trans[0])) for t in trans)
   if er<1e-7 and ep<adapter['unit_factor']*1e-6:candidates.append((np.linalg.norm(trans[0]),anchor,mats[0],trans[0],er,ep))
  if not candidates:omitted.append({'geom':gid,'reason':'no_rigid_cached_frame_across_random_q'});continue
  candidates.sort(key=lambda t:(0 if keep[t[1]]==bid else 1,t[0]));_,anchor,mat,trans,er,ep=candidates[0]
  mid=int(model.geom_dataid[gid]);mn=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_MESH,mid)
  if mn not in sources:raise ValueError('compiled mesh lacks original source evidence: '+str(mn))
  va=int(model.mesh_vertadr[mid]);nv=int(model.mesh_vertnum[mid]);fa=int(model.mesh_faceadr[mid]);nf=int(model.mesh_facenum[mid])
  vertices=model.mesh_vert[va:va+nv].astype(np.float64);faces=model.mesh_face[fa:fa+nf].astype(np.int64)
  tri=vertices[faces];cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);area=np.linalg.norm(cross,axis=-1)/2
  good=np.flatnonzero(area>1e-14)
  if not len(good):omitted.append({'geom':gid,'reason':'zero_area_mesh'});continue
  eligible.append(dict(gid=gid,mid=mid,mesh_name=mn,source=sources[mn],vertices=vertices,faces=faces,tri=tri,
   area=area,good=good,total_area=float(area[good].sum()),anchor=anchor,mat=mat,trans=trans,rot_error=er,pos_error=ep,
   compiled_mesh_sha256=digest_arrays(vertices,faces)))
 if not eligible:raise ValueError('no verified robot mesh surface')
 if len(eligible)>patches:raise ValueError('more geoms than patch budget; explicit allocation policy required')
 allocation=np.ones(len(eligible),int);remaining=patches-len(eligible);shares=np.array([e['total_area'] for e in eligible]);shares=remaining*shares/shares.sum();extra=np.floor(shares).astype(int);allocation+=extra
 for i in np.argsort(-(shares-extra))[:remaining-int(extra.sum())]:allocation[i]+=1
 out=dict(schema_version='stable_surface_v1',body=body,canonical_up='Y',scale_policy='geometry_only',binding_kind='rigid',evidence_kind='rigid_mjcf_mesh',
  joint_motion_kind='mixed_tree' if supported_slides else 'rotational_tree',length_unit=adapter['kinematic_unit'],geometry_scale=adapter['reach'],pose_scale=adapter['reach'],metric_valid=True,meters_per_unit=adapter['meters_per_unit'],
  patch_ids=[],valid=[],rest_xyz=[],rest_normal=[],area=[],anchor_joint=[],local_xyz=[],local_normal=[],semantic_valid=[],semantic_text_key=[],patch_sources=[],
  geometry_hashes={'model_xml':sha(xml),'kinematic_spec':adapter['spec_sha256']},
  normalization_provenance='maximum cumulative non-root rest-offset path length; geometry only, no motion-fit or heldout statistics',
  frame_provenance={'kind':adapter['kind'],'formula':'AX R(q) Rrest.T AX.T' if adapter['rest_rebase'] is not None else 'AX R(frame(q)) AX.T','AX':AX.tolist(),'joint_ids_kept':keep.tolist(),'rotation_frame_body_ids':frame.tolist(),'source':adapter['source_meta']})
 texts={};area_coverage=0
 for e,count in zip(eligible,allocation):
  good=e['good'];cum=np.cumsum(e['area'][good]);selected=good[np.minimum(np.searchsorted(cum,(np.arange(count)+.5)*cum[-1]/count),len(good)-1)]
  # Duplicate large triangles can occur in area quantiles. Different deterministic barycentric samples keep IDs unique.
  for occurrence,faceid in enumerate(selected):
   bary=np.array([1/3,1/3,1/3]) if list(selected).count(faceid)==1 else np.array([.2+.6*(occurrence+.5)/count,.1,.7-.6*(occurrence+.5)/count])
   triangle=e['tri'][faceid];point=bary@triangle;normal=np.cross(triangle[1]-triangle[0],triangle[2]-triangle[0]);normal/=np.linalg.norm(normal)
   local=e['mat']@(AX@point*adapter['unit_factor'])+e['trans'];local_n=e['mat']@(AX@normal)
   s0=snapshots[0];rest=s0['r'][e['anchor']]@local+s0['p'][e['anchor']];restn=s0['r'][e['anchor']]@local_n
   barytext=','.join(f'{x:.9g}' for x in bary);pid=f'{e["source"]["sha256"]}:compiled:{e["compiled_mesh_sha256"]}:geom:{e["gid"]}:face:{int(faceid)}:bary:{barytext}'
   text=role_text(body,adapter['names'][e['anchor']]);key=hashlib.sha256(text.encode()).hexdigest()
   texts.setdefault(key,{'key':key,'prompt':text,'body':body,'joint_name':adapter['names'][e['anchor']],'patch_ids':[],
    'source':{'xml':str(xml),'xml_sha256':sha(xml),'function_role_origin':'deterministic conservative description from verified model link name; no contact/tactile labels'}})['patch_ids'].append(pid)
   out['patch_ids'].append(pid);out['valid'].append(True);out['rest_xyz'].append(rest.tolist());out['rest_normal'].append(restn.tolist())
   # Representative equal-area stratum, not an assertion that one triangle covers the entire patch.
   out['area'].append(e['total_area']/count*adapter['unit_factor']**2);out['anchor_joint'].append(e['anchor']);out['local_xyz'].append(local.tolist());out['local_normal'].append(local_n.tolist())
   out['semantic_valid'].append(False);out['semantic_text_key'].append(key)
   out['patch_sources'].append({'mesh':e['source'],'mesh_name':e['mesh_name'],'compiled_mesh_sha256':e['compiled_mesh_sha256'],'geom_id':e['gid'],'compiled_face_index':int(faceid),'barycentric':bary.tolist(),
    'joint_name':adapter['names'][e['anchor']],'triangle_area_native2':float(e['area'][faceid]*adapter['unit_factor']**2),'represented_stratum_area_native2':float(e['total_area']/count*adapter['unit_factor']**2),
    'normal_kind':'actual_triangle_winding_normal','normal_valid':True,'contact_valid':False,'tactile_valid':False})
   out['geometry_hashes']['mesh_'+e['source']['sha256'][:16]]=e['source']['sha256']
  area_coverage+=e['total_area']
 joints=[]
 for jid in range(model.njnt):
  bid=int(model.jnt_bodyid[jid]);ancestor=bid
  while ancestor not in set(keep.tolist()) and ancestor>0:ancestor=int(model.body_parentid[ancestor])
  if ancestor==0:continue
  joints.append({'joint_id':jid,'name':mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,jid),'type':int(model.jnt_type[jid]),'body_id':bid,
   'axis_local':model.jnt_axis[jid].tolist(),'anchor_local_m':model.jnt_pos[jid].tolist(),'qpos_address':int(model.jnt_qposadr[jid]),'dof_address':int(model.jnt_dofadr[jid]),'limited':bool(model.jnt_limited[jid]),'range':model.jnt_range[jid].tolist()})
 out['binding_audit']={'status':'passed','mujoco_version':mujoco.__version__,'random_q_states':len(states),'random_q_fk_max_native':maxfk,
  'rigid_frame_rotation_max_abs':max(e['rot_error'] for e in eligible),'rigid_frame_translation_max_native':max(e['pos_error'] for e in eligible),
  'source_geoms_considered':len(visual),'bound_mesh_geoms':len(eligible),'omitted_geoms':omitted,'mesh_scope':mesh_scope,
  'represented_surface_area_native2':area_coverage*adapter['unit_factor']**2,'source_xmls':xmls,
  'evidence_boundary':'Actual source mesh and compiled frame consistency at deterministic geometry-only random q. No grasp success, contact, tactile or real-robot execution measured.',
  'q_to_cached_fk_verified':True,'global_rot_to_q_projection_implemented':False}
 out['kinematic_adapter']={'xml':str(xml),'xml_sha256':sha(xml),'source_spec_sha256':adapter['spec_sha256'],'joint_names':adapter['names'],
  'parents':adapter['parents'].tolist(),'kinematic_rest_off':adapter['offsets'].tolist(),'native_unit':adapter['kinematic_unit'],
  'mechanical_joints':joints,'body_ids':keep.tolist(),'frame_body_ids':frame.tolist(),
  'rest_joint_rotation':snapshots[0]['r'].tolist(),'rest_joint_position':snapshots[0]['p'].tolist(),
  'unsupported_prismatic_body_ids':sorted(excluded_slides),'supported_prismatic_body_ids':sorted(supported_slides),
  'rest_rebase_world_zup':None if adapter['rest_rebase'] is None else adapter['rest_rebase'].tolist(),
  'projection_status':'not implemented; q→compiled frames/FK verified, inverse requires an explicit optimizer/adapter'}
 if 'joint_type' in adapter:
  out['kinematic_adapter'].update({key:jsonable(adapter[key]) if isinstance(adapter[key],np.ndarray) else adapter[key] for key in ('joint_type','prismatic_axis','slide_lower','slide_upper','slide_reference','kinematic_fixed_rotation','root_fixed')})
  out['kinematic_adapter']['root_fixed_evidence']='root body and every ancestor have MuJoCo body_jntnum=0; rebased root I verified in all random states' if adapter['root_fixed'] else 'free or articulated root'
 if supported_slides:
  out['verification_states']=[{'qpos':s['qpos'].tolist(),'global_rotation':s['r'].tolist(),'position_native':s['p'].tolist(),'slide_native':s['slide_native'].tolist()} for s in snapshots]
 return out,list(texts.values())

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--bodies',default='unitree_g1_29dof,unitree_g1_23dof,booster_t1,booster_k1,fourier_n1,pnd_adam_lite,hightorque_hi')
 ap.add_argument('--adroit-meta');ap.add_argument('--franka-meta');ap.add_argument('--patches',type=int,default=96);a=ap.parse_args()
 result={'schema':'verified_mesh_patch_build_v1','bodies':{},'failures':{},'required_surface_texts':[]}
 jobs=[(n,None) for n in a.bodies.split(',') if n]
 if a.adroit_meta:jobs.append(('adroit_pen',a.adroit_meta))
 if a.franka_meta:jobs.append((json.loads(Path(a.franka_meta).read_text()).get('body_id','relay_franka_arm'),a.franka_meta))
 for body,meta in jobs:
  try:
   p,text=build_body(source_adapter(body,meta),a.patches);result['bodies'][body]=p;result['required_surface_texts']+=text
   print(json.dumps({'body':body,'status':'passed','patches':len(p['patch_ids']),'audit':p['binding_audit']}),file=sys.stderr,flush=True)
  except Exception as e:result['failures'][body]=repr(e);print(json.dumps({'body':body,'error':repr(e)}),file=sys.stderr,flush=True)
 print(json.dumps(result,default=jsonable))
if __name__=='__main__':main()
