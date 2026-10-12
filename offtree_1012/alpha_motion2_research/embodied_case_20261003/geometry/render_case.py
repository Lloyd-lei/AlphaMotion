"""Faithful MuJoCo geometry pictures and exact camera-projection metadata.

Source data stay outside this directory. PNGs are derived visualizations, not
simulation outcomes. The globals API accepts ARDY Y-up cm + column-6D rotations.
"""
from __future__ import annotations
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
import argparse, hashlib, json, math, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import mujoco

HERE = Path(__file__).resolve().parent
LEGACY = Path('/home/arenalabs/Desktop/"be water, robot"/pose_translator_design/exp_topo')
sys.path.insert(0, str(LEGACY))
import prep_gmr_cache as PG

AX = np.array([[0.,1.,0.],[0.,0.,1.],[1.,0.,0.]])
G1_XML = Path('/home/arenalabs/Desktop/"be water, robot"/alpha_motion2/external/GMR/assets/unitree_g1/g1_mocap_29dof.xml')
H2_XML = Path('/var/tmp/alphamotion/robot_models/unitree_ros/robots/h2_description/unitree_h2_mjcf.xml')
CLIP = 'fallAndGetUp3_subject1'
FRAMES = [1800, 2000, 2600]

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def rotation6d_matrix(v):
    v=np.asarray(v,dtype=float)
    a=v[...,:3]; a=a/np.maximum(np.linalg.norm(a,axis=-1,keepdims=True),1e-12)
    b=v[...,3:6]; b=b-(a*b).sum(-1,keepdims=True)*a
    b=b/np.maximum(np.linalg.norm(b,axis=-1,keepdims=True),1e-12)
    return np.stack([a,b,np.cross(a,b)],axis=-1)

def quaternion_matrix(q):
    out=np.empty(9);mujoco.mju_quat2Mat(out,np.asarray(q,dtype=float));return out.reshape(3,3)

def camera_metadata(scene,width,height):
    a,b=scene.camera
    pos=(a.pos.astype(float)+b.pos)/2
    forward=(a.forward.astype(float)+b.forward)/2;forward/=np.linalg.norm(forward)
    up=(a.up.astype(float)+b.up)/2;up/=np.linalg.norm(up)
    right=np.cross(forward,up);right/=np.linalg.norm(right)
    up=np.cross(right,forward)
    return dict(position_zup_m=pos.tolist(),forward=forward.tolist(),up=up.tolist(),right=right.tolist(),
                frustum_top=float(a.frustum_top),frustum_bottom=float(a.frustum_bottom),
                frustum_near=float(a.frustum_near),frustum_center=float(a.frustum_center),
                orthographic=bool(a.orthographic),width=width,height=height,
                pixel_origin='top_left',pixel_convention='continuous_pixel_coordinates',
                projection='MuJoCo actual monocular GL camera; midpoint of stereo camera pair')

def project_points(points,camera):
    p=np.asarray(points)-camera['position_zup_m']
    x=p@np.array(camera['right']);y=p@np.array(camera['up']);z=p@np.array(camera['forward'])
    halfheight=(camera['frustum_top']-camera['frustum_bottom'])/2
    halfwidth=halfheight*camera['width']/camera['height']
    if camera['orthographic']: sx=x;sy=y
    else: sx=x*camera['frustum_near']/z;sy=y*camera['frustum_near']/z
    nx=(sx-camera['frustum_center'])/halfwidth
    ny=(sy-(camera['frustum_top']+camera['frustum_bottom'])/2)/halfheight
    return np.stack([(nx+1)*camera['width']/2,(1-ny)*camera['height']/2],-1)

def _arrow(draw,start,end,color,width=5):
    s=np.asarray(start);e=np.asarray(end);draw.line([tuple(s),tuple(e)],fill=color,width=width)
    v=e-s;n=np.linalg.norm(v)
    if n>1e-6:
        v/=n;perp=np.array([-v[1],v[0]])
        draw.polygon([tuple(e),tuple(e-v*14+perp*6),tuple(e-v*14-perp*6)],fill=color)

def render_body_transforms(xml, body_positions_zup_m, body_rotations_zup,
                           output_prefix, *, skeleton_positions_zup_m=None,
                           skeleton_parents=None, skeleton_names=None,
                           qpos=None, target_joint='left_elbow_joint',
                           omit_body_ids=(), width=1000, height=1000,
                           azimuth=140, elevation=-18, camera_distance=None,
                           camera_lookat=None, label='derived rigid-link visualization'):
    """Render full compiled-model body arrays, no dynamics step or angle fitting.

    Returned JSON includes exact camera and target pixel/axis endpoints. Skeleton
    lines are a visible overlay; they are not depth-occluded anatomy.
    """
    xml=Path(xml);prefix=Path(output_prefix);prefix.parent.mkdir(parents=True,exist_ok=True)
    m=mujoco.MjModel.from_xml_path(str(xml));m.vis.global_.offwidth=width;m.vis.global_.offheight=height
    m.light_active[:]=0
    m.vis.headlight.ambient[:]=[0.3,0.3,0.3];m.vis.headlight.diffuse[:]=[0.6,0.6,0.6]
    m.vis.headlight.specular[:]=[0.08,0.08,0.08]
    d=mujoco.MjData(m)
    if qpos is not None:d.qpos[:]=qpos
    mujoco.mj_forward(m,d)
    visual_ids={g for g in range(m.ngeom)
                if int(m.geom_bodyid[g])>0 and int(m.geom_bodyid[g]) not in omit_body_ids
                and m.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH
                and m.geom_contype[g]==0 and m.geom_conaffinity[g]==0
                and m.geom_rgba[g,3]>0}
    if not visual_ids:raise ValueError('No explicit non-collision mesh visual geoms found')
    # Filter BEFORE mjv_updateScene: transparent collision helpers can still
    # write segmentation/depth. Move excluded geoms into a disabled group in
    # this in-memory model only; original MJCF and source data are untouched.
    for g in range(m.ngeom):m.geom_group[g]=1 if g in visual_ids else 5
    P=np.asarray(body_positions_zup_m,float);R=np.asarray(body_rotations_zup,float)
    assert P.shape==(m.nbody,3) and R.shape==(m.nbody,3,3)
    if skeleton_positions_zup_m is None:
        skP=P[1:];parents=np.asarray(m.body_parentid[1:])-1
        names=[m.body(i).name for i in range(1,m.nbody)]
    else:
        skP=np.asarray(skeleton_positions_zup_m);parents=np.asarray(skeleton_parents)
        names=list(skeleton_names)
    # Camera bounds from the posed mesh vertices; excludes world geoms.
    points=[skP]
    for g in range(m.ngeom):
        b=int(m.geom_bodyid[g])
        if g not in visual_ids:continue
        if m.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH:
            mid=int(m.geom_dataid[g]);a=int(m.mesh_vertadr[mid]);n=int(m.mesh_vertnum[mid])
            vv=m.mesh_vert[a:a+n: max(1,n//1000)]
            rg=R[b]@quaternion_matrix(m.geom_quat[g]);pg=P[b]+R[b]@m.geom_pos[g]
            points.append(vv@rg.T+pg)
    bound=np.concatenate(points);lo=bound.min(0);hi=bound.max(0);center=(lo+hi)/2
    if camera_lookat is not None:center=np.asarray(camera_lookat,float)
    camera=mujoco.MjvCamera();camera.lookat[:]=center
    camera.distance=camera_distance or max(float(np.linalg.norm(hi-lo))*1.35,1.0)
    camera.azimuth=azimuth;camera.elevation=elevation
    renderer=mujoco.Renderer(m,height=height,width=width)
    option=mujoco.MjvOption();option.geomgroup[:]=0;option.geomgroup[1]=1
    renderer.update_scene(d,camera=camera,scene_option=option)
    # Retain mesh visuals only: no opaque collision primitives over true meshes.
    visible_geom_ids=[]
    for i in range(renderer.scene.ngeom):
        g=renderer.scene.geoms[i]
        if g.objtype!=mujoco.mjtObj.mjOBJ_GEOM: g.rgba[3]=0;continue
        gi=int(g.objid);b=int(m.geom_bodyid[gi])
        if gi not in visual_ids:
            g.rgba[3]=0;continue
        g.pos[:]=P[b]+R[b]@m.geom_pos[gi]
        g.mat[:]=R[b]@quaternion_matrix(m.geom_quat[gi])
        g.rgba[3]=1
        visible_geom_ids.append(gi)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=0
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION]=0
    rgb=renderer.render().copy()
    renderer.enable_segmentation_rendering();seg=renderer.render().copy();renderer.disable_segmentation_rendering()
    mask=np.isin(seg[:,:,0],visible_geom_ids)&(seg[:,:,1]==int(mujoco.mjtObj.mjOBJ_GEOM))
    rgba=np.concatenate([rgb,np.uint8(mask[...,None])*255],-1);rgba[~mask,:3]=255
    white=rgb.copy();white[~mask]=255
    cam=camera_metadata(renderer.scene,width,height)
    renderer.close()
    base=Image.fromarray(white);transparent=Image.fromarray(rgba)
    base.save(str(prefix)+'_mesh.png');transparent.save(str(prefix)+'_mesh_transparent.png')
    xy=project_points(skP,cam)
    overlay=Image.new('RGBA',(width,height),(255,255,255,0));dr=ImageDraw.Draw(overlay)
    for j,p in enumerate(parents):
        if p>=0:dr.line([tuple(xy[p]),tuple(xy[j])],fill=(34,142,178,210),width=4)
    for pt in xy:
        x,y=pt;dr.ellipse([x-4,y-4,x+4,y+4],fill=(16,107,142,240),outline=(255,255,255,230),width=1)
    ji=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,target_joint)
    target=None
    if ji>=0:
        b=int(m.jnt_bodyid[ji]);anchor=P[b]+R[b]@m.jnt_pos[ji]
        axis=R[b]@m.jnt_axis[ji]
        length=0.16 if np.linalg.norm(hi-lo)>1.5 else 0.11
        endpoints=np.stack([anchor,anchor+axis*length]);exy=project_points(endpoints,cam)
        _arrow(dr,*exy,(221,83,37,255),5)
        x,y=exy[0];dr.ellipse([x-9,y-9,x+9,y+9],fill=(240,114,56,255),outline='white',width=3)
        target=dict(joint_name=target_joint,joint_id=int(ji),body_name=m.body(b).name,
                    axis_local_mjcf=m.jnt_axis[ji].tolist(),axis_world_zup=axis.tolist(),
                    anchor_world_zup_m=anchor.tolist(),anchor_pixel=exy[0].tolist(),
                    axis_endpoint_pixel=exy[1].tolist(),axis_length_m=length,
                    angle_rad=None if qpos is None else float(qpos[m.jnt_qposadr[ji]]),
                    angle_status='reference qpos' if qpos is not None else 'not_estimated; direct rotation is not a hinge angle',
                    axis_projection='camera_projected_actual_joint_axis_not_schematic')
    composite=Image.alpha_composite(base.convert('RGBA'),overlay)
    composite.save(str(prefix)+'_mesh_skeleton.png')
    Image.alpha_composite(transparent,overlay).save(str(prefix)+'_mesh_skeleton_transparent.png')
    skwhite=Image.alpha_composite(Image.new('RGBA',(width,height),'white'),overlay)
    skwhite.save(str(prefix)+'_skeleton.png');overlay.save(str(prefix)+'_skeleton_transparent.png')
    if target:
        cx,cy=target['anchor_pixel'];r=150
        box=(max(0,int(cx-r)),max(0,int(cy-r)),min(width,int(cx+r)),min(height,int(cy+r)))
        composite.crop(box).resize((600,600),Image.Resampling.LANCZOS).save(str(prefix)+'_joint_closeup.png')
        target['closeup_source_box_xyxy']=list(box)
        target['closeup_anchor_pixel']=[(cx-box[0])*600/(box[2]-box[0]),(cy-box[1])*600/(box[3]-box[1])]
    meta=dict(label=label,xml_path=str(xml),xml_sha256=sha256(xml),
              mesh_render='actual compiled MuJoCo mesh geometry with rigid body transforms',
              visual_geom_ids=sorted(visual_ids),
              visual_filter='body>0; explicit mesh; contype=conaffinity=0; group filtered before scene creation',
              skeleton_render='visible overlay, no occlusion; named origins, not surface semantics',
              camera=cam,target=target,skeleton_names=names,skeleton_parents=parents.tolist(),
              skeleton_pixels=xy.tolist(),skeleton_positions_zup_m=skP.tolist(),
              omitted_mesh_bodies=[m.body(int(i)).name for i in omit_body_ids],
              bounds_zup_m=[lo.tolist(),hi.tolist()])
    Path(str(prefix)+'_projection.json').write_text(json.dumps(meta,indent=2)+'\n')
    return meta

def render_qpos(xml,qpos,output_prefix,**kwargs):
    m=mujoco.MjModel.from_xml_path(str(xml));d=mujoco.MjData(m);d.qpos[:]=qpos;mujoco.mj_forward(m,d)
    return render_body_transforms(xml,d.xpos.copy(),d.xmat.reshape(-1,3,3).copy(),output_prefix,qpos=qpos,**kwargs)

def render_globals(xml, global_rotation, positions_cm, output_prefix, *,
                   root_world_yup_cm=None, expected_joint_names=None, **kwargs):
    """Unprojected fresh-MJCF PG.build_merge pose -> picture.

    Input global_rotation [J,6] = first two matrix COLUMNS or [J,3,3],
    positions_cm [J,3] in ARDY Y-up, usually root-relative FK positions.
    Optional root_world_yup_cm is added, no automatic root-height change.
    Frames: p_zup_m = (p_yup_cm/100) @ AX; R_zup=AX.T@R_yup@AX.
    No hinge fitting. A co-located slot's stored frame may differ from its named
    body: that unresolved body's mesh is omitted, not invented. Other rigid
    links use network transforms directly and can visibly disconnect.
    """
    m=mujoco.MjModel.from_xml_path(str(xml));keep,frame=PG.build_merge(m);spec=PG.build_spec(m,'renderer',keep)
    if expected_joint_names is not None and list(expected_joint_names)!=spec.joint_names:
        raise ValueError('Pose joint names do not match fresh MJCF merge spec')
    rot=np.asarray(global_rotation,float)
    if rot.shape==(len(keep),6):rot=rotation6d_matrix(rot)
    p=np.asarray(positions_cm,float)
    assert rot.shape==(len(keep),3,3) and p.shape==(len(keep),3)
    if root_world_yup_cm is not None:p=p+np.asarray(root_world_yup_cm)
    Rz=AX.T@rot@AX;Pz=p@AX/100
    d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    P=d.xpos.copy();R=d.xmat.reshape(-1,3,3).copy();resolved=set();omitted=[]
    for j,b in enumerate(keep):
        f=frame[b];P[f]=Pz[j];R[f]=Rz[j];resolved.add(f)
        if f!=b:P[b]=Pz[j];omitted.append(b)
    for b in range(1,m.nbody):
        if b not in resolved and b not in omitted:omitted.append(b)
    meta=render_body_transforms(xml,P,R,output_prefix,skeleton_positions_zup_m=Pz,
            skeleton_parents=spec.parents,skeleton_names=spec.joint_names,
            omit_body_ids=omitted,label='unprojected network per-link transforms; no manifold/hinge projection',**kwargs)
    meta['input_conventions']=dict(up='Y',unit='cm',rot6d='columns_0_and_1',
                                   root_translation_added=root_world_yup_cm is not None)
    meta['merge_keep_body_ids']=keep;meta['merge_frame_body_ids']=[frame[b] for b in keep]
    Path(str(output_prefix)+'_projection.json').write_text(json.dumps(meta,indent=2)+'\n')
    return meta

def render_skeleton(positions_cm,parents,names,output_prefix,*,camera=None,target_name=None):
    """Standalone direct-position-head or FK skeleton. No fabricated mesh binding."""
    p=np.asarray(positions_cm,float)@AX/100
    if camera is None:raise ValueError('Pass a camera from a mesh projection JSON for exact comparison')
    xy=project_points(p,camera);im=Image.new('RGBA',(camera['width'],camera['height']),'white');dr=ImageDraw.Draw(im)
    for j,pa in enumerate(parents):
        if pa>=0:dr.line([tuple(xy[pa]),tuple(xy[j])],fill=(34,142,178,255),width=4)
    for pt in xy:
        x,y=pt;dr.ellipse([x-4,y-4,x+4,y+4],fill=(16,107,142,255))
    im.save(str(output_prefix)+'.png')
    result=dict(camera=camera,skeleton_names=list(names),skeleton_parents=list(map(int,parents)),skeleton_pixels=xy.tolist(),
                label='direct supplied position skeleton; no FK or manifold correction')
    Path(str(output_prefix)+'_projection.json').write_text(json.dumps(result,indent=2)+'\n');return result

def reference_batch():
    import prep_xwbc_cache as PX
    xml=G1_XML;m=mujoco.MjModel.from_xml_path(str(xml));addrs,cols,zero=PX.joint_map(m,'unitree_g1_29dof')
    source=Path('/var/tmp/alphamotion/xwbc_lafan/extracted/unitree_g1_29dof')/(CLIP+'.npz')
    with np.load(source,allow_pickle=False) as z:
        allq=PX.rebuild_qpos(m,z,addrs,cols,len(z['joint_pos']));qs=allq[FRAMES]
        rawp=z['body_pos_w'][FRAMES].copy();rawquat=z['body_quat_w'][FRAMES].copy()
        joint_names=z['joint_names'].copy();jp=z['joint_pos'][FRAMES].copy()
    Ps=[];Rs=[];reports=[];errors=[]
    for f,q in zip(FRAMES,qs):
        d=mujoco.MjData(m);d.qpos[:]=q;mujoco.mj_forward(m,d);Ps.append(d.xpos.copy());Rs.append(d.xmat.reshape(-1,3,3).copy())
        ii=FRAMES.index(f);errors.append(float(np.linalg.norm(rawp[ii,:,None]-d.xpos[None],axis=-1).min(-1).max()*1000))
        reports.append(render_qpos(xml,q,HERE/f'g1_reference_frame_{f}',label=f'XWBC stored retargeted G1 reference; frame {f}; not AlphaMotion prediction'))
    hm=mujoco.MjModel.from_xml_path(str(H2_XML));hq=hm.qpos0.copy();hq[2]=PG.standing_height(hm)
    # A documented neutral configuration, not a sampled motion or model output.
    hreport=render_qpos(H2_XML,hq,HERE/'h2_official_neutral_reference',label='Unitree H2 official assets, locally derived MJCF neutral qpos; not model output')
    np.savez_compressed(HERE/'source_derived_frames.npz',frame_indices=np.array(FRAMES),fps=np.array(30),
        g1_qpos=qs,g1_joint_pos_rad=jp,g1_joint_names=joint_names,g1_full_body_positions_zup_m=np.array(Ps),
        g1_full_body_rotations_zup=np.array(Rs),g1_raw_body_positions_zup_m=rawp,g1_raw_body_quaternions_wxyz=rawquat,
        h2_neutral_qpos=hq,axis_transform_zup_to_yup=AX)
    trace=dict(clip=CLIP,frames=FRAMES,times_seconds=[f/30 for f in FRAMES],fps=30,
        source_npz=str(source),source_npz_sha256=sha256(source),source_length=3065,
        coordinate_transform=dict(raw_up='Z',raw_position_unit='m',raw_quaternion_order='wxyz',cache_up='Y',cache_unit='cm',AX=AX.tolist()),
        source_label='XWBC pre-retargeted G1 reference, not inference output',
        source_rebuild=dict(mapping='exact joint name to MJCF qpos address',zero_filled_joints=zero,
                           selected_frame_FK_nearest_body_max_error_mm=errors),
        g1_frames=[dict(frame=f,target=r['target'],camera=r['camera']) for f,r in zip(FRAMES,reports)],
        h2_reference=hreport,h2_cached_tree_used=False)
    (HERE/'case_geometry_trace.json').write_text(json.dumps(trace,indent=2)+'\n')
    return trace

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--references',action='store_true');args=ap.parse_args()
    if args.references:
        trace=reference_batch();print(json.dumps({'frames':trace['frames'],'g1_fk_error_mm':trace['source_rebuild']['selected_frame_FK_nearest_body_max_error_mm'],'output':str(HERE)}))
