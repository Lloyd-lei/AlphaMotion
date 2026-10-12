"""Real template/link geometry only; no UMR optimization or motion retargeting."""
from surface_retargeting.config import CFG as SRCFG  # machine paths: configs/sr.toml
from pathlib import Path
import os
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
import sys,json,hashlib,ast,urllib.request
import numpy as np
import trimesh,mujoco
from PIL import Image,ImageDraw
from xml.sax.saxutils import escape

HERE=Path(__file__).resolve().parent
GEOM=Path(f'{SRCFG.research_root}/alpha_motion2/research/embodied_case_20261003/geometry')
sys.path.insert(0,str(GEOM));import render_case as BASE
SMPLX=Path(f'{SRCFG.smplx_neutral}')
G1=Path(f'{SRCFG.gmr_root}/assets/unitree_g1')
SPHERE_RADIUS=.03
EXPECTED_GAP=.012
BARY=np.array([.2,.3,.5])

def record(path):
    p=Path(path);return dict(path=str(p),bytes=p.stat().st_size,sha256=BASE.sha256(p))

def names():
    cache=HERE/'smplx_joint_name_source.json'
    if cache.exists():return json.loads(cache.read_text())
    sha=json.load(urllib.request.urlopen('https://api.github.com/repos/vchoutas/smplx/commits/main'))['sha']
    url=f'https://raw.githubusercontent.com/vchoutas/smplx/{sha}/smplx/joint_names.py'
    source=urllib.request.urlopen(url).read();tree=ast.parse(source.decode())
    values=None
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='JOINT_NAMES' for t in node.targets):values=ast.literal_eval(node.value)
    assert values and values[20]=='left_wrist'
    out=dict(repository='https://github.com/vchoutas/smplx',revision=sha,url=url,
             source_sha256=hashlib.sha256(source).hexdigest(),names=values[:55])
    cache.write_text(json.dumps(out,indent=2)+'\n');return out

def choose_point(vertices,faces,desired_point,desired_normal,eligible=None):
    tri=vertices[faces];normals=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);area=np.linalg.norm(normals,axis=-1)
    normals=normals/np.maximum(area[:,None],1e-15);centers=tri.mean(1)
    mask=(area>1e-12)&((normals@np.asarray(desired_normal))>.5)
    if eligible is not None:mask &=eligible
    ids=np.flatnonzero(mask);assert len(ids)>0
    face=int(ids[np.linalg.norm(centers[ids]-desired_point,axis=-1).argmin()]);point=BARY@tri[face]
    return face,point,normals[face],tri[face]

def sphere_field(point,normal):
    center=point+normal*(SPHERE_RADIUS+EXPECTED_GAP)
    radial=point-center;distance=float(np.linalg.norm(radial));closest=center+SPHERE_RADIUS*radial/distance
    gap=distance-SPHERE_RADIUS;d=point-closest
    return dict(radius_m=SPHERE_RADIUS,rotation_world_from_object=np.eye(3).tolist(),center_world_m=center.tolist(),
        surface_point_world_m=point.tolist(),surface_point_object_m=(point-center).tolist(),
        sphere_closest_point_world_m=closest.tolist(),sphere_closest_point_object_m=(closest-center).tolist(),
        d_body_surface_minus_sphere_closest_world_m=d.tolist(),d_object_m=d.tolist(),
        signed_gap_m=gap,contact=bool(abs(gap)<=.002),contact_threshold_m=.002,
        valid_geometry=bool(np.isfinite(d).all() and abs(np.linalg.norm(closest-center)-SPHERE_RADIUS)<1e-12),
        vector_direction='object surface to body surface',signed_gap_definition='norm(body_point - sphere_center) - radius',
        placement='constructed reference sphere center = point + triangle_normal * (radius + 0.012m); not optimized',
        evidence='geometric validity/proximity only; not measured force, tactile, grasp success, dynamics, or UMR optimization')

def mesh_clearance(vertices,faces,center):
    triangles=vertices[faces]
    points=trimesh.triangles.closest_point(triangles,np.broadcast_to(center,(len(faces),3)))
    distances=np.linalg.norm(points-center,axis=-1);i=int(distances.argmin())
    return dict(method='exact closest point on each supplied triangle, exhaustively checked for this small local mesh',
        supplied_mesh_triangle_index=i,closest_body_point_world_m=points[i].tolist(),
        sphere_center_to_nearest_body_surface_m=float(distances[i]),
        nearest_surface_minus_radius_m=float(distances[i]-SPHERE_RADIUS),
        sphere_clear_of_all_supplied_triangles=bool(distances[i]>SPHERE_RADIUS),
        scope='rendered local hand mesh; distinct from selected point contact flag')

def render(label,vertices,faces,sample,normal,field,skeleton_positions,skeleton_parents,skeleton_names):
    # Submesh is held only inside this remote asset directory; no original model
    # or mesh is copied into the local figure bundle.
    unique=np.unique(faces);lookup=np.full(len(vertices),-1);lookup[unique]=np.arange(len(unique));vv=vertices[unique];ff=lookup[faces]
    meshfile=HERE/(label+'_render_subset.obj')
    with meshfile.open('w') as f:
        for v in vv:f.write('v '+' '.join(f'{x:.10g}' for x in v)+'\n')
        for t in ff:f.write('f '+' '.join(str(int(i)+1) for i in t)+'\n')
    c=np.array(field['center_world_m']);cp=np.array(field['sphere_closest_point_world_m'])
    rgb='0.75 0.75 0.78 1' if label.startswith('g1') else '0.76 0.61 0.49 1'
    escaped=escape(str(meshfile),{'"':'&quot;'})
    xml=f'''<mujoco><compiler angle="radian"/><visual><global offwidth="1000" offheight="1000"/><headlight ambient="0.35 0.35 0.35" diffuse="0.6 0.6 0.6" specular="0.08 0.08 0.08"/></visual><asset><mesh name="hand" file="{escaped}"/></asset><worldbody><body name="actual_hand_surface"><geom name="surface" type="mesh" mesh="hand" rgba="{rgb}" contype="0" conaffinity="0"/></body><body name="reference_sphere" pos="{' '.join(map(str,c))}"><geom name="sphere" type="sphere" size="{SPHERE_RADIUS}" rgba="0.40 0.72 0.80 1" contype="0" conaffinity="0"/></body></worldbody></mujoco>'''
    model=mujoco.MjModel.from_xml_string(xml);data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    bounds=np.concatenate([vv,c[None]-SPHERE_RADIUS,c[None]+SPHERE_RADIUS]);lo=bounds.min(0);hi=bounds.max(0)
    cam=mujoco.MjvCamera();cam.lookat[:]=(lo+hi)/2;cam.distance=max(np.linalg.norm(hi-lo)*1.45,.32);cam.azimuth=150;cam.elevation=-28
    renderer=mujoco.Renderer(model,1000,1000);renderer.update_scene(data,camera=cam)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=0
    meta=BASE.camera_metadata(renderer.scene,1000,1000)
    def image_set(include_sphere):
        # Remove omitted sphere before rendering/segmentation, not via alpha.
        model.geom_group[:]=1;model.geom_group[model.geom('sphere').id]=1 if include_sphere else 5
        option=mujoco.MjvOption();option.geomgroup[:]=0;option.geomgroup[1]=1
        renderer.update_scene(data,camera=cam,scene_option=option)
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=0
        rgbimage=renderer.render().copy();renderer.enable_segmentation_rendering();seg=renderer.render().copy();renderer.disable_segmentation_rendering()
        valid=[model.geom('surface').id]+([model.geom('sphere').id] if include_sphere else [])
        mask=np.isin(seg[:,:,0],valid)&(seg[:,:,1]==int(mujoco.mjtObj.mjOBJ_GEOM))
        rgba=np.concatenate([rgbimage,mask[...,None].astype(np.uint8)*255],-1);rgba[~mask,:3]=255
        white=rgbimage.copy();white[~mask]=255
        return Image.fromarray(white).convert('RGBA'),Image.fromarray(rgba)
    white,transparent=image_set(True)
    white.save(HERE/(label+'_mesh_sphere.png'));transparent.save(HERE/(label+'_mesh_sphere_transparent.png'))
    handwhite,handtrans=image_set(False);handwhite.save(HERE/(label+'_mesh.png'));handtrans.save(HERE/(label+'_mesh_transparent.png'));renderer.close()
    overlay=Image.new('RGBA',(1000,1000),(255,255,255,0));draw=ImageDraw.Draw(overlay)
    xy=BASE.project_points(skeleton_positions,meta)
    for i,p in enumerate(skeleton_parents):
        if p>=0:draw.line([tuple(xy[p]),tuple(xy[i])],fill=(30,137,174,225),width=3)
    for x,y in xy:draw.ellipse([x-4,y-4,x+4,y+4],fill=(30,137,174,255),outline='white',width=1)
    sxy,cxy,nxy=BASE.project_points(np.stack([sample,cp,sample+normal*.02]),meta)
    BASE._arrow(draw,cxy,sxy,(220,81,44,255),5)
    BASE._arrow(draw,sxy,nxy,(107,73,140,235),3)
    for pt,color in [(cxy,(39,123,144,255)),(sxy,(237,108,45,255))]:
        x,y=pt;draw.ellipse([x-8,y-8,x+8,y+8],fill=color,outline='white',width=3)
    combined=Image.alpha_composite(white,overlay);combined.save(HERE/(label+'_surface_overlay.png'))
    Image.alpha_composite(transparent,overlay).save(HERE/(label+'_surface_overlay_transparent.png'))
    Image.alpha_composite(handwhite,overlay).save(HERE/(label+'_hand_overlay.png'))
    cx,cy=sxy;r=180;box=[max(0,int(cx-r)),max(0,int(cy-r)),min(1000,int(cx+r)),min(1000,int(cy+r))]
    combined.crop(box).resize((720,720),Image.Resampling.LANCZOS).save(HERE/(label+'_palm_crop.png'))
    return dict(camera=meta,surface_point_pixel=sxy.tolist(),sphere_closest_point_pixel=cxy.tolist(),
                normal_endpoint_pixel=nxy.tolist(),normal_arrow_length_m=.02,crop_source_box_xyxy=box,
                skeleton_names=skeleton_names,skeleton_parents=list(map(int,skeleton_parents)),skeleton_pixels=xy.tolist(),
                overlay='landmarks and d/normal arrows stay visible through meshes for legibility; geometry is not depth modified',
                remote_only_render_subset=record(meshfile))

def main():
    HERE.mkdir(parents=True,exist_ok=True)
    assert BASE.sha256(SMPLX)=='376021446ddc86e99acacd795182bbef903e61d33b76b9d8b359c2b0865bd992'
    nref=names()
    with np.load(SMPLX,allow_pickle=False) as z:
        v=z['v_template'].astype(float);faces=z['f'].astype(int);weights=z['weights'];J=z['J_regressor']@v;parents=z['kintree_table'][0].astype(int)
    wrist=J[20];vertices=v-wrist
    hand_joints=[20]+list(range(25,40));hand_weight=weights[:,hand_joints].sum(1)
    eligibility=(hand_weight[faces].mean(1)>.5)&(vertices[faces].mean(1)[:,0]>-.025)
    face,p,normal,tri=choose_point(vertices,faces,np.array([.06,-.017,-.005]),np.array([0,-1,0]),eligibility)
    field=sphere_field(p,normal)
    human_clearance=mesh_clearance(vertices,faces[eligibility],np.array(field['center_world_m']))
    sk=J[hand_joints]-wrist;index={j:i for i,j in enumerate(hand_joints)};par=[index.get(int(parents[j]),-1) for j in hand_joints]
    human_projection=render('smplx_left_palm',vertices,faces[eligibility],p,normal,field,sk,par,[nref['names'][j] for j in hand_joints])
    human=dict(source=record(SMPLX),pose='direct neutral v_template; no pose blendshapes or LBS, not a posed motion frame',
        frame='world and body-local identity orientations, origin at J_regressor @ v_template left_wrist; original model is Y-up meters',
        world_from_model_translation_m=(-wrist).tolist(),world_from_model_rotation=np.eye(3).tolist(),
        triangle_index_in_original_f=face,triangle_vertex_indices=faces[face].tolist(),barycentric=BARY.tolist(),
        triangle_vertices_model_m=(tri+wrist).tolist(),triangle_vertices_body_local_m=tri.tolist(),
        point_model_m=(p+wrist).tolist(),point_body_local_m=p.tolist(),normal_body_local=normal.tolist(),normal_world=normal.tolist(),
        normal_source='unit normal from original triangle vertex winding',region='left palm-side reference selected geometrically near wrist; no UMR surface region predictor',
        selection_rule='hand skin-weight family 20,25..39 mean>0.5 and vertex-centroid x relative wrist > -0.025m; nearest candidate to [0.06,-0.017,-0.005], normal dot [0,-1,0] > .5',
        surface_field=field,mesh_clearance_audit=human_clearance,projection=human_projection,
        limitation='cropped boundary is open; normals are triangle-local. Neutral template with real finger skeleton, no motion/shape-person association. Not UMR optimized.')
    meshpath=G1/'meshes/left_rubber_hand.STL';mesh=trimesh.load(str(meshpath),force='mesh',process=False)
    gv=np.asarray(mesh.vertices);gf=np.asarray(mesh.faces)
    gfidx,gp,gn,gtri=choose_point(gv,gf,np.array([.045,-.025,.008]),np.array([0,-1,0]))
    gfield=sphere_field(gp,gn)
    robot_clearance=mesh_clearance(gv,gf,np.array(gfield['center_world_m']))
    # This is the actual fixed offset of the hand body from its parent wrist-yaw
    # link in GMR's G1 MJCF; there are no articulated finger joints in this model.
    gsk=np.array([[-.0415,-.003,0.],[0.,0.,0.]])
    robot_projection=render('g1_left_palm',gv,gf,gp,gn,gfield,gsk,[-1,0],['left_wrist_yaw_link','left_rubber_hand'])
    xml=G1/'g1_mocap_29dof.xml';m=mujoco.MjModel.from_xml_path(str(xml));b=m.body('left_rubber_hand').id
    assert m.body_jntnum[b]==0 and np.max(abs(m.body_pos[b]-[.0415,.003,0]))<1e-9
    robot=dict(source_mesh=record(meshpath),source_xml=record(xml),pose='isolated actual fixed hand link in link-local reference pose; world-from-link I, no full robot motion',
        body='left_rubber_hand',parent_body='left_wrist_yaw_link',world_from_body_rotation=np.eye(3).tolist(),world_from_body_translation_m=[0,0,0],
        movable_finger_dof=0,hand_body_joint_count=int(m.body_jntnum[b]),actual_parent_to_hand_offset_m=m.body_pos[b].tolist(),
        triangle_index_in_original_stl_load_order=gfidx,triangle_vertex_indices=gf[gfidx].tolist(),
        triangle_order_note='trimesh.load(process=False), duplicate STL vertices retained; source hash pins face order',
        barycentric=BARY.tolist(),triangle_vertices_link_local_m=gtri.tolist(),point_link_local_m=gp.tolist(),normal_link_local=gn.tolist(),normal_world=gn.tolist(),
        normal_source='unit normal from original STL triangle vertex winding',surface_field=gfield,mesh_clearance_audit=robot_clearance,projection=robot_projection,
        region='palm-side reference selected geometrically, not an articulated grasp correspondence',
        limitation='fixed rubber-hand visual mesh, not a dexterous hand. Duplicate-vertex STL reports non-watertight before merge; triangle normal is defined but whole-mesh signed volume not used.')
    validation={}
    for label,item in [('smplx',human),('g1',robot)]:
        ff=item['surface_field'];point=np.array(ff['surface_point_world_m']);closest=np.array(ff['sphere_closest_point_world_m']);center=np.array(ff['center_world_m']);dd=np.array(ff['d_body_surface_minus_sphere_closest_world_m'])
        tt=np.array(item.get('triangle_vertices_body_local_m',item.get('triangle_vertices_link_local_m')))
        validation[label]=dict(barycentric_sum_error=abs(BARY.sum()-1),barycentric_nonnegative=bool((BARY>=0).all()),
            barycentric_point_error_m=float(np.linalg.norm(BARY@tt-point)),sphere_radius_error_m=abs(np.linalg.norm(closest-center)-SPHERE_RADIUS),
            d_definition_error_m=float(np.linalg.norm(dd-(point-closest))),signed_gap_m=ff['signed_gap_m'],contact=ff['contact'],valid_geometry=ff['valid_geometry'])
        assert validation[label]['barycentric_point_error_m']<1e-12 and validation[label]['d_definition_error_m']<1e-12
        assert not ff['contact'] and ff['valid_geometry'] and abs(ff['signed_gap_m']-EXPECTED_GAP)<1e-12
    np.savez_compressed(HERE/'surface_reference_trace.npz',barycentric=BARY,sphere_radius_m=np.array(SPHERE_RADIUS),
        smplx_triangle_index=np.array(face),smplx_triangle_vertices_body_local_m=tri,smplx_surface_point_body_local_m=p,smplx_normal=normal,
        smplx_hand_joints_body_local_m=sk,smplx_hand_joint_ids=np.array(hand_joints),smplx_hand_joint_parents=np.array(par),
        smplx_sphere_center_world_m=np.array(field['center_world_m']),smplx_d=np.array(field['d_body_surface_minus_sphere_closest_world_m']),
        g1_triangle_index=np.array(gfidx),g1_triangle_vertices_link_local_m=gtri,g1_surface_point_link_local_m=gp,g1_normal=gn,
        g1_sphere_center_world_m=np.array(gfield['center_world_m']),g1_d=np.array(gfield['d_body_surface_minus_sphere_closest_world_m']))
    manifest=dict(schema='umr-geometry-reference-v1',status='reference geometry only; UMR optimization not run',smplx=human,g1=robot,
        relation='two independently constructed reference scenes; no optimized correspondence, transfer, contact trajectory, or dexterous grasp claim',
        common_definition=dict(sphere_radius_m=SPHERE_RADIUS,object_rotation_world_from_object=np.eye(3).tolist(),
            object_coordinate='object origin at sphere center in each reference scene; rotations identity; scenes have independently constructed centers',
            d='body_surface - sphere_closest_point (object to body)',contact='abs(signed_gap_m) <= 0.002',valid='geometric finite point/triangle/sphere calculations only'),
        source_joint_name_reference=nref,validation=validation,renderer=record(Path(__file__)),projection_helper=record(GEOM/'render_case.py'),backend=dict(python=sys.executable,mujoco_version=mujoco.__version__,MUJOCO_GL='egl'),
        license=dict(smplx='existing provided licensed model reused; account acceptance/provenance not revalidated; no model redistribution',
            smplx_model_license_url='https://smpl-x.is.tue.mpg.de/modellicense',smplx_license_page_fetch_status='web reader inaccessible on this audit',
            g1='existing GMR-vendored Unitree asset; GMR software MIT, robot-asset-specific license not independently re-audited',
            distribution='only derived PNGs, numerical trace and scripts copied locally; original models and render subset meshes stay remote'),
        files=[record(f) for f in HERE.iterdir() if f.is_file() and f.suffix in ['.png','.npz','.py']])
    (HERE/'surface_reference_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(result='complete',smplx_face=face,g1_face=gfidx,smplx_gap=field['signed_gap_m'],g1_gap=gfield['signed_gap_m'],validation=validation)))

if __name__=='__main__':main()
