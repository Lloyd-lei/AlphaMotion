"""Rebuild only this geometry subdirectory's derived assets and provenance."""
from pathlib import Path
import datetime, hashlib, json, subprocess, sys, xml.etree.ElementTree as ET
import numpy as np
import mujoco
import render_case as R
import prep_dof_features as PD
import eval_newbody as EN

HERE=Path(__file__).resolve().parent
CASE=HERE.parent

def record(path):
    p=Path(path)
    return dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=R.sha256(p))

def geometry_sources(xml,repo=None,revision=None):
    p=Path(xml);root=ET.parse(p).getroot();compiler=root.find('compiler')
    meshdir=p.parent/(compiler.get('meshdir','') if compiler is not None else '')
    sources=[record(p)]
    for item in root.findall('./asset/mesh'):
        if item.get('file'):sources.append(record(meshdir/item.get('file')))
    if repo:
        for x in sources:
            q=Path(x['path']);rel=q.relative_to(repo).as_posix()
            try:
                expected=subprocess.check_output(['git','-C',str(repo),'rev-parse',f'{revision}:{rel}'],text=True,stderr=subprocess.DEVNULL).strip()
                data=q.read_bytes();actual=hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()
                x.update(official_commit_blob_sha=expected,actual_git_blob_sha=actual,exact_match_to_revision=expected==actual)
            except subprocess.CalledProcessError:x['exact_match_to_revision']=False;x['provenance_note']='local conversion product not tracked at official revision'
    return sources

def reference_card(folder,filename,prefix,joint,azimuth):
    xml=HERE/'assets'/folder/filename;m=mujoco.MjModel.from_xml_path(str(xml));q=m.key_qpos[0].copy()
    meta=R.render_qpos(xml,q,HERE/prefix,target_joint=joint,azimuth=azimuth,elevation=-20,
                       label=f'{folder} provided keyframe reference; no model inference')
    d=mujoco.MjData(m);d.qpos[:]=q;mujoco.mj_forward(m,d)
    keep,frame=R.PG.build_merge(m);spec=R.PG.build_spec(m,folder,keep)
    D,slotnames,_,_,_,_,_,nonhinge,rest=PD.mjcf_dof(folder,str(xml),dict(parents=spec.parents,joint_names=spec.joint_names))
    gy,gp=R.PG.qpos_to_global(m,d,q[None],keep,frame)
    pose9=np.concatenate([gy[...,0],gy[...,1],gp/(spec.height+1e-6)],-1)
    np.savez_compressed(HERE/(prefix+'_data.npz'),qpos=q,joint_names=np.array([m.joint(i).name for i in range(m.njnt)]),
        body_names=np.array([m.body(i).name for i in range(m.nbody)]),full_body_parents=m.body_parentid,
        axes_local=m.jnt_axis,qpos_addresses=m.jnt_qposadr,full_body_positions_zup_m=d.xpos.copy(),
        full_body_rotations_zup=d.xmat.reshape(-1,3,3).copy(),descriptor_joint_names=np.array(spec.joint_names),
        descriptor_parents=spec.parents,descriptor_rest_offsets_cm_y=spec.rest_offsets,
        descriptor_dof19=D,descriptor_rest_rotations_y=rest,global_rotations_y=gy,
        root_relative_positions_cm_y=gp,pose9=pose9,height_sum_bonelength_cm=np.array(spec.height))
    return dict(model=folder,xml=str(xml),clip=None,pose_status='provided reference keyframe, not inference',
        reference_keyframe_index=0,reference_keyframe_name=m.key(0).name,qpos=q.tolist(),nq=m.nq,nv=m.nv,
        body_count_including_world=m.nbody,descriptor_joint_count=spec.J,
        descriptor_joint_names=spec.joint_names,descriptor_parents=spec.parents.tolist(),
        nonhinge_descriptor_unsupported=[dict(joint_name=n,mujoco_joint_type=int(t)) for n,t in nonhinge],
        descriptor_limits='nonhinge slides skipped by existing legacy mechanical descriptor; no full Panda adapter claim' if nonhinge else 'hinge descriptor only; reference pose no neural inference',
        target=meta['target'],geometry_sources=geometry_sources(xml),
        asset_license_file=record(HERE/'assets'/folder/'LICENSE'),upstream_readme=record(HERE/'assets'/folder/'README.md'))

def main():
    baseline=R.reference_batch()
    trace_json=CASE/'alpha1/trace.json';trace_npz=CASE/'alpha1/trace.npz'
    trace=json.loads(trace_json.read_text());pred=np.load(trace_npz,allow_pickle=False)
    target_outputs=[]
    for i,f in enumerate(R.FRAMES):
        meta=R.render_globals(R.H2_XML,pred['predicted_global_rotation_matrices_y'][i],
            pred['predicted_fk_positions_root_relative_cm_y'][i],HERE/f'h2_predicted_frame_{f}',
            expected_joint_names=trace['target']['joint_names'])
        target_outputs.append(dict(frame=f,prefix=f'h2_predicted_frame_{f}',target=meta['target'],
              omitted_mesh_bodies=meta['omitted_mesh_bodies'],root_visualization_gauge='root at (0,0,0); camera framing only; no floor or predicted XY/root-height claim',
              joint_global_rotation_matrix_y=pred['predicted_global_rotation_matrices_y'][i,22].tolist(),
              joint_global_rot6d_y=pred['predicted_global_rot6d_y'][i,22].tolist(),
              joint_fk_position_root_relative_cm_y=pred['predicted_fk_positions_root_relative_cm_y'][i,22].tolist(),
              joint_direct_position_head_raw=pred['predicted_position_head_raw'][i,22].tolist()))
    go2=reference_card('unitree_go2','go2.xml','go2_reference','FL_calf_joint',135)
    panda=reference_card('franka_emika_panda','panda.xml','panda_reference','joint4',130)
    repo=Path('/var/tmp/alphamotion/robot_models/unitree_ros');rev=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    h2sources=geometry_sources(R.H2_XML,repo,rev)
    h2urdf=repo/'robots/h2_description/H2.urdf';h2urdf_record=record(h2urdf)
    raw=h2urdf.read_bytes();blob=hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest()
    expected=subprocess.check_output(['git','-C',str(repo),'rev-parse',f'{rev}:robots/h2_description/H2.urdf'],text=True).strip()
    h2urdf_record.update(actual_git_blob_sha=blob,official_commit_blob_sha=expected,exact_match_to_revision=blob==expected)
    license_text=subprocess.check_output(['git','-C',str(repo),'show',f'{rev}:LICENSE'],text=True)
    (HERE/'LICENSE_Unitree_H2.txt').write_text(license_text)
    for folder,name in [('unitree_go2','LICENSE_Unitree_Go2.txt'),('franka_emika_panda','LICENSE_Franka_Panda.txt')]:
        (HERE/name).write_text((HERE/'assets'/folder/'LICENSE').read_text())
    g1license=Path('/home/arenalabs/Desktop/"be water, robot"/alpha_motion2/external/GMR/LICENSE')
    (HERE/'LICENSE_GMR_software.txt').write_text(g1license.read_text())
    # Fresh H2 descriptor and rest FK are separately saved; never substitute the
    # legacy H2 cached anonymous/data-tree skeleton for these named official links.
    hm=mujoco.MjModel.from_xml_path(str(R.H2_XML));hs,D,rest,slots,_=EN.build_from_mjcf(str(R.H2_XML),'unitree_h2_fresh')
    np.savez_compressed(HERE/'h2_fresh_geometry_descriptor.npz',joint_names=np.array(hs.joint_names),parents=hs.parents,
        rest_offsets_cm_y=hs.rest_offsets,dof19=D,rest_rotations_y=rest,full_model_body_names=np.array([hm.body(i).name for i in range(hm.nbody)]),
        full_model_body_parents=hm.body_parentid,full_model_joint_names=np.array([hm.joint(i).name for i in range(hm.njnt)]),
        full_model_joint_axis=hm.jnt_axis,full_model_joint_ranges=hm.jnt_range)
    geometry_manifest=dict(schema='embodied-case-geometry-v1',timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        actual_case='G1 cached LAFAN motion -> AlphaMotion checkpoint forward -> fresh H2 descriptor output',
        renderer=dict(file=record(HERE/'render_case.py'),python=sys.executable,mujoco_version=mujoco.__version__,backend='EGL',
            operation='CPU forward kinematics and offscreen geometry rendering; no dynamics step, training, queue or service change',
            source_mesh_filter='explicit noncollision visual meshes only; body>0; collision/world geoms excluded before scene creation'),
        g1=dict(model='Unitree G1 29 DOF',reference_clip=R.CLIP,frames=R.FRAMES,source=baseline,
            geometry_sources=geometry_sources(R.G1_XML),vendor_urdf=record(R.G1_XML.parent/'g1_custom_collision_29dof.urdf'),
            highlighted_joint='left_elbow_joint',highlighted_link='left_elbow_link',cached_spec_joint_index=23,
            license_note='vendored GMR software MIT; asset-specific upstream robot provenance not re-audited',gmr_software_license=record(g1license)),
        h2=dict(model='Unitree H2',xml=record(R.H2_XML),asset_upstream_repository='https://github.com/unitreerobotics/unitree_ros',
            asset_upstream_revision=rev,source_urdf=h2urdf_record,geometry_sources=h2sources,
            license='BSD-3-Clause, official Unitree repository LICENSE',license_copy=record(HERE/'LICENSE_Unitree_H2.txt'),
            mjcf_provenance='existing local untracked MJCF conversion, not a committed official MJCF; underlying URDF and all referenced meshes verified against pinned official git blobs',
            nq=hm.nq,nv=hm.nv,joints_including_free=hm.njnt,body_count_including_world=hm.nbody,mesh_count=hm.nmesh,
            fresh_descriptor_joint_count=hs.J,fresh_descriptor_joint_names=hs.joint_names,parents=hs.parents.tolist(),
            merged_slot=dict(index=14,slot_name='waist_roll_link',rotation_carrier='torso_link',lost_independent_mesh_orientation='waist_roll_link'),
            cached_h2_data_tree_used=False,highlighted_joint='left_elbow_joint',highlighted_link='left_elbow_link',fresh_spec_joint_index=22,
            prediction_trace_json=record(trace_json),prediction_trace_npz=record(trace_npz),outputs=target_outputs,
            evidence_boundary='direct network per-link rotation and FK geometry; no hinge fitting, IK, dynamics or robot validation; no zero-shot claim'),
        go2=go2,panda=panda,
        input_card_relationship='G1 is the only actual motion encoder input in this case. Go2 and Panda are standalone real-geometry reference cards, not fused inputs or completed cross-embodiment model runs.',
        coordinate_conventions=dict(raw_zup_meter=True,raw_quaternion_order='wxyz',cache_yup_cm=True,AX_zup_to_yup=R.AX.tolist(),
              matrix_global_zup='AX.T @ matrix_global_yup @ AX',position_zup_meter='positions_yup_cm @ AX / 100',
              rot6d='first two matrix columns concatenated; Gram-Schmidt if input is 6D',
              camera='each projection JSON is exact rendered camera; pixels top-left origin; arrows are projected actual MJCF axis',
              render_gauge='reference G1 keeps raw root position. H2 predictions use root-relative positions at origin; no hidden floor snap.'),
        semantic_joint_relation='Source G1 left_elbow_link and target H2 left_elbow_link share a name and anatomical role; this is descriptive evidence, not an imposed hard one-to-one decoder mapping',
        surface_semantics='meshes are robot visual rigid link assets; no skin weights or local surface-region supervision asserted',
        derived_files=[record(p) for p in HERE.iterdir() if p.is_file() and p.suffix in {'.png','.npz','.py'}])
    (HERE/'geometry_manifest.json').write_text(json.dumps(geometry_manifest,indent=2)+'\n')
    print(json.dumps(dict(result='complete',frames=R.FRAMES,h2_reference_meshes=len(h2sources)-1,
                         official_h2_meshes_exact=sum(x.get('exact_match_to_revision',False) for x in h2sources),
                         go2_J=go2['descriptor_joint_count'],panda_J=panda['descriptor_joint_count'],
                         panda_nonhinge=panda['nonhinge_descriptor_unsupported'])))

if __name__=='__main__':main()
