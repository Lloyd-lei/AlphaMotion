"""Frozen real-data contract for six runs; no fitting on heldout motion."""
from pathlib import Path
import json,sys,types,hashlib
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'base'))
sys.path.insert(0,str(ROOT))
from pilot_data import Body,load_bodies,load_payload,read_json,write_json,sha256
from surface.surface_contract import prepare_static_surface,expand_surface

def static_reach(parents,offsets):
    distance=np.zeros(len(parents),dtype=np.float64)
    for j,p in enumerate(parents):
        if p>=0:distance[j]=distance[p]+np.linalg.norm(offsets[j])
    if not np.isfinite(distance).all() or distance.max()<=0:raise ValueError('invalid static geometry')
    return float(distance.max())

def attach_geometry(body,surface=None):
    body._surface_static=surface
    def geom(self,batch):
        g={k:v[None].expand(batch,*v.shape) for k,v in self.static.items()}
        device=g['rest_off'].device
        g.update(parents=torch.as_tensor(self.parents,device=device),
            kinematic_rest_off=torch.as_tensor(self.offsets,device=device),
            pose_scale=torch.full((batch,),self.reach,device=device),joint_motion_kind='rotational_tree')
        for key,value in getattr(self,'_typed_static',{}).items():
            g[key]=torch.as_tensor(value,device=device)
        if self._surface_static is not None:g['surface']=expand_surface(self._surface_static,batch)
        return g
    body.geom=types.MethodType(geom,body)

def external_body(root,name,qwen,device):
    meta_path=root/'data/payloads'/f'{name}.json';meta=read_json(meta_path)
    p=root/'data/payloads'/f'{name}.npz'
    if sha256(p)!=meta['payload_sha256']:raise ValueError('external payload hash changed')
    a=load_payload(p);old_n=len(a['pose9'])
    stride=40 if name.startswith('relay_franka_') else 1
    rows=np.flatnonzero(a['frame_index']%stride==0)
    # Field names avoid accidentally treating a J-vector as rows if N happens to equal J.
    row_fields=['global_rot6','position_native','position_canonical','pose9','split','source_group_index',
                'frame_index','timestamp_seconds','fps','position_valid','rotation_valid','recorded_robot_qpos',
                'recorded_scene_qpos','recorded_scene_qvel','recorded_scene_ctrl','recorded_scene_mocap',
                'world_root_m_zup','slide_position_m','slide_normalized','slide_valid','aperture_m','aperture_signed_m','joint_position_native','joint_position_valid']
    for key in row_fields:
        if key in a and a[key].ndim and len(a[key])==old_n:a[key]=a[key][rows]
    a['payload_row_index']=rows
    if 'slide_position_m' in a:a['slide_native']=a['slide_position_m'].astype(np.float32)*100
    offsets=a['rest_offsets'].astype(np.float32)*100
    reach=static_reach(a['parents'],offsets)
    a['pose9']=np.concatenate([a['global_rot6'],a['position_native']*100/reach],axis=-1).astype(np.float32)
    N,J=a['pose9'].shape[:2]
    a.update(group=a['source_group_index'],pretrain_seen=np.zeros(N,bool),
             root_height_m=np.full(N,np.nan,np.float32),root_height_valid=np.zeros(N,bool),
             contact_sdist=np.full((N,J),np.nan,np.float32),contact_valid=np.zeros((N,J),bool),
             contact_soft=np.full((N,J),np.nan,np.float32),contact_hard=np.zeros((N,J),bool),
             contact_trust=np.zeros(N,bool),ground_valid=np.zeros(N,bool))
    meta.update(payload_path=str(p),metadata_sha256=sha256(meta_path),length_unit='cm',
        position_unit_conversion='native metres multiplied by100 for canonical training centimetres; qpos unchanged',
        source_native_unit='m',training_stride=stride,raw_frame_count=old_n,training_payload_rows=N,
        normalization='static maximum root-to-joint path length, no motion fitting',
        task_family='unpaired_same_body_reconstruction_or_sparse_completion',
        implementation_domain_key='unimate',actual_dataset=meta['source_dataset'])
    b=Body(name,'unimate',a,a['parents'],offsets,a['rest_offsets_descriptor'],list(a['joint_names'].astype(str)),
           reach,a['dof_features'],a['dof_valid'],True,meta)
    root_fixed=any(n['skeleton_index']==0 for n in meta['model'].get('fixed_body_nodes',[]))
    if not root_fixed:raise ValueError('external fixed-root source evidence missing')
    b._typed_static={'root_fixed':True}
    if 'joint_type' in a:
        b._typed_static.update({k:a[k] for k in ['joint_type','prismatic_axis','prismatic_fixed_rotation']})
        for key in ['slide_lower','slide_upper','slide_reference']:b._typed_static[key]=a[key]*100
    # The actual declared model endpoints are legal task targets; never infer by body nickname.
    if 'ee_indices' in a:
        ids=[int(i) for i in a['ee_indices']]
        if not ids or any(i<0 or i>=J for i in ids):raise ValueError('external endpoint metadata')
        b.endpoint={'indices':ids,'names':[b.names[i] for i in ids],
                    'policy':'recorded source model leaf joints or explicit named end_effector site',
                    'source':'hash-bound external payload ee_indices','physical_validation':False}
    b.setup(qwen,device)
    if 'mechanical_valid' in a:b.static['metadata_missing'][:,0]=torch.as_tensor(~a['mechanical_valid'],device=device)
    return b

def scaled_surface_payload(payload,factor):
    # Explicit conversion only. Descriptor scale is NOT a physical-unit estimate.
    from copy import deepcopy
    p=deepcopy(payload)
    for key in ['rest_xyz','local_xyz','bind_local_xyz']:
        if key in p:p[key]=(np.asarray(p[key])*factor).tolist()
    p['area']=(np.asarray(p['area'])*factor**2).tolist()
    p['geometry_scale']*=factor;p['pose_scale']*=factor
    if p.get('metric_valid'):p['meters_per_unit']/=factor
    return p

def load_scaling_bodies(root,config,device,include_surface=True):
    root=Path(root);pilot=Path(config['pilot_data_root'])
    old_q=pilot/config['qwen_cache']
    bodies,old_audit=load_bodies(pilot,old_q,config['qwen_sha256'],device)
    qpath=root/config.get('extended_qwen_cache','qwen/frozen_features.pt')
    if sha256(qpath)!=config['extended_qwen_sha256']:raise ValueError('extended frozen Qwen cache changed')
    features=torch.load(qpath,map_location='cpu',weights_only=True)
    qwen=features['joint_names'];patch_qwen=features['surface_texts']
    original=torch.load(old_q,map_location='cpu',weights_only=True)
    if any(n not in qwen or not torch.equal(qwen[n],v) for n,v in original.items()):raise ValueError('original584 frozen vectors changed')
    heldout=set(config['xwbc_body_holdout']);found=set()
    for b in bodies:
        previous=b.reach;b.reach=static_reach(b.parents,b.offsets)
        b.arrays['pose9']=b.arrays['pose9'].copy();b.arrays['pose9'][...,6:]*=previous/b.reach
        b.arrays['source_clip_split']=b.arrays['split'].copy()
        if b.name in heldout:
            found.add(b.name);b.arrays['split']=np.full(len(b.arrays['pose9']),2,np.int8)
        b.metadata.update(normalization='geometry_only_max_root_path',previous_motion_scale_unused=previous,
            source_native_unit='cm' if b.domain=='xwbc' else 'unknown_canonical',actual_dataset='XWBC' if b.domain=='xwbc' else 'UniMate')
        b.setup(qwen,device)
    if found!=heldout:raise ValueError('entire-body holdout not present')
    for name in ['adroit_pen','relay_franka_full_v2']:bodies.append(external_body(root,name,qwen,device))
    audits=[];artifacts=dict(old_audit['input_manifest_hashes'])
    for b in bodies:
        static=None;path=root/'surface/patches'/f'{b.name}.json'
        if include_surface and path.exists():
            p=read_json(path)
            if b.metadata.get('source_native_unit')=='m':p=scaled_surface_payload(p,100.)
            # Normalization is geometry-only, recomputed identically for A/B; patch geometry is unchanged.
            p['pose_scale']=b.reach
            p['qwen_sem']=torch.stack([patch_qwen[k] for k in p['semantic_text_key']])
            p['semantic_valid']=[True]*len(p['semantic_text_key'])
            static=prepare_static_surface(p,len(b.parents),device,max_patches=config['surface']['max_patches'])
            static.update(asset_sha256=sha256(path),qwen_cache_sha256=config['extended_qwen_sha256'])
            artifacts['surface/patches/'+path.name]=sha256(path)
        b.metadata['surface_patch_count']=0 if static is None else int(static['valid'].sum())
        attach_geometry(b,static)
        if 'slide_native' in b.arrays:
            from prismatic.kinematics import mixed_forward_kinematics
            fk_error=0.
            for start in range(0,len(b.arrays['pose9']),2048):
                rr=np.arange(start,min(start+2048,len(b.arrays['pose9'])))
                with torch.no_grad():
                    predicted=mixed_forward_kinematics(b.tensor('global_rot6',rr,device),b.geom(len(rr)),b.tensor('slide_native',rr,device))['joint_position']/b.reach
                    error=(predicted-b.tensor('pose9',rr,device)[...,6:]).norm(dim=-1)
                fk_error=max(fk_error,float(error.max()))
            if fk_error>2e-5:raise ValueError('typed FK payload mismatch')
        else:fk_error=b.audit_fk()
        a=b.arrays;N,J=a['pose9'].shape[:2]
        audits.append({'body':b.name,'dataset':b.metadata['actual_dataset'],'task_domain_key':b.domain,'joints':J,'frames':N,
            'frames_by_split':{str(s):int((a['split']==s).sum()) for s in [0,1,2]},
            'groups_by_split':{str(s):len(np.unique(a['group'][a['split']==s])) for s in [0,1,2]},
            'body_holdout':not len(b.eligible(True)), 'normalization_reach':b.reach,'normalization':'static_geometry_only',
            'metric':b.metric,'kinematic_unit':b.metadata.get('length_unit','canonical_unknown'),
            'fk_max_error_over_reach':fk_error,'payload_sha256':b.metadata['payload_sha256'],
            'valid_labels':{'rotation_joints':int(a['rotation_valid'].sum()),'position_joints':int(a['position_valid'].sum()),
                'root_height_frames':int(a['root_height_valid'].sum()),'contact_joints':int(a['contact_valid'].sum()),
                'ground_frames':int(a['ground_valid'].sum()),'surface_patches':b.metadata['surface_patch_count']},
            'endpoint':b.endpoint,'stride':b.metadata.get('training_stride',1),
            'motion_time_provenance':b.metadata.get('time_source','original sealed payload'),
            'cross_body_labels':b.domain=='xwbc','physical_robot_validation':False})
        if b.name in {'adroit_pen','relay_franka_full_v2'}:
            artifacts['data/payloads/'+b.name+'.json']=b.metadata['metadata_sha256']
            artifacts['data/payloads/'+b.name+'.npz']=b.metadata['payload_sha256']
    order=[b.name for b in bodies if b.domain=='xwbc' and len(b.eligible(True))]
    if len(order)!=config['model']['n_judges']:raise ValueError('training-only judge class count')
    patch_train=[b.name for b in bodies if b._surface_static is not None and len(b.eligible(True))]
    if include_surface and len(patch_train)<config['surface']['required_real_training_bodies']:raise ValueError('insufficient real mesh training bodies')
    return bodies,{'schema':'six_run_data_v1','bodies':audits,'judge_body_order':order,'patch_training_bodies':patch_train,
        'artifacts_sha256':artifacts,'qwen_sha256':config['extended_qwen_sha256'],
        'body_holdout':sorted(heldout),'normalization':'geometry-only; no heldout motion fitting',
        'split_policy':'source clips/episodes intact; unseen humanoid bodies reported separately by source-clip exposure',
        'cross_dataset_ancestry':'unknown; no claim of independence based solely on names',
        'upstream_data_audit':old_audit}
