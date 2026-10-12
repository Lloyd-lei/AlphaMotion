from pathlib import Path
import hashlib,json,importlib.metadata,subprocess,sys,os,copy
import numpy as np
from prepare import sha,write
STAGES=('reference','native','q0','q20','q100','default100')
LABELS=('记录参考','网络原始输出（自由姿态）','关节投影 / step 0','关节优化 / step 20','关节优化 / step 100','默认 q 起步 / step 100')

def load_bundle(path):
    path=Path(path);m=json.loads((path/'manifest.json').read_text())
    if m['schema']!='alphamotion-rollout-input-v1' or m['frame']!='canonical_root_relative_y_up' or m['position_unit']!='cm' or m['time_unit']!='seconds' or m['rotation']!='own_joint_global_rest_relative_SO3':raise ValueError('incompatible version, coordinate frame or units')
    for p,h in [(path/'arrays.npz',m['arrays_sha256']),(path.parent/'model.mjb',m['model_sha256']),(path.parent/'metadata.json',m['metadata_sha256'])]:
        if sha(p)!=h:raise ValueError('bad input reference/hash: '+str(p))
    z=dict(np.load(path/'arrays.npz',allow_pickle=False));meta=json.loads((path.parent/'metadata.json').read_text())
    f=z['frame_ids'];t=z['timestamps_seconds'];J=len(meta['model']['parents']);N=len(f)
    if N<2 or not np.array_equal(f,m['frame_ids']) or not np.array_equal(t,m['timestamps_seconds']):raise ValueError('frame/time manifest mismatch')
    if not np.isfinite(t).all() or not np.all(np.diff(t)>0) or not np.all(np.diff(f)==m['native_frame_stride']):raise ValueError('frame order/stride/time invalid')
    if len(set(meta['model']['joint_names']))!=J:raise ValueError('body joint names not unique')
    for s in STAGES:
        p=z[s+'_position'];R=z[s+'_rotation']
        if p.shape!=(N,J,3) or R.shape!=(N,J,3,3) or not np.isfinite(p).all() or not np.isfinite(R).all():raise ValueError('pose shape/finite')
        if not np.allclose(R.swapaxes(-1,-2)@R,np.eye(3),atol=2e-4) or not np.allclose(np.linalg.det(R),1,atol=2e-4):raise ValueError('rotation is not SO3')
    return m,z,meta

def cache_key(path,view,font):
    # Includes all runtime inputs, render settings, font and all package code.
    p=Path(path);h=hashlib.sha256()
    for q in [p/'manifest.json',p/'arrays.npz',p.parent/'metadata.json',p.parent/'model.mjb',Path(font),*[Path(__file__).parent/k for k in ('contracts.py','scene.py','render.py','encode.py')]]:h.update(sha(q).encode())
    versions={k:importlib.metadata.version(k) for k in ('numpy','Pillow','mujoco')}
    versions['ffmpeg']=subprocess.check_output(['ffmpeg','-version'],text=True).splitlines()[0];versions['python']=sys.version
    h.update(json.dumps(versions,sort_keys=True).encode())
    h.update(json.dumps({'view':view,'size':[1920,1080],'version':1},sort_keys=True).encode());return h.hexdigest()

def endpoint_metrics(z,stage,t):
    ix=z['condition_joint_indices'][t];p=z[stage+'_position'][t,ix];R=z[stage+'_rotation'][t,ix]
    Q=R.swapaxes(-1,-2)@z['condition_global_rotation'][t]
    skew=np.stack([Q[:,2,1]-Q[:,1,2],Q[:,0,2]-Q[:,2,0],Q[:,1,0]-Q[:,0,1]],-1)
    deg=np.degrees(np.arctan2(np.linalg.norm(skew,axis=-1)/2,np.clip((np.trace(Q,axis1=-2,axis2=-1)-1)/2,-1,1)))
    mm=np.linalg.norm(p-z['condition_position_cm'][t],axis=-1)*10
    return mm,deg

def capacity_guard(bundle,budget=512*1024**2):
    foundations=[p/'pretraining_foundation' for p in Path(bundle).parents if (p/'pretraining_foundation/configs/download_plan.json').exists()]
    if not foundations:return {'scope':'local offline render; no remote download volume involved'}
    root=foundations[0];sys.path.insert(0,str(root));import acquire
    plan,repos=acquire.load_plan(root/'configs/download_plan.json');follow,other=acquire.load_plan(root/'configs/download_followup_plan.json')
    if repos!=other:raise ValueError('download registry differs')
    combined=copy.deepcopy(plan);combined['stages']+=follow['stages'];st=os.statvfs('/home');cap=acquire.capacity(combined,repos,available=st.f_bavail*st.f_frsize)
    if cap['expected_unique_bytes']!=200505865539 or cap['available_bytes']<cap['remaining_bytes']+16*1024**3+budget:raise ValueError('insufficient guarded render/archive capacity')
    return {'combined_capacity':cap,'render_and_archive_budget_bytes':budget,'additional_reserve_bytes':16*1024**3}
