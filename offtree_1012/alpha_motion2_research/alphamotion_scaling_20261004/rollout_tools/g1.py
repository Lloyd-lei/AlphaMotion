"""Bounded G1 adapter: existing saved pair, source-verified rigid mesh binding."""
from pathlib import Path
import argparse,json,os,sys,tempfile,shutil
os.environ.setdefault('MUJOCO_GL','egl')
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import mujoco
from prepare import sha,write
from encode import encode
from contracts import capacity_guard

def prepare(root,out):
    sys.path[:0]=[str(root/'reports'),str(root/'surface')]
    from render_locomotion_pairs import load_matched
    from build_verified_patches import random_states
    runs=[f'{v}_{s}_seed0' for s in 'SML' for v in 'AB'];data,record,files=load_matched(root/'evaluation_phase2_v2/p0/locomotion',runs,6000,'unitree_g1_29dof')
    patchpath=root/'surface/patches/unitree_g1_29dof.json';patch=json.loads(patchpath.read_text());ka=patch['kinematic_adapter'];A=np.array(patch['frame_provenance']['AX'])
    if patch['binding_audit']['status']!='passed' or sha(ka['xml'])!=ka['xml_sha256']:raise ValueError('G1 mesh provenance stale')
    for p in patch['patch_sources']:
        if sha(p['mesh']['path'])!=p['mesh']['sha256']:raise ValueError('G1 source mesh changed')
    m=mujoco.MjModel.from_xml_path(ka['xml']);d=mujoco.MjData(m);keep=ka['body_ids'];frame=ka['frame_body_ids'];bind={};error=0.
    for si,q in enumerate(random_states(m)):
        d.qpos[:]=q;mujoco.mj_forward(m,d)
        R=A@d.xmat[frame].reshape(-1,3,3)@A.T;P=(d.xpos[keep]-d.xpos[keep[0]])@A.T*100
        for source in patch['patch_sources']:
            gid=source['geom_id'];j=ka['joint_names'].index(source['joint_name'])
            localR=R[j].T@A@d.geom_xmat[gid].reshape(3,3)@A.T;localP=R[j].T@((d.geom_xpos[gid]-d.xpos[keep[0]])@A.T*100-P[j])
            if not si:bind[gid]={'joint':j,'rotation':localR.tolist(),'position_cm':localP.tolist()}
            else:error=max(error,float(np.max(np.abs(localR-bind[gid]['rotation']))),float(np.max(np.abs(localP-bind[gid]['position_cm']))))
    if error>1e-5:raise ValueError('G1 binding not rigid')
    out.mkdir(parents=True,exist_ok=True);mujoco.mj_saveModel(m,str(out/'model.mjb'),None)
    first=data[runs[0]];arr={'frame_ids':first['frame_ids'],'timestamps_seconds':first['timestamps_seconds'],'parents':first['target_parents'],'reference_position':first['reference_position_native'],'reference_rotation':first['reference_global_rotation']}
    if list(first['target_joint_names'])!=ka['joint_names'] or not np.array_equal(first['target_parents'],ka['parents']):raise ValueError('G1 skeleton identity differs')
    for run,value in data.items():arr[run+'_position']=value['predicted_position_native'];arr[run+'_rotation']=value['predicted_global_rotation']
    np.savez_compressed(out/'arrays.npz',**arr)
    write(out/'manifest.json',{'schema':'alphamotion-g1-rollout-v1','runs':runs,'record':record,'frame':'canonical_root_relative_y_up','position_unit':'cm','AX':A.tolist(),'bindings':bind,'binding_max_error':error,'source_files_sha256':{str(p):sha(p) for p in [patchpath,*files.values()]},'mesh_xml_sha256':ka['xml_sha256'],'mesh_assets_sha256':{p['mesh']['path']:p['mesh']['sha256'] for p in patch['patch_sources']},'arrays_sha256':sha(out/'arrays.npz'),'model_sha256':sha(out/'model.mjb'),'time_policy':'source evaluation cache timestamps, 1x; original acquisition rate not independently revalidated','source_code_sha256':sha(__file__)})

def render(bundle,out,font):
    cap=capacity_guard(bundle);man=json.loads((bundle/'manifest.json').read_text())
    if man['schema']!='alphamotion-g1-rollout-v1' or man['position_unit']!='cm' or man['frame']!='canonical_root_relative_y_up':raise ValueError('G1 contract mismatch')
    for fn,h in [('arrays.npz',man['arrays_sha256']),('model.mjb',man['model_sha256'])]:
        if sha(bundle/fn)!=h:raise ValueError('G1 bundle corrupt')
    if out.exists():raise ValueError('G1 output exists; preserve prior evidence')
    out.parent.mkdir(exist_ok=True,parents=True);temp=Path(tempfile.mkdtemp(prefix='.g1-',dir=out.parent));frames=temp/'frames';frames.mkdir()
    z=dict(np.load(bundle/'arrays.npz',allow_pickle=False));A=np.array(man['AX']);m=mujoco.MjModel.from_binary_path(str(bundle/'model.mjb'));d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    m.light_active[:]=False;m.vis.headlight.ambient[:]=.25;m.vis.headlight.diffuse[:]=.65
    renderer=mujoco.Renderer(m,height=390,width=480,max_geom=10000);opt=mujoco.MjvOption();opt.geomgroup[:]=1
    cam=mujoco.MjvCamera();cam.type=mujoco.mjtCamera.mjCAMERA_FREE;cam.azimuth=145;cam.elevation=-13
    stages=['reference',*man['runs']];points=np.concatenate([z[s+'_position'] for s in stages],0)/100@A
    lo=points.min((0,1));hi=points.max((0,1));cam.lookat[:]=(lo+hi)/2;cam.distance=float(np.max(hi-lo))*1.9
    def mesh(s,t):
        renderer.update_scene(d,camera=cam,scene_option=opt);scene=renderer.scene;keepgeom=[]
        P=z[s+'_position'][t];R=z[s+'_rotation'][t]
        for k in range(scene.ngeom):
            g=scene.geoms[k];b=man['bindings'].get(str(g.objid)) if g.objtype==mujoco.mjtObj.mjOBJ_GEOM else None
            if b is None:continue
            j=b['joint'];g.pos[:]=A.T@(P[j]+R[j]@np.array(b['position_cm']))/100;g.mat[:]=A.T@R[j]@np.array(b['rotation'])@A;keepgeom.append(k)
        for i,k in enumerate(keepgeom):
            if i!=k:
                x=scene.geoms[k];y=scene.geoms[i]
                for f in ('type','dataid','matid','objtype','objid','category','texcoord','segid','emission','specular','shininess','reflectance','camdist','modelrbound','transparent'):setattr(y,f,getattr(x,f))
                for f in ('size','pos','mat','rgba'):getattr(y,f)[:]=getattr(x,f)
        scene.ngeom=len(keepgeom)
        for j,p in enumerate(z['parents']):
            if p<0:continue
            a=z['reference_position'][t,p]/100@A;b=z['reference_position'][t,j]/100@A
            g=scene.geoms[scene.ngeom];mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_CAPSULE,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array([.12,.5,.65,.65]));mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_CAPSULE,.003,a,b);scene.ngeom+=1
        rgb=renderer.render().copy();renderer.enable_depth_rendering();depth=renderer.render();renderer.disable_depth_rendering();rgb[depth>=m.vis.map.zfar*m.stat.extent*.99]=[241,244,246];return Image.fromarray(rgb)
    try:
        for t in range(len(z['frame_ids'])):
            im=Image.new('RGB',(1920,1080),'#f1f4f6');draw=ImageDraw.Draw(im);F=lambda size:ImageFont.truetype(font,size)
            draw.text((24,16),'Unitree G1 · 人到机器人配对映射 · 六组模型 · 运动学回放 / 非仿真执行',font=F(30),fill='#153442')
            draw.text((24,62),f'已见本体 / 未见动作  |  checkpoint 6000  |  来源缓存时间轴 1×  |  t={z["timestamps_seconds"][t]-z["timestamps_seconds"][0]:.6f}s  |  原帧 {int(z["frame_ids"][t])}',font=F(23),fill='#374d5e')
            for k,s in enumerate(stages):
                x=k%4*480;y=112+k//4*450;im.paste(mesh(s,t),(x,y+34));draw=ImageDraw.Draw(im)
                label='记录参考' if s=='reference' else s.replace('_seed0','');draw.text((x+12,y),label,font=F(25),fill='#153442')
                err=np.linalg.norm(z[s+'_position'][t]-z['reference_position'][t],axis=-1).mean()*10;draw.text((x+12,y+414),f'全身平均位置误差 {err:.1f} mm',font=F(20),fill='#374d5e')
            draw.text((1460,605),'同一固定片段，保留全部帧\n浅蓝骨架为记录参考\n网格为源 G1 visual 几何\n\n自由姿态不是可执行 q\n根相对坐标；无地面接触判定\n原采集速率未独立复核',font=F(21),fill='#374d5e',spacing=14)
            draw.text((24,1038),'仅展示冻结网络输出；未新增推理或训练。没有仿真控制、动力学、接触或真机成功结论。',font=F(23),fill='#526672')
            im.save(frames/f'{t:06d}.png')
            if t==len(z['frame_ids'])//2:im.save(temp/'poster.jpg',quality=94)
        timing=encode(frames,z['timestamps_seconds'],temp/'rollout.mp4');shutil.rmtree(frames)
        write(temp/'output_manifest.json',{'schema':'alphamotion-rollout-output-v1','source':man,'time':timing,'capacity':cap,'artifacts_sha256':{f:sha(temp/f) for f in ['rollout.mp4','poster.jpg']},'visual_qa':'pending','source_code_sha256':sha(__file__)});temp.rename(out);print('G1 mesh video complete',out,flush=True)
    finally:renderer.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--font',required=True);a=p.parse_args()
    if a.root:prepare(a.root,a.bundle)
    render(a.bundle,a.output,a.font)
