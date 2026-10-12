"""One offline entry for single/batch mesh playback. Output directories are atomic."""
from pathlib import Path
import argparse,fcntl,hashlib,json,os,shutil,tempfile,traceback
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from prepare import sha,write
from contracts import load_bundle,cache_key,endpoint_metrics,STAGES,LABELS,capacity_guard
from encode import encode

def draw_frame(scene,m,z,meta,t,font):
    image=Image.new('RGB',(1920,1080),'#f1f4f6');d=ImageDraw.Draw(image);F=lambda size:ImageFont.truetype(font,size)
    body='Adroit 灵巧手' if m['body']=='adroit_pen' else 'Franka 完整机械臂与夹爪'
    title=f'{body}   |   {m["run"].replace("_seed0", "")}   |   运动学回放 / 非仿真执行'
    d.text((25,14),title,font=F(32),fill='#152b38')
    elapsed=z['timestamps_seconds'][t]-z['timestamps_seconds'][0]
    time_label='标称控制周期回放' if m['body']=='adroit_pen' else '按记录时间戳回放'
    d.text((25,60),f'{time_label}，1×  |  动作 t={elapsed:.6f} s  |  原帧 {int(z["frame_ids"][t])}  |  checkpoint 6000  |  优化步数固定在各栏标题',font=F(23),fill='#374d5e')
    for k,(s,label) in enumerate(zip(STAGES,LABELS)):
        x=(k%3)*640;y=100+(k//3)*451
        image.paste(Image.fromarray(scene.render(s,t)),(x,y+38))
        if scene.view=='detail':image.paste(Image.fromarray(scene.render(s,t,full=True)).resize((192,106)),(x+442,y+43))
        d=ImageDraw.Draw(image)
        d.rectangle((x+1,y,x+638,y+36),fill='#deE6EA');d.text((x+14,y+3),label,font=F(23),fill='#153442')
        mm,deg=endpoint_metrics(z,s,t);ix=z['condition_joint_indices'][t]
        for c,j in enumerate(ix):
            name=meta['model']['joint_names'][j];name={'thdistal':'thdistal 拇指末节','ffdistal':'ffdistal 食指末节'}.get(name,name);passed=mm[c]<=10 and deg[c]<=5
            d.text((x+12,y+392+c*24),f'{name}: {mm[c]:.1f} mm / {deg[c]:.1f}°  '+('达标' if passed else '未达标'),font=F(18),fill='#18715d' if passed else '#9c3546')
        if m['body']=='relay_franka_full_v2':
            ids=meta['aperture_indices'];l=ids['left_finger'];rr=ids['right_finger'];pa=ids['parent'];axis=np.array(next(e['axis_rest_canonical'] for e in meta['model']['mechanical_joints'] if e['skeleton_index']==l))
            opening=float((z[s+'_position'][t,l]-z[s+'_position'][t,rr])@(z[s+'_rotation'][t,pa]@axis))*10
            d.rectangle((x+398,y+362,x+638,y+390),fill='#f1f4f6');d.text((x+405,y+364),f'开口 {opening:.1f} mm',font=F(20),fill='#153442')
    caption='局部形状视角：各栏相机跟随掌心/夹爪；右上角保留共同相机全景，位置误差见 mm。' if scene.view=='detail' else '浅蓝：参考骨架  |  红/紫：两个末端目标与轨迹  |  绿：当前姿态的末端轨迹  |  网格来自同一源模型'
    d.text((25,1011),caption,font=F(22),fill='#284451')
    d.text((25,1042),'自由姿态可能违反关节约束；关节优化仅检查已知限位。开口为指根轴向间距。未验证碰撞、接触、抓取、动力学或真机。',font=F(20),fill='#526672')
    return image

def run_job(bundle,output,view,font):
    from scene import MeshScene
    m,z,meta=load_bundle(bundle);key=cache_key(bundle,view,font);capacity=capacity_guard(bundle)
    output=Path(output);output.mkdir(parents=True,exist_ok=True);name=m['body']+'__'+m['run']+'__'+view;dest=output/name
    lock=(output/(name+'.lock')).open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (dest/'output_manifest.json').exists():
        old=json.loads((dest/'output_manifest.json').read_text())
        if old['cache_key']==key and all(sha(dest/k)==v for k,v in old['artifacts_sha256'].items()):return {'status':'cached','output':str(dest)}
        raise ValueError('existing output invalid/stale; choose a new output root (never overwrite evidence)')
    temp=output/('.'+name+'-'+key[:16]+'.work');temp.mkdir(exist_ok=True);scene=None
    try:
        frames=temp/'frames';frames.mkdir(exist_ok=True);scene=MeshScene(Path(bundle).parent/'model.mjb',meta,z,view)
        statefile=temp/'frame_state.json';state=json.loads(statefile.read_text()) if statefile.exists() else {}
        for t in range(len(z['frame_ids'])):
            framefile=frames/f'{t:06d}.png'
            if str(t) in state and framefile.exists() and sha(framefile)==state[str(t)]:continue
            im=draw_frame(scene,m,z,meta,t,font);im.save(framefile);state[str(t)]=sha(framefile);write(statefile,state)
            if t==len(z['frame_ids'])//2:im.save(temp/'poster.jpg',quality=94)
        timing=encode(frames,z['timestamps_seconds'],temp/'rollout.mp4');shutil.rmtree(frames)
        manifest={'schema':'alphamotion-rollout-output-v1','cache_key':key,'input_manifest_sha256':sha(Path(bundle)/'manifest.json'),'input_manifest':m,'view':view,'stage_labels':dict(zip(STAGES,LABELS)),'motion_time_vs_optimization':'video advances source motion time only; each panel uses one fixed optimizer step','mesh_mapping_max_error':scene.transform_error,'mesh_source':'same hash-pinned compiled source MJCF, rigid body transform from independently validated canonical FK','native_mesh_boundary':'free per-link transforms, may be infeasible; not a q execution','camera':{'lookat':scene.cam.lookat.tolist(),'distance':scene.cam.distance,'azimuth':scene.cam.azimuth,'elevation':scene.cam.elevation},'time':timing,'artifacts_sha256':{f:sha(temp/f) for f in ['rollout.mp4','poster.jpg']},'visual_qa':'pending'}
        manifest['capacity_before']=capacity;manifest['visual_geomgroup']=scene.visual_group;manifest['camera_policy']='detail follows local part separately; inset and full use identical global camera across six stages; metrics never aligned' if view=='detail' else 'one fixed camera shared across six stages'
        write(temp/'output_manifest.json',manifest);temp.rename(dest);return {'status':'rendered','output':str(dest),'frames':timing['encoded_frames'],'duration':timing['duration_seconds'],'max_pts_error':timing['max_pts_error_seconds']}
    except Exception as e:
        write(temp/'failure.json',{'error':repr(e),'traceback':traceback.format_exc(),'committed':False});raise
    finally:
        if scene:scene.close()
        fcntl.flock(lock,fcntl.LOCK_UN)

def main():
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--bundle',type=Path);g.add_argument('--batch',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--view',choices=['full','detail'],default='full');p.add_argument('--font',required=True);a=p.parse_args()
    jobs=[{'bundle':str(a.bundle),'view':a.view}] if a.bundle else json.loads(a.batch.read_text())['jobs']
    for job in jobs:print(json.dumps(run_job(job['bundle'],a.output,job.get('view','full'),a.font),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
