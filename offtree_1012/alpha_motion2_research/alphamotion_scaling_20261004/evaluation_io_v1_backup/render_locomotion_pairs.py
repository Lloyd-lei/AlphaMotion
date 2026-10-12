"""Render matched saved FK trajectories; no inference or model modification."""
from pathlib import Path
import argparse,hashlib,json,subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def pose(ax,position,parents,color,limits,reference=None):
    if reference is not None:
        for j,p in enumerate(parents):
            if p>=0:ax.plot(reference[[p,j],0],reference[[p,j],2],reference[[p,j],1],color='#b8bcc2',linewidth=1,alpha=.6)
    for j,p in enumerate(parents):
        if p>=0:ax.plot(position[[p,j],0],position[[p,j],2],position[[p,j],1],color=color,linewidth=2)
    ax.scatter(position[:,0],position[:,2],position[:,1],c=color,s=9)
    ax.set(xlim=limits[0],ylim=limits[2],zlim=limits[1]);ax.set_box_aspect((1,1,1));ax.view_init(elev=14,azim=-65);ax.set_axis_off()

def load_matched(root,runs,step,target):
    reports={run:json.loads((root/run/f'checkpoint_step{step:04d}'/'evaluation_summary.json').read_text()) for run in runs}
    lists={run:{r['case_id']:r for r in d['records'] if r['target_body']==target and r['input_mode']=='full_source' and r['clip_exposure']=='heldout_clip'} for run,d in reports.items()}
    common=set.intersection(*(set(v) for v in lists.values()))
    if not common:raise ValueError('no common full-source heldout clip for '+target)
    case_id=sorted(common)[0] # Fixed rule, independent of model quality.
    if any(lists[run][case_id]['status']!='measured' for run in runs):raise ValueError('preselected visual case failed; retain failure instead of selecting an easier replacement')
    files={run:root/run/f'checkpoint_step{step:04d}'/case_id/'paired_arrays.npz' for run in runs}
    data={run:dict(np.load(p,allow_pickle=False)) for run,p in files.items()};first=data[runs[0]]
    for run,d in data.items():
        for key in ['frame_ids','timestamps_seconds','target_parents','reference_position_native','reference_global_rotation']:
            if not np.array_equal(d[key],first[key]):raise ValueError('unmatched recorded reference '+key+' '+run)
    return data,lists[runs[0]][case_id],files

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--target',action='append',required=True);p.add_argument('--step',type=int,default=3000);p.add_argument('--video',action='store_true');a=p.parse_args()
    runs=[f'{group}_{size}_seed0' for size in ['S','M','L'] for group in ['A','B']];a.output.mkdir(parents=True,exist_ok=True);receipt=[]
    for target in a.target:
        data,record,files=load_matched(a.root,runs,a.step,target);first=data[runs[0]];ref=first['reference_position_native'];parents=first['target_parents'];times=first['timestamps_seconds'];frames=[0,len(ref)//2,len(ref)-1]
        allp=np.concatenate([ref]+[d['predicted_position_native'] for d in data.values()],axis=0);lo=allp.min((0,1));hi=allp.max((0,1));center=(lo+hi)/2;radius=max(float((hi-lo).max())*.55,1e-3);limits=np.stack([center-radius,center+radius],-1)
        unit='cm' if 'position_error_mm' in first else 'canonical units';paths=[]
        task='Human-to-robot mapping' if record['task_kind']=='paired_mapping' else 'Same-body reconstruction'
        label={'unitree_g1_29dof':'Unitree G1','kuavo_s45':'Kuavo S45'}.get(target,target if unit=='cm' else 'UniMate '+record['target_morphology'])
        exposure=record['body_exposure'].replace('_',' ')
        for size in ['S','M','L']:
            fig=plt.figure(figsize=(9,9),layout='constrained')
            for ri,t in enumerate(frames):
                for ci,(name,value,color) in enumerate([('Reference',ref[t],'#343d46'),(f'A-{size}',data[f'A_{size}_seed0']['predicted_position_native'][t],'#178f89'),(f'B-{size}',data[f'B_{size}_seed0']['predicted_position_native'][t],'#ce7140')]):
                    ax=fig.add_subplot(3,3,ri*3+ci+1,projection='3d');pose(ax,value,parents,color,limits,None if ci==0 else ref[t]);ax.set_title(f'{name}  |  t={times[t]-times[0]:.3f}s',fontsize=11)
            fig.suptitle(f'{label}: {task}; {exposure}\nFull source pose; fixed continuous subclip from heldout motion\nShared root-relative {unit}; gray reference; first / middle / last frames',fontsize=12)
            path=a.output/f'{target}_{size}_paired_step{a.step}.png';fig.savefig(path,dpi=150);plt.close(fig);paths.append(path)
        if a.video:
            fig=plt.figure(figsize=(16,8),dpi=100);axes=[fig.add_subplot(2,4,i+1,projection='3d') for i in range(7)];fig.subplots_adjust(left=.01,right=.99,bottom=.03,top=.84,wspace=.02,hspace=.07)
            dt=float(np.median(np.diff(times)))
            if dt<=0 or not np.allclose(np.diff(times),dt,atol=1e-6,rtol=1e-4):raise ValueError('Constant-FPS movie cannot preserve these nonuniform source timestamps')
            fps=1/dt;path=a.output/f'{target}_six_models_step{a.step}.mp4'
            width,height=fig.canvas.get_width_height();cmd=['ffmpeg','-loglevel','error','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}','-r',str(fps),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p',str(path)]
            child=subprocess.Popen(cmd,stdin=subprocess.PIPE)
            try:
                for t in range(len(ref)):
                    values=[('Reference',ref[t],'#343d46')]+[(run.replace('_seed0',''),data[run]['predicted_position_native'][t],'#178f89' if run[0]=='A' else '#ce7140') for run in runs]
                    for ax,(name,value,color) in zip(axes,values):ax.clear();pose(ax,value,parents,color,limits,ref[t] if name!='Reference' else None);ax.set_title(name,fontsize=13)
                    fig.suptitle(f'{label} | {task} | Step {a.step}\nHeldout motion subclip, frame {int(first["frame_ids"][t])}, t={times[t]-times[0]:.3f}s\nFull source pose; source sample timing; root-relative FK; gray reference',fontsize=13)
                    fig.canvas.draw();rgb=np.asarray(fig.canvas.buffer_rgba())[...,:3].copy();child.stdin.write(rgb.tobytes())
            finally:child.stdin.close();rc=child.wait();plt.close(fig)
            if rc:raise RuntimeError('ffmpeg failed '+str(rc))
            paths.append(path)
        receipt.append({'target':target,'case_id':record['case_id'],'clip_id':record['clip_id'],'input_mode':'full_source','source_body':record['source_body'],'selection':'first lexicographic common heldout case_id, independent of outcome','body_exposure':record['body_exposure'],'frames':len(ref),'source_files':{run:{'path':str(path),'sha256':digest(path)} for run,path in files.items()},'artifacts':[{'path':str(x),'sha256':digest(x)} for x in paths],'visual_review':'pending'})
    out=a.output/'paired_render_receipt.json';out.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'receipt':str(out),'targets':a.target}))

if __name__=='__main__':main()
