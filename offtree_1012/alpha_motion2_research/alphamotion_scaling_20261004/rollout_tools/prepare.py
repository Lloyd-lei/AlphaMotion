"""Derive visualization bundles from immutable P1 cases; never load a neural model."""
from pathlib import Path
import argparse, dataclasses, fcntl, hashlib, json, os, sys
import numpy as np

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def write(p,x):
    p=Path(p);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n');t.replace(p)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--run',action='append');a=ap.parse_args()
    r=a.root;sys.path[:0]=[str(r/'evaluation_p1'),str(r/'evaluation'),str(r)]
    import torch,mujoco
    from protocol import configuration,normalize_adapter,q_from_normalized
    from hinge_dof_adapter import SingleHingeTree,matrix_from_rot6
    from mixed_dof_adapter import MixedJointTree
    from trajectory_refinement import VisibleSE3,LegalConstraint,refine_trajectory,visible_se3_digest
    from prismatic.kinematics import mixed_forward_kinematics
    torch.set_num_threads(1)
    lease=(r/'gpu_queue.lock').open('a+');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if json.loads((r/'evaluation_phase2_v2/p1/logs/state.json').read_text())['status']!='completed':raise ValueError('P1 owner not completed')
    plan=json.loads((r/'evaluation_phase2_v2/p1/execution_plan.json').read_text())
    for rel,h in plan['source_sha256'].items():
        if sha(r/rel)!=h:raise ValueError('frozen source changed '+rel)
    a.output.mkdir(parents=True,exist_ok=True)
    runs=a.run or [f'{v}_{s}_seed0' for s in 'SML' for v in 'AB']
    for body,cls in [('adroit_pen',SingleHingeTree),('relay_franka_full_v2',MixedJointTree)]:
        md=r/'data/payloads'/f'{body}.json';doc=json.loads(md.read_text());m=doc['model'];tree=cls(md)
        payload=r/'data/payloads'/f'{body}.npz'
        if sha(payload)!=doc['payload_sha256']:raise ValueError('payload changed')
        with np.load(payload,allow_pickle=False) as z:
            g={'parents':torch.tensor(z['parents'],device='cuda'),'kinematic_rest_off':torch.tensor(z['rest_offsets'].astype(np.float32)*100,device='cuda'),'root_fixed':True}
            if 'joint_type' in z:
                for k in ['joint_type','prismatic_axis','prismatic_fixed_rotation']:g[k]=torch.tensor(z[k],device='cuda')
                for k in ['slide_lower','slide_upper','slide_reference']:g[k]=torch.tensor(z[k]*100,device='cuda')
        actual=tree.verified_adapter(r/'evaluation'/plan['adapter_receipts'][body])
        if body=='adroit_pen':
            fn=actual.project_free;actual=dataclasses.replace(actual,project_free=lambda x:fn(x[...,:6]))
        adapter=normalize_adapter(actual);cfg=configuration()
        assets=a.output/body;assets.mkdir(exist_ok=True)
        xml=Path(m['model_xml'])
        if sha(xml)!=m['model_xml_sha256']:raise ValueError('model xml changed')
        model=mujoco.MjModel.from_xml_path(str(xml));mujoco.mj_saveModel(model,str(assets/'model.mjb'),None)
        write(assets/'metadata.json',doc)
        for run in runs:
            folder=r/'evaluation_phase2_v2/p1/results/manipulation'/run/'checkpoint_step6000'
            candidates=[]
            for p in (folder/body).glob('*/case_result.json'):
                d=json.loads(p.read_text())
                if d.get('provenance',{}).get('reference_cohort')=='strict_limit_valid_control':candidates.append((d['case_id'],p,d))
            _,source,case=sorted(candidates)[0]
            dest=assets/run;dest.mkdir(exist_ok=True)
            if (dest/'manifest.json').exists():
                print('existing bundle retained',body,run,flush=True);continue
            arr=dict(np.load(source.parent/'case_arrays.npz',allow_pickle=False))
            t=lambda x:torch.tensor(x,device='cuda')
            oldcurve=json.loads((source.parent/'constrained_refinement/refinement_curve.json').read_text())
            cond=VisibleSE3(t(arr['condition_joint_indices']),t(arr['condition_position_cm']),t(arr['condition_global_rotation']),t(arr['condition_position_valid']),t(arr['condition_rotation_valid']),tuple(m['joint_names']),oldcurve['protocol']['condition_source'],'cm')
            if visible_se3_digest(cond)!=case['condition_sha256']:raise ValueError('condition identity mismatch')
            initial=torch.cat([t(arr['initial_model_global_rot6']),t(arr['initial_model_slide_cm'])[...,None]],-1)
            with torch.no_grad():
                raw=mixed_forward_kinematics(initial[...,:6],g,initial[...,6]);q0=adapter.clamp(adapter.project_free(initial))
            snapshots={'reference_position':arr['reference_position_cm'],'reference_rotation':matrix_from_rot6(t(arr['reference_global_rot6'])).cpu().numpy(),'native_position':raw['joint_position'].cpu().numpy(),'native_rotation':raw['joint_rotation'].cpu().numpy()}
            for key in ['frame_ids','timestamps_seconds','condition_joint_indices','condition_position_cm','condition_global_rotation','condition_position_valid','condition_rotation_valid']:snapshots[key]=arr[key]
            checks={};files={str(source):sha(source),str(source.parent/'case_arrays.npz'):sha(source.parent/'case_arrays.npz'),str(md):sha(md),**actual.source_files_sha256}
            for stage,init in [('constrained_refinement',q0),('default_valid_q',adapter.default_q.to(initial).expand(len(initial),-1).clone())]:
                def observe(step,state,p,R):
                    if step in (0,20,100):
                        prefix=('q' if stage=='constrained_refinement' else 'default')+str(step)
                        snapshots[prefix+'_position']=p.cpu().numpy();snapshots[prefix+'_rotation']=R.cpu().numpy();snapshots[prefix+'_native_q']=q_from_normalized(state,actual).cpu().numpy()
                known=LegalConstraint('verified_joint_limits','known_joint_limits',adapter.violations,'; '.join(sorted(adapter.source_files_sha256)),validity_description='all actual model DOF',unit='per-coordinate declared q unit')
                result=refine_trajectory(init,adapter.decode_q,cond,config=cfg,constraints=(known,),state_projector=adapter.clamp,projection_source='; '.join(sorted(adapter.source_files_sha256)),state_observer=observe)
                original=source.parent/stage/'refinement_arrays.npz';old=np.load(original,allow_pickle=False);files[str(original)]=sha(original)
                delta=float(np.max(np.abs(result['final_state'].cpu().numpy()-old['final_state'])))
                curve=source.parent/stage/'refinement_curve.json';saved=json.loads(curve.read_text());files[str(curve)]=sha(curve)
                metric_delta=max(abs(float(rec['metrics'][k])-float(saved['records'][i]['metrics'][k])) for i,rec in enumerate(result['records']) for k in ['selected_position_mean','selected_rotation_mean_deg'])
                if delta>2e-6 or metric_delta>2e-4:raise ValueError(f'replay differs {body} {run} {stage}: {delta} {metric_delta}')
                checks[stage]={'max_final_normalized_state_delta':delta,'max_curve_metric_delta':metric_delta,'all_101_steps_checked':True}
            np.savez_compressed(dest/'arrays.npz',**snapshots)
            summary=json.loads((folder/'evaluation_summary.json').read_text());files[str(folder/'evaluation_summary.json')]=sha(folder/'evaluation_summary.json')
            manifest={'schema':'alphamotion-rollout-input-v1','body':body,'run':run,'case_id':case['case_id'],'selection':'first lexicographic strict_limit_valid_control case; no result-based selection','checkpoint_sha256':summary['checkpoint_sha256'],'checkpoint_step':6000,'condition_sha256':case['condition_sha256'],'frame':'canonical_root_relative_y_up','position_unit':'cm','rotation':'own_joint_global_rest_relative_SO3','time_unit':'seconds','time_source':doc['time_source'],'native_frame_stride':case['provenance']['native_frame_stride'],'frame_ids':arr['frame_ids'].tolist(),'timestamps_seconds':arr['timestamps_seconds'].tolist(),'arrays_sha256':sha(dest/'arrays.npz'),'model_sha256':sha(assets/'model.mjb'),'metadata_sha256':sha(assets/'metadata.json'),'source_xml_sha256':sha(xml),'source_files_sha256':files,'replay_validation':checks,'training_updates':0,'neural_inference':False,'simulator_steps':0,'evidence':'kinematic playback only; free native posture is not feasible q; no object/contact/dynamics/physical execution','source_code_sha256':{str(Path(__file__)):sha(__file__),**{str(r/k):v for k,v in plan['source_sha256'].items()}}}
            write(dest/'manifest.json',manifest);print(json.dumps({'prepared':str(dest),'checks':checks}),flush=True)

if __name__=='__main__':main()
