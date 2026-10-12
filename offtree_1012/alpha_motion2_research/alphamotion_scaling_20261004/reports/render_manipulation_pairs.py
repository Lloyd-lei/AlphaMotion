"""Visualize saved reference and final FK arrays, without inference or optimization."""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from render_locomotion_pairs import pose

RUNS=[f'{v}_{s}_seed0' for s in 'SML' for v in 'AB']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--step',type=int,default=3000)
    p.add_argument('--body',action='append',required=True);p.add_argument('--cohort',default='strict_limit_valid_control')
    p.add_argument('--detail',action='store_true',help='Show the named palm/gripper subtree with shared limits')
    p.add_argument('--results-root',type=Path)
    p.add_argument('--comparison-results-root',type=Path,help='Use common display bounds with this other protocol; never select a different case/frame')
    p.add_argument('--protocol-label',default='P0')
    p.add_argument('--diagnostic-input',type=Path);a=p.parse_args()
    runs=['A_S_seed0'] if a.diagnostic_input else RUNS
    results_root=a.results_root or a.root/'evaluation/results/manipulation'
    folders={run:a.diagnostic_input or results_root/run/f'checkpoint_step{a.step:04d}' for run in runs}
    reports={run:json.loads((folder/'evaluation_summary.json').read_text()) for run,folder in folders.items()}
    a.output.mkdir(parents=True,exist_ok=True);receipts=[]
    for body in a.body:
        cases={}
        for run,folder in folders.items():
            cases[run]={}
            for path in folder.rglob('case_result.json'):
                doc=json.loads(path.read_text())
                if doc['body']==body and doc.get('provenance',{}).get('reference_cohort')==a.cohort:
                    cases[run][doc['case_id']]=(path,doc)
        first=next(iter(cases))
        expected=set(cases[first])
        if not expected or any(set(v)!=expected for v in cases.values()):
            raise ValueError('Complete identical cohort case sets required')
        chosen=sorted(expected)[0]  # Fixed rule independent of performance.
        selected={run:cases[run][chosen] for run in runs}
        if any(d.get('status')=='failed_case_retained' for _,d in selected.values()):
            raise ValueError('Preselected visual case failed; no easier substitute')
        files={};arrays={};stage_arrays={}
        for run,(path,doc) in selected.items():
            source=path.parent/'case_arrays.npz';files[str(source)]=sha(source)
            arrays[run]=dict(np.load(source,allow_pickle=False));stage_arrays[run]={}
            for stage in ['raw_free_posture','constrained_refinement','default_valid_q']:
                source=path.parent/stage/'refinement_arrays.npz';files[str(source)]=sha(source)
                stage_arrays[run][stage]=dict(np.load(source,allow_pickle=False))
                if reports[run]['refinement_config']['steps']!=doc['stages'][stage]['steps']:
                    raise ValueError('Refinement budget mismatch')
                if doc['stages'][stage]['final_observation'] is None:
                    raise ValueError('Preselected final refinement stage failed; no substitute')
                if stage in ['constrained_refinement','default_valid_q']:
                    limits=doc['stages'][stage]['final_constraints']['verified_joint_limits']
                    if limits['values']<=0 or limits['violations']!=0:
                        raise ValueError('Cannot label known bounds satisfied')
        reference=arrays[first]
        keys=['reference_position_cm','reference_global_rot6','frame_ids','timestamps_seconds',
              'condition_joint_indices','condition_position_cm','condition_position_valid',
              'condition_global_rotation','condition_rotation_valid']
        for run,arr in arrays.items():
            for key in keys:
                if not np.array_equal(arr[key],reference[key]):raise ValueError('Unmatched reference/conditions: '+key)
        payload=a.root/'data/payloads'/f'{body}.npz'
        with np.load(payload,allow_pickle=False) as data:
            parents=data['parents'].copy();names=list(data['joint_names'].astype(str))
        files[str(payload)]=sha(payload)
        ref=reference['reference_position_cm'];t=len(ref)//2
        indices=list(range(len(parents)))
        if a.detail:
            anchor=names.index({'adroit_pen':'palm','relay_franka_full_v2':'panda0_link7'}[body])
            indices=[anchor]
            for j,parent in enumerate(parents):
                if parent in indices:indices.append(j)
        lookup={j:i for i,j in enumerate(indices)}
        shown_parents=np.asarray([lookup.get(int(parents[j]),-1) for j in indices])
        target=reference['condition_position_cm'][t][reference['condition_position_valid'][t]]
        target_indices=reference['condition_joint_indices'][t]
        target_names=[names[int(j)] for j in target_indices]
        default_delta=max(float(np.max(np.abs(stage_arrays[run]['default_valid_q']['final_positions']-stage_arrays[first]['default_valid_q']['final_positions']))) for run in runs)
        if default_delta>1e-4:raise ValueError('Default q baseline differs across matched evaluations; inspect before shared display')
        paths=[]
        for stage in ['raw_free_posture','constrained_refinement']:
            points=[ref]+[value[s]['final_positions'] for value in stage_arrays.values() for s in [stage,'default_valid_q']]
            shown=[value[t,indices] for value in points]
            if a.comparison_results_root:
                for run in runs:
                    other=a.comparison_results_root/run/f'checkpoint_step{a.step:04d}'/body/chosen
                    with np.load(other/'case_arrays.npz',allow_pickle=False) as comp:
                        for key in keys:
                            if not np.array_equal(comp[key],reference[key]):raise ValueError('Cross-protocol visual reference differs')
                    for s in [stage,'default_valid_q']:
                        source=other/s/'refinement_arrays.npz';files[str(source)]=sha(source)
                        with np.load(source,allow_pickle=False) as comp:shown.append(comp['final_positions'][t,indices])
            allp=np.concatenate(shown,0);lo=allp.min(0);hi=allp.max(0);mid=(lo+hi)/2
            radius=max(float((hi-lo).max())*.55,1e-3);limits=np.stack([mid-radius,mid+radius],-1)
            sizes=['S'] if a.diagnostic_input else list('SML')
            cols=3 if a.diagnostic_input else 4
            fig=plt.figure(figsize=(cols*3.5,len(sizes)*3.7+1.0))
            fig.subplots_adjust(left=.015,right=.985,bottom=.065,top=.87,wspace=.04,hspace=.23)
            for row,size in enumerate(sizes):
                run_a=f'A_{size}_seed0';run_b=f'B_{size}_seed0'
                values=[('Reference posture',ref[t],'#343d46'),
                    ('Default q + refinement',stage_arrays[run_a]['default_valid_q']['final_positions'][t],'#777777'),
                    (f'A-{size}',stage_arrays[run_a][stage]['final_positions'][t],'#178f89')]
                if not a.diagnostic_input:values.append((f'B-{size}',stage_arrays[run_b][stage]['final_positions'][t],'#ce7140'))
                for col,(label,value,color) in enumerate(values):
                    ax=fig.add_subplot(len(sizes),cols,row*cols+col+1,projection='3d')
                    pose(ax,value[indices],shown_parents,color,limits,None if col==0 else ref[t,indices])
                    ax.scatter(target[:,0],target[:,2],target[:,1],c='#a62f3f',marker='x',s=48,depthshade=False)
                    for number,point in enumerate(target,1):
                        delta=radius*.06*(1 if number==1 else -1)
                        ax.text(point[0]+delta,point[2],point[1]+delta,str(number),color='#a62f3f',fontsize=10)
                    ax.text2D(.5,1.01,label,transform=ax.transAxes,ha='center',fontsize=11)
                    if col==0:
                        annotation='\n'.join(f'{i+1}: {n}' for i,n in enumerate(target_names))
                    else:
                        run=run_a if col in (1,2) else run_b
                        selected_stage='default_valid_q' if col==1 else stage
                        predR=stage_arrays[run][selected_stage]['final_rotations'][t,target_indices]
                        gtR=reference['condition_global_rotation'][t]
                        Q=np.swapaxes(predR,-1,-2)@gtR
                        skew=np.stack([Q[:,2,1]-Q[:,1,2],Q[:,0,2]-Q[:,2,0],Q[:,1,0]-Q[:,0,1]],-1)
                        angles=np.degrees(np.arctan2(np.linalg.norm(skew,axis=-1)*.5,np.clip((np.trace(Q,axis1=-2,axis2=-1)-1)*.5,-1,1)))
                        error=np.linalg.norm(value[target_indices]-reference['condition_position_cm'][t],axis=-1)*10
                        annotation='\n'.join(f'{i+1}: {e:.1f} mm / {theta:.1f} deg' for i,(e,theta) in enumerate(zip(error,angles)))
                    ax.text2D(.5,-.005,annotation,transform=ax.transAxes,ha='center',va='top',fontsize=8)
            N=reports[first]['refinement_config']['steps'];heading='Free posture; q feasibility uncertified' if stage=='raw_free_posture' else 'Joint-space refinement; known bounds satisfied'
            label={'adroit_pen':'Adroit hand','relay_franka_full_v2':'Franka arm and gripper'}[body]
            scope='DIAGNOSTIC INTEGRATION ONLY\n' if a.diagnostic_input else ''
            detail='Palm / gripper detail' if a.detail else 'Full skeleton'
            shared='Common P0/P1 display scale' if a.comparison_results_root else 'Common display scale across all six models'
            fig.suptitle(f'{scope}{a.protocol_label} | {label} | {heading} | {N} updates\n{detail}; fixed middle frame; red numbers = named targets; gray = reference\n{shared}; root-relative cm; annotations show position / orientation error\nContact, dynamics and physical execution untested',fontsize=11,y=.99)
            suffix='__detail' if a.detail else ''
            path=a.output/f'{body}__{a.cohort}__{stage}{suffix}.png';fig.savefig(path,dpi=160,bbox_inches='tight',pad_inches=.15);plt.close(fig);paths.append(path)
        receipts.append(dict(body=body,case_id=chosen,cohort=a.cohort,selection='lexicographic first case, independent of outcome',
            protocol_label=a.protocol_label,comparison_results_root=str(a.comparison_results_root) if a.comparison_results_root else None,
            display_bounds='same preselected middle frame, union of all shown protocols/models; no metric or sample selection change',
            frame_id=int(reference['frame_ids'][t]),source_files_sha256=files,shown_joint_names=[names[j] for j in indices],
            default_baseline_max_position_delta_cm=default_delta,artifacts=[{'path':str(x),'sha256':sha(x)} for x in paths],
            visual_review='pending',scope='diagnostic integration only' if a.diagnostic_input else 'saved formal FK visualization'))
    path=a.output/('manipulation_render_detail_receipt.json' if a.detail else 'manipulation_render_receipt.json');path.write_text(json.dumps(receipts,indent=2)+'\n')
    print(json.dumps({'receipt':str(path),'bodies':a.body,'visual_review':'pending'}))


if __name__=='__main__':main()
