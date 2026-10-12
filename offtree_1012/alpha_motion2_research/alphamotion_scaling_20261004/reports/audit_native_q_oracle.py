"""CPU-only oracle implementation diagnostic. Never a formal model score.

Recorded q is permitted only in this isolated reachability/gradient audit.
No model is instantiated, no training weights or formal outputs are changed.
"""
from pathlib import Path
import argparse, dataclasses, datetime, hashlib, json, math, sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'evaluation'))
import run_scaling_eval as formal
from scaling_data import load_scaling_bodies
from trajectory_refinement import VisibleSE3, RefinementConfig, _targets, refine_trajectory
from prismatic.kinematics import mixed_forward_kinematics


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(2**20),b''):h.update(chunk)
    return h.hexdigest()


def objective(q,tree,condition,cfg):
    pos,rot=tree.decode_q(q)
    return _targets(pos,rot,condition,cfg)


def components(q,tree,condition,cfg):
    loss,metrics=objective(q,tree,condition,cfg)
    p,_=objective(q,tree,condition,dataclasses.replace(cfg,rotation_weight=0))
    r,_=objective(q,tree,condition,dataclasses.replace(cfg,position_weight=0))
    return dict(total=float(loss.detach()),position_term=float(p.detach()),rotation_term=float(r.detach()),metrics=metrics)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    torch.set_num_threads(1)
    config=json.loads((ROOT/'runs/A_S_seed0/config.json').read_text())
    bodies,audit=load_scaling_bodies(ROOT,config,torch.device('cpu'));by={b.name:b for b in bodies}
    source_files={};records=[];projections=[];cfg=RefinementConfig()
    for name,kind,receipt in [('adroit_pen',formal.SingleHingeTree,'adroit_dof_validation.json'),
                              ('relay_franka_full_v2',formal.MixedJointTree,'relay_franka_full_v2_dof_validation.json')]:
        metadata=ROOT/'data/payloads'/f'{name}.json';payload=metadata.with_suffix('.npz')
        tree=kind(metadata);adapter=tree.verified_adapter(ROOT/'evaluation'/receipt)
        body=by[name];formal.validate_loader_adapter(body,tree)
        source_files[str(metadata)]=sha(metadata);source_files[str(payload)]=sha(payload)
        assert source_files[str(payload)]==tree.document['payload_sha256']
        with np.load(payload,allow_pickle=False) as z:
            saved_q=z['recorded_robot_qpos'][:,tree.q_addresses].copy()
            groups=z['source_group_index'].copy();frames=z['frame_index'].copy()
        lookup={(int(g),int(f)):i for i,(g,f) in enumerate(zip(groups,frames))}
        paths=sorted((ROOT/'evaluation/results/manipulation/A_S_seed0/checkpoint_step3000'/name).glob('*/case_result.json'))
        for path in paths:
            case=json.loads(path.read_text())
            if case['provenance']['reference_cohort']!='strict_limit_valid_control':continue
            source_files[str(path)]=sha(path)
            arrpath=path.parent/'case_arrays.npz';source_files[str(arrpath)]=sha(arrpath)
            with np.load(arrpath,allow_pickle=False) as z:arr={k:z[k].copy() for k in z.files}
            ix=torch.tensor(arr['condition_joint_indices'],dtype=torch.long)
            condition=VisibleSE3(ix,torch.tensor(arr['condition_position_cm'],dtype=torch.float64),
                torch.tensor(arr['condition_global_rotation'],dtype=torch.float64),
                torch.tensor(arr['condition_position_valid']),torch.tensor(arr['condition_rotation_valid']),
                tuple(body.names),'DIAGNOSTIC ONLY: exact saved formal visible targets, independently recorded q oracle','cm')
            gid=case['provenance']['group_index'];rows=[lookup[(gid,int(f))] for f in arr['frame_ids']]
            q=torch.tensor(saved_q[rows],dtype=torch.float64)
            assert bool(((q>=tree.lower)&(q<=tree.upper)).all())
            p,R=tree.decode_q(q)
            rot6=formal.rot6_from_matrix(R)
            slides=q.new_zeros(len(q),len(tree.names))
            for j in getattr(tree,'slide_bodies',()):slides[:,j]=q[:,tree.body_slot[j]]*100.
            packed=torch.cat([rot6,slides[...,None]],-1)
            projected=tree.project_free(packed if hasattr(tree,'slide_bodies') else rot6)
            typed=mixed_forward_kinematics(rot6.float(),body.geom(len(q)),slides.float())
            typed_loss,typed_metrics=_targets(typed['joint_position'].double(),typed['joint_rotation'].double(),condition,cfg)
            typed_position_max=float((typed['joint_position'].double()-p).abs().max())
            typed_rotation_max=float((typed['joint_rotation'].double()-R).abs().max())
            oracle=components(q,tree,condition,cfg)
            original_position_max=float((p-torch.tensor(arr['reference_position_cm'])).abs().max())
            units=getattr(tree,'units',('radian',)*len(tree.q_names))
            scale=q.new_tensor([1e-3 if u=='radian' else 1e-5 for u in units])
            gen=torch.Generator().manual_seed(20261004)
            perturb=torch.randn(q.shape,generator=gen,dtype=q.dtype)*scale
            qp=adapter.clamp(q+perturb).detach().requires_grad_(True)
            loss,_=objective(qp,tree,condition,cfg);grad=torch.autograd.grad(loss,qp)[0]
            direction=-grad*scale.square()
            direction=direction/(direction.abs()/scale).max().clamp_min(1e-30)*.1
            directional=float((grad*direction).sum())
            eps=1e-3
            fd=float((objective(qp.detach()+eps*direction,tree,condition,cfg)[0]-objective(qp.detach()-eps*direction,tree,condition,cfg)[0])/(2*eps))
            candidates=[];accepted=None
            for alpha in [1.,.5,.25,.125,.0625]:
                test=adapter.clamp(qp.detach()+alpha*direction)
                value=float(objective(test,tree,condition,cfg)[0])
                candidates.append(dict(alpha=alpha,loss=value))
                if value<float(loss.detach()):accepted=alpha;break
            near=refine_trajectory(qp.detach(),tree.decode_q,condition,config=cfg,
                state_projector=adapter.clamp,projection_source='DIAGNOSTIC: validated actual bounds; perturbed recorded q, not learned initialization')
            bound=lambda v:int(((v<=tree.lower.to(v)+1e-10)|(v>=tree.upper.to(v)-1e-10)).sum())
            rec=dict(case_id=case['case_id'],body=name,frames=len(q),source_group=gid,oracle=oracle,
                max_fullbody_position_difference_cm=original_position_max,
                own_global_rotation_to_q_roundtrip_max=float((projected-q).abs().max()),
                shared_typed_fk_position_max_difference_cm=typed_position_max,
                shared_typed_fk_rotation_matrix_max_difference=typed_rotation_max,
                shared_typed_fk_visible_loss=float(typed_loss),shared_typed_fk_visible_metrics=typed_metrics,
                perturbation_sigma_by_q_unit=dict(radian=.001,metre=.00001),
                perturbed=components(qp,tree,condition,cfg),gradient_finite=bool(torch.isfinite(grad).all()),
                gradient_directional_derivative=directional,finite_difference_directional_derivative=fd,
                relative_directional_derivative_error=abs(fd-directional)/max(abs(fd),abs(directional),1e-15),
                small_projected_descent_candidates=candidates,small_projected_descent_accepted=accepted,
                exact_formal_adam_from_near_oracle=dict(initial=near['records'][0]['metrics'],
                    final=near['records'][-1].get('metrics'),initial_loss=near['records'][0]['loss'],
                    final_loss=near['records'][-1].get('loss'),status=near['status'],
                    coordinate_bound_counts=dict(reference=bound(q),perturbed=bound(qp),final=bound(near['final_state'])),
                    training_updates=0,diagnostic_state_updates=near['executed_optimizer_steps']))
            rec['implementation_checks_passed']=bool(original_position_max<1e-3 and typed_position_max<1e-3 and typed_rotation_max<1e-5
                and float((projected-q).abs().max())<1e-5 and oracle['metrics']['selected_position_max']<1e-3
                and oracle['metrics']['selected_rotation_max_deg']<1e-3 and rec['gradient_finite']
                and rec['relative_directional_derivative_error']<1e-4 and accepted is not None)
            records.append(rec)
            for run in [f'{v}_{s}_seed0' for s in 'SML' for v in 'AB']:
                pp=ROOT/'evaluation/results/manipulation'/run/'checkpoint_step3000'/name/path.parent.name/'case_arrays.npz'
                with np.load(pp,allow_pickle=False) as z:
                    state=torch.cat([torch.tensor(z['initial_model_global_rot6'],dtype=torch.float64),torch.tensor(z['initial_model_slide_cm'],dtype=torch.float64)[...,None]],-1)
                free_q=tree.project_free(state if hasattr(tree,'slide_bodies') else state[...,:6]);valid_q=adapter.clamp(free_q)
                projections.append(dict(run=run,case_id=case['case_id'],coordinates=free_q.numel(),
                    before_clamp_violations=int(((free_q<tree.lower)|(free_q>tree.upper)).sum()),
                    after_clamp_at_bounds=bound(valid_q),max_absolute_q_clamp_delta=float((valid_q-free_q).abs().max()),
                    unit_by_coordinate=list(units),clamped_coordinate_counts=((free_q<tree.lower)|(free_q>tree.upper)).sum(0).tolist()))
    output=dict(schema='native_q_oracle_implementation_diagnostic_v1',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        status='passed' if records and all(r['implementation_checks_passed'] for r in records) else 'failed',
        scope='DIAGNOSTIC ONLY: oracle q and near-q perturbations, never formal learned-model results',
        source_files_sha256=source_files,source_code_sha256={str(p):sha(p) for p in [Path(__file__),ROOT/'evaluation/trajectory_refinement.py',ROOT/'evaluation/hinge_dof_adapter.py',ROOT/'evaluation/mixed_dof_adapter.py',ROOT/'prismatic/kinematics.py']},
        model_loaded=False,training_updates=0,formal_output_changes=False,cuda_initialized=torch.cuda.is_initialized(),
        fullbody_GT_in_formal_optimizer=False,records=records,learned_initial_projection_clamp=projections,
        loss_scale=dict(position_units='cm',position_scale=1.,position_weight=1.,rotation_weight=1.,
            at_1cm_position_tolerance=1.,at_5deg_rotation_tolerance=2*(1-math.cos(math.radians(5))),
            ratio_at_tolerances=1/(2*(1-math.cos(math.radians(5)))),
            interpretation='Frozen objective weights are not normalized to equal position/rotation acceptance tolerances; diagnostic only, no protocol change.'),
        rotation_convention='Own-joint global rotations rebased to rest-canonical coordinates; relative hinge M=R_parent^T R_joint about rest-canonical axis; actual q0 and closest 2pi branch; hard clamp. Not a coupled global IK projection.')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':output['status'],'output':str(args.output),'cases':len(records),'checks':[{k:r[k] for k in ['body','case_id','implementation_checks_passed','max_fullbody_position_difference_cm','shared_typed_fk_position_max_difference_cm','relative_directional_derivative_error']} for r in records],'loss_scale':output['loss_scale'],'cuda_initialized':output['cuda_initialized']}))


if __name__=='__main__':main()
