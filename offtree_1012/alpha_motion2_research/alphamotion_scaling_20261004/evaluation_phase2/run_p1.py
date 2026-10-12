"""Execute the sealed P1 supplement from immutable P0 learned initializations.

Own GPU lease; preserve every P0 file. This script never creates a model,
trains, downloads resources, or receives oracle q as an optimizer input.
"""
from pathlib import Path
import argparse,dataclasses,datetime,fcntl,hashlib,importlib.util,json,os,sys,time,traceback
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'evaluation_p1'))
from protocol import ROOT,configuration,normalize_adapter,q_from_normalized
from common import capacity as phase_capacity, checkpoint_receipt
import run_scaling_eval as formal
from scaling_data import load_scaling_bodies
from dof_protocol import compare_learned_vs_default
from prismatic.kinematics import mixed_forward_kinematics

PHASE=ROOT/'evaluation_phase2/p1'
ORDER=[f'{v}_{s}_seed0' for s in 'SML' for v in 'AB']


def sha(p):return formal.file_sha(p)
def write(p,d):formal.json_write(p,d)
def read(p):return json.loads(Path(p).read_text())
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def identity():
    p=Path('/proc')/str(os.getpid());s=(p/'stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=os.getpid(),start_ticks=s[19],boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        argv=[x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x])


def verify(plan):
    if plan['status']!='frozen_after_synthetic_probe' or plan['run_order']!=ORDER:raise RuntimeError('P1 sealed plan required')
    for rel,h in plan['source_sha256'].items():
        if sha(ROOT/rel)!=h:raise RuntimeError('P1 source changed: '+rel)
    for rel,h in plan['input_sha256'].items():
        if sha(ROOT/rel)!=h:raise RuntimeError('P1 input changed: '+rel)
    state=read(ROOT/'logs/evaluation_queue/state.json')
    if state['status']!='completed' or len(state['jobs'])!=18 or any(j['status']!='completed' for j in state['jobs']):
        raise RuntimeError('All eighteen original formal jobs must be complete')
    probe=read(ROOT/'evaluation_p1/synthetic_probe_receipt.json')
    if probe['status']!='passed' or probe['registration_sha256']!=sha(PHASE/'preregistration.json'):raise RuntimeError('P1 synthetic gate stale')
    for path,h in probe['source_sha256'].items():
        if sha(path)!=h:raise RuntimeError('P1 probe source stale')


def capacity(plan):
    return phase_capacity(plan['output_total_budget_bytes'])


def same_arrays(a,b):
    with np.load(a,allow_pickle=False) as x,np.load(b,allow_pickle=False) as y:
        return set(x.files)==set(y.files) and all(x[k].dtype==y[k].dtype and np.array_equal(x[k],y[k]) for k in x.files)


def run_one(name,bodies,audit,adapters,trees,plan):
    olddir=ROOT/'evaluation_phase2/p0/manipulation'/name/'checkpoint_step6000'
    out=PHASE/'results/manipulation'/name/'checkpoint_step6000';out.mkdir(parents=True,exist_ok=False)
    oldreport=read(olddir/'evaluation_summary.json');cfg=configuration();cases=[];results=[]
    declared=checkpoint_receipt(name,6000)
    if oldreport['checkpoint_sha256']!=declared['sha256']:raise RuntimeError('P0 report has wrong phase2 checkpoint')
    by={b.name:b for b in bodies};start=time.perf_counter()
    for source in sorted(olddir.rglob('case_result.json')):
        capacity(plan);old=read(source)
        if old.get('status')=='failed_case_retained':raise RuntimeError('P0 initialization unavailable; retain failure, do not substitute')
        body=by[old['body']];tree=trees[body.name];adapter=adapters[body.name]
        src=source.parent/'case_arrays.npz'
        with np.load(src,allow_pickle=False) as z:arr={k:z[k].copy() for k in z.files}
        selection=dict(group_index=old['provenance']['group_index'],rows=arr['loader_rows'],
            loaded_episode_frames=old['provenance']['loaded_episode_frames'],sampling_stride=old['provenance']['native_frame_stride'],
            reference_cohort=old['provenance']['reference_cohort'],native_reference_limit_qc=old['provenance']['native_reference_limit_qc'],
            selection_reason=old['provenance']['selection'])
        case,condition,gt6,rows=formal.make_case_and_condition(body,selection,torch.device('cuda'),2)
        if case.case_id!=old['case_id'] or formal.visible_se3_digest(condition)!=old['condition_sha256']:
            raise RuntimeError('P1 visible condition/case differs from P0')
        cases.append(case);directory=out/body.name/case.case_id
        initial=torch.cat([torch.tensor(arr['initial_model_global_rot6'],device='cuda'),
            torch.tensor(arr['initial_model_slide_cm'],device='cuda')[...,None]],-1)
        def raw_decode(state):
            value=mixed_forward_kinematics(state[...,:6],body.geom(len(state)),state[...,6])
            return value['joint_position'],value['joint_rotation']
        observer=formal.make_observer(case,body,condition,cfg,mixed_tree=tree if body.name=='relay_franka_full_v2' else None)
        try:
            protocol=compare_learned_vs_default(initial,raw_decode,condition,adapter=adapter,config=cfg,observer=observer)
            timing=dict(old['initial_inference'],reused_from_P0=True,measured_in_this_execution=False)
            initial_eval=dict(old['no_refine_initial_model'],reused_from_P0=True)
            saved=formal.save_case(directory,case,condition,initial,gt6,rows,protocol,timing,initial_eval,
                dict(old['fixed_patch_surface'],reused_from_P0=True))
            if not same_arrays(src,directory/'case_arrays.npz'):raise RuntimeError('P1 common reference/initialization arrays changed')
            link=directory/'case_arrays.shared.tmp.npz';os.link(src,link);link.replace(directory/'case_arrays.npz')
            saved['protocol_id']='P1_threshold_and_joint_range_normalized'
            saved['registration_sha256']=sha(PHASE/'preregistration.json')
            saved['P0_case_source']=dict(path=str(source),sha256=sha(source),shared_arrays_sha256=sha(src),
                hardlink_verified=os.stat(src).st_ino==os.stat(directory/'case_arrays.npz').st_ino)
            saved['timing_provenance']='Reuse measured P0 inference; new projection and refinement measured here. Composite path estimate, not one new end-to-end inference measurement.'
            saved['default_control_comparison']['timing_provenance']=saved['timing_provenance']
            for stage,result in [('constrained_refinement',protocol['learned']['constrained_refinement']),
                                  ('default_valid_q',protocol['default_valid_q_baseline'])]:
                z=result['final_state'];actual=trees[body.name].verified_adapter(ROOT/'evaluation'/plan['adapter_receipts'][body.name])
                q=q_from_normalized(z,actual)
                artifact=directory/stage/'terminal_actual_q.npz'
                np.savez_compressed(artifact,q_native=q.cpu().numpy(),optimizer_normalized_z=z.cpu().numpy())
                saved['stages'][stage]['terminal_actual_q']=dict(file=artifact.name,sha256=sha(artifact),
                    coordinate_names=list(actual.joint_names),coordinate_units=list(actual.coordinate_units),
                    normalized_state_definition='z=(q-center)/half_range; refinement_arrays.final_state stores z',
                    max_native_bound_roundoff=float(actual.violations(q).max()),
                    normalized_bound_violations=int((adapter.violations(z)>0).sum()))
            write(directory/'case_result.json',saved);results.append(saved)
        except Exception as exc:
            failure=formal.failure_result(case,repr(exc));record=dict(case_id=case.case_id,clip_id=case.clip_id,body=body.name,
                status='failed_case_retained',error=repr(exc),traceback=traceback.format_exc(),no_refine_initial_model=failure,
                stages={n:dict(status='failed_case_retained',final_observation=failure) for n in ['raw_free_posture','dof_projected','constrained_refinement','default_valid_q']})
            write(directory/'case_result.json',record);results.append(record)
    report=formal.summarize_checkpoint(cases,results,bootstrap_samples=1000)
    if len(cases)!=oldreport['case_count'] or report['evaluation_set_sha256']!=oldreport['evaluation_set_sha256']:
        raise RuntimeError('P1 formal evaluation cases differ')
    report.update(protocol_id='P1_threshold_and_joint_range_normalized',supplemental_not_replacement_of_P0=True,
        checkpoint=oldreport['checkpoint'],checkpoint_sha256=oldreport['checkpoint_sha256'],checkpoint_step=6000,
        variant=name[0],wall_seconds=time.perf_counter()-start,refinement_config=dataclasses.asdict(cfg),
        device='cuda',data_audit=audit,case_count=len(cases),training_updates=0,fullbody_GT_used_by_optimizer=False,
        failed_case_ids=[r['case_id'] for r in results if r.get('status')=='failed_case_retained'],
        P0_summary_source=dict(path=str(olddir/'evaluation_summary.json'),sha256=sha(olddir/'evaluation_summary.json')),
        no_new_model_inference=True,initialization='Exact saved P0 learned rot6 and slide output; no GT q',
        plan_sha256=sha(PHASE/'execution_plan.json'),source_sha256=plan['source_sha256'])
    write(out/'evaluation_summary.json',report)
    return dict(run=name,status='completed',report=str(out/'evaluation_summary.json'),sha256=sha(out/'evaluation_summary.json'),
        case_count=len(cases),failed_cases=report['failed_case_ids'],wall_seconds=report['wall_seconds'])


def main():
    torch.set_num_threads(1);logs=PHASE/'logs';logs.mkdir(exist_ok=True)
    if (logs/'state.json').exists():raise RuntimeError('P1 existing state requires explicit review; no duplicate or implicit restart')
    plan=read(PHASE/'execution_plan.json');lease=(ROOT/'gpu_queue.lock').open('a+')
    fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);verify(plan);capacity(plan)
    if not torch.cuda.is_available():raise RuntimeError('P1 declared CUDA unavailable')
    state=dict(status='running',started_at_utc=utc(),identity=identity(),plan_sha256=sha(PHASE/'execution_plan.json'),runs=[],training_updates=0)
    write(logs/'state.json',state)
    try:
        config=read(ROOT/'runs/A_S_seed0/config.json')
        bodies,audit=load_scaling_bodies(ROOT,config,torch.device('cuda'))
        trees={'adroit_pen':formal.SingleHingeTree(ROOT/'data/payloads/adroit_pen.json'),
            'relay_franka_full_v2':formal.MixedJointTree(ROOT/'data/payloads/relay_franka_full_v2.json')}
        adapters={}
        for body,tree in trees.items():
            actual=tree.verified_adapter(ROOT/'evaluation'/plan['adapter_receipts'][body])
            if body=='adroit_pen':
                method=actual.project_free;actual=dataclasses.replace(actual,project_free=lambda value,fn=method:fn(value[...,:6]))
            adapters[body]=normalize_adapter(actual)
            formal.validate_loader_adapter(next(b for b in bodies if b.name==body),tree)
        for name in ORDER:
            verify(plan);state['current_run']=name;state['capacity']=capacity(plan);write(logs/'state.json',state)
            result=run_one(name,bodies,audit,adapters,trees,plan);state['runs'].append(result);write(logs/'state.json',state)
        state.update(status='completed',finished_at_utc=utc());write(logs/'state.json',state)
    except Exception as exc:
        state.update(status='failed',error=repr(exc),traceback=traceback.format_exc(),finished_at_utc=utc());write(logs/'state.json',state);raise
    finally:fcntl.flock(lease,fcntl.LOCK_UN)


if __name__=='__main__':main()
