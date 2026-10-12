"""Pre-registered P1 synthetic legal-q probe; no recorded motion or model load."""
from pathlib import Path
import dataclasses,datetime,hashlib,json,sys
import numpy as np
import torch
from protocol import ROOT,configuration,normalize_adapter,normalized_from_q,q_from_normalized
from hinge_dof_adapter import SingleHingeTree
from mixed_dof_adapter import MixedJointTree
from trajectory_refinement import VisibleSE3,refine_trajectory,_angles


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1);registration=json.loads((ROOT/'evaluation_p1/preregistration.json').read_text())
    records=[];cfg=configuration()
    for body,kind,receipt,names in [('adroit_pen',SingleHingeTree,'adroit_dof_validation.json',['thdistal','ffdistal']),
        ('relay_franka_full_v2',MixedJointTree,'relay_franka_full_v2_dof_validation.json',['panda0_leftfinger','panda0_rightfinger'])]:
        tree=kind(ROOT/'data/payloads'/f'{body}.json');actual=tree.verified_adapter(ROOT/'evaluation'/receipt)
        adapter=normalize_adapter(actual);Q=len(actual.joint_names);T=64
        gen=torch.Generator().manual_seed(registration['synthetic_probe']['seed'])
        z=torch.rand((T,Q),generator=gen,dtype=torch.float64)*1.8-.9
        for i in range(T-16,T):z[i,i%Q]=(-1. if i%2 else 1.)
        q=q_from_normalized(z,actual);p,R=actual.decode_q(q)
        ix=torch.tensor([tree.names.index(n) for n in names])[None].expand(T,-1)
        b=torch.arange(T)[:,None]
        condition=VisibleSE3(ix,p[b,ix],R[b,ix],torch.ones_like(ix,dtype=torch.bool),
            torch.ones_like(ix,dtype=torch.bool),tree.names,'P1 preregistered synthetic legal-q FK, seed '+str(registration['synthetic_probe']['seed']),'cm')
        zp=adapter.clamp(z+torch.randn(z.shape,generator=gen,dtype=z.dtype)*.01)
        result=refine_trajectory(zp,adapter.decode_q,condition,config=cfg,
            state_projector=adapter.clamp,projection_source='P1 exact actual-q range normalization and bounds')
        final=result['final_state'];qfinal=q_from_normalized(final,actual);pf,rf=actual.decode_q(qfinal)
        errors=(pf[b,ix]-p[b,ix]).norm(dim=-1);angles=_angles(rf[b,ix],R[b,ix])
        violations=int(((qfinal<actual.lower-1e-12)|(qfinal>actual.upper+1e-12)).sum())
        rt=float((normalized_from_q(q,actual)-z).abs().max())
        default_delta=float((q_from_normalized(adapter.default_q,actual)-actual.default_q).abs().max())
        rec=dict(body=body,frames=T,q_coordinates=Q,seed=registration['synthetic_probe']['seed'],
            initial=result['records'][0]['metrics'],final=result['records'][-1]['metrics'],
            initial_loss=result['records'][0]['loss'],final_loss=result['records'][-1]['loss'],
            max_normalization_roundtrip_error=rt,default_q_native_difference=default_delta,
            final_bound_violations=violations,final_bound_contacts=int((final.abs()>=1-1e-10).sum()),
            frames_meeting_both_tolerances=int(((errors<=1.)&(angles<=5.)).all(1).sum()),
            optimizer_steps=result['executed_optimizer_steps'],numerical_failure=result['failure_reason'],
            native_step_per_0p01_z=[float(v) for v in ((actual.upper-actual.lower)/2*.01)],
            actual_q_units=list(actual.coordinate_units),actual_q_names=list(actual.joint_names),
            source_sha256=actual.source_files_sha256)
        rec['passed']=bool(rt<1e-12 and default_delta<1e-12 and violations==0 and result['failure_reason'] is None
            and rec['final_loss']<rec['initial_loss'] and rec['frames_meeting_both_tolerances']==T)
        records.append(rec)
    out=dict(schema='P1_preregistered_synthetic_probe_v1',status='passed' if all(r['passed'] for r in records) else 'failed',
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),records=records,
        registration_sha256=sha(ROOT/'evaluation_p1/preregistration.json'),
        source_sha256={str(p):sha(p) for p in [Path(__file__),ROOT/'evaluation_p1/protocol.py',ROOT/'evaluation/trajectory_refinement.py']},
        refinement_config=dataclasses.asdict(cfg),recorded_motion_loaded=False,model_loaded=False,training_updates=0,
        cuda_initialized=torch.cuda.is_initialized(),formal_model_scores=False)
    dest=ROOT/'evaluation_p1/synthetic_probe_receipt.json';dest.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print(json.dumps(out))
    if out['status']!='passed':raise RuntimeError('P1 synthetic gate did not pass; do not launch formal supplement')


if __name__=='__main__':main()
