"""Fixed P0 Franka cases: actual model logits, saturation and opening errors."""
from pathlib import Path
import gc,json,sys
import numpy as np
import torch
from common import ROOT,PHASE,TRAIN,ORDER,read,sha,write,capacity,checkpoint_receipt
sys.path.insert(0,str(ROOT/'evaluation'))
import run_scaling_eval as formal
from scaling_data import load_scaling_bodies
from prismatic.kinematics import mixed_forward_kinematics

def stats(values):
    x=np.asarray(values,dtype=np.float64).reshape(-1)
    return dict(count=len(x),min=float(x.min()),max=float(x.max()),mean=float(x.mean()),p05=float(np.quantile(x,.05)),p50=float(np.median(x)),p95=float(np.quantile(x,.95)))
def main():
    torch.set_num_threads(1);device=torch.device('cuda');out=PHASE/'slide_diagnostic';out.mkdir(exist_ok=True)
    body=None;records=[]
    for name in ORDER:
        for step in [3000,4500,6000]:
            capacity();path=ROOT/'runs'/name/f'checkpoint_step{step:04d}.pt' if step==3000 else TRAIN/'runs'/name/f'checkpoint_step{step:04d}.pt'
            if step>3000:checkpoint_receipt(name,step)
            else:
                declared=next(r for r in read(ROOT/'runs'/name/'result.json')['intermediate_checkpoints'] if r['step']==3000)
                if sha(path)!=declared['sha256']:raise RuntimeError('Initial diagnostic checkpoint changed')
            payload=formal.load_evaluation_checkpoint(path);model=formal.model_factory(payload['config'],name[0]);model.load_state_dict(payload['model'],strict=True);model.to(device).eval()
            if body is None:
                bodies,audit=load_scaling_bodies(ROOT,payload['config'],device);body=next(b for b in bodies if b.name=='relay_franka_full_v2')
                tree=formal.MixedJointTree(ROOT/'data/payloads/relay_franka_full_v2.json')
                names=list(body.names) if hasattr(body,'names') else list(np.load(ROOT/'data/payloads/relay_franka_full_v2.npz',allow_pickle=False)['joint_names'].astype(str))
                fingers=[names.index('panda0_leftfinger'),names.index('panda0_rightfinger')]
            for src in sorted((ROOT/'evaluation/results/manipulation'/name/'checkpoint_step3000/relay_franka_full_v2').rglob('case_result.json')):
                old=read(src);arr=dict(np.load(src.parent/'case_arrays.npz',allow_pickle=False))
                selection=dict(group_index=old['provenance']['group_index'],rows=arr['loader_rows'],loaded_episode_frames=old['provenance']['loaded_episode_frames'],sampling_stride=old['provenance']['native_frame_stride'],reference_cohort=old['provenance']['reference_cohort'],native_reference_limit_qc=old['provenance']['native_reference_limit_qc'],selection_reason=old['provenance']['selection'])
                case,condition,gt6,rows=formal.make_case_and_condition(body,selection,device,2)
                assert case.case_id==old['case_id'] and formal.visible_se3_digest(condition)==old['condition_sha256']
                captured=[];sigmoids=[]
                def capture(m,a,v):
                    captured.append(v.detach().float().cpu().numpy());sigmoids.append(torch.sigmoid(v.detach().float()).cpu().numpy())
                hook=model.slide_head.register_forward_hook(capture)
                initial,prediction,timing,raw_aux=formal.sparse_model_initialization(model,body,condition,32);hook.remove()
                logits=np.concatenate(captured,0)[...,0][:,fingers]
                with torch.no_grad():
                    value=mixed_forward_kinematics(initial[...,:6],body.geom(len(initial)),initial[...,6]);p=value['joint_position'];R=value['joint_rotation']
                    predicted=tree.aperture_cm(p,R).cpu().numpy();reference=tree.aperture_cm(torch.tensor(case.reference_positions,device=device,dtype=p.dtype),torch.tensor(case.reference_rotations,device=device,dtype=R.dtype)).cpu().numpy()
                sigmoid=np.concatenate(sigmoids,0)[...,0][:,fingers];error=np.abs(predicted-reference)*10
                artifact=out/f'{name}_{step}_{case.case_id}.npz'
                np.savez_compressed(artifact,logits=logits,float32_sigmoid=sigmoid,predicted_opening_cm=predicted,reference_opening_cm=reference,absolute_opening_error_mm=error,frame_ids=case.frame_ids,timestamps_seconds=case.timestamps_seconds)
                records.append(dict(run=name,step=step,case_id=case.case_id,cohort=old['provenance']['reference_cohort'],checkpoint_sha256=sha(path),condition_sha256=old['condition_sha256'],reference_case_sha256=sha(src),
                    logit_distribution=stats(logits),exact_saturation_count=int(((sigmoid==0)|(sigmoid==1)).sum()),logit_count=int(logits.size),exact_saturation_fraction=float(((sigmoid==0)|(sigmoid==1)).mean()),opening_error_mm=stats(error),artifact=str(artifact),artifact_sha256=sha(artifact)))
            del model,payload;gc.collect();torch.cuda.empty_cache()
    write(out/'summary.json',dict(records=records,training_updates=0,case_source='Unchanged initial P0 Franka cases; same visible SE3, frames and cohorts',
        opening_definition=tree.document['aperture_definition'],scope='Native network prediction through typed FK before real-q projection/refinement; P0/P1 reports separately establish actual-q final SE3 and opening errors',
        contact_pad_clearance=False,saturation_recovery_claim=False,source_sha256=sha(__file__)))
    print(json.dumps(dict(records=len(records),output=str(out/'summary.json'))))
if __name__=='__main__':main()
