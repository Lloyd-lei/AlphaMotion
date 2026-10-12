"""Read-only restored-checkpoint gradient diagnosis; zero optimizer updates."""
from pathlib import Path
import contextlib,fcntl,json,sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'continuation_v2'))
import train_common_phase as r
o=r.original

def tensor_info(t):
    x=t.detach().float()
    return dict(shape=list(x.shape),min=float(x.min()),max=float(x.max()),mean=float(x.mean()),finite=bool(torch.isfinite(x).all()))

def main():
    lease=(ROOT/'gpu_queue.lock').open('a+');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(4);device=torch.device('cuda');cfg=r.read(ROOT/'runs/A_S_seed0/config.json')
    bodies,audit=o.load_scaling_bodies(ROOT,cfg,device);b=next(b for b in bodies if b.name=='relay_franka_full_v2')
    records=[]
    for name in r.ORDER:
        cfg=r.read(ROOT/'runs'/name/'config.json');model,opt=r.load_run(name,cfg,device)
        for amp in [True,False]:
            rng=np.random.default_rng(811);rows=o.sample_rows(b,rng,cfg['batch_size'])
            inp,target=o.make_inputs(b,b,rows,rng,device,'full',0);captured={};hooks=[]
            def hook(key):
                def receive(module,args,out):
                    if torch.is_tensor(out):
                        captured[key]=out;out.retain_grad()
                return receive
            for key in ['slide_head','source_slide_projection','slide_geometry_projection']:
                hooks.append(getattr(model,key).register_forward_hook(hook(key)))
            model.zero_grad(set_to_none=True)
            with o.autocast(device) if amp else contextlib.nullcontext():out=o.forward(model,b,b,inp)
            loss,terms=o.loss_function(cfg)(out,target,b,rows,inp['condition'],cfg['loss']);loss.backward()
            valid=out['slide_valid'];logits=captured['slide_head'].squeeze(-1)[valid].detach().float()
            grads={prefix:sum(float(p.grad.abs().sum()) for n,p in model.named_parameters() if n.startswith(prefix+'.') and p.grad is not None) for prefix in ['source_slide_projection','slide_geometry_projection','slide_head']}
            rec=dict(run=name,autocast_bfloat16=amp,checkpoint_sha256=r.sha(ROOT/'runs'/name/'checkpoint.pt'),
                loss=float(loss.detach()),finite_global_gradient=o.finite_backward(model),gradient_sums=grads,
                slide_logits=tensor_info(logits),slide_prediction=tensor_info(out['slide_normalized'][valid]),
                sigmoid_exact_zero_derivatives=int((torch.sigmoid(logits)*(1-torch.sigmoid(logits))==0).sum()),
                valid_values=int(valid.sum()),source_slide=tensor_info(inp['source_slide_normalized']),
                modules={k:dict(value=tensor_info(t),gradient_sum=None if t.grad is None else float(t.grad.abs().sum())) for k,t in captured.items()},
                rows_sha256=o.digest(rows.tolist()),optimizer_updates=0)
            records.append(rec);print(json.dumps(rec),flush=True)
            for h in hooks:h.remove()
        del model,opt;torch.cuda.empty_cache()
    r.write(ROOT/'reports/continuation_prismatic_gradient_diagnostic.json',dict(records=records,optimizer_updates=0,source_sha256=r.seal(),diagnostic_sha256=r.sha(__file__)))

if __name__=='__main__':main()
