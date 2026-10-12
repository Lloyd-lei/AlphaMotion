"""Independent high-precision derivative and all-six real-checkpoint checks."""
from pathlib import Path
from decimal import Decimal,localcontext
import fcntl,gc,hashlib,json,sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];PHASE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'orchestration'))
import train_scaling as o
import pilot_model
import stable_slide_gradient as repair

ORDER=[f'{v}_{s}_seed0' for s in 'SML' for v in 'AB']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())

def main():
    lock=(ROOT/'gpu_queue.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(4);device=torch.device('cuda');old=pilot_model.decode_slide
    grid=torch.linspace(-80,80,4097,device=device);assert torch.equal(torch.sigmoid(grid),repair.StableSigmoid.apply(grid))
    x=torch.tensor([-80.,-34.,-20.,-1.,0.,1.,20.,34.,80.],device=device,requires_grad=True)
    repair.StableSigmoid.apply(x).sum().backward();actual=x.grad.detach().cpu().numpy()
    expected=[]
    with localcontext() as ctx:
        ctx.prec=110;h=Decimal('0.00001')
        def f(x):return Decimal(1)/(Decimal(1)+(-x).exp())
        for v in x.detach().cpu().tolist():
            d=Decimal(str(v));expected.append(float((f(d+h)-f(d-h))/(2*h)))
    relative=float(np.max(np.abs(actual-np.array(expected))/np.array(expected)));assert relative<1e-5
    probe=torch.tensor([-10.,-1.,0.,1.,10.],device=device,dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(repair.StableSigmoid.apply,(probe,),eps=1e-4,atol=1e-7,rtol=1e-4)
    cfg=read(ROOT/'runs/A_S_seed0/config.json');bodies,audit=o.load_scaling_bodies(ROOT,cfg,device)
    body=next(b for b in bodies if b.name=='relay_franka_full_v2');records=[]
    for name in ORDER:
        cfg=read(ROOT/'runs'/name/'config.json');payload=torch.load(ROOT/'runs'/name/'checkpoint.pt',map_location='cpu',weights_only=True)
        model=o.build_model(cfg,device);model.load_state_dict(payload['model'],strict=True);rng=np.random.default_rng(811)
        rows=o.sample_rows(body,rng,cfg['batch_size']);inp,target=o.make_inputs(body,body,rows,rng,device,'full',0)
        pilot_model.decode_slide=old;torch.manual_seed(4119)
        with torch.no_grad(),o.autocast(device):before=o.forward(model,body,body,inp)
        repair.install();model.zero_grad(set_to_none=True);torch.manual_seed(4119)
        with o.autocast(device):after=o.forward(model,body,body,inp)
        equal={k:torch.equal(before[k],v.detach()) for k,v in after.items() if torch.is_tensor(v)}
        if not all(equal.values()):raise RuntimeError('Inference changed under numeric repair: '+name)
        loss,_=o.loss_function(cfg)(after,target,body,rows,inp['condition'],cfg['loss']);loss.backward()
        grads={pfx:sum(float(p.grad.abs().sum()) for n,p in model.named_parameters() if n.startswith(pfx+'.') and p.grad is not None) for pfx in ['source_slide_projection','slide_geometry_projection','slide_head']}
        assert all(np.isfinite(v) and v>0 for v in grads.values()),grads
        unchanged=all(torch.equal(v.detach().cpu(),payload['model'][k]) for k,v in model.state_dict().items());assert unchanged
        records.append(dict(run=name,checkpoint_sha256=sha(ROOT/'runs'/name/'checkpoint.pt'),all_forward_tensor_fields_bit_exact=equal,
            gradient_sums=grads,finite_global_gradient=o.finite_backward(model),model_parameters_unchanged=unchanged,optimizer_updates=0))
        del model,payload,before,after;gc.collect();torch.cuda.empty_cache()
    result=dict(status='passed',optimizer_updates=0,kernel_sha256=sha(PHASE/'stable_slide_gradient.py'),
        validator_sha256=sha(__file__),decimal110_central_difference_max_relative_error=relative,
        double_precision_gradcheck=True,float32_forward_grid_bit_exact=4097,real_checkpoint_records=records,
        limitation='Restores mathematically nonzero but possibly very small derivatives. Does not reset logits, reinitialize weights or establish that saturated models will recover.',
        original_failure_retained='continuation_v2/logs/attempt_20261004T121611Z_exit.json')
    o.write_json(PHASE/'numeric_repair_receipt.json',result);print(json.dumps(result))

if __name__=='__main__':main()
