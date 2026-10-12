"""Original single-frame geometric floor recipe with explicit missingness.

Reference: dual_common.py::contact_terms (Y up, centimetre FK, metre sole
vectors, GT root gap), train_dual.py root-position channel0 SmoothL1. Weights
from L_M20k_0903: pen=200, cgeo=1, rootpos=3. Invalid measurements are selected
out before arithmetic. No metric floor is inferred for UniMate.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def _mat(rot6):
    a1,a2=rot6[...,:3],rot6[...,3:]
    b1=F.normalize(a1,dim=-1)
    b2=F.normalize(a2-(b1*a2).sum(-1,keepdim=True)*b1,dim=-1)
    return torch.stack([b1,b2,torch.cross(b1,b2,dim=-1)],-1)


def ground_losses(pred_global_rot6, pred_fk_cm, pred_root_height_m, ground,
                  *, w_pen=200., w_cgeo=1., w_rootpos=3.):
    """Return {'total': scalar tensor, 'terms': unweighted tensors, 'counts':...}.

    ground keys: root_height_m/valid[B], ground_valid[B], contact_trust[B],
    contact_soft/valid/hard[B,J], sole_j[K], sole_off_m[K,3],
    contact_threshold_m scalar, metric_length_valid bool. contact_hard is
    preserved for old masking/cmaint callers, but original cgeo uses soft.
    ``pred_root_height_m`` is [B] = position_head[:,root_index,0], NOT XYZ Y.
    Geometry uses GT root_height_m and is independent of the root head.
    Call w_rootpos=0 if another objective already adds that loss.
    """
    rot=pred_global_rot6.float();pos=pred_fk_cm.float()
    if rot.ndim!=3 or rot.shape[-1]!=6 or pos.shape!=rot.shape[:-1]+(3,):
        raise ValueError('ground prediction shape')
    B,J=rot.shape[:2];device=rot.device
    # A differentiable empty sum is zero without multiplying NaN predictions.
    zero=rot.reshape(-1)[:0].sum();total=zero;terms={}
    counts={'ground_rows':0,'contact_values':0,'root_height_rows':0}
    if not bool(ground['metric_length_valid']):
        if bool(torch.as_tensor(ground['ground_valid']).any()) or bool(torch.as_tensor(ground['root_height_valid']).any()) or bool(torch.as_tensor(ground['contact_valid']).any()):
            raise ValueError('nonmetric body claims metric ground/root validity')
        return {'total':total,'terms':terms,'counts':counts}
    def tensor(key,dtype=None):return torch.as_tensor(ground[key],device=device,dtype=dtype)
    root=tensor('root_height_m',torch.float32);rv=tensor('root_height_valid',torch.bool)
    gv=tensor('ground_valid',torch.bool);trust=tensor('contact_trust',torch.bool)
    soft=tensor('contact_soft',torch.float32);cv=tensor('contact_valid',torch.bool)
    hard=tensor('contact_hard',torch.bool)
    if any(v.shape!=(B,) for v in [root,rv,gv,trust]) or any(v.shape!=(B,J) for v in [soft,cv,hard]):
        raise ValueError('ground batch target shape')
    if bool((rv & ~torch.isfinite(root)).any()):raise ValueError('valid root target nonfinite')
    if bool((gv & ~(rv & trust)).any()):raise ValueError('ground row missing trusted root gap')
    if bool((cv & ~torch.isfinite(soft)).any()):raise ValueError('valid contact soft target nonfinite')
    if bool(((soft[cv]<0)|(soft[cv]>1)).any()):raise ValueError('contact soft outside [0,1]')
    if w_rootpos and pred_root_height_m is not None and bool(rv.any()):
        pred=pred_root_height_m.float()
        if pred.shape!=(B,):raise ValueError('root head must be [B] channel0')
        if not bool(torch.isfinite(pred[rv]).all()):raise ValueError('nonfinite valid root prediction')
        terms['rootpos']=F.smooth_l1_loss(pred[rv],root[rv],beta=1.,reduction='mean')
        total=total+w_rootpos*terms['rootpos'];counts['root_height_rows']=int(rv.sum())
    rows=gv & trust & rv
    if (w_pen or w_cgeo) and bool(rows.any()):
        if not ground.get('ground_geometry_verified',False):raise ValueError('ground geometry lacks verified original sole mapping')
        if not bool(torch.isfinite(rot[rows]).all() and torch.isfinite(pos[rows]).all()):raise ValueError('nonfinite valid geometric prediction')
        # Select trusted rows BEFORE adding root values, preserving invalid NaNs.
        h=pos[rows,:,1]/100. + root[rows,None]
        js=tensor('sole_j',torch.long);off=tensor('sole_off_m',torch.float32)
        if js.ndim!=1 or off.shape!=(len(js),3) or len(torch.unique(js))!=len(js) or bool(((js<0)|(js>=J)).any()) or not bool(torch.isfinite(off).all()):
            raise ValueError('invalid verified sole mapping')
        if len(js):
            dh=torch.einsum('bkxy,ky->bkx',_mat(rot[rows][:,js]),off)[...,1]
            h=h.index_add(1,js,dh)
        counts['ground_rows']=int(rows.sum())
        if w_pen:
            terms['pen']=torch.relu(-h).square().mean()
            total=total+w_pen*terms['pen']
        valid=cv[rows]
        if w_cgeo and bool(valid.any()):
            thr=tensor('contact_threshold_m',torch.float32)
            if thr.ndim!=0 or not bool(torch.isfinite(thr) and thr>0):raise ValueError('invalid contact threshold')
            terms['cgeo']=F.binary_cross_entropy_with_logits((thr-h[valid])/(thr/3.),soft[rows][valid])
            total=total+w_cgeo*terms['cgeo'];counts['contact_values']=int(valid.sum())
    return {'total':total,'terms':terms,'counts':counts}
