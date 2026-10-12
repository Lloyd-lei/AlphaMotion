"""B: shared dual-FSQ translator with real stable mesh-patch/joint token fusion.

All outputs remain global rotations/raw root followed by FK+verified mesh binding.
There is no independently drifting surface coordinate head or embodiment decoder.
"""
from __future__ import annotations
from pathlib import Path
from contextlib import contextmanager
import sys
sys.dont_write_bytecode=True
_BASE=Path(__file__).resolve().parents[1]/'base'
sys.path.insert(0,str(_BASE));sys.path.insert(0,str(_BASE/'legacy'))
import torch
from torch import nn
from pilot_model import ConditionalMegatron,model_parameter_counts
try:
 from .surface_contract import checked_surface
 from .surface_geometry import bound_surface
except ImportError:
 from surface_contract import checked_surface
 from surface_geometry import bound_surface


class AttachmentCrossAttention(nn.Module):
    """Shared joint-query/patch-key attention with binding-joint hop-distance bias."""
    def __init__(self,d,heads):
        super().__init__();self.heads=heads;self.dk=d//heads
        self.q=nn.Linear(d,d);self.k=nn.Linear(d,d);self.v=nn.Linear(d,d);self.out=nn.Linear(d,d)
        self.nq=nn.LayerNorm(d);self.nk=nn.LayerNorm(d);self.hop_bias=nn.Embedding(18,heads)
    def forward(self,joint,patch,hop,valid,ablate_bias=False):
        b,j,d=joint.shape;p=patch.shape[1]
        q=self.q(self.nq(joint)).view(b,j,self.heads,self.dk).transpose(1,2)
        k=self.k(self.nk(patch)).view(b,p,self.heads,self.dk).transpose(1,2)
        v=self.v(self.nk(patch)).view(b,p,self.heads,self.dk).transpose(1,2)
        logits=q@k.transpose(-1,-2)/(self.dk**.5)
        if not ablate_bias:logits=logits+self.hop_bias(hop.clamp(max=17)).permute(0,3,1,2)
        any_valid=valid.any(1);safe_valid=valid.clone();safe_valid[~any_valid,0]=True
        logits=logits.masked_fill(~safe_valid[:,None,None,:],-torch.inf)
        out=self.out((logits.softmax(-1)@v).transpose(1,2).reshape(b,j,d))
        return torch.where(any_valid[:,None,None],out,torch.zeros_like(out))


class SurfaceMegatron(ConditionalMegatron):
    def __init__(self,max_patches=96,**kwargs):
        kwargs.setdefault('n_judges',12)
        super().__init__(**kwargs);d=kwargs['d_model'];heads=kwargs['n_heads']
        self.max_patches=max_patches;self.surface_ablation='none'
        # xyz3, normal3, log-area1, binding entropy1, semantic-valid1.
        self.patch_geometry=nn.Sequential(nn.Linear(9,d),nn.GELU(),nn.Linear(d,d))
        self.patch_semantic=nn.Sequential(nn.Linear(1024,d),nn.GELU(),nn.Linear(d,d))
        # Visible posed position3, normal3, observed bit1. No velocity/pressure invented.
        self.patch_motion=nn.Sequential(nn.Linear(7,d),nn.GELU(),nn.Linear(d,d))
        self.patch_type=nn.Parameter(torch.randn(d)*.01)
        self.target_surface_attention=AttachmentCrossAttention(d,heads)

    @contextmanager
    def ablate_surface(self,mode):
        if mode not in {'none','drop_surface','static_only','no_patch_semantics','no_attachment_bias'}:raise ValueError('surface ablation')
        previous=self.surface_ablation;self.surface_ablation=mode
        try:yield self
        finally:self.surface_ablation=previous

    def _surface(self,g,batch,joints):
        if self.surface_ablation=='drop_surface':return None
        s=checked_surface(g,batch,joints)
        if s is not None and s['valid'].shape[1]>self.max_patches:raise ValueError('patch token budget')
        return s

    def _patch_static(self,joint,g,s):
        weights=s['binding_weights'];scale=s['geometry_scale'][:,None]
        entropy=-(weights*weights.clamp_min(1e-12).log()).sum(-1)
        feat=torch.cat([s['rest_xyz']/scale[...,None],s['rest_normal'],
            (s['area']/scale.square()).clamp_min(1e-12).log()[...,None],entropy[...,None],s['semantic_valid'][...,None].float()],-1)
        token=self.patch_geometry(feat)+torch.einsum('bpj,bjd->bpd',weights,joint)+self.patch_type
        if self.surface_ablation!='no_patch_semantics':token=token+self.patch_semantic(s['qwen_sem'].detach())*s['semantic_valid'][...,None]
        return torch.where(s['valid'][...,None],token,torch.zeros_like(token))

    def descriptor(self,g):
        joint=super().descriptor(g);b,j,_=joint.shape;s=self._surface(g,b,j)
        if s is None:return joint
        patch=self._patch_static(joint,g,s);anchor=s['binding_weights'].argmax(-1)
        hop=torch.gather(g['hop'],2,anchor[:,None,:].expand(-1,j,-1))+1
        return joint+self.target_surface_attention(joint,patch,hop,s['valid'],self.surface_ablation=='no_attachment_bias')

    def _joint_patch_trunk(self,x,g,joint_mask=None,con=None,source_slide_normalized=None):
        x,con=self._split_con(x,con);b,j,_=x.shape
        s=self._surface(g,b,j)
        if s is None:return super()._trunk(x,g,joint_mask,con,source_slide_normalized),None
        if joint_mask is None:joint_mask=torch.zeros((b,j),device=x.device,dtype=torch.bool)
        if joint_mask.dtype!=torch.bool or joint_mask.shape!=(b,j):raise ValueError('source hidden mask')
        x=torch.where(joint_mask[...,None],torch.zeros_like(x),x)
        if not bool(torch.isfinite(x).all()):raise ValueError('visible source pose must be finite')
        descriptor=super().descriptor(g)
        joint=self.inp(x);joint=torch.where(joint_mask[...,None],self.mask_tok.expand_as(joint),joint)
        if con is not None:
            valid=torch.isfinite(con);joint=joint+self.con_emb(torch.where(valid,con,torch.zeros_like(con)))*valid
        joint=joint+self.source_slide_tokens(source_slide_normalized,g,b,j,x.device,joint_mask)
        joint=joint+descriptor
        patch=self._patch_static(descriptor,g,s)
        if self.surface_ablation!='static_only':
            observed=s['valid']&~((s['binding_weights']>0)&joint_mask[:,None,:]).any(-1)
            posed=bound_surface(x[...,:6],x[...,6:]*s['pose_scale'][:,None,None],s)
            dynamics=torch.cat([posed['position']/s['pose_scale'][:,None,None],posed['normal'],observed[...,None].float()],-1)
            dynamics=torch.where(observed[...,None],dynamics,torch.zeros_like(dynamics))
            # Mask after MLP too: its biases cannot impersonate a dynamic observation.
            patch=patch+self.patch_motion(dynamics)*observed[...,None]
        h=torch.cat([joint,patch],1);p=patch.shape[1]
        anchor=s['binding_weights'].argmax(-1);node=torch.cat([torch.arange(j,device=x.device)[None].expand(b,-1),anchor],1)
        batch=torch.arange(b,device=x.device)[:,None,None]
        hop=g['hop'][batch,node[:,:,None],node[:,None,:]]
        extra=torch.cat([torch.zeros(j,device=x.device,dtype=torch.long),torch.ones(p,device=x.device,dtype=torch.long)])
        hop=hop+extra[None,:,None]+extra[None,None,:]
        diag=torch.arange(j+p,device=x.device);hop[:,diag,diag]=0
        if self.surface_ablation=='no_attachment_bias':
            mixed=(extra[:,None]+extra[None,:])>0
            hop=torch.where(mixed[None],torch.zeros_like(hop),hop)
        kpm=torch.cat([torch.zeros((b,j),device=x.device,dtype=torch.bool),~s['valid']],1)
        for block in self.enc:h=block(h,hop,kpm)
        return h,kpm

    def _trunk(self,x,g,joint_mask=None,con=None,source_slide_normalized=None):
        # For compatibility with inherited diagnostic hooks; encode retains padding mask.
        return self._joint_patch_trunk(x,g,joint_mask,con,source_slide_normalized)[0]

    def encode(self,x,g,joint_mask=None,return_codes=False,con=None,source_slide_normalized=None):
        h,kpm=self._joint_patch_trunk(x,g,joint_mask,con,source_slide_normalized);b=len(h)
        zp,cp=self.fsqp(self.poolp(self.qp[None].expand(b,-1,-1),h,kpm))
        zr,cr=self.fsqr(self.poolr(self.qr[None].expand(b,-1,-1),h,kpm))
        return (zp,zr,(cp,cr)) if return_codes else (zp,zr)

    def encode_features(self,x,g,joint_mask=None,con=None,source_slide_normalized=None):
        h,kpm=self._joint_patch_trunk(x,g,joint_mask,con,source_slide_normalized);b=len(h)
        return (self.fsqp.features(self.poolp(self.qp[None].expand(b,-1,-1),h,kpm)),
                self.fsqr.features(self.poolr(self.qr[None].expand(b,-1,-1),h,kpm)))

    def surface_cost(self,source_geom,target_geom,condition_count=0):
        def counts(g):
            j=g['rest_off'].shape[1];s=g.get('surface');p=0 if s is None else s['valid'].shape[1]
            valid=0 if s is None else int(s['valid'].sum().detach())
            return j,p,valid
        js,ps,vs=counts(source_geom);jt,pt,vt=counts(target_geom)
        return {'source_joint_tokens':js,'source_patch_tokens_allocated':ps,'source_patch_tokens_valid_in_batch':vs,
                'encoder_tokens':js+ps,'target_joint_queries':jt,'target_static_patch_tokens':pt,'target_patch_tokens_valid_in_batch':vt,
                'position_fsq_slots':self.qp.shape[0],'rotation_fsq_slots':self.qr.shape[0],
                'condition_tokens':condition_count,'encoder_attention_cells_per_layer_per_head':(js+ps)**2,
                'target_patch_query_cells':jt*pt,'surface_ablation':self.surface_ablation}


def surface_parameter_counts(model):
    counts=model_parameter_counts(model)
    prefixes=('patch_geometry.','patch_semantic.','patch_motion.','patch_type','target_surface_attention.')
    extra=sum(p.numel() for n,p in model.named_parameters() if n.startswith(prefixes))
    return {**counts,'surface_added':extra,'shared_skeleton_main':counts['main_excluding_judge']-extra}
