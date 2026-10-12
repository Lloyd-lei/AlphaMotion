"""Single-frame depth-expanded Megatron, with explicitly sparse target SE(3).
Shared decoder has no per-body parameters; the separate16-class auxiliary judge
keeps its original structure with a fresh, explicitly mapped class head.
No flow, diffusion, surface network or temporal backbone.
"""
import math
import torch
from torch import nn
from model_dual import DualTranslator

class ConditionalMegatron(DualTranslator):
    def __init__(self, judge_mode='grad', judge_body_order=None, **kwargs):
        if judge_mode not in {'off','observe','grad'}:raise ValueError('judge mode')
        supported={'use_sem':True,'use_dof':True,'n_judges':16,
                   'single_codebook':False,'rot_only':False,'rot_head':'free',
                   'vis_residual':False,'root_head':False,'contact_chan':True}
        for key,value in supported.items():
            if key in kwargs and kwargs[key]!=value:
                raise ValueError(f'pilot requires {key}={value!r}')
            kwargs[key]=value
        super().__init__(**kwargs)
        self.judge_mode=judge_mode
        self.judge_body_order=None
        if judge_body_order is not None:self.set_judge_body_order(judge_body_order)
        d=kwargs['d_model']
        self.condition_projection=nn.Sequential(nn.Linear(9,d),nn.GELU(),nn.Linear(d,d))
        self.condition_type=nn.Parameter(torch.randn(d)*.01)
        # Two declared missingness bits: mechanics unknown, metric length unavailable.
        self.metadata_missing_projection=nn.Linear(2,d,bias=False)
        nn.init.normal_(self.metadata_missing_projection.weight,std=.01)

    def set_judge_body_order(self,names):
        # Class meanings come from the current validated chosen_split.body_names.
        # The historical16-column checkpoint does not prove the old class meanings.
        names=tuple(names)
        if len(names)!=16 or len(set(names))!=16 or not all(isinstance(n,str) and n for n in names):
            raise ValueError('judge requires the validated current16-name class order')
        if self.judge_body_order is not None and self.judge_body_order!=names:
            raise ValueError('cannot silently reorder initialized judge classes')
        self.judge_body_order=names

    def judge_logits(self,zp,zr,mode=None):
        mode=self.judge_mode if mode is None else mode
        if mode=='off':return None
        if mode not in {'observe','grad'}:raise ValueError('judge mode')
        if mode=='observe':zp,zr=zp.detach(),zr.detach()
        return self.judge(zp,zr)

    def decode(self,zp,zr,g,obs=None,vis=None,ret_con=False):
        # Original unconditional cycle API; raw hpos, never FK-substituted coordinates.
        if obs is not None or vis is not None:raise ValueError('pilot has no visible-residual decoder')
        rot,pos,con=self.decode_conditional(zp,zr,g,condition=None)
        return (rot,pos,con) if ret_con else (rot,pos)

    def descriptor(self,g):
        x=self.jf(g['rest_off'],g['bonelen'],g['depth'],g.get('sem'),g.get('dof'))
        if 'metadata_missing' in g:
            x=x+self.metadata_missing_projection(g['metadata_missing'].to(x.dtype))
        return x

    def _trunk(self,x,g,joint_mask=None,con=None):
        x,con=self._split_con(x,con)
        if joint_mask is not None:
            if joint_mask.dtype!=torch.bool or joint_mask.shape!=x.shape[:2]:raise ValueError('source hidden mask contract')
            x=torch.where(joint_mask[...,None],torch.zeros_like(x),x)
        if not bool(torch.isfinite(x).all()):raise ValueError('visible source pose must be finite')
        h=self.inp(x)
        if joint_mask is not None:h=torch.where(joint_mask[...,None],self.mask_tok.expand_as(h),h)
        if con is not None:
            valid=torch.isfinite(con)
            safe_con=torch.where(valid,con,torch.zeros_like(con))
            h=h+self.con_emb(safe_con)*valid.to(h.dtype)
        h=h+self.descriptor(g)
        for block in self.enc:h=block(h,g['hop'],None)
        return h

    def decode_conditional(self,zp,zr,g,condition=None):
        qp=self.descriptor(g);qr=qp.clone();kp=kr=None
        if condition is not None:
            # The only dynamic target input is C<=2 declared sparse SE(3) values.
            ix=condition['joint_indices'];v=condition['values'];valid=condition['valid']
            if ix.ndim!=2 or v.shape!=(*ix.shape,9) or valid.shape!=ix.shape:
                raise ValueError('sparse condition shape contract violated')
            if ix.shape[0]!=qp.shape[0] or ix.shape[1]>2 or valid.dtype!=torch.bool:
                raise ValueError('only zero, one or two declared SE3 conditions are accepted')
            if ix.dtype!=torch.long:raise TypeError('condition joint indices must be int64')
            if bool(((ix[valid]<0)|(ix[valid]>=qp.shape[1])).any()):
                raise ValueError('valid condition index outside target joint tree')
            if not bool(torch.isfinite(v[valid]).all()):raise ValueError('valid condition values must be finite')
            ix=torch.where(valid,ix,torch.zeros_like(ix))
            v=torch.where(valid[...,None],v,torch.zeros_like(v))
            descriptor=torch.gather(qp,1,ix[...,None].expand(-1,-1,qp.shape[-1]))
            tokens=self.condition_projection(v)+descriptor+self.condition_type
            tokens=torch.where(valid[...,None],tokens,torch.zeros_like(tokens))
            kp=torch.cat([torch.zeros(zp.shape[:2],device=zp.device,dtype=torch.bool),~valid],1)
            kr=torch.cat([torch.zeros(zr.shape[:2],device=zr.device,dtype=torch.bool),~valid],1)
            zp=torch.cat([zp,tokens],1);zr=torch.cat([zr,tokens],1)
        for cb,sb in zip(self.decp,self.selfp):qp=sb(cb(qp,zp,kp),g['hop'],None)
        for cb,sb in zip(self.decr,self.selfr):qr=sb(cb(qr,zr,kr),g['hop'],None)
        rot=self.hrot(qr);pos=self.hpos(qp)
        self.last_pos=pos
        con=self.hcon(qr).squeeze(-1) if self.contact_chan else None
        return rot,pos,con

    def forward(self,source,source_geom,target_geom,source_hidden=None,source_contact=None,condition=None):
        zp,zr=self.encode(source,source_geom,joint_mask=source_hidden,con=source_contact)
        return self.decode_conditional(zp,zr,target_geom,condition)

def initialize_added_identity_blocks(model,first_new=4):
    changed=[]
    for family in ['enc','selfp','selfr','decp','decr']:
        for i,block in enumerate(getattr(model,family)):
            if i<first_new:continue
            projection=block.a.o if hasattr(block,'a') else block.mha.out_proj
            for component,module in [('attention_output',projection),('feedforward_output',block.ff[2])]:
                nn.init.zeros_(module.weight);nn.init.zeros_(module.bias)
                changed.append(f'{family}.{i}.{component}')
    return changed

def load_reference_weights(model,reference,judge_body_order=None,judge_init_seed=0):
    # Verify the complete known reference schema before considering warm-start exclusions.
    template=DualTranslator(d_model=256,n_heads=8,enc_layers=4,dec_layers=4,
        n_latent=128,n_latent_p=4,fsq_stages=1,fsq_levels=(32,)*13,
        use_sem=True,use_dof=True,n_judges=16,contact_chan=True,sem_dim=768)
    expected=template.state_dict()
    if set(reference)!=set(expected):
        raise ValueError({'missing_reference_keys':sorted(set(expected)-set(reference)),
                          'unexpected_reference_keys':sorted(set(reference)-set(expected))})
    for key,value in reference.items():
        if value.shape!=expected[key].shape:raise ValueError('reference schema shape mismatch: '+key)
    del template,expected
    if judge_body_order is not None:model.set_judge_body_order(judge_body_order)
    state=model.state_dict();accepted={};skipped={}
    for key,value in reference.items():
        if key.startswith('judge.'):
            skipped[key]='historical class order unverified; new16-class judge uses current validated body mapping'
        elif key.startswith('jf.sem_proj.'):
            skipped[key]='entire SigLIP projection reinitialized for pinned Qwen1024'
        else:
            if key not in state:raise KeyError('unexplained reference key '+key)
            if state[key].shape!=value.shape:raise ValueError('unexplained shape mismatch '+key)
            accepted[key]=value
    state.update(accepted);model.load_state_dict(state,strict=True)
    initialize_new_judge(model,judge_init_seed)
    changed=initialize_added_identity_blocks(model)
    return {'loaded_reference_keys':sorted(accepted),'skipped_reference_keys':skipped,
            'backbone_loaded_keys':sorted(accepted),
            'semantic_projection_new_keys':sorted(k for k in state if k.startswith('jf.sem_proj.')),
            'judge_dropped_reference_keys':sorted(k for k in reference if k.startswith('judge.')),
            'judge_new_keys':sorted(k for k in state if k.startswith('judge.')),
            'judge_initialization_seed':int(judge_init_seed),
            'new_keys':sorted(set(state)-set(reference)),
            'identity_initialized_added_blocks':changed,
            'judge_body_order':list(model.judge_body_order) if model.judge_body_order is not None else None,
            'judge_label_order_status':'new judge mapping from current validated chosen_split; no historical-label reuse' if model.judge_body_order is not None else 'new judge; current validated chosen_split mapping must be supplied before auxiliary training',
            'parameter_counts':model_parameter_counts(model),
            'claim':'Compatible shared backbone/FSQ tensors loaded; added residual blocks start as identity. Entire Qwen projection and16-class judge are newly initialized, as are sparse-condition modules. No numerical identity or old judge-label equivalence is claimed.'}


def initialize_new_judge(model,seed=0):
    """Reproducible fresh head, independent of global RNG; never reuse unknown labels."""
    if model.judge is None:raise ValueError('16-class auxiliary head required')
    generator=torch.Generator(device=next(model.judge.parameters()).device).manual_seed(int(seed))
    for module in model.judge.modules():
        if isinstance(module,nn.Linear):
            nn.init.kaiming_uniform_(module.weight,a=math.sqrt(5),generator=generator)
            bound=1/math.sqrt(module.weight.shape[1])
            nn.init.uniform_(module.bias,-bound,bound,generator=generator)



def model_parameter_counts(model):
    total=sum(p.numel() for p in model.parameters())
    judge=sum(p.numel() for name,p in model.named_parameters() if name.startswith('judge.'))
    return {'total':total,'main_excluding_judge':total-judge,'judge_auxiliary':judge,
            'trainable_total':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'trainable_judge':sum(p.numel() for n,p in model.named_parameters() if n.startswith('judge.') and p.requires_grad)}
