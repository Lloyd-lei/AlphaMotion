"""Single-frame depth-expanded Megatron, with explicitly sparse target SE(3).
No per-body parameters; no flow, diffusion, surface network or temporal backbone.
"""
import torch
from torch import nn
from model_dual import DualTranslator

class ConditionalMegatron(DualTranslator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        d=kwargs['d_model']
        self.condition_projection=nn.Sequential(nn.Linear(9,d),nn.GELU(),nn.Linear(d,d))
        self.condition_type=nn.Parameter(torch.randn(d)*.01)
        # Two declared missingness bits: mechanics unknown, metric length unavailable.
        self.metadata_missing_projection=nn.Linear(2,d,bias=False)
        nn.init.normal_(self.metadata_missing_projection.weight,std=.01)

    def descriptor(self,g):
        x=self.jf(g['rest_off'],g['bonelen'],g['depth'],g.get('sem'),g.get('dof'))
        if 'metadata_missing' in g:
            x=x+self.metadata_missing_projection(g['metadata_missing'].to(x.dtype))
        return x

    def _trunk(self,x,g,joint_mask=None,con=None):
        x,con=self._split_con(x,con)
        h=self.inp(x)
        if joint_mask is not None:h=torch.where(joint_mask[...,None],self.mask_tok.expand_as(h),h)
        if con is not None:
            valid=torch.isfinite(con)
            h=h+self.con_emb(torch.nan_to_num(con))*valid.to(h.dtype)
        h=h+self.descriptor(g)
        for block in self.enc:h=block(h,g['hop'],None)
        return h

    def decode_conditional(self,zp,zr,g,condition=None):
        qp=self.descriptor(g);qr=qp.clone();kp=kr=None
        if condition is not None:
            # The only dynamic target input is C<=2 declared sparse SE(3) values.
            ix=condition['joint_indices'];v=condition['values'];valid=condition['valid']
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

def load_reference_weights(model,reference):
    state=model.state_dict();accepted={};skipped={}
    for key,value in reference.items():
        if key.startswith('judge.'):
            skipped[key]='auxiliary per-body judge removed; shared translator retained'
        elif key.startswith('jf.sem_proj.'):
            skipped[key]='entire SigLIP projection reinitialized for pinned Qwen1024'
        else:
            if key not in state:raise KeyError('unexplained reference key '+key)
            if state[key].shape!=value.shape:raise ValueError('unexplained shape mismatch '+key)
            accepted[key]=value
    state.update(accepted);model.load_state_dict(state,strict=True)
    changed=initialize_added_identity_blocks(model)
    return {'loaded_reference_keys':sorted(accepted),'skipped_reference_keys':skipped,'new_keys':sorted(set(state)-set(reference)),'identity_initialized_added_blocks':changed,'claim':'Added residual blocks start as identity. Qwen projection and sparse-condition modules are new; the entire new model is not numerically identical to the reference.'}
