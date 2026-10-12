"""Same float32 sigmoid forward; cancellation-free analytic sigmoid derivative.

Original y*(1-y) backward becomes exactly zero after float32 y rounds to 1.
This changes no inference values, parameters, joint bounds or mathematical loss.
The hook applies only to the named slide decoder, never globally to torch.
"""
import torch
import kinematics
import pilot_model

class StableSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x):
        ctx.save_for_backward(x)
        return torch.sigmoid(x)
    @staticmethod
    def backward(ctx,gradient):
        (x,)=ctx.saved_tensors
        e=torch.exp(-x.abs())
        return gradient*e/(1+e).square()

@kinematics.geometry_float32
def decode_slide(logits,g,metadata=None):
    b,j=logits.shape;m=metadata or kinematics.joint_metadata(g,b,j,logits.device)
    normalized=torch.where(m['slide_valid'],StableSigmoid.apply(logits.float()),torch.zeros_like(logits,dtype=torch.float32))
    native=torch.where(m['slide_valid'],m['lower']+normalized*(m['upper']-m['lower']),torch.zeros_like(normalized))
    return normalized,native,m['slide_valid']

def install():
    pilot_model.decode_slide=decode_slide
