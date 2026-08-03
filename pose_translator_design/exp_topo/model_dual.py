"""Dual-stream topology-agnostic translator: SEPARATE FSQ codes for position and
rotation, so rotation (esp. twist) gets protected bits instead of being starved by
position in a shared bottleneck. Confirmed to unfreeze rotation learning.

encode(any source) -> (z_pos, z_rot) two FSQ code streams -> decode(any target).
"""
from __future__ import annotations
import torch, torch.nn as nn
from model import EncBlock, CrossBlock, FSQ, ResidualFSQ, JointFeature


class DualTranslator(nn.Module):
    def __init__(self, d_model=256, n_heads=8, enc_layers=4, dec_layers=4,
                 n_latent=32, fsq_stages=2, fsq_levels=(8, 8, 8, 5, 5, 5), use_sem=True,
                 single_codebook=False):
        super().__init__()
        # use_sem=True: SigLIP joint-name semantics as cross-topology bridge; drop harmful index emb
        self.jf = JointFeature(d_model, use_index=not use_sem, use_sem=use_sem)
        self.inp = nn.Linear(9, d_model)                      # global 6D rot + root-rel pos
        # E0b masked pretraining: replaces a hidden joint's POSE embedding; geometry stays visible
        self.mask_tok = nn.Parameter(torch.randn(d_model) * 0.02)
        self.enc = nn.ModuleList([EncBlock(d_model, n_heads) for _ in range(enc_layers)])
        self.single_codebook = single_codebook
        if single_codebook:
            # E0c: ONE shared query bank (2K tokens) + ONE FSQ, same total bit budget as dual.
            # encode returns (z, z) so decoders/training code are unchanged.
            self.qp = nn.Parameter(torch.randn(2 * n_latent, d_model) * 0.02)
            self.poolp = CrossBlock(d_model, n_heads)
            self.fsqp = ResidualFSQ(d_model, fsq_stages, fsq_levels) if fsq_stages > 1 else FSQ(d_model, fsq_levels)
            self.bits = self.fsqp.bits * 2 * n_latent
        else:
            self.qp = nn.Parameter(torch.randn(n_latent, d_model) * 0.02)
            self.qr = nn.Parameter(torch.randn(n_latent, d_model) * 0.02)
            self.poolp = CrossBlock(d_model, n_heads)
            self.poolr = CrossBlock(d_model, n_heads)
            self.fsqp = ResidualFSQ(d_model, fsq_stages, fsq_levels) if fsq_stages > 1 else FSQ(d_model, fsq_levels)
            self.fsqr = ResidualFSQ(d_model, fsq_stages, fsq_levels) if fsq_stages > 1 else FSQ(d_model, fsq_levels)
            self.bits = (self.fsqp.bits + self.fsqr.bits) * n_latent
        self.decp = nn.ModuleList([CrossBlock(d_model, n_heads) for _ in range(dec_layers)])
        self.decr = nn.ModuleList([CrossBlock(d_model, n_heads) for _ in range(dec_layers)])
        self.selfp = nn.ModuleList([EncBlock(d_model, n_heads) for _ in range(dec_layers)])
        self.selfr = nn.ModuleList([EncBlock(d_model, n_heads) for _ in range(dec_layers)])
        self.hpos = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 3))
        self.hrot = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 6))

    def encode(self, x, g, joint_mask=None, return_codes=False):
        """joint_mask [B,J] bool (True = pose hidden -> mask token). return_codes: also
        return the quantized grid values (probe input: leakage / partial-obs stability)."""
        h = self.inp(x)
        if joint_mask is not None:
            h = torch.where(joint_mask[..., None], self.mask_tok.expand_as(h), h)
        h = h + self.jf(g["rest_off"], g["bonelen"], g["depth"], g.get("sem"))
        for b in self.enc:
            h = b(h, g["hop"], None)
        B = h.shape[0]
        if self.single_codebook:
            z, cp = self.fsqp(self.poolp(self.qp[None].expand(B, -1, -1), h))
            zr, cr = z, cp
            zp = z
        else:
            zp, cp = self.fsqp(self.poolp(self.qp[None].expand(B, -1, -1), h))
            zr, cr = self.fsqr(self.poolr(self.qr[None].expand(B, -1, -1), h))
        if return_codes:
            return zp, zr, (cp, cr)
        return zp, zr

    def decode(self, zp, zr, g):
        qp = self.jf(g["rest_off"], g["bonelen"], g["depth"], g.get("sem"))
        qr = qp.clone()
        for cb, sb in zip(self.decp, self.selfp):
            qp = cb(qp, zp); qp = sb(qp, g["hop"], None)
        for cb, sb in zip(self.decr, self.selfr):
            qr = cb(qr, zr); qr = sb(qr, g["hop"], None)
        return self.hrot(qr), self.hpos(qp)         # (rot6d, pos)
