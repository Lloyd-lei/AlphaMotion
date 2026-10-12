"""Arm S: static dense-surface group tokens (encoder) + target descriptor attachment (decoder).

Encoder (body has g['dense_surface']):
  joint tokens  = exactly the SF preamble (inp, mask_tok, con_emb, slide tokens, plain descriptor)
  group tokens  = PointNet-lite over the K members [(p-c)/(0.05 reach), normal, region one-hot]
                  (shared MLP 24->d->d, max-pool) + MLP(centre/reach) + region_emb
                  + plain descriptor[group_anchor] + type_emb            (static shape, no posed dynamics)
  per encoder layer (sealed EncBlock weights shared by both streams, B-style concat semantics):
    joints : SF attention over joint keys (the sealed block.a call, bit-identical)
             + Z_l( attention(joint queries -> group keys/values, hop bias via anchors) )
             with Z_l a per-layer Linear(d,d,bias=False) initialised to ZERO
    groups : one softmax over [joint ; group] keys with the B hop construction
             (hop[i,anchor]+1, hop[anchor_p,anchor_q]+2, diag 0; surface_model.py:107-116)
  FSQ pooling reads the joint tokens only (as in SF).
Decoder: descriptor(target) + AttachmentCrossAttention(descriptor, target group tokens,
  hop(joint, group_anchor)+1)  (sealed class, output projection zero-initialised).

Why the split softmax for joint queries: a single softmax over [joint; group] keys changes
every joint token at init (normalisation over 256 extra keys), so S could never equal SF at
init.  With the zero-initialised Z_l, S == SF bit-for-bit at init (tested) while the group
branch receives gradient from step 1.  Bodies without a dense surface run the SF trunk.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from ._sealed import ConditionalMegatron, AttachmentCrossAttention, model_parameter_counts
from .slide_fix import SFMegatron
from .surface_api import N_REGIONS

GROUP_FEATURE_DIM = 3 + 3 + N_REGIONS  # 24 (plan text says 18: 3+3+12; REGIONS has 18 entries)


def _attention(q, k, v, bias):
    """softmax(q k^T / sqrt(dk) + bias) v ; bias broadcast [1,H,Lq,Lk]."""
    return F.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype))


class DenseSurfaceMegatron(SFMegatron):
    """Arm S (parent SF)."""

    def __init__(self, *args, dense_group_attention='full', **kwargs):
        super().__init__(*args, **kwargs)
        if dense_group_attention not in {'full', 'joints_only'}: raise ValueError('dense_group_attention')
        d = kwargs['d_model']; heads = kwargs['n_heads']
        self.dense_group_attention = dense_group_attention
        # SDPA arithmetic of the group branches (ComputeCounter only sees their Linear calls);
        # plain python int, multiply-add=2 convention, forward only; the trainer may read/reset it.
        self.dense_attention_flops = 0
        # all new modules are created AFTER the parent constructor (parent RNG stream intact)
        self.group_point_mlp = nn.Sequential(nn.Linear(GROUP_FEATURE_DIM, d), nn.GELU(), nn.Linear(d, d))
        self.group_centre_mlp = nn.Sequential(nn.Linear(3, d), nn.GELU(), nn.Linear(d, d))
        self.group_region_emb = nn.Embedding(N_REGIONS, d)
        nn.init.normal_(self.group_region_emb.weight, std=0.02)
        self.group_type = nn.Parameter(torch.randn(d) * .01)
        self.group_to_joint = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(len(self.enc))])
        for m in self.group_to_joint: nn.init.zeros_(m.weight)
        self.target_dense_attachment = AttachmentCrossAttention(d, heads)
        nn.init.zeros_(self.target_dense_attachment.out.weight); nn.init.zeros_(self.target_dense_attachment.out.bias)

    # ------------------------------------------------------------------ static tokens
    def group_static_tokens(self, st, descriptor_row):
        """[G,d] static group tokens for one body (row-invariant)."""
        pn = self.group_point_mlp(st['member_feat']).amax(1)
        tok = pn + self.group_centre_mlp(st['centre_feat']) + self.group_region_emb(st['group_region'].clamp(0, N_REGIONS - 1))
        return tok + descriptor_row[st['group_anchor']] + self.group_type

    @staticmethod
    def _dense(g):
        ds = g.get('dense_surface') if isinstance(g, dict) else None
        return None if ds is None else ds['static']

    # ------------------------------------------------------------------ decoder
    def descriptor(self, g):
        joint = ConditionalMegatron.descriptor(self, g)
        st = self._dense(g)
        if st is None: return joint
        b, j, _ = joint.shape
        tok = self.group_static_tokens(st, joint[0])
        patch = tok[None].expand(b, -1, -1)
        hop = (g['hop'][0][:, st['group_anchor']] + 1)[None].expand(b, -1, -1)
        valid = st['group_valid'][None].expand(b, -1)
        return joint + self.target_dense_attachment(joint, patch, hop, valid)

    # ------------------------------------------------------------------ encoder
    def relational_group_tokens(self, x, g, st, joint_mask, relational):
        """Hook for arm I; S adds nothing."""
        return None

    def _trunk(self, x, g, joint_mask=None, con=None, source_slide_normalized=None, relational=None):
        st = self._dense(g)
        if st is None:
            return super()._trunk(x, g, joint_mask, con, source_slide_normalized)
        # --- verbatim SF preamble (pilot_model.py:86-99) ---
        x, con = self._split_con(x, con)
        if joint_mask is not None:
            if joint_mask.dtype != torch.bool or joint_mask.shape != x.shape[:2]: raise ValueError('source hidden mask contract')
            x = torch.where(joint_mask[..., None], torch.zeros_like(x), x)
        if not bool(torch.isfinite(x).all()): raise ValueError('visible source pose must be finite')
        h = self.inp(x)
        if joint_mask is not None: h = torch.where(joint_mask[..., None], self.mask_tok.expand_as(h), h)
        if con is not None:
            valid = torch.isfinite(con)
            safe_con = torch.where(valid, con, torch.zeros_like(con))
            h = h + self.con_emb(safe_con) * valid.to(h.dtype)
        h = h + self.source_slide_tokens(source_slide_normalized, g, *x.shape[:2], x.device, joint_mask)
        desc = ConditionalMegatron.descriptor(self, g)
        h = h + desc
        # --- group stream ---
        b = h.shape[0]
        hp = self.group_static_tokens(st, desc[0])[None].expand(b, -1, -1)
        rel = self.relational_group_tokens(x, g, st, joint_mask, relational)
        if rel is not None: hp = hp + rel
        hop = g['hop'][0]; ga = st['group_anchor']; G = ga.shape[0]
        hop_jp = hop[:, ga] + 1
        hop_pj = hop[ga, :] + 1
        hop_pp = hop[ga][:, ga] + 2
        hop_pp.fill_diagonal_(0)
        for block, z in zip(self.enc, self.group_to_joint):
            h, hp = self._dense_block(block, z, h, hp, g['hop'], hop_jp, hop_pj, hop_pp)
        return h

    def _dense_block(self, block, z, h, hp, hop_j, hop_jp, hop_pj, hop_pp):
        attn = block.a; B, J, D = h.shape; G = hp.shape[1]; H = attn.h; dk = attn.dk; cap = attn.max_hop + 1
        n1h = block.n1(h); n1p = block.n1(hp)
        joint_self = attn(n1h, hop_j, None)  # sealed SF computation (bit-identical)
        qj = attn.q(n1h).view(B, J, H, dk).transpose(1, 2)
        kp = attn.k(n1p).view(B, G, H, dk).transpose(1, 2)
        vp = attn.v(n1p).view(B, G, H, dk).transpose(1, 2)
        bias_jp = attn.hop_emb(hop_jp.clamp(max=cap)).permute(2, 0, 1)[None]
        cross = _attention(qj, kp, vp, bias_jp).transpose(1, 2).reshape(B, J, D)
        h_new = h + (joint_self + z(cross))
        h_new = h_new + block.ff(block.n2(h_new))
        # group queries (B-style concatenation semantics: previous-layer keys of both streams)
        qp = attn.q(n1p).view(B, G, H, dk).transpose(1, 2)
        kj = attn.k(n1h).view(B, J, H, dk).transpose(1, 2)
        vj = attn.v(n1h).view(B, J, H, dk).transpose(1, 2)
        if self.dense_group_attention == 'full':
            keys = torch.cat([kj, kp], 2); vals = torch.cat([vj, vp], 2)
            hb = torch.cat([hop_pj, hop_pp], 1)
        else:
            keys, vals, hb = kj, vj, hop_pj
        bias_p = attn.hop_emb(hb.clamp(max=cap)).permute(2, 0, 1)[None]
        gout = _attention(qp, keys, vals, bias_p).transpose(1, 2).reshape(B, G, D)
        self.dense_attention_flops += 4 * B * D * (J * G + G * keys.shape[2])
        hp_new = hp + attn.o(gout)
        hp_new = hp_new + block.ff(block.n2(hp_new))
        return h_new, hp_new

    def encode(self, x, g, joint_mask=None, return_codes=False, con=None, source_slide_normalized=None, relational=None):
        h = self._trunk(x, g, joint_mask, con, source_slide_normalized, relational); b = len(h)
        zp, cp = self.fsqp(self.poolp(self.qp[None].expand(b, -1, -1), h))
        zr, cr = self.fsqr(self.poolr(self.qr[None].expand(b, -1, -1), h))
        return (zp, zr, (cp, cr)) if return_codes else (zp, zr)

    def encode_features(self, x, g, joint_mask=None, con=None, source_slide_normalized=None, relational=None):
        h = self._trunk(x, g, joint_mask, con, source_slide_normalized, relational); b = len(h)
        return (self.fsqp.features(self.poolp(self.qp[None].expand(b, -1, -1), h)),
                self.fsqr.features(self.poolr(self.qr[None].expand(b, -1, -1), h)))

    def _relational_context(self, source_root_height_m, roll_relations):
        return None

    def forward(self, source, source_geom, target_geom, source_hidden=None, source_contact=None, condition=None,
                source_slide_normalized=None, source_root_height_m=None, roll_relations=None):
        zp, zr = self.encode(source, source_geom, joint_mask=source_hidden, con=source_contact,
                             source_slide_normalized=source_slide_normalized,
                             relational=self._relational_context(source_root_height_m, roll_relations))
        return self.decode_conditional(zp, zr, target_geom, condition)

    def forward_kinematic(self, source, source_geom, target_geom, source_hidden=None, source_contact=None,
                          condition=None, source_slide_normalized=None, source_root_height_m=None, roll_relations=None):
        zp, zr = self.encode(source, source_geom, joint_mask=source_hidden, con=source_contact,
                             source_slide_normalized=source_slide_normalized,
                             relational=self._relational_context(source_root_height_m, roll_relations))
        return self.decode_kinematic(zp, zr, target_geom, condition)


ARM_PREFIXES = {
    'slide_fix': ('slide_head.',),
    'dense_static': ('group_point_mlp.', 'group_centre_mlp.', 'group_region_emb.', 'group_type'),
    'dense_encoder_gates': ('group_to_joint.',),
    'dense_decoder_attachment': ('target_dense_attachment.',),
    'relational_input': ('relational_mlp.',),
}


def arm_parameter_counts(model):
    counts = model_parameter_counts(model)
    by = {k: sum(p.numel() for n, p in model.named_parameters() if n.startswith(v)) for k, v in ARM_PREFIXES.items()}
    return {**counts, 'added_by_component': by}
