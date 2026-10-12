"""Arm I: source-side relational inputs on the posed SOURCE dense surface.

Per group g (members m in g, posed from the masked source pose9; hidden joints are zeros):
  ground (needs the root-height sensor): h_m = y_m + 100*rh - beta_vis_cm   [cm]
     [slog(min h), mean h / 50, frac(h < 3 cm), mean n_y of members with h < 3 cm]
  self-contact: nearest ALLOWED, VISIBLE point of the sub1024 subset (allowed_anchor_pairs)
     [ln(1 + dmin/2)/ln 26 (dmin clamped to 100 cm), unit direction member->partner in the
      group-anchor frame (R_anchor^T), partner-region one-hot (18)]
  valid bits [ground_valid, self_valid]                                   -> 28 features
  ground_valid = sensor finite & anchor visible & >=1 visible member
  self_valid   = anchor visible & an allowed visible partner exists
Point visibility = every joint that moves the point (dependency mask) is visible.
MLP 28->d->d (last layer zero-initialised) added to the group token, masked with torch.where.

The features are computed under no_grad from data (no gradient into the pose), only on calls
that pass ``relational=...`` (forward/forward_kinematic); bare encode() (contrast, judge, cycle
re-encode of raw_hpos) never computes them.
Roll twin (I_roll / eval reliance test): row i receives the features of row (i+B//2) % B
(each block already masked by row j's own validity), then gated by row i's OWN validity.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from ._sealed import rot6d_to_matrix
from .dense_surface import DenseSurfaceMegatron
from . import surface_api as sa

REL_FEATURE_DIM = 4 + 4 + sa.N_REGIONS + 2  # 28
NEAR_GROUND_CM = 3.0
BIG = 1.0e4
DMIN_CLAMP_CM = 100.0
CDIST_CHUNK = 16


def slog(x):
    return torch.sign(x) * torch.log1p(x.abs())


@torch.no_grad()
def relational_features(x, joint_mask, g, st, root_height_m=None, roll=False, chunk=CDIST_CHUNK):
    """x: masked source pose9 [B,J,9] (hidden rows already zero). Returns (feat [B,G,28], valid [B,G])."""
    asset = st.get('asset')
    if asset is None:
        raise ValueError('relational features need a real (non-deranged) dense surface for ' + str(st.get('body')))
    B, J, _ = x.shape
    dev = x.device
    with torch.autocast(device_type=dev.type, enabled=False):
        x = x.float()
        hidden = torch.zeros(B, J, dtype=torch.bool, device=dev) if joint_mask is None else joint_mask
        rot = rot6d_to_matrix(x[..., :6])
        scale = torch.as_tensor(g['pose_scale'], device=dev, dtype=torch.float32).reshape(-1)
        if scale.numel() == 1: scale = scale.expand(B)
        pos = x[..., 6:] * scale[:, None, None]
        members = st['group_members']; G, K = members.shape
        flat = st['member_flat']; sub = st['sub']; S = sub.shape[0]
        mp, mn = sa.pose_surface_subset(asset, rot, pos, flat)  # [B,GK,3]
        sp, _ = sa.pose_surface_subset(asset, rot, pos, sub)  # [B,S,3]
        mp = mp.float(); mn = F.normalize(mn.float(), dim=-1); sp = sp.float()
        dep = st['point_dep'].float()  # [P,J]
        point_hidden = (hidden.float() @ dep.T) > 0  # [B,P]
        if st.get('point_trusted') is not None:  # [integration] badly bound points are treated as never visible
            point_hidden = point_hidden | ~st['point_trusted'][None]
        mvis = ~point_hidden[:, flat]  # [B,GK]
        svis = ~point_hidden[:, sub]  # [B,S]
        anchor_vis = ~hidden[:, st['group_anchor']]  # [B,G]
        mvis3 = mvis.view(B, G, K)
        cnt = mvis3.sum(-1)
        # ---------------- ground ----------------
        if root_height_m is None:
            rh = torch.full((B,), float('nan'), device=dev)
        else:
            rh = torch.as_tensor(root_height_m, device=dev, dtype=torch.float32).reshape(B)
        rh_ok = torch.isfinite(rh)
        rh_safe = torch.where(rh_ok, rh, torch.zeros_like(rh))
        h = sa.ground_clearance(mp, rh_safe, st['beta_vis_cm']).view(B, G, K)
        hmin = torch.where(mvis3, h, torch.full_like(h, BIG)).amin(-1)
        hmean = torch.where(mvis3, h, torch.zeros_like(h)).sum(-1) / cnt.clamp_min(1)
        near = mvis3 & (h < NEAR_GROUND_CM)
        nnear = near.sum(-1)
        frac = nnear / cnt.clamp_min(1)
        ny = torch.where(near, mn.view(B, G, K, 3)[..., 1], torch.zeros_like(h)).sum(-1) / nnear.clamp_min(1)
        gvalid = rh_ok[:, None] & anchor_vis & (cnt > 0)
        ground = torch.stack([slog(hmin), hmean / 50.0, frac, ny], -1)
        ground = torch.where(gvalid[..., None], ground, torch.zeros_like(ground))
        # ---------------- self contact ----------------
        allowed = st['self_allowed']  # [GK,S]
        dmin = torch.empty(B, G * K, device=dev); arg = torch.empty(B, G * K, dtype=torch.long, device=dev)
        for s0 in range(0, B, chunk):
            s1 = min(B, s0 + chunk)
            # matmul form (the exact-difference CUDA kernel is ~50x slower); the selected pair's
            # distance is recomputed exactly below, so only near-ties can pick another partner.
            d = torch.cdist(mp[s0:s1], sp[s0:s1], compute_mode='use_mm_for_euclid_dist')
            ok = allowed[None] & svis[s0:s1, None, :]
            d = torch.where(ok, d, torch.full_like(d, BIG))
            dmin[s0:s1], arg[s0:s1] = d.min(-1)
        dmin = torch.where(mvis, dmin, torch.full_like(dmin, BIG)).view(B, G, K)
        dg_approx, kstar = dmin.min(-1)  # [B,G]
        partner = arg.view(B, G, K).gather(-1, kstar[..., None]).squeeze(-1)  # index into sub
        mpt = mp.view(B, G, K, 3).gather(2, kstar[..., None, None].expand(-1, -1, 1, 3)).squeeze(2)
        ppt = sp.gather(1, partner[..., None].expand(-1, -1, 3))
        vec = ppt - mpt
        dexact = vec.norm(dim=-1)
        dg = torch.where(dg_approx < BIG / 2, dexact, dg_approx)
        dirn = vec / dexact[..., None].clamp_min(1e-6)
        Ra = rot[:, st['group_anchor']]  # [B,G,3,3]
        dir_local = torch.einsum('bgji,bgj->bgi', Ra, dirn)  # R^T d
        svalid = anchor_vis & (dg < BIG / 2)
        dfeat = torch.log1p(dg.clamp(max=DMIN_CLAMP_CM) / 2.0) / math.log(26.0)
        selff = torch.cat([dfeat[..., None], dir_local], -1)
        selff = torch.where(svalid[..., None], selff, torch.zeros_like(selff))
        preg = st['region'][sub][partner].clamp(0, sa.N_REGIONS - 1)
        onehot = F.one_hot(preg, sa.N_REGIONS).float()
        onehot = torch.where(svalid[..., None], onehot, torch.zeros_like(onehot))
        bits = torch.stack([gvalid.float(), svalid.float()], -1)
        gblock = torch.cat([ground, bits[..., :1]], -1)  # 5
        sblock = torch.cat([selff, onehot, bits[..., 1:]], -1)  # 4+18+1
        if roll and B > 1:
            shift = -(B // 2)  # out[i] = in[(i + B//2) % B]
            gblock = torch.roll(gblock, shift, 0); sblock = torch.roll(sblock, shift, 0)
            gblock = torch.where(gvalid[..., None], gblock, torch.zeros_like(gblock))
            sblock = torch.where(svalid[..., None], sblock, torch.zeros_like(sblock))
        feat = torch.cat([gblock[..., :4], sblock[..., :4 + sa.N_REGIONS], gblock[..., 4:], sblock[..., -1:]], -1)
        return feat, gvalid | svalid


class RelationalInputMegatron(DenseSurfaceMegatron):
    """Arm I (parent S): relational source features added to the encoder group tokens."""

    def __init__(self, *args, roll_relations=False, **kwargs):
        super().__init__(*args, **kwargs)
        d = kwargs['d_model']
        self.roll_relations_default = bool(roll_relations)
        self.relational_mlp = nn.Sequential(nn.Linear(REL_FEATURE_DIM, d), nn.GELU(), nn.Linear(d, d))
        nn.init.zeros_(self.relational_mlp[2].weight); nn.init.zeros_(self.relational_mlp[2].bias)
        self.last_relational = None

    def _relational_context(self, source_root_height_m, roll_relations):
        return {'root_height_m': source_root_height_m,
                'roll': self.roll_relations_default if roll_relations is None else bool(roll_relations)}

    def relational_group_tokens(self, x, g, st, joint_mask, relational):
        if relational is None:
            self.last_relational = None
            return None
        feat, valid = relational_features(x, joint_mask, g, st, relational.get('root_height_m'), relational.get('roll', False))
        # tensors (no host sync); diagnostics only
        self.last_relational = {'valid_fraction': valid.float().mean(),
                                'ground_valid_fraction': feat[..., 4 + 4 + sa.N_REGIONS].mean(),
                                'self_valid_fraction': feat[..., -1].mean()}
        out = self.relational_mlp(feat)
        return torch.where(valid[..., None], out, torch.zeros_like(out))
