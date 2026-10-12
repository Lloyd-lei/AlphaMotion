"""Joint-mask sampling for E0b masked pretraining. Mask semantics: True = this
joint's POSE is hidden from the encoder (replaced by a learned mask token). The
joint's geometry/semantics conditioning stays visible — the skeleton is always
known at deployment; only the pose observation is partial.

Families (mirror the product surface):
  rand    keep a random joint subset (keep ratio ~ U(0.2, 0.8))
  ee      keep only root + kinematic-tree leaves (end effectors; topology-agnostic)
  root    keep only the root
  sparse  keep only root + 1..3 random leaves — the "single-wrist SE3 constraint"
          product case. Added 2026-08-03 after the demo showed root+wrist (2.5%
          keep) was outside rand's 20-80% range: the mask family distribution IS
          the product surface.

Greenwich-v2 接触族 (0831, 需要调用方传 contact=[B,J]; 不列进 mix 就一行代码都不走,
既往 run 逐位不变):
  cgive   接触关节可见, 其余全遮 —— 「这几个点在地上, 把身体补出来」。
  cask    接触关节被遮, 但它们的接触通道置 1 —— 「这个关节必须踩地, 你自己想办法」。
          lloyd 选定的主方案: 坐标是具身相关的, 「踩没踩」这个标签是具身无关的,
          所以 cask 给的约束能跨具身直接搬运, 而 cgive 给的坐标不能。
两族都不消耗任何 RNG (给定 contact 就是确定的), 所以混进 mix 不改其它族的随机流。
Root is never masked (it anchors the global frame).
"""
from __future__ import annotations
import numpy as np, torch

DEFAULT_MIX = {"rand": 0.4, "ee": 0.3, "root": 0.3}
SPARSE_MIX = {"rand": 0.3, "ee": 0.25, "root": 0.15, "sparse": 0.3}


def parse_mix(s):
    """'rand:0.4,ee:0.3,root:0.3' -> dict"""
    out = {}
    for part in str(s).split(","):
        k, v = part.split(":")
        out[k.strip()] = float(v)
    return out


def leaf_mask(spec):
    """[J] bool, True at kinematic-tree leaves."""
    J = spec.J
    is_parent = np.zeros(J, bool)
    for p in spec.parents:
        if p >= 0:
            is_parent[p] = True
    return torch.as_tensor(~is_parent)


def sample_joint_mask(spec, B, mix=None, device="cuda", generator=None,
                      contact=None, contact_bin=None, return_con=False):
    """[B,J] bool mask (True = pose hidden). Each row draws one family from `mix`.

    contact [B,J] ([0,1] 软标签, 进第 10 通道), contact_bin [B,J] bool (可选, 用于
    "算不算接触"的二值判定; 不给就用 contact>0.5)。分两个入口的理由: 通道要软值
    (信息更全), 而族判定要标注器过完滞回+去毛刺的 hard 标签(更保守可信)。cgive/cask
    两族需要它; 其它族一概不看。return_con=True 时返回 (mask, con_over), con_over
    [B,J] float = 要喂进第 10 通道的接触值 —— cask 行上被遮关节被写成 1.0, 其余
    位置保持真实标签。默认 return_con=False, 返回值与既往完全一致。
    """
    mix = mix or DEFAULT_MIX
    J = spec.J
    root = int(np.where(spec.parents < 0)[0][0])
    need_c = any(f in mix for f in ("cgive", "cask"))
    if need_c and contact is None:
        raise ValueError("mask_mix 里有 cgive/cask, 但没传 contact=[B,J]")
    cb = con_over = None
    if contact is not None:
        contact = torch.as_tensor(contact, device=device)
        # 0831: 必须 clone。contact 已是 float32 时 .float() 返回**自身**, 后面的
        # con_over[rows]=... 会就地改写调用方的 csoft, 使 train_dual 的门控还原
        # 成为空操作 (没被 mask_prob 选中的 cask 行, 通道也被硬化成 1.0)。
        con_over = contact.float().clone()
        cb = (torch.as_tensor(contact_bin, device=device) if contact_bin is not None
              else (contact > 0.5 if contact.dtype != torch.bool else contact))
    fams = list(mix.keys())
    w = torch.tensor([mix[f] for f in fams], device=device)
    fam_idx = torch.multinomial(w.expand(B, -1), 1, generator=generator).squeeze(1)  # [B]
    mask = torch.zeros(B, J, dtype=torch.bool, device=device)
    for fi, fam in enumerate(fams):
        rows = fam_idx == fi
        if not rows.any():
            continue
        n = int(rows.sum())
        if fam == "rand":
            keep = torch.rand(n, 1, device=device, generator=generator) * 0.6 + 0.2  # U(0.2,0.8)
            mask[rows] = torch.rand(n, J, device=device, generator=generator) > keep
        elif fam == "ee":
            hide = ~leaf_mask(spec).to(device)          # hide non-leaves
            mask[rows] = hide[None].expand(n, -1)
        elif fam == "root":
            mask[rows] = True
        elif fam == "cgive":
            # 接触关节可见, 其余遮住。没有任何接触的帧(腾空)退化成"只剩 root",
            # 语义上正是"这一帧什么都没碰到", 不另做特判。
            mask[rows] = ~cb[rows]
        elif fam == "cask":
            # 接触关节遮住 + 接触位置 1。注意通道能活下来靠的是 model_dual._trunk
            # 里"con_emb 加在 mask 之后"那一步, 两处必须同时成立。
            mask[rows] = cb[rows]
            con_over[rows] = torch.where(cb[rows], torch.ones_like(con_over[rows]),
                                         con_over[rows])
        elif fam == "sparse":
            # v2: hide all, reveal 1..3 random joints from the WHOLE body (not just leaves).
            # v1 sampled tree leaves only — on soma77 those are fingertips, so a wrist
            # constraint stayed out-of-distribution (demo: fingertip 2.65cm vs wrist 16.7cm).
            # The product contract is "any joint can be a constraint": sample accordingly.
            # Vectorized: rank random scores per row, reveal the k lowest-ranked joints.
            k = torch.randint(1, 4, (n, 1), device=device, generator=generator)
            rank = torch.rand(n, J, device=device, generator=generator).argsort(1).argsort(1)
            mask[rows] = rank >= k
        else:
            raise ValueError(f"unknown mask family {fam}")
    mask[:, root] = False
    if return_con:
        return mask, con_over
    return mask
