"""Dual-stream topology-agnostic translator: SEPARATE FSQ codes for position and
rotation, so rotation (esp. twist) gets protected bits instead of being starved by
position in a shared bottleneck. Confirmed to unfreeze rotation learning.

encode(any source) -> (z_pos, z_rot) two FSQ code streams -> decode(any target).
"""
from __future__ import annotations
import torch, torch.nn as nn
from model import EncBlock, CrossBlock, FSQ, ResidualFSQ, JointFeature
from common import rot6d_to_matrix, matrix_to_rot6d


class JudgeHead(nn.Module):
    """Per-embodiment sigmoid critics reading the quantized codes ("像不像我负责的具身").
    Sits between codes and decoder; whether its gradients reach the backbone is the
    caller's choice (detach or not) — that is the E14 experimental variable."""
    def __init__(self, d_model, n_emb, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4 * d_model, hidden), nn.GELU(),
                                 nn.Linear(hidden, n_emb))

    def forward(self, zp, zr):                          # [B,K,d] x2 -> logits [B,n_emb]
        h = torch.cat([zp.mean(1), zp.amax(1), zr.mean(1), zr.amax(1)], -1)
        return self.net(h)


class DualTranslator(nn.Module):
    def __init__(self, d_model=256, n_heads=8, enc_layers=4, dec_layers=4,
                 n_latent=32, fsq_stages=2, fsq_levels=(8, 8, 8, 5, 5, 5), use_sem=True,
                 single_codebook=False, n_judges=0, use_dof=False, rot_head="free",
                 sem_dim=768, vis_residual=False, rot_only=False,
                 contact_chan=False, root_head=False, n_latent_p=0):
        super().__init__()
        self.rot_head = rot_head
        # 0812: with rot_only the position branch is gone and ALL n_latent
        # slots feed the rotation stream — the ablation showed the position
        # codes cannot reach the render at all, so this spends the same code
        # budget on the half that can.
        self.rot_only = rot_only
        # E28 vis_residual: joints whose pose the ENCODER actually saw are decoded as a
        # RESIDUAL on the observed pose (final = observed (+) delta) instead of being
        # re-invented from the bottleneck — attacks the measured E19 failure where the
        # model's own encode-decode of FULLY VISIBLE input cost 11.88 cm at the wrist.
        # Default False = byte-exact reproduction of every prior run (no extra modules,
        # so old checkpoints strict-load either way).
        self.vis_residual = vis_residual
        if vis_residual:
            # zero-init weights; rot bias = identity in 6D ([1,0,0,0,1,0]) so at init the
            # residual path is exactly "copy the observed pose".
            self.hrot_res = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 6))
            self.hpos_res = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 3))
            nn.init.zeros_(self.hrot_res[1].weight)
            nn.init.zeros_(self.hpos_res[1].weight)
            nn.init.zeros_(self.hpos_res[1].bias)
            with torch.no_grad():
                self.hrot_res[1].bias.copy_(torch.tensor([1., 0., 0., 0., 1., 0.]))
        self.judge = JudgeHead(d_model, n_judges) if n_judges else None
        # Greenwich-v2 (0831, contact_chan=False 默认 = 无新模块 = 老 ckpt 逐位 strict-load):
        #   con_emb  第 10 通道(逐关节软接触标签 in [0,1])的嵌入。**加在 mask 替换之后**,
        #            所以 cask(关节被遮但接触位=1)的信息能穿过 mask_tok 活下来 —— 这是
        #            cask 方案能成立的机械前提, 写在 _trunk 里。零初始化: 开关刚打开时
        #            函数与不开时逐位相同, 但梯度 dL/dW = grad_out * x 非零, 能自己长出来。
        #   hcon     解码器接触头 (BCE)。读 qr(旋转支路): 接触是姿态的确定性函数, 而
        #            旋转支路才是出画的那支; rot_only 下 qp 根本不算, 读 qr 两种模式通用。
        self.contact_chan = contact_chan
        self.root_head = root_head
        # use_sem=True: frozen joint-name text semantics as cross-topology bridge; drop harmful
        # index emb. sem_dim = the text encoder's width (768 SigLIP / 1024 Qwen3-Embedding).
        self.jf = JointFeature(d_model, use_index=not use_sem, use_sem=use_sem, use_dof=use_dof,
                               sem_dim=sem_dim)
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
            # n_latent_p (lloyd 0901): 位置流的码位预算, 独立于旋转流。Y3 砍到 4:
            # 位置头只管 root 一个槽 (高度+航向速度), 不该占旋转的仓库。0=旧行为。
            self.n_latent_p = n_latent_p or n_latent
            self.qp = nn.Parameter(torch.randn(self.n_latent_p, d_model) * 0.02)
            self.qr = nn.Parameter(torch.randn(n_latent, d_model) * 0.02)
            self.poolp = CrossBlock(d_model, n_heads)
            self.poolr = CrossBlock(d_model, n_heads)
            self.fsqp = ResidualFSQ(d_model, fsq_stages, fsq_levels) if fsq_stages > 1 else FSQ(d_model, fsq_levels)
            self.fsqr = ResidualFSQ(d_model, fsq_stages, fsq_levels) if fsq_stages > 1 else FSQ(d_model, fsq_levels)
            self.bits = self.fsqp.bits * self.n_latent_p + self.fsqr.bits * n_latent
        self.decp = nn.ModuleList([CrossBlock(d_model, n_heads) for _ in range(dec_layers)])
        self.decr = nn.ModuleList([CrossBlock(d_model, n_heads) for _ in range(dec_layers)])
        self.selfp = nn.ModuleList([EncBlock(d_model, n_heads) for _ in range(dec_layers)])
        self.selfr = nn.ModuleList([EncBlock(d_model, n_heads) for _ in range(dec_layers)])
        self.hpos = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 3))
        # rot_head="free"   : 6D GLOBAL rotation per joint (all runs up to E24).
        # rot_head="angles" : E25 arm A. 6D + 3 hinge angles. The 6D branch is used only
        #   for the root (a free joint) and for genuine 3-DOF joints; every hinge joint
        #   gets its LOCAL rotation from the angles, FK'd to global in decode(). The
        #   whole architectural delta is 3 extra output units = 3*d_model+3 params.
        self.hrot = nn.Sequential(nn.LayerNorm(d_model),
                                  nn.Linear(d_model, 9 if rot_head == "angles" else 6))

        # 0831: 接触模块必须建在**最后**。它们原本建在 jf/inp/mask_tok/enc/qp/qr 之前,
        # 于是 contact_chan 一开一关, 后面每个模块抽到的随机数全部错位 —— 实测同一
        # seed 下 188/298 个共享张量不同。消融的"chan 减 cons"就同时包含了机制本身和
        # 整个主干重新掷骰, 8000 步单种子下根本分不开。挪到末尾即可消除这一半混淆。
        if contact_chan:
            self.con_emb = nn.Linear(1, d_model)
            nn.init.zeros_(self.con_emb.weight); nn.init.zeros_(self.con_emb.bias)
            self.hcon = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        # 0901 lloyd: 根高槽位。逐帧可解的唯一世界量 (水平位移单帧无解, 属 Equator)。
        # 读旋转支路的根 query, 出一个标量 = 根离地高 (米)。推理时用于落地, 替换
        # studio 里"逐帧按最低点贴地"的弹跳式兜底。建在 __init__ 最末, RNG 流不动。
        if root_head:
            self.hroot = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def encode(self, x, g, joint_mask=None, return_codes=False, con=None):
        """joint_mask [B,J] bool (True = pose hidden -> mask token). return_codes: also
        return the quantized grid values (probe input: leakage / partial-obs stability).
        x 可以是 [B,J,9] (老口径) 或 [B,J,10] (v2: 第 10 通道 = 逐关节软接触标签);
        也可以走 con= 显式传 [B,J] / [B,J,1]。9 通道 + con=None => 与既往逐位一致。"""
        h = self._trunk(x, g, joint_mask, con)
        B = h.shape[0]
        if self.single_codebook:
            z, cp = self.fsqp(self.poolp(self.qp[None].expand(B, -1, -1), h))
            zr, cr = z, cp
            zp = z
        else:
            if self.rot_only:
                zr, cr = self.fsqr(self.poolr(self.qr[None].expand(B, -1, -1), h))
                zp, cp = zr, cr           # alias: no separate position code
            else:
                zp, cp = self.fsqp(self.poolp(self.qp[None].expand(B, -1, -1), h))
                zr, cr = self.fsqr(self.poolr(self.qr[None].expand(B, -1, -1), h))
        if return_codes:
            return zp, zr, (cp, cr)
        return zp, zr

    def _split_con(self, x, con=None):
        """[B,J,10] -> ([B,J,9], [B,J,1])。9 通道且 con=None 时原样返回 (老路径)。"""
        if x.shape[-1] > 9:
            assert self.contact_chan, \
                "传了 10 通道姿态但模型不是 contact_chan 建的 (会静默丢掉接触位)"
            assert con is None, "接触位不要同时用第 10 通道和 con= 传"
            x, con = x[..., :9], x[..., 9:10]
        if con is not None:
            assert self.contact_chan, "con= 需要 contact_chan=True 的模型"
            if con.dim() == 2:
                con = con[..., None]
        return x, con

    def _trunk(self, x, g, joint_mask=None, con=None):
        x, con = self._split_con(x, con)
        h = self.inp(x)
        if joint_mask is not None:
            h = torch.where(joint_mask[..., None], self.mask_tok.expand_as(h), h)
        if con is not None:
            # 顺序是纪律: 接触位必须加在 mask 替换之后。cask 的语义是
            # 「这个关节被遮住了, 但它必须踩在地上」—— 若先加后遮, mask_tok 会把
            # 接触位一起抹掉, cask 退化成普通遮挡, 整个方案静默失效。
            # 0901: NaN = "这一格没有接触信息" —— 嵌入贡献置零, 与推理时 con=None
            # 完全等价。之前 dropout 用 -1 当哨兵, 在距离语义里 -1 = "所有关节确定
            # 悬空", 30% 的训练步在喂与姿态矛盾的断言; 而考试 con=None 又是模型从没
            # 见过的第三态 -> Y1 无通道评测崩到 40.6°, 翻转方向命中 51% (掷硬币)。
            _v = (~torch.isnan(con)).float()
            h = h + self.con_emb(torch.nan_to_num(con)) * _v
        h = h + self.jf(g["rest_off"], g["bonelen"], g["depth"], g.get("sem"), g.get("dof"))
        for b in self.enc:
            h = b(h, g["hop"], None)
        return h

    def encode_features(self, x, g, joint_mask=None, con=None):
        """Pre-quantization continuous features per stream: (fp, fr) each [B,K,f].
        This is the diffusion compute space; FSQ rounding of these = the token interface."""
        h = self._trunk(x, g, joint_mask, con)
        B = h.shape[0]
        fp = self.fsqp.features(self.poolp(self.qp[None].expand(B, -1, -1), h))
        if self.single_codebook:
            return fp, fp
        fr = self.fsqr.features(self.poolr(self.qr[None].expand(B, -1, -1), h))
        return fp, fr

    def ints_from_features(self, fp, fr):
        """(fp, fr) -> integer level tensors ([S,B,K,f], [S,B,K,f])."""
        return self.fsqp.ints(fp), (self.fsqp if self.single_codebook else self.fsqr).ints(fr)

    def decode_from_ints(self, ip, ir, g):
        """Integer levels -> full-body (rot6d, pos) on target skeleton g."""
        zp = self.fsqp.from_ints(ip)
        zr = (self.fsqp if self.single_codebook else self.fsqr).from_ints(ir)
        return self.decode(zp, zr, g)

    def decode(self, zp, zr, g, obs=None, vis=None, ret_con=False):
        """obs [B,J,9] / vis [B,J] (True = the ENCODER saw this joint's pose): E28
        vis_residual inputs, SAME-SKELETON decode ONLY. For visible joints the output
        becomes observed (+) delta (rotation: R_obs @ R_delta with R_delta init'd at
        identity; position: p_obs + dp). Masked joints decode from the codes as always.

        CROSS-BODY decode (different target skeleton) must pass obs=None: there is no
        observed target pose, the visibility mask is source-side, and the codes must
        keep carrying the full motion for translation — the residual path is therefore
        structurally unable to become an identity shortcut for cross-body decode.
        Callers (dual_common.run_dir, eval harnesses) enforce src == tgt before
        passing obs."""
        qp = self.jf(g["rest_off"], g["bonelen"], g["depth"], g.get("sem"), g.get("dof"))
        qr = qp.clone()
        if not self.rot_only:
            for cb, sb in zip(self.decp, self.selfp):
                qp = cb(qp, zp); qp = sb(qp, g["hop"], None)
        for cb, sb in zip(self.decr, self.selfr):
            qr = cb(qr, zr); qr = sb(qr, g["hop"], None)
        h = self.hrot(qr)
        if self.rot_head == "angles":               # E25 arm A: hinge angles -> FK -> global
            from dof_head import head_to_global6d
            assert g.get("kin") is not None, "rot_head='angles' needs g['kin'] (DOF descriptor)"
            h = head_to_global6d(h[..., :6], h[..., 6:9], g["kin"])
        if self.rot_only:
            # 0812: the position head is gone. Returning zeros here was a silent
            # trap — evaluate() computes |pos - gt| and would report |gt| as if
            # it were an error (it printed 21-48 cm of pure garbage in the smoke
            # run). Return the FK positions instead: the same quantity every
            # reported number already comes through, so any caller that wants
            # positions gets the correct ones.
            from dual_common import fk_pos_torch
            pos = fk_pos_torch(h, g["spec"]) / g["reach"] if "spec" in g \
                else h.new_zeros(h.shape[0], h.shape[1], 3)
        else:
            pos = self.hpos(qp)
        if self.vis_residual and obs is not None:
            assert vis is not None, "vis_residual decode with obs needs the visibility mask"
            R_obs = rot6d_to_matrix(obs[..., :6])
            R_delta = rot6d_to_matrix(self.hrot_res(qr))
            h_res = matrix_to_rot6d(R_obs @ R_delta)
            p_res = obs[..., 6:9] + self.hpos_res(qp)
            m = vis[..., None]
            h = torch.where(m, h_res, h)
            pos = torch.where(m, p_res, pos)
        self.last_pos = pos                         # Y3 root 槽监督从这里读
        if getattr(self, "root_head", False):
            # 根 query 在 qr 的下标由调用方 g 提供 (spec.parents<0); 缺省取 0 (根总在 0 位)
            self.last_root_h = self.hroot(qr[:, g.get("root_idx", 0)]).squeeze(-1)
        if ret_con:
            # v2 接触头: [B,J] logit。始终由调用方显式索取 (默认 False), 所以
            # 所有既有调用点 (eval_dual/bridge_*/v2_judge/amplitude_report) 一字不改。
            assert self.contact_chan, "ret_con=True 需要 contact_chan=True 的模型"
            return h, pos, self.hcon(qr).squeeze(-1)
        return h, pos                               # (global rot6d, pos)
