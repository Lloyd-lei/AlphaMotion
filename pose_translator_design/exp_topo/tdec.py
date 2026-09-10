"""时序解码器 (0901, Z 系列 ③-lite): 码不动, 解码器看相邻 K 帧的码。

为什么: WBC 可训性验收 (wbc_ready_audit) 判定走路数据滑冰 (踩地脚世界系 0.35 m/s, 真值 0.02)。
根因两半 (chk_slipw/chk_avg 神谕实验, Y5b, LAFAN 走路段):
  * 跳过 FSQ 取整直接解码 -> 0.33 -> 0.17 m/s : 一半是相邻帧码 ±1 格抖动;
  * 零训练把相邻 5 帧的码平均再解 -> 0.37 -> 0.23 m/s, 姿态误差不变 : 时序上下文能消抖;
  * G1 自编码和人体->G1 一样滑 (0.33 vs 0.35) : 不是翻译的病, 是单帧解码器的精度地板
    (脚点相邻帧不一致 0.7cm × 30fps = 0.22 m/s)。
单帧解码器只训解码器 (Z1/Z2) 世界脚速卡在 0.38, 权重 2 -> 10 只挪 0.01: 不是权重问题, 是结构问题。

做法: 编码器、码本、每帧的码全部不变 (Equator / 文本层 / token 接口零影响)。解码前加一个
"逐码位沿时间轴的小 transformer" (axial: 每个码位只看自己在 K 帧里的值), 输出投影零初始化
=> 初始时逐位等于 Y5b 的单帧解码, 训练再学会用邻帧消抖 + 把踩地脚钉住 (世界系脚速损失)。
接口: 训练时 batch 是 W 个窗 × K 帧压平成 [W*K] 行 (行序 = 窗内时间序); decode 看 self.K
把 [W*K] 还原成 [W,K]。推理: decode_clip_windows —— 每帧取以它为中心的 K 帧窗, 只留中心输出,
所以相邻帧的上下文几乎相同 (无窗边界跳变)。片段头尾复制首/末帧补齐。
"""
import json, os
import numpy as np
import torch
import torch.nn as nn


def window_starts(cache, N, tr_idx, K, max_rows=0):
    """行 s..s+K-1 都在同一片段且都是训练行 的起点表 (K=2 即 slip_pair_starts)。"""
    caches = [c for c in str(cache).split(",") if c]
    ok = np.zeros(N, dtype=bool)
    off = 0
    for c in caches:
        sq = f"{c}/sequences.json"
        assert os.path.exists(sq), f"需要 {sq} (行->片段映射); 该缓存没有"
        seqs = json.load(open(sq))
        for s in seqs:
            a, b = off + s["start"], off + s["start"] + s["len"]
            if a >= N:
                break
            b = min(b, N)
            if b - a >= K:
                ok[a:b - K + 1] = True
        off += sum(s["len"] for s in seqs)
    is_tr = np.zeros(N, dtype=bool)
    is_tr[np.asarray(tr_idx.cpu() if torch.is_tensor(tr_idx) else tr_idx)] = True
    allt = is_tr.copy()
    for k in range(1, K):
        allt &= np.roll(is_tr, -k)
    ok &= allt
    starts = np.flatnonzero(ok)
    assert len(starts) > 0, f"K={K}: 没有任何合法窗"
    return starts


class TemporalFuse(nn.Module):
    """[B,K,T,d] -> [B,K,T,d]: 每个码位 T 独立, 沿 K 做 n_layers 层自注意力; 零初始化残差。"""
    def __init__(self, d, K, n_heads=4, n_layers=2):
        super().__init__()
        self.K = K
        self.temb = nn.Parameter(torch.zeros(K, d))
        nn.init.normal_(self.temb, std=0.02)
        layer = nn.TransformerEncoderLayer(d, n_heads, 2 * d, dropout=0.0,
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.out = nn.Linear(d, d)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, z):                                   # [B,K,T,d]
        B, K, T, d = z.shape
        x = z.permute(0, 2, 1, 3).reshape(B * T, K, d) + self.temb[:K][None]
        x = self.enc(x)
        x = self.out(x).reshape(B, T, K, d).permute(0, 2, 1, 3)
        return z + x


class TemporalDecoder(nn.Module):
    """包住 DualTranslator: encode 原样; decode 在 [W*K] 行上先做时序融合再走原解码器。
    属性访问缺省转发到 base (last_pos / judges / fsqp ...), 所以 run_dir/evaluate 一字不改。"""
    def __init__(self, base, K, n_layers=2):
        super().__init__()
        self.base = base
        self.K = K
        d = base.fsqp.proj_out.out_features
        self.tfp = TemporalFuse(d, K, n_layers=n_layers)
        self.tfr = self.tfp if base.single_codebook else TemporalFuse(d, K, n_layers=n_layers)
        self.window_mode = True
        self.last_prot = None

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("base"), name)

    def encode(self, *a, **k):
        return self.base.encode(*a, **k)

    def encode_features(self, *a, **k):
        return self.base.encode_features(*a, **k)

    def fuse(self, zp, zr):
        K = self.K
        B = zp.shape[0]
        assert B % K == 0, f"窗模式要求行数是 K={K} 的倍数, 得到 {B}"
        W = B // K
        zp = self.tfp(zp.reshape(W, K, *zp.shape[1:])).reshape(B, *zp.shape[1:])
        zr = self.tfr(zr.reshape(W, K, *zr.shape[1:])).reshape(B, *zr.shape[1:])
        return zp, zr

    def decode(self, zp, zr, g, obs=None, vis=None, ret_con=False):
        if self.window_mode:
            zp, zr = self.fuse(zp, zr)
        out = self.base.decode(zp, zr, g, obs=obs, vis=vis, ret_con=ret_con)
        self.last_prot = out[0]
        return out


def wrap_if_tdec(model, cfg):
    """按 config 决定要不要包时序解码器 (老 config 没 tdec 键 -> 原模型, 逐位不变)。"""
    K = int(cfg.get("tdec", 0) or 0)
    if K <= 1:
        return model
    return TemporalDecoder(model, K, n_layers=int(cfg.get("tdec_layers", 2))).to(
        next(model.parameters()).device)


def window_index(L, K, device):
    """长度 L 的片段 -> [L,K] 行下标 (以每帧为中心, 头尾复制)。中心位 = K//2。"""
    h = K // 2
    t = torch.arange(L, device=device)[:, None] + (torch.arange(K, device=device) - h)[None]
    return t.clamp_(0, L - 1)


@torch.no_grad()
def decode_clip_windows(model, zp, zr, g_fn, K, bs_rows=4096):
    """片段的码 zp/zr [L,...] -> 逐帧中心输出 (rot6d [L,J,6], pos [L,J,3])。
    g_fn(n) 给 n 行的目标几何 (= lambda n: geom(data, tgt, n))。"""
    L = zp.shape[0]
    idx = window_index(L, K, zp.device)                       # [L,K]
    wpb = max(1, bs_rows // K)                                # 每批窗数
    outs_r, outs_p = [], []
    for i in range(0, L, wpb):
        w = idx[i:i + wpb].reshape(-1)
        r, p = model.decode(zp[w], zr[w], g_fn(len(w)))[:2]
        n = len(w) // K
        outs_r.append(r.reshape(n, K, *r.shape[1:])[:, K // 2])
        outs_p.append(p.reshape(n, K, *p.shape[1:])[:, K // 2])
    return torch.cat(outs_r), torch.cat(outs_p)


def slip_world(prot0, prot1, rw0, rw1, con, rows0, rows1, spec, rowmask=None):
    """世界系接触关节水平速度损失 (裸量, (m/s)^2) 和 [B] 有效掩码。
    v = (Δ脚底点/100 + Δ根世界) 的水平模 × FPS; 权 = soft × 两帧硬接触 × 两帧可信; 逐帧按接触数归一。"""
    from dual_common import fk_pos_torch, CONTACT_HORIZ, CONTACT_FPS
    from common import rot6d_to_matrix
    rows = torch.cat([rows0, rows1]); B = len(rows0)
    ch = con.hard(rows); cs = con.soft(rows); ctr = con.trust(rows)
    ww = cs[:B] * (ch[:B] & ch[B:]).float() * (ctr[:B] & ctr[B:]).float()[:, None]
    if rowmask is not None:                                 # [B] bool: 只统计这些帧对
        ww = ww * rowmask.to(ww.device).float()[:, None]
    prot = torch.cat([prot0, prot1])
    pw = fk_pos_torch(prot, spec)
    if con.sole_j is not None:
        js = con.sole_j
        Rj = rot6d_to_matrix(prot[:, js])
        pw = pw.clone()
        pw[:, js] = pw[:, js] + torch.einsum("bkxy,ky->bkx", Rj, con.sole_off) * 100.0
    dxw = (pw[B:] - pw[:B]) / 100.0 + (rw1 - rw0)[:, None, :]
    vw = dxw[..., CONTACT_HORIZ].norm(dim=-1) * CONTACT_FPS
    den = ww.sum(1); act = den > 1e-6
    if not bool(act.any()):
        return None
    return ((ww * vw ** 2).sum(1)[act] / den[act]).mean()
