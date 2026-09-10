"""Shared data/eval logic for DualTranslator experiments (E0/E5 ablations).

Factored out of train_dual.py so training and the standalone eval harness
(eval_dual.py) run literally the same code path. Behavior-identical to the
FIX_* runs: same load order, same metrics.

RNG contract (load-bearing for split reproduction): the FIX split is
    torch.manual_seed(0) -> [data loading, consumes NO rng] -> torch.randperm(N)
Data loading consumes no RNG only because the SigLIP name-embedding disk cache
(semantics.CACHE) hits; it existed for all FIX-era runs. split_indices() must
therefore be called immediately after manual_seed(0) + load, BEFORE model init.
"""
from __future__ import annotations
import os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(__file__))
from common import rot6d_to_matrix, geodesic_deg
from model_dual import DualTranslator
from train_ae import static_cond
from train_global import build_global

FINGER_KW = ("Hand", "Thumb", "Index", "Middle", "Ring", "Pinky", "Finger")

# Greenwich-v2 接触层 (0831)。轴向口径与 contact_label.py / slip 副批完全一致:
# 缓存是 ardy Y-up, 所以"高"= 分量 1, "水平"= 分量 (0,2)。三处若不一致, 穿地和滑移
# 会互相抵消成一个看不出错的假数字, 所以常量只在这里定义一次。
CONTACT_UP = 1
CONTACT_HORIZ = (0, 2)
CONTACT_FPS = 30.0        # lafan/amass 统一, 与 slip 副批同值


class ContactStore:
    """一具身一份接触标注 (contact_label.py 产的 <emb>_contact.npz)。

    内存口径 (18 具身 x 5.2M 行的生产规模下这是必须算的账):
      soft  [N,J] fp16 CPU, 取批时 .float() 上卡 —— 和 PoseStore 同一套路。
      hard  [N,ceil(J/8)] uint8 **位打包**: 布尔标签用 fp16 存要 16 倍内存, 而它
            本来就只有 1 bit 信息量。取批时 np.unpackbits 一次 256xJ, 开销可忽略。
      gap_root [N] fp32 直接驻卡 (5M 行 = 20MB), 因为每步每方向都要它。

    为什么只留 gap 的根那一列: 见 fk_height_m —— 损失需要的"离地高"能由
    可微 FK + 根的 gap 精确重建, 其余 J-1 列在训练里是冗余的。
    """

    def __init__(self, soft16, hard_packed, gap_root, J, device, trust=None, sd16=None,
                 thr=0.04, sole_j=None, sole_off=None):
        self.s, self.hp, self.gr, self.J, self.device = soft16, hard_packed, gap_root, J, device
        # sd [N,J] fp16: 对数压缩的符号距离 (lloyd 0831 定, "距离进、置信出")。
        #   c = sign(thr-gap) * log1p(|thr-gap|/thr) / log1p(D/thr)
        # 输入通道喂它, 不喂概率: 概率会把"离地5cm"和"举在3m"都压成同一个数,
        # 实测 89% 的关节帧恒为 0, 通道嵌入零初始化下梯度死掉 (chan 臂聋的病根)。
        # 距离不饱和: 远处各带各的高度, 近处 log 给高分辨率。c>0=在阈值下(踩)。
        self.sd_ = sd16
        # 尺子 (0901 lloyd): thr 是该具身自己的接触阈值; sole_j/sole_off 是脚部关节
        # 的"关节中心 -> 真实脚底"局部偏移 (ardy 基, 米)。标签按脚底量, 损失也必须按
        # 脚底量 —— 之前损失用关节中心, 与标签差 3.5-16.8cm, cmaint 等于把身体往一个
        # 假地面上拽 (X3 在 gr3 幅度 1.30 的真凶)。
        self.thr = float(thr); self.tau = float(thr) / 3.0
        self.sole_j = sole_j                       # LongTensor [k] 或 None
        self.sole_off = sole_off                   # [k,3] 米, device 上
        # trust [N] bool: False = 标注器建不出这一段的地面 (contact_label.py 的第三态)。
        # 这**不是**"没有接触"—— 把它当 0 监督, 等于教模型"这个人走路不碰地"。
        # 三项接触损失必须整段屏蔽掉这些帧。缺该字段的老 npz -> 全 True (向后兼容)。
        self.tr = trust

    @staticmethod
    def _cpu(idx):
        return idx.cpu() if torch.is_tensor(idx) and idx.device.type != "cpu" else idx

    def soft(self, idx):
        return self.s[self._cpu(idx)].to(self.device, non_blocking=True).float()

    def hard(self, idx):
        i = np.asarray(self._cpu(idx))
        u = np.unpackbits(self.hp[i], axis=1, count=self.J)
        return torch.from_numpy(u.astype(bool)).to(self.device)

    def sdist(self, idx):
        """[B,J] float32 on device: 输入通道用的符号距离。"""
        assert self.sd_ is not None, "这份 ContactStore 没带 sd (老 npz?)"
        return self.sd_[self._cpu(idx)].to(self.device, non_blocking=True).float()

    def trust(self, idx):
        """[B] bool, True = 这一帧的接触标注可信, 可以进损失。"""
        if self.tr is None:
            n = len(idx) if hasattr(idx, "__len__") else int(idx.shape[0])
            return torch.ones(n, dtype=torch.bool, device=self.device)
        i = np.asarray(self._cpu(idx))
        return torch.from_numpy(self.tr[i]).to(self.device)

    def gap_root(self, idx):
        i = idx if (torch.is_tensor(idx) and idx.device == self.gr.device) else \
            torch.as_tensor(np.asarray(self._cpu(idx))).to(self.gr.device)
        return self.gr[i]


def fk_height_m(rot6d, spec, gap_root):
    """可微的"离地高" [B,J] (米): 从预测的全局旋转做 FK, 再用 GT 根关节的 gap 定地板。

    恒等式 (0831 在 LAFAN human_smpl 上实测过, 最大偏差 0.7mm = fp16 存储精度):
        h_j - h0  ==  fk_y_j(根相对, 米) + gap_root
    因为 gap_root = Y_root_world - h0, 而 fk 的根就在原点。
    意义: 训练缓存里根本没有世界根 (pose 是根相对的), 单靠 FK 拿不到"离地多高";
    这条恒等式让我们只借一个标量(GT 根的 gap, 常量、不回传梯度)就补上了那一维,
    于是穿模/接触/滑移三项都能经可微 FK 把梯度送回码本 —— spec §7 的全部机关。
    """
    p = fk_pos_torch(rot6d, spec)                       # [B,J,3] cm, 根在原点
    return p[..., CONTACT_UP] / 100.0 + gap_root[:, None]


def build_pose16(parts, spec, device, chunk=4096):
    """内存版 build_global: parts = 各缓存的 local6d (可 mmap), 按顺序拼成一张 fp16 [N,J,9]
    (global rot6d | global pos/reach). 第一遍只算 reach (= 所有帧根相对位置模长最大值,
    与 build_global 同定义), 第二遍填表. 与 build_global 的结果逐位一致."""
    from train_global import fk_global, matrix_to_rot6d
    if torch.is_tensor(parts): parts = [parts]
    N = sum(p.shape[0] for p in parts); J = parts[0].shape[1]
    reach = 0.0
    for p in parts:
        for i in range(0, p.shape[0], chunk):
            _, gp = fk_global(p[i:i + chunk].to(device), spec)
            reach = max(reach, gp.norm(dim=-1).max().item())
    reach = max(reach, 1e-6)
    out = torch.empty(N, J, 9, dtype=torch.float16); o = 0
    for p in parts:
        for i in range(0, p.shape[0], chunk):
            gR, gp = fk_global(p[i:i + chunk].to(device), spec); n = gR.shape[0]
            out[o:o + n, :, :6] = matrix_to_rot6d(gR).cpu().half()      # 与 build_global 同: 回 CPU 再转
            out[o:o + n, :, 6:] = (gp.cpu() / reach).half(); o += n
    assert o == N
    return out, reach


class PoseStore:
    """0809 full-13k surgery: 5.1M rows x 496 joints no longer fit resident on GPU.
    Rows live fp16 on CPU; indexing gathers the batch and returns fp32 on `device` —
    every consumer does data[n]["pose"][idx], so this is a drop-in. ~36MB/step of
    PCIe traffic at bs 256 x 8 combos, invisible next to the forward pass."""

    def __init__(self, pose_cpu16, device):
        self.t = pose_cpu16
        self.device = device

    @property
    def shape(self):
        return self.t.shape

    def __len__(self):
        return self.t.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx) and idx.device.type != "cpu":
            idx = idx.cpu()
        return self.t[idx].to(self.device, non_blocking=True).float()


def load_paired_data(cache, device, max_rows=0, sem_encoder="siglip", pose_device=None):
    """Load a paired cache -> per-embodiment dict. Consumes no RNG (see module doc).
    Manifest schema 2 lists N embodiments; legacy manifests imply [soma77, g1_name].
    max_rows > 0 truncates every embodiment to the first max_rows frames — SMOKE TESTS
    ONLY (it changes N and therefore the split); 0 = the real, unchanged path.
    sem_encoder picks the frozen joint-name text encoder ("siglip" 768 / "qwen3" 1024);
    its disk cache must already exist or the load consumes RNG and breaks the split.
    `cache` may be a comma list of caches with identical embodiment sets — rows are
    concatenated in the listed order (lafan,amass_full for the 0809 arms).
    pose_device: "gpu" = legacy resident-on-GPU fp32; "cpu16" = PoseStore; None = auto
    (cpu16 once total rows exceed 2.5M, printed loudly)."""
    import json, numpy as np, gc
    from common import SkeletonSpec
    caches = [c for c in str(cache).split(",") if c]
    man = json.load(open(f"{caches[0]}/manifest.json"))
    names = man.get("embodiments") or ["soma77", man["g1_name"]]
    # 0831 lloyd: pal_talos 的 URDF 有问题(标签侧也测到 h0=15.8cm 全场最高、
    # 16.9% 片段整段零接触), 整个语料层面剔除, 不训练也不当 holdout。
    # 覆盖用 AM_EXCLUDE_EMB=a,b (留空则不排除任何具身)。
    _ex = {x for x in os.environ.get("AM_EXCLUDE_EMB", "pal_talos").split(",") if x}
    if _ex:
        names = [n for n in names if n not in _ex]
    for c in caches[1:]:
        m2 = json.load(open(f"{c}/manifest.json"))
        n2 = m2.get("embodiments") or ["soma77", m2["g1_name"]]
        n2 = [n for n in n2 if n not in _ex]          # 0903: 排除表对每个缓存同样生效
        assert set(n2) == set(names), f"embodiment mismatch: {c}"
    specs = {}
    for name in names:
        d = np.load(f"{caches[0]}/{name}_spec.npz", allow_pickle=True)
        specs[name] = SkeletonSpec(str(d["name"]), d["parents"], d["rest_offsets"],
                                   list(d["joint_names"]))
    from semantics import build_name_embeddings, embed_matrix
    lut = build_name_embeddings(sum([list(specs[n].joint_names) for n in names], []),
                                device=device, encoder=sem_encoder)
    if pose_device is None:
        pose_device = os.environ.get("POSE_DEVICE") or None   # smoke-test override
    if pose_device is None:
        rows0 = sum(torch.load(f"{c}/{names[0]}_paired_local6d.pt", mmap=True).shape[0]
                    for c in caches)
        pose_device = "cpu16" if (rows0 if not max_rows else min(rows0, max_rows)) > 2_500_000 \
            else "gpu"
        if pose_device == "cpu16":
            print(f"[load_paired_data] {rows0} rows -> pose resident fp16 on CPU "
                  f"(PoseStore)", flush=True)
    data = {}; N = None
    for name in names:
        spec = specs[name]
        st = static_cond(spec, device)
        sem = embed_matrix(spec.joint_names, lut).to(device)     # [J,D] joint-name text semantics
        if pose_device == "cpu16":
            # 0903: 5.2M 行时 build_global 的 fp32 中间量 (list+cat+pose) 一台机器人就 20+GB,
            # 叠在已驻留的 fp16 上被 OOM 杀 (anon-rss 50.8GB). 改成: 各缓存 mmap 读 (页缓存,
            # 可回收, 不 cat), 分块 FK 直接写进预分配 fp16, 两遍 FK (第一遍只求 reach).
            # 数值与原路径一致到 fp32 分块噪声 (build_global 自己换 chunk 也差 1e-6 比例的 1 ulp).
            parts = [torch.load(f"{c}/{name}_paired_local6d.pt", mmap=True) for c in caches]
            if max_rows: parts = [parts[0][:max_rows]]
            pose16, reach = build_pose16(parts, spec, device)
            entry = {"spec": spec, "st": st, "pose": PoseStore(pose16, device), "reach": reach, "sem": sem}
            l6 = local6d = gr6 = gp = pose = None; N_ = pose16.shape[0]
        else:
            l6 = torch.cat([torch.load(f"{c}/{name}_paired_local6d.pt") for c in caches]) \
                if len(caches) > 1 else torch.load(f"{caches[0]}/{name}_paired_local6d.pt")
            local6d = l6[:max_rows].contiguous() if max_rows else l6; N_ = local6d.shape[0]
            gr6, gp, reach = build_global(local6d, spec, device)
            pose = torch.cat([gr6, gp / reach], -1)
            entry = {"spec": spec, "st": st, "pose": pose.to(device), "reach": reach, "sem": sem}
        dof_path = f"{caches[0]}/{name}_dof.npz"                 # mechanical spec (axes + limits)
        if os.path.exists(dof_path):
            entry["dof"] = torch.as_tensor(np.load(dof_path)["dof"], dtype=torch.float32, device=device)
            from dof_head import build_kin                       # E25: precomputed hinge kinematics
            entry["kin"] = build_kin(caches[0], name, spec, device=device)
        data[name] = entry
        N = N_; print(f"[{name}] J={spec.J} rows={N} reach={reach:.1f}cm sem={tuple(sem.shape)}")
        del l6, local6d, gr6, gp, pose
        gc.collect()
    return data, names, N


def split_indices(N, max_val=5000):
    """The FIX train/val split. Call order contract in module docstring.
    max_val caps eval size for big caches (251k rows -> 25k val OOMed the one-shot
    eval encode); FIX-era caches (48k -> 4800) are below the cap, bitwise unchanged."""
    perm = torch.randperm(N); nval = min(max(128, N // 10), max_val)
    return perm[:nval], perm[nval:]


def parse_fsq_levels(s):
    """'8,8,8,5,5,5' -> (8,8,8,5,5,5)"""
    return tuple(int(x) for x in str(s).split(","))


def build_model(cfg, device):
    """Construct DualTranslator from a config dict (train args or {run}/config.json).
    Unknown/missing keys fall back to FIX-era defaults so old configs stay loadable."""
    from semantics import SEM_DIM
    model = DualTranslator(
        d_model=cfg.get("d_model", 256),
        n_heads=cfg.get("n_heads", 8),
        enc_layers=cfg.get("enc_layers", 4),
        dec_layers=cfg.get("dec_layers", 4),
        n_latent=cfg.get("n_latent", 32),
        fsq_stages=cfg.get("fsq_stages", 2),
        fsq_levels=parse_fsq_levels(cfg.get("fsq_levels", "8,8,8,5,5,5")),
        single_codebook=cfg.get("single_codebook", False),
        n_judges=cfg.get("n_judges", 0),
        use_dof=cfg.get("use_dof", False),
        rot_head=cfg.get("rot_head", "free"),        # E25 arm A ("angles"); default = every prior run
        root_head=cfg.get("root_head", False),       # 0901 根高槽位
        n_latent_p=cfg.get("n_latent_p", 0),         # 0901 位置流码位预算 (Y3: 4)
        # E27: text-encoder width. Configs with no "sem_encoder" key -> 768, i.e. SigLIP.
        sem_dim=SEM_DIM[cfg.get("sem_encoder", "siglip")],
        # E28: residual decode of encoder-visible joints. Missing key -> False -> the
        # model has no extra modules and behaves bit-for-bit like every prior run.
        vis_residual=cfg.get("vis_residual", False),
                          rot_only=cfg.get("rot_only", False),
        # v2 (0831): 第 10 通道 + 接触头。老 config 没这个 key -> False -> 无新模块,
        # 既往 checkpoint 照旧 strict-load。
        contact_chan=cfg.get("contact_chan", False),
    ).to(device)
    return model


def geom(data, name, B):
    st = data[name]["st"]
    g = dict(rest_off=st["rest_off"][None].expand(B, -1, -1), bonelen=st["bonelen"][None].expand(B, -1),
             depth=st["depth"][None].expand(B, -1), hop=st["hop"][None].expand(B, -1, -1),
             sem=data[name]["sem"][None].expand(B, -1, -1))
    g["spec"] = data[name]["spec"]        # static; rot_only decode needs it for FK
    g["reach"] = data[name]["reach"]
    if "dof" in data[name]:
        g["dof"] = data[name]["dof"][None].expand(B, -1, -1)
        g["kin"] = data[name]["kin"]         # static, not batch-expanded (E25)
    return g


def fk_pos_torch(rot6d, spec):
    """DIFFERENTIABLE FK from GLOBAL rot6d -> joint positions (root at origin).

    Same recipe as v5_common.fk_pos (the numpy yardstick every reported position
    comes through), written in torch so the SAME quantity can enter the loss.
    Added 0812: the audit showed the 5.0-weighted position loss trains hpos,
    whose output is discarded at render time, leaving hrot — the head that
    actually determines the picture — supervised only by a per-joint geodesic
    that is blind to lever arms. This closes that loop."""
    R = rot6d_to_matrix(rot6d)                                  # [B,J,3,3]
    par = list(spec.parents)
    off = torch.as_tensor(np.asarray(spec.rest_offsets),
                          dtype=R.dtype, device=R.device)       # [J,3]
    B, J = R.shape[0], R.shape[1]
    cols = [None] * J
    order = sorted(range(J), key=lambda j: _depth(par, j))      # parents first
    for j in order:
        pj = par[j]
        if pj < 0:
            cols[j] = R.new_zeros(B, 3)
        else:
            cols[j] = cols[pj] + torch.einsum("bik,k->bi", R[:, pj], off[j])
    return torch.stack(cols, 1)                                 # [B,J,3]


def _depth(par, j):
    d = 0
    while par[j] >= 0:
        j = par[j]; d += 1
    return d


def run_dir(model, data, src, tgt, idx, joint_mask=None, visible_boost=0.0, dof_loss=False,
            extras=None, tail_alpha=0.0, arm_boost=0.0, ee_boost=0.0,
            fk_loss=False, pos_w=None, ee_fk_w=0.0,
            contact=False, con_over=None, cextras=None, con_bce=False):
    """One direction src->tgt on rows idx. Returns (mean geodesic deg, mean pos in reach units).
    extras (N3f --var_floor): pass a dict and it gains "var_floor" = the hinge
    max(0, std_gt - std_decoded) on the per-joint/per-dim std ACROSS THE BATCH of the
    decoded pose (rot6d + pos together), the anti-amplitude-compression penalty whose
    target is the GT batch's own variance. None (default) = bit-exact prior behavior.
    joint_mask [B,J_src]: E0b masked pretraining — encoder sees partial pose, loss stays full-body.
    visible_boost > 0 (same-skeleton directions only): upweight the VISIBLE joints' loss by
    (1+boost) so the completion learns to honor given constraints, not just plausibility.
    dof_loss (E25 arm B, TRAINING ONLY): before the geodesic, project prediction and target
    local rotations onto each joint's hinge manifold, so the loss never charges the model
    for components the mechanism cannot realize. Genuine 3-DOF joints and the root pass
    through untouched. Eval never sets this — the yardstick stays the raw global geodesic.

    v2 接触 (0831, 三个参数全默认 = 与既往逐位一致):
      contact=True  编码器输入接成 [B,J,10], 第 10 通道 = SOURCE 具身该帧的软接触标签
                    (缺标注的具身喂零 -> 通道在结构上还在, 信息为空)。
      con_over      [B,J] 覆盖上面那个通道 (cask 用: 被遮关节写 1)。
      cextras       传个 dict 就地填 TARGET 具身上的三项接触量 (pen/cmaint/cbce);
                    权重由调用方乘, 这里只出裸量。con_bce=True 时额外要解码器的接触头。
    """
    B = len(idx)
    x9 = data[src]["pose"][idx]
    xin = x9
    if contact:
        cs = con_over
        if cs is None:
            cs = (data[src]["con"].soft(idx) if data[src].get("con_ok")
                  else x9.new_zeros(B, x9.shape[1]))
        xin = torch.cat([x9, cs[..., None].to(x9.dtype)], -1)
    zp, zr = model.encode(xin, geom(data, src, B), joint_mask=joint_mask)
    obs = vis = None
    if getattr(model, "vis_residual", False) and src == tgt:
        # E28: SAME-SKELETON decode only — the decoder may lean on the pose the encoder
        # actually saw (final = observed + delta). joint_mask is SOURCE-side, so for a
        # cross-body direction (src != tgt) there is no observed target pose and the
        # residual path must stay off: translation still runs purely from the codes,
        # which keeps the codes honest (no identity shortcut can leak across bodies).
        obs = x9
        vis = ~joint_mask if joint_mask is not None else \
            torch.ones(B, obs.shape[1], dtype=torch.bool, device=obs.device)
    clog = None
    if cextras is not None and con_bce:
        prot, ppos, clog = model.decode(zp, zr, geom(data, tgt, B), obs=obs, vis=vis,
                                        ret_con=True)
    else:
        prot, ppos = model.decode(zp, zr, geom(data, tgt, B), obs=obs, vis=vis)
    gt = data[tgt]["pose"][idx]
    if extras is not None:
        # N3f: "match the data's variance" — decoded per-joint/dim std over the
        # batch must not fall below the GT batch's own (regression-to-mean cure).
        pv = torch.cat([prot, ppos], -1)                                    # [B,J,9]
        extras["var_floor"] = torch.relu(gt.std(0).detach() - pv.std(0)).mean()
    Rp, Rt = rot6d_to_matrix(prot), rot6d_to_matrix(gt[..., :6])
    if dof_loss and "kin" in data[tgt]:
        from dof_head import project_global6d
        kin = data[tgt]["kin"]
        Rp = project_global6d(Rp, kin)
        Rt = project_global6d(Rt, kin).detach()
    rot_j = geodesic_deg(Rp, Rt)                                               # [B,J]
    if fk_loss:
        # 0812: the position term is computed by running FK on the PREDICTED
        # rotations instead of reading the discarded hpos head, so the metric
        # error (which knows about lever arms) supervises the head that renders.
        # Same units as before: the cached pose stores gp / reach.
        fk_cm = fk_pos_torch(prot, data[tgt]["spec"])                            # [B,J,3] cm
        fkp = fk_cm / data[tgt]["reach"]
        fkg = fk_pos_torch(gt[..., :6], data[tgt]["spec"]) / data[tgt]["reach"]
        pos_j = (fkp - fkg).norm(dim=-1)                                        # [B,J]
    else:
        fk_cm = None
        pos_j = (ppos - gt[..., 6:9]).norm(dim=-1)                              # [B,J]
    if cextras is not None:
        contact_terms(cextras, data, tgt, idx, prot, clog, fk_cm)
    if pos_w is not None:
        # P1 方差归一化 (--var_norm, 0830 v2 计划): 每个关节的位置误差除以它自己
        # 在训练数据里的位移幅度 sigma_j。为什么: 无加权 L2 里腿(动得多、误差大)
        # 天然吃掉大头梯度, 手腕这种"动得多但杠杆长"的尾巴买不到梯度 —— 用数据统计
        # 代替 arm_boost/ee_boost 的关键词表, 不用人手挑关节。pos_w 已在 train_dual
        # 侧做过 1/sigma + 地板 + 均值归一(见 pos_sigma_weights), 这里只做逐关节乘法。
        # None (默认) = 与既往所有 run 逐位一致。
        pos_j = pos_j * pos_w[None]
    if ee_fk_w > 0:
        # P2 (0831): 末端-杠杆位置权重, 拓扑自动推 (见 ee_fk_weights)。均值 1,
        # 只重新分配位置项的梯度预算, 不改总量。0 (默认) = 逐位一致。
        ew = data[tgt].get("_ee_fk_w")
        if ew is None or float(data[tgt].get("_ee_fk_w_val", -1)) != float(ee_fk_w):
            ew = ee_fk_weights(data[tgt]["spec"], pos_j.device, ee_fk_w)
            data[tgt]["_ee_fk_w"] = ew; data[tgt]["_ee_fk_w_val"] = float(ee_fk_w)
        pos_j = pos_j * ew[None]
    if arm_boost > 0 or ee_boost > 0:
        # decoder post-train (0811): static per-joint pressure on the arm chain.
        # ee_boost hits the wrist/hand POSITION term only — the 0811 probes
        # showed ~17% of the wrist p95 tail is per-frame recoverable and it is
        # positional. mean-1 normalized; fingers excluded; composes with
        # tail_alpha below.
        w2 = data[tgt].get("_arm_w2")
        if w2 is None:
            KW = ("shoulder", "elbow", "wrist", "hand", "arm", "clavicle",
                  "collar", "scap", "bicep")
            EE = ("wrist", "hand", "palm")
            nl = [n.lower() for n in data[tgt]["spec"].joint_names]
            fin = [any(k.lower() in n for k in FINGER_KW) for n in nl]
            m = torch.tensor([any(k in n for k in KW) and not f
                              for n, f in zip(nl, fin)], device=rot_j.device)
            me = torch.tensor([any(k in n for k in EE) and not f
                               for n, f in zip(nl, fin)], device=rot_j.device)
            aw = 1.0 + arm_boost * m.float()
            pw = 1.0 + arm_boost * m.float() + ee_boost * me.float()
            w2 = (aw / aw.mean(), pw / pw.mean())
            data[tgt]["_arm_w2"] = w2
        rot_j = rot_j * w2[0][None]
        pos_j = pos_j * w2[1][None]
    if visible_boost > 0 and joint_mask is not None and src == tgt:
        w = 1.0 + visible_boost * (~joint_mask).float()
        w = w / w.mean()
        return (rot_j * w).mean(), (pos_j * w).mean()
    if tail_alpha > 0:
        # 0809 tail cure (backfist verdict): a mean objective lets the model abandon
        # low-density pose regions. Reweight each sample-joint by its OWN detached
        # relative error ^ alpha (mean-1 normalized), so hard poses buy gradient in
        # proportion to how badly they are being fit. TRAINING ONLY — evaluate()
        # never passes tail_alpha; the yardstick stays the plain mean.
        wr = (rot_j.detach() / (rot_j.detach().mean() + 1e-9)) ** tail_alpha
        wp = (pos_j.detach() / (pos_j.detach().mean() + 1e-9)) ** tail_alpha
        return (rot_j * (wr / wr.mean())).mean(), (pos_j * (wp / wp.mean())).mean()
    return rot_j.mean(), pos_j.mean()


def amplitude_report(model, data, dirs, pools, n_win=24, T=90, seed=0, starts=None):
    """AMPLITUDE RATIO — how much of the real motion's range survives decoding.

    Added 0813 after lloyd watched a punch render and said the robot barely
    moved. Mean position error cannot see this: collapsing toward the average
    pose is a LOW-error strategy, so a model can score well while producing
    motion that is visibly dead. Measured on contiguous windows (the metric is
    temporal; evaluate()'s random rows cannot express it).

    ratio = std_over_time(decoded FK positions) / std_over_time(GT FK positions),
    per joint, averaged. 1.0 = the real range; 0.2 = a body that barely moves.
    """
    if starts is None:
        g = np.random.default_rng(seed)
        ok = [(a, b) for a, b in pools if b - a >= T + 2]
        starts = [int(g.integers(a, b - T)) for a, b in
                  (ok[i] for i in g.choice(len(ok), n_win, replace=False))]
    else:
        # v2 判官 (0830): 窗口起点由调用方传死。为什么: P0 消融要求四条臂在
        # "同一批固定窗口"上对比, 抽样版每次靠 pools+seed 重画, 换 pools 就换卷子;
        # 判官把窗口列表落盘后原样传进来。None (默认) = 老抽样路径, amp_duel 不受影响。
        starts = [int(s) for s in starts]
    ARM = ("shoulder", "elbow", "wrist", "arm", "hand", "clavicle")
    LEG = ("hip", "knee", "ankle", "foot", "leg", "toe", "shank")
    out = {}
    was = model.training
    model.eval()
    with torch.no_grad():
        for src, tgt in dirs:
            spec = data[tgt]["spec"]
            nl = [x.lower() for x in spec.joint_names]
            am = torch.tensor([any(k in x for k in ARM) for x in nl])
            lm = torch.tensor([any(k in x for k in LEG) for x in nl])
            acc = {"all": [], "arm": [], "leg": []}
            for s0 in starts:
                rows = torch.arange(s0, s0 + T)
                p9 = data[src]["pose"][rows]
                prot, _ = model.decode(*model.encode(p9, geom(data, src, T)),
                                       geom(data, tgt, T))
                fp = fk_pos_torch(prot, spec).cpu()
                fg = fk_pos_torch(data[tgt]["pose"][rows][..., :6], spec).cpu()
                sp_, sg = fp.std(0).norm(dim=-1), fg.std(0).norm(dim=-1)
                for k, msk in (("all", None), ("arm", am), ("leg", lm)):
                    a_ = sp_ if msk is None else sp_[msk]
                    b_ = sg if msk is None else sg[msk]
                    if len(a_):
                        acc[k].append(float(a_.mean() / (b_.mean() + 1e-9)))
            out[f"{src}->{tgt}"] = {k: round(float(np.mean(v)), 3)
                                    for k, v in acc.items() if v}
    if was:
        model.train()
    return out


def swing_twist_report(model, data, name, idx):
    d = data[name]; B = len(idx)
    zp, zr = model.encode(d["pose"][idx], geom(data, name, B))
    prot, ppos = model.decode(zp, zr, geom(data, name, B))
    pr = rot6d_to_matrix(prot); gt = rot6d_to_matrix(d["pose"][idx][..., :6])
    a0 = torch.nn.functional.normalize(d["st"]["rest_off"], dim=-1)[None].expand(B, -1, -1)
    full = geodesic_deg(pr, gt)                          # [B,J]
    # swing = bone-direction error (position-observable); residual ~ twist
    vp = (pr @ a0.unsqueeze(-1)).squeeze(-1); vg = (gt @ a0.unsqueeze(-1)).squeeze(-1)
    swing = torch.rad2deg(torch.arccos((vp * vg).sum(-1).clamp(-1 + 1e-6, 1 - 1e-6)))
    finger = torch.tensor([any(k in nm for k in FINGER_KW) for nm in d["spec"].joint_names], device=pr.device)
    body = ~finger
    return {"full_rot": round(float(full[:, body].mean()), 1),
            "swing": round(float(swing[:, body].mean()), 1),
            "twist_residual": round(float(full[:, body].mean() - swing[:, body].mean()), 1)}


def evaluate(model, data, names, combos, val_idx):
    """The FIX eval dict: 4-direction rot/pos, swing/twist per embodiment, code alignment.
    code_align_dist generalizes to N embodiments as the mean over unordered pairs
    (identical value to the FIX single-pair number when len(names)==2)."""
    was_training = model.training
    model.eval(); out = {}
    with torch.no_grad():
        for src, tgt in combos:
            rot, pos = run_dir(model, data, src, tgt, val_idx)
            # full names: [:4] truncation collided for soma77/soma_scaled/soma_nohand
            out[f"{src}->{tgt}"] = {"rot": round(float(rot), 1), "pos_cm": round(float(pos * data[tgt]["reach"]), 2)}
        for name in names:                              # swing/twist breakdown (self-recon)
            out[f"ST_{name}"] = swing_twist_report(model, data, name, val_idx)
        # code alignment: how close are codes of the SAME motion across embodiments?
        zs = {n: model.encode(data[n]["pose"][val_idx], geom(data, n, len(val_idx))) for n in names}
        dists, scale = [], None
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                (zp_a, zr_a), (zp_b, zr_b) = zs[names[i]], zs[names[j]]
                dists.append(float(((zp_a - zp_b) ** 2).mean() + ((zr_a - zr_b) ** 2).mean()))
        zp0, zr0 = zs[names[0]]
        out["code_align_dist"] = round(sum(dists) / max(len(dists), 1), 4)
        out["code_scale"] = round(float((zp0 ** 2).mean() + (zr0 ** 2).mean()), 4)
        if getattr(model, "judge", None) is not None:    # E14: per-embodiment critic readout
            # judge indices follow TRAIN-time embodiment order, which need not match this
            # cache's names (e.g. evaluating an 18-judge model on the 2-body BONES cache)
            sub = val_idx[:1024]; jr = {}
            n_j = model.judge.net[-1].out_features
            for n in names:
                zp_j, zr_j = model.encode(data[n]["pose"][sub], geom(data, n, len(sub)))
                act = torch.sigmoid(model.judge(zp_j, zr_j)).mean(0)
                top = int(act.argmax())
                self_i = names.index(n) if len(names) == n_j else None
                jr[n] = {"self": round(float(act[self_i]), 3) if self_i is not None else None,
                         "top_idx": top, "top_val": round(float(act[top]), 3)}
            out["judges"] = jr
    if was_training:
        model.train()
    return out


def pos_sigma_weights(data, names, cache, tr_idx, floor, device, max_rows=0):
    """P1 方差归一化 (--var_norm) 的权重表: 每具身一个 [J] 张量。

    sigma_j = 训练行上第 j 关节位置(gp/reach 单位)逐维 std 的模长 —— 和
    amplitude_report 度量"动了多少"用同一把尺子。权重 = 1/sigma_j.clamp_min(floor),
    再做均值归一。为什么均值归一: 不归一的话 1/sigma 平均在 10 倍上下, w_pos=5 的
    位置项整体放大 ≈ 改了学习率, 消融就混进第二个变量; arm_boost/ee_boost 的旧例
    也是 mean-1(见 run_dir), 沿用同一纪律, 只重新分配、不改总量。
    floor: sigma 地板 (--var_norm_floor, 默认 1e-4 = (0.01)^2 量级, reach 单位下
    约 0.01cm), 防死关节(std≈0)把权重打到无穷。

    sigma 只算一次并写盘到数据缓存旁 (caches[0]/v2_sigma_<name>_<N>.npz), 因为
    统计只依赖数据不依赖模型; 行数进文件名, smoke 的 max_rows 截断不会污染全量缓存。
    行子采样用固定步长(≤20 万行), 不碰任何 RNG —— 开关关掉时逐位不变的保证之一。
    """
    caches = [c for c in str(cache).split(",") if c]
    sub = tr_idx[::max(1, len(tr_idx) // 200_000)]
    out = {}
    for name in names:
        N = len(data[name]["pose"])
        f = f"{caches[0]}/v2_sigma_{name}_{N}.npz"
        if os.path.exists(f):
            sig = torch.as_tensor(np.load(f)["sigma"], dtype=torch.float32)
        else:
            ps = data[name]["pose"][sub][..., 6:9]           # [n,J,3] 已是 gp/reach
            sig = ps.std(0).norm(dim=-1).cpu()               # [J]
            np.savez(f, sigma=sig.numpy())
        # 钉死关节例外 (smoke 实测发现): 根的位置按数据约定恒为原点, sigma 精确 = 0,
        # 打到地板后权重 1/1e-4 = 1e4, 均值归一时把其余所有关节压到 ~0 —— 整个位置
        # 损失塌成"学根在原点"。钉死 (sigma < 1e-6) 的关节给中性权重 1, 不参与归一。
        pinned = sig < 1e-6
        w = 1.0 / sig.clamp_min(floor)
        w = w / w[~pinned].mean()
        w[pinned] = 1.0
        out[name] = w.to(device)
        print(f"[var_norm] {name}: sigma [{sig.min():.4f},{sig.max():.4f}] "
              f"pinned={int(pinned.sum())} w [{w.min():.2f},{w.max():.2f}]")
    return out


def slip_pair_starts(cache, N, tr_idx, max_rows=0):
    """P1.5 slip 副批 (--slip_pairs) 的合法相邻行对起点表 -> int64 数组 (行 r, r+1 成对)。

    行到片段的映射来自各缓存的 sequences.json (行区间 [start, start+len));
    多缓存按 train_dual 的拼接顺序累加偏移。为什么必须查表: 相邻行只有在同一
    片段内才是时间上相邻的两帧, 跨片段的 r/r+1 是两段无关动作, 对它们压位移
    等于教模型冻结。另外两行都得落在训练行里(不许偷看 val 行)。
    没有 sequences.json 的老缓存(如 cache_paired)直接报错 —— 宁可拒跑, 不编数据。
    """
    caches = [c for c in str(cache).split(",") if c]
    ok = np.zeros(N, dtype=bool)
    off = 0
    import json as _j
    for c in caches:
        sq = f"{c}/sequences.json"
        assert os.path.exists(sq), f"--slip_pairs 需要 {sq} (行->片段映射); 该缓存没有"
        for s in _j.load(open(sq)):
            a, b = off + s["start"], off + s["start"] + s["len"]
            if a >= N:
                break                                        # max_rows 截断之外
            ok[a:min(b, N) - 1] = True                       # r 与 r+1 同片段
        off += sum(s["len"] for s in _j.load(open(sq)))
    is_tr = np.zeros(N, dtype=bool)
    is_tr[tr_idx.cpu().numpy()] = True
    ok &= is_tr & np.roll(is_tr, -1)                         # r、r+1 都是训练行
    starts = np.flatnonzero(ok)
    assert len(starts) > 0, "--slip_pairs: 没有任何合法相邻行对"
    return starts


# ---- v2 P0 探针 C (0831): 逐组梯度份额 ----
GRP_ARM_KW = ("shoulder", "elbow", "wrist", "arm", "hand", "clavicle", "collar",
              "scap", "bicep")
GRP_LEG_KW = ("hip", "knee", "ankle", "foot", "leg", "toe", "shank", "thigh")


def group_masks(spec, device):
    """[J] bool x3: arm / leg / torso(其余)。关键词表 = amplitude_report 用的同一套,
    不另立口径。"""
    nl = [n.lower() for n in spec.joint_names]
    arm = torch.tensor([any(k in n for k in GRP_ARM_KW) for n in nl], device=device)
    leg = torch.tensor([any(k in n for k in GRP_LEG_KW) for n in nl], device=device)
    leg = leg & ~arm
    return arm, leg, ~(arm | leg)


def grad_share(model, data, src, tgt, idx, params, w_pos, fk_loss=False, dof_loss=False):
    """一次独立前向 (自己的图, 不碰训练的 .grad), 把位置项和旋转项按 arm/leg/torso
    拆开分别求梯度, 返回各组的全局梯度范数 + 份额。

    份额的定义要和真实损失一致: 总损失里的位置项是 w_pos * pos_j.mean() (对 B*J 求
    平均), 所以第 g 组的贡献 = w_pos * pos_j[:, g].sum() / (B*J) —— 分母是全体关节数,
    不是组内关节数。这样三组相加 = 完整位置项, 份额才有意义。
    """
    B = len(idx)
    zp, zr = model.encode(data[src]["pose"][idx], geom(data, src, B))
    prot, ppos = model.decode(zp, zr, geom(data, tgt, B))
    gt = data[tgt]["pose"][idx]
    Rp, Rt = rot6d_to_matrix(prot), rot6d_to_matrix(gt[..., :6])
    if dof_loss and "kin" in data[tgt]:
        from dof_head import project_global6d
        kin = data[tgt]["kin"]
        Rp = project_global6d(Rp, kin); Rt = project_global6d(Rt, kin).detach()
    rot_j = geodesic_deg(Rp, Rt)
    if fk_loss:
        spec, reach = data[tgt]["spec"], data[tgt]["reach"]
        pos_j = (fk_pos_torch(prot, spec) / reach - fk_pos_torch(gt[..., :6], spec) / reach).norm(dim=-1)
    else:
        pos_j = (ppos - gt[..., 6:9]).norm(dim=-1)
    arm, leg, tor = group_masks(data[tgt]["spec"], rot_j.device)
    denom = rot_j.numel()
    out = {}
    for gname, m in (("arm", arm), ("leg", leg), ("torso", tor)):
        out[f"n_{gname}"] = int(m.sum())
        for tname, term, scale in (("pos", pos_j, w_pos), ("rot", rot_j, 1.0 / 30.0)):
            L = scale * term[:, m].sum() / denom
            g = torch.autograd.grad(L, params, retain_graph=True, allow_unused=True)
            n = torch.sqrt(sum((x ** 2).sum() for x in g if x is not None))
            out[f"{tname}_{gname}_gnorm"] = float(n)
            out[f"{tname}_{gname}_loss"] = float(L)
    for tname in ("pos", "rot"):
        tot = sum(out[f"{tname}_{g}_gnorm"] for g in ("arm", "leg", "torso")) + 1e-12
        for g in ("arm", "leg", "torso"):
            out[f"{tname}_{g}_share"] = round(out[f"{tname}_{g}_gnorm"] / tot, 4)
    return out


def ee_fk_weights(spec, device, w=1.0):
    """P2 (0831, --ee_fk_w): 末端-杠杆位置权重, 全部从骨架树自动推, 不用关键词表。

    为什么要有它 (探针 B 的实测结论): 码里有手的信息 (g1 手部 探针 2.85cm vs 解码器
    7.32cm), 病在解码器交付, 不在瓶颈。而 pos_j 是 L2 范数再对关节取平均 —— 对位置的
    梯度是单位向量除以 B*J, 每个关节拿到的位置梯度完全相等。相等的 cm 梯度 = 不相等
    的角度精度要求: 半径 48cm 的手要达到和半径 10cm 的髋一样的 cm 误差, 需要细得多的
    角精度, 却只买到同样多的梯度。这个权重把预算按杠杆臂重新分配。

    构造 (只用 parents + rest_offsets, 无关键词):
      1. r_j = 静止姿势下根到 j 的累计骨长 (cm) = 杠杆臂
      2. 链长归一化 rho_j = r_j / max(r): 权重表变成无量纲的, 大机器人和小机器人拿到
         同一张表 —— 多具身同批训练时这一步是必须的, 否则 reach 大的具身整体吃更多梯度
      3. 树叶 (parents 里没出现过的关节 = 末端, 拓扑判定) 再乘 2: 末端是链上误差的
         累积点, 也是渲染里最显眼的地方
      4. 均值归一并按 w 插值: w_j = 1 + w * (raw_j/mean(raw) - 1)
         w=0 -> 全 1 (调用方在 w<=0 时根本不乘, 逐位一致); w=1 -> raw/mean(raw)
    """
    par = list(spec.parents)
    off = np.asarray(spec.rest_offsets, dtype=np.float64)
    J = spec.J
    r = np.zeros(J)
    for j in sorted(range(J), key=lambda x: _depth(par, x)):
        p = par[j]
        r[j] = 0.0 if p < 0 else r[p] + float(np.linalg.norm(off[j]))
    rho = r / max(r.max(), 1e-9)
    is_parent = np.zeros(J, bool)
    for p in par:
        if p >= 0:
            is_parent[p] = True
    raw = rho * (1.0 + (~is_parent).astype(np.float64))
    m = raw.mean()
    if m <= 0:
        return torch.ones(J, device=device)
    ww = 1.0 + w * (raw / m - 1.0)
    return torch.as_tensor(np.clip(ww, 0.0, None), dtype=torch.float32, device=device)



# ================= Greenwich-v2 接触层 (0831, spec §7/§8) =================
def contact_terms(cx, data, tgt, idx, prot, clog=None, fk_cm=None):
    """就地往 cx 里填 TARGET 具身上的裸接触量 (不带权重, 权重在 train_dual 侧乘)。

    三项的形状按 spec §7 的三条纠正来:
      pen     relu(-h)^2 对**所有**关节 —— 不等式, 不是回归。§8.1: 源数据自己就穿模
              2.5-4.3cm, 回归会把脏数据学进去, 不等式只罚"穿到地下"这一侧, 所以训出来
              的模型可以比训练数据更干净。这是接触损失唯一能超过数据质量的机制,
              也因此它**永远开着**, 不管有没有接触标签。
              归一化用全体关节均值 (不是逐帧接触数): 它是全身普适律, 不是接触事件。
      cmaint  |h| 只在有标签的关节上, **逐帧按当帧接触数归一**。为什么必须逐帧归一:
              躺着的帧有 6 个接触点、站着的帧只有 2 个, 不归一的话一帧躺姿的权重是
              一帧站姿的三倍, 损失会被地面动作里"接触多"的那几帧牵着走。
      cbce    解码器接触头对软标签的 BCE, 全关节。**永远开着**的理由 (lloyd):
              推理时没有输入接触标签, 模型得自己预测 —— 跟"稀疏关节推全身"是同一范式。

    置信掩码 (§8.3): 权重取 soft * hard, hard 是标注器过完滞回+去毛刺的保守判定。
    模糊帧 (soft 中间值但 hard=False) 权重为 0 = 不给接触监督, 宁可少教不可教错。
    """
    d = data[tgt]
    if not d.get("con_ok"):
        return                                   # 缺标注 -> 该具身所有接触项静默失效
    p = fk_cm if fk_cm is not None else fk_pos_torch(prot, d["spec"])
    h = p[..., CONTACT_UP] / 100.0 + d["con"].gap_root(idx)[:, None]      # [B,J] 米
    # 0831 lloyd: 第三态。标注器建不出地面的帧 (trust=False) 一律不进任何接触损失 ——
    # 它们**不是**"没有接触", 当 0 监督等于主动教错。三项全部屏蔽, 不只是 cmaint:
    # pen 也要屏蔽, 因为 trust=False 往往正是地面高度本身不可信的那些片段, 罚"穿地"
    # 会以一个错误的地面为基准。
    tr = d["con"].trust(idx)                                              # [B] bool
    cx["ctrust"] = float(tr.float().mean())
    if not bool(tr.any()):
        return                                   # 整批都不可信 -> 本步无接触监督
    # ---- 0901 尺子修正: 脚部关节的高度按真实脚底算, 与标签同一口径。
    # sole = joint + R_global @ off_local; prot 就是全局 rot6d, 偏移是常量, 全程可微。
    if d["con"].sole_j is not None:
        js = d["con"].sole_j
        Rj = rot6d_to_matrix(prot[:, js])                              # [B,k,3,3]
        dh = torch.einsum("bkxy,ky->bkx", Rj, d["con"].sole_off)[..., CONTACT_UP]
        h = h.clone()
        h[:, js] = h[:, js] + dh                                       # 米
    hT = h[tr]
    cx["pen"] = (torch.relu(-hT) ** 2).mean()
    soft = d["con"].soft(idx)[tr]
    w = soft * d["con"].hard(idx)[tr].float()
    den = w.sum(1)
    act = den > 1e-6
    cx["cframe"] = float(act.float().mean())
    if bool(act.any()):
        cx["cmaint"] = ((w * hT.abs()).sum(1)[act] / den[act]).mean()
    # ---- 0901 lloyd: 接触不再独立预测。接触输出 = 解码几何的可微读数
    #   c_hat = sigmoid((thr - h_sole)/tau)
    # 对标签软值做 BCE: "该踩你却悬空" -> 梯度往下压; "没踩你却穿了" -> 往上抬。
    # 平滑双向, 一项顶掉 cmaint(单向硬拽) + 独立 hcon 头(会和 6rot 打架)。
    cx["cgeo"] = torch.nn.functional.binary_cross_entropy_with_logits(
        (d["con"].thr - hT) / d["con"].tau, soft)
    if clog is not None:
        cx["cbce"] = torch.nn.functional.binary_cross_entropy_with_logits(clog[tr], soft)


def load_contact(cache, data, names, N, device, max_rows=0):
    """把 <emb>_contact.npz 挂进 data (就地)。缺文件 = con_ok False = 全部接触项失效。

    拼接口径与 load_paired_data 逐字对齐: 先按 cache 列表顺序 concat, 再 max_rows 截断。
    对不上就直接断言失败 —— 行号错位会让"某一帧的接触"配到另一段动作上, 这种错在
    数字上完全看不出来, 只能拒跑。
    """
    import json
    caches = [c for c in str(cache).split(",") if c]
    tot = 0
    for name in names:
        fs = [f"{c}/{name}_contact.npz" for c in caches]
        if not all(os.path.exists(f) for f in fs):
            data[name]["con_ok"] = False
            print(f"[contact] {name}: 缺 {os.path.basename(fs[0])} -> 接触通道喂零, "
                  f"该具身 pen/cmaint/slip_c/cbce 全部失效", flush=True)
            continue
        spec = data[name]["spec"]
        root = int(np.where(np.asarray(spec.parents) < 0)[0][0])
        S, H, G, T, SD, hdr = [], [], [], [], [], None
        for f in fs:
            z = np.load(f)
            hdr = json.loads(str(z["header"]))
            assert int(hdr["J"]) == spec.J, f"{f}: J={hdr['J']} 与 spec 的 {spec.J} 不符"
            # 0903: h0 健全性闸逐文件检查 (原来只看最后一个缓存的 header, 拼多缓存时
            # 前面缓存的坏地面会漏过去)。
            _h0f = float(hdr.get("h0_m", 0.0))
            assert abs(_h0f) < 0.30, (
                f"{name}: h0={_h0f*100:.1f}cm 远超合理范围 (其余具身 0.3-14cm)。"
                f"多半是该具身在这个缓存里的重定向坏了 —— 先修缓存或把它剔出语料, "
                f"不要带着这个地面训练。缓存: {f}")
            # 训练通道 (lloyd 0831 定): contact | (gap<0)。
            #   contact      = 高度门 AND 速度门。实测单脚占空比 34/34 组合全部满足
            #                  腾空相物理上限 (<0.50), 而只用高度门的 contact_low 有
            #                  33/34 违反 —— 它的迟滞离开线 1.5*thr 比开合跳的实际
            #                  腾空高度还高, 把腾空相整段吃掉。
            #   gap<0        = 已经穿到地面以下。速度门会把这些帧否掉 (穿模时脚往往
            #                  还在滑), 但穿地的关节一定是接触的, 所以并集兜底。
            # 并集必须**软硬一起做**: 所有损失项的权重是 soft*hard, 只并 hard 的话
            # 新纳入的穿地帧 soft 中位数只有 0.00-0.02, 权重≈0, 等于没并 (kuavo_s45
            # 上 hard 多了 34.2%, 其中 86% 的 soft 低于 0.1)。更糟的是 cask 会把这些
            # 关节的输入通道置 1.0, 而 cbce 同时被训去预测 soft≈0 —— 自相矛盾。
            pen_ = z["gap"] < 0
            S.append(np.maximum(z["contact_soft"], pen_.astype(z["contact_soft"].dtype)))
            H.append(z["contact"] | pen_)
            _thr = float(hdr["thr_m"]); _D = float(hdr["D_used_m"])
            _s = (_thr - z["gap"].astype(np.float32))
            SD.append((np.sign(_s) * np.log1p(np.abs(_s) / _thr)
                       / np.log1p(_D / _thr)).astype(np.float16))
            G.append(z["gap"][:, root].astype(np.float32))
            T.append(z["trust"] if "trust" in z.files
                     else np.ones(z["contact"].shape[0], bool))
            del z
        soft = np.concatenate(S) if len(S) > 1 else S[0]
        hard = np.concatenate(H) if len(H) > 1 else H[0]
        gr = np.concatenate(G) if len(G) > 1 else G[0]
        tru = np.concatenate(T) if len(T) > 1 else T[0]
        sd = np.concatenate(SD) if len(SD) > 1 else SD[0]
        if max_rows:
            soft, hard, gr, tru, sd = (soft[:max_rows], hard[:max_rows], gr[:max_rows],
                                       tru[:max_rows], sd[:max_rows])
        assert soft.shape[0] == N, f"{name}: 接触标注 {soft.shape[0]} 行 vs 姿态 {N} 行"
        # 0831 健全性闸: engineai_pm01 在 AMASS 缓存里骨盆中位高只有 0.036m (别的具身
        # 0.77-0.91m), 98.8% 的片段整个人埋在地下 72cm, h0 被标成 -0.711m。这是缓存
        # 本身的缺陷, 不是这轮改出来的, 但训练侧此前没有任何检查会拦它 —— 全量 xwbc
        # 一进来就会拿一个错到离谱的地面去算 pen/cmaint。越界即拒跑, 不静默降级。
        _h0 = float(hdr.get("h0_m", 0.0))
        assert abs(_h0) < 0.30, (
            f"{name}: h0={_h0*100:.1f}cm 远超合理范围 (其余具身 0.3-14cm)。"
            f"多半是该具身在这个缓存里的重定向坏了 —— 先修缓存或把它剔出语料, "
            f"不要带着这个地面训练。缓存: {fs}")
        _gm = float(np.median(gr))
        assert -0.30 < _gm < 2.0, (
            f"{name}: 根关节离地中位 {_gm*100:.1f}cm 不合理 (人形应在 40-120cm)。"
            f"该具身的重定向把整个人放错了高度。缓存: {fs}")
        packed = np.packbits(hard, axis=1)                       # [N,ceil(J/8)] uint8
        from contact_label import sole_offsets as _sole
        _map = _sole(name, spec)
        _sj = (torch.as_tensor(sorted(_map), dtype=torch.long, device=device)
               if _map else None)
        _so = (torch.stack([torch.as_tensor(_map[int(j)], dtype=torch.float32)
                            for j in sorted(_map)]).to(device) if _map else None)
        data[name]["con"] = ContactStore(
            torch.from_numpy(np.ascontiguousarray(soft)),
            np.ascontiguousarray(packed),
            torch.from_numpy(gr).to(device), spec.J, device,
            trust=np.ascontiguousarray(tru),
            sd16=torch.from_numpy(np.ascontiguousarray(sd)),
            thr=float(hdr["thr_m"]), sole_j=_sj, sole_off=_so)
        data[name]["con_ok"] = True
        if not tru.all():
            print(f"[contact] {name}: {100*(~tru).mean():.2f}% 帧标为不可信, "
                  f"已从 pen/cmaint/cbce 全部屏蔽", flush=True)
        data[name]["con_groups"] = list(hdr.get("limb_groups") or spec.joint_names)
        data[name]["con_h0_m"] = float(hdr.get("h0_m", 0.0))
        mb = (soft.nbytes + packed.nbytes + gr.nbytes) / 1e6
        tot += mb
        print(f"[contact] {name}: rate={float(hard.mean())*100:.1f}% "
              f"any={float(hard.any(1).mean())*100:.1f}% h0={hdr.get('h0_m', 0)*100:.1f}cm "
              f"thr={hdr.get('thr_m', 0)*100:.1f}cm {mb:.0f}MB", flush=True)
        del soft, hard, gr, packed, S, H, G
    print(f"[contact] 常驻内存合计 {tot/1000:.2f}GB", flush=True)


def contact_balance_weights(data, ref, tr_idx, target=0.10, wcap=50.0, verbose=True):
    """--contact_balance 的逐行采样权重 (对 tr_idx, 和为 1) + 一份统计。

    立论: 生产语料里"地面帧"约 1.2%, bs=256 每批只有 3 帧, 还分散在爬/躺/跪几种构型
    上 —— 这不是损失权重能治的, 是采样问题 (再大的权重也乘不出不存在的样本)。

    分桶用**接触构型签名**: 把 J 个关节按标注器自己的 limb_groups 归并成十几个肢体组
    (脚/膝/髋/手/肘/肩/躯干/头), 每帧的签名 = 哪些组在接触的位掩码。为什么按组不按
    关节: 各家骨架一只脚有 1~3 个关节, 按关节分桶等于给关节多的具身更多桶, 桶就不可比了。

    两级配比:
      1. 先按"是否地面帧"分两池。地面帧 = 有**非脚**的组在接触 (手/膝/躯干/头着地),
         正是判官那份卷子里一帧都没有的那类。
      2. 池内按签名频率的倒数配权 (均值归一 -> 截到 wcap -> 归一), 让爬/躺/跪各自
         都抽得到, 而不是全被最常见的那种构型吃掉。
      3. 两池按 target 配总质量 (默认地面帧占 10%)。
    截断 wcap 是防"某个签名只出现 3 帧"把整批变成那 3 帧的复读。
    """
    d = data[ref]
    assert d.get("con_ok"), f"--contact_balance 需要 {ref} 的接触标注, 但它没加载上"
    groups = d["con_groups"]
    uniq = sorted(set(groups))
    assert len(uniq) < 62, f"肢体组 {len(uniq)} 个, 位掩码放不下"
    gid = np.array([uniq.index(g) for g in groups])
    is_foot = np.array([u.endswith("foot") for u in uniq])
    rows = np.asarray(tr_idx.cpu())
    n = len(rows)
    sig = np.zeros(n, np.int64); nonfoot = np.zeros(n, bool)
    bit = (1 << np.arange(len(uniq), dtype=np.int64))
    for i in range(0, n, 200_000):
        r = rows[i:i + 200_000]
        u = np.unpackbits(d["con"].hp[r], axis=1, count=len(groups)).astype(bool)
        gm = np.stack([u[:, gid == k].any(1) for k in range(len(uniq))], 1)
        sig[i:i + len(r)] = (gm * bit).sum(1)
        nonfoot[i:i + len(r)] = gm[:, ~is_foot].any(1)
    uq, inv, cnt = np.unique(sig, return_inverse=True, return_counts=True)
    raw = 1.0 / cnt[inv].astype(np.float64)
    w = np.zeros(n, np.float64)
    for m in (nonfoot, ~nonfoot):
        if not m.any():
            continue
        r = raw[m]; r = r / r.mean(); r = np.minimum(r, wcap); w[m] = r / r.sum()
    tg = float(target)
    if not nonfoot.any():
        tg = 0.0
    elif not (~nonfoot).any():
        tg = 1.0
    w[nonfoot] *= tg; w[~nonfoot] *= (1.0 - tg)
    w = w / w.sum()
    st = {"ref": ref, "n_sig": int(len(uq)), "ground_rate_data": round(float(nonfoot.mean()), 5),
          "ground_rate_sampled": round(float(w[nonfoot].sum()), 4), "target": tg,
          "wcap": wcap, "max_row_w_x": round(float(w.max() * n), 1),
          "groups": uniq}
    if verbose:
        print(f"[contact_balance] ref={ref} 签名 {st['n_sig']} 种; 地面帧 "
              f"{st['ground_rate_data']*100:.2f}% -> 采样 {st['ground_rate_sampled']*100:.1f}% "
              f"(target {tg*100:.0f}%); 单行最大权重 {st['max_row_w_x']:.0f}x 均值", flush=True)
    return w, st
