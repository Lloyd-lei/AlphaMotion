"""手/上肢诊断探针的公共加载层 (0831)。

只做一件事: 用和 v2_judge 完全一样的口径 (reach = 训练缓存全量行 max, 走
v2_judge.cache_reach 的落盘表) 把 ckpt + 任意行号变成 pose9, 再喂 encode/decode。
不新建任何度量口径 —— 上肢/腿关节掩码直接复用 v2_judge.kw_mask + UL_KW/HIP_KW/KNEE_KW,
免得"换了把尺子"这种伪发现。
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import CACHE_ROOT, SkeletonSpec, fk_global, matrix_to_rot6d   # noqa: E402
from train_ae import static_cond                                          # noqa: E402
from dual_common import build_model, geom, fk_pos_torch                    # noqa: E402
import v2_judge as VJ                                                     # noqa: E402

RUNS = f"{CACHE_ROOT}/runs"
LAFAN = f"{CACHE_ROOT}/cache_xwbc_lafan"
SRC = "human_smpl"
TGTS = ["unitree_g1_29dof", "fourier_gr3"]

# human_smpl (22 关节) 的三分组 —— 名字表在缓存里是固定的, 直接写死并在加载时断言。
H_TORSO = ["Pelvis", "Torso", "Spine", "Chest", "Neck", "Head"]
H_LEG = ["L_Hip", "R_Hip", "L_Knee", "R_Knee", "L_Ankle", "R_Ankle", "L_Toe", "R_Toe"]
H_ARM = ["L_Thorax", "R_Thorax", "L_Shoulder", "R_Shoulder",
         "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist"]


class Body:
    """一个具身的懒加载视图: 行号 -> pose9 (与训练逐位同口径)。"""

    def __init__(self, name, caches, lut, device):
        d = np.load(f"{caches[0]}/{name}_spec.npz", allow_pickle=True)
        self.name = name
        self.spec = SkeletonSpec(str(d["name"]), d["parents"], d["rest_offsets"],
                                 list(d["joint_names"]))
        self.reach = max(VJ.cache_reach(c, name, self.spec, device) for c in caches)
        self.L = torch.load(f"{LAFAN}/{name}_paired_local6d.pt", mmap=True)
        self.device = device
        from semantics import embed_matrix
        self.st = static_cond(self.spec, device)
        self.sem = embed_matrix(self.spec.joint_names, lut).to(device)
        self.dof = None
        dof_path = f"{caches[0]}/{name}_dof.npz"
        if os.path.exists(dof_path):
            self.dof = torch.as_tensor(np.load(dof_path)["dof"], dtype=torch.float32,
                                       device=device)
            from dof_head import build_kin
            self.kin = build_kin(caches[0], name, self.spec, device=device)

    def pose9(self, rows, bs=4096):
        out = []
        rows = torch.as_tensor(rows, dtype=torch.long)
        with torch.no_grad():
            for i in range(0, len(rows), bs):
                l6 = self.L[rows[i:i + bs]].to(self.device).float()
                gR, gp = fk_global(l6, self.spec)
                out.append(torch.cat([matrix_to_rot6d(gR), gp / self.reach], -1))
        return torch.cat(out)

    def fkpos_cm(self, rows, bs=4096):
        """GT 全局 FK 位置 (cm, 根在原点) —— 和 v2_judge 的 b 项同一个量。"""
        p9 = self.pose9(rows, bs)
        return fk_pos_torch(p9[..., :6], self.spec)

    def geom(self, B):
        g = dict(rest_off=self.st["rest_off"][None].expand(B, -1, -1),
                 bonelen=self.st["bonelen"][None].expand(B, -1),
                 depth=self.st["depth"][None].expand(B, -1),
                 hop=self.st["hop"][None].expand(B, -1, -1),
                 sem=self.sem[None].expand(B, -1, -1))
        g["spec"] = self.spec
        g["reach"] = self.reach
        if self.dof is not None:
            g["dof"] = self.dof[None].expand(B, -1, -1)
            g["kin"] = self.kin
        return g


def load_ckpt(ckpt, device="cuda", bodies=None):
    ck = ckpt if os.path.isdir(ckpt) else f"{RUNS}/{ckpt}"
    cfg = json.load(open(f"{ck}/config.json"))
    caches = [c for c in str(cfg.get("cache", LAFAN)).split(",") if c]
    assert "cache_xwbc_lafan" in os.path.basename(caches[0].rstrip("/")), caches[0]
    model = build_model(cfg, device)
    model.load_state_dict(torch.load(f"{ck}/model.pt", map_location=device))
    model.eval()
    names = bodies or ([SRC] + TGTS)
    from semantics import build_name_embeddings
    names_all = []
    for n in names:
        d = np.load(f"{caches[0]}/{n}_spec.npz", allow_pickle=True)
        names_all += [str(x) for x in d["joint_names"]]
    lut = build_name_embeddings(names_all, device=device,
                                encoder=cfg.get("sem_encoder", "siglip"))
    B = {n: Body(n, caches, lut, device) for n in names}
    return ck, cfg, model, B


def clip_of_row(n_rows):
    seqs = json.load(open(f"{LAFAN}/sequences.json"))
    c = np.full(n_rows, -1, dtype=np.int32)
    for i, s in enumerate(seqs):
        c[s["start"]:s["start"] + s["len"]] = i
    return c, seqs


def val_clip_ids():
    """v2_judge 的固定 val 片段 (default_rng(123) 取 10%) —— 探针的留出必须和判官同一批。"""
    seqs = json.load(open(f"{LAFAN}/sequences.json"))
    rng = np.random.default_rng(123)
    val = sorted(rng.permutation(len(seqs))[: len(seqs) // 10].tolist())
    return set(val), seqs


def judge_rows():
    w = json.load(open(VJ.WIN_FILE))
    return torch.cat([torch.arange(s["start"], s["start"] + w["T"]) for s in w["windows"]])


def hgroups(spec):
    nl = list(spec.joint_names)
    idx = lambda names: torch.tensor([nl.index(x) for x in names], dtype=torch.long)
    return idx(H_TORSO), idx(H_LEG), idx(H_ARM)


def leaf_mask_np(spec):
    J = spec.J
    isp = np.zeros(J, bool)
    for p in spec.parents:
        if p >= 0:
            isp[p] = True
    return ~isp


def chain_leaf_mask(spec, kws, device):
    """拓扑法取"手/脚"这一层: 祖先链上出现过 kws 关节的树叶。

    为什么不用关键词直接点名: 判官的 UL_KW 里 "hand" 会先被 FINGER_KW 过滤掉,
    g1 的 left_rubber_hand、gr3 的 *_end_effector_link 其实都没进上肢指标 ——
    也就是说"手"从来没被量过。这里按骨架树自动认末端, 不用每家维护名单。
    """
    nl = [n.lower() for n in spec.joint_names]
    hit = [any(k in n for k in kws) for n in nl]
    par = list(spec.parents)
    leaves = leaf_mask_np(spec)
    out = np.zeros(spec.J, bool)
    for j in range(spec.J):
        if not leaves[j]:
            continue
        k = j
        while k >= 0:
            if hit[k]:
                out[j] = True
                break
            k = par[k]
    return torch.as_tensor(out, device=device)
