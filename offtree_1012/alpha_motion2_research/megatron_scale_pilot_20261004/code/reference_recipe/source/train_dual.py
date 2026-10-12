"""Any-to-any cross-embodiment translation with DUAL FSQ (separate position/rotation
codes). Trained on paired SOMA<->G1, all 4 direction combos, with train/val split.
Reports rot + pos per direction (cross directions = the real translation metric).

Shared data/eval logic lives in dual_common.py (same code path as eval_dual.py).
RNG order is load-bearing for FIX reproducibility: seed(0) -> load -> split -> model.
"""
from __future__ import annotations
import os, sys, json, time, argparse, itertools
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from common import CACHE_ROOT
from dual_common import load_paired_data, split_indices, build_model, geom, run_dir, evaluate


def main():
    # @file argfile support (N3): one token per line, e.g. runs/qwen_recipe.args
    # reproduces E27_qwen_0806's exact flags; later CLI flags override earlier ones.
    ap = argparse.ArgumentParser(fromfile_prefix_chars="@")
    ap.add_argument("--cache", default=f"{CACHE_ROOT}/cache_paired")
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n_latent", type=int, default=32)
    ap.add_argument("--fsq_stages", type=int, default=2)
    ap.add_argument("--fsq_levels", default="8,8,8,5,5,5")   # E0a bit-budget sweep
    ap.add_argument("--single_codebook", action="store_true")  # E0c: one shared FSQ, same total bits
    ap.add_argument("--mask_prob", type=float, default=0.0)  # E0b: per-sample prob of masked encoder input
    ap.add_argument("--mask_mix", default="rand:0.4,ee:0.3,root:0.3")
    ap.add_argument("--visible_boost", type=float, default=0.0)  # upweight visible joints under mask (constraint honoring)
    ap.add_argument("--tail_alpha", type=float, default=0.0)     # 0809 tail cure: per-sample-joint error^alpha reweighting (training only)
    ap.add_argument("--w_pos", type=float, default=5.0)
    ap.add_argument("--w_align", type=float, default=0.0)    # L2 same-motion code pull (collapses if strong)
    ap.add_argument("--w_contrast", type=float, default=0.0) # InfoNCE: same-motion attract, others repel (no collapse)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--self_only", action="store_true")   # only reconstruction (soma->soma, g1->g1)
    ap.add_argument("--holdout", default=None)            # E5: comma-separated embodiments excluded from training; first one gets zero-shot eval
    ap.add_argument("--combo_sample", type=int, default=0)  # E5: combos per step (0 = all; bounds N^2 growth)
    ap.add_argument("--contrast_sample", type=int, default=0)  # InfoNCE embodiments per step (0 = all; bounds memory at large N)
    ap.add_argument("--use_dof", action="store_true")  # feed per-joint rotation axes + limits
    # E27: which frozen text encoder produces the joint-name semantics (same prompt both ways).
    ap.add_argument("--sem_encoder", choices=["siglip", "qwen3"], default="siglip")
    # E25 DOF-masking arms (default = every prior run, bit-for-bit).
    #   --rot_head angles : arm A, decoder emits hinge ANGLES -> FK -> global rotations
    #   --dof_loss        : arm B, free 6D head, loss projected onto the hinge manifold
    ap.add_argument("--rot_head", choices=["free", "angles"], default="free")
    ap.add_argument("--dof_loss", action="store_true")
    ap.add_argument("--judge_mode", choices=["off", "observe", "grad"], default="off")  # E14: sigmoid critics
    # E28: decode encoder-VISIBLE joints as a residual on the observed pose (same-skeleton
    # directions only; cross-body decode is untouched). Default off = every prior run.
    ap.add_argument("--vis_residual", action="store_true")
    ap.add_argument("--w_judge", type=float, default=0.1)
    # N3e (--reenc_consistency W): RE-ENCODE CYCLE CONSISTENCY. The measured failure
    # it exists to fix: encode(decode(z)) explodes even same-body (round-trip error
    # ~8.7 -> ~38.8 deg), which kills any synthetic-data pipeline. With prob
    # --reenc_prob per step: pick one direction, decode, RE-ENCODE the decoded pose
    # (no stop-grad anywhere — the encoder must learn to read decoded input AND the
    # decoder must emit re-encodable poses) and penalize the z-space distance to the
    # original codes with weight W. Runs at HALF batch to bound the double forward.
    # 0 = off = bit-exact prior behavior.
    ap.add_argument("--reenc_consistency", type=float, default=0.0)
    ap.add_argument("--reenc_prob", type=float, default=0.3)
    # N3f (--var_floor W): OUTPUT-VARIANCE FLOOR (VICReg's variance term applied to
    # OUTPUTS): hinge max(0, std_gt - std_decoded) per joint/dim across the batch,
    # std target = the GT batch's own std. Treats amplitude compression (the kick
    # that never lifts). 0 = off = bit-exact prior behavior.
    ap.add_argument("--var_floor", type=float, default=0.0)
    ap.add_argument("--eval_every", type=int, default=1000)
    # 0903 scaling 阶梯 (lloyd): 训练侧此前从不记逐步 loss (各项分别 backward, 没有总
    # 标量), 只有每 eval_every 一条 EVAL —— scaling law 要的是 loss-vs-step 曲线。
    #   --log_every N     每 N 步把各项 loss 的区间均值 + 墙钟 + s/步 写一行 loss_log.jsonl
    #   --exclude_set F   sequences.json: 这些片段的行从训练/验证里挖掉 (语料外考卷不能
    #                     进训练; 在 split_indices 之后过滤, 不消耗 RNG, FIX 分割契约不变)
    #   --probe_steps N   只跑 N 步就退出并报 s/步, 用来算 ETA
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--exclude_set", default=None)
    ap.add_argument("--probe_steps", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)  # training randomness only; the val split stays seed-0
    # E24 parameter axis. Everything before it scaled the BOTTLENECK (n_latent, fsq levels)
    # or the data, never the backbone — every run from K=16 to K=128 sat at 17.95-18.01M params.
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--enc_layers", type=int, default=4)
    ap.add_argument("--dec_layers", type=int, default=4)
    ap.add_argument("--out", default=f"{CACHE_ROOT}/runs/D_dual")
    ap.add_argument("--device", default="cuda")   # "cpu" = smoke test while the GPU queue runs
    ap.add_argument("--max_rows", type=int, default=0)  # 0 = all; smoke tests only
    # decoder post-train (0811): warm-start a finished run and touch ONLY the decode
    # side. The code lattice (enc/pool/FSQ-in, shared jf) stays byte-frozen, so every
    # downstream consumer (BRIDGE A3/B3, text layer) keeps working unchanged.
    ap.add_argument("--init", default=None, help="path to model.pt to warm-start from")
    ap.add_argument("--train_scope", choices=["all", "decoder"], default="all")
    ap.add_argument("--arm_boost", type=float, default=0.0)
    ap.add_argument("--ee_boost", type=float, default=0.0)
    # 0812 experiments (lloyd): the position head cannot reach the render, so
    #   --fk_loss   supervise the ROTATION head with metric error via FK
    #   --rot_only  spend every code slot on rotation instead of half on a
    #               stream whose output is discarded
    ap.add_argument("--fk_loss", action="store_true")
    ap.add_argument("--rot_only", action="store_true")
    # Greenwich-v2 P1 (0830, 默认关): 位置误差按关节自身的数据位移幅度归一
    # (pos_j * 归一化的 1/sigma_j)。为什么: 用数据统计替换 arm/ee_boost 的关键词表,
    # 见 dual_common.pos_sigma_weights 的注释。floor = sigma 地板, 默认 1e-4
    # ((0.01)^2 量级), 防 std≈0 的死关节权重爆炸。
    ap.add_argument("--var_norm", action="store_true")
    ap.add_argument("--var_norm_floor", type=float, default=1e-4)
    # Greenwich-v2 P2 (0831, 默认 0 = 关 = 逐位一致): 末端-杠杆位置权重。树叶由
    # parents 自动认, 权重 = 归一化杠杆臂 x (树叶再乘 2), 均值 1, 全程无关键词表。
    # 立论见 dual_common.ee_fk_weights 和 ../HAND_DIAGNOSIS.md 的探针 B。
    ap.add_argument("--ee_fk_w", type=float, default=0.0)
    # Greenwich-v2 P1.5 (0830, 默认关): slip 副批。主批保持严格单帧; 每步另抽 1/4
    # batch 的"片段内相邻行对", 两帧各自解码, 只对 GT 脚在踩地(低且慢: 水平脚速
    # <0.10 m/s)的样本罚解码脚的水平位移。为什么: 精修器A 实测支撑相 21-44cm/s 的
    # 漂移来自相邻帧解码不一致 —— 码没病, Equator 也够不到 decoder, 只能在这里治。
    # 推理仍是单帧, 副批只在训练时存在。--slip_w 0 (默认) = 逐位一致的老路径。
    ap.add_argument("--slip_pairs", action="store_true")
    ap.add_argument("--slip_w", type=float, default=0.0)
    # 副批大小 = slip_frac * bs 对 (默认 1/4 = 老值)。每步只抽一个 (src,tgt) 组合,
    # 17 具身时每个目标只分到 1/17 的步数, 世界系脚速信号太稀 -> Z 系列用 0.5。
    ap.add_argument("--slip_frac", type=float, default=0.25)
    # ---- Greenwich-v2 P3 接触层 (0831, spec §7/§8; 每个开关默认关, 全关 = 逐位一致) ----
    # 结构开关: 编码器输入 [B,J,9]->[B,J,10] (第 10 通道 = 逐关节软接触标签, 不是一个
    # 全局标志位) + 解码器接触头。缺 <emb>_contact.npz 的具身自动喂零并关掉全部接触项。
    ap.add_argument("--contact_chan", action="store_true")
    # 三项一致性损失 + 一个预测头 (裸量见 dual_common.contact_terms):
    #   w_pen     relu(-h)^2 全关节穿模罚, 米^2 (3cm -> 9e-4, 所以权重要上百量级)
    #   w_contact |h| 该碰必须碰到, 只在有标签处, 逐帧按接触数归一, 米
    #   w_slip_c  接触关节的相邻帧水平位移, (m/s)^2 —— 走 --slip_pairs 那个副批
    #   w_cbce    解码器接触头对软标签的 BCE (推理时没有输入标签, 模型得自己预测)
    ap.add_argument("--w_pen", type=float, default=0.0)
    ap.add_argument("--w_contact", type=float, default=0.0)
    # 0901 lloyd: 接触输出 = 几何读数 (无独立头)。BCE((thr-h_sole)/tau, soft)。
    ap.add_argument("--w_cgeo", type=float, default=0.0)
    # 0901 根高槽位: hroot 读旋转支路根 query -> 根离地高 (米)。目标 = 标签的 gap_root。
    ap.add_argument("--root_head", action="store_true")
    ap.add_argument("--w_root", type=float, default=0.0)
    # 0901 Y3 (lloyd): 位置头砍到只有 root 槽, 位置码本预算独立砍小 (--n_latent_p)。
    # root 槽 = hpos 在根 query 的 3 维输出, 语义: [离地高 m, 航向前速 m/s, 航向侧速 m/s]
    # (不过 reach 归一)。速度积分出位移 -> 位移归模型own, 播放不再抄人体轨迹。
    ap.add_argument("--n_latent_p", type=int, default=0)
    ap.add_argument("--root_pos", action="store_true")
    ap.add_argument("--w_rootpos", type=float, default=0.0)
    ap.add_argument("--w_slip_c", type=float, default=0.0)
    # 0901 Z 系列 (WBC 可训性验收后): **世界系**接触脚速。w_slip_c / slip_w 都是根相对
    # 位移 —— 走路支撑相里踩住的脚相对根必须以 -根速 后退, 罚它等于教滑冰
    # (P3V2_cons_0831 腿部 22° vs 8° 就是这么坏的)。这里把目标 GT 根世界轨迹
    # (root_pos 驻卡的 rw) 的相邻帧位移加回来: v_world = Δfk_sole/100 + Δrw, 水平分量,
    # 只在标注器判两帧都硬接触 (soft*hard, trust 双帧) 的关节上罚 (m/s)^2, 逐帧按接触数归一。
    # 脚部关节用脚底点 (sole_offsets, 与标签同口径), 其它接触关节用关节点。需要 --root_pos。
    ap.add_argument("--w_slip_w", type=float, default=0.0)
    # 0902 根高成对损失 (lloyd: "位置高度一直在跳变"): 在 --slip_pairs 副批的相邻两帧上罚
    #   (Δ解码根高 - ΔGT根高)*30 的均方, 单位 (m/s)^2。模型仍是单帧, 只是训练时看相邻对
    #   (与 w_slip_w 同一套路)。主批的 w_rootpos 只管每帧绝对高度, 管不住逐帧跳变。
    ap.add_argument("--w_hpair", type=float, default=0.0)
    # 0901 ③-lite 时序解码器 (tdec.py): --tdec K>1 时 batch = (bs//K) 个 K 帧窗压平, 解码器
    # 先沿时间融合码再解; 世界系脚速损失直接在主批的窗内相邻帧上算 (不用 --slip_pairs)。
    # 0 = 关 = 逐位一致的老路径。
    ap.add_argument("--tdec", type=int, default=0)
    ap.add_argument("--tdec_layers", type=int, default=2)
    ap.add_argument("--code_noise", type=float, default=0.0, help="训练时量化值加 ±noise 格均匀噪声 (单帧抗抖动解码器)")
    ap.add_argument("--w_cbce", type=float, default=0.0)
    # 0831 lloyd 授权按 A 方案: 第 10 通道的 dropout。不加它的话训练每一步都白送真值
    # 接触标签, 而 run_dir/v2_judge 考试时一律不喂 -> 模型(a)永远不需要自己预测接触,
    # w_cbce 的接触头等于白训; (b)被扔进一个从没见过的配置里考, 通道读到"全身没碰地"。
    # 掉线时通道置零(= "不知道"), 与考试时的输入分布一致。
    ap.add_argument("--con_dropout", type=float, default=0.5,
                    help="训练时以此概率把整行的接触通道置零; 0=永远给(旧行为)")
    # 接触构型均衡采样: 地面帧在生产语料里约 1.2%, bs=256 一批只有 3 帧且分散在
    # 爬/躺/跪几种构型上 —— 损失权重乘不出不存在的样本, 只能在采样上治。
    ap.add_argument("--contact_balance", action="store_true")
    ap.add_argument("--contact_target", type=float, default=0.10)  # 目标地面帧占比
    ap.add_argument("--contact_wcap", type=float, default=50.0)    # 单签名权重上限
    ap.add_argument("--contact_ref", default=None)                 # 签名参照具身, 默认第一个
    # v2 探针 C (0831, 默认 0 = 关 = 逐位一致): 每 N 步单独做一次前向, 把位置项/旋转项
    # 按 arm/leg/torso 拆开测梯度范数份额。用 autograd.grad 在独立的图上算, 不写 .grad、
    # 不进 optimizer, 训练轨迹分毫不变。
    ap.add_argument("--grad_share_every", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = args.device; torch.manual_seed(0)

    data, names, N = load_paired_data(args.cache, device, max_rows=args.max_rows,
                                      sem_encoder=args.sem_encoder)
    val_idx, tr_idx = split_indices(N)          # consumes global RNG -> must stay at seed 0
    torch.manual_seed(args.seed)                # only init/batching/masking vary across seeds
    if args.exclude_set:
        # 语料外集合按片段名匹配各缓存的 sequences.json, 行号加上前置缓存的行偏移
        # (与 load_paired_data / slip_pair_starts 的 concat 口径逐字一致)。
        _ex_names = {q["name"] for q in json.load(open(args.exclude_set))}
        _exm = np.zeros(N, dtype=bool); _off = 0; _hit = 0
        for _c in [c for c in str(args.cache).split(",") if c]:
            _sq = json.load(open(f"{_c}/sequences.json"))
            for q in _sq:
                if q["name"] in _ex_names:
                    a_, b_ = _off + q["start"], _off + q["start"] + q["len"]
                    if a_ < N:
                        _exm[a_:min(b_, N)] = True; _hit += 1
            _off += sum(q["len"] for q in _sq)
        _n0 = len(tr_idx) + len(val_idx)
        tr_idx = tr_idx[~torch.as_tensor(_exm)[tr_idx]]
        val_idx = val_idx[~torch.as_tensor(_exm)[val_idx]]
        print(f"[exclude_set] {len(_ex_names)} 段名单, 命中 {_hit} 段 / {int(_exm.sum())} 行; "
              f"训练+验证行 {_n0} -> {len(tr_idx) + len(val_idx)}", flush=True)
        assert _hit > 0 or args.max_rows, f"--exclude_set {args.exclude_set}: 没有任何片段名命中缓存"

    cfg = vars(args)
    if args.judge_mode != "off":
        cfg["n_judges"] = len(names)                     # judges cover ALL embodiments incl. holdouts
    emb_idx = {n: i for i, n in enumerate(names)}
    model = build_model(cfg, device)
    if args.init:
        _sd = torch.load(args.init, map_location=device)
        _tdec_ck = any(k.startswith("base.") for k in _sd)
        if not _tdec_ck:
            model.load_state_dict(_sd)
        print(f"warm-start <- {args.init}" + (" (tdec ckpt)" if _tdec_ck else ""))
    if args.code_noise > 0:
        for _m in (model.fsqp, getattr(model, "fsqr", None)):
            if _m is not None: _m.noise = args.code_noise
        print(f"[code_noise] ±{args.code_noise} 格", flush=True)
    if args.tdec > 1:
        from tdec import TemporalDecoder, window_starts, slip_world
        model = TemporalDecoder(model, args.tdec, n_layers=args.tdec_layers).to(device)
        if args.init and _tdec_ck:
            model.load_state_dict(_sd)
        assert args.bs % args.tdec == 0, "--bs 必须是 --tdec 的倍数"
        assert not args.slip_pairs and not args.contact_balance, "tdec 与 slip_pairs/contact_balance 不兼容"
        print(f"[tdec] K={args.tdec} 层={args.tdec_layers} 窗/批={args.bs // args.tdec}")
    train_params = list(model.parameters())
    if args.train_scope == "decoder":
        DEC = ("decp.", "decr.", "selfp.", "selfr.", "hrot", "hpos",
               "fsqp.proj_out", "fsqr.proj_out", "tfp.", "tfr.")
        n_tr = 0
        for nm, prm in model.named_parameters():
            on = nm[5:].startswith(DEC) if nm.startswith("base.") else nm.startswith(DEC)
            prm.requires_grad_(on)
            n_tr += prm.numel() if on else 0
        train_params = [prm for prm in model.parameters() if prm.requires_grad]
        print(f"train_scope=decoder: {n_tr/1e6:.2f}M trainable, lattice frozen")
    print(f"params {sum(p.numel() for p in model.parameters())/1e6:.2f}M dual-FSQ bits/frame={model.bits:.0f}")
    opt = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    holdouts = [h for h in (args.holdout or "").split(",") if h]
    for h in holdouts:
        assert h in names, f"holdout {h} not in {names}"
    train_names = [n for n in names if n not in holdouts]
    combos = [(n, n) for n in train_names] if args.self_only else list(itertools.product(train_names, train_names))
    if len(combos) <= 20:
        eval_combos = list(combos)
    else:
        # big-N caches: full product eval (169 directions at n=13) costs more than training.
        # Eval = all self-recon + soma-source cross translations; training combos unaffected.
        eval_combos = [(n, n) for n in train_names] + [(train_names[0], t) for t in train_names[1:]]
    if holdouts:                                         # zero-shot: self-recon + translation into first holdout
        eval_combos += [(holdouts[0], holdouts[0]), (train_names[0], holdouts[0])]
    combo_rng = np.random.default_rng(4)                 # own stream: torch RNG order untouched
    reenc_rng = np.random.default_rng(args.seed * 1000 + 77)   # N3e's own stream
    # v2 P1/P1.5 准备 (都在 split 之后、且不碰 torch RNG —— 开关全关时训练逐位不变)
    pos_w = None
    if args.var_norm:
        from dual_common import pos_sigma_weights
        pos_w = pos_sigma_weights(data, names, args.cache, tr_idx,
                                  args.var_norm_floor, device, max_rows=args.max_rows)
    # v2 P3: 接触标注加载 (纯 numpy, 不碰 torch RNG; 所有接触开关都关时根本不进这里)
    contact_on = bool(args.contact_chan or args.contact_balance or args.w_pen > 0 or args.w_cgeo > 0
                      or args.w_contact > 0 or args.w_slip_c > 0 or args.w_cbce > 0
                      or args.w_slip_w > 0)
    assert not (args.w_slip_w > 0 and not (args.root_pos and (args.slip_pairs or args.tdec > 1))), \
        "--w_slip_w 需要 --root_pos (GT 根世界轨迹) 和 --slip_pairs 或 --tdec (相邻帧)"
    closs_on = bool(args.w_pen > 0 or args.w_contact > 0 or args.w_cbce > 0 or args.w_cgeo > 0)
    bal_cum = bal_rng = None
    rp_tgt = {}
    if args.root_pos:
        # 航向系速度: 世界速度旋进根 yaw 系。逐具身一次性算好驻卡。
        from common import rot6d_to_matrix as _r2m
        _h0_tab = []
        for _n in names:
            # 0903: 多缓存按列表顺序 concat (原来只读第一个缓存再 [:N] —— 拼 lafan+amass
            # 时越界/错位); 每个 (缓存, 具身) 的根高减去该缓存接触标签量出的 h0, 让
            # "地面 = 0" 在所有语料里成立 (ArenaLabs gauge; 实测修正, 不是手调常数,
            # 下面整表打印)。没有标签的具身不动, h0 记 None。
            _parts = []
            for _c in [c for c in str(args.cache).split(",") if c]:
                _r = torch.load(f"{_c}/{_n}_root_world.pt", mmap=True).float() / 100.0
                _cf = f"{_c}/{_n}_contact.npz"
                _h0 = None
                if os.path.exists(_cf):
                    _h0 = float(json.loads(str(np.load(_cf)["header"])).get("h0_m", 0.0))
                    _r = _r.clone(); _r[:, 1] -= _h0
                _h0_tab.append((os.path.basename(_c.rstrip("/")), _n, _h0))
                _parts.append(_r)
            _rw = (torch.cat(_parts) if len(_parts) > 1 else _parts[0])[:N].to(device)
            assert _rw.shape[0] == N, f"{_n}: root_world 行数 {_rw.shape[0]} != {N}"
            # 0901 v2 (lloyd): 不用速度积分 —— 积分漂移, 开合跳走成散步。
            # root 槽 = [离地高, 相对源根轨迹的偏移] : 每帧独立可算, 误差有界不累积。
            # 偏移在目标 GT 根 yaw 系里表达 (播放时用解码 yaw 转回世界)。
            _R = _r2m(data[_n]["pose"][:, 0, :6])
            _yaw = torch.atan2(_R[:, 0, 2], _R[:, 2, 2])
            rp_tgt[_n] = {"rw": _rw, "yaw": _yaw}
        print(f"[root_pos] {len(rp_tgt)} 具身的 root 槽监督已驻卡", flush=True)
        print("[root_pos] 地面归零 h0 表 (缓存, 具身, h0 cm):")
        for _c, _n, _h0 in _h0_tab:
            print(f"    {_c:28s} {_n:20s} {'n/a' if _h0 is None else f'{_h0*100:+.2f}'}")
        json.dump([{"cache": c_, "emb": n_, "h0_m": h_} for c_, n_, h_ in _h0_tab],
                  open(f"{args.out}/h0_table.json", "w"), indent=1)
    if contact_on:
        from dual_common import load_contact
        load_contact(args.cache, data, names, N, device, max_rows=args.max_rows)
        if args.contact_balance:
            from dual_common import contact_balance_weights
            ref = args.contact_ref or names[0]
            bw, bstat = contact_balance_weights(data, ref, tr_idx, args.contact_target,
                                                args.contact_wcap)
            bal_cum = np.cumsum(bw); bal_cum[-1] = 1.0
            bal_rng = np.random.default_rng(args.seed * 1000 + 91)   # 自有流, 不动 torch RNG
            json.dump(bstat, open(f"{args.out}/contact_balance.json", "w"), indent=1)
    slip_starts = None
    if args.slip_pairs:
        from dual_common import (slip_pair_starts, fk_pos_torch,
                                 CONTACT_HORIZ, CONTACT_FPS)
        slip_starts = slip_pair_starts(args.cache, N, tr_idx, max_rows=args.max_rows)
        slip_rng = np.random.default_rng(args.seed * 1000 + 55)   # 自有流, 不动 torch RNG
        slip_log = []; slipc_log = []; slipw_log = []; hpair_log = []
        print(f"[slip_pairs] {len(slip_starts)} 个合法相邻行对起点")
    print("combos:", combos, "eval extra:", eval_combos[len(combos):])
    win_starts = None
    if args.tdec > 1:
        win_starts = window_starts(args.cache, N, tr_idx, args.tdec, max_rows=args.max_rows)
        win_rng = np.random.default_rng(args.seed * 1000 + 77)
        # 评测窗: 以 val 行为窗首 (其余 K-1 行多半是训练行 —— 不是纯 held-out; 真判官是
        # wbc_ready_audit 的语料外 AMASS)。取 max_val//K 个窗, 行数与老 val 同量级。
        _vs = window_starts(args.cache, N, torch.arange(N), args.tdec, max_rows=args.max_rows)
        _vok = np.zeros(N, bool); _vok[_vs] = True
        _vsel = val_idx[torch.as_tensor(_vok[val_idx.cpu().numpy()])][: max(16, len(val_idx) // args.tdec)]
        val_idx = (_vsel[:, None] + torch.arange(args.tdec)[None]).reshape(-1)
        slipw_log = []
        print(f"[tdec] {len(win_starts)} 个合法训练窗, 评测 {len(_vsel)} 窗")

    mask_mix = None
    mask_need_con = False
    if args.mask_prob > 0:
        from masking import parse_mix, sample_joint_mask
        mask_mix = parse_mix(args.mask_mix)
        mask_need_con = any(f in mask_mix for f in ("cgive", "cask"))
        if mask_need_con and not contact_on:
            raise SystemExit("--mask_mix 里有 cgive/cask, 但没开任何接触开关 "
                             "(至少要 --contact_chan) —— 接触族没有标签可用")

    log = []; gs_log = []; t0 = time.time()
    clog_acc = {k: [] for k in ("pen", "cmaint", "cbce", "cframe", "ctrust", "cgeo", "rooth", "rootpos")}
    # 0903 逐步 loss 记录: 各项 backward 前的标量累进 (已算好的数, 零开销), 每 log_every
    # 步写一行区间均值到 loss_log.jsonl。"total" = 主批各项按各自权重加起来的和 (仅记录用,
    # 训练仍是分项 backward, 配方不变)。
    _sl = {}; _t_log = time.time()
    def _acc(k, v):
        if v is None: return
        _sl.setdefault(k, []).append(float(v.detach()) if torch.is_tensor(v) else float(v))
    _lf = open(f"{args.out}/loss_log.jsonl", "a")
    for step in range(args.steps):
        opt.zero_grad()
        if bal_cum is not None:
            # 接触构型均衡: 用自己的 numpy 流 + 累积分布 (searchsorted, O(bs log N)),
            # 不碰 torch RNG, 所以掩码/初始化那几路随机流与非均衡运行保持同源。
            sel = tr_idx[torch.as_tensor(np.searchsorted(bal_cum, bal_rng.random(args.bs)))]
        elif win_starts is not None:
            _w0 = torch.as_tensor(win_starts[win_rng.integers(len(win_starts), size=args.bs // args.tdec)])
            sel = (_w0[:, None] + torch.arange(args.tdec)[None]).reshape(-1)
            # torch.randint 照旧消费一次 RNG, 让掩码/其它随机流与非 tdec 运行同源
            torch.randint(0, len(tr_idx), (args.bs,))
        else:
            sel = tr_idx[torch.randint(0, len(tr_idx), (args.bs,))]
        step_combos = combos
        if args.combo_sample and args.combo_sample < len(combos):
            pick = combo_rng.choice(len(combos), args.combo_sample, replace=False)
            step_combos = [combos[i] for i in pick]
        for src, tgt in step_combos:
            jm = None; co = None
            if mask_mix is not None:
                # per-sample: with prob mask_prob the encoder sees a partial pose; loss stays full-body
                if mask_need_con:
                    ok_s = bool(data[src].get("con_ok"))
                    J_s = data[src]["spec"].J
                    csoft = (data[src]["con"].soft(sel) if ok_s
                             else torch.zeros(args.bs, J_s, device=device))
                    cbin = (data[src]["con"].hard(sel) if ok_s
                            else torch.zeros(args.bs, J_s, dtype=torch.bool, device=device))
                    jm, co = sample_joint_mask(data[src]["spec"], args.bs, mask_mix,
                                               device=device, contact=csoft,
                                               contact_bin=cbin, return_con=True)
                else:
                    jm = sample_joint_mask(data[src]["spec"], args.bs, mask_mix, device=device)
                gate = (torch.rand(args.bs, 1, device=device) < args.mask_prob)
                jm &= gate
            # ---- 输入通道 = 符号距离 (lloyd 0831 "距离进、置信出") ------------------
            # 通道值来自标签 (几何事实), 与姿态遮没遮无关, 所以 cask 行不再需要任何
            # "置 1"覆盖 —— 被遮关节的"我踩着地"就写在它的 sd>0 里。旧的概率覆盖逻辑
            # (masking.py 的 con_over) 弃用: co 丢弃, 统一喂 sd。
            # 置信度 (soft) 只留在监督端: cbce 的目标、cmaint 的权重。
            _con_in = None
            if args.contact_chan:
                if data[src].get("con_ok"):
                    _con_in = (data[src]["con"].soft(sel) if os.environ.get("AM_OLDCHAN")
                               else data[src]["con"].sdist(sel))
                    # dropout: 模拟"没有接触规格"的推理模式 (通道置于哨兵值 -1 = 远离
                    # 地面处处成立的最保守读数)。cask 行照旧放过 —— 被遮关节只能靠通道。
                    if args.con_dropout > 0:
                        _keep = jm.any(1) if jm is not None else torch.zeros(
                            args.bs, dtype=torch.bool, device=device)
                        _drop = (torch.rand(args.bs, 1, device=device) < args.con_dropout) \
                                & ~_keep[:, None]
                        _con_in = torch.where(_drop, torch.full_like(_con_in, float('nan')), _con_in)
                else:
                    # 该具身没有标签 -> 整批"无信息", 不是"全悬空"
                    _con_in = torch.full((args.bs, data[src]["spec"].J), float("nan"),
                                         device=device)
            ex = {} if args.var_floor > 0 else None                 # N3f
            cex = {} if closs_on else None                          # v2 P3
            rot, pos = run_dir(model, data, src, tgt, sel, joint_mask=jm,
                               visible_boost=args.visible_boost, dof_loss=args.dof_loss,
                               extras=ex, tail_alpha=args.tail_alpha,
                               arm_boost=args.arm_boost, ee_boost=args.ee_boost,
                               fk_loss=args.fk_loss, ee_fk_w=args.ee_fk_w,
                               pos_w=None if pos_w is None else pos_w[tgt],
                               contact=args.contact_chan, con_over=_con_in,
                               cextras=cex, con_bce=args.w_cbce > 0)
            term = rot / 30.0 + args.w_pos * pos
            _acc("rot", rot); _acc("pos", pos)
            if ex is not None:
                term = term + args.var_floor * ex["var_floor"]
                _acc("var_floor", ex["var_floor"])
            if cex:
                if args.w_pen > 0 and "pen" in cex:
                    term = term + args.w_pen * cex["pen"]
                if args.w_contact > 0 and "cmaint" in cex:
                    term = term + args.w_contact * cex["cmaint"]
                if args.w_cgeo > 0 and "cgeo" in cex:
                    term = term + args.w_cgeo * cex["cgeo"]
                pos_out_last = getattr(model, "last_pos", None)
                if args.root_pos and args.w_rootpos > 0 and tgt in rp_tgt and src in rp_tgt:
                    _ri = int(np.where(np.asarray(data[tgt]["spec"].parents) < 0)[0][0])
                    _pr = pos_out_last[:, _ri] if pos_out_last is not None else None
                    if _pr is not None:
                        _dw = rp_tgt[tgt]["rw"][sel] - rp_tgt[src]["rw"][sel]      # 世界偏移
                        _yw = rp_tgt[tgt]["yaw"][sel]
                        _c2, _s3 = torch.cos(_yw), torch.sin(_yw)
                        _h = (data[tgt]["con"].gap_root(sel) if data[tgt].get("con_ok")
                              else rp_tgt[tgt]["rw"][sel][:, 1])
                        # 0901 Y5b (lloyd): 只监督高度。实测水平偏移 98% 是 retarget 原点
                        # 常数 (中位 79cm, 不可学), 姿态真正决定的只有 1.5cm —— 没东西可学,
                        # 硬学只会用噪声梯度淹掉高度 (Y5: 高度被淹到 19.9cm)。
                        _lrp = torch.nn.functional.smooth_l1_loss(_pr[:, 0], _h)
                        term = term + args.w_rootpos * _lrp
                        if cex is not None: cex["rootpos"] = _lrp
                if args.root_head and args.w_root > 0 and data[tgt].get("con_ok") \
                        and getattr(model, "last_root_h", None) is not None:
                    _rh = data[tgt]["con"].gap_root(sel)
                    _tr = data[tgt]["con"].trust(sel)
                    if bool(_tr.any()):
                        _lr = torch.nn.functional.smooth_l1_loss(
                            model.last_root_h[_tr], _rh[_tr])
                        term = term + args.w_root * _lr
                        if cex is not None: cex["rooth"] = _lr
                if args.w_cbce > 0 and "cbce" in cex:
                    term = term + args.w_cbce * cex["cbce"]
                for k in ("pen", "cmaint", "cgeo", "rootpos", "rooth", "cbce"):
                    if k in cex: _acc(k, cex[k])
                for k in clog_acc:
                    if k in cex:
                        v_ = cex[k]
                        clog_acc[k].append(float(v_.detach()) if torch.is_tensor(v_) else float(v_))
            if win_starts is not None and args.w_slip_w > 0 and data[tgt].get("con_ok") \
                    and tgt in rp_tgt:
                # 窗内相邻帧 (k, k+1) 的世界系接触脚速; 行序 = 窗内时间序
                K_ = args.tdec; W_ = args.bs // K_
                _pr6 = model.last_prot.reshape(W_, K_, *model.last_prot.shape[1:])
                _sw = sel.reshape(W_, K_)
                _r0 = _sw[:, :-1].reshape(-1); _r1 = _sw[:, 1:].reshape(-1)
                _lws = slip_world(_pr6[:, :-1].reshape(-1, *_pr6.shape[2:]),
                                  _pr6[:, 1:].reshape(-1, *_pr6.shape[2:]),
                                  rp_tgt[tgt]["rw"][_r0], rp_tgt[tgt]["rw"][_r1],
                                  data[tgt]["con"], _r0, _r1, data[tgt]["spec"])
                if _lws is not None:
                    term = term + args.w_slip_w * _lws
                    _acc("slip_w_main", _lws)
                    # 日志只记"两帧都没遮"的样本 (与 Z1/Z2 副批同口径); 被遮的帧对照罚
                    # (物理先验不看有没有观测), 但它们的读数不代表正常解码精度
                    if jm is not None:
                        _um = ~jm.any(1).reshape(W_, K_)
                        _um = (_um[:, :-1] & _um[:, 1:]).reshape(-1)
                        with torch.no_grad():
                            _lu = slip_world(_pr6[:, :-1].reshape(-1, *_pr6.shape[2:]),
                                             _pr6[:, 1:].reshape(-1, *_pr6.shape[2:]),
                                             rp_tgt[tgt]["rw"][_r0], rp_tgt[tgt]["rw"][_r1],
                                             data[tgt]["con"], _r0, _r1, data[tgt]["spec"],
                                             rowmask=_um)
                        if _lu is not None:
                            slipw_log.append(float(_lu) ** 0.5)
                    else:
                        slipw_log.append(float(_lws) ** 0.5)
            _acc("total_main", term)
            (term / len(step_combos)).backward()
        if args.reenc_consistency > 0 and float(reenc_rng.random()) < args.reenc_prob:
            # N3e: decode -> re-encode cycle, half batch, one random direction.
            B2 = max(8, args.bs // 2)
            sel2 = sel[:B2]
            a, b = combos[int(reenc_rng.integers(len(combos)))]
            zp_c, zr_c = model.encode(data[a]["pose"][sel2], geom(data, a, B2))
            prot_c, ppos_c = model.decode(zp_c, zr_c, geom(data, b, B2))
            re_in = torch.cat([prot_c, ppos_c], -1)                 # decoded pose [B,J,9]
            zp_r, zr_r = model.encode(re_in, geom(data, b, B2))
            lre = ((zp_r - zp_c) ** 2).mean() + ((zr_r - zr_c) ** 2).mean()
            _acc("reenc", lre)
            (args.reenc_consistency * lre).backward()
        if slip_starts is not None:
            # v2 P1.5 slip 副批: 相邻两帧 (r, r+1) 各自单帧解码, 罚"GT 脚踩地时解码
            # 脚还在水平漂"。两帧拼成一个 2*B4 的 batch 走同一次前向, 省一半开销。
            # 脚 = 关节名含 ankle/foot/toe (gr3 只有 foot_pitch, 也覆盖到)。
            # 踩地判据 = 低且慢规则的"慢"半边: GT 脚水平速度 < 0.10 m/s
            # (缓存是根相对坐标、无世界根, 这是能拿到的最忠实版本, 已记录在案)。
            # 损失单位 = (m/s)^2 的水平解码脚速均方 —— 和验收线 (<0.15 m/s) 同尺。
            # 水平 = x/z (缓存 Y-up, 见 prep_xwbc_cache); FPS=30 (lafan/amass 统一)。
            B4 = max(4, int(args.bs * args.slip_frac))
            r0 = torch.as_tensor(
                slip_starts[slip_rng.integers(len(slip_starts), size=B4)])
            a, b = combos[int(slip_rng.integers(len(combos)))]
            rows = torch.cat([r0, r0 + 1])
            con_s = None
            if args.w_slip_w > 0 and args.contact_chan and data[a].get("con_ok"):
                # 与主批同口径: 源侧喂 sdist 通道, 按 con_dropout 整对丢弃 (NaN = 无信息)
                con_s = data[a]["con"].sdist(rows)
                if args.con_dropout > 0:
                    _dp = (torch.rand(B4, 1, device=device) < args.con_dropout).repeat(2, 1)
                    con_s = torch.where(_dp, torch.full_like(con_s, float("nan")), con_s)
            zp_s, zr_s = model.encode(data[a]["pose"][rows], geom(data, a, 2 * B4), con=con_s)
            prot_s, _ = model.decode(zp_s, zr_s, geom(data, b, 2 * B4))
            spec_b, reach_b = data[b]["spec"], data[b]["reach"]
            ft = data[b].get("_feet_m")
            if ft is None:
                ft = torch.tensor([any(k in n.lower() for k in ("ankle", "foot", "toe"))
                                   for n in spec_b.joint_names], device=device)
                data[b]["_feet_m"] = ft
            gtp = data[b]["pose"][rows][..., 6:9] * reach_b        # [2B4,J,3] cm
            gvel = (gtp[:B4] - gtp[B4:])[..., (0, 2)].norm(dim=-1) * 30.0 / 100.0
            planted = (gvel < 0.10)[:, ft]                          # [B4,F] GT 踩地
            # v2 P3 无滑移: 复用同一个副批和同一次前向, 但作用域从"脚 + GT 速度门"
            # 换成"标注器说这两帧都在接触的**任意**关节"(手掌撑地、膝盖跪地一样算)。
            # 两项合成一次 backward: 与分别 backward 的梯度是同一个和, 且 w_slip_c=0
            # 时整段退化成逐位一致的老路径 (loss_s 就是 slip_w*lslip 那一个)。
            loss_s = None
            use_c = args.w_slip_c > 0 and bool(data[b].get("con_ok"))
            use_w = args.w_slip_w > 0 and bool(data[b].get("con_ok")) and b in rp_tgt
            if args.w_hpair > 0 and args.root_pos and b in rp_tgt \
                    and getattr(model, "last_pos", None) is not None:
                _rib = int(np.where(np.asarray(spec_b.parents) < 0)[0][0])
                _hp = model.last_pos[:, _rib, 0]                        # [2B4] 解码根高 (m)
                _hg = (data[b]["con"].gap_root(rows) if data[b].get("con_ok")
                       else rp_tgt[b]["rw"][rows][:, 1])                # 与主批 w_rootpos 同口径
                _dh = ((_hp[B4:] - _hp[:B4]) - (_hg[B4:] - _hg[:B4])) * 30.0   # m/s
                lhp = (_dh ** 2).mean()
                _acc("hpair", lhp)
                loss_s = args.w_hpair * lhp
                hpair_log.append(float(lhp) ** 0.5)
            need_fk = (args.slip_w > 0 and bool(planted.any())) or use_c or use_w
            if need_fk:
                fks = fk_pos_torch(prot_s, spec_b)                  # [2B4,J,3] cm
            if args.slip_w > 0 and planted.any():
                dvel = ((fks[:B4] - fks[B4:])[..., (0, 2)].norm(dim=-1)
                        * 30.0 / 100.0)[:, ft]                      # 解码脚水平速度 m/s
                lslip = (dvel[planted] ** 2).mean()
                _acc("slip", lslip)
                loss_s = args.slip_w * lslip
                slip_log.append(float(lslip) ** 0.5)
            if use_c:
                ch = data[b]["con"].hard(rows)                       # [2B4,J]
                cs = data[b]["con"].soft(rows)
                wc = cs[:B4] * (ch[:B4] & ch[B4:]).float()           # 两帧都接触才算支撑相
                dv = ((fks[:B4] - fks[B4:])[..., CONTACT_HORIZ].norm(dim=-1)
                      * CONTACT_FPS / 100.0)                         # [B4,J] m/s
                den = wc.sum(1); actc = den > 1e-6
                if bool(actc.any()):                                 # 逐帧按接触数归一
                    lcs = ((wc * dv ** 2).sum(1)[actc] / den[actc]).mean()
                    _acc("slip_c", lcs)
                    loss_s = args.w_slip_c * lcs if loss_s is None \
                        else loss_s + args.w_slip_c * lcs
                    slipc_log.append(float(lcs) ** 0.5)
            if use_w:
                conb = data[b]["con"]
                ch = conb.hard(rows); cs = conb.soft(rows); ctr = conb.trust(rows)
                ww = cs[:B4] * (ch[:B4] & ch[B4:]).float() \
                     * (ctr[:B4] & ctr[B4:]).float()[:, None]          # [B4,J]
                pw = fks
                if conb.sole_j is not None:                             # 脚 -> 脚底点 (cm)
                    js = conb.sole_j
                    Rj = _r2m(prot_s[:, js])
                    pw = pw.clone()
                    pw[:, js] = pw[:, js] + torch.einsum("bkxy,ky->bkx", Rj, conb.sole_off) * 100.0
                drw = rp_tgt[b]["rw"][r0 + 1] - rp_tgt[b]["rw"][r0]     # [B4,3] 米, GT 根世界位移
                dxw = (pw[B4:] - pw[:B4]) / 100.0 + drw[:, None, :]      # [B4,J,3] 世界系位移
                vw = dxw[..., CONTACT_HORIZ].norm(dim=-1) * CONTACT_FPS  # m/s
                denw = ww.sum(1); actw = denw > 1e-6
                if bool(actw.any()):
                    lws = ((ww * vw ** 2).sum(1)[actw] / denw[actw]).mean()
                    _acc("slip_w", lws)
                    loss_s = args.w_slip_w * lws if loss_s is None \
                        else loss_s + args.w_slip_w * lws
                    slipw_log.append(float(lws) ** 0.5)
            if loss_s is not None:
                loss_s.backward()
        if args.w_align > 0 or args.w_contrast > 0:
            B = len(sel)
            cnames = train_names
            if args.contrast_sample and args.contrast_sample < len(train_names):
                pick = combo_rng.choice(len(train_names), args.contrast_sample, replace=False)
                cnames = [train_names[i] for i in pick]
            zs = {n: model.encode(data[n]["pose"][sel], geom(data, n, B)) for n in cnames}
            pairs = [(a, b) for i, a in enumerate(cnames) for b in cnames[i + 1:]]
            extra = 0.0
            for a, b in pairs:                       # mean over unordered embodiment pairs
                (zp_s, zr_s), (zp_g, zr_g) = zs[a], zs[b]
                if args.w_align > 0:      # L2 pull (baseline, collapses if strong)
                    extra = extra + args.w_align / len(pairs) * (
                        ((zp_s - zp_g) ** 2).mean() + ((zr_s - zr_g) ** 2).mean())
                if args.w_contrast > 0:   # InfoNCE: same-motion attract, other-motion repel -> no collapse
                    vs = torch.cat([zp_s.reshape(B, -1), zr_s.reshape(B, -1)], -1)
                    vg = torch.cat([zp_g.reshape(B, -1), zr_g.reshape(B, -1)], -1)
                    vs = torch.nn.functional.normalize(vs, dim=-1); vg = torch.nn.functional.normalize(vg, dim=-1)
                    logits = (vs @ vg.t()) / args.tau        # [B,B], diagonal = same motion
                    labels = torch.arange(B, device=vs.device)
                    nce = 0.5 * (torch.nn.functional.cross_entropy(logits, labels) +
                                 torch.nn.functional.cross_entropy(logits.t(), labels))
                    extra = extra + args.w_contrast / len(pairs) * nce
                    _acc("contrast", nce)
            if args.judge_mode != "off":             # E14 critics: same graph, single backward
                jloss = 0.0
                for n in cnames:
                    zp_j, zr_j = zs[n]
                    if args.judge_mode == "observe": # critics learn, backbone untouched
                        zp_j, zr_j = zp_j.detach(), zr_j.detach()
                    jl = model.judge(zp_j, zr_j)
                    tgt = torch.zeros_like(jl); tgt[:, emb_idx[n]] = 1.0
                    jloss = jloss + torch.nn.functional.binary_cross_entropy_with_logits(jl, tgt)
                extra = extra + args.w_judge * jloss / len(cnames)
                _acc("judge", jloss / len(cnames))
            extra.backward()
        if args.grad_share_every and (step + 1) % args.grad_share_every == 0:
            from dual_common import grad_share
            rows = []
            for src, tgt in step_combos:
                rows.append(grad_share(model, data, src, tgt, sel, train_params,
                                       args.w_pos, fk_loss=args.fk_loss,
                                       dof_loss=args.dof_loss))
            agg = {"step": step + 1}
            for k in rows[0]:
                agg[k] = round(float(np.mean([r[k] for r in rows])), 6)
            agg["combos"] = [f"{a_}->{b_}" for a_, b_ in step_combos]
            gs_log.append(agg)
            json.dump(gs_log, open(f"{args.out}/grad_share.json", "w"), indent=1)
            print("GRADSHARE", json.dumps({k: agg[k] for k in agg
                                           if k.endswith("share") or k == "step"}), flush=True)
        torch.nn.utils.clip_grad_norm_(train_params, 1.0)
        opt.step(); sched.step()
        if args.log_every and ((step + 1) % args.log_every == 0 or step + 1 == args.steps):
            _now = time.time()
            _row = {"step": step + 1, "wall_s": round(_now - t0, 1),
                    "s_per_step": round((_now - _t_log) / args.log_every, 4),
                    "lr": float(opt.param_groups[0]["lr"])}
            _row.update({k: round(float(np.mean(v)), 6) for k, v in _sl.items() if v})
            _lf.write(json.dumps(_row) + "\n"); _lf.flush()
            _sl.clear(); _t_log = _now
            if args.probe_steps and step + 1 >= args.probe_steps:
                print(f"PROBE {json.dumps(_row)}", flush=True)
                print(f"PROBE s/step={_row['s_per_step']:.3f} "
                      f"gpu_peak_GB={torch.cuda.max_memory_allocated() / 1e9:.1f}", flush=True)
                return
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            ev = evaluate(model, data, names, eval_combos, val_idx); ev["step"] = step + 1
            ev["wall_s"] = round(time.time() - t0, 1)
            ev["s_per_step"] = round(ev["wall_s"] / (step + 1), 4)
            # 0903: 中途存档 (原来只在最后存) —— lloyd 规矩: 问进度必附视频, 中途要能
            # 被 eval_dual / showcase_render 直接读。
            torch.save(model.state_dict(), f"{args.out}/model.pt")
            json.dump(vars(args), open(f"{args.out}/config.json", "w"), indent=2)
            # 0831: clog_acc 原来从不清空, 于是 c_pen/c_cmaint 是**从第 1 步起的累计
            # 平均**, 被随机初始化那几百步 (关节乱飞到地下, pen 高达几十) 永久拖着 ——
            # 实测末尾报 2.2, 而同一个 checkpoint 真实的 pen 只有 0.0001。区间平均才是
            # "现在什么水平", 累计平均只会让人以为损失爆了。
            if slip_starts is not None and slip_log:
                # 踩地样本上"解码脚水平速度"的近期均值 (m/s) —— slip 治没治好看这行
                ev["slip_dec_ms"] = round(float(np.mean(slip_log[-200:])), 4)
            if slip_starts is not None and slipc_log:
                ev["slipc_dec_ms"] = round(float(np.mean(slipc_log[-200:])), 4)
            if slip_starts is not None and hpair_log:
                # 相邻帧解码根高变化 vs GT 的均方根 (m/s) —— 根高跳变治没治好看这行
                ev["hpair_ms"] = round(float(np.mean(hpair_log[-200:])), 4)
            if (slip_starts is not None or win_starts is not None) and slipw_log:
                # 世界系接触脚速 (m/s) 近期均值 —— 走路滑冰治没治好看这行
                ev["slipw_dec_ms"] = round(float(np.mean(slipw_log[-200:])), 4)
            for k, v in clog_acc.items():
                if v:
                    # 训练侧接触量的近期均值。pen 是米^2, 开方 x100 = 平均穿深 cm,
                    # 直接和 spec §4 的 "穿地 p95 <1cm" 同尺; cmaint 米; cbce 无量纲。
                    ev[f"c_{k}"] = round(float(np.mean(v[-200 * max(1, len(step_combos)):])), 6)
                    v.clear()
            log.append(ev)
            print("EVAL", json.dumps(ev)); json.dump(log, open(f"{args.out}/eval_log.json", "w"), indent=2)
    torch.save(model.state_dict(), f"{args.out}/model.pt")
    json.dump(vars(args), open(f"{args.out}/config.json", "w"), indent=2)
    _lf.close()
    print(f"DONE {args.out} wall={time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
