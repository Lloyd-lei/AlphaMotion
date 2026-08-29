"""COMPOSITION demo/verdict for the B5 cross-attention wrapper.

    composition_demo.py --stage_b BRIDGE_B5_xattn_0814

lloyd's acceptance target 0814: "walk 和 wave hand 在数据集里面都没有出现过,但是
prompt 输入可以把边走边挥手给组合出来".

No goal is given — the goal slot is dropped, so the ONLY thing steering the
generation is the prompt.  Same start pose and same seed for every column, so
any difference between columns is text and nothing else.

Two outputs, because the video alone cannot settle it:
  * a 4-column video (mesh, X2) — lloyd judges by eye
  * a GT-free similarity matrix: each generated clip is embedded by the FROZEN
    TEXT_ALIGN_v3 retriever and scored against all prompts.  If the diagonal
    wins, text steered the draw; if a column looks the same under every prompt,
    it did not.  The composite row is the one that matters: the composite clip
    must beat BOTH parents on the composite prompt.
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES",
                      "/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
os.environ.setdefault("V5_AMASS", "/var/tmp/alphamotion/topo_exp/cache_xwbc_amass_full")
import v5_common as V                                                # noqa: E402
import eval_newbody as E                                             # noqa: E402
import manifold_project as MP                                        # noqa: E402
import render_newbody as RN                                          # noqa: E402
import greenwich_carnival as GC                                      # noqa: E402
from bridge_a import CodeData, load_bridge_a                         # noqa: E402
from bridge_b import BridgeB                                         # noqa: E402
from bridge_b5 import BridgeB5, Critic                               # noqa: E402
from bridge_b import V5 as V5x                                       # noqa: E402
from semantics import QWEN_HF_HOME, QWEN_MODEL                       # noqa: E402
from spatial5 import pad_band, font                                  # noqa: E402
from common import rot6d_to_matrix                                   # noqa: E402
from PIL import Image, ImageDraw                                     # noqa: E402

RUNS = V.RUNS
RM = "/var/tmp/alphamotion/robot_models"
X2 = f"{RM}/agibot_x2_urdf/X2_URDF-v1.4.0/agibot_x2_ultra_mjcf.xml"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "R4_VIZ", "mp4", "bridge_demo")

# Caption co-occurrence audited 0814 over all 25,936 captions. lloyd assumed
# walk+wave never co-occurs; it does, 46 times (0.18%) — low-resource, not
# zero-shot. run+clap is the honest zero-shot pair: 1339 and 257 parent
# captions, ZERO captions containing both. Both sets are run and reported.
SETS = {
    "walkwave": (["a person walks forward",
                  "a person waves with their right hand",
                  "a person walks forward while waving with their right hand",
                  "",
                  "a person"],
                 ["walk (7404 caps)", "wave (744 caps)",
                  "WALK + WAVE  [46 co-occur]", "(no text)",
                  "neutral@same CFG"]),
    "runclap": (["a person runs forward",
                 "a person claps their hands",
                 "a person runs forward while clapping their hands",
                 "",
                 "a person"],
                ["run (1339 caps)", "clap (257 caps)",
                 "RUN + CLAP  [0 co-occur]", "(no text)",
                 "neutral@same CFG"]),
}


def encode_prompts(prompts, dev, maxl=32):
    """same path as prep_text_seq.py: keep the whole BPE word sequence."""
    os.environ["HF_HOME"] = QWEN_HF_HOME
    from transformers import AutoTokenizer, AutoModel
    hub = f"{QWEN_HF_HOME}/hub"
    tk = AutoTokenizer.from_pretrained(QWEN_MODEL, cache_dir=hub,
                                       padding_side="right")
    mo = AutoModel.from_pretrained(QWEN_MODEL, cache_dir=hub,
                                   dtype=torch.float16).to(dev).eval()
    with torch.no_grad():
        inp = tk([p or "x" for p in prompts], padding="max_length",
                 truncation=True, max_length=maxl, return_tensors="pt").to(dev)
        h = mo(**inp).last_hidden_state.float()
        pooled = torch.stack([h[i, int(inp["attention_mask"][i].sum()) - 1]
                              for i in range(len(prompts))])
    ln = inp["attention_mask"].sum(1).long()
    del mo
    torch.cuda.empty_cache()
    return h, ln, pooled


def sample_tokens(logits_fn, dev, temp=0.9, seed=0):
    """one-by-one confidence-ordered exact sampling, GOAL DROPPED.

    Takes a closure so the B5 (cross-attention) and B4 (8 input slots) arms go
    through byte-identical sampling — otherwise the duel measures the sampler."""
    gen = torch.Generator(device=dev).manual_seed(seed)
    tok = torch.zeros(1, 32, dtype=torch.long, device=dev)
    mask = torch.ones(1, 32, dtype=torch.bool, device=dev)
    for _ in range(32):
        logits = logits_fn(tok, mask)
        probs = F.softmax(logits[0] / temp, -1)
        conf = probs.max(-1).values.masked_fill(~mask[0], -1)
        pos = int(conf.argmax())
        tok[0, pos] = torch.multinomial(probs[pos], 1, generator=gen)
        mask[0, pos] = False
    return tok[0]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage_a", default="BRIDGE_A3_prod_0810")
    ap.add_argument("--stage_b", default="BRIDGE_B5_xattn_0814")
    ap.add_argument("--spatial", default="PROD_c1_91k_dim20_tail_0810")
    ap.add_argument("--critic", default="TEXT_ALIGN_v3_fixed_0812")
    ap.add_argument("--T", type=int, default=90)
    ap.add_argument("--n_seeds", type=int, default=6)
    ap.add_argument("--temp", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set", dest="pset", default="runclap",
                    choices=tuple(SETS))
    ap.add_argument("--arch", default="b5", choices=("b5", "b4"),
                    help="b4 = the old 8-input-slot wrapper, run as\n                         the control arm through identical sampling")
    ap.add_argument("--cfg", type=float, default=1.0,
                    help="classifier-free guidance weight. Both arms were\n                         trained with text dropout, so the null branch\n                         exists in each: logits = null + w*(text - null).\n                         w=1 is plain conditioning. Text carries only\n                         ~0.03-0.06 nats against a strong motion prior,\n                         so w>1 is how that signal is made visible.")
    ap.add_argument("--cfg_list", default="",
                    help="extra guidance weights to sweep in the same\n                         process, e.g. 1,2,4,8; only --cfg is rendered")
    ap.add_argument("--keep_goal", action="store_true",
                    help="HOMECOMING mode: start == goal == the same\n                         standing pose, so the endpoints carry no action\n                         information and text is the only thing that can\n                         decide what happens in the 90 interior frames.\n                         Without this the run sits in the no-goal regime,\n                         which D4 documents as collapsed — the 0814 B4v5\n                         control produced 8-17/32 distinct codes and four\n                         kinematically identical columns.")
    ap.add_argument("--tag", default="b5")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev, T = args.device, args.T
    PROMPTS, LABELS = SETS[args.pset]
    os.makedirs(OUT, exist_ok=True)

    ma, _ = load_bridge_a(f"{RUNS}/{args.stage_a}", dev)
    if args.arch == "b5":
        mb = BridgeB5(ep_dim=ma.d).to(dev)
    else:
        from bridge_b4 import BridgeB4
        mb = BridgeB4(ep_dim=ma.d).to(dev)
    mb.load_state_dict(torch.load(f"{RUNS}/{args.stage_b}/model.pt",
                                  map_location=dev)); mb.eval()
    critic = Critic(args.critic, dev)
    sp, spcfg = E.load_run(f"{RUNS}/{args.spatial}", dev)
    data = CodeData(args.spatial, use_root=True)
    spec_t, dof_t, rest_t, qn_t, xml_t = E.get_target(X2, E.CACHE_AMASS)
    g_t = E.target_geom(spec_t, dof_t, dev, sem_encoder=spcfg.get("sem_encoder"))
    dof_d = torch.as_tensor(dof_t, device=dev, dtype=torch.float64)
    rest_d = torch.as_tensor(rest_t, device=dev, dtype=torch.float64)
    W = 5 ** torch.arange(6, device=dev)
    seqs, lens, pooled = encode_prompts(PROMPTS, dev)

    # shared neutral start: one standing frame pair, identical for every column
    kd = V.KeyData(V.AMASS, ["human_smpl"], device="cpu")
    s0, _ = kd.sample_starts(1, np.random.default_rng(args.seed), "val",
                             span=4, families=["walk"])
    r0 = int(s0[0])
    cp = torch.from_numpy(np.ascontiguousarray(data.c[r0:r0 + 2])).long()
    pk = torch.from_numpy(np.ascontiguousarray(data.cr[r0:r0 + 2])).long()
    rr = torch.empty_like(cp)
    rr[..., 0::2] = pk & 0x0F; rr[..., 1::2] = (pk >> 4) & 0x0F
    c01 = torch.cat([cp, rr], 1)
    ep_c = torch.stack([c01[0], c01[1], c01[0], c01[1]])[None].to(dev)
    ep = ma.endpoints(ep_c)
    nf = torch.full((1,), float(T), device=dev)

    kj, kn, _ = V.key_joints(spec_t)
    KI = dict(zip(("root", "head", "l_wrist", "r_wrist", "l_ankle", "r_ankle"),
                  kj))
    rp = V.rest_positions(spec_t)                 # decide the vertical axis
    UP = int(np.argmax(np.abs(rp[KI["head"]] - rp[KI["l_ankle"]])))
    print(f"[joints] {dict(zip(('root','head','lw','rw','la','ra'), kn))}  "
          f"vertical axis = {'xyz'[UP]}", flush=True)

    def decode_q(q):
        zp = sp.fsqp.proj_out(q[:, :128]); zr = sp.fsqr.proj_out(q[:, 128:])
        rot, _ = sp.decode(zp, zr, E.geom_batch(g_t, q.shape[0]))
        Rg = rot6d_to_matrix(rot).double()
        _, _, qj = MP.project(Rg, spec_t, dof_d, rest=rest_d, method="global")
        gR, gp = MP.fk_from_angles(qj, spec_t, dof_d, rest=rest_d,
                                   root_R=Rg[:, 0])
        return qj, gR, gp

    def render_from(qj, gR, gp):
        return GC.render_tile(xml_t, spec_t, qn_t, qj.cpu().numpy(),
                              gR[:, 0].cpu().numpy(), gp.cpu().numpy(),
                              qj.shape[0], status="nodataset")

    def tok_to_q(tk):
        z = ma.fsq.from_ints(((tk[:, None] // W) % 5)[None].float())[None]
        prob = F.softmax(ma.decode(z, ep, T)[0], -1)
        lv = (torch.arange(9, device=dev).float() - 4.0) / 3.5
        q = (prob * lv).sum(-1)
        q[0] = (ep_c[0, 1].float() - 4.0) / 3.5
        return q

    def signature(gp):
        """MODEL-FREE kinematics off the FK ruler — no learned embedding, so it
        cannot be gamed by the critic that is part of the training loss.
          hand_osc  std of the two-wrist distance   -> clapping
          foot_osc  std of the two-ankle separation -> stepping/running
          arm_lift  wrist height above the head     -> waving
        Both signatures must light up in the composite column; a parent column
        should light up in one only."""
        hor = [a for a in range(3) if a != UP]
        hand = np.linalg.norm(gp[:, KI["l_wrist"]] - gp[:, KI["r_wrist"]], axis=-1)
        foot = np.linalg.norm(gp[:, KI["l_ankle"], hor] -
                              gp[:, KI["r_ankle"], hor], axis=-1)
        lift = np.maximum(gp[:, KI["l_wrist"], UP], gp[:, KI["r_wrist"], UP]) \
            - gp[:, KI["head"], UP]
        return {"hand_osc": float(hand.std()), "hand_min": float(hand.min()),
                "foot_osc": float(foot.std()), "foot_max": float(foot.max()),
                "arm_lift": float(lift.max())}

    gd1 = torch.full((1,), not args.keep_goal, dtype=torch.bool,
                     device=dev)
    if args.arch == "b4":                       # 8 FSQ text ids from the pooled vec
        # little-endian 5^j packing, verified 0814 to reproduce
        # text_codes.npy from TEXT_ALIGN_v3 at 100% on 256 captions
        tcodes = (critic.tc.codes(pooled).long() * W).sum(-1)

    def raw(i, on):
        if args.arch == "b5":
            td = torch.full((1,), not on, dtype=torch.bool, device=dev)
            return lambda tk, mk: mb(tk, mk, ep, nf, seqs[i:i + 1],
                                     lens[i:i + 1], goal_drop=gd1, text_drop=td)
        tx = tcodes[i:i + 1] if on else \
            torch.full((1, 8), V5x, dtype=torch.long, device=dev)
        return lambda tk, mk: mb(tk, mk, ep, nf, goal_drop=gd1, text=tx)

    def make_fn(i, on, w):
        f_t = raw(i, on)
        if w == 1.0 or not on:
            return f_t
        f_0 = raw(i, False)
        return lambda tk, mk: f_0(tk, mk) + w * (f_t(tk, mk) - f_0(tk, mk))

    def generate(w, render):
        """one full 4-column draw at guidance weight w."""
        toks, tiles, sigs = [], [], []
        for i, pr in enumerate(PROMPTS):
            fn = make_fn(i, bool(pr), w)
            cand = [sample_tokens(fn, dev, temp=args.temp,
                                  seed=args.seed * 97 + i * 13 + s)
                    for s in range(args.n_seeds)]
            toks.append(torch.stack(cand))
            # signature over EVERY seed (one clip proves nothing — 0813's three
            # retractions were all single-sample readings); render only seed 0
            ss = []
            for s, tk in enumerate(cand):
                qj, gR, gp = decode_q(tok_to_q(tk))
                ss.append(signature(gp.cpu().numpy()))
                if s == 0 and render:
                    tiles.append(render_from(qj, gR, gp))
            # mean AND spread across seeds: a gap smaller than the seed spread
            # is not a finding (0813 cost three retractions to that rule)
            sigs.append({k: float(np.mean([x[k] for x in ss])) for k in ss[0]}
                        | {k + "_sd": float(np.std([x[k] for x in ss]))
                           for k in ss[0]})
            print(f"[gen w={w}] {LABELS[i]}: "
                  f"{len(set(cand[0].tolist()))}/32 distinct codes", flush=True)
        return toks, tiles, sigs

    sweep = {}
    wl = {float(x) for x in args.cfg_list.split(",") if x.strip()}
    for w in sorted(wl | {args.cfg}):
        tk_, ti_, sg_ = generate(w, render=(w == args.cfg))
        sweep[w] = sg_
        if w == args.cfg:
            toks, tiles, sigs = tk_, ti_, sg_
    if len(sweep) > 1:
        print("\n== CFG 扫描:组合列是否【两个签名同时亮】(cm) ==")
        print(f"{'w':>5}{'run:foot':>12}{'clap:hand':>12}{'组合:foot':>12}"
              f"{'组合:hand':>12}{'无文本:foot':>13}{'无文本:hand':>12}")
        for w, sg in sweep.items():
            print(f"{w:5.1f}{sg[0]['foot_osc']:12.2f}{sg[1]['hand_osc']:12.2f}"
                  f"{sg[2]['foot_osc']:12.2f}{sg[2]['hand_osc']:12.2f}"
                  f"{sg[3]['foot_osc']:13.2f}{sg[3]['hand_osc']:12.2f}")

    # ---- GT-free verdict: retriever similarity, generations x prompts ----
    zt = critic.text(pooled)                                   # [4,256]
    NP_ = len(PROMPTS)
    M = np.zeros((NP_, NP_))
    for i in range(NP_):
        zm = critic.me(toks[i])                                # [n_seeds,256]
        M[i] = (zm @ zt.T).mean(0).cpu().numpy()
    names = [l.split(" ")[0].lower() for l in LABELS]
    print("\n== 相似度矩阵 (行=生成用的prompt, 列=打分用的prompt) ==")
    print("            " + "".join(f"{n:>12}" for n in names))
    for i in range(NP_):
        star = "  <-- 对角最高" if int(M[i].argmax()) == i else ""
        print(f"{names[i]:>10}  " + "".join(f"{M[i, j]:12.3f}" for j in range(NP_))
              + star)
    # A composition is the clip that satisfies BOTH parent descriptions at once,
    # so the criterion is row-wise min(score vs parent A, score vs parent B):
    # the parents each satisfy one description and fail the other, the composite
    # should be the only column that clears both. Scoring the composite against
    # the composite prompt (the earlier M[2,2] test) rewards matching one long
    # sentence, which either parent can also do.
    both = [float(min(M[i, 0], M[i, 1])) for i in range(NP_)]
    print(f"\n组合判据 min(对亲本A, 对亲本B): "
          + "  ".join(f"{names[i]} {both[i]:.3f}" for i in range(NP_))
          + f"  -> {'通过' if both[2] > max([both[j] for j in range(NP_) if j != 2]) else '未通过'}"
          + " (组合列必须是唯一两边都不低的)")
    diag = float(np.mean([M[i, i] > np.delete(M[i], i).max() for i in range(3)]))
    print(f"文本可辨性: 3 条 prompt 中 {diag*3:.0f} 条对角最高")
    # ---- the judge that training cannot game: raw FK kinematics ----
    print("\n== 运动学签名 (FK 直接量,与任何学习到的嵌入无关; cm) ==")
    keys = ["foot_osc", "foot_max", "hand_osc", "hand_min", "arm_lift"]
    print(f"{'':>12}" + "".join(f"{k:>12}" for k in keys))
    for i in range(NP_):
        print(f"{names[i]:>12}" + "".join(
            f"{sigs[i][k]:7.2f}±{sigs[i][k+'_sd']:<4.2f}" for k in keys))
    # Effect size against the no-text control, in units of the pooled seed
    # spread. A gap of 1.3 cm on top of a +-1.0 cm seed spread is not a result;
    # printing d stops that from being read as one.
    def d(i, k, b=3):
        s = np.hypot(sigs[i][k + "_sd"], sigs[b][k + "_sd"]) / np.sqrt(2) + 1e-9
        return (sigs[i][k] - sigs[b][k]) / s

    for b, lab in ((3, "无文本对照"),
                   (4, "中性 prompt @同等引导强度")) if NP_ > 4 else ((3, "无文本对照"),):
        print(f"\n== 相对【{lab}】的效应量 d (>1 才算看得见, <0.5 是噪声) ==")
        for k in ("foot_osc", "hand_osc", "hand_min", "arm_lift"):
            print(f"{k:>10}: " + "  ".join(f"{names[i]} {d(i, k, b):+5.2f}"
                                           for i in range(3)))
    B_ = 4 if NP_ > 4 else 3      # guided-neutral if we have it, else null
    print(f"\n组合列要两个签名同时亮 (基线={names[B_]}) -> 下肢 d "
          f"{d(2,'foot_osc',B_):+.2f} (亲本 {d(0,'foot_osc',B_):+.2f}), 上肢 d "
          f"{d(2,'hand_osc',B_):+.2f} (亲本 {d(1,'hand_osc',B_):+.2f})  -> "
          f"{'通过' if min(d(2,'foot_osc',B_), d(2,'hand_osc',B_)) > 1.0 else '未通过'}")
    print("注: 上面的 gen_R5 和相似度矩阵用的检索器同时是训练损失的一部分,"
          "只能当参考;运动学签名才是独立证据。")
    json.dump({"matrix": M.tolist(), "signatures": sigs,
               "sweep": {str(k): v for k, v in sweep.items()},
               "prompts": PROMPTS,
               "stage_b": args.stage_b, "T": T, "temp": args.temp,
               "cfg": args.cfg, "keep_goal": args.keep_goal,
               "arch": args.arch,
               "n_seeds": args.n_seeds},
              open(f"{OUT}/composition_{args.pset}_{args.tag}.json", "w"), indent=2)

    # ---- video ----
    L = max(len(t) for t in tiles)
    frames = []
    for t in range(L):
        row = np.concatenate([pad_band(ti[min(t, len(ti) - 1)])
                              for ti in tiles], 1)
        im = Image.fromarray(row).convert("RGB")
        dr = ImageDraw.Draw(im)
        for k, lb in enumerate(LABELS):
            col = (255, 220, 120) if k == 2 else (235, 235, 240)
            dr.text((k * GC.TILE_W + 8, 6), lb, font=font(15), fill=col)
            dr.text((k * GC.TILE_W + 8, 28), f'"{PROMPTS[k][:44]}"'
                    if PROMPTS[k] else "(null text control)",
                    font=font(10, False), fill=(160, 200, 255))
            dr.text((k * GC.TILE_W + 8, 44), f"self-sim {M[k, k]:.3f}",
                    font=font(10, False), fill=(150, 220, 170))
        dr.text((8, GC.TILE_H + 24),
                f"{args.stage_b}  |  "
                f"{'start==goal (homecoming): endpoints carry no action' if args.keep_goal else 'no goal'}"
                f", identical start + seed in every column: the only "
                f"difference is the prompt  |  no GT",
                font=font(12, False), fill=(210, 210, 215))
        frames.append(np.asarray(im))
    p = f"{OUT}/composition_{args.pset}_{args.tag}_x2.mp4"
    RN.write_mp4(p, np.stack(frames), 20)
    print("->", p, flush=True)


if __name__ == "__main__":
    main()
