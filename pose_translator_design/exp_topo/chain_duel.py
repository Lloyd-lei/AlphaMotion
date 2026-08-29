"""Which chain: GEM->Greenwich, or GEM->Equator->Greenwich?

    chain_duel.py --clip /var/tmp/alphamotion/genmo_bridge/prompt_runclap.pt

Three arms, so the cost of the token layer is separated from the cost of
quantisation (without the middle arm the duel blames Equator for both):

  A   GEM -> Greenwich model.encode() -> robot
  A'  GEM -> Greenwich explicit CODES -> robot
  B   GEM -> codes -> Equator 32 tokens -> codes -> robot (what buys controllability)

MEASURED 0814: A and A' come out bit-identical (0.00 cm on every joint), because
model.encode() already returns proj_in(proj_out(q)) — the quantised latent. There
is no continuous arm to have; Greenwich is quantised by construction. So the real
duel is two chains, and the whole gap belongs to Equator. A' is kept only as the
standing check that this stays true.

Rulers are the GT-free ones this project already trusts, plus the standing rules:
positions ONLY through FK; cross-embodiment quantities normalised by each body's
own reach; no same-skeleton no-observation decode anywhere.
  跟随度   time correlation of each key joint's DIRECTION from the root
           (scale-free, so human vs robot is a legal comparison)
  幅度比   robot amplitude / human amplitude, both in body-reach units
  jerk / 限位  physical plausibility
  链间差   chains against each other on the SAME robot, in cm
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
import v5_common as V                                            # noqa: E402
import eval_newbody as E                                         # noqa: E402
import manifold_project as MP                                    # noqa: E402
import render_newbody as RN                                      # noqa: E402
import greenwich_carnival as GC                                  # noqa: E402
from bridge_a import load_bridge_a                               # noqa: E402
from bridge_b import tokens_of                                   # noqa: E402
from manifold_project import _limits, fit_angles                 # noqa: E402
from common import rot6d_to_matrix                               # noqa: E402
from spatial5 import pad_band                                    # noqa: E402
from genmo_to_robot import human_tile, font                      # noqa: E402
from PIL import Image, ImageDraw                                 # noqa: E402

RUNS = V.RUNS
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "R4_VIZ", "mp4", "genmo")
# the tokenizer pinned this; using a per-clip reach would put the codes in a
# different space from the one A3/B3 were trained on (bridge_codes/meta.json)
REACH = 103.64282989501953
KEYS = ("head", "l_wrist", "r_wrist", "l_ankle", "r_ankle")


def dir_series(p, kidx, root):
    """[T,J,3] -> {key: unit direction from root [T,3]} — scale free."""
    out = {}
    for k in KEYS:
        v = p[:, kidx[k]] - p[:, root]
        out[k] = v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)
    return out


def amp_series(p, kidx, root, reach):
    return {k: np.linalg.norm(p[:, kidx[k]] - p[:, root], axis=-1) / reach
            for k in KEYS}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="/var/tmp/alphamotion/genmo_bridge/prompt_runclap.pt")
    ap.add_argument("--spatial", default="PROD_c1_91k_dim20_tail_0810")
    ap.add_argument("--stage_a", default="BRIDGE_A3_prod_0810")
    ap.add_argument("--T", type=int, default=120, help="one Stage-A window "
                    "(A3 was trained on n in 15..120; longer needs chunking)")
    ap.add_argument("--robot", default="Unitree G1")
    ap.add_argument("--multi", default=None,
                    help="see chain_duel_multi.py for the many-clip aggregate")
    ap.add_argument("--tag", default="runclap")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev, T = args.device, args.T
    os.makedirs(OUT, exist_ok=True)

    d = torch.load(args.clip, map_location="cpu", weights_only=False)
    gr6 = d["rot6d"][:T]
    T = gr6.shape[0]
    sp = np.load(f"{os.environ['V5_AMASS']}/human_smpl_spec.npz", allow_pickle=True)
    sspec = E.SkeletonSpec(str(sp["name"]), sp["parents"], sp["rest_offsets"],
                           [str(x) for x in sp["joint_names"]])

    class S:
        parents = sp["parents"]; rest_offsets = sp["rest_offsets"]
    hp = V.fk_pos(gr6.numpy(), S)                       # [T,22,3] cm
    pose = torch.cat([gr6, torch.from_numpy(hp).float() / REACH], -1).to(dev)

    model, cfg = E.load_run(f"{RUNS}/{args.spatial}", dev)
    enc = cfg.get("sem_encoder")
    sdof = np.load(f"{os.environ['V5_AMASS']}/human_smpl_dof.npz")["dof"]
    sg = E.target_geom(sspec, sdof, dev, sem_encoder=enc)
    ma, _acfg = load_bridge_a(f"{RUNS}/{args.stage_a}", dev)

    # ---- the three latents ----
    zp_a, zr_a = model.encode(pose, E.geom_batch(sg, T))
    fp, fr = model.encode_features(pose, E.geom_batch(sg, T))
    ipS, irS = model.ints_from_features(fp, fr)         # ([S,T,K,f], [S,T,K,f])
    ip, ir = ipS[0], irS[0]                             # stage 0 = the REAL codes
    assert ip.shape[1:] == (128, 20) and ir.shape[1:] == (128, 20), \
        (ip.shape, ir.shape)                            # print-the-shapes rule
    qa = (torch.cat([ip, ir], 1).float() - 4.0) / 3.5   # [T,256,20]
    zp_q, zr_q = model.fsqp.proj_out(qa[:, :128]), model.fsqr.proj_out(qa[:, 128:])

    codes = torch.cat([ip, ir], 1).long()[None]         # [1,T,256,20]
    tok, ep = tokens_of(ma, codes.to(dev))
    W = 5 ** torch.arange(6, device=dev)
    z = ma.fsq.from_ints(((tok[0][:, None] // W) % 5)[None].float())[None]
    prob = F.softmax(ma.decode(z, ep, T)[0], -1)
    lv = (torch.arange(9, device=dev).float() - 4.0) / 3.5
    qb = (prob * lv).sum(-1)
    qb[0] = qa[0]                                       # A3's endpoint pass-through
    zp_b, zr_b = model.fsqp.proj_out(qb[:, :128]), model.fsqr.proj_out(qb[:, 128:])
    agree = float((tok[0].unique().numel()))
    print(f"frames {T}  Equator token 用了 {agree}/32 个不同码", flush=True)

    # ---- decode all three to the robot ----
    name, xml, status = [r for r in GC.ROSTER if r[0] == args.robot][0]
    spec_t, dof_t, rest_t, qn_t, xml_t = E.get_target(xml, E.CACHE_AMASS)
    g_t = E.target_geom(spec_t, dof_t, dev, sem_encoder=enc)
    dof_d = torch.as_tensor(dof_t, device=dev, dtype=torch.float64)
    rest_d = torch.as_tensor(rest_t, device=dev, dtype=torch.float64)
    lo_, hi_, lim_ = _limits(dof_d, 1.0)
    span_ = (hi_ - lo_).clamp(min=1e-6)
    kj_t, _n, _t = V.key_joints(spec_t)
    kidx_t = dict(zip(("root",) + KEYS, kj_t))
    kj_h, _n, _t = V.key_joints(sspec)
    kidx_h = dict(zip(("root",) + KEYS, kj_h))

    res, tiles = {}, [human_tile(hp, sspec, T)]
    labels = ["SOURCE  GEM prompt output"]
    for tag, (zp, zr) in (("A 连续", (zp_a, zr_a)), ("A' 量化码", (zp_q, zr_q)),
                          ("B 过Equator", (zp_b, zr_b))):
        rot, _ = model.decode(zp, zr, E.geom_batch(g_t, T))
        Rg = rot6d_to_matrix(rot).double()
        qa_, _ = fit_angles(Rg, spec_t, dof_d, rest=rest_d, clamp=True,
                            method="greedy")
        gR, gpos = MP.fk_from_angles(qa_, spec_t, dof_d, rest=rest_d,
                                     root_R=Rg[:, 0])
        p = gpos.cpu().numpy()
        near = (((qa_ - lo_[None]) / span_[None] < 0.02) |
                ((hi_[None] - qa_) / span_[None] < 0.02)) & lim_[None]
        reach_r = float(np.linalg.norm(
            p[:, [kidx_t[k] for k in KEYS]] - p[:, kidx_t["root"]][:, None],
            axis=-1).max())
        res[tag] = {
            "pos": p,
            "jerk": float(np.abs(np.diff(p, 3, axis=0)).mean()),
            "limit": float(near[:, lim_].float().mean()) if lim_.any() else 0.0,
            "dir": dir_series(p, kidx_t, kidx_t["root"]),
            "amp": amp_series(p, kidx_t, kidx_t["root"], reach_r),
        }
        _, _, qj = MP.project(Rg, spec_t, dof_d, rest=rest_d, method="global")
        gR2, gp2 = MP.fk_from_angles(qj, spec_t, dof_d, rest=rest_d,
                                     root_R=Rg[:, 0])
        rt = GC.render_tile(xml_t, spec_t, qn_t, qj.cpu().numpy(),
                            gR2[:, 0].cpu().numpy(), gp2.cpu().numpy(), T,
                            status=status)
        tiles.append([pad_band(f) if f.shape[0] != tiles[0][0].shape[0] else f
                      for f in rt])
        labels.append(tag)
        print(f"  {tag} decoded", flush=True)

    reach_h = float(np.linalg.norm(
        hp[:, [kidx_h[k] for k in KEYS]] - hp[:, kidx_h["root"]][:, None],
        axis=-1).max())
    hdir = dir_series(hp, kidx_h, kidx_h["root"])
    hamp = amp_series(hp, kidx_h, kidx_h["root"], reach_h)

    print(f"\n== 对人体源动作的跟随度 (方向时间相关, 1.00=完全跟随) ==")
    print(f"{'':>12}" + "".join(f"{k:>11}" for k in KEYS) + f"{'平均':>9}")
    table = {}
    for tag in res:
        cs = []
        for k in KEYS:
            c = np.mean([np.corrcoef(hdir[k][:, a], res[tag]["dir"][k][:, a])[0, 1]
                         for a in range(3)])
            cs.append(c)
        table[tag] = cs
        print(f"{tag:>12}" + "".join(f"{c:11.3f}" for c in cs)
              + f"{np.mean(cs):9.3f}")
    print(f"\n== 幅度比 (机器人/人体, 各自按自身臂展归一; 1.00=不缩水) ==")
    print(f"{'':>12}" + "".join(f"{k:>11}" for k in KEYS))
    for tag in res:
        rs = [float(res[tag]["amp"][k].std() / (hamp[k].std() + 1e-9))
              for k in KEYS]
        print(f"{tag:>12}" + "".join(f"{r:11.2f}" for r in rs))
    print(f"\n== 物理合理性 ==")
    for tag in res:
        print(f"{tag:>12}  jerk {res[tag]['jerk']:.4f}   限位命中 "
              f"{res[tag]['limit']*100:.1f}%")
    print(f"\n== 链间差 (同一台机器人上, 关节位置中位差 cm) ==")
    ks = list(res)
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            dd = np.median(np.linalg.norm(res[ks[i]]["pos"] - res[ks[j]]["pos"],
                                          axis=-1))
            print(f"   {ks[i]:>12} vs {ks[j]:<12} {dd:6.2f} cm")
    json.dump({t: {"follow": table[t],
                   "jerk": res[t]["jerk"], "limit": res[t]["limit"]}
               for t in res} | {"robot": name, "T": T, "clip": args.clip},
              open(f"{OUT}/chain_duel_{args.tag}.json", "w"), indent=2)

    L = min(len(t) for t in tiles)
    frames = []
    for t in range(L):
        row = np.concatenate([ti[t] for ti in tiles], 1)
        im = Image.fromarray(row).convert("RGB")
        dr = ImageDraw.Draw(im)
        dr.text((8, 6), f"链路对决 on {name}", font=font(15), fill=(255, 230, 150))
        for k, lb in enumerate(labels):
            dr.text((k * GC.TILE_W + 8, 26), lb, font=font(12),
                    fill=(255, 220, 120) if k == 3 else (215, 220, 230))
        frames.append(np.asarray(im))
    p = f"{OUT}/chain_duel_{args.tag}.mp4"
    RN.write_mp4(p, np.stack(frames), 30)
    print("->", p, flush=True)


if __name__ == "__main__":
    main()
