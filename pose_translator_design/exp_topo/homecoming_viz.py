"""HOMECOMING visuals + recovery-horizon sweep.

    homecoming_viz.py

Fig 1  danger map      freedom ratio vs n per family (recovery-horizon curves,
                       crossing of ratio=1 marked per family)
Fig 2  fan chart       two contrasting poses (most-committed vs freest), 8
                       return paths each, wrist/ankle height traces overlaid
Video  danger side-by-side: locked pose (many return samples overlap) vs free
                       pose (samples spray), rendered on X2, GT human as source

All frozen models; zero training.
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
from bridge_a import CodeData, load_bridge_a, FULL                   # noqa: E402
from bridge_b import BridgeB                                         # noqa: E402
from homecoming_probe import sample_nll                              # noqa: E402
from spatial5 import capsule_tile, pad_band, font                    # noqa: E402
from train_global import build_global                                # noqa: E402
from common import rot6d_to_matrix                                   # noqa: E402
from PIL import Image, ImageDraw                                     # noqa: E402

RUNS = V.RUNS
RM = "/var/tmp/alphamotion/robot_models"
X2 = f"{RM}/agibot_x2_urdf/X2_URDF-v1.4.0/agibot_x2_ultra_mjcf.xml"
VIZ = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "R4_VIZ")
OUT_MP4 = f"{VIZ}/mp4/bridge_demo"
OUT_PNG = f"{VIZ}/png"
FAMS = ["jacks", "kick", "dance", "walk", "sit", "run", "punch", "jump"]
NS = [15, 30, 45, 60, 90, 120]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage_a", default="BRIDGE_A3_prod_0810")
    ap.add_argument("--stage_b", default="BRIDGE_B3_prod_0810")
    ap.add_argument("--spatial", default="PROD_c1_91k_dim20_tail_0810")
    ap.add_argument("--n_start", type=int, default=8)
    ap.add_argument("--n_samp", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device
    os.makedirs(OUT_MP4, exist_ok=True); os.makedirs(OUT_PNG, exist_ok=True)

    ma, _ = load_bridge_a(f"{RUNS}/{args.stage_a}", dev)
    bcfg = json.load(open(f"{RUNS}/{args.stage_b}/config.json"))
    mb = BridgeB(ep_dim=bcfg.get("ep_dim")).to(dev)
    mb.load_state_dict(torch.load(f"{RUNS}/{args.stage_b}/model.pt",
                                  map_location=dev)); mb.eval()
    sp, spcfg = E.load_run(f"{RUNS}/{args.spatial}", dev)
    enc = spcfg.get("sem_encoder")
    data = CodeData(args.spatial, use_root=True)
    kd = V.KeyData(V.AMASS, ["human_smpl"], device="cpu")
    sspec = V.load_spec(FULL, "human_smpl")
    sdof = np.load(f"{FULL}/human_smpl_dof.npz")["dof"]
    g_h = E.target_geom(sspec, sdof, dev, sem_encoder=enc)
    l6h = torch.load(f"{FULL}/human_smpl_paired_local6d.pt", mmap=True)
    spec_t, dof_t, rest_t, qn_t, xml_t = E.get_target(X2, E.CACHE_AMASS)
    g_t = E.target_geom(spec_t, dof_t, dev, sem_encoder=enc)
    dof_d = torch.as_tensor(dof_t, device=dev, dtype=torch.float64)
    rest_d = torch.as_tensor(rest_t, device=dev, dtype=torch.float64)
    W = 5 ** torch.arange(6, device=dev)
    lv = (torch.arange(9, device=dev).float() - 4.0) / 3.5

    def codes_at(r0, n):
        cp = torch.from_numpy(np.ascontiguousarray(data.c[r0:r0 + n])).long()
        pk = torch.from_numpy(np.ascontiguousarray(data.cr[r0:r0 + n])).long()
        r = torch.empty_like(cp)
        r[..., 0::2] = pk & 0x0F; r[..., 1::2] = (pk >> 4) & 0x0F
        return torch.cat([cp, r], 1)

    def q_of(tok, ep, n):
        z = ma.fsq.from_ints(((tok[:, None] // W) % 5)[None].float())[None]
        p = F.softmax(ma.decode(z, ep, n)[0], -1)
        return (p * lv).sum(-1)

    def fk_human(q):
        zp = sp.fsqp.proj_out(q[:, :128]); zr = sp.fsqr.proj_out(q[:, 128:])
        rot, _ = sp.decode(zp, zr, E.geom_batch(g_h, q.shape[0]))
        return torch.as_tensor(V.fk_pos(rot.float().cpu(), sspec)).float()

    def render_x2(q):
        zp = sp.fsqp.proj_out(q[:, :128]); zr = sp.fsqr.proj_out(q[:, 128:])
        rot, _ = sp.decode(zp, zr, E.geom_batch(g_t, q.shape[0]))
        Rg = rot6d_to_matrix(rot).double()
        _, _, qj = MP.project(Rg, spec_t, dof_d, rest=rest_d, method="global")
        gR, gp = MP.fk_from_angles(qj, spec_t, dof_d, rest=rest_d,
                                   root_R=Rg[:, 0])
        return GC.render_tile(xml_t, spec_t, qn_t, qj.cpu().numpy(),
                              gR[:, 0].cpu().numpy(), gp.cpu().numpy(),
                              qj.shape[0], status="nodataset")

    # ---------- sweep: recovery horizon ----------
    gg = np.random.default_rng(0)
    starts = {}
    for fam in FAMS:
        try:
            s, _ = kd.sample_starts(args.n_start, gg, "val", span=130,
                                    families=[fam])
            starts[fam] = [int(x) for x in s.tolist()]
        except Exception:
            pass
    curves, keep = {}, {}
    for fam, rows in starts.items():
        curves[fam] = []
        for n in NS:
            rr = []
            for i, r0 in enumerate(rows):
                c = codes_at(r0, 130)
                eph = ma.endpoints(torch.stack([c[0], c[1], c[0], c[1]])[None].to(dev))
                epo = ma.endpoints(torch.stack([c[0], c[1], c[120], c[121]])[None].to(dev))
                nf = torch.full((1,), float(n), device=dev)
                sp_ = {}
                for tag, ep in (("home", eph), ("open", epo)):
                    fks = []
                    for s_ in range(args.n_samp):
                        tk, _ = sample_nll(mb, ep, nf, dev, seed=i * 100 + s_)
                        fks.append(fk_human(q_of(tk, ep, n)))
                    f = torch.stack(fks)
                    sp_[tag] = float((f - f.mean(0, keepdim=True)).norm(dim=-1).mean())
                    if tag == "home" and n == 45:
                        keep.setdefault(fam, []).append((r0, sp_["home"], fks))
                rr.append(sp_["home"] / max(sp_["open"], 1e-6))
            curves[fam].append(float(np.mean(rr)))
        print(f"[sweep] {fam}: {[round(x,2) for x in curves[fam]]}", flush=True)

    # ---------- Fig 1: recovery-horizon curves ----------
    fig, ax = plt.subplots(figsize=(9, 5.5))
    cmap = plt.get_cmap("turbo")
    order = sorted(curves, key=lambda f: curves[f][2])
    for k, fam in enumerate(order):
        y = curves[fam]
        ax.plot(NS, y, "o-", lw=2, color=cmap(k / max(len(order) - 1, 1)),
                label=f"{fam}")
        # recovery horizon: first n where ratio >= 1
        for a, b, na, nb in zip(y, y[1:], NS, NS[1:]):
            if a < 1 <= b:
                nh = na + (1 - a) / (b - a) * (nb - na)
                ax.plot([nh], [1.0], "*", ms=16,
                        color=cmap(k / max(len(order) - 1, 1)))
                ax.annotate(f"{fam} n*={nh:.0f}", (nh, 1.0),
                            textcoords="offset points", xytext=(6, -14),
                            fontsize=8, color=cmap(k / max(len(order) - 1, 1)))
                break
    ax.axhline(1.0, color="k", ls="--", lw=1)
    ax.text(NS[0], 1.02, "ratio = 1: returning is as free as going anywhere",
            fontsize=8)
    ax.set_xlabel("n  (frames allotted to come home)")
    ax.set_ylabel("freedom ratio   spread(return) / spread(open-loop)")
    ax.set_title("POSE FREEDOM & RECOVERY HORIZON — frozen Equator, no GT, no falls")
    ax.grid(alpha=0.3); ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    p1 = f"{OUT_PNG}/homecoming_recovery_horizon.png"
    fig.savefig(p1, dpi=150); plt.close(fig)
    print("->", p1, flush=True)

    # ---------- Fig 2: fan chart, locked vs free ----------
    ranked = sorted(curves, key=lambda f: curves[f][2])
    lock_f, free_f = ranked[0], ranked[-1]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    picks = {}
    for ax, fam in zip(axes, (lock_f, free_f)):
        r0, _, fks = keep[fam][0]
        picks[fam] = (r0, fks)
        jn = list(sspec.joint_names)
        jw = jn.index("R_Wrist")
        for f in fks:
            h = f[:, jw, 2].numpy()
            ax.plot(np.arange(len(h)), h, lw=1.4, alpha=0.85)
        ax.set_title(f"{fam}   ratio={curves[fam][2]:.2f}"
                     f"   ({'LOCKED' if fam == lock_f else 'FREE'})")
        ax.set_xlabel("frame (n=45 return)"); ax.grid(alpha=0.3)
    axes[0].set_ylabel("right-wrist height (cm)")
    fig.suptitle("8 RETURN PATHS from the same pose — narrow fan = only one way home",
                 fontsize=11)
    fig.tight_layout()
    p2 = f"{OUT_PNG}/homecoming_fan_locked_vs_free.png"
    fig.savefig(p2, dpi=150); plt.close(fig)
    print("->", p2, flush=True)

    # ---------- Video: locked vs free, 3 return samples each on X2 ----------
    n = 45
    tiles, labels = [], []
    for fam in (lock_f, free_f):
        r0 = picks[fam][0]
        c = codes_at(r0, 130)
        ep = ma.endpoints(torch.stack([c[0], c[1], c[0], c[1]])[None].to(dev))
        nf = torch.full((1,), float(n), device=dev)
        g6, _, _ = build_global(l6h[torch.as_tensor(np.arange(r0, r0 + 1))],
                                sspec, dev)
        src = capsule_tile(np.repeat(V.fk_pos(g6, sspec), n, 0), sspec.parents,
                           f"SOURCE pose ({fam})")
        tiles.append(src); labels.append(f"{fam}: start pose (held)")
        for s_ in range(3):
            tk, _ = sample_nll(mb, ep, nf, dev, seed=s_ + 7)
            q = q_of(tk, ep, n)
            q[0] = (c[0].float().to(dev) - 4.0) / 3.5
            q[-1] = (c[0].float().to(dev) - 4.0) / 3.5
            tiles.append(render_x2(q))
            labels.append(f"{fam}: return #{s_+1}")
    frames = []
    for t in range(n):
        row = np.concatenate([pad_band(ti[min(t, len(ti) - 1)]) if k % 4 else
                              ti[min(t, len(ti) - 1)]
                              for k, ti in enumerate(tiles)], 1)
        im = Image.fromarray(row).convert("RGB")
        dr = ImageDraw.Draw(im)
        for k, lb in enumerate(labels):
            col = (255, 130, 130) if lb.startswith(lock_f) else (140, 230, 255)
            dr.text((k * GC.TILE_W + 8, 6), lb, font=font(12), fill=col)
        dr.text((8, GC.TILE_H + 24),
                f"curr -> n={n} -> curr.  LEFT block {lock_f} "
                f"(ratio {curves[lock_f][2]:.2f}, paths nearly identical) | "
                f"RIGHT block {free_f} (ratio {curves[free_f][2]:.2f}, paths differ)",
                font=font(12, False), fill=(215, 215, 220))
        frames.append(np.asarray(im))
    p3 = f"{OUT_MP4}/homecoming_locked_vs_free_x2.mp4"
    RN.write_mp4(p3, np.stack(frames), 20)
    print("->", p3, flush=True)
    json.dump({"curves": curves, "ns": NS, "locked": lock_f, "free": free_f},
              open(f"{RUNS}/HOMECOMING_CURVES_0811.json", "w"), indent=2)


if __name__ == "__main__":
    main()
