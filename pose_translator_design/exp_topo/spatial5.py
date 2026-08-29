"""spatial5 — the five-part Greenwich acceptance (skill: spatial-eval, lloyd 0810).

    spatial5.py --run PROD_c1_91k_dim20_tail_0810 --part 1|2|3|4

  1  PCA-clustered families from the OUT-OF-CORPUS bones-seed cache, 3 motions per
     cluster, each rolled out on g1/x2/adam/gr3 as a 4-tile video. Visual verdict.
  2  Cross-embodiment GT deviation (holdout bodies) + GT|recon side-by-side video.
  3  SE(3) constraints, NO refinement, four tiers (1 wrist / 2 wrists / 2 ankles /
     all 6 anchors) x three trajectory sources (martial-arts GT, hand circle,
     root sink). Rendered on X2. Sketch-through-encoder (m4 v2) only.
  4  Robot -> robot: encode one robot's real retarget, decode to another.
     g1->adam (GT exists: deviation) and adam->x2 (no-dataset body).

Part 5 (carnival merge) is assembled afterwards by carnival_v2 + ffmpeg concat.
Outputs: R4_VIZ/mp4/spatial5/<run>/*.mp4 + runs/SPATIAL5_<run>.txt (append per part).
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES",
                      "/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
os.environ.setdefault("V5_AMASS", "/var/tmp/alphamotion/topo_exp/cache_xwbc_amass_full")
import v5_common as V                                                # noqa: E402
import eval_newbody as E                                             # noqa: E402
import manifold_project as MP                                        # noqa: E402
import eval_bones_body_e19 as EB                                     # noqa: E402
import render_newbody as RN                                          # noqa: E402
import greenwich_carnival as GC                                      # noqa: E402
from common import CACHE_ROOT, rot6d_to_matrix                       # noqa: E402
from semantics import build_name_embeddings                          # noqa: E402
from PIL import Image, ImageDraw, ImageFont                          # noqa: E402

BONES = f"{CACHE_ROOT}/cache_bones_body"
FULL = "/var/tmp/alphamotion/topo_exp/cache_xwbc_amass_full"
GMR = "/var/tmp/alphamotion/config_harvest/GMR/assets"
RM = "/var/tmp/alphamotion/robot_models"
FOUR = [("Unitree G1 (heldout)", f"{GMR}/unitree_g1/g1_mocap_29dof.xml", "heldout"),
        ("AgiBot X2 (no data)", f"{RM}/agibot_x2_urdf/X2_URDF-v1.4.0/"
                                "agibot_x2_ultra_mjcf.xml", "nodataset"),
        ("PND Adam (trained)", f"{GMR}/pnd_adam_lite/adam_lite.xml", "trained"),
        ("Fourier GR3 (trained)", f"{GMR}/fourier_gr3v2_1_1/mjcf/gr3v2_1_1.xml",
         "trained")]
VIZ_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "R4_VIZ", "mp4", "spatial5")


def font(s, b=True):
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf"
                              % ("-Bold" if b else ""), s)


class Ctx:
    def __init__(self, run, dev):
        self.dev = dev
        self.run = run
        self.model, self.cfg = E.load_run(f"{V.RUNS}/{run}", dev)
        self.enc = self.cfg.get("sem_encoder", "siglip")
        self.out = f"{VIZ_ROOT}/{run}"
        os.makedirs(self.out, exist_ok=True)
        self.verdict = f"{V.RUNS}/SPATIAL5_{run}.txt"
        self.targets = {}

    def target(self, name, xml):
        if name not in self.targets:
            spec, dof, rest, qnames, xml2 = E.get_target(xml, E.CACHE_AMASS)
            g = E.target_geom(spec, dof, self.dev, sem_encoder=self.enc)
            self.targets[name] = dict(
                spec=spec, dof=dof, qnames=qnames, xml=xml2, g=g,
                dof_d=torch.as_tensor(dof, device=self.dev, dtype=torch.float64),
                rest_d=torch.as_tensor(rest, device=self.dev, dtype=torch.float64))
        return self.targets[name]

    def decode_to(self, zp, zr, t):
        rot, _ = self.model.decode(zp, zr, E.geom_batch(t["g"], zp.shape[0]))
        Rg = rot6d_to_matrix(rot).double()
        _, _, q = MP.project(Rg, t["spec"], t["dof_d"], rest=t["rest_d"],
                             method="global")
        gR, gp = MP.fk_from_angles(q, t["spec"], t["dof_d"], rest=t["rest_d"],
                                   root_R=Rg[:, 0])
        return q.cpu().numpy(), gR[:, 0].cpu().numpy(), gp.cpu().numpy()

    def say(self, *lines):
        with open(self.verdict, "a") as f:
            for l in lines:
                print(l, flush=True)
                f.write(l + "\n")


def pad_band(f_):
    """raw render_tile frame (TILE_H) -> banded height to match capsule_tile."""
    return np.pad(f_, ((0, GC.BAND), (0, 0), (0, 0)), constant_values=14)


def capsule_tile(hp_cm, parents, label):
    """human/source reference tile: positions [T,J,3] cm (Y-up) -> capsule frames."""
    import mujoco as mj
    hp = np.asarray(hp_cm) / 100.0 @ RN.AX
    hp[..., 2] -= hp[..., 2].min(axis=1)[:, None] - 0.02
    sp = mj.MjSpec()
    sp.visual.global_.offwidth = GC.TILE_W
    sp.visual.global_.offheight = GC.TILE_H
    sp.worldbody.add_light(pos=[0, -2, 3], dir=[0, 0.5, -1])
    gpl = sp.worldbody.add_geom()
    gpl.type = mj.mjtGeom.mjGEOM_PLANE; gpl.size = [8, 8, 0.1]
    gpl.rgba = [0.9, 0.9, 0.9, 1]
    m = sp.compile(); d = mj.MjData(m); mj.mj_forward(m, d)
    rend = mj.Renderer(m, height=GC.TILE_H, width=GC.TILE_W)
    cam = mj.MjvCamera(); cam.lookat[:] = [0, 0, 0.85]
    cam.distance, cam.elevation, cam.azimuth = 3.2, -8, 180
    par = np.asarray(parents)
    out = []
    for t in range(hp.shape[0]):
        rend.update_scene(d, cam)
        sc = rend.scene
        for j in range(len(par)):
            if par[j] < 0 or sc.ngeom >= sc.maxgeom:
                continue
            gg = sc.geoms[sc.ngeom]; sc.ngeom += 1
            mj.mjv_initGeom(gg, mj.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                            np.zeros(3), np.eye(3).flatten(),
                            np.array([0.30, 0.30, 0.33, 1], np.float32))
            mj.mjv_connector(gg, mj.mjtGeom.mjGEOM_CAPSULE, 0.030,
                             hp[t, par[j]], hp[t, j])
        im = Image.new("RGB", (GC.TILE_W, GC.TILE_H + GC.BAND), (14, 14, 16))
        im.paste(Image.fromarray(rend.render().copy()), (0, 0))
        dr = ImageDraw.Draw(im)
        dr.rectangle([0, GC.TILE_H, GC.TILE_W, GC.TILE_H + 4], fill=(230, 230, 235))
        dr.text((8, GC.TILE_H + 8), label, font=font(15), fill=(255, 255, 255))
        out.append(np.asarray(im))
    del rend
    return out


def grid_video(ctx, streams, path, header, fps=20, src_tile=None):
    """streams: list of (label, status, xml, spec, qnames, (q, rootR, gp)) -> 2x2."""
    tiles = []
    T = max(s[5][0].shape[0] for s in streams)
    if src_tile is not None:
        st_ = src_tile
        st_ = st_ + [st_[-1]] * (T - len(st_)) if len(st_) < T else st_[:T]
        tiles.append(st_)
    for label, status, xml, spec, qnames, (q, rootR, gp) in streams:
        fr = GC.render_tile(xml, spec, qnames, q, rootR, gp, T, status=status)
        lab = []
        col = {"trained": (70, 130, 255), "heldout": (250, 200, 60),
               "nodataset": (255, 90, 90)}[status]
        for f_ in fr:
            im = Image.new("RGB", (GC.TILE_W, GC.TILE_H + GC.BAND), (14, 14, 16))
            im.paste(Image.fromarray(f_), (0, 0))
            dr = ImageDraw.Draw(im)
            dr.rectangle([0, GC.TILE_H, GC.TILE_W, GC.TILE_H + 4], fill=col)
            dr.text((8, GC.TILE_H + 8), label, font=font(15), fill=(255, 255, 255))
            lab.append(np.asarray(im))
        tiles.append(lab)
    frames = []
    for t in range(T):
        fr = np.concatenate([ti[min(t, len(ti) - 1)] for ti in tiles], 1)
        im = Image.new("RGB", (fr.shape[1], fr.shape[0] + 36), (10, 10, 12))
        im.paste(Image.fromarray(fr), (0, 36))
        dr = ImageDraw.Draw(im)
        dr.text((10, 8), f"{header} | f{t}/{T-1}", font=font(18),
                fill=(255, 255, 255))
        frames.append(np.asarray(im))
    RN.write_mp4(path, np.stack(frames), fps)
    return path


# ---------------------------------------------------------------- part 1
@torch.no_grad()
def part1(ctx, n_clusters=3, per_cluster=3, max_T=150):
    dev = ctx.dev
    man = json.load(open(f"{BONES}/manifest.json"))
    spec_b = EB.load_spec(BONES, "soma_body")
    lut = build_name_embeddings(list(spec_b.joint_names), device=dev,
                                encoder=ctx.enc)
    entry = EB.entry(BONES, "soma_body", spec_b, dev, lut)
    entry["reach"] = man["reach_cm"]
    data = {"soma_body": entry}
    from dual_common import geom
    g9 = torch.load(f"{BONES}/soma_body_global9d.pt")
    reach = man["reach_cm"]
    pose_all = torch.cat([g9[..., :6], g9[..., 6:9] / reach], -1)
    seqs = json.load(open(f"{BONES}/sequences.json"))
    # clip embedding = mean pooled zp over subsampled frames
    embs = []
    for s in seqs:
        rows = torch.arange(s["start"], s["start"] + s["len"], 10)
        x = pose_all[rows].to(dev).float()
        zp, _ = ctx.model.encode(x, geom(data, "soma_body", len(x)))
        embs.append(zp.mean((0, 1)).cpu().numpy())
    X = np.stack(embs)
    X = X - X.mean(0)
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    P = PCA(min(16, len(seqs) - 1)).fit_transform(X)
    km = KMeans(n_clusters, n_init=8, random_state=0).fit(P)
    g = np.random.default_rng(3)
    ctx.say("=" * 88, f"PART 1 — zero-shot families (bones-seed, PCA k={n_clusters})"
            f" on 4 bodies", "=" * 88)
    for cl in range(n_clusters):
        idx = [i for i in range(len(seqs)) if km.labels_[i] == cl]
        pick = g.choice(idx, min(per_cluster, len(idx)), replace=False)
        fams = sorted({seqs[i].get("family", "?") for i in idx})
        ctx.say(f"[cluster {cl}] {len(idx)} clips, families seen inside: {fams}")
        for i in pick:
            s = seqs[i]
            T = min(s["len"], max_T)
            rows = torch.arange(s["start"], s["start"] + T)
            x = pose_all[rows].to(dev).float()
            zp, zr = ctx.model.encode(x, geom(data, "soma_body", T))
            streams = []
            for label, xml, status in FOUR:
                t = ctx.target(label, xml)
                streams.append((label, status, t["xml"], t["spec"], t["qnames"],
                                ctx.decode_to(zp, zr, t)))
            src_t = capsule_tile(g9[rows][..., 6:9].numpy(), spec_b.parents,
                                 "SOURCE (bones-seed human)")
            p = f"{ctx.out}/p1_c{cl}_{s['name'][:40]}.mp4"
            grid_video(ctx, streams, p,
                       f"PART1 zero-shot family c{cl} | {s['name'][:60]}",
                       src_tile=src_t)
            ctx.say(f"  -> {p}")


# ---------------------------------------------------------------- part 2
@torch.no_grad()
def part2(ctx, T=150):
    """holdout-body GT vs recon, numbers + the mandatory visual gate."""
    dev = ctx.dev
    from train_global import build_global
    seqs = json.load(open(f"{FULL}/sequences.json"))
    rng = np.random.default_rng(V.SPLIT_SEED)
    val = sorted(set(rng.permutation(len(seqs))[: len(seqs) // 10].tolist()))
    s = seqs[val[3]]
    rows = torch.arange(s["start"], s["start"] + min(s["len"], T))
    sspec = V.load_spec(FULL, "human_smpl")
    sdof = np.load(f"{FULL}/human_smpl_dof.npz")["dof"]
    sg = E.target_geom(sspec, sdof, dev, sem_encoder=ctx.enc)
    l6h = torch.load(f"{FULL}/human_smpl_paired_local6d.pt", mmap=True)[rows]
    gr6, gp, reach = build_global(l6h, sspec, dev)
    pose = torch.cat([gr6, gp / reach], -1).to(dev).float()
    zp, zr = ctx.model.encode(pose, E.geom_batch(sg, pose.shape[0]))
    ctx.say("=" * 88, f"PART 2 — holdout GT deviation + visual gate | {s['name']}",
            "=" * 88)
    for label, xml, emb in (("Unitree G1 (heldout)",
                             f"{GMR}/unitree_g1/g1_mocap_29dof.xml",
                             "unitree_g1_29dof"),):
        t = ctx.target(label, xml)
        q, rootR, gp_t = ctx.decode_to(zp, zr, t)
        spec_t = V.load_spec(FULL, emb)
        l6t = torch.load(f"{FULL}/{emb}_paired_local6d.pt", mmap=True)[rows]
        gr6t, _, _ = build_global(l6t, spec_t, dev)
        Rg = rot6d_to_matrix(gr6t.to(dev)).double()
        qg, _ = MP.fit_angles(Rg, t["spec"], t["dof_d"], rest=t["rest_d"],
                              clamp=False, method="greedy")
        gRt, gpt = MP.fk_from_angles(qg, t["spec"], t["dof_d"], rest=t["rest_d"],
                                     root_R=Rg[:, 0])
        gt = (qg.cpu().numpy(), gRt[:, 0].cpu().numpy(), gpt.cpu().numpy())
        d = np.linalg.norm(gp_t - gt[2], axis=-1)
        frA = GC.render_tile(t["xml"], t["spec"], t["qnames"], *gt,
                             gt[0].shape[0], status="heldout")
        frB = GC.render_tile(t["xml"], t["spec"], t["qnames"], q, rootR, gp_t,
                             q.shape[0], status="heldout")
        src_t = capsule_tile(V.fk_pos(gr6, sspec), sspec.parents, "SOURCE human")
        frames = []
        for k2, (a, b) in enumerate(zip(frA, frB)):
            sfr = src_t[min(k2, len(src_t) - 1)]
            im = Image.fromarray(np.concatenate([sfr, pad_band(a), pad_band(b)], 1)).convert("RGB")
            dr = ImageDraw.Draw(im)
            dr.text((GC.TILE_W + 8, 4), "GT retarget", font=font(14), fill=(255, 255, 255))
            dr.text((2 * GC.TILE_W + 8, 4), "recon (full-body)", font=font(14),
                    fill=(255, 255, 255))
            frames.append(np.asarray(im))
        p = f"{ctx.out}/p2_{emb}_gt_vs_recon.mp4"
        RN.write_mp4(p, np.stack(frames), 20)
        ctx.say(f"[{emb}] FK dev vs GT: mean {d.mean():.2f}cm "
                f"p95 {np.percentile(d,95):.2f}cm -> {p}")


# ---------------------------------------------------------------- part 3 helpers
def sketch_encode(ctx, x_full, vis_idx, sspec, sg):
    """SE3 sketch: only vis_idx joints enter the encoder (m4 v2 pathway)."""
    x_full = x_full.to(ctx.dev).float()
    B = x_full.shape[0]
    x = torch.zeros_like(x_full)
    vi = torch.as_tensor(sorted(set(vis_idx)), dtype=torch.long)
    x[:, vi] = x_full[:, vi]
    jm = torch.ones(B, sspec.J, dtype=torch.bool, device=x.device)
    jm[:, vi] = False
    return ctx.model.encode(x, E.geom_batch(sg, B), joint_mask=jm)


@torch.no_grad()
def part3(ctx, T=120):
    dev = ctx.dev
    kd = V.KeyData(V.AMASS, ["human_smpl"], device="cpu")
    sspec = V.load_spec(FULL, "human_smpl")
    sdof = np.load(f"{FULL}/human_smpl_dof.npz")["dof"]
    sg = E.target_geom(sspec, sdof, dev, sem_encoder=ctx.enc)
    names = [n.lower() for n in sspec.joint_names]
    root = int(np.where(np.asarray(sspec.parents) < 0)[0][0])
    wl = [i for i, n in enumerate(names) if n == "l_wrist"]
    wr = [i for i, n in enumerate(names) if n == "r_wrist"]
    al = [i for i, n in enumerate(names) if n == "l_ankle"]
    ar = [i for i, n in enumerate(names) if n == "r_ankle"]
    head = [i for i, n in enumerate(names) if n == "head"]
    TIERS = [("1wrist", wr), ("2wrists", wl + wr), ("2ankles", al + ar),
             ("6anchors", [root] + head + wl + wr + al + ar)]
    # --- trajectory sources ---
    g = np.random.default_rng(11)
    st, cid = kd.sample_starts(1, g, "val", span=T, families=["punch"])
    martial = V.full_pose_of(FULL, "human_smpl", list(range(int(st[0]),
                             int(st[0]) + T)), dev, reach=kd.reach["human_smpl"])[0]
    stand = martial[0:1].clone()
    circle = stand.expand(T, -1, -1).clone()
    th = torch.linspace(0, 4 * np.pi, T)
    base = circle[0, wr[0], 6:9].clone()
    circle[:, wr[0], 6] = base[0] + 0.25 * torch.cos(th)
    circle[:, wr[0], 7] = base[1] + 0.25 * torch.sin(th)
    sink = stand.expand(T, -1, -1).clone()
    dz = 0.22 * (0.5 - 0.5 * torch.cos(torch.linspace(0, 2 * np.pi, T)))
    sink[:, root, 7] = sink[:, root, 7] - dz            # y-up: lower root height
    SRC = [("martialGT", martial), ("handcircle", circle), ("rootsink", sink)]
    t = ctx.target(*[(l, x) for l, x, _ in FOUR][1][:2])  # X2
    xml, spec = t["xml"], t["spec"]
    ctx.say("=" * 88, "PART 3 — SE(3) constraint tiers on X2, sketch-through-encoder,"
            " NO refinement", "=" * 88)
    for sname, src in SRC:
        for tname, vis in TIERS:
            zp, zr = sketch_encode(ctx, src, vis, sspec, sg)
            q, rootR, gp = ctx.decode_to(zp, zr, t)
            fr = GC.render_tile(t["xml"], t["spec"], t["qnames"], q, rootR, gp,
                                q.shape[0], status="nodataset")
            hp = V.fk_pos(src[..., :6].cpu(), sspec)
            src_t = capsule_tile(hp, sspec.parents,
                                 f"SOURCE constraint ({sname})")
            rowf = [np.concatenate([src_t[min(i, len(src_t) - 1)], pad_band(f_)], 1)
                    for i, f_ in enumerate(fr)]
            p = f"{ctx.out}/p3_{sname}_{tname}.mp4"
            RN.write_mp4(p, np.stack(rowf), 20)
            ctx.say(f"[{sname} | {tname}] visible={len(vis)} -> {p}")


# ---------------------------------------------------------------- part 4
@torch.no_grad()
def part4(ctx, T=150):
    dev = ctx.dev
    from train_global import build_global
    seqs = json.load(open(f"{FULL}/sequences.json"))
    rng = np.random.default_rng(V.SPLIT_SEED)
    val = set(rng.permutation(len(seqs))[: len(seqs) // 10].tolist())
    vi = sorted(val)[7]
    s = seqs[vi]
    rows = torch.arange(s["start"], s["start"] + min(s["len"], T))
    ctx.say("=" * 88, f"PART 4 — robot->robot transfer | clip {s['name']}", "=" * 88)
    pairs = [("unitree_g1_29dof", f"{GMR}/unitree_g1/g1_mocap_29dof.xml",
              "PND Adam (trained)", "pnd_adam_lite"),
             ("pnd_adam_lite", f"{GMR}/pnd_adam_lite/adam_lite.xml",
              "AgiBot X2 (no data)", None)]
    for src_emb, src_xml, tgt_label, tgt_emb in pairs:
        spec_s = V.load_spec(FULL, src_emb)
        l6 = torch.load(f"{FULL}/{src_emb}_paired_local6d.pt", mmap=True)[rows]
        gr6, gp, reach = build_global(l6, spec_s, dev)
        pose = torch.cat([gr6, gp / reach], -1).to(dev).float()
        lut = build_name_embeddings(list(spec_s.joint_names), device=dev,
                                    encoder=ctx.enc)
        from train_ae import static_cond
        from semantics import embed_matrix
        sdof_s = np.load(f"{FULL}/{src_emb}_dof.npz")["dof"]
        g_s = E.target_geom(spec_s, sdof_s, dev, sem_encoder=ctx.enc)
        zp, zr = ctx.model.encode(pose, E.geom_batch(g_s, pose.shape[0]))
        tmap = {l: (l, x) for l, x, _ in FOUR}
        t = ctx.target(*tmap[tgt_label])
        q, rootR, gp_t = ctx.decode_to(zp, zr, t)
        line = f"[{src_emb} -> {tgt_label}]"
        if tgt_emb:                                     # GT exists: deviation
            spec_t = V.load_spec(FULL, tgt_emb)
            l6t = torch.load(f"{FULL}/{tgt_emb}_paired_local6d.pt", mmap=True)[rows]
            gr6t, _, _ = build_global(l6t, spec_t, dev)
            gt_fk = V.fk_pos(gr6t, spec_t)
            names_t = [n.lower() for n in spec_t.joint_names]
            wlist = [i for i, n in enumerate(names_t) if "wrist" in n or "hand" in n]
            d = np.linalg.norm(gp_t[:, wlist] - gt_fk[:, wlist], axis=-1)
            line += f" wrist vs GT: mean {d.mean():.2f}cm p95 {np.percentile(d,95):.2f}cm"
        fr = GC.render_tile(t["xml"], t["spec"], t["qnames"], q, rootR, gp_t,
                            q.shape[0], status="trained")
        src_t = capsule_tile(V.fk_pos(gr6.cpu(), spec_s), spec_s.parents,
                             f"SOURCE robot ({src_emb})")
        fr = [np.concatenate([src_t[min(i, len(src_t) - 1)], pad_band(f_)], 1)
              for i, f_ in enumerate(fr)]
        p = f"{ctx.out}/p4_{src_emb}_to_{tgt_label.split()[0]}.mp4"
        RN.write_mp4(p, np.stack(fr), 20)
        ctx.say(line + f" -> {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="PROD_c1_91k_dim20_tail_0810")
    ap.add_argument("--part", type=int, required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    ctx = Ctx(args.run, args.device)
    {1: part1, 2: part2, 3: part3, 4: part4}[args.part](ctx)


if __name__ == "__main__":
    main()
