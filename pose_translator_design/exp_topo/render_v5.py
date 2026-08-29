"""V5 rollout videos — the FOUR zero-shot dimensions of ARCH_V5 §6, on real meshes.

  render_v5.py --run /var/tmp/alphamotion/topo_exp/runs/V5_pathint_0806

What is rendered, in every panel, without exception:
  * FK from the decoded ROTATIONS only. The decoder's raw position stream is never
    drawn (p1_fkfix: drawing it gave rubber bones and sunken heads and cost a day).
  * manifold_project onto the target robot's own mechanical manifold BEFORE the mesh,
    so every frame is executable; the pre-projection violation rate and the worst
    overshoot are burned into the frame, not hidden in a log.
  * real MJCF meshes via the validated showcase/carnival renderer.

The generation contract, for all four sections (ARCH_V5 §6): take a GT window, hand
the model the FIRST frame as the current state and the LAST as the target, let it
produce every frame in between from its PATH and TIME tokens, and put it next to GT.
Nothing is text-conditioned; this round is two-point boundary generation.

  (a) CROSS-EMBODIMENT   one token stream -> unitree_g1_29dof, agibot_x2_ultra,
      agibot_a3_t3d0, pnd_adam_lite. X2 and A3 are built straight from vendor MJCF
      and appear in NO training set and NO public retargeting dataset; G1 is in
      E19's holdout list. Same tokens, four bodies.
  (b) CROSS-MOTION       AMASS families the run never trained on, plus kimodo's
      GENERATED clips (GENMO/outputs/investor_demo) — a genuinely different
      distribution, not a held-out slice of the same one.
  (c) CROSS-SPEED        the SAME path token decoded with a DIFFERENT time token
      (taken from a fast and a slow reading of the same path), side by side.
  (d) CONSTRAINT DENSITY first+last only -> +1 middle -> +3 middle. Each extra
      constraint splits the window into more two-point problems; the completion
      should converge to GT and the printed error should fall monotonically.

  MONEY SHOT: GT | model-generated-from-endpoints, two big panels.

Every section is independently guarded — one failing body or one missing clip never
costs the other three sections.

Output: R4_VIZ/mp4/v5/  (paths printed; lloyd judges with his eyes, per ARCH_V5 §8)
"""
from __future__ import annotations
import os, sys, json, time, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v5_common as V                                        # noqa: E402
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES",
                      "/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
sys.path.insert(0, V.scratchpad())                           # showcase/carnival/agibot
import mujoco as mj                                          # noqa: E402
import showcase_all as SA                                    # noqa: E402
import carnival as CV                                        # noqa: E402
import agibot as AG                                          # noqa: E402
import prep_xwbc_cache as PX                                 # noqa: E402
from common import (rot6d_to_matrix, matrix_to_rot6d, geodesic_deg,
                    fk_global)                               # noqa: E402
from train_v5 import load_run, to_states                     # noqa: E402
from p1_verdict import ffmpeg_write, downsample, compose, cjk_font  # noqa: E402
from PIL import Image, ImageDraw                             # noqa: E402

HUM = "human_smpl"
BODIES = ["unitree_g1_29dof", "agibot_x2_ultra", "agibot_a3_t3d0", "pnd_adam_lite"]
NEW_BODIES = {"agibot_x2_ultra", "agibot_a3_t3d0"}           # no dataset anywhere
HELD_BODIES = {"unitree_g1_29dof"}                           # in E19's holdout list
LABEL = {"unitree_g1_29dof": "UNITREE G1 29DOF", "agibot_x2_ultra": "AGIBOT X2 ULTRA",
         "agibot_a3_t3d0": "AGIBOT A3 T3.0", "pnd_adam_lite": "PND ADAM LITE"}
AZIM, ELEV = 145, -10
FPS, STEP, LOOPS, SS = 15, 2, 3, 2
COL_GT = SA.COL_SRC
COL_GEN = [0.16, 0.62, 0.42, 1.0]
COL_ALT = SA.COL_HELD
RGB_NEW = (185, 15, 95)
RGB_HELD = (200, 70, 0)
RGB_TRN = (72, 78, 90)


def rgb(c):
    return tuple(int(255 * v) for v in c[:3])


# --------------------------------------------------------------------------- bodies
_BCACHE = {}


def get_body(name, cache):
    """-> dict(spec, dof, rest, slot_names, model, tag). Cache bodies come from the
    paired cache exactly as E19 saw them (so the geometry conditioning matches
    training) with the vendor MJCF used only for FK and rendering; the AgiBots are
    built from raw MJCF by agibot.build and exist in no dataset at all."""
    if name in _BCACHE:
        return _BCACHE[name]
    if name in AG.CANDIDATES and name in NEW_BODIES:
        b = CV.new_body(name)
        out = {"spec": b["spec"], "dof": b["dof"], "rest": b["rest"],
               "slot_names": b["slot_names"], "model": b["model"], "tag": "new"}
    else:
        xml = PX.resolve_xml(name)
        spec = V.load_spec(cache, name)
        z = np.load(f"{cache}/{name}_dof.npz", allow_pickle=True)
        model = mj.MjModel.from_xml_path(xml)
        assert model.jnt_type[0] == mj.mjtJoint.mjJNT_FREE, name
        AG.CANDIDATES[name] = xml                    # so render_new_panel finds it
        out = {"spec": spec, "dof": z["dof"], "rest": z["rest"],
               "slot_names": list(z["slot_joints"]), "model": model,
               "tag": "held" if name in HELD_BODIES else "train"}
    _BCACHE[name] = out
    return out


def target_entry(spec, dof, dev):
    from train_ae import static_cond
    from semantics import embed_matrix
    lut = CV.name_lut(list(spec.joint_names), device="cpu")
    return {"spec": spec, "st": static_cond(spec, dev),
            "sem": embed_matrix(spec.joint_names, lut).to(dev),
            "dof": torch.as_tensor(np.asarray(dof), dtype=torch.float32, device=dev)}


# --------------------------------------------------------------------------- panels
_TAN_H = None
_ORIG_MKCAM = SA.mkcam
PAD = [1.0]


def _mkcam_pad(height, azimuth, elev):
    c = _ORIG_MKCAM(height, azimuth, elev)
    c.distance *= PAD[0]
    return c


SA.mkcam = _mkcam_pad


def pad_for(height, r_xy, pw, ph, margin=1.06):
    tan_h = np.tan(np.radians(45 / 2)) * (pw / ph)
    base = 1.72 * max(float(height), 0.55)
    return max(1.0, margin * float(r_xy) / tan_h / base)


def human_panel(spec, pos_cm, col, pw, ph):
    """showcase capsules with the SMPL head/chest capsule fix (p1_gen.fix_caps_scene)."""
    from g1x_mesh import set_caps
    p = (pos_cm / 100.0) @ SA.AX
    p[:, :, 2] -= (p[:, :, 2].min(1, keepdims=True) - 0.042)
    height = float(p[:, :, 2].max())
    scene = SA.panel_scene(None, [], [("C", spec, col)], pw, ph)
    off = np.asarray(spec.rest_offsets, np.float64)
    for j, par in enumerate(spec.parents):
        if int(par) < 0:
            continue
        bid = mj.mj_name2id(scene, mj.mjtObj.mjOBJ_BODY, f"C_b{j}")
        gid = next((g for g in range(scene.ngeom) if scene.geom_bodyid[g] == bid), -1)
        if gid < 0:
            continue
        Ln = float(np.linalg.norm(off[j])) / 100.0
        if spec.joint_names[j] == "Head":
            scene.geom_type[gid] = mj.mjtGeom.mjGEOM_SPHERE
            scene.geom_size[gid, 0] = 0.088
        else:
            scene.geom_size[gid, 0] = float(np.clip(0.30 * Ln, 0.026, 0.045))
    d = mj.MjData(scene)
    r = SA.mk_renderer(scene, ph, pw)
    cam = SA.mkcam(height, AZIM, ELEV)
    out = np.empty((len(p), ph, pw, 3), np.uint8)
    for t in range(len(p)):
        set_caps(scene, d, "C", spec, p[t])
        mj.mj_forward(scene, d)
        r.update_scene(d, cam)
        out[t] = r.render()
    try:
        r.close()
    except Exception:
        pass
    return out, height, float(np.linalg.norm(p[:, :, :2], axis=-1).max())


def fixed_height(body, qposes, idx):
    d0 = mj.MjData(body["model"])
    hh, rr = [], []
    for qp in qposes:
        xp = []
        for t in idx:
            d0.qpos[:] = qp[t]
            mj.mj_forward(body["model"], d0)
            xp.append(d0.xpos[1:].copy())
        xp = np.asarray(xp)
        hh.append(float(xp[:, :, 2].max() - xp[:, :, 2].min()))
        rr.append(float(np.linalg.norm(xp[:, :, :2] - xp[:, :1, :2], axis=-1).max()))
    return float(np.max(hh)), float(np.max(rr))


# --------------------------------------------------------------------------- generate
@torch.no_grad()
def gen_from_endpoints(model, key, dev, T_out=None, time_from=None, hidden=None):
    """key [T,nk,9] GT window -> generated key trajectory [T_out,nk,9].

    The model sees the window (to produce its tokens) plus the first and last frames
    as the boundary condition; every intermediate frame is the model's. `time_from`
    swaps in the TIME token of another reading of the same path — that is the
    cross-speed knob and it is the whole point of having two codebooks.
    """
    w = key[None].to(dev).float()
    cur, tgt = w[:, 0], w[:, -1]
    kn = torch.ones(1, device=dev)
    e = model.encode(w, cur, tgt, kn)
    tz = e["time_z"] if time_from is None else \
        model.encode(time_from[None].to(dev).float(), cur, tgt, kn)["time_z"]
    o = model.decode(e["path_z"], tz, cur, tgt, kn, T_out or w.shape[1], hidden)
    return to_states(o, model, cur, tgt, kn)[0].float().cpu()


@torch.no_grad()
def gen_piecewise(model, key, dev, anchors):
    """Constraint-density rollout: GT frames at `anchors` are GIVEN, each consecutive
    pair is its own two-point problem, and the segments are concatenated."""
    segs = []
    for a, b in zip(anchors[:-1], anchors[1:]):
        s = gen_from_endpoints(model, key[a:b + 1], dev)
        segs.append(s if not segs else s[1:])
    return torch.cat(segs, 0)


def to_robot(tok, src_ent, tgt_ent, key_traj, kidx, body, dev, cache=None):
    """generated key trajectory -> (rot6d, qpos, limit stats) on one robot."""
    rot, _ = V.complete_from_key(tok, src_ent, tgt_ent, key_traj, kidx, dev)
    r6, q, st = CV.project_new(rot, body["spec"], body["dof"], body["rest"])
    root_R = rot6d_to_matrix(torch.as_tensor(np.asarray(rot),
                                             dtype=torch.float64))[:, 0].numpy()
    qpos, nw = CV.qpos_from_angles(body["model"], body["spec"], body["slot_names"],
                                   q.numpy(), root_R)
    return torch.as_tensor(r6), qpos, dict(st) | {"joints_written": int(nw)}


# --------------------------------------------------------------------------- compose
def build_video(path, panels, headers, title, sub, note, pw, ph, hdr=118):
    """panels: list of [T,ph,pw,3]; headers: list of (caption, colour, subline,
    tag_rgb, tag_text)."""
    n = len(panels)
    W, H = n * pw, hdr + ph
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    dr = ImageDraw.Draw(ov)
    dr.rectangle([0, 0, W, hdr], fill=(24, 27, 33, 255))
    dr.text((16, 6), title, font=cjk_font(26), fill=(255, 255, 255, 255))
    dr.text((16, 42), sub[:230], font=SA.font(13, False), fill=(190, 196, 206, 255))
    dr.text((16, 64), note[:250], font=SA.font(12, False), fill=(150, 156, 168, 255))
    dr.text((16, 84), "FK from decoded ROTATIONS only (never the raw position stream); "
                      "joint limits enforced by manifold projection before rendering.",
            font=SA.font(12, False), fill=(150, 156, 168, 255))
    for k, (cap, col, subl, tagc, tagt) in enumerate(headers):
        dr.rectangle([k * pw, hdr, k * pw + pw - 1, hdr + 30], fill=rgb(col) + (228,))
        dr.text((k * pw + pw // 2, hdr + 5), cap, font=SA.font(15),
                fill=(255, 255, 255, 255), anchor="ma")
        if tagc:
            dr.rectangle([k * pw, hdr + 30, k * pw + pw - 1, hdr + 49],
                         fill=tagc + (235,))
            dr.text((k * pw + pw // 2, hdr + 32), tagt, font=SA.font(11),
                    fill=(255, 255, 255, 255), anchor="ma")
        dr.text((k * pw + pw // 2, H - 26), subl, font=SA.font(11),
                fill=(60, 64, 72, 255), anchor="ma")
        if k:
            dr.line([k * pw, hdr, k * pw, H], fill=(200, 204, 212, 255), width=1)
    frames = compose(panels, [(k * pw, hdr) for k in range(n)], W, H, np.asarray(ov))
    mp4 = ffmpeg_write(path, frames, FPS, LOOPS)
    import imageio.v2 as imageio
    png = path.replace(".mp4", "_still.png")
    imageio.imwrite(png, frames[len(frames) // 2])
    print("MP4:", mp4, flush=True)
    return mp4, png


def limit_sub(st, extra=""):
    return (f"limits pre {st['pre_viol_dof_pct']:.2f}% (worst "
            f"{st['pre_worst_overshoot_deg']:.0f}deg) -> post "
            f"{st['post_viol_dof_pct']:.3f}%   proj {st['projection_moved_mean_deg']:.1f}deg"
            + extra)


# --------------------------------------------------------------------------- sources
def kimodo_clips(limit=3):
    import glob
    hits = sorted(glob.glob(f"{V.GEN_ROOT}/**/kimodo_smplx*.npz", recursive=True))
    seen, out = set(), []
    for p in hits:
        n = os.path.basename(os.path.dirname(p))
        if n in seen:
            continue
        seen.add(n); out.append((p, n))
        if len(out) >= limit:
            break
    return out


def kimodo_key(path, spec, kidx, reach, dev):
    """kimodo generated SMPL-X npz -> key states [T,nk,9] + FK positions cm.
    Same 22-joint order and Y-up convention as the human_smpl cache (verified in
    p1_gen), so applying its local rotations to the cache's rest offsets is exactly
    what build_global does to the cached AMASS local6d."""
    z = np.load(path, allow_pickle=True)
    lr = np.asarray(z["local_rot_mats"])
    if lr.ndim == 5:
        lr = lr[0]
    assert lr.shape[1] == len(spec.parents), (lr.shape, len(spec.parents))
    l6 = matrix_to_rot6d(torch.as_tensor(lr, dtype=torch.float32))
    gR, gp = fk_global(l6.to(dev), spec)
    key = torch.cat([matrix_to_rot6d(gR[:, kidx]), gp[:, kidx] / reach], -1).cpu()
    return key.float(), gp.cpu().numpy()


def best_window(pos_cm, T):
    n = pos_cm.shape[0]
    if n <= T:
        return 0
    sp = np.linalg.norm(np.diff(pos_cm, axis=0), axis=-1).mean(1)
    c = np.concatenate([[0.0], np.cumsum(sp)])
    return int(np.argmax(c[T - 1:] - c[:n - T + 1]))


def pick_clip(kd, fam, T, prefer_val=True):
    cand = [c for c in range(len(kd.clips))
            if kd.family[c] == fam and kd.clips[c]["len"] >= T + 4]
    if not cand:
        return None
    cand.sort(key=lambda c: (0 if (c in kd.val_clips) == prefer_val else 1,
                             -kd.clips[c]["len"]))
    return cand[0]


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--cache", default=V.AMASS)
    ap.add_argument("--out", default=V.VIZ_OUT)
    ap.add_argument("--bodies", default=",".join(BODIES))
    ap.add_argument("--only", default="", help="comma list of sections a,b,c,d,money")
    ap.add_argument("--n_motion", type=int, default=4, help="(b) AMASS clips")
    ap.add_argument("--n_gen", type=int, default=2, help="(b) kimodo clips")
    ap.add_argument("--pw", type=int, default=460)
    ap.add_argument("--ph", type=int, default=900)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tiny", action="store_true")
    args = ap.parse_args()
    if args.tiny:
        args.bodies = "unitree_g1_29dof"
        args.n_motion, args.n_gen = 1, 1
        args.pw, args.ph, args.device = 200, 380, "cpu"
    t0, dev = time.time(), args.device
    os.makedirs(args.out, exist_ok=True)
    do = set(x for x in args.only.split(",") if x) or {"a", "b", "c", "d", "money"}
    PW, PH = args.pw, args.ph

    model, cfg = load_run(args.run, dev)
    T = cfg.get("T", V.T_WIN)
    ren = np.arange(0, T, STEP)
    kd = V.KeyData(args.cache, [HUM], device=dev)
    spec_h, kidx, reach = kd.spec[HUM], kd.kidx[HUM], kd.reach[HUM]
    tok, _ = V.load_frozen(dev)
    ent_h = V.body_entry(args.cache, HUM, dev, reach)
    bodies = [b for b in args.bodies.split(",") if b]
    print(f"[run] {args.run}  T={T} budgets={model.budgets} arm={cfg.get('arm')}")
    print(f"[bodies] {bodies}", flush=True)

    ents, bods = {}, {}
    for b in bodies:
        try:
            bods[b] = get_body(b, args.cache)
            ents[b] = target_entry(bods[b]["spec"], bods[b]["dof"], dev)
            print(f"  [{b}] J={bods[b]['spec'].J} tag={bods[b]['tag']}", flush=True)
        except Exception as e:
            print(f"  [{b}] UNAVAILABLE: {type(e).__name__}: {e}", flush=True)

    index = {"run": args.run, "arm": cfg.get("arm"), "frozen": V.FROZEN,
             "contract": "first frame = current state, last frame = target state; "
                         "every frame in between is generated from the PATH and TIME "
                         "tokens", "sections": {}}
    idx_path = f"{args.out}/_index.json"

    def save():
        json.dump(index, open(idx_path, "w"), indent=1, default=str)

    def gt_and_gen(key):
        gen = gen_from_endpoints(model, key, dev)
        return gen

    def human_pos(key_full_rot6d):
        return None

    # ================================================================= (a) bodies
    if "a" in do and bods:
        try:
            cid = pick_clip(kd, "walk", T) or pick_clip(kd, "dance", T)
            s = kd.clips[cid]["start"]
            rows = np.arange(s + (kd.clips[cid]["len"] - T) // 2,
                             s + (kd.clips[cid]["len"] - T) // 2 + T)
            key = kd.key[HUM][torch.from_numpy(rows)]
            gen = gen_from_endpoints(model, key, dev)
            gt_full, _ = V.full_pose_of(args.cache, HUM, rows, dev, reach)
            gt_pos = V.fk_pos(gt_full[..., :6], spec_h)
            panels, heads, qp, stats = [], [], {}, {}
            for b in bods:
                r6, q, st = to_robot(tok, ent_h, ents[b], gen, kidx, bods[b], dev)
                qp[b] = q; stats[b] = st
            hs = {b: fixed_height(bods[b], [qp[b]], ren) for b in qp}
            PAD[0] = max([pad_for(hs[b][0], hs[b][1], PW, PH) for b in qp] +
                         [pad_for(1.7, np.abs(gt_pos[:, :, [0, 2]]).max() / 100, PW, PH)])
            fr, _, _ = human_panel(spec_h, gt_pos[ren], COL_GT, PW * SS, PH * SS)
            panels.append(downsample(fr, PH, PW, SS))
            heads.append(("HUMAN GT  ·  reference", COL_GT, "FK on GT rotations",
                          None, ""))
            for b in qp:
                fr, _ = CV.render_new_panel(b, qp[b][ren], bods[b]["model"], COL_GEN,
                                            AZIM, ELEV, PW * SS, PH * SS,
                                            fixed_height=hs[b][0])
                panels.append(downsample(fr, PH, PW, SS))
                tg = (RGB_NEW, "NOT IN ANY DATASET") if b in NEW_BODIES else \
                     ((RGB_HELD, "HELD OUT OF E19") if b in HELD_BODIES else
                      (RGB_TRN, "IN E19'S TRAINING SET"))
                heads.append((LABEL.get(b, b), COL_GEN, limit_sub(stats[b]),
                              tg[0], tg[1]))
            mp4, png = build_video(
                f"{args.out}/a_cross_embodiment.mp4", panels, heads,
                "ONE TOKEN STREAM  ->  FOUR BODIES",
                f"{kd.clips[cid]['name'][:90]}  ·  {T} frames @30fps  ·  generated "
                f"from the first and last frame only",
                "X2 Ultra and A3 T3.0 are built from vendor MJCF and appear in no "
                "training set and no public retargeting dataset.", PW, PH)
            index["sections"]["a_cross_embodiment"] = {
                "mp4": mp4, "still": png, "clip": kd.clips[cid]["name"],
                "limits": stats}
            save()
        except Exception as e:
            import traceback; traceback.print_exc()
            index["sections"]["a_cross_embodiment"] = {"error": str(e)}; save()

    # ================================================================= (b) motions
    if "b" in do and bods:
        b0 = list(bods)[0]
        fams = [f for f in ("kick", "dance", "punch", "jacks", "run", "walk")][:args.n_motion]
        jobs = []
        for f in fams:
            c = pick_clip(kd, f, T)
            if c is not None:
                jobs.append(("amass", c, f, kd.clips[c]["name"]))
        for p, n in kimodo_clips(args.n_gen):
            jobs.append(("gen", p, "generated", n))
        out_b = []
        for kind, ref, fam, name in jobs:
            try:
                if kind == "amass":
                    s = kd.clips[ref]["start"]; L = kd.clips[ref]["len"]
                    rows = np.arange(s + (L - T) // 2, s + (L - T) // 2 + T)
                    key = kd.key[HUM][torch.from_numpy(rows)]
                    gt_full, _ = V.full_pose_of(args.cache, HUM, rows, dev, reach)
                    gt_pos = V.fk_pos(gt_full[..., :6], spec_h)
                    src = f"AMASS  {name[:70]}"
                else:
                    kall, gp = kimodo_key(ref, spec_h, kidx, reach, dev)
                    o = best_window(gp, T)
                    key = kall[o:o + T]
                    gt_pos = gp[o:o + T]
                    src = f"kimodo GENERATED  {name}"
                gen = gen_from_endpoints(model, key, dev)
                pe, re = V.key_se3_err(gen[None], key[None], reach)
                r6, q, st = to_robot(tok, ent_h, ents[b0], gen, kidx, bods[b0], dev)
                r6g, qg, stg = to_robot(tok, ent_h, ents[b0], key, kidx, bods[b0], dev)
                h, rr = fixed_height(bods[b0], [q, qg], ren)
                PAD[0] = max(pad_for(h, rr, PW, PH),
                             pad_for(1.7, np.abs(gt_pos[:, :, [0, 2]]).max() / 100,
                                     PW, PH))
                fr, _, _ = human_panel(spec_h, gt_pos[ren], COL_GT, PW * SS, PH * SS)
                panels = [downsample(fr, PH, PW, SS)]
                heads = [("HUMAN GT", COL_GT, "FK on GT rotations", None, "")]
                for tag, qq, ss, col in ((f"{LABEL.get(b0,b0)}  ·  GT KEY JOINTS", qg,
                                          stg, COL_ALT),
                                         (f"{LABEL.get(b0,b0)}  ·  GENERATED", q, st,
                                          COL_GEN)):
                    f2, _ = CV.render_new_panel(b0, qq[ren], bods[b0]["model"], col,
                                                AZIM, ELEV, PW * SS, PH * SS,
                                                fixed_height=h)
                    panels.append(downsample(f2, PH, PW, SS))
                    heads.append((tag, col, limit_sub(ss), None, ""))
                mp4, png = build_video(
                    f"{args.out}/b_cross_motion_{fam}_{name[:32].replace('/','_')}.mp4",
                    panels, heads, f"CROSS-MOTION  ·  {fam.upper()}", src,
                    f"generated key-joint error vs GT: {pe:.2f} cm / {re:.1f} deg. "
                    f"Middle panel is the same pipeline fed the TRUE key joints — the "
                    f"ceiling the temporal tokens are chasing.", PW, PH)
                out_b.append({"family": fam, "source": src, "mp4": mp4, "still": png,
                              "key_pos_cm": round(pe, 2), "key_rot_deg": round(re, 1)})
                save()
            except Exception as e:
                import traceback; traceback.print_exc()
                out_b.append({"family": fam, "source": name, "error": str(e)})
        index["sections"]["b_cross_motion"] = out_b
        save()

    # ================================================================= (c) speed
    if "c" in do and bods:
        try:
            b0 = list(bods)[0]
            cid = pick_clip(kd, "punch", T) or pick_clip(kd, "kick", T) or \
                pick_clip(kd, "walk", T)
            s = kd.clips[cid]["start"]; L = kd.clips[cid]["len"]
            rows = np.arange(s + (L - T) // 2, s + (L - T) // 2 + T)
            key = kd.key[HUM][torch.from_numpy(rows)]
            # the SAME path read out at 30 / 60 / 120 frames -> three time tokens,
            # one shared path token. Decoded at the same length so the difference on
            # screen is purely the timing law.
            views = {"fast (path read in 30 frames)": kd.window(HUM, rows[:1], T // 2, T)[0],
                     "natural (60 frames)": key,
                     "slow (path read in 120 frames)": kd.window(HUM, rows[:1], 2 * T, T)[0]}
            gens, tstat = {}, {}
            for tag, vw in views.items():
                g = gen_from_endpoints(model, key, dev, T_out=T, time_from=vw)
                gens[tag] = g
                c = V.arclen_curve(g[None].float())[0]
                tstat[tag] = {"path_len": round(float(c[-1]), 3),
                              "half_way_frame": int(np.searchsorted(
                                  (c / c[-1].clamp_min(1e-6)).numpy(), 0.5))}
            qp, st = {}, {}
            for tag, g in gens.items():
                _, q, s_ = to_robot(tok, ent_h, ents[b0], g, kidx, bods[b0], dev)
                qp[tag] = q; st[tag] = s_
            h, rr = fixed_height(bods[b0], list(qp.values()), ren)
            PAD[0] = pad_for(h, rr, PW, PH)
            panels, heads = [], []
            for i, tag in enumerate(views):
                fr, _ = CV.render_new_panel(b0, qp[tag][ren], bods[b0]["model"],
                                            [COL_ALT, COL_GEN, COL_GT][i], AZIM, ELEV,
                                            PW * SS, PH * SS, fixed_height=h)
                panels.append(downsample(fr, PH, PW, SS))
                heads.append((tag.upper(), [COL_ALT, COL_GEN, COL_GT][i],
                              f"path length {tstat[tag]['path_len']:.2f} · half-way at "
                              f"frame {tstat[tag]['half_way_frame']}/{T}", None, ""))
            mp4, png = build_video(
                f"{args.out}/c_cross_speed.mp4", panels, heads,
                "SAME PATH TOKEN  ·  THREE TIME TOKENS",
                f"{kd.clips[cid]['name'][:90]}  ·  identical path and identical "
                f"endpoints; only the TIME codebook entry differs",
                "If the two codebooks are separated, these three should trace the "
                "same shape and differ only in when they get there.", PW, PH)
            index["sections"]["c_cross_speed"] = {"mp4": mp4, "still": png,
                                                  "time_stats": tstat}
            save()
        except Exception as e:
            import traceback; traceback.print_exc()
            index["sections"]["c_cross_speed"] = {"error": str(e)}; save()

    # ================================================================= (d) density
    if "d" in do and bods:
        try:
            b0 = list(bods)[0]
            cid = pick_clip(kd, "dance", T) or pick_clip(kd, "kick", T) or \
                pick_clip(kd, "walk", T)
            s = kd.clips[cid]["start"]; L = kd.clips[cid]["len"]
            rows = np.arange(s + (L - T) // 2, s + (L - T) // 2 + T)
            key = kd.key[HUM][torch.from_numpy(rows)]
            gt_full, _ = V.full_pose_of(args.cache, HUM, rows, dev, reach)
            gt_pos = V.fk_pos(gt_full[..., :6], spec_h)
            schemes = [("first + last only", [0, T - 1]),
                       ("+ 1 middle", [0, T // 2, T - 1]),
                       ("+ 3 middle", [0, T // 4, T // 2, 3 * T // 4, T - 1])]
            gens, errs, qp, st = {}, {}, {}, {}
            for tag, anch in schemes:
                g = gen_piecewise(model, key, dev, anch)
                gens[tag] = g
                pe, re = V.key_se3_err(g[None], key[None], reach)
                errs[tag] = {"n_constraints": len(anch), "key_pos_cm": round(pe, 2),
                             "key_rot_deg": round(re, 1)}
                _, q, s_ = to_robot(tok, ent_h, ents[b0], g, kidx, bods[b0], dev)
                qp[tag] = q; st[tag] = s_
            h, rr = fixed_height(bods[b0], list(qp.values()), ren)
            PAD[0] = max(pad_for(h, rr, PW, PH),
                         pad_for(1.7, np.abs(gt_pos[:, :, [0, 2]]).max() / 100, PW, PH))
            fr, _, _ = human_panel(spec_h, gt_pos[ren], COL_GT, PW * SS, PH * SS)
            panels = [downsample(fr, PH, PW, SS)]
            heads = [("HUMAN GT  ·  the target", COL_GT, "FK on GT rotations", None, "")]
            for i, (tag, _) in enumerate(schemes):
                f2, _ = CV.render_new_panel(b0, qp[tag][ren], bods[b0]["model"],
                                            [COL_ALT, COL_GEN, COL_GEN][i], AZIM, ELEV,
                                            PW * SS, PH * SS, fixed_height=h)
                panels.append(downsample(f2, PH, PW, SS))
                heads.append((tag.upper(), [COL_ALT, COL_GEN, COL_GEN][i],
                              f"{errs[tag]['key_pos_cm']:.2f} cm / "
                              f"{errs[tag]['key_rot_deg']:.1f} deg vs GT", None, ""))
            mp4, png = build_video(
                f"{args.out}/d_constraint_density.mp4", panels, heads,
                "CONSTRAINT DENSITY  ·  CONVERGING TO GROUND TRUTH",
                f"{kd.clips[cid]['name'][:90]}  ·  every extra constraint splits the "
                f"window into more two-point problems",
                "Errors should fall monotonically left to right; if they do not, the "
                "tokens are not what is carrying the motion.", PW, PH)
            index["sections"]["d_constraint_density"] = {"mp4": mp4, "still": png,
                                                         "errors": errs}
            save()
        except Exception as e:
            import traceback; traceback.print_exc()
            index["sections"]["d_constraint_density"] = {"error": str(e)}; save()

    # ================================================================= money shot
    if "money" in do and bods:
        try:
            b0 = list(bods)[0]
            cid = pick_clip(kd, "kick", T) or pick_clip(kd, "dance", T) or \
                pick_clip(kd, "walk", T)
            s = kd.clips[cid]["start"]; L = kd.clips[cid]["len"]
            rows = np.arange(s + (L - T) // 2, s + (L - T) // 2 + T)
            key = kd.key[HUM][torch.from_numpy(rows)]
            gen = gen_from_endpoints(model, key, dev)
            pe, re = V.key_se3_err(gen[None], key[None], reach)
            _, qg, sg = to_robot(tok, ent_h, ents[b0], key, kidx, bods[b0], dev)
            _, qp_, sp = to_robot(tok, ent_h, ents[b0], gen, kidx, bods[b0], dev)
            PW2, PH2 = 840, 960
            h, rr = fixed_height(bods[b0], [qg, qp_], ren)
            PAD[0] = pad_for(h, rr, PW2, PH2)
            panels, heads = [], []
            for tag, q, s_, col in (("GROUND TRUTH", qg, sg, COL_GT),
                                    ("GENERATED FROM THE TWO ENDPOINTS", qp_, sp,
                                     COL_GEN)):
                fr, _ = CV.render_new_panel(b0, q[ren], bods[b0]["model"], col, AZIM,
                                            ELEV, PW2 * SS, PH2 * SS, fixed_height=h)
                panels.append(downsample(fr, PH2, PW2, SS))
                heads.append((tag, col, limit_sub(s_), None, ""))
            mp4, png = build_video(
                f"{args.out}/MONEY_gt_vs_generated.mp4", panels, heads,
                "两个端点 + 一串 token = 中间的全部动作",
                f"{kd.clips[cid]['name'][:80]}  ·  {LABEL.get(b0,b0)}  ·  the right "
                f"panel saw only the first and last frame plus "
                f"{cfg['bit_budget']['total_bits_per_window']:.0f} bits",
                f"key-joint error {pe:.2f} cm / {re:.1f} deg vs GT.", PW2, PH2)
            index["sections"]["money"] = {"mp4": mp4, "still": png,
                                          "key_pos_cm": round(pe, 2),
                                          "key_rot_deg": round(re, 1)}
            save()
        except Exception as e:
            import traceback; traceback.print_exc()
            index["sections"]["money"] = {"error": str(e)}; save()

    save()
    print(f"\nINDEX: {idx_path}  ({time.time()-t0:.0f}s)")
    for k, v in index["sections"].items():
        if isinstance(v, dict) and "mp4" in v:
            print(f"  {k}: {v['mp4']}")
        elif isinstance(v, list):
            for x in v:
                if "mp4" in x:
                    print(f"  {k}: {x['mp4']}")
        else:
            print(f"  {k}: FAILED {v}")


if __name__ == "__main__":
    main()
