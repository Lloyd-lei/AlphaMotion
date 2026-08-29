"""NIGHT2 A/B video: soft SE(3) constraint mode (root + left wrist commanded, all
other joints invented) — E19_K128_0805 control vs the NIGHT2 verdict winner
(fallback E28_resid_0807), SAME frames, real Unitree G1 mesh, wrist target
crosshair, per-frame SOFT wrist error printed in-frame. Deliberately NO hard pin:
the whole point is how far the raw tokenizer output lands from the command.

Frames: a contiguous window from cache_xwbc_amass (E19/E28/E29 trained only on
LAFAN -> the clip is unseen; unitree_g1_29dof is additionally a holdout body).
Reach is pinned to the LAFAN cache's G1 value (the models' training-time scale).

Output: R4_VIZ/mp4/night2/night2_constraint_ab.mp4 (+ still + metrics json).
"""
from __future__ import annotations
import os, sys, json, argparse, subprocess
import numpy as np
import torch

EXP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXP)
NIGHT2_LIB = "/var/tmp/alphamotion/topo_exp/night2_lib"
if os.path.isdir(NIGHT2_LIB):
    sys.path.append(NIGHT2_LIB)
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES",
                      "/usr/share/glvnd/egl_vendor.d/10_nvidia.json")

import mujoco as mj                                                    # noqa: E402
from common import SkeletonSpec, fk_global, matrix_to_rot6d           # noqa: E402
from train_ae import static_cond                                       # noqa: E402
from dual_common import build_model, geom                              # noqa: E402
from dof_head import build_kin                                         # noqa: E402
from semantics import build_name_embeddings, embed_matrix              # noqa: E402
import showcase_all as SA                                              # noqa: E402
from g1x_mesh import G1XwbcMapper, G1_XML, AX                          # noqa: E402

ROOT = "/var/tmp/alphamotion/topo_exp"
RUNS = f"{ROOT}/runs"
LAFAN = f"{ROOT}/cache_xwbc_lafan"
AMASS = f"{ROOT}/cache_xwbc_amass"
G1 = "unitree_g1_29dof"
WRIST = "left_rubber_hand"
OUT = ('/home/arenalabs/Desktop/"be water, robot"/pose_translator_design/'
       'R4_VIZ/mp4/night2')
CONTROL = "E19_K128_0805"
FALLBACKS = ["E28_resid_0807", "E29_wristaug_0807"]

W, H, HDR, FTR = 1600, 960, 118, 62
PW, PH = W // 2, H - HDR - FTR
COL_A = [0.62, 0.65, 0.70, 1.0]           # control: steel grey
COL_B = [0.95, 0.55, 0.15, 1.0]           # winner: orange
CROSS = (0.10, 0.80, 0.25, 1.0)           # commanded wrist: green crosshair
BG = (247, 248, 250)


def pick_run():
    """Verdict winner if finished, else the first finished fallback."""
    winner = None
    vp = f"{RUNS}/NIGHT2_verdict.json"
    if os.path.exists(vp):
        try:
            winner = json.load(open(vp)).get("winner")
        except Exception:
            winner = None
    for cand in ([winner] if winner else []) + FALLBACKS:
        if cand and os.path.exists(f"{RUNS}/{cand}/model.pt"):
            return cand
    raise SystemExit("night2_render: no finished E28/E29 run to compare against E19")


def load_g1(device):
    """G1 conditioning from the LAFAN cache (the models' training-time geometry,
    semantics, dof and reach)."""
    d = np.load(f"{LAFAN}/{G1}_spec.npz", allow_pickle=True)
    spec = SkeletonSpec(str(d["name"]), d["parents"], d["rest_offsets"], list(d["joint_names"]))
    lut = build_name_embeddings(list(spec.joint_names), device="cpu")
    entry = {"spec": spec, "st": static_cond(spec, device),
             "sem": embed_matrix(spec.joint_names, lut).to(device),
             "dof": torch.as_tensor(np.load(f"{LAFAN}/{G1}_dof.npz")["dof"],
                                    dtype=torch.float32, device=device),
             "kin": build_kin(LAFAN, G1, spec, device=device)}
    # reach: LAFAN value (training-time normalization). Computing it needs the max
    # over all rows; cheap chunked pass over the LAFAN rows.
    l6 = torch.load(f"{LAFAN}/{G1}_paired_local6d.pt")
    mx = 0.0
    for i in range(0, l6.shape[0], 65536):
        _, gp = fk_global(l6[i:i + 65536], spec)
        mx = max(mx, float(gp.norm(dim=-1).max()))
    entry["reach"] = max(mx, 1e-6)
    return entry


def pick_window(l6, spec, jw, T, n_cand=200, seed=11):
    """Contiguous T-frame window with lively wrist motion; windows containing a
    wrist jump > 20 cm between probe frames (clip boundary) are rejected."""
    N = l6.shape[0]
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, N - T, n_cand)
    probe_t = np.arange(0, T, 6)
    rows = (starts[:, None] + probe_t[None, :]).reshape(-1)
    _, gp = fk_global(l6[torch.as_tensor(rows)], spec)
    wp = gp[:, jw].reshape(n_cand, len(probe_t), 3).numpy()
    jump = np.linalg.norm(np.diff(wp, axis=1), axis=-1).max(1)
    spread = wp.std(1).sum(-1)
    spread[jump > 20.0] = -1.0
    k = int(np.argmax(spread))
    assert spread[k] > 0, "no smooth window found"
    return int(starts[k])


@torch.no_grad()
def constraint_decode(run, data, x, mask, device):
    rd = run if os.path.isabs(run) else f"{RUNS}/{run}"
    cfg = json.load(open(f"{rd}/config.json"))
    model = build_model(cfg, device)
    model.load_state_dict(torch.load(f"{rd}/model.pt", map_location=device), strict=True)
    model.eval()
    outs = []
    for i in range(0, x.shape[0], 128):
        xs, ms = x[i:i + 128], mask[i:i + 128]
        g = geom(data, G1, len(xs))
        zp, zr = model.encode(xs, g, joint_mask=ms)
        obs = vis = None
        if getattr(model, "vis_residual", False):       # same-skeleton decode -> residual
            obs, vis = xs, ~ms
        prot, _ = model.decode(zp, zr, g, obs=obs, vis=vis)
        outs.append(prot.float().cpu())
    del model
    return torch.cat(outs), bool(cfg.get("vis_residual", False))


def marker_scene(xml, pw, ph):
    """G1 MJCF + studio sky + one green crosshair (x2_core.marker_scene recipe)."""
    scene = mj.MjSpec()
    scene.visual.global_.offwidth = max(pw, 1920)
    scene.visual.global_.offheight = max(ph, 1920)
    scene.visual.headlight.ambient = [0.45, 0.45, 0.45]
    scene.visual.headlight.diffuse = [0.6, 0.6, 0.6]
    child = mj.MjSpec.from_file(xml)
    for t in list(child.textures):
        if t.type == mj.mjtTexture.mjTEXTURE_SKYBOX:
            child.delete(t)
    f = scene.worldbody.add_frame()
    f.attach_body(child.worldbody.first_body(), "P_", "")
    tex = scene.add_texture()
    tex.name = "sky"; tex.type = mj.mjtTexture.mjTEXTURE_SKYBOX
    tex.builtin = mj.mjtBuiltin.mjBUILTIN_GRADIENT
    tex.width, tex.height = 256, 256
    tex.rgb1 = [0.72, 0.77, 0.85]; tex.rgb2 = [0.96, 0.965, 0.97]
    b = scene.worldbody.add_body(name="TGT")
    jt = b.add_freejoint(); jt.name = "TGT"
    for ax in range(3):
        v = np.zeros(3); v[ax] = 0.12
        g = b.add_geom()
        g.type = mj.mjtGeom.mjGEOM_CAPSULE
        g.size[0] = 0.006
        g.fromto = [*(-v), *v]
        g.rgba = list(CROSS); g.contype = 0; g.conaffinity = 0
    g = b.add_geom()
    g.type = mj.mjtGeom.mjGEOM_SPHERE
    g.size[0] = 0.024
    g.rgba = [CROSS[0], CROSS[1], CROSS[2], 0.55]
    g.contype = 0; g.conaffinity = 0
    scene.worldbody.add_light(pos=[2.5, -3, 4], dir=[-0.4, 0.6, -1])
    scene.worldbody.add_light(pos=[-2.5, -3, 4], dir=[0.4, 0.6, -1])
    g = scene.worldbody.add_geom()
    g.type = mj.mjtGeom.mjGEOM_PLANE
    g.size = [12, 12, 0.1]
    g.rgba = [0.90, 0.91, 0.93, 1]
    return scene.compile()


def render_panel(mapper, g6, tgt_cm, col, azim, elev, pw, ph, fixed_height=None,
                 cam_dist=1.55, cam_look=0.55):
    """One robot + wrist crosshair. tgt_cm [T,3] root-relative cache cm."""
    q, resid = mapper.qpos_from_global(np.asarray(g6))
    xp = mapper.fk(q)
    lift = SA.sole_lift(mapper.m, q, mapper.root_adr, mapper.body)
    z = -xp[:, :, 2].min(1) + lift
    height = float((xp[:, :, 2] + z[:, None]).max()) if fixed_height is None else fixed_height
    scene = marker_scene(mapper.xml, pw, ph)
    SA.tint_all(scene, "P_", col)
    d = mj.MjData(scene)
    jf = SA.root_free_joint(scene, "P_")
    tadr = int(scene.jnt_qposadr[mj.mj_name2id(scene, mj.mjtObj.mjOBJ_JOINT, "TGT")])
    tgt_m = (np.asarray(tgt_cm, np.float64) / 100.0) @ AX          # Y-up cm -> Z-up m
    r = SA.mk_renderer(scene, ph, pw)
    cam = mj.MjvCamera()
    cam.lookat[:] = [0, 0, cam_look * height]
    cam.distance = cam_dist * max(height, 0.55)
    cam.elevation = elev; cam.azimuth = azim
    out = np.empty((len(q), ph, pw, 3), np.uint8)
    a = int(scene.jnt_qposadr[jf])
    for t in range(len(q)):
        d.qpos[a:a + 3] = [0.0, 0.0, z[t]]
        d.qpos[a + 3:a + 7] = q[t, mapper.root_adr + 3:mapper.root_adr + 7]
        for j in range(1, mapper.m.njnt):
            nm = mj.mj_id2name(mapper.m, mj.mjtObj.mjOBJ_JOINT, j)
            jid = mj.mj_name2id(scene, mj.mjtObj.mjOBJ_JOINT, "P_" + nm)
            if jid >= 0:
                d.qpos[scene.jnt_qposadr[jid]] = q[t, mapper.m.jnt_qposadr[j]]
        d.qpos[tadr:tadr + 3] = tgt_m[t] + [0, 0, z[t]]
        d.qpos[tadr + 3:tadr + 7] = [1, 0, 0, 0]
        mj.mj_forward(scene, d)
        r.update_scene(d, cam)
        out[t] = r.render()
    try:
        r.close()
    except Exception:
        pass
    return out, height, float(resid.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=240)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--azimuth", type=float, default=205)
    ap.add_argument("--elev", type=float, default=-10)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--probe", action="store_true", help="3 stills instead of the mp4")
    ap.add_argument("--winner", default=None, help="override the verdict winner (smoke tests)")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = args.device
    T = args.frames

    winner = args.winner or pick_run()
    wname = os.path.basename(winner.rstrip("/"))         # display/file-safe name
    print(f"[night2_render] E19 vs {wname}", flush=True)
    data = {G1: load_g1(device)}
    spec, reach = data[G1]["spec"], data[G1]["reach"]
    jw = list(spec.joint_names).index(WRIST)
    root = int(np.where(spec.parents < 0)[0][0])

    # AMASS spec must match (same robot, different clips)
    d = np.load(f"{AMASS}/{G1}_spec.npz", allow_pickle=True)
    assert list(d["joint_names"]) == list(spec.joint_names)
    l6 = torch.load(f"{AMASS}/{G1}_paired_local6d.pt")
    s0 = pick_window(l6, spec, jw, T)
    gR, gp = fk_global(l6[s0:s0 + T], spec)
    x = torch.cat([matrix_to_rot6d(gR), gp / reach], -1).float().to(device)
    gt_cm = gp.numpy()                                   # [T,J,3] root-relative cm
    print(f"[clip] AMASS rows {s0}..{s0 + T} wrist spread ok, reach={reach:.1f}cm", flush=True)

    mask = torch.ones(T, spec.J, dtype=torch.bool, device=device)
    mask[:, [root, jw]] = False                          # encoder sees ONLY root + wrist
    preds, is_resid = {}, {}
    for run in [CONTROL, winner]:
        preds[run], is_resid[run] = constraint_decode(run, data, x, mask, device)
        print(f"[decode] {run} done (vis_residual={is_resid[run]})", flush=True)

    # soft wrist error from the model's own FK (what the mesh shows, before any solver)
    err = {}
    for run, g6 in preds.items():
        p = SA.fk_from_global_rot(g6.numpy(), spec)      # [T,J,3] cm
        err[run] = np.linalg.norm(p[:, jw] - gt_cm[:, jw], axis=-1)

    mapper = G1XwbcMapper(spec, G1_XML)
    frA, hA, residA = render_panel(mapper, preds[CONTROL].numpy(), gt_cm[:, jw], COL_A,
                                   args.azimuth, args.elev, PW, PH)
    frB, _, residB = render_panel(mapper, preds[winner].numpy(), gt_cm[:, jw], COL_B,
                                  args.azimuth, args.elev, PW, PH, fixed_height=hA)
    print(f"[render] panels done (chain-fit resid {residA:.2f}/{residB:.2f} deg)", flush=True)

    from PIL import Image, ImageDraw
    F = SA.font
    tagB = wname + (" — residual decode" if is_resid[winner] else "")

    def frame(t):
        canvas = np.empty((H, W, 3), np.uint8); canvas[:] = BG
        canvas[HDR:HDR + PH, 0:PW] = frA[t]
        canvas[HDR:HDR + PH, PW:W] = frB[t]
        canvas[HDR:HDR + PH, PW - 2:PW + 2] = (24, 27, 33)
        im = Image.new("RGBA", (W, H), (0, 0, 0, 0)); dr = ImageDraw.Draw(im)
        dr.rectangle([0, 0, W, HDR], fill=(24, 27, 33, 255))
        dr.text((22, 8), "SOFT SE(3) CONSTRAINT — ROOT + LEFT WRIST COMMANDED, BODY INVENTED",
                font=F(29), fill=(255, 255, 255, 255))
        dr.text((23, 48), "Unitree G1 (held-out body), unseen AMASS clip. GREEN CROSSHAIR = the "
                          "commanded wrist. NO hard pin anywhere —", font=F(16, False),
                fill=(196, 202, 212, 255))
        dr.text((23, 72), f"the printed error is the raw tokenizer output; the {wname} question is "
                          "whether residual decode collapses it.", font=F(16, False),
                fill=(196, 202, 212, 255))
        dr.rectangle([14, HDR + 12, 380, HDR + 44], fill=(70, 76, 86, 230))
        dr.text((26, HDR + 17), f"{CONTROL} (control)", font=F(20), fill=(255, 255, 255, 255))
        dr.rectangle([PW + 14, HDR + 12, PW + 620, HDR + 44], fill=(190, 95, 15, 230))
        dr.text((PW + 26, HDR + 17), tagB, font=F(20), fill=(255, 255, 255, 255))
        for x0, run in ((26, CONTROL), (PW + 26, winner)):
            dr.text((x0, HDR + 56), f"wrist soft error  {err[run][t]:6.2f} cm",
                    font=F(24), fill=(30, 34, 40, 255))
        dr.rectangle([0, H - FTR, W, H], fill=(238, 240, 243, 255))
        dr.text((22, H - FTR + 8),
                f"mean wrist soft error:  {CONTROL} {err[CONTROL].mean():.2f} cm   vs   "
                f"{wname} {err[winner].mean():.2f} cm   (max {err[CONTROL].max():.1f} / "
                f"{err[winner].max():.1f} cm)", font=F(18), fill=(50, 56, 66, 255))
        dr.text((22, H - FTR + 34),
                "same codes-only completion everywhere else; only the commanded joints differ "
                "in decode path", font=F(14, False), fill=(110, 116, 126, 255))
        ov = np.asarray(im)
        a_ = ov[..., 3:4].astype(np.float32) / 255.0
        return (canvas * (1 - a_) + ov[..., :3] * a_).astype(np.uint8)

    stem = f"night2_constraint_ab_{CONTROL}_vs_{wname}"
    meta = {"control": CONTROL, "winner": wname, "amass_rows": [int(s0), int(s0 + T)],
            "mean_wrist_soft_err_cm": {r: round(float(err[r].mean()), 2) for r in err},
            "max_wrist_soft_err_cm": {r: round(float(err[r].max()), 2) for r in err},
            "chain_fit_resid_deg": {CONTROL: round(residA, 3), winner: round(residB, 3)}}
    if args.probe:
        import imageio.v2 as imageio
        for f_ in (0.1, 0.5, 0.9):
            t = int(f_ * (T - 1))
            p = f"{args.out}/_probe_{stem}_{int(f_ * 100)}.png"
            imageio.imwrite(p, frame(t)); print("PROBE", p, flush=True)
        print(json.dumps(meta, indent=1))
        return
    mp4 = f"{args.out}/{stem}.mp4"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(args.fps), "-i", "-", "-an", "-c:v", "libx264",
           "-preset", "slow", "-crf", "26", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", mp4]
    pr = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for t in range(T):
        pr.stdin.write(np.ascontiguousarray(frame(t)).tobytes())
    pr.stdin.close(); assert pr.wait() == 0
    import imageio.v2 as imageio
    png = f"{args.out}/still_{stem}.png"
    imageio.imwrite(png, frame(T // 2))
    meta["mp4"] = mp4; meta["png"] = png
    json.dump(meta, open(f"{args.out}/_metrics_{stem}.json", "w"), indent=1)
    print("MP4", mp4, "\nPNG", png, flush=True)


if __name__ == "__main__":
    main()
