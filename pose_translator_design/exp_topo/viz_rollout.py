"""E1 duel VIDEO: AR (arm A) vs chunked latent diffusion (arm B) rolling out over
time, decoded onto the REAL Unitree G1 mesh.

Same protocol as eval_rollout.py (56-frame context -> 32 generated frames, same
frozen tokenizer, same val split), but instead of numbers it renders 4 mesh panels:
    GT | AR pred | DIFFUSION pred | GT (reference)
During the context all prediction panels show the tokenizer's cross-embodiment
decode of the SAME context codes, so the panels are identical until generation
starts; the divergence you see afterwards is purely the temporal model.

Arm B here uses v-parameterization (the E1B_vpred checkpoint), unlike
eval_rollout.py's eps-branch.

Usage:
  python viz_rollout.py --seq_pick 0 1
"""
from __future__ import annotations
import os, sys, json, math, time, argparse
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", "/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
import mujoco as mj
from common import rot6d_to_matrix, geodesic_deg
from dual_common import load_paired_data, build_model, geom
from train_ar_lm import ARLM, dequant
from train_latent_diffusion import ChunkDenoiser, cosine_alphas, CHUNK
from g1_qpos import G1QposMapper, build_scene, set_robot, STAND_H, ARDY_G1_XML

CTX, HOR = 56, 32
RUNS = "/var/tmp/alphamotion/topo_exp/runs"
COL = {"gt": [0.55, 0.57, 0.60, 1], "ar": [0.95, 0.55, 0.15, 1], "df": [0.25, 0.75, 0.35, 1]}
TXT = {"gt": (200, 205, 212), "ar": (250, 160, 60), "df": (90, 210, 110)}


# ------------------------------------------------------------------ models ----
def load_all(device, tokenizer, tokens, runA, runB):
    tcfg = json.load(open(f"{tokenizer}/config.json"))
    tmeta = json.load(open(f"{tokens}/meta.json"))
    torch.manual_seed(0)
    data, names, _ = load_paired_data(tmeta["cache"], device)
    tok = build_model(tcfg, device)
    tok.load_state_dict(torch.load(f"{tokenizer}/model.pt", map_location=device), strict=False)
    tok.eval()
    F = torch.load(f"{tokens}/soma77_f.pt").float().to(device)     # [N,2,K,f] pre-round
    I = torch.load(f"{tokens}/soma77_ints.pt").long().to(device)   # [N,2,S,K,f]
    levels = torch.tensor(tmeta["levels"], dtype=torch.float32)
    seqs = json.load(open(f"{tokens}/sequences.json"))

    cfgA = json.load(open(f"{runA}/config.json"))
    P = int(np.prod(I.shape[1:])); L = int(I.max().item()) + 1
    A = ARLM(P, P, L, d_model=cfgA["d_model"], layers=cfgA["layers"], max_w=cfgA["window"] + 1).to(device)
    A.load_state_dict(torch.load(f"{runA}/model_best.pt", map_location=device)); A.eval()

    cfgB = json.load(open(f"{runB}/config.json"))
    D = F.shape[1] * F.shape[2] * F.shape[3]
    B = ChunkDenoiser(D, cfgB["d_model"], cfgB["layers"], n_frames=cfgB["window"]).to(device)
    B.load_state_dict(torch.load(f"{runB}/model_best.pt", map_location=device)); B.eval()
    nrm = torch.load(f"{runB}/norm.pt")
    return dict(data=data, tok=tok, F=F, I=I, levels=levels, seqs=seqs, A=A, cfgA=cfgA,
                B=B, cfgB=cfgB, mu=nrm["mu"].to(device), sd=nrm["sd"].to(device))


def val_picks(seqs, n=64):
    """eval_rollout.py's split, verbatim."""
    rng = np.random.default_rng(0)
    order = rng.permutation(len(seqs))
    nval = max(8, len(seqs) // 10)
    val = [seqs[i] for i in order[:nval] if seqs[i]["len"] >= CTX + HOR]
    return val[:n]


def decode_g1(tok, data, ip, ir, T):
    with torch.no_grad():
        prot, ppos = tok.decode_from_ints(ip, ir, geom(data, "g1_30", T))
    return prot.cpu().numpy()


def rollout_A(M, rows, device):
    A, cfgA, I, levels, tok, data = M["A"], M["cfgA"], M["I"], M["levels"], M["tok"], M["data"]
    X_all = dequant(I.cpu(), levels).to(device)
    win = cfgA["window"]
    D2, S, K, f = I.shape[1:]
    cur = X_all[rows[:CTX]]
    pred = []
    t0 = time.time()
    with torch.no_grad():
        for t in range(HOR):
            logits = A(cur[-win:][None])[0, -1]                    # [P,L]
            nxt_int = logits.argmax(-1)
            x_next = dequant(nxt_int.reshape(1, D2, S, K, f).cpu(), levels).to(device)[0]
            cur = torch.cat([cur, x_next[None]], 0)
            pred.append(nxt_int.reshape(D2, S, K, f))
    ms = (time.time() - t0) / HOR * 1000
    ints = torch.stack(pred, 0)                                     # [HOR,2,S,K,f]
    # code-level freeze indicator: fraction of token slots identical to the previous frame
    same = (ints[1:] == ints[:-1]).float().mean().item()
    g6 = decode_g1(tok, data, ints[:, 0].permute(1, 0, 2, 3), ints[:, 1].permute(1, 0, 2, 3), HOR)
    return g6, ms, same


def rollout_B(M, rows, device, ddim_steps=50, seed=0):
    B, cfgB, F, tok, data = M["B"], M["cfgB"], M["F"], M["tok"], M["data"]
    mu, sd = M["mu"], M["sd"]
    ab = cosine_alphas(cfgB["T"]).to(device)
    T = cfgB["T"]
    x = (F[rows].reshape(CTX + HOR, -1) - mu) / sd
    ctx = x[:CTX]
    out = []
    g = torch.Generator(device=device); g.manual_seed(seed)
    t0 = time.time()
    with torch.no_grad():
        for c in range(HOR // CHUNK):
            ctx_in = ctx[-(cfgB["window"] - CHUNK):][None]
            z = torch.randn(1, CHUNK, ctx.shape[-1], device=device, generator=g)
            ts = torch.linspace(T, 1, ddim_steps, dtype=torch.long, device=device)
            for i, t in enumerate(ts):
                a = ab[t]
                v = B(ctx_in, z, t[None].float() / T)               # v-parameterization
                x0 = (a.sqrt() * z - (1 - a).sqrt() * v).clamp(-4, 4)
                eps = (z - a.sqrt() * x0) / (1 - a).sqrt()
                a_prev = ab[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=device)
                z = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps
            out.append(z[0]); ctx = torch.cat([ctx, z[0]], 0)
    ms = (time.time() - t0) / HOR * 1000
    pred = (torch.cat(out, 0) * sd + mu).reshape(HOR, *F.shape[1:])
    fp, fr = pred[:, 0], pred[:, 1]
    ip = tok.fsqp.ints(fp)
    ir = (tok.fsqp if tok.single_codebook else tok.fsqr).ints(fr)
    return decode_g1(tok, data, ip, ir, HOR), ms


def decode_context(M, rows, device):
    """Tokenizer decode of the CONTEXT codes onto g1_30 (identical for both arms)."""
    I = M["I"][rows[:CTX]]
    return decode_g1(M["tok"], M["data"], I[:, 0].permute(1, 0, 2, 3), I[:, 1].permute(1, 0, 2, 3), CTX)


# ------------------------------------------------------------------ render ----
def tint(scene, prefix, rgba):
    for gi in range(scene.ngeom):
        bname = mj.mj_id2name(scene, mj.mjtObj.mjOBJ_BODY, scene.geom_bodyid[gi])
        if bname and bname.startswith(prefix) and scene.geom_group[gi] == 2:
            scene.geom_rgba[gi] = rgba
            scene.geom_matid[gi] = -1


def render_seq(mapper, streams, size, gap, azimuth, elev):
    """streams: list of (prefix, rgba, rot6d [T,J,6]). Returns [T,H,W,3] uint8."""
    n = len(streams)
    ys = [(i - (n - 1) / 2) * gap for i in range(n)]
    scene = build_scene(mapper.xml, [(p,) for p, _, _ in streams], [])
    for (p, rgba, _) in streams:
        tint(scene, p, rgba)
    mdata = mj.MjData(scene)
    renderer = mj.Renderer(scene, height=size[1], width=size[0])
    cam = mj.MjvCamera()
    cam.lookat[:] = [0, 0, 0.80]
    cam.distance = 3.15
    cam.elevation = elev; cam.azimuth = azimuth
    poses = []
    for (_, _, g6) in streams:
        ang, quat, res = mapper.angles_from_global(g6)
        poses.append((ang, quat, res))
    T = len(streams[0][2])
    frames = []
    for t in range(T):
        for (pre, _, _), (ang, quat, _), y in zip(streams, poses, ys):
            set_robot(scene, mdata, pre, mapper, ang[t], quat[t], [0, y, STAND_H])
        mj.mj_forward(scene, mdata)
        renderer.update_scene(mdata, cam)
        frames.append(renderer.render().copy())
    resid = [float(r.mean()) for (_, _, r) in poses]
    return np.stack(frames), resid


def write_mp4(path, frames, fps):
    """Pipe raw RGB to system ffmpeg (imageio's pyav plugin has no crf knob here)."""
    import subprocess
    h, w = frames[0].shape[:2]
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-crf", "28",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f, dtype=np.uint8).tobytes())
    p.stdin.close(); assert p.wait() == 0, "ffmpeg failed"


def panel_centers(frame, W):
    """Locate the 4 panels in pixel space from the orange/green robots' color masks
    (panels are equally spaced in world y, so the two grey ones extrapolate linearly)."""
    r, g, b = [frame[..., i].astype(np.int16) for i in range(3)]
    orange = (r - g > 50) & (r - b > 90) & (r > 120)
    green = (g - r > 50) & (g - b > 50) & (g > 90)
    try:
        x1 = float(np.nonzero(orange)[1].mean()); x2 = float(np.nonzero(green)[1].mean())
        dx = x2 - x1
        assert 40 < dx < W / 2
        return [x1 - dx, x1, x2, x2 + dx]
    except Exception:
        return [(i + 0.5) * W / 4 for i in range(4)]


def annotate(img, labels, header, sub, panel_x, err_txt=None):
    from PIL import Image, ImageDraw, ImageFont
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    W, H = im.size

    def font(sz, bold=True):
        p = "/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf" % ("-Bold" if bold else "")
        try:
            return ImageFont.truetype(p, sz)
        except Exception:
            return ImageFont.load_default()
    f_lab, f_hd, f_sub, f_err = font(19), font(26), font(17, False), font(17)
    d.rectangle([0, 0, W, 40], fill=(18, 18, 20))
    d.rectangle([0, H - 68, W, H], fill=(18, 18, 20))
    d.text((14, 8), header[0], font=f_hd, fill=header[1])
    d.text((W - 14 - d.textlength(sub, font=f_sub), 12), sub, font=f_sub, fill=(150, 155, 162))
    for (txt, col), x in zip(labels, panel_x):
        w = d.textlength(txt, font=f_lab)
        d.text((x - w / 2, H - 62), txt, font=f_lab, fill=col)
    if err_txt:
        for (txt, col), x in zip(err_txt, panel_x):
            if not txt:
                continue
            w = d.textlength(txt, font=f_err)
            d.text((x - w / 2, H - 40), txt, font=f_err, fill=col)
    leg = ("grey = ground truth   |   orange = AR, discrete next-frame tokens   |   "
           "green = diffusion, 8-frame latent chunks (v-pred, 50 DDIM steps)")
    d.text(((W - d.textlength(leg, font=f_sub)) / 2, H - 20), leg, font=f_sub, fill=(140, 145, 152))
    return np.asarray(im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default=f"{RUNS}/E0b_sparse2_0803")
    ap.add_argument("--tokens", default=f"{RUNS}/E0b_sparse2_0803/tokens_seq10k")
    ap.add_argument("--runA", default=f"{RUNS}/E1A_v2_0804")
    ap.add_argument("--runB", default=f"{RUNS}/E1B_vpred_0804")
    ap.add_argument("--seq_pick", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--list", action="store_true", help="just print the first val seq names")
    ap.add_argument("--test_frame", action="store_true")
    ap.add_argument("--size", type=int, nargs=2, default=[1400, 560])
    ap.add_argument("--gap", type=float, default=0.85)
    ap.add_argument("--azimuth", type=float, default=180)
    ap.add_argument("--elev", type=float, default=-10)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--out", default='/home/arenalabs/Desktop/"be water, robot"'
                                     '/pose_translator_design/R4_VIZ/mp4/e1_rollout')
    args = ap.parse_args()
    device = "cuda"
    os.makedirs(args.out, exist_ok=True)

    seqs = json.load(open(f"{args.tokens}/sequences.json"))
    picks = val_picks(seqs)
    if args.list:
        for i, s in enumerate(picks[:40]):
            print(i, s["name"], s["len"])
        return

    M = load_all(device, args.tokenizer, args.tokens, args.runA, args.runB)
    mapper = G1QposMapper(M["data"]["g1_30"]["spec"])
    gtpose = M["data"]["g1_30"]["pose"]
    reach = float(M["data"]["g1_30"]["reach"])

    for k in args.seq_pick:
        s = picks[k]
        rows = torch.arange(s["start"], s["start"] + CTX + HOR, device=device)
        gt6 = gtpose[rows][..., :6].cpu().numpy()
        gtp = gtpose[rows][..., 6:9].cpu().numpy()
        ctx6 = decode_context(M, rows, device)
        ar6, msA, ar_same = rollout_A(M, rows, device)
        df6, msB = rollout_B(M, rows, device, seed=k)
        ar_full = np.concatenate([ctx6, ar6], 0)
        df_full = np.concatenate([ctx6, df6], 0)

        # per-frame rotation error on g1_30 (the decode target), deg
        def rerr(p6):
            a = rot6d_to_matrix(torch.as_tensor(p6)); b = rot6d_to_matrix(torch.as_tensor(gt6))
            return geodesic_deg(a, b).mean(-1).numpy()
        eA, eB, eC = rerr(ar_full), rerr(df_full), rerr(np.concatenate([ctx6, gt6[CTX:]], 0))
        print(f"\n=== [{k}] {s['name']} len={s['len']}  (ms/frame A={msA:.1f} B={msB:.1f})")
        print("  ctx tokenizer recon rot err (g1_30): %.1f deg" % eC[:CTX].mean())
        for h in (1, 4, 8, 16, 24, 32):
            print(f"  h{h:>2}: AR {eA[CTX + h - 1]:6.1f} deg | DIFF {eB[CTX + h - 1]:6.1f} deg")
        # freeze / jitter diagnostics on the generated window (g1 rot velocity, deg/frame)
        def vel(p6):
            R = rot6d_to_matrix(torch.as_tensor(p6))
            return geodesic_deg(R[1:], R[:-1]).mean(-1).numpy()
        vA, vB, vG = vel(ar_full), vel(df_full), vel(np.concatenate([ctx6, gt6[CTX:]], 0))
        vGT = vel(gt6)
        print("  mean |rot velocity| deg/frame  ctx=%.2f | GT-gen=%.2f | AR-gen=%.2f | DIFF-gen=%.2f"
              % (vG[:CTX - 1].mean(), vGT[CTX:].mean(), vA[CTX:].mean(), vB[CTX:].mean()))
        print("  velocity by 8-frame block (GT / AR / DIFF):", " | ".join(
            "%.1f/%.1f/%.1f" % (vGT[CTX + i:CTX + i + 8].mean(), vA[CTX + i:CTX + i + 8].mean(),
                                vB[CTX + i:CTX + i + 8].mean()) for i in range(0, HOR, 8)))
        print("  AR token slots repeated from previous frame: %.1f%%" % (100 * ar_same))

        streams = [("gtL_", COL["gt"], np.concatenate([ctx6, gt6[CTX:]], 0)),
                   ("ar_", COL["ar"], ar_full),
                   ("df_", COL["df"], df_full),
                   ("gtR_", COL["gt"], np.concatenate([ctx6, gt6[CTX:]], 0))]
        # panel 0/3 show the GT continuation; during context they show the same tokenizer
        # decode as the prediction panels so all four are identical until t=CTX.
        raw, resid = render_seq(mapper, streams, args.size, args.gap, args.azimuth, args.elev)
        print("  hinge residual (deg, off-manifold part dropped by twist projection):",
              [round(r, 2) for r in resid])
        if args.test_frame:
            import imageio.v2 as imageio
            for t in (0, CTX + 20):
                p = f"{args.out}/_test_s{k}_f{t}.png"
                imageio.imwrite(p, raw[t]); print("PNG", p)
            return

        W = args.size[0]
        panel_x = panel_centers(raw[CTX // 2], W)
        labels = [("GT", TXT["gt"]), ("AR", TXT["ar"]), ("DIFFUSION", TXT["df"]), ("GT", TXT["gt"])]
        name = s["name"]

        def compose(t, phase, extra=""):
            if t < CTX:
                hd = (f"CONTEXT  frame {t + 1}/{CTX}   (given to both models)", (200, 205, 212))
                err = None
            else:
                g = t - CTX + 1
                hd = (f"GENERATED  +{g}/{HOR} frames{extra}", (255, 210, 120))
                err = [("", TXT["gt"]), (f"{eA[t]:.0f}deg err", TXT["ar"]),
                       (f"{eB[t]:.0f}deg err", TXT["df"]), ("", TXT["gt"])]
            sub = f"{name[:34]}   |   soma77 context -> G1 mesh decode   |   0.4x speed"
            return annotate(raw[t], labels, hd, sub, panel_x, err)

        out_frames = [compose(t, "ctx") for t in range(CTX)]
        banner = annotate(raw[CTX - 1], labels,
                          ("GENERATION STARTS  --  models now run free", (255, 120, 90)),
                          f"{name}   |   context ends, 32 frames predicted from here",
                          panel_x, None)
        out_frames += [banner] * 16
        out_frames += [compose(t, "gen") for t in range(CTX, CTX + HOR)]
        out_frames += [compose(CTX + HOR - 1, "gen")] * 14
        # replay: last bit of context + the whole generated window again
        out_frames += [compose(t, "ctx") for t in range(CTX - 16, CTX)]
        out_frames += [compose(t, "gen", "   (replay)") for t in range(CTX, CTX + HOR)]
        out_frames += [compose(CTX + HOR - 1, "gen", "   (replay)")] * 12

        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)[:60]
        mp4 = f"{args.out}/{k:02d}_{safe}.mp4"
        write_mp4(mp4, out_frames, args.fps)
        print("MP4:", mp4, f"({len(out_frames)} frames, {len(out_frames)/args.fps:.1f}s)")


if __name__ == "__main__":
    main()
