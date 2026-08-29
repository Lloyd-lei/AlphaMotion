"""V7 BRIDGE — can we chain two motions in real time? JEPA vs Mask-FSQ.

    v7_bridge.py --runs V7_jepa_0807,V7_maskfsq_0807

THE QUESTION (lloyd, 2026-08-07): "两个没见过的动作片段concate在一起（这个是最重要
的可能 直接决定我们能不能连续实时控制）". Everything else V7 measures is a
reconstruction score. THIS is the deployment question: a controller finishing clip A
and being handed clip B has to invent the frames in between, live, with no ground
truth for them, and the joint has to be smooth enough to actually run on hardware.

WHAT IS ACTUALLY BEING TESTED, stated plainly so nobody over-reads the pictures:
  * Clip A and clip B are two DIFFERENT held-out (val) motions, different families,
    never seen in training. They are shown as REAL data — untouched GT frames.
  * The middle segment is 100% invented by the model from two things only: A's last
    frame (`cur`), B's first frame (`tgt`), plus the quantized tokens.
  * There is NO ground truth for that middle, by construction — two unrelated clips
    have no "true" transition. So the bridge cannot be scored against GT, and this
    file never pretends to. It is scored GT-free (velocity handoff, teleports,
    limpness) and, above all, by eye.
  * WHERE THE BRIDGE TOKENS COME FROM: in a deployed system the generator emits them.
    We have no generator yet, so the bridge is tokenized from the ENDPOINT
    INTERPOLATION — the zero-knowledge plan, i.e. "what the model's motion prior does
    to a naive controller's straight line". That is the honest floor, not a best case:
    a real generator can only do better than a straight line. Labelled `interp-token`
    everywhere.

FOUR PANELS PER PAIR, which is the whole argument in one row:
  hard cut     hold A's last pose, then snap to B  — what you get with no model
  interp       endpoints only, ZERO tokens         — the naive controller
  jepa         endpoints + 125 bits                — arm 1
  maskfsq      endpoints + 125 bits                — arm 2

THE BRIDGE IS MARKED: during invented frames every skeleton turns red, the banner
reads "MODEL BRIDGE (invented)", and a timeline strip under the figure shows
A | BRIDGE | B with a moving cursor. Real GT frames are drawn in the normal colour.
You should never have to guess which frames the model made up.

TWO BODIES:
  human_smpl        in training
  agibot_x2_ultra   ZERO-SHOT — X2 is in no training set of any run, ever. Same
                    tokens, decoded onto a skeleton the model has never seen.

Output: R4_VIZ/mp4/v7_bridge/*.gif + BRIDGE_verdict.txt. Judge with your eyes.
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
import torch

EXP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXP)
import v5_common as V                                             # noqa: E402
from train_v5 import load_run, to_states                          # noqa: E402

OUT = ('/home/arenalabs/Desktop/"be water, robot"/pose_translator_design/'
       'R4_VIZ/mp4/v7_bridge')
EMB = "human_smpl"
HL = "#d1495b"          # the invented frames are drawn in this colour, always

# Pairs of DIFFERENT families. The second element is where we are handing control to.
PAIRS = [("walk", "kick"), ("run", "jacks"), ("kick", "walk"), ("dance", "run")]


# ----------------------------------------------------------------- bodies ------
def x2_entry(device):
    """AgiBot X2 conditioning from its raw MJCF. Zero-shot: X2 is in no training set.
    Same construction night2_eval.build_x2_entry uses."""
    import night2_eval as N2
    return N2.build_x2_entry(device)


@torch.no_grad()
def to_full_body(tok, src_ent, tgt_ent, key, kidx, dev, spec):
    """[T,nk,9] key states -> FK positions [T,J,3] in CENTIMETRES.

    NOTE ON UNITS, because it bit this file once: spec.rest_offsets are already in cm
    (human_smpl absmax 99 cm), so V.fk_pos returns cm directly. v7_render multiplies
    its panels by `reach` — harmless there because every panel gets the same factor
    and the axes autoscale, but it inflates any DISTANCE metric by ~104x. Nothing here
    is scaled.
    """
    rot, _ = V.complete_from_key(tok, src_ent, tgt_ent, key.to(dev).float(), kidx, dev)
    return V.fk_pos(rot.cpu().numpy(), spec)


def yaw_align(B, ref):
    """Rotate clip B about the vertical so its first frame faces the same way as `ref`
    (clip A's last frame). [T,nk,9] global rot6d + root-relative pos.

    A controller handing over from A to B does not get to teleport the robot's
    heading — the body keeps the yaw it already has and the new clip is played from
    there. Without this the "bridge" is dominated by an arbitrary 180-degree spin
    between two unrelated recordings, which is a property of the dataset, not of the
    model. Only yaw is touched; pitch/roll and all relative geometry are untouched.
    """
    from common import rot6d_to_matrix as r2m, matrix_to_rot6d as m2r
    Rb = r2m(B[0, 0, :6].double())            # clip B root, first frame
    Ra = r2m(ref[0, :6].double())             # clip A root, last frame

    def yaw_of(R):
        f = R[:, 2].clone(); f[1] = 0.0
        n = torch.linalg.norm(f)
        return f / n if n > 1e-6 else None
    fb, fa = yaw_of(Rb), yaw_of(Ra)
    if fb is None or fa is None:
        return B
    c = float(fa @ fb)                        # rotate fb onto fa about +Y
    s = float(fa[0] * fb[2] - fa[2] * fb[0])
    Ry = torch.tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=torch.float64)
    R = r2m(B[..., :6].double())              # [T,nk,3,3]
    out = B.clone()
    out[..., :6] = m2r(Ry @ R).to(B.dtype)
    out[..., 6:9] = (B[..., 6:9].double() @ Ry.T).to(B.dtype)
    return out


# ---------------------------------------------------------------- metrics ------
def step_cm(p):
    """[T,J,3] -> [T-1] mean per-joint displacement between consecutive frames."""
    return np.linalg.norm(np.diff(p, axis=0), axis=-1).mean(-1)


def bridge_metrics(p, la, lb_start):
    """GT-free. p [T,J,3]; frames [0,la) are real A, [la,lb_start) invented,
    [lb_start,T) real B.

    handoff_in / handoff_out are the ONLY numbers here that speak to real-time
    control: the ratio of motion speed just after a join to just before it. 1.0 = the
    body carries its velocity through the handoff. Far below 1 = it stalls at the
    seam; far above = it lurches to catch up. Both look terrible on hardware."""
    s = step_cm(p)
    pre_in = s[max(0, la - 4):la - 1].mean()
    post_in = s[la:la + 3].mean()
    pre_out = s[lb_start - 4:lb_start - 1].mean()
    post_out = s[lb_start:lb_start + 3].mean()
    eps = 1e-6
    br = p[la:lb_start]
    real = np.concatenate([p[:la], p[lb_start:]], 0)
    return {
        "handoff_in": round(float(post_in / (pre_in + eps)), 3),
        "handoff_out": round(float(post_out / (pre_out + eps)), 3),
        "max_step_cm": round(float(s.max()), 2),
        "max_step_at": int(s.argmax()),
        "bridge_energy_ratio": round(float(br.std(0).mean() / (real.std(0).mean() + eps)), 3),
        "bridge_jerk_p95": round(float(np.percentile(
            np.abs(np.diff(br, n=3, axis=0)).mean(-1), 95)), 4),
    }


# ------------------------------------------------------------------ render -----
def animate(panels, titles, path, parents, la, lb_start, fps=30, note=""):
    """panels: list of [T,J,3]. Front + side rows, plus a timeline strip.
    Frames in [la, lb_start) are drawn in HL and banner-marked as invented."""
    import matplotlib
    matplotlib.use("Agg")
    plt = V.fig_style()
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.patches import Rectangle
    n = len(panels)
    T = min(p.shape[0] for p in panels)
    allp = np.concatenate([p[:T] for p in panels], 0)
    lo, hi = allp.min(0).min(0), allp.max(0).max(0)
    pad = 0.10 * max(hi - lo)
    VIEWS = ((0, 1, "front"), (2, 1, "side"))

    fig = plt.figure(figsize=(2.9 * n, 9.1))
    gs = fig.add_gridspec(3, n, height_ratios=[1.0, 1.0, 0.11], hspace=0.14)
    lines, dots, base_col = [], [], []
    for r, (h_, v_, vname) in enumerate(VIEWS):
        for i in range(n):
            a = fig.add_subplot(gs[r, i])
            a.set_xlim(lo[h_] - pad, hi[h_] + pad); a.set_ylim(lo[v_] - pad, hi[v_] + pad)
            a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([]); a.grid(False)
            a.set_title(f"{titles[i]} ({vname})", fontsize=10)
            col = "#111" if i == 0 else V.SERIES[(i - 1) % len(V.SERIES)]
            base_col.append(col)
            lines.append([a.plot([], [], "-", color=col, lw=2.0)[0]
                          for j in range(len(parents)) if parents[j] >= 0])
            dots.append(a.plot([], [], "o", color=col, ms=2.6)[0])

    # timeline strip: A | BRIDGE | B with a moving cursor
    tl = fig.add_subplot(gs[2, :])
    tl.set_xlim(0, T - 1); tl.set_ylim(0, 1)
    tl.set_yticks([]); tl.set_xticks([]); tl.grid(False)
    tl.add_patch(Rectangle((0, 0), la, 1, color="#c9c7c2"))
    tl.add_patch(Rectangle((la, 0), lb_start - la, 1, color=HL))
    tl.add_patch(Rectangle((lb_start, 0), T - lb_start, 1, color="#c9c7c2"))
    tl.text(la / 2, 0.5, "clip A (real)", ha="center", va="center", fontsize=8)
    tl.text((la + lb_start) / 2, 0.5, "MODEL BRIDGE", ha="center", va="center",
            fontsize=8, color="white", fontweight="bold")
    tl.text((lb_start + T) / 2, 0.5, "clip B (real)", ha="center", va="center", fontsize=8)
    cursor = tl.axvline(0, color="#111", lw=2.0)

    txt = fig.text(0.5, 0.977, "", ha="center", fontsize=11)
    if note:
        fig.text(0.5, 0.008, note, ha="center", fontsize=8, color="#52514e")

    def upd(t):
        inb = la <= t < lb_start
        for r, (h_, v_, _) in enumerate(VIEWS):
            for i, p in enumerate(panels):
                q = p[min(t, p.shape[0] - 1)]
                s_ = r * n + i
                c = HL if inb else base_col[s_]
                k = 0
                for j in range(len(parents)):
                    if parents[j] < 0:
                        continue
                    lines[s_][k].set_data([q[parents[j], h_], q[j, h_]],
                                          [q[parents[j], v_], q[j, v_]])
                    lines[s_][k].set_color(c)
                    k += 1
                dots[s_].set_data(q[:, h_], q[:, v_])
                dots[s_].set_color(c)
        cursor.set_xdata([t, t])
        txt.set_text(f"frame {t}/{T-1}   " +
                     ("<<< MODEL BRIDGE — every pose here is invented >>>" if inb
                      else "clip A (real data)" if t < la else "clip B (real data)"))
        txt.set_color(HL if inb else "#111")
        return sum(lines, []) + dots + [txt, cursor]

    os.makedirs(os.path.dirname(path), exist_ok=True)
    FuncAnimation(fig, upd, frames=T, interval=1000 / fps, blit=False).save(
        path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return path


# -------------------------------------------------------------------- main -----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="V7_jepa_0807,V7_maskfsq_0807")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--la", type=int, default=30, help="real frames from clip A")
    ap.add_argument("--lbb", type=int, default=30, help="real frames from clip B")
    ap.add_argument("--bridge", type=int, default=30,
                    help="invented frames. 30 = 1 s, the control-relevant case; the "
                         "arms were TRAINED at T=60, so both spans are reported.")
    ap.add_argument("--also_span", type=int, default=60,
                    help="second bridge span for the metrics table (in-distribution)")
    ap.add_argument("--no_x2", action="store_true")
    args = ap.parse_args()
    dev = args.device
    os.makedirs(args.out, exist_ok=True)

    arms = []
    for a in [r for r in args.runs.split(",") if r]:
        rd = a if os.path.isabs(a) else f"{V.RUNS}/{a}"
        if os.path.exists(f"{rd}/model.pt"):
            arms.append((a, rd))
        else:
            print(f"[skip] {a}: no model.pt", flush=True)
    if not arms:
        raise SystemExit("no finished arm")

    kd = V.KeyData(V.AMASS, [EMB], device="cpu")
    tok, _ = V.load_frozen(dev)
    src_ent = V.body_entry(V.AMASS, EMB, dev, kd.reach[EMB])
    kidx, reach = kd.kidx[EMB], kd.reach[EMB]
    models = {a: load_run(rd, dev)[0] for a, rd in arms}

    bodies = [("human_smpl", src_ent, kd.spec[EMB], reach)]
    if not args.no_x2:
        x2 = x2_entry(dev)
        if x2 is not None:
            bodies.append(("agibot_x2_ultra_ZEROSHOT", x2, x2["spec"], None))
            print(f"[X2] ok, J={x2['spec'].J}", flush=True)

    rows, paths = {}, []

    @torch.no_grad()
    def build_bridges(a_end, b_start, span, donor=None):
        """-> dict name -> [span,nk,9] key states for the invented middle.

        Two token sources, and the gap between them is the real finding:
          interp-token : tokens encoded from the endpoint straight line. Zero
                         knowledge. Guarantees "model ~= interp" is uninformative —
                         it is what we FED it.
          donor-token  : tokens encoded from a REAL held-out motion of B's family.
                         This is what a trained generator would emit, so it is the
                         CEILING the architecture can reach on this task. Without
                         this row one cannot tell "the model cannot bridge" from
                         "we handed it a straight line and it gave one back".
        """
        cur = a_end[None].to(dev).float()
        tgt = b_start[None].to(dev).float()
        kn = torch.ones(1, device=dev)
        out = {}
        m0 = next(iter(models.values()))
        interp = m0.baseline(cur, tgt, kn, span).float()
        out["interp (0 bits)"] = interp[0].cpu()
        for a, m in models.items():
            nm = a.replace("_0807", "")
            e = m.encode(interp, cur, tgt, kn)
            o = m.decode(e["path_z"], e["time_z"], cur, tgt, kn, span)
            out[f"{nm} interp-tok"] = to_states(o, m, cur, tgt, kn).float()[0].cpu()
            if donor is not None:
                d = donor[None].to(dev).float()
                ed = m.encode(d, d[:, 0], d[:, -1], kn)
                od = m.decode(ed["path_z"], ed["time_z"], cur, tgt, kn, span)
                out[f"{nm} donor-tok"] = to_states(od, m, cur, tgt, kn).float()[0].cpu()
        return out

    g = np.random.default_rng(2027)
    for famA, famB in PAIRS:
        try:
            sa, _ = kd.sample_starts(1, g, "val", span=args.la, families=[famA])
            sb, _ = kd.sample_starts(1, g, "val", span=args.lbb, families=[famB])
            A = kd.window(EMB, sa, args.la, args.la)[0]        # [la,nk,9] real
            B = kd.window(EMB, sb, args.lbb, args.lbb)[0]      # [lb,nk,9] real
            B = yaw_align(B, A[-1])
        except Exception as e:
            print(f"WARN sample {famA}->{famB}: {e}", flush=True)
            continue

        for span in dict.fromkeys([args.bridge, args.also_span]):
            try:
                try:                       # a REAL held-out motion of B's family
                    sd, _ = kd.sample_starts(1, g, "val", span=span, families=[famB])
                    donor = kd.window(EMB, sd, span, span)[0]
                except Exception:
                    donor = None
                br = build_bridges(A[-1], B[0], span, donor)
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"WARN bridge {famA}->{famB} span={span}: {e}", flush=True)
                continue
            # hard cut: hold A's last pose for the same duration, then snap to B
            hold = A[-1][None].expand(span, -1, -1).clone()
            seqs = {"hard cut (no model)": hold}
            seqs.update(br)
            full = {k: torch.cat([A, v, B], 0) for k, v in seqs.items()}
            la, lbs = args.la, args.la + span

            for bname, tgt_ent, spec, rsc in bodies:
                pan, ttl = [], []
                for k, v in full.items():
                    p = to_full_body(tok, src_ent, tgt_ent, v, kidx, dev, spec)
                    pan.append(p); ttl.append(k)                 # already cm
                    key = f"{famA}->{famB} | span{span} | {bname} | {k}"
                    rows[key] = bridge_metrics(pan[-1], la, lbs)
                if span != args.bridge:            # metrics for both spans, render one
                    continue
                try:
                    from common import rot6d_to_matrix as _r2m
                    import v7_render as R7
                    R0 = _r2m(A[0, 0, :6].double())[None].numpy()[0]
                    pan = R7.deyaw(pan, R0)
                    p = animate(pan, ttl,
                                f"{args.out}/bridge_{famA}_{famB}_{bname}.gif",
                                np.asarray(spec.parents), la, lbs,
                                note=(f"{famA} -> {famB}, two DIFFERENT held-out clips. "
                                      f"Red frames {la}-{lbs-1} are invented from A's last "
                                      f"frame + B's first frame + 125 bits. No ground "
                                      f"truth exists for them."))
                    paths.append(p); print(f"  -> {p}", flush=True)
                except Exception as e:
                    import traceback; traceback.print_exc()
                    print(f"WARN render {famA}->{famB} {bname}: {e}", flush=True)

    # ------------------------------------------------------------- verdict -----
    L = []
    L.append("=" * 110)
    L.append("V7 BRIDGE — two DIFFERENT held-out clips chained, middle invented by the model")
    L.append("=" * 110)
    L.append("The bridge has NO ground truth by construction, so every number here is")
    L.append("GT-free. handoff_in/out = motion speed just after a join / just before it;")
    L.append("1.0 means velocity carries through. bridge_energy_ratio << 1 = the model")
    L.append("goes limp in the invented part. Bridge tokens come from the ENDPOINT")
    L.append("INTERPOLATION (no generator exists yet) — the floor, not a best case.")
    L.append("X2 is ZERO-SHOT: in no training set of any run.")
    L.append("-" * 110)
    hdr = f"{'case':<62}{'hand_in':>9}{'hand_out':>10}{'maxstep':>9}{'br_energy':>11}"
    L.append(hdr); L.append("-" * 110)
    for k in sorted(rows):
        r = rows[k]
        L.append(f"{k:<62}{r['handoff_in']:>9}{r['handoff_out']:>10}"
                 f"{r['max_step_cm']:>9}{r['bridge_energy_ratio']:>11}")
    txt = "\n".join(L)
    print("\n" + txt, flush=True)
    open(f"{V.RUNS}/V7_BRIDGE_verdict.txt", "w").write(txt + "\n")
    json.dump(rows, open(f"{V.RUNS}/V7_BRIDGE_verdict.json", "w"), indent=1)
    json.dump(paths, open(f"{args.out}/_index.json", "w"), indent=1)
    print(f"\nverdict: {V.RUNS}/V7_BRIDGE_verdict.txt", flush=True)
    for p in paths:
        print("  " + p, flush=True)


if __name__ == "__main__":
    main()
