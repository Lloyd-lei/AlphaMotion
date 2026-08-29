"""V7 eyeball evidence — animated stick figures, GT next to each arm.

    v7_render.py --runs V7_jepa_0807,V7_maskfsq_0807

Numbers say "energy ratio 0.6". Only a moving picture says whether that is a small
jumping jack or a person standing still while their endpoints move. Two sections:

  (a) TRANSITIONS   the spliced walk->kick / run->stand / walk->jacks windows from
                    v7_eval.transition_test, seam hidden at frame 30. Watch for the
                    model going limp through the second half.
  (b) FAMILIES      ordinary held-out windows from the families where mean regression
                    shows (jacks / run / kick), reconstructed from the endpoints plus
                    the 125 quantized bits.

Every frame is FK from the decoded ROTATIONS ONLY, through the frozen spatial layer's
masked-completion path (key joints -> full body), which is exactly the pipeline the
ladder's frame_code_floor row prices. The decoder's raw position stream is never
drawn (p1_fkfix).

Output: R4_VIZ/mp4/v7/*.gif — paths are printed. Judge with your eyes.
"""
from __future__ import annotations
import os, sys, json, time, argparse
import numpy as np
import torch

EXP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXP)
import v5_common as V                                             # noqa: E402
from train_v5 import load_run, to_states                          # noqa: E402
import v7_eval as E7                                              # noqa: E402

OUT = ('/home/arenalabs/Desktop/"be water, robot"/pose_translator_design/'
       'R4_VIZ/mp4/v7')
EMB = "human_smpl"


@torch.no_grad()
def to_full_body(tok, ent, key, kidx, dev, spec):
    """[T,nk,9] key states -> FK positions [T,J,3] cm, through the FROZEN spatial
    layer's masked completion. Rotations only; the position stream is discarded."""
    rot, _ = V.complete_from_key(tok, ent, ent, key.to(dev).float(), kidx, dev)
    return V.fk_pos(rot.cpu().numpy(), spec)


def deyaw(panels, ref_R0):
    """Rotate every panel by the inverse of the GT window's FIRST-frame root yaw.

    The cache stores GLOBAL rotations with the root at the origin, so a window whose
    subject happens to face away from +Z renders as a stick figure seen from behind
    at an arbitrary angle. One CONSTANT rotation about the vertical axis, taken from
    GT frame 0 and applied identically to GT and to every arm, puts the subject in a
    fixed frame without touching anything that is being compared."""
    f = ref_R0[:, 2].copy()                       # body forward column, projected
    f[1] = 0.0
    n = np.linalg.norm(f)
    if n < 1e-6:
        return panels
    f /= n
    c, s = f[2], f[0]                             # rotate f onto +Z
    Ry = np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])
    return [p @ Ry.T for p in panels]


def animate(panels, titles, path, parents, fps=30, seam=None, note=""):
    """panels: list of [T,J,3] cm arrays. Two rows: front (x,y) and side (z,y)."""
    import matplotlib
    matplotlib.use("Agg")
    plt = V.fig_style()
    from matplotlib.animation import FuncAnimation, PillowWriter
    n = len(panels)
    T = min(p.shape[0] for p in panels)
    allp = np.concatenate([p[:T] for p in panels], 0)
    lo, hi = allp.min(0).min(0), allp.max(0).max(0)
    pad = 0.10 * max(hi - lo)
    VIEWS = ((0, 1, "front"), (2, 1, "side"))
    fig, ax = plt.subplots(2, n, figsize=(2.9 * n, 8.4), squeeze=False)
    lines, dots = [], []
    for r, (h_, v_, vname) in enumerate(VIEWS):
        for i in range(n):
            a = ax[r][i]
            a.set_xlim(lo[h_] - pad, hi[h_] + pad); a.set_ylim(lo[v_] - pad, hi[v_] + pad)
            a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([]); a.grid(False)
            a.set_title(f"{titles[i]} ({vname})", fontsize=10)
            col = "#111" if i == 0 else V.SERIES[(i - 1) % len(V.SERIES)]
            lines.append([a.plot([], [], "-", color=col, lw=2.0)[0]
                          for j in range(len(parents)) if parents[j] >= 0])
            dots.append(a.plot([], [], "o", color=col, ms=2.6)[0])
    txt = fig.text(0.5, 0.975, "", ha="center", fontsize=10)
    if note:
        fig.text(0.5, 0.012, note, ha="center", fontsize=8, color="#52514e")

    def upd(t):
        for r, (h_, v_, _) in enumerate(VIEWS):
            for i, p in enumerate(panels):
                q = p[min(t, p.shape[0] - 1)]
                s_ = r * n + i
                k = 0
                for j in range(len(parents)):
                    if parents[j] < 0:
                        continue
                    lines[s_][k].set_data([q[parents[j], h_], q[j, h_]],
                                          [q[parents[j], v_], q[j, v_]])
                    k += 1
                dots[s_].set_data(q[:, h_], q[:, v_])
        mark = "   <<< SEAM" if (seam is not None and abs(t - seam) <= 1) else ""
        txt.set_text(f"frame {t}/{T-1}{mark}")
        return sum(lines, []) + dots + [txt]

    os.makedirs(os.path.dirname(path), exist_ok=True)
    FuncAnimation(fig, upd, frames=T, interval=1000 / fps, blit=False).save(
        path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="V7_jepa_0807,V7_maskfsq_0807")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n_each", type=int, default=2)
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
    ent = V.body_entry(V.AMASS, EMB, dev, kd.reach[EMB])
    spec, kidx, reach = kd.spec[EMB], kd.kidx[EMB], kd.reach[EMB]
    parents = np.asarray(spec.parents)
    models = {a: load_run(rd, dev) for a, rd in arms}
    T = V.T_WIN
    paths = []

    def render_set(w, tag, seam=None, note=""):
        """w [n,T,nk,9] GT windows -> one gif per sample."""
        w = w.to(dev).float()
        cur, tgt = w[:, 0], w[:, -1]
        kn = torch.ones(w.shape[0], device=dev)
        preds = {}
        for a, (m, _) in models.items():
            preds[a] = to_states(m(w, cur, tgt, kn)[0], m, cur, tgt, kn).float().cpu()
        for i in range(w.shape[0]):
            try:
                pan = [to_full_body(tok, ent, w[i].cpu() * 1.0, kidx, dev, spec)]
                ttl = ["GT"]
                for a in preds:
                    pan.append(to_full_body(tok, ent, preds[a][i], kidx, dev, spec))
                    ttl.append(a.replace("_0807", ""))
                pan = [p * reach for p in pan]
                from common import rot6d_to_matrix as _r2m
                R0 = _r2m(w[i, 0, 0, :6].double().cpu())[None].numpy()[0]
                pan = deyaw(pan, R0)
                p = animate(pan, ttl, f"{args.out}/{tag}_{i}.gif", parents,
                            seam=seam, note=note)
                paths.append(p); print(f"  -> {p}", flush=True)
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"WARN {tag}_{i}: {e}", flush=True)

    # ---- (a) transitions, seam hidden -------------------------------------
    for famA, famB in E7.TRANSITIONS:
        try:
            w, h = E7.splice(kd, EMB, famA, famB, args.n_each, T, seed=101)
            render_set(w, f"transition_{famA}_{famB}", seam=h,
                       note=f"{famA} -> {famB} spliced at frame {h}; the model is "
                            f"given ONLY frame 0 and frame {T-1} plus 125 bits")
        except Exception as e:
            print(f"WARN transition {famA}->{famB}: {e}", flush=True)

    # ---- (b) the families where mean regression shows ----------------------
    g = np.random.default_rng(41)
    for fam in ("jacks", "run", "kick"):
        try:
            st, _ = kd.sample_starts(args.n_each, g, "val", span=T, families=[fam])
            render_set(kd.window(EMB, st, T, T), f"family_{fam}",
                       note=f"held-out {fam}: endpoints + 125 bits -> 60 frames")
        except Exception as e:
            print(f"WARN family {fam}: {e}", flush=True)

    print("\nGIFS (lloyd judges with his eyes):", flush=True)
    for p in paths:
        print("  " + p, flush=True)
    json.dump(paths, open(f"{args.out}/_index.json", "w"), indent=1)


if __name__ == "__main__":
    main()
