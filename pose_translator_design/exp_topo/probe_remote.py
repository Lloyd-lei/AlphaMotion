"""THE REMOTE-CONTROL DETECTIVE — which body motion does each token control?
(lloyd's level-2 analysis, 2026-08-07: "token — 遥控器")

    probe_remote.py

INTERVENTION, not correlation: encode N val windows, then for one register at a
time SWAP that register's code with another window's (everything else untouched),
decode, and measure WHERE the output moved — per frame and per key joint. The
aggregated displacement field is the register's "control surface".

What each map answers:
  register x FRAME map    maskfsq claims its 9 codes are time-local (group k = a
                          ~0.22 s slice). If true, swapping code k moves ONLY the
                          frames in group k — the map is block-diagonal. jepa's 8
                          unordered codes should smear across all 60 frames.
  register x JOINT map    does any code specialize on arms vs legs vs root? A flat
                          map = holographic code; structure = a button layout.
  per-DIGIT probe         step ONE FSQ digit of one register by +-1 level, decode:
                          the smallest button on the remote. Reported for the
                          register with the largest control surface.

Also settles ARCH_V5 §七-1 (code-frequency vs information) and §七-3 (V7 jepa's
TIME register distribution), both owed since the spec was written.

Output: runs/REMOTE_verdict.txt (+ .json), heatmap PNG in R4_VIZ/fig/.
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v5_common as V                                              # noqa: E402
from train_v5 import load_run, to_states                           # noqa: E402

EMB = "human_smpl"
FIG = ('/home/arenalabs/Desktop/"be water, robot"/pose_translator_design/'
       'R4_VIZ/fig')


@torch.no_grad()
def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    kd = V.KeyData(V.AMASS, [EMB], device="cpu")
    T = V.T_WIN
    n = 256
    g = np.random.default_rng(23)
    st, cids = kd.sample_starts(n, g, "val", span=T)
    w = kd.window(EMB, st, T, T).to(dev).float()
    cur, tgt = w[:, 0], w[:, -1]
    kn = torch.ones(n, device=dev)
    knames = kd.knames[EMB]
    out = {}

    for run in ("V7_maskfsq_0807", "V7_jepa_0807"):
        nm = run.replace("_0807", "")
        m = load_run(f"{V.RUNS}/{run}", dev)[0]
        e = m.encode(w, cur, tgt, kn)
        base = to_states(m.decode(e["path_z"], e["time_z"], cur, tgt, kn, T),
                         m, cur, tgt, kn).float()
        R = e["path_z"].shape[1]
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(3)).to(dev)

        # ---- register x frame / register x joint maps ------------------------
        fmap = np.zeros((R, T))                       # mean key displacement [cm]
        jmap = np.zeros((R, len(knames)))
        for r in range(R):
            pz = e["path_z"].clone()
            pz[:, r] = pz[perm, r]                    # swap ONE register
            p = to_states(m.decode(pz, e["time_z"], cur, tgt, kn, T),
                          m, cur, tgt, kn).float()
            d = (p[..., 6:9] - base[..., 6:9]).norm(dim=-1) * kd.reach[EMB]  # [n,T,nk]
            fmap[r] = d.mean((0, 2)).cpu().numpy()
            jmap[r] = d.mean((0, 1)).cpu().numpy()

        # time-locality score: fraction of a register's total effect that lands in
        # its OWN 60/R-frame slice (maskfsq should be high, jepa ~1/R)
        bounds = np.round(np.linspace(0, T, R + 1)).astype(int)
        own = [fmap[r, bounds[r]:bounds[r + 1]].sum() / max(fmap[r].sum(), 1e-9)
               for r in range(R)]

        # ---- time register (jepa only): swap the WHOLE time bank -------------
        trow = None
        if e["time_z"].shape[1]:
            tz = e["time_z"][perm]
            p = to_states(m.decode(e["path_z"], tz, cur, tgt, kn, T),
                          m, cur, tgt, kn).float()
            d = (p[..., 6:9] - base[..., 6:9]).norm(dim=-1) * kd.reach[EMB]
            trow = d.mean((0, 2)).cpu().numpy()

        # ---- per-digit probe on the strongest register -----------------------
        r_star = int(np.argmax(fmap.sum(1)))
        digit_eff = []
        q0 = e["path_q"].clone()
        half = (m.fsq_path._levels - 1) / 2
        step = 1.0 / half                              # one level in q units
        for dgt in range(q0.shape[-1]):
            q = q0.clone()
            q[:, r_star, dgt] = (q[:, r_star, dgt] + step[dgt]).clamp(-1, 1)
            p = to_states(m.decode(m.fsq_path.proj_out(q), e["time_z"], cur, tgt,
                                   kn, T), m, cur, tgt, kn).float()
            d = (p[..., 6:9] - base[..., 6:9]).norm(dim=-1) * kd.reach[EMB]
            digit_eff.append([round(float(x), 3) for x in d.mean((0, 1)).cpu()])

        # ---- §七-1: code frequency vs information ---------------------------
        pi, _ = m.token_ints(e)
        ints = pi.reshape(n, R, -1).cpu().numpy()
        freq_info = []
        for r in range(R):
            tup, cnt = np.unique([tuple(x) for x in ints[:, r]], axis=0,
                                 return_counts=True)
            freq_info.append({"distinct": int(len(cnt)),
                              "max_count": int(cnt.max()),
                              "singleton_frac": round(float((cnt == 1).mean()), 3)})

        out[nm] = {
            "register_x_frame_cm": np.round(fmap, 2).tolist(),
            "register_x_joint_cm": {knames[j]: [round(float(jmap[r, j]), 2)
                                                for r in range(R)]
                                    for j in range(len(knames))},
            "time_locality_own_slice_frac": [round(float(x), 3) for x in own],
            "time_locality_chance": round(1.0 / R, 3),
            "time_register_swap_frame_cm": (np.round(trow, 2).tolist()
                                            if trow is not None else None),
            "strongest_register": r_star,
            "digit_step_per_joint_cm": {f"dim{d}": v
                                        for d, v in enumerate(digit_eff)},
            "code_freq_vs_info": freq_info,
        }
        del m; torch.cuda.empty_cache()

    # ---------------------------------------------------------------- figure --
    os.makedirs(FIG, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, len(out), figsize=(6.4 * len(out), 4.4))
    for a, (nm, o) in zip(np.atleast_1d(ax), out.items()):
        fm = np.asarray(o["register_x_frame_cm"])
        im = a.imshow(fm, aspect="auto", cmap="magma")
        a.set_title(f"{nm}: swap register r -> displacement per frame [cm]\n"
                    f"own-slice frac {np.mean(o['time_locality_own_slice_frac']):.2f}"
                    f" (chance {o['time_locality_chance']:.2f})", fontsize=10)
        a.set_xlabel("frame"); a.set_ylabel("register swapped")
        plt.colorbar(im, ax=a, fraction=0.03)
    fig.tight_layout()
    p = f"{FIG}/remote_control_maps.png"
    fig.savefig(p, dpi=130); plt.close(fig)

    hdr = ["=" * 100, "REMOTE-CONTROL DETECTIVE — swap one register, watch the body",
           "=" * 100]
    lines = []
    for nm, o in out.items():
        lines.append(f"\n[{nm}]  own-slice locality per register "
                     f"(chance {o['time_locality_chance']}):")
        lines.append("  " + " ".join(f"{x:.2f}" for x in
                                     o["time_locality_own_slice_frac"]))
        lines.append(f"  strongest register: #{o['strongest_register']}")
        lines.append(f"  joint map (cm, rows=joint, per register):")
        for jn, vals in o["register_x_joint_cm"].items():
            lines.append(f"    {jn:<10} " + " ".join(f"{v:5.2f}" for v in vals))
    txt = "\n".join(hdr + lines)
    print(txt, flush=True)
    open(f"{V.RUNS}/REMOTE_verdict.txt", "w").write(txt + "\n")
    json.dump(out, open(f"{V.RUNS}/REMOTE_verdict.json", "w"), indent=1)
    print(f"\nfigure: {p}\nverdict: {V.RUNS}/REMOTE_verdict.txt", flush=True)


if __name__ == "__main__":
    main()
