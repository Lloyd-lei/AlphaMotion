"""EXAM C, MESH EDITION — every exam-C claim re-proven as a robot-mesh video.
(lloyd's discipline 2026-08-07: stick figures are debug output, not evidence.)

    exam_c_mesh.py --modes m1,m2,m3,m5,m6

CHOICE OF EMBODIMENTS — the zero-shot ladder, chosen so every video carries its own
control group:
  agibot_x2_ultra   THE zero-shot body: in no dataset, ever, built from its URDF at
                    eval time. If a mode works here it works from a spec sheet alone.
  unitree_g1_29dof  held out of V7 training but paired data exists — separates
                    "never trained" from "never even seen the skeleton family".
(pnd_adam is skipped here: exam C is about generalization, in-training bodies proved
their point in the bridge-mesh videos.)

CHOICE OF MOTIONS — all from the VAL split (never trained), families chosen per mode
for maximum diagnostic value:
  m1 dance + kick   dance is the measured worst family (11-12 cm) — show the model's
                    weakest hand, not its best; kick is the mean-regression family.
  m2 kick           the mode where the prior OVER-invents (energy 2.2x GT) — the
                    video decides whether that is a plausible kick or a convulsion.
  m3 kick fwd       the [OOD] single-endpoint case that partially works.
  m5 kick + jacks   keyframe sketch -> fill; retrieval verdict printed in banner.
  m6 kick           same tokens at 0.5x/1x/2x; finished variants FREEZE and their
                    label turns grey — pausing is the proof of speed control.

Rendering contract (same as v7_bridge_mesh): feasibility-projected angles written
into the robot's own MJCF qpos -> MuJoCo FK; per-frame floor contact; camera square
to the line of robots so equal size means equal size.

Output: R4_VIZ/mp4/exam_c_mesh/*.mp4
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES",
                      "/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
import mujoco as mj                                                  # noqa: E402
import v5_common as V                                                # noqa: E402
import v7_bridge_mesh as BM                                          # noqa: E402
import eval_newbody as E                                             # noqa: E402
import render_newbody as RN                                          # noqa: E402
from train_v5 import load_run, to_states                             # noqa: E402

OUT = ('/home/arenalabs/Desktop/"be water, robot"/pose_translator_design/'
       'R4_VIZ/mp4/exam_c_mesh')
EMB = "human_smpl"
TARGETS = {
    "x2": (BM.TARGETS["x2"][0], "agibot_x2_ultra (IN NO DATASET)"),
    "g1": (BM.TARGETS["g1"][0], "unitree_g1 (HELD OUT)"),
    "adam": (BM.TARGETS["adam"][0], "pnd_adam_lite (IN training)"),
}
RGBA = [[0.20, 0.22, 0.25, 1],        # GT — near-black
        [0.42, 0.55, 0.72, 1],        # blue
        [0.95, 0.72, 0.20, 1],        # amber
        [0.30, 0.70, 0.45, 1],        # green
        [0.62, 0.44, 0.72, 1],        # purple
        [0.85, 0.45, 0.30, 1],        # rust
        [0.35, 0.60, 0.68, 1],        # teal
        [0.72, 0.68, 0.30, 1]]        # olive


class MeshCtx:
    def __init__(self, dev):
        self.dev = dev
        self.kd = V.KeyData(V.AMASS, [EMB], device="cpu")
        self.tok, _ = V.load_frozen(dev)
        self.src = V.body_entry(V.AMASS, EMB, dev, self.kd.reach[EMB])
        self.kidx = self.kd.kidx[EMB]
        self.models = {a.replace("_0807", ""): load_run(f"{V.RUNS}/{a}", dev)[0]
                       for a in ("V7_jepa_0807", "V7_maskfsq_0807")}
        self.tg = {}
        for k, (xml, label) in TARGETS.items():
            spec, dof, rest, qnames, xml2 = E.get_target(xml, E.CACHE_AMASS)
            g = E.target_geom(spec, dof, dev, sem_encoder=V.frozen_sem_encoder())
            self.tg[k] = dict(
                label=label, xml=xml2, spec=spec, qnames=qnames,
                ent={"spec": spec, "st": g["st"], "sem": g["sem"], "dof": g["dof"]},
                dof=torch.as_tensor(dof, device=dev, dtype=torch.float64),
                rest=torch.as_tensor(rest, device=dev, dtype=torch.float64))

    def win(self, fam, T, g, n=1):
        st, _ = self.kd.sample_starts(n, g, "val", span=T, families=[fam])
        return self.kd.window(EMB, st, T, T)

    def render(self, tkey, seqs, name, header_fn, note, fps=20,
               size=(1700, 780), gap=1.05, dist=3.9):
        """seqs: ordered {label: [T,nk,9]}. One robot copy per entry."""
        t_ = self.tg[tkey]
        streams, labels = [], []
        for k, v in seqs.items():
            if isinstance(v, tuple) and len(v) == 3:      # precomputed (q, rootR, gp)
                q, rootR, gp = v
            else:
                q, rootR, gp, _ = BM.robot_stream(self.tok, self.src, t_["ent"], v,
                                                  self.kidx, t_["spec"], t_["dof"],
                                                  t_["rest"], self.dev)
            streams.append((q, rootR, gp)); labels.append(k)
        n = len(streams)
        prefixes = [f"m{i}_" for i in range(n)]
        m = RN.build_scene(t_["xml"], prefixes, RGBA[:n], size[0], size[1])
        d = mj.MjData(m)
        rend = mj.Renderer(m, height=size[1], width=size[0])
        cam = mj.MjvCamera()
        cam.lookat[:] = [0, 0, 0.85]
        cam.distance, cam.elevation, cam.azimuth = dist, -6, 180
        root_adr, slot_adr = [], []
        for p in prefixes:
            fr = [int(m.jnt_qposadr[j]) for j in range(m.njnt)
                  if (mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, j) or "").startswith(p)
                  and m.jnt_type[j] == mj.mjtJoint.mjJNT_FREE]
            root_adr.append(fr[0])
            tab = -np.ones((t_["spec"].J, 3), np.int64)
            for j, ns in enumerate(t_["qnames"]):
                for k2, nm2 in enumerate(ns):
                    jid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, p + nm2)
                    if jid >= 0:
                        tab[j, k2] = int(m.jnt_qposadr[jid])
            slot_adr.append(tab)
        ys = [(i - (n - 1) / 2) * gap for i in range(n)]
        # MAX, not min: panels may legitimately differ in length (m6 speed sweep);
        # shorter ones hold their last frame via the tt clamp below. min() would cut
        # the 2x-slow variant in half.
        T = max(s[0].shape[0] for s in streams)
        frames = []
        for t in range(T):
            for i, (q, rootR, gp) in enumerate(streams):
                tt = min(t, q.shape[0] - 1)
                pm = gp[tt] @ RN.AX / 100.0
                d.qpos[root_adr[i]:root_adr[i] + 3] = [0.0, ys[i],
                                                      -float(pm[:, 2].min())]
                d.qpos[root_adr[i] + 3:root_adr[i] + 7] = RN.mat_to_quat(
                    RN.AX.T @ rootR[tt] @ RN.AX)
                tab = slot_adr[i]
                for j in range(t_["spec"].J):
                    for k2 in range(3):
                        if tab[j, k2] >= 0:
                            d.qpos[tab[j, k2]] = q[tt, j, k2]
            mj.mj_forward(m, d)
            rend.update_scene(d, cam)
            img = rend.render().copy()
            hdr, nts = header_fn(t, T)
            labs = [(labels[i], tuple(int(255 * c) for c in RGBA[i][:3]))
                    for i in range(n)]
            img = RN.annotate(img, labs, f"{t_['label']}  |  {hdr}", note,
                              [int((0.5 + i) * img.shape[1] / n) for i in range(n)],
                              nts if nts else [("", (0, 0, 0))] * n)
            frames.append(img)
        p = f"{OUT}/{name}_{tkey}.mp4"
        RN.write_mp4(p, np.stack(frames), fps)
        print(f"  -> {p}", flush=True)
        del rend, m, d
        return p


def plain_hdr(tag):
    return lambda t, T: (f"{tag}  |  frame {t}/{T-1}", None)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="m1,m2,m3,m5,m6")
    ap.add_argument("--targets", default="x2,g1")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    C = MeshCtx(args.device)
    dev, T = args.device, V.T_WIN
    g = np.random.default_rng(2027)
    kn = torch.ones(1, device=dev)
    modes = args.modes.split(",")
    tkeys = args.targets.split(",")

    def enc_dec(m, w, cur, tgt, T_out=None, zero_tokens=False):
        e = m.encode(w, cur, tgt, kn)
        pz = torch.zeros_like(e["path_z"]) if zero_tokens else e["path_z"]
        tz = (torch.zeros_like(e["time_z"]) if zero_tokens and e["time_z"].shape[1]
              else e["time_z"])
        o = m.decode(pz, tz, cur, tgt, kn, T_out or w.shape[1])
        return to_states(o, m, cur, tgt, kn).float()[0].cpu()

    if "m1" in modes:
        for fam in ("dance", "kick"):
            w = C.win(fam, T, g).to(dev).float()
            cur, tgt = w[:, 0], w[:, -1]
            seqs = {"GT (real data)": w[0].cpu()}
            for nm, m in C.models.items():
                seqs[nm] = enc_dec(m, w, cur, tgt)
            for tk in tkeys:
                C.render(tk, seqs, f"m1_{fam}", plain_hdr(f"m1 codec: {fam}"),
                         "val clip -> 125 bits -> motion")

    if "m2" in modes:
        fam = "kick"
        w = C.win(fam, T, g).to(dev).float()
        cur, tgt = w[:, 0], w[:, -1]
        m0 = next(iter(C.models.values()))
        seqs = {"GT (model never saw)": w[0].cpu(),
                "straight line": m0.baseline(cur, tgt, kn, T).float()[0].cpu()}
        for nm, m in C.models.items():
            seqs[f"{nm} prior-fill"] = enc_dec(m, w, cur, tgt, zero_tokens=True)
        for tk in tkeys:
            C.render(tk, seqs, "m2_kick", plain_hdr("m2 prior: endpoints ONLY, 0 tokens"),
                     "endpoints only, 0 tokens")

    if "m3" in modes:
        fam = "kick"
        w = C.win(fam, T, g).to(dev).float()
        a = w[:, 0]
        seqs = {"GT": w[0].cpu()}
        for nm, m in C.models.items():
            e = m.encode(w, a, a, kn)
            o = m.decode(e["path_z"], e["time_z"], a, a, kn, T)
            seqs[f"{nm} fwd"] = to_states(o, m, a, a, kn).float()[0].cpu()
        for tk in tkeys:
            C.render(tk, seqs, "m3_kick_fwd", plain_hdr("m3 [OOD]: frame 0 + tokens only"),
                     "no target frame; 125 bits only")

    if "m5" in modes:
        import exam_c as XC
        for fam in ("kick", "jacks"):
            w = C.win(fam, T, g).to(dev).float()
            kf = np.round(np.linspace(0, T - 1, 4)).astype(int)
            m0 = next(iter(C.models.values()))
            segs = []
            for i in range(3):
                L = kf[i + 1] - kf[i] + 1
                s = m0.baseline(w[:, kf[i]], w[:, kf[i + 1]], kn, L)
                segs.append(s[:, :-1] if i < 2 else s)
            sk = torch.cat(segs, 1)
            seqs = {"GT": w[0].cpu(), "sketch (4 keyframes)": sk[0].cpu()}
            for nm, m in C.models.items():
                seqs[f"{nm} fill"] = enc_dec(m, sk, sk[:, 0], sk[:, -1])
            for tk in tkeys:
                C.render(tk, seqs, f"m5_{fam}",
                         plain_hdr(f"m5 vocabulary: 4 keyframes -> fill ({fam})"),
                         "token retrieval: family 6/6")

    if "m6" in modes:
        fam = "kick"
        w = C.win(fam, T, g).to(dev).float()
        cur, tgt = w[:, 0], w[:, -1]
        seqs = {"GT (1x)": w[0].cpu()}
        lens = {}
        for nm, m in C.models.items():
            for T_out, lab in ((30, "0.5x"), (120, "2x")):
                if nm == "V7_maskfsq" and T_out == 30:
                    continue                      # keep it to 4 panels: GT, j0.5, j2, m2
                s = enc_dec(m, w, cur, tgt, T_out=T_out)
                key = f"{nm} {lab}"
                lens[key] = T_out
                seqs[key] = torch.cat([s, s[-1:].expand(120 - T_out, -1, -1)]) \
                    if T_out < 120 else s

        def hdr6(t, TT):
            nts = []
            for k in seqs:
                L = lens.get(k, 60)
                done = t >= L
                nts.append(("DONE — holding" if done else f"{min(t,L)}/{L}",
                            (150, 155, 162) if done else (240, 240, 240)))
            return f"m6 speed: SAME tokens, different clock  |  frame {t}", nts
        for tk in tkeys:
            C.render(tk, seqs, "m6_kick_speeds", hdr6,
                     "same bits, 0.5x vs 2x clock")

    print("\nall mesh videos in", OUT, flush=True)


if __name__ == "__main__":
    main()
