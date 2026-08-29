"""MOTION LAB — an interactive viser front end for judging the temporal models.

    lab_server.py            # then open http://localhost:8114

lloyd, 2026-08-08: "双向补全的质量，不是你说数值谁好就谁好的" — so this is a viewer,
not a report. Everything the exams measured is a knob here, and you look at it.

WHAT YOU CAN DO
  * pick any ROBOT (X2 / G1 / Adam / T1 / H1 / Kuavo / HighTorque / H1-2) and any
    MOTION (any family, any val clip), reload live
  * build a TIMELINE by hand: the clip is cut into 4 segments and you choose what
    each segment is — a real motion, or a MODEL-FILLED gap. That is the
    "由我来选择每一个时刻要放上什么动作" part.
  * for every gap, choose what the encoder is allowed to SEE:
      full        the whole window (plain reconstruction)
      endpoints   only the two boundary frames
      prefix/suffix/middle/sparse-k   partial observation, the JEPA-vs-MSBA decider
  * compare ARMS side by side in the same scene (jepa / maskfsq / arclen / V9),
    each its own colour, plus GT in grey. Toggle any of them.
  * scrub / play / loop, and a per-arm live readout of error on the frames the
    encoder never saw.

Everything is decoded through the frozen spatial layer and feasibility-projected
into the selected robot's own joint limits, i.e. the same pipeline the mp4s use.
"""
from __future__ import annotations
import os, sys, time, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES",
                      "/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
import mujoco as mj                                                  # noqa: E402
import v5_common as V                                                # noqa: E402
import v7_bridge_mesh as BM                                          # noqa: E402
import v7_bridge as B7                                               # noqa: E402
import eval_newbody as E                                             # noqa: E402
import render_newbody as RN                                          # noqa: E402
import manifold_project as MP                                        # noqa: E402
from common import rot6d_to_matrix                                   # noqa: E402
from train_v5 import load_run, to_states                             # noqa: E402

PORT = 8114
EMB = "human_smpl"
GMR = "/var/tmp/alphamotion/config_harvest/GMR/assets"
ROBOTS = {
    "AgiBot X2  (no dataset)":  BM.TARGETS["x2"][0],
    "Unitree G1  (held out)":   BM.TARGETS["g1"][0],
    "PND Adam":                 BM.TARGETS["adam"][0],
    "Booster T1":               f"{GMR}/booster_t1/T1_locomotion.xml",
    "Unitree H1":               f"{GMR}/unitree_h1/h1_with_hand.xml",
    "Kuavo S45":                f"{GMR}/kuavo_s45/biped_s45_collision.xml",
    "HighTorque Hi":            f"{GMR}/hightorque_hi/hi_25dof.xml",
    "Unitree H1-2":             f"{GMR}/unitree_h1_2/h1_2_handless.xml",
}
ARM_RUNS = [("jepa", "V7_jepa_0807", (245, 190, 60)),
            ("maskfsq", "V7_maskfsq_0807", (90, 190, 130)),
            ("arclen (V8)", "V8_arclen_0807", (110, 170, 235)),
            ("V9 arclen pure-timing", "V9_arclen_pt_0808", (200, 130, 220)),
            ("V9 jepa pure-timing", "V9_jepa_pt_0808", (230, 120, 110))]
OBS_MODES = ["full", "endpoints only", "prefix", "suffix", "middle", "sparse"]
FAMS = ["kick", "walk", "run", "jacks", "dance", "punch", "other"]


def enc_mask_for(mode, k, T, dev):
    """-> bool [T] True = hidden from the encoder, or None for 'full'."""
    if mode == "full":
        return None
    m = np.ones(T, bool)
    if mode == "endpoints only":
        m[0] = m[-1] = False
    elif mode == "prefix":
        m[:k] = False
    elif mode == "suffix":
        m[T - k:] = False
    elif mode == "middle":
        s = (T - k) // 2
        m[s:s + k] = False
    elif mode == "sparse":
        m[np.linspace(0, T - 1, k).round().astype(int)] = False
    return torch.as_tensor(m, device=dev)


class Lab:
    def __init__(self, dev):
        self.dev = dev
        self.T = V.T_WIN
        self.kd = V.KeyData(V.AMASS, [EMB], device="cpu")
        self.tok, _ = V.load_frozen(dev)
        self.src = V.body_entry(V.AMASS, EMB, dev, self.kd.reach[EMB])
        self.kidx = self.kd.kidx[EMB]
        self.reach = self.kd.reach[EMB]
        self.models = {}
        for label, run, col in ARM_RUNS:
            rd = f"{V.RUNS}/{run}"
            if os.path.exists(f"{rd}/model.pt"):
                try:
                    # model.pt is saved PERIODICALLY, so a still-training run loads a
                    # half-baked checkpoint. Say so in the label rather than letting it
                    # masquerade as a finished arm.
                    lg = f"{V.RUNS}/{run}.log"
                    done = os.path.exists(lg) and "DONE" in open(lg).read()[-4000:]
                    step = ""
                    if not done and os.path.exists(lg):
                        import re
                        mm = re.findall(r"step (\d+)", open(lg).read()[-4000:])
                        step = f" [TRAINING @{mm[-1]}]" if mm else " [TRAINING]"
                    self.models[label + step] = (load_run(rd, dev)[0], col)
                    print(f"[arm] {label}{step}", flush=True)
                except Exception as ex:
                    print(f"[arm] {label} SKIP {ex}", flush=True)
        self.robot_cache = {}

    def robot(self, name):
        if name not in self.robot_cache:
            spec, dof, rest, qnames, xml = E.get_target(ROBOTS[name], E.CACHE_AMASS)
            g = E.target_geom(spec, dof, self.dev)
            self.robot_cache[name] = dict(
                spec=spec, qnames=qnames, xml=xml,
                ent={"spec": spec, "st": g["st"], "sem": g["sem"], "dof": g["dof"]},
                dof=torch.as_tensor(dof, device=self.dev, dtype=torch.float64),
                rest=torch.as_tensor(rest, device=self.dev, dtype=torch.float64))
            print(f"[robot] {name} J={spec.J}", flush=True)
        return self.robot_cache[name]

    def clips(self, fam):
        idx = [i for i, f in enumerate(self.kd.family) if f == fam]
        return idx[:200]

    def window(self, clip_id, off=0):
        c = self.kd.clips[clip_id]
        st = np.array([c["start"] + min(off, max(c["len"] - self.T - 1, 0))])
        return self.kd.window(EMB, st, self.T, self.T).to(self.dev).float()

    @torch.no_grad()
    def decode(self, label, w, obs_mode, k):
        m, _ = self.models[label]
        kn = torch.ones(w.shape[0], device=self.dev)
        em = enc_mask_for(obs_mode, k, self.T, self.dev)
        em = None if em is None else em[None].expand(w.shape[0], -1)
        e = m.encode(w, w[:, 0], w[:, -1], kn, enc_mask=em)
        o = m.decode(e["path_z"], e["time_z"], w[:, 0], w[:, -1], kn, self.T)
        return to_states(o, m, w[:, 0], w[:, -1], kn).float(), em

    @torch.no_grad()
    def to_robot(self, rb, key):
        """key [T,nk,9] -> (qpos-ready q, rootR, gp) via spatial completion +
        feasibility projection into THIS robot's limits."""
        return BM.robot_stream(self.tok, self.src, rb["ent"], key.cpu(), self.kidx,
                               rb["spec"], rb["dof"], rb["rest"], self.dev)[:3]


def mesh_geoms(rb, q, rootR, gp):
    """-> list of per-frame (position, wxyz) for each geom, + static meshes."""
    m = mj.MjModel.from_xml_path(rb["xml"])
    d = mj.MjData(m)
    fr = [j for j in range(m.njnt) if m.jnt_type[j] == mj.mjtJoint.mjJNT_FREE]
    ra = int(m.jnt_qposadr[fr[0]])
    tab = -np.ones((rb["spec"].J, 3), np.int64)
    for j, ns in enumerate(rb["qnames"]):
        for kk, nm in enumerate(ns):
            jid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, nm)
            if jid >= 0:
                tab[j, kk] = int(m.jnt_qposadr[jid])
    T = q.shape[0]
    keep = [gi for gi in range(m.ngeom) if m.geom_dataid[gi] >= 0 or
            m.geom_type[gi] in (mj.mjtGeom.mjGEOM_BOX, mj.mjtGeom.mjGEOM_CAPSULE,
                                mj.mjtGeom.mjGEOM_CYLINDER, mj.mjtGeom.mjGEOM_SPHERE)]
    pos = np.zeros((T, len(keep), 3), np.float32)
    quat = np.zeros((T, len(keep), 4), np.float32)
    for t in range(T):
        pm = gp[t] @ RN.AX / 100.0
        d.qpos[ra:ra + 3] = [0.0, 0.0, -float(pm[:, 2].min())]
        d.qpos[ra + 3:ra + 7] = RN.mat_to_quat(RN.AX.T @ rootR[t] @ RN.AX)
        for j in range(rb["spec"].J):
            for kk in range(3):
                if tab[j, kk] >= 0:
                    d.qpos[tab[j, kk]] = q[t, j, kk]
        mj.mj_forward(m, d)
        for i, gi in enumerate(keep):
            pos[t, i] = d.geom_xpos[gi]
            quat[t, i] = RN.mat_to_quat(d.geom_xmat[gi].reshape(3, 3))
    # static geometry for the meshes themselves
    import trimesh
    tris = []
    for gi in keep:
        did = int(m.geom_dataid[gi])
        if did >= 0:
            va = int(m.mesh_vertadr[did]); vn = int(m.mesh_vertnum[did])
            fa = int(m.mesh_faceadr[did]); fn = int(m.mesh_facenum[did])
            v = m.mesh_vert[va:va + vn].astype(np.float32)
            f = m.mesh_face[fa:fa + fn].astype(np.int32)
        else:
            b = trimesh.creation.box(extents=np.maximum(m.geom_size[gi] * 2, 1e-3))
            v, f = np.asarray(b.vertices, np.float32), np.asarray(b.faces, np.int32)
        tris.append((v, f))
    return tris, pos, quat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    import viser
    lab = Lab(args.device)
    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/floor", width=6, height=6, plane="xy")

    gui = server.gui
    with gui.add_folder("source"):
        g_robot = gui.add_dropdown("robot", tuple(ROBOTS), initial_value=list(ROBOTS)[0])
        g_fam = gui.add_dropdown("family", tuple(FAMS), initial_value="kick")
        g_clip = gui.add_slider("clip #", min=0, max=50, step=1, initial_value=0)
        g_off = gui.add_slider("offset in clip", min=0, max=300, step=10, initial_value=0)
    with gui.add_folder("what the encoder sees"):
        g_obs = gui.add_dropdown("observation", tuple(OBS_MODES), initial_value="full")
        g_k = gui.add_slider("k frames", min=2, max=45, step=1, initial_value=15)
    with gui.add_folder("arms"):
        g_arms = {lab_: gui.add_checkbox(lab_, initial_value=(i < 2))
                  for i, lab_ in enumerate(lab.models)}
        g_gt = gui.add_checkbox("GT (grey)", initial_value=True)
    with gui.add_folder("playback"):
        g_play = gui.add_checkbox("play", initial_value=True)
        g_frame = gui.add_slider("frame", min=0, max=lab.T - 1, step=1, initial_value=0)
        g_load = gui.add_button("RELOAD")
    g_stat = gui.add_markdown("*(load a clip)*")

    state = {"tris": {}, "pos": {}, "quat": {}, "handles": {}, "T": lab.T}

    busy = {"v": False}

    def rebuild():
        # show() walks state["handles"] from the play loop and from every on_update,
        # so mutating those dicts in place raced with it ("dictionary changed size
        # during iteration"). Build fresh, swap, and gate show() on a busy flag.
        if busy["v"]:
            return
        busy["v"] = True
        old = list(state["handles"].values())
        state["handles"], state["pos"], state["quat"] = {}, {}, {}
        for h in old:
            for x in h:
                try:
                    x.remove()
                except Exception:
                    pass
        rb = lab.robot(g_robot.value)
        ids = lab.clips(g_fam.value)
        if not ids:
            busy["v"] = False
            g_stat.content = f"**no clips for {g_fam.value}**"; return
        cid = ids[min(g_clip.value, len(ids) - 1)]
        w = lab.window(cid, g_off.value)
        name = lab.kd.clips[cid]["name"]

        lanes, msg = [], []
        if g_gt.value:
            lanes.append(("GT", w[0], (150, 150, 155)))
        for label in lab.models:
            if not g_arms[label].value:
                continue
            out, em = lab.decode(label, w, g_obs.value, int(g_k.value))
            col = lab.models[label][1]
            lanes.append((label, out[0], col))
            if em is not None:
                sel = em[0].cpu().numpy()
                err = float((out[:, sel, :, 6:9] - w[:, sel, :, 6:9]
                             ).norm(dim=-1).mean() * lab.reach)
                msg.append(f"- **{label}** unseen-frame err **{err:.2f} cm**")
            else:
                err = float((out[..., 6:9] - w[..., 6:9]).norm(dim=-1).mean() * lab.reach)
                msg.append(f"- **{label}** full-window err **{err:.2f} cm**")

        gap = 1.0
        for i, (label, key, col) in enumerate(lanes):
            q, rootR, gp = lab.to_robot(rb, key)
            tris, pos, quat = mesh_geoms(rb, q, rootR, gp)
            pos = pos + np.array([0.0, (i - (len(lanes) - 1) / 2) * gap, 0.0], np.float32)
            hs = [server.scene.add_mesh_simple(
                    f"/{i}_{label}/g{j}", vertices=v, faces=f, color=col)
                  for j, (v, f) in enumerate(tris)]
            state["pos"][label] = pos
            state["quat"][label] = quat
            state["handles"][label] = hs      # handles LAST: show() keys off it
        busy["v"] = False
        state["T"] = min(p.shape[0] for p in state["pos"].values()) if state["pos"] else lab.T
        g_frame.max = state["T"] - 1
        g_stat.content = (f"**{name}**  \n{g_obs.value}"
                          + (f" (k={int(g_k.value)})" if g_obs.value not in
                             ("full", "endpoints only") else "")
                          + "  \n" + "\n".join(msg))

    def show(t):
        # DEFENSIVE ON PURPOSE: rebuild() fills handles/pos/quat lane by
        # lane while the play loop drives show() from another thread, so it
        # can land on a half-inserted lane. The busy flag alone did not
        # close that window (KeyError on a lane in handles but not pos).
        if busy["v"]:
            return
        for label in list(state["handles"]):
            hs = state["handles"].get(label)
            p = state["pos"].get(label)
            qt = state["quat"].get(label)
            if hs is None or p is None or qt is None:
                continue
            tt = min(t, p.shape[0] - 1)
            for j2, h in enumerate(hs):
                try:
                    h.position = p[tt, j2]
                    h.wxyz = qt[tt, j2]
                except Exception:
                    return

    g_load.on_click(lambda _: (rebuild(), show(0)))
    for h in (g_robot, g_fam, g_obs):
        h.on_update(lambda _: (rebuild(), show(int(g_frame.value))))
    g_frame.on_update(lambda _: show(int(g_frame.value)))

    rebuild(); show(0)
    print(f"\nMOTION LAB up:  http://localhost:{args.port}\n", flush=True)
    if args.check:
        time.sleep(2); print("CHECK OK"); return
    while True:
        try:
            if g_play.value and state["T"] > 1:
                g_frame.value = (int(g_frame.value) + 1) % state["T"]
        except Exception as ex:
            print(f"[play] {type(ex).__name__}: {ex}", flush=True)
        time.sleep(1 / 20)


if __name__ == "__main__":
    main()
