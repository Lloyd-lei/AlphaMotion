"""Side-by-side DECODE videos for the Greenwich-v2 pilot ablation arms (0831).

4 panels, kinematic qpos playback on GMR's g1_mocap_29dof.xml:
    GT (native G1 retarget from the paired cache) | base | fk | fkvn

Decode path = v2_judge's exactly (same fixed windows v2_windows.json, same
load_emb/reach/geom, human_smpl -> unitree_g1_29dof, the holdout direction).
Rotations -> qpos = the repo's hinge projection: manifold_project.fit_angles
(method="global", whole-chain LM — the documented choice for model predictions)
with the cache's {emb}_dof.npz (slot_joints maps slots -> MJCF hinge names).

ROOT: all four panels play the GT world root (pos from *_root_world.pt, quat from
the GT global root rotation). The pilots do not decode world root; differences
shown are pure joint decode quality.

Output (raw, labels added afterwards with ffmpeg drawtext):
    <scratch>/raw_<tag>.mp4        1920x480, 30fps, [normal speed + 0.5x replay]
    prints an 8-frame index list for the grid PNG step.
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco as mj
import imageio.v2 as iio
from scipy.spatial.transform import Rotation

from common import rot6d_to_matrix
from dual_common import build_model, geom
from manifold_project import fit_angles, load_dof
import v2_judge as VJ

AX = np.array([[0., 1, 0], [0, 0, 1], [1, 0, 0]])       # v_ardy = AX @ v_mj
XML = "/var/tmp/alphamotion/config_harvest/GMR/assets/unitree_g1/g1_mocap_29dof.xml"
RUNS = "/var/tmp/alphamotion/topo_exp/runs"
ARMS = [("base", f"{RUNS}/P0V2_base_0830"),
        ("fk", f"{RUNS}/P0V2_fk_0830"),
        ("fkvn", f"{RUNS}/P1V2_fkvn_0830")]
TGT = "unitree_g1_29dof"
SRC = VJ.SRC
T = VJ.T
# window index in v2_windows.json -> content tag (picked by source clip name)
PICKS = {"walk": 7,      # walk3_subject1 @460017 (root path 1.6 m, steady gait)
         "ground": 17,   # aiming1_subject4 @11023 (pelvis 37-42 cm the whole window = ground-level work)
         "dyn": 15}      # sprint1_subject4 @402964 (root path 9.7 m, real sprint)
SCRATCH = sys.argv[1] if len(sys.argv) > 1 else "/tmp"
PANEL_W, PANEL_H = 480, 480
GRID_IDX = np.linspace(0, T - 1, 8).round().astype(int)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    win = json.load(open(VJ.WIN_FILE))
    assert win["T"] == T
    picks = {tag: win["windows"][i] for tag, i in PICKS.items()}
    for tag, w in picks.items():
        print(f"[pick] {tag}: {w['clip']} @ {w['start']}")

    rows_per = {tag: torch.arange(w["start"], w["start"] + T) for tag, w in picks.items()}
    rows = torch.cat(list(rows_per.values()))
    seg = {tag: slice(i * T, (i + 1) * T) for i, tag in enumerate(picks)}

    # ---- data via the judge's own loader (config of any arm; caches identical) ----
    cfg0 = json.load(open(f"{ARMS[0][1]}/config.json"))
    caches = [c for c in str(cfg0["cache"]).split(",") if c]
    assert "cache_xwbc_lafan" in os.path.basename(caches[0].rstrip("/"))
    from semantics import build_name_embeddings
    names_all = []
    for n in [SRC, TGT]:
        d = np.load(f"{caches[0]}/{n}_spec.npz", allow_pickle=True)
        names_all += [str(x) for x in d["joint_names"]]
    lut = build_name_embeddings(names_all, device=dev, encoder=cfg0.get("sem_encoder", "siglip"))
    data = {n: VJ.load_emb(n, rows, caches, cfg0, lut, dev) for n in [SRC, TGT]}
    spec = data[TGT]["spec"]

    # ---- decode each arm on all picked rows (judge's decode call, verbatim) ----
    streams = {"GT": data[TGT]["pose"][..., :6]}
    for tag, ck in ARMS:
        cfg = json.load(open(f"{ck}/config.json"))
        model = build_model(cfg, dev)
        model.load_state_dict(torch.load(f"{ck}/model.pt", map_location=dev))
        model.eval()
        with torch.no_grad():
            B = len(rows)
            prot, _ = model.decode(*model.encode(data[SRC]["pose"], geom(data, SRC, B)),
                                   geom(data, TGT, B))
        streams[tag] = prot
        del model

    # ---- rotations -> hinge angles (whole-chain LM projection, repo default) ----
    dofz = np.load(f"{caches[0]}/{TGT}_dof.npz", allow_pickle=True)
    slot_joints = [str(s) for s in dofz["slot_joints"]]
    dof, rest = load_dof(caches[0], TGT, device=dev)
    angles = {}
    for tag, r6 in streams.items():
        q, active = fit_angles(r6, spec, dof, rest=rest, clamp=True, method="global")
        angles[tag] = q.cpu().numpy()
        # sanity: FK-vs-target geodesic after projection (GT should be ~0)
        from manifold_project import fk_from_angles
        with torch.no_grad():
            gR, _ = fk_from_angles(q, spec, dof, rest=rest,
                                   root_R=rot6d_to_matrix(r6)[:, 0])
            from common import geodesic_deg
            gd = geodesic_deg(gR, rot6d_to_matrix(r6))
        print(f"[proj] {tag}: post-projection geodesic mean={gd.mean():.2f} deg "
              f"max={gd.max():.2f} deg")

    # ---- GT root (world) for ALL panels ----
    rw = torch.load(f"{caches[0]}/{TGT}_root_world.pt", mmap=True)[rows].numpy()  # cm ardy Y-up
    root_pos = (rw / 100.0) @ AX                      # v_mj = AX^T v_ardy  (row-vec: @ AX)
    Rroot = rot6d_to_matrix(streams["GT"][:, 0]).cpu().numpy()
    Rmj = np.einsum("ij,tjk,kl->til", AX.T, Rroot, AX)
    root_quat = Rotation.from_matrix(Rmj).as_quat(scalar_first=True)  # wxyz

    # ---- qpos assembly ----
    model0 = mj.MjModel.from_xml_path(XML)
    adr = {}
    for k, sj in enumerate(slot_joints):
        for s, jn in enumerate([x for x in sj.split(",") if x]):
            jid = mj.mj_name2id(model0, mj.mjtObj.mjOBJ_JOINT, jn)
            assert jid >= 0 and model0.jnt_type[jid] == mj.mjtJoint.mjJNT_HINGE, jn
            adr[(k, s)] = int(model0.jnt_qposadr[jid])
    nq = model0.nq
    qpos = {}
    for tag, q in angles.items():
        Q = np.zeros((len(rows), nq), np.float32)
        Q[:, 0:3] = root_pos
        Q[:, 3:7] = root_quat
        for (k, s), a in adr.items():
            Q[:, a] = q[:, k, s]
        qpos[tag] = Q

    # ---- render scene ----
    sp = mj.MjSpec.from_file(XML)              # XML has its own checker floor
    sp.worldbody.add_light(name="_sun", pos=[0, -1, 3], dir=[0, 0.25, -1],
                           diffuse=[0.7, 0.7, 0.7], castshadow=True)
    scene = sp.compile()
    sdata = mj.MjData(scene)
    ren = mj.Renderer(scene, PANEL_H, PANEL_W)
    cam = mj.MjvCamera()
    cam.distance, cam.elevation, cam.azimuth = 3.0, -14, 135
    order = ["GT", "base", "fk", "fkvn"]

    for tag_v, sl in seg.items():
        panels = []
        for arm in order:
            F = []
            for t in range(sl.start, sl.stop):
                sdata.qpos[:] = qpos[arm][t]
                mj.mj_forward(scene, sdata)
                cam.lookat[:] = sdata.qpos[:3] + np.array([0, 0, 0.15])
                ren.update_scene(sdata, cam)
                F.append(ren.render().copy())
            panels.append(F)
        comp = [np.concatenate([p[t] for p in panels], axis=1) for t in range(T)]
        out = f"{SCRATCH}/raw_{tag_v}.mp4"
        w = iio.get_writer(out, fps=30, codec="libx264", quality=8)
        for f in comp:                      # normal speed
            w.append_data(f)
        for f in comp:                      # 0.5x replay
            w.append_data(f); w.append_data(f)
        w.close()
        print("WROTE", out, len(comp), "frames (+0.5x replay)")
        grid = np.concatenate(
            [np.concatenate([comp[i] for i in GRID_IDX[r * 2:r * 2 + 2]], axis=1)
             for r in range(4)], axis=0)    # 4 rows x 2 composites
        iio.imwrite(f"{SCRATCH}/grid_{tag_v}.png", grid[::2, ::2])
        print("WROTE", f"{SCRATCH}/grid_{tag_v}.png")


if __name__ == "__main__":
    main()
