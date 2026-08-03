"""Interactive viser viewer: real G1 mesh (GT + model output) + capsule human,
frame slider + play. Meshes come straight from the compiled MuJoCo model
(model.mesh_vert/mesh_face — already in the frame that geom_xpos/xmat expect),
posed per frame by FK over the qpos produced by g1_qpos.G1QposMapper.

Launch:
  /home/arenalabs/miniconda3/envs/ardy/bin/python viser_mesh_viewer.py \
      --mode translate --seq_idx 12          # then open http://localhost:8093
Flags: --mode {translate,recon} --seq_idx N [--check] (--check: start, verify, exit)
"""
from __future__ import annotations
import os, sys, json, time, argparse
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco as mj
from g1_qpos import (ARDY_G1_XML, G1QposMapper, STAND_H, COL_PRED,
                     set_robot, ardy_to_mj_pos, model_out, gt_out)
from dual_common import load_paired_data, build_model, geom

PORT = 8093


def robot_frame_geoms(mapper, angles, root_quat, root_pos):
    """FK a whole sequence -> per-frame poses of every visual mesh geom.
    Returns (meshes: [(verts, faces)], pos [T,G,3], wxyz [T,G,4])."""
    model = mj.MjModel.from_xml_path(mapper.xml)
    data = mj.MjData(model)
    gids = [g for g in range(model.ngeom)
            if model.geom_type[g] == mj.mjtGeom.mjGEOM_MESH and model.geom_group[g] == 2]
    meshes = []
    for g in gids:
        m = model.geom_dataid[g]
        va, vn = model.mesh_vertadr[m], model.mesh_vertnum[m]
        fa, fn = model.mesh_faceadr[m], model.mesh_facenum[m]
        meshes.append((model.mesh_vert[va:va + vn].copy(), model.mesh_face[fa:fa + fn].copy()))
    T = len(angles)
    pos = np.zeros((T, len(gids), 3)); wxyz = np.zeros((T, len(gids), 4))
    for t in range(T):
        set_robot(model, data, "", mapper, angles[t], root_quat[t], root_pos)
        mj.mj_forward(model, data)
        for k, g in enumerate(gids):
            pos[t, k] = data.geom_xpos[g]
            R = data.geom_xmat[g].reshape(3, 3)
            from scipy.spatial.transform import Rotation
            q = Rotation.from_matrix(R).as_quat(scalar_first=True)
            wxyz[t, k] = q
    return meshes, pos, wxyz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="/var/tmp/alphamotion/topo_exp/runs/FIX_translate")
    ap.add_argument("--cache", default="/var/tmp/alphamotion/topo_exp/cache_paired_seq")
    ap.add_argument("--mode", choices=["translate", "recon"], default="translate")
    ap.add_argument("--seq_idx", type=int, default=12)
    ap.add_argument("--max_frames", type=int, default=120)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--check", action="store_true", help="start server, verify, exit")
    args = ap.parse_args()
    device = "cuda"

    data, names, N = load_paired_data(args.cache, device)
    g1 = "g1_30"
    mapper = G1QposMapper(data[g1]["spec"])
    seqs = json.load(open(f"{args.cache}/sequences.json"))
    s = seqs[args.seq_idx]
    rows = torch.arange(s["start"], s["start"] + min(s["len"], args.max_frames), args.stride)
    print(f"sequence '{s['name']}' {len(rows)} frames")

    cfg = json.load(open(f"{args.run}/config.json"))
    model = build_model(cfg, device)
    model.load_state_dict(torch.load(f"{args.run}/model.pt", map_location=device), strict=False)
    model.eval()
    src = "soma77" if args.mode == "translate" else g1
    pr_rot, _ = model_out(model, data, src, g1, rows)
    gt_rot, _ = gt_out(data, g1, rows)
    pr_ang, pr_quat, _ = mapper.angles_from_global(pr_rot)
    gt_ang, gt_quat, _ = mapper.angles_from_global(gt_rot)

    gap = 1.0
    streams = {"pred": (pr_ang, pr_quat, np.array([0, -gap / 2, STAND_H]), (242, 140, 40)),
               "gt": (gt_ang, gt_quat, np.array([0, gap / 2, STAND_H]), (150, 150, 155))}
    geo = {k: robot_frame_geoms(mapper, a, q, p) for k, (a, q, p, _) in streams.items()}

    # capsule human (soma77 GT) as line segments
    human_segs = None
    if args.mode == "translate":
        _, spos = gt_out(data, "soma77", rows)
        pw = ardy_to_mj_pos(spos) + np.array([0, -3 * gap / 2, STAND_H])
        cspec = data["soma77"]["spec"]
        pairs = [(p, j) for j, p in enumerate(cspec.parents) if p >= 0]
        human_segs = np.stack([np.stack([pw[:, p], pw[:, j]], axis=1) for p, j in pairs], axis=1)

    import viser
    server = viser.ViserServer(port=PORT)
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/floor", width=8, height=8, plane="xy")
    handles = {}
    for k, (meshes, pos, wxyz) in geo.items():
        col = streams[k][3]
        handles[k] = [server.scene.add_mesh_simple(f"/{k}/g{i}", vertices=v, faces=f, color=col)
                      for i, (v, f) in enumerate(meshes)]
    hseg = None
    if human_segs is not None:
        hseg = server.scene.add_line_segments(
            "/human", points=human_segs[0].astype(np.float32),
            colors=np.full_like(human_segs[0], 120, dtype=np.uint8))

    T = len(rows)
    slider = server.gui.add_slider("frame", min=0, max=T - 1, step=1, initial_value=0)
    play = server.gui.add_checkbox("play", initial_value=True)

    def show(t):
        for k, (meshes, pos, wxyz) in geo.items():
            for i, h in enumerate(handles[k]):
                h.position = pos[t, i]
                h.wxyz = wxyz[t, i]
        if hseg is not None:
            hseg.points = human_segs[t].astype(np.float32)

    slider.on_update(lambda _: show(slider.value))
    show(0)
    print(f"viser viewer up: http://localhost:{PORT}  ({args.mode}, seq {args.seq_idx})")
    if args.check:
        time.sleep(2)
        print("CHECK OK"); server.stop(); return
    while True:
        if play.value:
            slider.value = (slider.value + 1) % T
        time.sleep(args.stride / 30.0)


if __name__ == "__main__":
    main()
