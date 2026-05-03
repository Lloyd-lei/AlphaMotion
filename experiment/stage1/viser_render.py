"""Standalone viser renderer for MotionFormer reconstructions.

Shows joint spheres + skeletal bones in the browser at http://localhost:7861.
Can compare ground truth vs model reconstruction side-by-side.

Usage:
    python viser_render.py runs/reconstruction/compare_sample100.gif   # (npz)
    python viser_render.py runs/reconstruction/sample100_random.npz
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import viser

from soma_skeleton import build_parent_index, JOINT_NAMES


def main():
    p = argparse.ArgumentParser()
    p.add_argument("npz", help="NPZ from visualize_reconstruction.py")
    p.add_argument("--port", type=int, default=7861)
    p.add_argument("--bone-thickness", type=float, default=0.012)
    p.add_argument("--joint-radius", type=float, default=0.020)
    p.add_argument("--center-root", action="store_true", default=True,
                   help="Subtract root position at each frame so person stays in place.")
    args = p.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    gt = data["gt"]                      # [T, J, 3]
    pred = data["composite"]             # [T, J, 3]
    mask = data["mask"] if "mask" in data else None
    prompt = str(data["prompt"]) if "prompt" in data else "(no prompt)"

    if args.center_root:
        gt = gt - gt[:, :1, :]
        pred = pred - pred[:, :1, :]

    T, J, _ = gt.shape
    parents = build_parent_index()

    # Compute a reasonable offset to put GT and pred side by side
    extent_x = max(gt[..., 0].max() - gt[..., 0].min(),
                   pred[..., 0].max() - pred[..., 0].min())
    offset_x = extent_x * 1.6 + 0.5

    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    print(f"\n>>> Viser running at  http://localhost:{args.port}\n")
    print(f"Prompt:  {prompt}")
    print(f"Frames:  {T}\n")

    # GUI: a playback slider + auto-play toggle
    with server.gui.add_folder("Playback"):
        gui_frame = server.gui.add_slider("Frame", min=0, max=T - 1, step=1, initial_value=0)
        gui_play = server.gui.add_checkbox("Auto-play", initial_value=True)
        gui_fps = server.gui.add_slider("FPS", min=5, max=60, step=1, initial_value=20)
    with server.gui.add_folder("Info"):
        server.gui.add_markdown(f"**Prompt:** {prompt}")
        server.gui.add_markdown(f"**T** = {T}, **J** = {J}")
        server.gui.add_markdown("Left (green): ground truth\n\nRight (blue): model reconstruction")
        if mask is not None:
            server.gui.add_markdown(f"**Masked fraction (avg):** {mask.mean():.2%}")

    # Pre-register joint and bone handles for each side
    # We just use spheres for joints and line_segments for bones.
    joint_handles_gt = []
    joint_handles_pr = []
    bones_gt = None
    bones_pr = None

    def make_bones_points(joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (N_bones, 2, 3) point pairs for each bone."""
        pts = []
        for j, p in enumerate(parents):
            if p >= 0:
                pts.append(np.stack([joints[j], joints[p]], axis=0))
        return np.stack(pts, axis=0) if pts else np.zeros((0, 2, 3))

    def set_frame(frame_idx: int):
        nonlocal bones_gt, bones_pr
        joints_gt_f = gt[frame_idx].copy()
        joints_pr_f = pred[frame_idx].copy()
        # Offset pred to the right
        joints_pr_f = joints_pr_f + np.array([offset_x, 0, 0])

        # Spheres (one batched "point cloud" node per side)
        server.scene.add_point_cloud(
            "/gt_joints", points=joints_gt_f,
            colors=np.tile(np.array([[60, 220, 100]]), (J, 1)),
            point_size=args.joint_radius, point_shape="sparkle",
        )
        server.scene.add_point_cloud(
            "/pr_joints", points=joints_pr_f,
            colors=np.tile(np.array([[70, 130, 240]]), (J, 1)),
            point_size=args.joint_radius, point_shape="sparkle",
        )

        # Bones as line segments
        bones_gt_pts = make_bones_points(joints_gt_f)
        bones_pr_pts = make_bones_points(joints_pr_f)
        server.scene.add_line_segments(
            "/gt_bones",
            points=bones_gt_pts,
            colors=np.tile(np.array([[40, 180, 80]]), (bones_gt_pts.shape[0], 2, 1)),
            line_width=3.0,
        )
        server.scene.add_line_segments(
            "/pr_bones",
            points=bones_pr_pts,
            colors=np.tile(np.array([[50, 100, 220]]), (bones_pr_pts.shape[0], 2, 1)),
            line_width=3.0,
        )

    # Labels
    server.scene.add_label("/gt_label", text="ground truth",
                           position=(0, 0, 1.5))
    server.scene.add_label("/pr_label", text="reconstruction",
                           position=(offset_x, 0, 1.5))

    set_frame(0)

    # Playback loop
    last_frame_time = time.time()
    try:
        while True:
            now = time.time()
            dt = now - last_frame_time
            if gui_play.value and dt >= 1.0 / gui_fps.value:
                new_frame = (gui_frame.value + 1) % T
                gui_frame.value = int(new_frame)
                last_frame_time = now
            set_frame(int(gui_frame.value))
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("Shutting down viser server")


if __name__ == "__main__":
    main()
