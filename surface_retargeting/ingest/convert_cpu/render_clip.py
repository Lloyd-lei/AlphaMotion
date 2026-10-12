"""Render a 12-frame SMPL-X mesh + object grid for a converted clip (either layout).
usage: render_clip.py <converted_seq_dir> <out.png>
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import smplx_qa  # noqa: E402

d = Path(sys.argv[1]); out = sys.argv[2]
qa = json.loads((d / "qa.json").read_text())
objects = []
if (d / "motion_actor_smplx.npz").exists():  # HiPHI layout
    z = np.load(d / "motion_actor_smplx.npz")
    poses, trans, betas, fps = z["poses"], z["trans"], z["betas"], float(z["mocap_framerate"])
    meta = json.loads((d / "metadata.json").read_text())
    root = d.parents[3]
    for o in meta.get("objects", []):
        ov, of = smplx_qa.load_obj_cm(root / o["mesh_path"])
        rows = np.genfromtxt(d / o["trajectory_path"], delimiter=",", names=True)
        tr = np.stack([rows[k] for k in ("px", "py", "pz", "qx", "qy", "qz", "qw")], 1)
        objects.append((o["object_id"], ov, of, tr))
else:  # OmniContact flat layout
    poses, trans, betas = np.load(d / "poses.npy"), np.load(d / "transl.npy"), np.load(d / "betas.npy")
    fps = float(np.load(d / "mocap_framerate.npy"))
    for csvp in sorted(d.glob("prop_*.csv")):
        name = csvp.stem[5:]
        ov, of = smplx_qa.load_obj_cm(d / f"{name}.obj")
        rows = np.genfromtxt(csvp, delimiter=",", names=True)
        tr = np.stack([rows[k] for k in ("px", "py", "pz", "qx", "qy", "qz", "qw")], 1)
        objects.append((name, ov, of, tr))
floor = qa["floor"]["floor_y_m"]
rows = smplx_qa.render_grid(out, poses, trans, betas, fps, floor, objects,
                            title=f"{qa['clip_id']}  floor_y={floor:+.3f} m  (grey plane = per-clip floor estimate)")
qa["render"] = {"png": out, "frames": rows}
(d / "qa.json").write_text(json.dumps(qa, indent=1))
print(json.dumps(rows))
