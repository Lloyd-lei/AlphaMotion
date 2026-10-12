"""Human(SMPL-X)-object penetration audit for converted clips (sampled frames).
Object solid = convex hull of the OBJ (exact for OmniContact boxes/ball; an upper bound for
concave HiPHI furniture -> reported with hull=True). Body parts from SMPL-X LBS argmax.
usage: obj_pen_audit.py <seq_dir> [n_frames=60]   -> prints JSON, writes <seq_dir>/obj_pen.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import warnings
warnings.filterwarnings("ignore")
import trimesh

sys.path.insert(0, str(Path(__file__).parent))
import smplx_qa  # noqa: E402

GROUP = {"hand": set(range(20, 22)) | set(range(25, 55)), "forearm": {18, 19}, "upper_arm": {16, 17},
         "torso": {0, 3, 6, 9, 12, 13, 14}, "head": {15, 22, 23, 24}, "leg": {1, 2, 4, 5}, "foot": {7, 8, 10, 11}}


def load(d):
    d = Path(d)
    objs = []
    if (d / "motion_actor_smplx.npz").exists():
        z = np.load(d / "motion_actor_smplx.npz")
        P, Tr, B = z["poses"], z["trans"], z["betas"]
        meta = json.loads((d / "metadata.json").read_text())
        for o in meta.get("objects", []):
            ov, of = smplx_qa.load_obj_cm(d.parents[3] / o["mesh_path"])
            r = np.genfromtxt(d / o["trajectory_path"], delimiter=",", names=True)
            objs.append((o["object_id"], ov, of, np.stack([r[k] for k in ("px", "py", "pz", "qx", "qy", "qz", "qw")], 1)))
    else:
        P, Tr, B = np.load(d / "poses.npy"), np.load(d / "transl.npy"), np.load(d / "betas.npy")
        for c in sorted(d.glob("prop_*.csv")):
            n = c.stem[5:]
            ov, of = smplx_qa.load_obj_cm(d / f"{n}.obj")
            r = np.genfromtxt(c, delimiter=",", names=True)
            objs.append((n, ov, of, np.stack([r[k] for k in ("px", "py", "pz", "qx", "qy", "qz", "qw")], 1)))
    return P, Tr, B, objs


def main():
    d = Path(sys.argv[1]); n = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    P, Tr, B, objs = load(d)
    fr = np.linspace(0, len(P) - 1, n).round().astype(int)
    V, _ = smplx_qa.smplx_verts(P, Tr, B, frames=fr)
    W = smplx_qa.body_model(1).lbs_weights.numpy().argmax(1)
    gid = np.full(len(W), "other", object)
    for g, s in GROUP.items():
        gid[np.isin(W, list(s))] = g
    out = {"clip": str(d), "frames_sampled": int(n), "objects": []}
    for name, ov, of, tr in objs:
        m0 = trimesh.Trimesh(ov, of, process=True)
        # signed_distance needs a clean closed solid; OBJ exports here have split UV seams /
        # inconsistent winding, so use the convex hull unless the mesh is watertight AND clearly
        # concave (volume < 85% of hull).  Hull = exact for boxes/balls, upper bound otherwise.
        hull = True
        if m0.is_watertight and m0.is_winding_consistent and m0.volume > 0:
            hull = m0.volume / m0.convex_hull.volume > 0.85
        per = {g: [] for g in GROUP}
        touch = 0
        for k, t in enumerate(fr):
            w = smplx_qa.pose_obj(ov, tr[t])
            if hull:
                m = trimesh.Trimesh(w, process=True).convex_hull
            else:
                m = trimesh.Trimesh(smplx_qa.pose_obj(np.asarray(m0.vertices), tr[t]), m0.faces, process=False)
            c = w.mean(0); rad = np.linalg.norm(w - c, axis=1).max()
            near = np.linalg.norm(V[k] - c, axis=1) < rad + 0.05
            if not near.any():
                continue
            sd = trimesh.proximity.signed_distance(m, V[k][near])  # >0 inside
            if (sd > -0.02).any():
                touch += 1
            for g in GROUP:
                sel = gid[near] == g
                if sel.any():
                    per[g].append(max(0.0, float(sd[sel].max())))
        res = {"object": name, "convex_hull_used": bool(hull), "frames_within_2cm": touch}
        for g, v in per.items():
            v = np.asarray(v)
            if len(v):
                res[g] = {"pen_max_cm": round(float(v.max() * 100), 2),
                          "frames_pen_gt1cm": int((v > 0.01).sum())}
        out["objects"].append(res)
    (d / "obj_pen.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
