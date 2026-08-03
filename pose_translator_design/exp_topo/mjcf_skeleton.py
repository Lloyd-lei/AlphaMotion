"""Bridge GMR's robot output (qpos = motor angles) into the tokenizer's input
(skeleton + per-joint GLOBAL rotations).

Every MJCF robot carries a kinematic tree — bodies/links are the bones, body_pos
is the rest offset from the parent, and MuJoCo FK turns qpos into a global frame
per body. That is exactly the SkeletonSpec + global-rot6d format the tokenizer
already eats for SOMA and g1_30; only the conversion was missing.

Two structural caveats, both handled here:
  * Multi-DOF joints (3-DOF hip, 2-DOF ankle) are modelled as stacked co-located
    hinge bodies, giving ZERO-LENGTH bones (0-3 per robot). A zero rest offset has
    no direction, so JointFeature's normalize() yields a null direction vector.
    `--merge_colocated` folds such a body into its parent (their rotations compose
    through FK anyway), producing a cleaner bone tree.
  * Robot joints are 1-DOF hinges, but the decoder emits full 3-DOF rotations.
    Nothing is enforced here — measuring that off-manifold component is the point
    (see g1_qpos.py); projection belongs on the decode side.
"""
from __future__ import annotations
import os, sys, argparse
import numpy as np
import mujoco
sys.path.insert(0, os.path.dirname(__file__))
from common import SkeletonSpec, matrix_to_rot6d
import torch

CM = 100.0          # MuJoCo metres -> cache centimetres


def spec_from_mjcf(model, name, merge_colocated=True, eps=1e-3):
    """MJCF body tree -> (SkeletonSpec in cm, kept body ids).

    Root = body 1 (the floating base); world (body 0) is dropped. Its rest offset
    is set to the standing height so the root bone has a length like the other
    caches' root bone.
    """
    nb = model.nbody
    parent = model.body_parentid.copy()
    pos = model.body_pos.copy()                              # offset from parent frame (m)
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or f"body{i}" for i in range(nb)]

    keep = [i for i in range(1, nb)]
    if merge_colocated:
        # drop bodies sitting exactly on their parent's origin (stacked hinge dummies)
        drop = {i for i in range(2, nb) if np.linalg.norm(pos[i]) < eps}
        keep = [i for i in keep if i not in drop]

    kept_set = set(keep)
    remap = {b: k for k, b in enumerate(keep)}

    def nearest_kept_ancestor(b):
        p = parent[b]
        while p >= 1 and p not in kept_set:
            p = parent[p]
        return p

    parents, offsets, jnames = [], [], []
    for b in keep:
        p = nearest_kept_ancestor(b)
        parents.append(remap[p] if p in kept_set else -1)
        # offset accumulates through any dropped intermediate bodies
        acc = pos[b].copy(); cur = parent[b]
        while cur >= 1 and cur not in kept_set:
            acc = acc + pos[cur]; cur = parent[cur]
        offsets.append(acc * CM)
        jnames.append(names[b])
    offsets = np.asarray(offsets, np.float32)
    if len(offsets):                                   # root bone: give it the standing height
        offsets[0] = np.array([0.0, 0.0, float(model.body_pos[1][2] or 0.8) * CM], np.float32)
    return SkeletonSpec(name, np.asarray(parents, np.int64), offsets, jnames), keep


def qpos_to_global(model, data, qpos_seq, keep, batch=None):
    """[T, nq] qpos -> (global_rot6d [T,J,6], global_pos_cm [T,J,3], root-relative)."""
    T = qpos_seq.shape[0]
    R = np.empty((T, len(keep), 3, 3), np.float32)
    P = np.empty((T, len(keep), 3), np.float32)
    for t in range(T):
        data.qpos[:] = qpos_seq[t]
        mujoco.mj_kinematics(model, data)
        R[t] = data.xmat[keep].reshape(-1, 3, 3)
        P[t] = data.xpos[keep]
    P = (P - P[:, :1]) * CM                                   # root-relative, cm (cache convention)
    rot6d = matrix_to_rot6d(torch.as_tensor(R))
    return rot6d, torch.as_tensor(P)


def load_robot(xml_path, name=None, merge_colocated=True):
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    spec, keep = spec_from_mjcf(model, name or os.path.basename(os.path.dirname(xml_path)),
                                merge_colocated=merge_colocated)
    return model, data, spec, keep


def convert_npz(xml_path, npz_path, merge_colocated=True):
    model, data, spec, keep = load_robot(xml_path, merge_colocated=merge_colocated)
    q = np.load(npz_path)["qpos"].astype(np.float64)
    assert q.shape[1] == model.nq, f"qpos width {q.shape[1]} != model.nq {model.nq}"
    rot6d, pos = qpos_to_global(model, data, q, keep)
    return spec, rot6d, pos


def main():
    ap = argparse.ArgumentParser(description="Inspect/convert a robot MJCF + qpos into skeleton form")
    ap.add_argument("--xml", required=True)
    ap.add_argument("--npz", default=None, help="a GMR production qpos npz to convert as a check")
    ap.add_argument("--no_merge", action="store_true")
    args = ap.parse_args()
    model, data, spec, keep = load_robot(args.xml, merge_colocated=not args.no_merge)
    L = np.linalg.norm(spec.rest_offsets, axis=-1)
    print(f"{spec.name}: J={spec.J} (from {model.nbody-1} bodies)  bone cm: "
          f"min={L.min():.2f} med={np.median(L):.1f} max={L.max():.1f}  zero={(L < 1e-3).sum()}")
    spec.order_root_first()
    if args.npz:
        s, rot6d, pos = convert_npz(args.xml, args.npz, merge_colocated=not args.no_merge)
        blen = (pos[:, 1:] - pos[:, [max(int(p), 0) for p in s.parents[1:]]]).norm(dim=-1)
        print(f"  {os.path.basename(args.npz)}: T={rot6d.shape[0]} rot6d={tuple(rot6d.shape)} "
              f"pos={tuple(pos.shape)}  NaN={bool(torch.isnan(rot6d).any())}")
        print(f"  bone-length drift over time (should be ~0): {float(blen.std(0).max()):.4f} cm")
        print(f"  root-rel extent cm: {float(pos.norm(dim=-1).max()):.1f}")


if __name__ == "__main__":
    main()
