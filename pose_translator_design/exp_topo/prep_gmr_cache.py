"""Build the 14-embodiment strictly row-aligned paired cache (soma77 + g1_30 + 12 GMR
robots) — the data foundation for E5-full / E9.

Sources
  human side : cache_paired_seq (soma77 + g1_30, fps mode 30fps capped 240/clip)
  robot side : /var/tmp/alphamotion/gmr_production/{robot}/{clip}.npz  qpos [T,nq] @30fps

Clip selection: intersection of clips ACCEPTED by all 12 robots (per-robot manifest
accept flag + npz on disk), in cache_paired_seq/sequences.json order. Each surviving
clip contributes exactly the human cache's "len" rows: robot qpos is resampled by
relative time to len, so row i is the SAME instant in all 14 embodiments.

LOCAL rotations (mandatory — downstream FKs local6d with the spec):
    local_j = R_parent^T @ R_j        (root: local = global)
subtlety: mjcf_skeleton merges co-located bodies (stacked hinge dummies), but the
kept body's own xmat is the WRONG frame for its children's rest offsets — children
hang off the deepest co-located (dropped) body's frame, so the naive conversion FKs
to ~4cm position error. Fix here:
  * each merged joint's global rotation = xmat of the unique body its kept children
    attach to (the co-located chain end); leaves use their chain end (composes all
    hinge DOFs into the stored rotation).
  * if a kept body's children attach to DIFFERENT frames (g1 pelvis: hips on pelvis,
    torso on dropped waist stack) no single rotation is exact -> iteratively un-drop
    the conflicting attach bodies (adds a zero-length bone, FK becomes exact).
Verified per clip by the FK round-trip gate: fk_global(local6d, spec) positions must
match MuJoCo positions to <0.1cm.

CPU only, multiprocessing (nice 10). No trainings started here.
"""
from __future__ import annotations
import os, sys, json, time, glob, argparse, importlib.util
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SkeletonSpec, matrix_to_rot6d, rot6d_to_matrix, fk_global, CACHE_ROOT

CM = 100.0
GMR_PROD = "/var/tmp/alphamotion/gmr_production"
SRC_CACHE = f"{CACHE_ROOT}/cache_paired_seq"
OUT_DEFAULT = f"{CACHE_ROOT}/cache_paired_13"
GMR_REPO = ('/home/arenalabs/Desktop/"be water, robot"/'
            "alpha_motion2/external/GMR")
ARDY_G1_XML = ('/home/arenalabs/Desktop/"be water, robot"/'
               "experiment/ardy/ardy/assets/skeletons/g1skel34/xml/g1.xml")
FK_GATE_CM = 0.1


def rel_time_idx(T, K):
    if T <= 1:
        return np.zeros(K, dtype=np.int64)
    return np.clip(np.round(np.linspace(0, T - 1, K)).astype(np.int64), 0, T - 1)


def save_spec(spec, path):
    np.savez(path, name=spec.name, parents=spec.parents, rest_offsets=spec.rest_offsets,
             norm_bonelen=spec.norm_bonelen,
             joint_names=np.array(spec.joint_names, dtype=object), height=spec.height)


def robot_xml(robot):
    """Resolve the MJCF for a robot. unitree_g1 -> the ardy XML our existing g1
    pipeline uses (verified: same 29-hinge joint order, FK-identical to GMR's
    g1_mocap_29dof.xml on production qpos); everything else -> GMR params.py."""
    if robot == "unitree_g1":
        return ARDY_G1_XML
    p = f"{GMR_REPO}/general_motion_retargeting/params.py"
    s = importlib.util.spec_from_file_location("gmr_params", p)
    mod = importlib.util.module_from_spec(s)
    s.loader.exec_module(mod)          # params.py only needs pathlib, no mink
    return str(mod.ROBOT_XML_DICT[robot])


# ------------------------- merge plan (keep + frame map) -------------------------

def build_merge(model, eps=1e-3):
    """Decide which co-located bodies to merge and which frame carries each kept joint.

    Returns (keep, frame) — keep: kept body ids ascending (topological in MJCF);
    frame[i]: body id whose xmat is the kept joint's global rotation.
    """
    import mujoco
    nb = model.nbody
    parent = model.body_parentid
    pos = model.body_pos
    drop = {i for i in range(2, nb) if np.linalg.norm(pos[i]) < eps}

    def compute(drop):
        keep = [i for i in range(1, nb) if i not in drop]
        kset = set(keep)

        def nka(b):                                   # nearest kept ancestor
            p = int(parent[b])
            while p >= 1 and p not in kset:
                p = int(parent[p])
            return p

        attach = {b: set() for b in keep}             # frames kept children hang from
        for c in keep:
            a = nka(c)
            if a >= 1:
                attach[a].add(int(parent[c]))
        conflicts = {b: s for b, s in attach.items() if len(s) > 1}
        return keep, attach, conflicts

    keep, attach, conflicts = compute(drop)
    while conflicts:                                  # un-drop conflicting attach points
        for b, s in conflicts.items():
            for x in s:
                drop.discard(x)
        keep, attach, conflicts = compute(drop)

    dropped_children = {}
    for d in sorted(drop):
        dropped_children.setdefault(int(parent[d]), []).append(d)

    def chain_end(b):                                 # deepest single co-located chain
        cur = b
        while len(dropped_children.get(cur, [])) == 1:
            cur = dropped_children[cur][0]
        return cur

    frame = {}
    for b in keep:
        s = attach[b]
        frame[b] = next(iter(s)) if len(s) == 1 else chain_end(b)
    return keep, frame


def build_spec(model, name, keep):
    """keep -> SkeletonSpec (cm), same conventions as mjcf_skeleton.spec_from_mjcf
    (offset accumulation through dropped bodies, root bone = standing height)."""
    import mujoco
    parent = model.body_parentid
    pos = model.body_pos
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or f"body{i}"
             for i in range(model.nbody)]
    kset = set(keep)
    remap = {b: k for k, b in enumerate(keep)}
    parents, offsets, jnames = [], [], []
    for b in keep:
        p = int(parent[b])
        while p >= 1 and p not in kset:
            p = int(parent[p])
        parents.append(remap[p] if p in kset else -1)
        acc = pos[b].copy()
        cur = int(parent[b])
        while cur >= 1 and cur not in kset:
            acc = acc + pos[cur]
            cur = int(parent[cur])
        offsets.append(acc * CM)
        jnames.append(names[b])
    offsets = np.asarray(offsets, np.float32)
    offsets[0] = np.array([0.0, 0.0, float(model.body_pos[1][2] or 0.8) * CM], np.float32)
    spec = SkeletonSpec(name, np.asarray(parents, np.int64), offsets, jnames)
    spec.order_root_first()
    return spec


def load_robot(robot):
    import mujoco
    model = mujoco.MjModel.from_xml_path(robot_xml(robot))
    data = mujoco.MjData(model)
    keep, frame = build_merge(model)
    spec = build_spec(model, robot, keep)
    return model, data, spec, keep, frame


def qpos_to_global(model, data, qpos_seq, keep, frame):
    """[T,nq] -> (global R [T,J,3,3], root-relative positions cm [T,J,3])."""
    import mujoco
    T = qpos_seq.shape[0]
    fids = np.array([frame[b] for b in keep])
    kids = np.array(keep)
    R = np.empty((T, len(keep), 3, 3), np.float32)
    P = np.empty((T, len(keep), 3), np.float32)
    for t in range(T):
        data.qpos[:] = qpos_seq[t]
        mujoco.mj_kinematics(model, data)
        R[t] = data.xmat[fids].reshape(-1, 3, 3)
        P[t] = data.xpos[kids]
    P = (P - P[:, :1]) * CM
    return R, P


def to_local6d(R, parents):
    """global R [T,J,3,3] -> local rot6d [T,J,6]; local_j = R_parent^T @ R_j."""
    R = torch.as_tensor(R)
    L = R.clone()
    for j in range(1, R.shape[1]):
        p = int(parents[j])
        L[:, j] = R[:, p].transpose(-1, -2) @ R[:, j]
    return matrix_to_rot6d(L)


# ----------------------------------- workers -----------------------------------

def _worker_init():
    os.nice(10)
    torch.set_num_threads(1)


def process_chunk(task):
    """One (robot, chunk of clips) -> local6d npy + per-chunk validation stats."""
    robot, chunk_id, clips, out_npy = task
    model, data, spec, keep, frame = load_robot(robot)
    parts, fk_err, drift, nan_frames, rows = [], 0.0, 0.0, 0, 0
    for name, K in clips:
        q = np.load(f"{GMR_PROD}/{robot}/{name}.npz")["qpos"].astype(np.float64)
        assert q.shape[1] == model.nq, \
            f"{robot}/{name}: qpos width {q.shape[1]} != model.nq {model.nq}"
        q = q[rel_time_idx(q.shape[0], K)]            # frame-align to the human cache
        R, P = qpos_to_global(model, data, q, keep, frame)
        l6 = to_local6d(R, spec.parents)              # [K,J,6]
        nan_frames += int(torch.isnan(l6).any(-1).any(-1).sum())
        P_t = torch.as_tensor(P)
        _, gp = fk_global(l6, spec)                   # FK round-trip gate
        fk_err = max(fk_err, float((gp - P_t).norm(dim=-1).max()))
        par = spec.parents
        blen = (P_t[:, 1:] - P_t[:, [max(int(p), 0) for p in par[1:]]]).norm(dim=-1)
        drift = max(drift, float(blen.std(0).max()))  # rigid bones -> ~0
        parts.append(l6.numpy().astype(np.float32))
        rows += K
    np.save(out_npy, np.concatenate(parts, 0))
    return {"robot": robot, "chunk": chunk_id, "rows": rows, "fk_err_cm": fk_err,
            "bone_drift_cm": drift, "nan_frames": nan_frames, "npy": out_npy}


# ------------------------------------ main -------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--chunk_clips", type=int, default=64)
    ap.add_argument("--limit_clips", type=int, default=0, help="debug: only first N clips")
    args = ap.parse_args()
    t0 = time.time()
    os.makedirs(args.out, exist_ok=True)
    tmp = f"{args.out}/_chunks"
    os.makedirs(tmp, exist_ok=True)

    robots = sorted(d for d in os.listdir(GMR_PROD)
                    if os.path.isdir(f"{GMR_PROD}/{d}") and d != "logs")
    assert len(robots) == 12, robots

    # ---- 1. clip selection: accepted by ALL robots, in human-cache order ----
    seq = json.load(open(f"{SRC_CACHE}/sequences.json"))
    accept = {}
    for r in robots:
        man = json.load(open(f"{GMR_PROD}/{r}/manifest.json"))
        accept[r] = {c["name"] for c in man["clips"] if c["accept"]}
    surviving = [s for s in seq
                 if all(s["name"] in accept[r] and
                        os.path.exists(f"{GMR_PROD}/{r}/{s['name']}.npz")
                        for r in robots)]
    if args.limit_clips:
        surviving = surviving[:args.limit_clips]
    n_rows = sum(s["len"] for s in surviving)
    print(f"[select] {len(surviving)}/{len(seq)} clips accepted by all {len(robots)} robots"
          f" -> {n_rows} rows", flush=True)

    # ---- 2+3. robots: qpos -> LOCAL rot6d, chunked multiprocessing ----
    clip_kl = [(s["name"], int(s["len"])) for s in surviving]
    tasks = []
    for r in robots:
        for ci in range(0, len(clip_kl), args.chunk_clips):
            cid = ci // args.chunk_clips
            tasks.append((r, cid, clip_kl[ci:ci + args.chunk_clips],
                          f"{tmp}/{r}_{cid:04d}.npy"))
    print(f"[fk] {len(tasks)} tasks ({args.workers} workers)", flush=True)

    import multiprocessing as mp
    stats = {r: {"fk_err_cm": 0.0, "bone_drift_cm": 0.0, "nan_frames": 0, "rows": 0}
             for r in robots}
    chunks = {r: {} for r in robots}
    with mp.Pool(args.workers, initializer=_worker_init) as pool:
        done = 0
        for res in pool.imap_unordered(process_chunk, tasks):
            r = res["robot"]
            chunks[r][res["chunk"]] = res["npy"]
            stats[r]["fk_err_cm"] = max(stats[r]["fk_err_cm"], res["fk_err_cm"])
            stats[r]["bone_drift_cm"] = max(stats[r]["bone_drift_cm"], res["bone_drift_cm"])
            stats[r]["nan_frames"] += res["nan_frames"]
            stats[r]["rows"] += res["rows"]
            done += 1
            if done % 25 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)} ({time.time()-t0:.0f}s)", flush=True)

    for r in robots:
        model, data, spec, keep, frame = load_robot(r)
        arr = np.concatenate([np.load(chunks[r][c]) for c in sorted(chunks[r])], 0)
        assert arr.shape == (n_rows, spec.J, 6), (r, arr.shape, n_rows, spec.J)
        torch.save(torch.from_numpy(arr), f"{args.out}/{r}_paired_local6d.pt")
        save_spec(spec, f"{args.out}/{r}_spec.npz")
        stats[r]["J"] = spec.J
        print(f"[{r}] J={spec.J} rows={arr.shape[0]} fk_err={stats[r]['fk_err_cm']:.4f}cm "
              f"drift={stats[r]['bone_drift_cm']:.4f}cm nan={stats[r]['nan_frames']}", flush=True)

    # ---- 4. subset the human/g1 sides to the surviving clips' row ranges ----
    new_seq, start = [], 0
    for s in surviving:
        new_seq.append({"name": s["name"], "start": start, "len": int(s["len"])})
        start += int(s["len"])
    human_stats = {}
    for name in ["soma77", "g1_30"]:
        full = torch.load(f"{SRC_CACHE}/{name}_paired_local6d.pt")
        sub = torch.cat([full[s["start"]:s["start"] + s["len"]] for s in surviving], 0)
        assert sub.shape[0] == n_rows, (name, sub.shape, n_rows)
        torch.save(sub.contiguous(), f"{args.out}/{name}_paired_local6d.pt")
        d = np.load(f"{SRC_CACHE}/{name}_spec.npz", allow_pickle=True)
        save_spec(SkeletonSpec(str(d["name"]), d["parents"], d["rest_offsets"],
                               list(d["joint_names"])), f"{args.out}/{name}_spec.npz")
        human_stats[name] = {"J": int(sub.shape[1]), "rows": int(sub.shape[0]),
                             "nan_frames": int(torch.isnan(sub).any(-1).any(-1).sum())}
        print(f"[{name}] J={sub.shape[1]} rows={sub.shape[0]} "
              f"nan={human_stats[name]['nan_frames']}", flush=True)
        del full, sub

    emb = ["soma77", "g1_30"] + robots
    json.dump({"schema": 2, "embodiments": emb, "rows": int(n_rows),
               "clips": len(surviving), "mode": "fps", "fps": 30.0,
               "g1_name": "g1_30", "src_human": SRC_CACHE, "src_robot": GMR_PROD},
              open(f"{args.out}/manifest.json", "w"), indent=2)
    json.dump(new_seq, open(f"{args.out}/sequences.json", "w"))

    # ---- 5. validation gate ----
    for f in glob.glob(f"{tmp}/*.npy"):
        os.remove(f)
    os.rmdir(tmp)
    disk = sum(os.path.getsize(f"{args.out}/{f}") for f in os.listdir(args.out))
    rng = np.random.default_rng(0)
    probe = new_seq[int(rng.integers(len(new_seq)))]
    ok = all(s["fk_err_cm"] < FK_GATE_CM and s["nan_frames"] == 0
             and s["rows"] == n_rows for s in stats.values()) \
        and all(h["nan_frames"] == 0 and h["rows"] == n_rows for h in human_stats.values())
    val = {"pass": bool(ok), "fk_gate_cm": FK_GATE_CM, "clips": len(surviving),
           "rows": int(n_rows), "embodiments": emb, "robots": stats,
           "human": human_stats, "probe_clip": probe,
           "disk_bytes": int(disk), "wall_s": round(time.time() - t0, 1)}
    json.dump(val, open(f"{args.out}/VALIDATION.json", "w"), indent=2)
    print(f"[validate] PASS={ok}  disk={disk/1e9:.2f}GB  wall={time.time()-t0:.0f}s")
    print("SAVED", args.out)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
