"""GPU-batched BVH -> SMPL-X converter v2 (HiPHI / OmniContact), output layout identical to
umr_factory_data/scripts/convert_hiphi.py / convert_omni.py (+ optional 30 fps variant for HiPHI).

usage (genmo-gem python, read-only use of that env):
  convert.py --kind hiphi [--parts 0002,0003 | --clips FILE] [--out DIR] [--fps30 DIR] [--include-cpu-done] [--force]
             [--beta gpu|cpu-npz] [--lanes 1024] [--watch --idle-exit 1800] [--max-wall 21600]
  convert.py --kind omni [--out /home/arenalabs/umr_factory_data/smplx_gpu/omnicontact] ...

Skip logic (coordination with the CPU converter): a clip is skipped if the CPU path already wrote
smplx/<kind>/.../qa.json, or if this converter already wrote <out>/.../qa.json (unless --force). Clips of a part are
fed in REVERSE order of the CPU path's todo list, so a CPU run still working on the same part overlaps least.

Pipeline (all stages overlap):
  prep pool (CPU, BVH -> targets/copied rotations/frame-0 beta targets)
  -> beta thread (GPU stream 2: batched float64 Newton beta fit; CPU: shaped rest joints + foot targets)
  -> IK loop (GPU main stream: continuous batching, one lane = one clip at a time, a lane that finishes a clip
     starts the next one at the next frame slot; the per-clip frame chain is exactly the CPU path's sequential
     chain -> no approximation from batching; K frames per CUDA-graph replay)
  -> post thread (GPU stream 3: SMPL-X lowest-vertex floor QA, gpu_lowest.LeanLowest) -> io pool (npz / metadata / object tracks / qa.json).
"""
from __future__ import annotations

import os

os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/home/arenalabs/gpu_convert/.inductor_cache")
import argparse
import collections
import json
import multiprocessing as mp
import queue
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
DATA = Path("/home/arenalabs/umr_factory_data")
FPS_SRC = 90.0


# ------------------------------------------------------------------------------------------------ discovery
def hiphi_part_dirs(parts=None):
    root = DATA / "raw/hiphi/extracted"
    ps = sorted(root.glob("part_*")) if not parts else [root / f"part_{p}" for p in parts]
    return [p for p in ps if (p / ".extracted").exists()]


def discover_hiphi(parts=None):
    out = []
    for part in hiphi_part_dirs(parts):
        dirs = sorted(m.parent for m in part.glob("HiPHI/data/*/*/*/metadata.json"))
        todo = DATA / "logs" / f"hiphi_{part.name.split('_')[1]}_todo.txt"
        if todo.exists():   # the CPU path walks its todo list in order -> start from the other end
            order = {l.strip(): i for i, l in enumerate(todo.read_text().split())}
            dirs.sort(key=lambda d: -order.get(str(d), 1e9))
        out += dirs
    return out


def discover_omni():
    raw = DATA / "raw/omnicontact/raw_mocap"
    out = []
    for m in sorted(raw.glob("*/*/*/capture_meta.json")):
        d = m.parent
        meta = json.loads(m.read_text())
        need = [meta["human_motion_file"], meta["human_timestamps_file"]] + [o["source_file"] for o in meta.get("objects", [])]
        if all((d / f).exists() for f in need):
            out.append(d)
    return out


def load_meta(kind, d):
    return json.loads((Path(d) / ("metadata.json" if kind == "hiphi" else "capture_meta.json")).read_text())


def dst_dir(kind, out_root, meta):
    if kind == "hiphi":
        return Path(out_root) / "data" / meta["frame"] / meta["lu"] / meta["motion_id"]
    return Path(out_root) / meta["domain"] / meta["normalized_case"] / meta["capture_id"]


def cpu_dir(kind, meta):
    return dst_dir(kind, DATA / "smplx" / ("hiphi" if kind == "hiphi" else "omnicontact"), meta)


def bvh_of(kind, d, meta):
    return str(Path(d) / ("motion_actor.bvh" if kind == "hiphi" else meta["human_motion_file"]))


def rec_base(kind, src, meta):
    if kind == "hiphi":
        return {"clip_id": f"hiphi/{meta['frame']}/{meta['lu']}/{meta['motion_id']}", "source": "HiPHI", "raw_dir": str(src),
                "actor_id": meta.get("actor_id"), "actor_metadata": meta.get("actor_metadata"),
                "is_hoi": meta.get("is_hoi"), "mirrored": meta.get("mirrored")}
    return {"clip_id": f"omnicontact/{meta['domain']}/{meta['normalized_case']}/{meta['capture_id']}",
            "source": "OmniContact", "raw_dir": str(src)}


# ------------------------------------------------------------------------------------------------ workers
def _winit():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""   # workers never touch the GPU
    import prep  # noqa: F401


def _prep_task(a):
    d, bvh, max_frames, beta_cpu = a
    import prep
    try:
        r = prep.prep_clip(bvh, max_frames=max_frames)
        if beta_cpu:   # upstream float32 LBFGS fit_betas, exactly as hiphi2smplx.pipeline.fit_bvh calls it
            from hiphi2smplx.beta_fitting import BetaFitConfig, fit_betas
            t0 = time.time()
            r["betas_cpu"] = fit_betas(r["targets"][0], prep.neutral().rest_joints, prep.MODEL,
                                       config=BetaFitConfig(device="cpu", data_dir=prep.bvh2smplx.BETA_DATA)).betas
            r["beta_cpu_s"] = time.time() - t0
        return d, r
    except Exception as e:  # recorded in qa.json like the CPU path (missing files are only logged: retry later)
        return d, {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-2000:],
                   "missing": isinstance(e, FileNotFoundError)}


def _write_task(job):
    import prep
    r = job.get("res")
    if r is not None and "pose22" in r:   # expand in the worker (keeps pickling + GIL work out of the GPU process)
        F = len(r["pose22"])
        p55 = np.zeros((F, 55, 3), np.float32)
        p55[:, :22] = r.pop("pose22")
        r["poses55"] = p55
    return prep.write_hiphi(job) if job["kind"] == "hiphi" else prep.write_omni(job)


def _shape_task(a):
    import prep
    key, betas, targets, reference = a
    t0 = time.time()
    ft, rest = prep.shape_clip(betas, targets, reference)
    return key, ft, rest, time.time() - t0


# ------------------------------------------------------------------------------------------------ GPU monitor
class GpuMon(threading.Thread):
    """device utilisation + this process' SM share (nvidia-smi pmon), ~1 s cadence."""

    def __init__(self, pid):
        super().__init__(daemon=True)
        self.pid, self.stop, self.dev, self.mine, self.mem, self.t = str(pid), False, [], [], [], []

    def run(self):
        while not self.stop:
            try:
                o = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True, timeout=5).stdout.strip().split(",")
                p = subprocess.run(["nvidia-smi", "pmon", "-c", "1", "-s", "u"], capture_output=True, text=True, timeout=5).stdout
                sm = 0.0
                for ln in p.splitlines():
                    f = ln.split()
                    if len(f) > 3 and f[1] == self.pid and f[3] not in ("-",):
                        sm = float(f[3])
                self.dev.append(float(o[0])); self.mem.append(float(o[1])); self.mine.append(sm); self.t.append(time.time())
            except Exception:
                pass
            time.sleep(1.0)

    def summary(self, t0=None, t1=None):
        sel = [i for i, t in enumerate(self.t) if (t0 is None or t >= t0) and (t1 is None or t <= t1)]
        dv = [self.dev[i] for i in sel]; mi = [self.mine[i] for i in sel]
        a = lambda x: (round(float(np.mean(x)), 1) if x else None)
        return dict(device_util_mean=a(dv), device_util_p50=(float(np.median(dv)) if dv else None),
                    this_process_sm_mean=a(mi), this_process_sm_p50=(float(np.median(mi)) if mi else None),
                    mem_used_mib_max=(max(self.mem) if self.mem else None), samples=len(sel))


# ------------------------------------------------------------------------------------------------ arena
class Arena:
    """first-fit allocator of contiguous frame ranges in [0, cap)."""

    def __init__(self, cap):
        self.cap = int(cap)
        self.free = [(0, self.cap)]

    def alloc(self, n):
        for i, (s, e) in enumerate(self.free):
            if e - s >= n:
                if s + n == e:
                    self.free.pop(i)
                else:
                    self.free[i] = (s + n, e)
                return s
        return None

    def release(self, s, n):
        self.free.append((s, s + n))
        self.free.sort()
        merged = []
        for a, b in self.free:
            if merged and merged[-1][1] == a:
                merged[-1] = (merged[-1][0], b)
            else:
                merged.append((a, b))
        self.free = merged

    def used(self):
        return self.cap - sum(e - s for s, e in self.free)


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["hiphi", "omni"], default="hiphi")
    ap.add_argument("--parts", default=None)
    ap.add_argument("--clips", default=None, help="file with raw clip dirs (one per line)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fps30", default=None, help="also write a 30 fps (stride 3) variant here (hiphi only)")
    ap.add_argument("--include-cpu-done", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--beta", choices=["gpu", "cpu", "cpu-npz"], default="gpu",
                    help="gpu: float64 Newton on the GPU (fast); cpu: upstream float32 LBFGS in the prep workers (bitwise "
                         "CPU-path betas, ~6.6 CPU-s/clip); cpu-npz: copy betas of an existing CPU output (parity tests)")
    ap.add_argument("--lanes", type=int, default=1024)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--arena-frames", type=float, default=7e6)
    ap.add_argument("--beta-batch", type=int, default=192)
    ap.add_argument("--prep-workers", type=int, default=14)
    ap.add_argument("--io-workers", type=int, default=6)
    ap.add_argument("--qa-chunk", type=int, default=2048)
    ap.add_argument("--watch", action="store_true", help="keep discovering newly extracted HiPHI parts")
    ap.add_argument("--idle-exit", type=float, default=1800.0)
    ap.add_argument("--max-wall", type=float, default=6 * 3600.0)
    ap.add_argument("--log", default=None)
    ap.add_argument("--summary", default=None)
    args = ap.parse_args()
    kind = args.kind
    out_root = Path(args.out or (DATA / "smplx_gpu" / ("hiphi" if kind == "hiphi" else "omnicontact")))
    out_root.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    deadline = t_start + args.max_wall
    parts = args.parts.split(",") if args.parts else None

    if kind == "hiphi":
        for root in [out_root] + ([Path(args.fps30)] if args.fps30 else []):
            om = root / "object_meshes"
            root.mkdir(parents=True, exist_ok=True)
            if not om.exists():
                r = subprocess.run(["cp", "-al", str(DATA / "raw/hiphi/object_meshes"), str(om)], capture_output=True)
                if r.returncode != 0:   # other filesystem: symlink the shared mesh dir instead
                    subprocess.run(["rm", "-rf", str(om)]); om.symlink_to(DATA / "raw/hiphi/object_meshes")

    seen = set()
    counts = collections.Counter()

    def select(dirs):
        todo = []
        for d in dirs:
            d = Path(d)
            if str(d) in seen:
                continue
            try:
                meta = load_meta(kind, d)
            except Exception:
                counts["meta_unreadable"] += 1; continue
            seen.add(str(d))
            if not args.include_cpu_done and (cpu_dir(kind, meta) / "qa.json").exists():
                counts["skip_cpu_done"] += 1; continue
            if not args.force and (dst_dir(kind, out_root, meta) / "qa.json").exists():
                counts["skip_gpu_done"] += 1; continue
            todo.append((d, meta))
        return todo

    def discover():
        if args.clips:
            return [Path(x.strip()) for x in open(args.clips) if x.strip()]
        return discover_hiphi(parts) if kind == "hiphi" else discover_omni()

    first = select(discover())
    if args.limit:
        first = first[:args.limit]
    # longest clips first: the clips that enter the lanes last (and set the end-of-run tail) are the short ones
    first.sort(key=lambda x: -int(x[1].get("frame_count") or 0))
    print(f"[gpu_convert] {kind}: {len(seen)} found, {counts['skip_cpu_done']} skipped (CPU done), "
          f"{counts['skip_gpu_done']} skipped (GPU done), {len(first)} todo", flush=True)
    if not first and not args.watch:
        return

    import torch
    import gpu_beta
    import gpu_ik
    import gpu_lowest
    import prep
    import seg_ik

    dev = "cuda"
    B, K = args.lanes, args.K
    ctx = mp.get_context("spawn")
    prep_pool = ctx.Pool(args.prep_workers, initializer=_winit, maxtasksperchild=100)
    io_pool = ctx.Pool(args.io_workers, initializer=_winit, maxtasksperchild=200)
    shape_pool = ctx.Pool(2, initializer=_winit, maxtasksperchild=500)   # never queued behind prep or write tasks
    mon = GpuMon(os.getpid()); mon.start()
    logf = open(args.log or (out_root / "convert_log.jsonl"), "a")
    log_lock = threading.Lock()
    stats = collections.Counter()
    tstat = collections.defaultdict(float)
    errors = []

    def log(rec):
        with log_lock:
            logf.write(json.dumps(rec) + "\n"); logf.flush()

    def write_done(r):
        log(r)
        stats["clips_ok" if r.get("ok") else "clips_err"] += 1

    def write_err(e):
        errors.append(repr(e)); stats["write_exceptions"] += 1

    def dispatch_write(job):
        stats["writes_dispatched"] += 1
        io_pool.apply_async(_write_task, (job,), callback=write_done, error_callback=write_err)

    def make_job(src, meta, res, lows, gpu, seconds, error=None):
        return dict(kind=kind, src=str(src), out_root=str(out_root), meta=meta, rec_base=rec_base(kind, src, meta), res=res,
                    lows=lows, gpu=gpu, seconds=seconds, fps30_root=(args.fps30 if kind == "hiphi" else None), error=error)

    # ---------------------------------------------------------------- stage 1: prep feeder
    prep_q: queue.Queue = queue.Queue()
    # clips between submission and arena upload (host RAM bound: ~4.5 MB each)
    inflight = threading.Semaphore(args.prep_workers * 4 + args.beta_batch + 256)
    feeder_done = threading.Event()
    meta_of = {}

    def submit(items):
        for d, meta in items:
            if time.time() > deadline:
                break
            inflight.acquire()
            meta_of[str(d)] = meta
            stats["submitted"] += 1
            prep_pool.apply_async(_prep_task, ((str(d), bvh_of(kind, d, meta), args.max_frames, args.beta == "cpu"),),
                                  callback=prep_q.put,
                                  error_callback=lambda e, d=str(d): prep_q.put((d, {"error": repr(e), "missing": False})))

    def feeder():
        try:
            submit(first)
            if args.watch:
                last_new = time.time()
                while time.time() < deadline:
                    time.sleep(60)
                    new = select(discover())
                    if new:
                        new.sort(key=lambda x: -int(x[1].get("frame_count") or 0))
                        print(f"[gpu_convert] watch: {len(new)} new clips", flush=True)
                        submit(new); last_new = time.time()
                    elif time.time() - last_new > args.idle_exit:
                        break
        except Exception:
            errors.append(traceback.format_exc()); print(traceback.format_exc(), flush=True)
        finally:
            feeder_done.set()

    # ---------------------------------------------------------------- stage 2: betas + shape
    ready_q: queue.Queue = queue.Queue()
    beta_done = threading.Event()
    bf = gpu_beta.BatchedBetaFit(dev) if args.beta == "gpu" else None
    beta_stream = torch.cuda.Stream()

    shaping = {}

    def shaped(r):
        key, ft, rest, dt = r
        it = shaping.pop(key)
        it.update(ft=ft, rest=rest, t_ready=time.time())
        tstat["shape_s"] += dt
        ready_q.put(it)

    def shape_failed(key, e):
        it = shaping.pop(key)
        inflight.release()
        dispatch_write(make_job(it["d"], it["meta"], None, None, {}, 0.0, error=f"shape: {e!r}"))

    def beta_worker():
        try:
            while True:
                batch = []
                try:
                    batch.append(prep_q.get(timeout=1.0))
                except queue.Empty:
                    if feeder_done.is_set() and stats["submitted"] == stats["prepped"]:
                        break
                    continue
                t_w = time.time()
                # gather up to beta_batch; flush early when the IK side is running low
                # batch = everything that arrives within ~2 s (more while the IK side has a backlog); the
                # fit time is nearly flat in batch size, so batches grow to match the prep rate
                horizon = 2.0 if ready_q.qsize() < B // 4 else 8.0
                while len(batch) < args.beta_batch and time.time() - t_w < horizon:
                    try:
                        batch.append(prep_q.get(timeout=0.1))
                    except queue.Empty:
                        pass
                stats["prepped"] += len(batch)
                ok = []
                for d, pr in batch:
                    meta = meta_of[d]
                    if "error" in pr:
                        inflight.release()
                        if pr.get("missing"):
                            stats["missing_raw"] += 1; log({"clip_dir": d, "skipped_missing": pr["error"]}); continue
                        dispatch_write(make_job(d, meta, None, None, {}, 0.0, error=pr["error"]))
                        continue
                    tstat["prep_s_sum"] += pr["prep_s"]
                    ok.append((d, meta, pr))
                if not ok:
                    continue
                t0 = time.time()
                if bf is not None:
                    with torch.cuda.stream(beta_stream):
                        br = bf.fit(np.stack([pr["beta_target"] for _, _, pr in ok]))
                    binfo = [dict(beta_fit="gpu_f64_multistart_newton", beta_loss_w5=float(br["loss_w5"][k]),
                                  beta_loss_w0=float(br["loss_w0"][k]), beta_mean_joint_err_cm=round(float(br["mean_joint_err"][k] * 100), 3),
                                  beta_grad_max=float(br["grad_max"][k]), beta_start=int(br["start_used"][k])) for k in range(len(ok))]
                    betas = list(br["betas"])
                elif args.beta == "cpu":
                    betas = [pr["betas_cpu"] for _, _, pr in ok]
                    binfo = [dict(beta_fit="cpu_upstream_lbfgs_f32 (hiphi2smplx fit_betas in prep worker)",
                                  beta_cpu_s=round(pr["beta_cpu_s"], 2)) for _, _, pr in ok]
                else:
                    betas, binfo = [], []
                    for d, meta, pr in ok:
                        f = cpu_dir(kind, meta) / ("motion_actor_smplx.npz" if kind == "hiphi" else "betas.npy")
                        b = np.load(f)
                        betas.append((b["betas"] if hasattr(b, "files") else b).astype(np.float32))
                        binfo.append(dict(beta_fit="cpu-npz (betas copied from the CPU converter output)"))
                tstat["beta_s"] += time.time() - t0
                stats["beta_batches"] += 1
                for k, (d, meta, pr) in enumerate(ok):
                    bt = np.asarray(betas[k], np.float32)
                    if not np.isfinite(bt).all():
                        inflight.release()
                        dispatch_write(make_job(d, meta, None, None, {}, 0.0, error="non-finite betas from GPU beta fit")); continue
                    item = dict(d=d, meta=meta, F=int(pr["frames"]), ref=pr["reference"], betas=bt, binfo=binfo[k],
                                frame_time=pr["frame_time"], layout=pr["layout"], prep_s=pr["prep_s"])
                    shaping[d] = item
                    shape_pool.apply_async(_shape_task, ((d, bt, pr["targets"], pr["reference"]),), callback=shaped,
                                          error_callback=lambda e, d=d: shape_failed(d, e))
        except Exception:
            errors.append(traceback.format_exc()); print(traceback.format_exc(), flush=True)
        finally:
            beta_done.set()

    # ---------------------------------------------------------------- stage 4: QA + writes
    post_q: queue.Queue = queue.Queue()
    sm = gpu_lowest.LeanLowest(dev, chunk=args.qa_chunk)
    post_stream = torch.cuda.Stream()

    def post_worker():
        try:
            stop = False
            while not stop:
                it = post_q.get()
                if it is None:
                    break
                items = [it]
                nfr = it["F"]
                while nfr < 60000:
                    try:
                        x = post_q.get(timeout=0.5)
                    except queue.Empty:
                        break
                    if x is None:
                        stop = True; break
                    items.append(x); nfr += x["F"]
                t0 = time.time()
                with torch.cuda.stream(post_stream):
                    P = np.concatenate([x["pose"] for x in items]); T = np.concatenate([x["trans"] for x in items])
                    Bt = np.concatenate([np.repeat(x["betas"][None], x["F"], 0) for x in items])
                    lows = sm.lowest_pf(P, T, Bt)
                tstat["qa_s"] += time.time() - t0
                o = 0
                for x in items:
                    F = x["F"]
                    res = dict(pose22=x["pose"], trans=x["trans"], betas=x["betas"], joints=x["joints"], losses=x["loss"],
                               fit_targets=x["ft"], frame_time=x["frame_time"], layout=x["layout"])
                    gpu = dict(x["binfo"], ik_iters_mean=round(float(x["nit"].mean()), 3), lanes=B, prep_s=round(x["prep_s"], 2),
                               pipeline="gpu_convert v2 (continuous batching, exact per-clip chain)")
                    dispatch_write(make_job(x["d"], x["meta"], res, lows[o:o + F], gpu, round(time.time() - x["t_ready"], 2)))
                    o += F
        except Exception:
            errors.append(traceback.format_exc()); print(traceback.format_exc(), flush=True)

    # ---------------------------------------------------------------- stage 3: IK (main thread)
    neutral_rest = np.asarray(prep.neutral().rest_joints)
    ik = gpu_ik.FastMinkIK(np.repeat(neutral_rest[None], B, 0), device=dev)
    tc = time.time()
    runner = seg_ik.SegRunner(ik, K=K)
    print(f"[gpu_convert] IK compiled+captured for {B} lanes x K={K} in {time.time() - tc:.1f}s", flush=True)
    cap = int(args.arena_frames)
    arena = Arena(cap)
    f32 = dict(dtype=torch.float32, device=dev)
    A_tgt = torch.zeros(cap + 1, 22, 3, **f32); A_ref = torch.zeros(cap + 1, 22, 3, **f32)
    A_pose = torch.zeros(cap + 1, 22, 3, **f32); A_trans = torch.zeros(cap + 1, 3, **f32)
    A_joints = torch.zeros(cap + 1, 22, 3, **f32); A_loss = torch.zeros(cap + 1, **f32)
    A_nit = torch.zeros(cap + 1, dtype=torch.int32, device=dev)
    A_tgt[cap] = torch.as_tensor(neutral_rest, **f32)   # idle slots: T-pose targets (finite; masked anyway)
    NSLOT = 4 * B + 1
    rest_tab = torch.as_tensor(np.repeat(gpu_ik.mjcf_rest_offsets(neutral_rest[None])[0], NSLOT, 0), dtype=torch.float64, device=dev)
    off_tab = torch.as_tensor(np.repeat(gpu_ik.mjcf_rest_offsets(neutral_rest[None])[1], NSLOT, 0), dtype=torch.float64, device=dev)
    IDLE_SLOT = NSLOT - 1
    free_slots = list(range(NSLOT - 1))
    lo_pri, hi_pri = torch.cuda.Stream.priority_range()
    up_stream = torch.cuda.Stream()
    cp_stream = torch.cuda.Stream()
    main_stream = torch.cuda.Stream(priority=hi_pri)   # IK replays: highest priority (QA / beta GEMMs yield)
    main_stream.wait_stream(torch.cuda.current_stream())
    torch.cuda.set_stream(main_stream)

    threading.Thread(target=feeder, daemon=True).start()
    bt_thread = threading.Thread(target=beta_worker, daemon=True); bt_thread.start()
    pt_thread = threading.Thread(target=post_worker, daemon=True); pt_thread.start()

    host_idx = [torch.zeros(4, B, K, dtype=torch.int64).pin_memory() for _ in range(4)]
    lane_clip = np.full(B, -1, np.int64)
    lane_t = np.zeros(B, np.int64)
    clips = {}                     # cid -> dict(item, start, slot)
    pending = collections.deque()  # loaded into the arena, not yet assigned to a lane
    holdover = [None]
    next_cid = [0]
    groups = 0
    frames_done = 0
    busy_slots = 0
    events = collections.deque()   # (event, [completed cids]) per launched group
    t_ik0 = None
    t_first_ready = [None]
    timeline = []

    def load_ready(block):
        loaded = 0
        while len(pending) < B:
            if holdover[0] is None:
                try:
                    holdover[0] = ready_q.get(timeout=1.0) if (block and loaded == 0) else ready_q.get_nowait()
                except queue.Empty:
                    break
            it = holdover[0]
            if not free_slots:
                break
            s = arena.alloc(it["F"])
            if s is None:
                break
            holdover[0] = None
            slot = free_slots.pop()
            up_stream.wait_stream(cp_stream)    # arena range may have just been read out on the copy stream
            with torch.cuda.stream(up_stream):
                A_tgt[s:s + it["F"]].copy_(torch.from_numpy(it["ft"]))
                A_ref[s:s + it["F"]].copy_(torch.from_numpy(it["ref"]))
                rj, off = gpu_ik.mjcf_rest_offsets(it["rest"][None])
                rest_tab[slot].copy_(torch.from_numpy(rj[0])); off_tab[slot].copy_(torch.from_numpy(off[0]))
            inflight.release()
            it.pop("ref", None); it.pop("ft", None)   # inputs live in the arena now
            cid = next_cid[0]; next_cid[0] += 1
            clips[cid] = dict(item=it, start=s, slot=slot)
            pending.append(cid)
            loaded += 1
            if t_first_ready[0] is None:
                t_first_ready[0] = time.time()
        if loaded:
            main_stream.wait_stream(up_stream)
        return loaded

    def finish(cids):
        """copy finished clips' outputs to host (copy stream), free their arena ranges, hand to the post thread."""
        for cid in cids:
            c = clips.pop(cid)
            it, s, F = c["item"], c["start"], c["item"]["F"]
            with torch.cuda.stream(cp_stream):
                out = dict(pose=A_pose[s:s + F].cpu().numpy(), trans=A_trans[s:s + F].cpu().numpy(),
                           joints=A_joints[s:s + F].cpu().numpy(), loss=A_loss[s:s + F].cpu().numpy(),
                           nit=A_nit[s:s + F].cpu().numpy(), ft=A_tgt[s:s + F].cpu().numpy())
            arena.release(s, F); free_slots.append(c["slot"])
            post_q.put(dict(it, **out))

    warned = False
    while True:
        if time.time() > deadline and not warned:
            print("[gpu_convert] max wall reached; no new submissions, draining", flush=True); warned = True
        load_ready(block=False)
        if (lane_clip < 0).all() and not pending:
            while events:   # nothing to run: drain completions, then wait for input or exit
                e0, dc = events.popleft(); e0.synchronize(); finish(dc)
            if beta_done.is_set() and ready_q.empty() and holdover[0] is None:
                break
            load_ready(block=True)
            continue
        if t_ik0 is None:
            t_ik0 = time.time()
        # ---- schedule K frame slots (host bookkeeping only)
        G = np.full((B, K), cap, np.int64); S = np.full((B, K), IDLE_SLOT, np.int64)
        Fst = np.zeros((B, K), bool); Act = np.zeros((B, K), bool)
        done_c = []
        for k in range(K):
            if pending:
                for b in np.flatnonzero(lane_clip < 0):
                    if not pending:
                        break
                    lane_clip[b] = pending.popleft(); lane_t[b] = 0
            ab = np.flatnonzero(lane_clip >= 0)
            if not len(ab):
                break
            cl = [clips[c] for c in lane_clip[ab]]
            st = np.fromiter((c["start"] for c in cl), np.int64, len(cl))
            sl = np.fromiter((c["slot"] for c in cl), np.int64, len(cl))
            Fc = np.fromiter((c["item"]["F"] for c in cl), np.int64, len(cl))
            G[ab, k] = st + lane_t[ab]; S[ab, k] = sl; Fst[ab, k] = lane_t[ab] == 0; Act[ab, k] = True
            lane_t[ab] += 1
            for b in ab[lane_t[ab] >= Fc]:
                done_c.append(int(lane_clip[b])); lane_clip[b] = -1
            busy_slots += len(ab)
        hb = host_idx[groups % len(host_idx)]           # pinned staging, reused only after its group has finished
        hb[0].copy_(torch.from_numpy(G)); hb[1].copy_(torch.from_numpy(S))
        hb[2].copy_(torch.from_numpy(Fst.astype(np.int64))); hb[3].copy_(torch.from_numpy(Act.astype(np.int64)))
        Gt = hb.to(dev, non_blocking=True)
        g, sl, fs, ac = Gt[0], Gt[1], Gt[2].bool(), Gt[3].bool()
        runner.tgt.copy_(A_tgt[g]); runner.ref.copy_(A_ref[g])
        runner.rest.copy_(rest_tab[sl]); runner.off.copy_(off_tab[sl])
        runner.first.copy_(fs); runner.active.copy_(ac)
        runner.run()
        d = g.reshape(-1)
        A_pose.index_copy_(0, d, runner.o_pose.reshape(-1, 22, 3)); A_trans.index_copy_(0, d, runner.o_trans.reshape(-1, 3))
        A_joints.index_copy_(0, d, runner.o_joints.reshape(-1, 22, 3)); A_loss.index_copy_(0, d, runner.o_loss.reshape(-1))
        A_nit.index_copy_(0, d, runner.o_nit.reshape(-1))
        ev = torch.cuda.Event(); ev.record(main_stream)
        events.append((ev, done_c))
        groups += 1
        frames_done += int(Act.sum())
        while len(events) > 2:   # bounded run-ahead: finish clips completed >= 2 groups ago
            e0, dc = events.popleft(); e0.synchronize(); finish(dc)
        if groups % 100 == 0:
            el = time.time() - t_ik0
            timeline.append(dict(t=round(time.time() - t_start, 1), groups=groups, frames=frames_done, ik_fps=round(frames_done / el),
                                 lane_busy=round(busy_slots / (groups * B * K), 3), ready_q=ready_q.qsize(), pending=len(pending),
                                 prep_q=prep_q.qsize(), arena_used=arena.used(), ok=stats["clips_ok"]))
            print("[gpu_convert] ik", json.dumps(timeline[-1]), flush=True)
    t_ik = time.time() - (t_ik0 or time.time())
    post_q.put(None)
    pt_thread.join()
    prep_pool.close(); io_pool.close(); shape_pool.close(); prep_pool.join(); io_pool.join(); shape_pool.join()
    mon.stop = True
    wall = time.time() - t_start
    out = dict(clips_ok=stats.get("clips_ok", 0), clips_err=stats.get("clips_err", 0), frames=frames_done,
               source_hours=round(frames_done / FPS_SRC / 3600, 3), wall_s=round(wall, 1),
               source_hours_per_hour_wall=round(frames_done / FPS_SRC / wall, 1),
               ik_phase_s=round(t_ik, 1), ik_phase_frames_per_s=round(frames_done / max(t_ik, 1e-9)),
               source_hours_per_hour_ik_phase=round(frames_done / FPS_SRC / max(t_ik, 1e-9), 1),
               t_first_ready_s=round((t_first_ready[0] or t_start) - t_start, 1),
               lane_busy=round(busy_slots / max(1, groups * B * K), 3), groups=groups,
               stage_seconds={k: round(v, 1) for k, v in tstat.items()}, counters=dict(stats), discovery=dict(counts),
               gpu_whole_run=mon.summary(), gpu_ik_phase=(mon.summary(t_ik0, t_ik0 + t_ik) if t_ik0 else None),
               peak_gpu_mem_gb_this_process=round(torch.cuda.max_memory_allocated() / 1e9, 2),
               errors=errors[:5], timeline=timeline[-30:], args=vars(args))
    print("[gpu_convert] summary", json.dumps({k: v for k, v in out.items() if k != "timeline"}), flush=True)
    if args.summary:
        Path(args.summary).write_text(json.dumps(out, indent=1))
    logf.close()


if __name__ == "__main__":
    main()
