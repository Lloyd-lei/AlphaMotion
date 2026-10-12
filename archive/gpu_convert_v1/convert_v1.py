"""GPU-batched BVH -> SMPL-X converter (HiPHI / OmniContact), output layout identical to
umr_factory_data/scripts/convert_hiphi.py / convert_omni.py.

usage (genmo-gem python, read-only use of that env):
  convert.py --kind hiphi [--parts 0002] [--out /home/arenalabs/umr_factory_data/smplx_gpu/hiphi]
             [--fps30 /home/arenalabs/umr_factory_data/smplx_gpu/hiphi_30fps] [--include-cpu-done] [--force]
             [--clips FILE] [--limit N] [--beta gpu|cpu-npz] [--lanes 1024] [--prep-workers 10] [--io-workers 6]
  convert.py --kind omni  [--out /home/arenalabs/umr_factory_data/smplx_gpu/omnicontact] ...

Default skip logic (coordination with the CPU converter): a clip is skipped if the CPU path already wrote
smplx/<kind>/.../qa.json, or if this converter already wrote <out>/.../qa.json (unless --force).
Pipeline per wave (<= lanes clips): CPU pool prep (BVH -> targets) -> GPU beta fit (batched float64) -> CPU pool
shape (rest joints + foot targets) -> GPU IK (one lane per clip, CUDA graph, all lanes in lock-step) -> GPU SMPL-X
lowest-vertex QA -> CPU pool writers (npz, copies, qa.json, 30 fps variant).
"""
from __future__ import annotations

import os

os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/home/arenalabs/gpu_convert/.inductor_cache")
import argparse
import json
import multiprocessing as mp
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
DATA = Path("/home/arenalabs/umr_factory_data")


# ------------------------------------------------------------------------------------------------ discovery
def discover_hiphi(args):
    parts = sorted((DATA / "raw/hiphi/extracted").glob("part_*")) if not args.parts else \
        [DATA / f"raw/hiphi/extracted/part_{p}" for p in args.parts.split(",")]
    out = []
    for part in parts:
        if not (part / ".extracted").exists():
            continue
        for m in sorted(part.glob("HiPHI/data/*/*/*/metadata.json")):
            out.append(m.parent)
    return out


def clip_rel_hiphi(meta):
    return Path(meta["frame"]) / meta["lu"] / meta["motion_id"]


def discover_omni(args):
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
    return json.loads((d / ("metadata.json" if kind == "hiphi" else "capture_meta.json")).read_text())


def dst_dir(kind, out_root, meta):
    if kind == "hiphi":
        return out_root / "data" / clip_rel_hiphi(meta)
    return out_root / meta["domain"] / meta["normalized_case"] / meta["capture_id"]


def cpu_dir(kind, meta):
    return dst_dir(kind, DATA / "smplx" / ("hiphi" if kind == "hiphi" else "omnicontact"), meta)


def bvh_of(kind, d, meta):
    return str(d / ("motion_actor.bvh" if kind == "hiphi" else meta["human_motion_file"]))


def n_frames(kind, meta):
    return int(meta.get("frame_count") or 0)


# ------------------------------------------------------------------------------------------------ workers
def _winit():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""   # workers never touch the GPU
    import prep  # noqa: F401


def _prep_task(a):
    kind, d, bvh, max_frames = a
    import prep
    try:
        return d, prep.prep_clip(bvh, max_frames=max_frames)
    except Exception as e:  # recorded in qa.json like the CPU path
        import traceback
        return d, {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-2000:]}


def _shape_task(a):
    import prep
    betas, targets, reference = a
    return prep.shape_clip(betas, targets, reference)


def _write_task(job):
    import prep
    return prep.write_hiphi(job) if job["kind"] == "hiphi" else prep.write_omni(job)


# ------------------------------------------------------------------------------------------------ GPU monitor
class GpuMon(threading.Thread):
    """samples device utilisation and this process' SM share (nvidia-smi pmon) every ~1 s."""

    def __init__(self, pid):
        super().__init__(daemon=True)
        self.pid, self.stop, self.dev, self.mine, self.mem = str(pid), False, [], [], []

    def run(self):
        while not self.stop:
            try:
                o = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True, timeout=5).stdout.strip().split(",")
                self.dev.append(float(o[0])); self.mem.append(float(o[1]))
                p = subprocess.run(["nvidia-smi", "pmon", "-c", "1", "-s", "u"], capture_output=True, text=True, timeout=5).stdout
                sm = 0.0
                for ln in p.splitlines():
                    f = ln.split()
                    if len(f) > 3 and f[1] == self.pid and f[3] not in ("-",):
                        sm = float(f[3])
                self.mine.append(sm)
            except Exception:
                pass
            time.sleep(1.0)

    def summary(self):
        a = lambda x: (round(float(np.mean(x)), 1) if x else None)
        return dict(device_util_mean=a(self.dev), device_util_p50=(float(np.median(self.dev)) if self.dev else None),
                    this_process_sm_mean=a(self.mine), this_process_sm_p50=(float(np.median(self.mine)) if self.mine else None),
                    mem_used_mib_max=(max(self.mem) if self.mem else None), samples=len(self.dev))


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
    ap.add_argument("--beta", choices=["gpu", "cpu-npz"], default="gpu")
    ap.add_argument("--lanes", type=int, default=1024)
    ap.add_argument("--max-slots", type=float, default=6.5e6, help="lanes x max frames per wave (GPU memory bound)")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--prep-workers", type=int, default=10)
    ap.add_argument("--io-workers", type=int, default=6)
    ap.add_argument("--log", default=None)
    ap.add_argument("--summary", default=None)
    args = ap.parse_args()
    kind = args.kind
    out_root = Path(args.out or (DATA / "smplx_gpu" / ("hiphi" if kind == "hiphi" else "omnicontact")))
    out_root.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # ---- clip list
    dirs = [Path(x.strip()) for x in open(args.clips)] if args.clips else (discover_hiphi(args) if kind == "hiphi" else discover_omni(args))
    todo, skipped_cpu, skipped_done = [], 0, 0
    for d in dirs:
        meta = load_meta(kind, d)
        if not args.include_cpu_done and (cpu_dir(kind, meta) / "qa.json").exists():
            skipped_cpu += 1; continue
        if not args.force and (dst_dir(kind, out_root, meta) / "qa.json").exists():
            skipped_done += 1; continue
        todo.append((d, meta))
    if args.limit:
        todo = todo[:args.limit]
    print(f"[gpu_convert] {kind}: {len(dirs)} found, {skipped_cpu} skipped (CPU done), {skipped_done} skipped (GPU done), {len(todo)} todo", flush=True)
    if not todo:
        return
    if kind == "hiphi":
        for root in [out_root] + ([Path(args.fps30)] if args.fps30 else []):
            om = root / "object_meshes"
            root.mkdir(parents=True, exist_ok=True)
            if not om.exists():
                r = subprocess.run(["cp", "-al", str(DATA / "raw/hiphi/object_meshes"), str(om)], capture_output=True)
                if r.returncode != 0:   # other filesystem: symlink the shared mesh dir instead
                    subprocess.run(["rm", "-rf", str(om)]); om.symlink_to(DATA / "raw/hiphi/object_meshes")

    # ---- waves (longest first, bounded by lanes and lanes x frames)
    todo.sort(key=lambda x: -n_frames(kind, x[1]))
    waves, cur = [], []
    for item in todo:
        f = n_frames(kind, item[1]) or 10000
        fmax = max([n_frames(kind, m) or 10000 for _, m in cur] + [f])
        if cur and (len(cur) + 1 > args.lanes or (len(cur) + 1) * fmax > args.max_slots):
            waves.append(cur); cur = []
        cur.append(item)
    if cur:
        waves.append(cur)
    lanes = max(len(w) for w in waves)
    lanes = int(np.ceil(lanes / 128) * 128)  # lane buckets of 128 -> few distinct compiled shapes
    print(f"[gpu_convert] {len(waves)} wave(s), sizes {[len(w) for w in waves]}, lanes {lanes}", flush=True)

    import torch
    import gpu_ik
    import gpu_beta
    import gpu_smplx

    ctx = mp.get_context("spawn")
    prep_pool = ctx.Pool(args.prep_workers, initializer=_winit, maxtasksperchild=200)
    io_pool = ctx.Pool(args.io_workers, initializer=_winit, maxtasksperchild=200)
    mon = GpuMon(os.getpid()); mon.start()
    logf = open(args.log or (out_root / "convert_log.jsonl"), "a")
    flat = [(kind, d, bvh_of(kind, d, m), args.max_frames) for w in waves for d, m in w]
    prep_iter = prep_pool.imap(_prep_task, flat, chunksize=1)
    bf = gpu_beta.BatchedBetaFit("cuda")
    sm = gpu_smplx.GpuSmplx("cuda")
    ik = runner = None
    stats = dict(waves=[], clips_ok=0, clips_err=0, frames=0)
    pending = []
    for wi, wave in enumerate(waves):
        tw = time.time()
        preps, metas, srcs = [], [], []
        for d, meta in wave:
            dd, pr = next(prep_iter)
            assert dd == d
            preps.append(pr); metas.append(meta); srcs.append(d)
        t_prep = time.time() - tw
        ok_idx = [i for i, p in enumerate(preps) if "error" not in p]
        # ---- betas
        t0 = time.time()
        if args.beta == "gpu":
            br = bf.fit(np.stack([preps[i]["beta_target"] for i in ok_idx])) if ok_idx else None
            betas = {i: br["betas"][k] for k, i in enumerate(ok_idx)}
            binfo = {i: dict(beta_fit="gpu_f64_multistart_newton", beta_loss_w5=float(br["loss_w5"][k]),
                             beta_loss_w0=float(br["loss_w0"][k]), beta_mean_joint_err_cm=round(float(br["mean_joint_err"][k] * 100), 3),
                             beta_grad_max=float(br["grad_max"][k]), beta_start=int(br["start_used"][k]))
                     for k, i in enumerate(ok_idx)}
        else:
            betas = {i: np.load(cpu_dir(kind, metas[i]) / ("motion_actor_smplx.npz" if kind == "hiphi" else "betas.npy"))
                     for i in ok_idx}
            betas = {i: (b["betas"] if hasattr(b, "files") else b).astype(np.float32) for i, b in betas.items()}
            binfo = {i: dict(beta_fit="cpu-npz (betas copied from the CPU converter output)") for i in ok_idx}
        torch.cuda.synchronize(); t_beta = time.time() - t0
        # ---- shape (CPU pool)
        t0 = time.time()
        shp = io_pool.map(_shape_task, [(betas[i], preps[i]["targets"], preps[i]["reference"]) for i in ok_idx], chunksize=4)
        fit_t = {i: s[0] for i, s in zip(ok_idx, shp)}
        rest = {i: s[1] for i, s in zip(ok_idx, shp)}
        t_shape = time.time() - t0
        # ---- IK
        t0 = time.time()
        L = len(ok_idx)
        Fs = [preps[i]["frames"] for i in ok_idx]
        K = args.K
        Fmax = int(np.ceil(max(Fs) / K) * K) if Fs else 0
        if ik is None:
            neutral_rest = np.repeat(np.asarray(__import__("prep").neutral().rest_joints)[None], lanes, 0)
            ik = gpu_ik.FastMinkIK(neutral_rest, device="cuda")
            tc = time.time()
            runner = gpu_ik.GraphRunner(ik, K=K)
            print(f"[gpu_convert] IK compiled+captured for {lanes} lanes in {time.time() - tc:.1f}s", flush=True)
        rj = np.repeat(np.asarray(__import__("prep").neutral().rest_joints)[None], lanes, 0)
        for k, i in enumerate(ok_idx):
            rj[k] = rest[i]
        ik.set_rest(rj)
        dev = "cuda"
        Tg = torch.zeros(lanes, Fmax, 22, 3, dtype=torch.float32, device=dev)
        Rg = torch.zeros(lanes, Fmax, 22, 3, dtype=torch.float32, device=dev)
        for k, i in enumerate(ok_idx):
            Tg[k, :Fs[k]] = torch.from_numpy(fit_t[i]).to(dev, non_blocking=False)
            Rg[k, :Fs[k]] = torch.from_numpy(preps[i]["reference"]).to(dev)
            if Fs[k] < Fmax:   # pad with the last frame (inactive anyway)
                Tg[k, Fs[k]:] = Tg[k, Fs[k] - 1]; Rg[k, Fs[k]:] = Rg[k, Fs[k] - 1]
        lens = torch.zeros(lanes, dtype=torch.long, device=dev)
        lens[:L] = torch.tensor(Fs, device=dev)
        Op = torch.zeros(lanes, Fmax, 22, 3, dtype=torch.float32, device=dev)
        Ot = torch.zeros(lanes, Fmax, 3, dtype=torch.float32, device=dev)
        Oj = torch.zeros(lanes, Fmax, 22, 3, dtype=torch.float32, device=dev)
        Ol = torch.zeros(lanes, Fmax, dtype=torch.float32, device=dev)
        On = torch.zeros(lanes, Fmax, dtype=torch.int32, device=dev)
        arK = torch.arange(K, device=dev)
        torch.cuda.synchronize(); t_upload = time.time() - t0; t0 = time.time()
        for s in range(0, Fmax, K):
            runner.tgt.copy_(Tg[:, s:s + K]); runner.ref.copy_(Rg[:, s:s + K])
            fr = s + arK
            runner.active.copy_(fr[None, :] < lens[:, None])
            runner.first.copy_((fr == 0)[None, :].expand(lanes, K))
            runner.run()
            Op[:, s:s + K] = runner.o_pose; Ot[:, s:s + K] = runner.o_trans; Oj[:, s:s + K] = runner.o_joints
            Ol[:, s:s + K] = runner.o_loss; On[:, s:s + K] = runner.o_nit
        torch.cuda.synchronize(); t_ik = time.time() - t0
        # ---- QA lows + writers
        t0 = time.time()
        Opc, Otc, Ojc, Olc, Onc = (x[:L].cpu().numpy() for x in (Op, Ot, Oj, Ol, On))
        del Tg, Rg, Op, Ot, Oj, Ol, On
        wave_frames = int(sum(Fs))
        per_clip_s = (time.time() - tw) / max(1, len(wave))
        for k, i in enumerate(ok_idx):
            F = Fs[k]
            poses55 = np.zeros((F, 55, 3), np.float32)
            poses55[:, :22] = Opc[k, :F]
            lows = sm.lowest(poses55.reshape(F, 165), Otc[k, :F], betas[i])
            res = dict(poses55=poses55, trans=Otc[k, :F], betas=np.asarray(betas[i], np.float32), joints=Ojc[k, :F],
                       losses=Olc[k, :F], fit_targets=fit_t[i], frame_time=preps[i]["frame_time"], layout=preps[i]["layout"])
            gpu = dict(binfo[i], ik_iters_mean=round(float(Onc[k, :F].mean()), 3), wave=wi, lanes=lanes,
                       prep_s=round(preps[i]["prep_s"], 2), pipeline="gpu_convert v1")
            pending.append(io_pool.apply_async(_write_task, (_job(kind, srcs[i], out_root, metas[i], res, lows, gpu, per_clip_s, args),)))
        for i, p in enumerate(preps):
            if "error" in p:
                pending.append(io_pool.apply_async(_write_task, (_job(kind, srcs[i], out_root, metas[i], None, None, {}, 0.0, args, error=p["error"]),)))
        t_qa = time.time() - t0
        w = dict(wave=wi, clips=len(wave), ok=L, frames=wave_frames, Fmax=Fmax, lanes=lanes, t_prep_wait=round(t_prep, 1),
                 t_beta=round(t_beta, 1), t_shape=round(t_shape, 1), t_upload=round(t_upload, 1), t_ik=round(t_ik, 1),
                 ik_frames_per_s=round(wave_frames / max(t_ik, 1e-9)), ik_ms_per_step=round(1000 * t_ik / max(1, Fmax), 2),
                 t_qa_dispatch=round(t_qa, 1), t_wave=round(time.time() - tw, 1))
        stats["waves"].append(w); stats["frames"] += wave_frames
        print("[gpu_convert] wave", json.dumps(w), flush=True)
    tw = time.time()
    for p in pending:
        r = p.get(timeout=3600)
        logf.write(json.dumps(r) + "\n")
        stats["clips_ok" if r.get("ok") else "clips_err"] += 1
    logf.close()
    stats["t_write_drain"] = round(time.time() - tw, 1)
    prep_pool.close(); io_pool.close(); prep_pool.join(); io_pool.join()
    mon.stop = True
    wall = time.time() - t_start
    fps_src = 90.0 if kind == "hiphi" else 90.0
    stats.update(wall_s=round(wall, 1), source_hours=round(stats["frames"] / fps_src / 3600, 3),
                 source_hours_per_hour=round(stats["frames"] / fps_src / 3600 / (wall / 3600), 1), gpu=mon.summary(),
                 peak_gpu_mem_gb_this_process=round(torch.cuda.max_memory_allocated() / 1e9, 2),
                 args=vars(args))
    print("[gpu_convert] summary", json.dumps(stats), flush=True)
    if args.summary:
        Path(args.summary).write_text(json.dumps(stats, indent=1))


def _job(kind, src, out_root, meta, res, lows, gpu, seconds, args, error=None):
    if kind == "hiphi":
        rec_base = {"clip_id": f"hiphi/{meta['frame']}/{meta['lu']}/{meta['motion_id']}", "source": "HiPHI", "raw_dir": str(src),
                    "actor_id": meta.get("actor_id"), "actor_metadata": meta.get("actor_metadata"),
                    "is_hoi": meta.get("is_hoi"), "mirrored": meta.get("mirrored")}
    else:
        rec_base = {"clip_id": f"omnicontact/{meta['domain']}/{meta['normalized_case']}/{meta['capture_id']}",
                    "source": "OmniContact", "raw_dir": str(src)}
    return dict(kind=kind, src=str(src), out_root=str(out_root), meta=meta, rec_base=rec_base, res=res, lows=lows, gpu=gpu,
                seconds=seconds, fps30_root=(args.fps30 if kind == "hiphi" else None), error=error)


if __name__ == "__main__":
    main()
