"""Segmentation parity: SegRunner (B lanes, warm-up W) with the CPU path's own betas vs the CPU npz outputs.
usage: test_seg.py clips.txt B W1,W2,... [K]"""
from surface_retargeting.config import CFG as SRCFG  # machine paths: configs/sr.toml
import os
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", f'{SRCFG.convert_work}/.inductor_cache')
import sys, json, time
import multiprocessing as mp
from pathlib import Path
import numpy as np

DATA = Path(f'{SRCFG.data_root}')


def _p(d):
    from surface_retargeting.ingest.convert_gpu import prep
    meta = json.loads((Path(d) / "metadata.json").read_text())
    cdir = DATA / "smplx/hiphi/data" / meta["frame"] / meta["lu"] / meta["motion_id"]
    z = np.load(cdir / "motion_actor_smplx.npz")
    p = prep.prep_clip(str(Path(d) / "motion_actor.bvh"))
    ft, rest = prep.shape_clip(z["betas"], p["targets"], p["reference"])
    return dict(ft=ft, ref=p["reference"], rest=rest, cpu_pose=z["poses"].reshape(-1, 55, 3)[:, :22], cpu_trans=z["trans"],
                cpu_joints=z["fitted_joints"], cpu_loss=z["final_losses"])


def main():
    clips = [l.strip() for l in open(sys.argv[1]) if l.strip()]
    B = int(sys.argv[2]); global EXACT; EXACT = os.environ.get("EXACT", "1") == "1"; Ws = [int(x) for x in sys.argv[3].split(",")]; K = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    with mp.get_context("spawn").Pool(8) as pool:
        P = pool.map(_p, clips, chunksize=1)
    import torch, gpu_ik, seg_ik
    dev = "cuda"
    ik = gpu_ik.FastMinkIK(np.repeat(P[0]["rest"][None], B, 0), device=dev)
    t = time.time(); run = seg_ik.SegRunner(ik, K=K); print("compile+capture", round(time.time() - t, 1), flush=True)
    Fs = [len(p["ft"]) for p in P]
    tgt = torch.as_tensor(np.concatenate([p["ft"] for p in P]), device=dev)
    ref = torch.as_tensor(np.concatenate([p["ref"] for p in P]), device=dev)
    rest_tab, off_tab = seg_ik.rest_tables([p["rest"] for p in P], dev)
    cpu_pose = np.concatenate([p["cpu_pose"] for p in P]); cpu_j = np.concatenate([p["cpu_joints"] for p in P])
    cpu_t = np.concatenate([p["cpu_trans"] for p in P])
    for W in Ws:
        torch.cuda.synchronize(); t = time.time()
        out, st = run.solve_wave(tgt, ref, rest_tab, off_tab, Fs, W=W, exact=EXACT)
        torch.cuda.synchronize(); dt = time.time() - t
        pose = out["pose"].cpu().numpy(); jj = out["joints"].cpu().numpy(); tr = out["trans"].cpu().numpy()
        dj = np.linalg.norm(jj - cpu_j, axis=-1).max(-1) * 1000      # mm, per frame (max over 22 joints)
        dtr = np.linalg.norm(tr - cpu_t, axis=-1) * 1000
        dp = np.abs(pose - cpu_pose).max((-1, -2))
        print(json.dumps(dict(W=W, B=B, K=K, sched=st, secs=round(dt, 2),
                              frames_per_s=round(st["N"] / dt), pose_bitwise_frames=float((dp == 0).mean()),
                              joints_bitwise_frames=float((dj == 0).mean()), joint_mm_max=float(dj.max()),
                              joint_mm_p999=float(np.percentile(dj, 99.9)), joint_mm_median=float(np.median(dj)),
                              trans_mm_max=float(dtr.max()), pose_rad_max=float(dp.max()),
                              frames_gt_0p01mm=int((dj > 0.01).sum()), frames_gt_0p1mm=int((dj > 0.1).sum()))), flush=True)


if __name__ == "__main__":
    main()
