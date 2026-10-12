"""Control arm for the beta parity: the CPU path's own float32 fit_betas (upstream code, unchanged) on first-frame
targets perturbed by +1e-5 m (0.01 mm). Shows how far the CPU betas move under input noise far below mocap
precision.  usage: beta_control.py clips.txt out.npy [workers]   (clips.txt = raw clip dirs)"""
import os, sys, json, multiprocessing as mp
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np


def one(d):
    import torch
    torch.set_num_threads(1)
    from surface_retargeting.ingest.convert_gpu import prep
    from hiphi2smplx.beta_fitting import BetaFitConfig, fit_betas
    meta = json.load(open(os.path.join(d, "metadata.json")))
    r = prep.prep_clip(os.path.join(d, "motion_actor.bvh"), max_frames=1)
    nb = prep.neutral()
    b = fit_betas(r["targets"][0] + np.float32(1e-5), nb.rest_joints, prep.MODEL,
                  config=BetaFitConfig(device="cpu", data_dir=prep.bvh2smplx.BETA_DATA)).betas
    return f"data/{meta['frame']}/{meta['lu']}/{meta['motion_id']}", b


if __name__ == "__main__":
    clips = [l.strip() for l in open(sys.argv[1]) if l.strip()]
    with mp.get_context("spawn").Pool(int(sys.argv[3]) if len(sys.argv) > 3 else 6) as p:
        res = dict(p.map(one, clips, chunksize=1))
    np.save(sys.argv[2], res, allow_pickle=True)
    print("control betas", len(res))
