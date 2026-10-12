"""Beta-fit objective check: is the GPU float64 Newton solution a better minimiser of the SAME objective that the
CPU path's float32 LBFGS minimises?

For each clip: re-run the upstream CPU fit_betas stages verbatim (float32 LBFGS, same config; returns the full
nuisance state that BetaFitResult drops) -> evaluate the stage-2 objective (pose-preserve weight 5) in float64 at
the CPU solution with gpu_beta.loss_body, and compare with the GPU solution's objective. Also checks that the
re-run betas are bitwise equal to the CPU converter's stored betas (so the state is the CPU path's state).
usage: beta_objective_check.py clips.txt out.json [workers]
"""
import os, sys, json, multiprocessing as mp
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np

DATA = "/home/arenalabs/umr_factory_data"


def cpu_state(d):
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0, "/home/arenalabs/gpu_convert")
    import prep
    import h5py
    from pathlib import Path
    import hiphi2smplx.beta_fitting as BF
    meta = json.load(open(os.path.join(d, "metadata.json")))
    r = prep.prep_clip(os.path.join(d, "motion_actor.bvh"), max_frames=1)
    nb = prep.neutral()
    cfg = BF.BetaFitConfig(device="cpu", data_dir=prep.bvh2smplx.BETA_DATA)
    mean_path, prior_path = BF._resolve_fitting_resources(cfg.data_dir)
    targets_array = BF._prepare_target_joints(r["targets"][0], nb.rest_joints)
    device = torch.device("cpu")
    model = BF._create_skeleton_model(Path(prep.MODEL).resolve(), device)
    pose_prior = BF._MaxMixturePrior(prior_path).to(device)
    with h5py.File(mean_path, "r") as h:
        initial_pose = torch.from_numpy(h["pose"][:]).reshape(1, 72).float()
    targets = torch.as_tensor(targets_array).unsqueeze(0)
    body_pose = initial_pose[:, 3:].detach().clone(); global_orient = initial_pose[:, :3].detach().clone()
    betas = torch.zeros(1, 10)
    out0 = model(global_orient=global_orient, body_pose=body_pose, betas=betas)
    est = (targets[:, BF._ALIGNMENT_JOINTS] - out0.joints[:, BF._ALIGNMENT_JOINTS]).mean(dim=1, keepdim=True).detach()
    cam = est.clone(); preserve = body_pose.detach().clone()
    global_orient.requires_grad = True; cam.requires_grad = True
    opt = torch.optim.LBFGS((global_orient, cam), max_iter=cfg.iterations, lr=cfg.step_size, line_search_fn="strong_wolfe")
    for _ in range(int(os.environ.get("JOINT2SMPL_CAMERA_OUTER_ITERS", "10"))):
        def c1():
            opt.zero_grad()
            cur = model(global_orient=global_orient, body_pose=body_pose, betas=betas)
            l = BF._camera_fitting_loss(cur.joints, cam, est, targets); l.backward(); return l
        opt.step(c1)
    body_pose.requires_grad = True; betas.requires_grad = True
    opt = torch.optim.LBFGS((body_pose, betas, global_orient, cam), max_iter=cfg.iterations, lr=cfg.step_size, line_search_fn="strong_wolfe")
    conf = torch.ones(22)

    def bl(w):
        cur = model(global_orient=global_orient, body_pose=body_pose, betas=betas)
        return BF._body_fitting_loss(body_pose, preserve, betas, cur.joints, cam, targets, pose_prior, conf, pose_preserve_weight=w)
    for _ in range(int(os.environ.get("JOINT2SMPL_BODY_OUTER_ITERS", str(cfg.iterations)))):
        def c2():
            opt.zero_grad(); l = bl(5.0); l.backward(); return l
        opt.step(c2)
    with torch.no_grad():
        l5 = float(bl(5.0)); l0 = float(bl(0.0))
    x = np.concatenate([body_pose.detach().numpy()[0], betas.detach().numpy()[0], global_orient.detach().numpy()[0], cam.detach().numpy().reshape(3)])
    stored = np.load(os.path.join(DATA, "smplx/hiphi/data", meta["frame"], meta["lu"], meta["motion_id"], "motion_actor_smplx.npz"))["betas"]
    return d, dict(x=x, cpu_loss_w5_f32=l5, cpu_loss_w0_f32=l0, target=targets_array, betas_equal_stored=bool(np.array_equal(x[69:79].astype(np.float32), stored)))


def main():
    clips = [l.strip() for l in open(sys.argv[1]) if l.strip()]
    with mp.get_context("spawn").Pool(int(sys.argv[3]) if len(sys.argv) > 3 else 8) as p:
        res = dict(p.map(cpu_state, clips, chunksize=1))
    import torch
    sys.path.insert(0, "/home/arenalabs/gpu_convert")
    import gpu_beta
    bf = gpu_beta.BatchedBetaFit("cuda")
    X = np.stack([res[c]["target"] for c in clips])
    g = bf.fit(X)
    kw = dict(dtype=torch.float64, device="cuda")
    bp0 = bf.mean_pose[3:]
    rows = []
    for i, c in enumerate(clips):
        x = torch.tensor(res[c]["x"], **kw); t = torch.tensor(res[c]["target"], **kw)
        f_cpu = float(bf.loss_body(x, t, bp0, 5.0))
        rows.append(dict(clip=c, betas_equal_stored=res[c]["betas_equal_stored"], obj_cpu_f64=f_cpu, obj_cpu_f32=res[c]["cpu_loss_w5_f32"],
                         obj_gpu_f64=float(g["loss_w5"][i]), gpu_minus_cpu=float(g["loss_w5"][i]) - f_cpu,
                         betas_maxabs_diff=float(np.abs(g["betas"][i] - res[c]["x"][69:79]).max())))
    d = np.array([r["gpu_minus_cpu"] for r in rows]); rel = d / np.array([r["obj_cpu_f64"] for r in rows])
    summ = dict(n=len(rows), cpu_state_reproduces_stored_betas=sum(r["betas_equal_stored"] for r in rows),
                gpu_objective_lower_or_equal=int((d <= 1e-6 * np.abs([r["obj_cpu_f64"] for r in rows])).sum()),
                rel_diff_median=float(np.median(rel)), rel_diff_min=float(rel.min()), rel_diff_max=float(rel.max()),
                f64_vs_f32_eval_reldiff_max=float(max(abs(r["obj_cpu_f64"] - r["obj_cpu_f32"]) / r["obj_cpu_f64"] for r in rows)))
    json.dump(dict(summary=summ, clips=rows), open(sys.argv[2], "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
