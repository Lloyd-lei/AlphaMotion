"""train_runner.py — train ONE variant with **conditional flow matching (FM)**.

Stage 1 v2 — FM (replaces deterministic regression).

Per-step:
    1. Sample x_0 ~ N(0, I) for each stream (pos / rot6d / root)
    2. Sample t ~ U(0, 1) per batch element
    3. x_t = (1 - t) * x_0 + t * x_1     (linear interpolation)
       v_target = x_1 - x_0              (linear-FM target velocity)
    4. v_pred = model(x_pos_t, x_rot_t, x_root_t, t)
    5. Endpoint extrapolation:  x̂_1 = x_t + (1 - t) * v_pred
       For perfect v_pred, x̂_1 ≡ x_1 exactly.
    6. 4-component loss with user-chosen weights:
       L_pos = MSE(v_pos_pred, v_pos_target)             × λ_pos = 10
       L_rot = MSE(v_rot_pred, v_rot_target)             × λ_rot = 1
       L_root = MSE(v_root_pred, v_root_target)          × λ_root = 2
       L_FK = λ(t) · MSE(FK(R̂_endpoint), p̂_endpoint_un)  × λ_fk  = 2
       (FK is computed in un-normalized world meters)

       λ(t) = t²  — emphasize endpoint correctness (t ≈ 1 = sample target)

Robustness (preserved from v1):
    - SIGINT → save → exit cleanly
    - OOM → halve batch + double grad_accum, capped at 5 retries then 'failed_oom'
    - Resume from latest.pt
    - Atomic status.json + loss_history.json writes
    - 4 loss components individually logged to TB and loss_history.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

SCRIPTS_01 = Path(__file__).resolve().parent.parent.parent / "01_resources/scripts"
sys.path.insert(0, str(SCRIPTS_01))

from model_builder  import build_from_spec
from data_loader    import load_amass_canonical
from soma_skeleton  import rot6d_to_matrix, forward_kinematics_torch


MODE_PRESETS = {
    "smoke":    {"n_steps":   1000, "batch": 32,
                 "variants":  ["full", "opm_only", "triangle_only", "pair_static",
                               "axial_only", "baseline"]},
    "headline": {"n_steps":  30000, "batch": 64,
                 "variants":  ["full", "opm_only", "triangle_only", "pair_static"]},
    "full":     {"n_steps":  60000, "batch": 64,
                 "variants":  ["full", "opm_only", "triangle_only", "pair_static",
                               "axial_only", "baseline"]},
}


# ---------- atomic JSON writes (monitor-readable) ---------------------------

def _atomic_write_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def _write_status(status_path: Path, **fields) -> None:
    base = {}
    if status_path.exists():
        try: base = json.loads(status_path.read_text())
        except Exception: base = {}
    base.update(fields)
    base["last_update"] = time.time()
    _atomic_write_json(status_path, base)


# ---------- main training routine -------------------------------------------

def train_variant(
    variant: str,
    *,
    mode: str = "smoke",
    n_steps: Optional[int] = None,
    batch: Optional[int] = None,
    resume: bool = True,
    use_fp16: bool = True,
    use_compile: bool = False,
    seed: int = 0,
    device: str = "cuda",
    log_every: int = 50,
    save_every: int = 5000,
    ckpt_root: Optional[Path] = None,
) -> dict:
    REPO  = Path(__file__).resolve().parent.parent.parent.parent
    CFG   = REPO / "notebook/01_resources/configs"
    DATA  = json.loads((CFG / "data.json").read_text())
    VARS  = json.loads((CFG / "variants.json").read_text())
    TRAIN_DEFAULT = json.loads((CFG / "train_default.json").read_text())

    if variant not in VARS:
        raise ValueError(f"unknown variant: {variant}")
    spec = VARS[variant]["model"]

    preset = MODE_PRESETS[mode]
    n_steps = n_steps if n_steps is not None else preset["n_steps"]
    batch   = batch   if batch   is not None else preset["batch"]

    # Loss weights from TRAIN_DEFAULT (NB1 §0.4)
    lw = TRAIN_DEFAULT["loss"]
    LAM_POS  = float(lw["pos_weight"])
    LAM_ROT  = float(lw["rot_weight"])
    LAM_ROOT = float(lw["root_weight"])
    LAM_FK   = float(lw["fk_weight"])

    if ckpt_root is None:
        ckpt_root = REPO / "notebook/02_resources/checkpoints"
    ckpt_dir   = Path(ckpt_root) / variant
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tb_dir     = REPO / f"notebook/02_resources/tensorboard/{variant}"
    tb_dir.mkdir(parents=True, exist_ok=True)
    status_path        = ckpt_dir / "status.json"
    loss_history_path  = ckpt_dir / "loss_history.json"
    latest_pt          = ckpt_dir / "latest.pt"

    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    use_cuda = device == "cuda" and torch.cuda.is_available()
    dev = torch.device("cuda" if use_cuda else "cpu")

    # ----- data -----
    train_set, _ = load_amass_canonical(DATA)
    base_ds = train_set.dataset
    sample = train_set[0]
    T, J, C = sample["pos"].shape

    # Normalization stats (for un-normalize at FK)
    pos_mean  = torch.from_numpy(base_ds.mean[0]).to(dev)         # [J, 3]
    pos_std   = torch.from_numpy(base_ds.std[0]).to(dev)
    root_mean = torch.from_numpy(base_ds.root_mean[0]).to(dev)    # [3]
    root_std  = torch.from_numpy(base_ds.root_std[0]).to(dev)
    bone_offsets = torch.from_numpy(base_ds.bone_offsets).float().to(dev)  # [J, 3]
    parents      = torch.from_numpy(base_ds.parents).to(dev)               # [J]

    loader_iter = _make_infinite_loader(train_set, batch, seed)

    # ----- model + optim -----
    model, _ = build_from_spec(spec, T=T, J=J, C=C)
    model = model.to(dev).train()
    if use_compile:
        try:    model = torch.compile(model)
        except Exception as e:
            print(f"[{variant}] torch.compile failed: {e}; continuing without compile")

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(TRAIN_DEFAULT["optim"]["lr"]),
        betas=tuple(TRAIN_DEFAULT["optim"]["betas"]),
        weight_decay=float(TRAIN_DEFAULT["optim"]["weight_decay"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(use_fp16 and use_cuda))

    # ----- resume -----
    start_step = 0
    if resume and latest_pt.exists():
        try:
            ck = torch.load(latest_pt, map_location=dev, weights_only=False)
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["optim"])
            if ck.get("scaler"):
                scaler.load_state_dict(ck["scaler"])
            start_step = ck.get("step", 0)
            print(f"[{variant}] resumed from step {start_step}")
        except Exception as e:
            print(f"[{variant}] resume failed ({e}); starting fresh (likely architecture change)")
            start_step = 0
    else:
        print(f"[{variant}] starting fresh")

    if start_step >= n_steps:
        print(f"[{variant}] already at {start_step} >= {n_steps}, nothing to do")
        return _final_status(status_path, variant, start_step, n_steps, ckpt_dir, "already_done")

    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(tb_dir), purge_step=start_step)
    except ImportError:
        writer = None
        print(f"[{variant}] tensorboard not available; skipping TB logs")

    interrupted = {"flag": False}
    def _on_sigint(sig, frame):
        print(f"\n[{variant}] SIGINT received; will save latest checkpoint then exit")
        interrupted["flag"] = True
    signal.signal(signal.SIGINT, _on_sigint)

    _write_status(
        status_path,
        variant=variant, mode=mode, status="running",
        step=start_step, n_steps=n_steps, batch=batch,
        use_fp16=use_fp16, pid=os.getpid(),
        started_at=time.time(),
        device=str(dev),
        loss_weights={"pos": LAM_POS, "rot": LAM_ROT, "root": LAM_ROOT, "fk": LAM_FK},
        regime="flow_matching_v1",
    )

    cur_batch  = batch
    grad_accum = 1
    oom_retries = 0
    MAX_OOM_RETRIES = 5
    losses_window = []
    components_window = {"pos": [], "rot": [], "root": [], "fk": []}
    step = start_step
    t_window = time.time()

    loss_history: list[dict] = []
    if resume and loss_history_path.exists():
        try:
            loss_history = json.loads(loss_history_path.read_text()).get("entries", [])
            loss_history = [e for e in loss_history if e.get("step", 0) <= start_step]
        except Exception:
            loss_history = []
    started_wall = time.time()

    print(f"[{variant}] FM training {start_step}→{n_steps}  bs={cur_batch}  fp16={use_fp16}  "
           f"weights pos={LAM_POS} rot={LAM_ROT} root={LAM_ROOT} fk={LAM_FK}  device={dev}")

    while step < n_steps and not interrupted["flag"]:
        try:
            batch_d = next(loader_iter)
            x_pos_1  = batch_d["pos"].to(dev, non_blocking=True)        # [B, T, J, 3]
            x_rot_1  = batch_d["rot6d"].to(dev, non_blocking=True)      # [B, T, J, 6]
            x_root_1 = batch_d["root"].to(dev, non_blocking=True)       # [B, T, 3]
            B = x_pos_1.shape[0]

            # ---- FM training step ----
            x_pos_0  = torch.randn_like(x_pos_1)
            x_rot_0  = torch.randn_like(x_rot_1)
            x_root_0 = torch.randn_like(x_root_1)

            t = torch.rand(B, device=dev)                          # [B] in [0, 1]
            t_pos  = t.view(B, 1, 1, 1)                            # for [B,T,J,3]
            t_rot  = t.view(B, 1, 1, 1)                            # for [B,T,J,6]
            t_root = t.view(B, 1, 1)                               # for [B,T,3]

            x_pos_t  = (1 - t_pos)  * x_pos_0  + t_pos  * x_pos_1
            x_rot_t  = (1 - t_rot)  * x_rot_0  + t_rot  * x_rot_1
            x_root_t = (1 - t_root) * x_root_0 + t_root * x_root_1

            v_pos_target  = x_pos_1  - x_pos_0
            v_rot_target  = x_rot_1  - x_rot_0
            v_root_target = x_root_1 - x_root_0

            opt.zero_grad(set_to_none=True)
            for _ga in range(grad_accum):
                with torch.amp.autocast("cuda", dtype=torch.float16,
                                         enabled=(use_fp16 and use_cuda)):
                    out = model(x_pos_t, x_rot_t, x_root_t, t)
                    v_pos_pred  = out["pos"]
                    v_rot_pred  = out["rot6d"]
                    v_root_pred = out["root"]

                    L_pos  = F.mse_loss(v_pos_pred,  v_pos_target)
                    L_rot  = F.mse_loss(v_rot_pred,  v_rot_target)
                    L_root = F.mse_loss(v_root_pred, v_root_target)

                    # ---- FK consistency at extrapolated endpoint ----
                    # x̂_1 = x_t + (1 - t) * v_pred
                    pos_endpoint  = x_pos_t  + (1 - t_pos)  * v_pos_pred
                    rot_endpoint  = x_rot_t  + (1 - t_rot)  * v_rot_pred
                    root_endpoint = x_root_t + (1 - t_root) * v_root_pred

                    # Un-normalize for FK
                    pos_endpoint_un  = pos_endpoint  * pos_std  + pos_mean        # [B,T,J,3]
                    root_endpoint_un = root_endpoint * root_std + root_mean       # [B,T,3]
                    R_hat = rot6d_to_matrix(rot_endpoint)                          # [B,T,J,3,3]

                    # FK in fp32 for stability (autocast disabled inside)
                    with torch.amp.autocast("cuda", enabled=False):
                        R_hat32   = R_hat.float()
                        root32_un = root_endpoint_un.float()
                        pos32_un  = pos_endpoint_un.float()
                        fk_pos, _ = forward_kinematics_torch(
                            R_hat32, root32_un, bone_offsets, parents,
                        )
                        # λ(t) = t²  — heavier near sample endpoint
                        lam_t = (t.float() ** 2).view(B, 1, 1, 1)
                        L_FK  = (lam_t * (fk_pos - pos32_un) ** 2).mean()

                    total = (LAM_POS * L_pos + LAM_ROT * L_rot
                            + LAM_ROOT * L_root + LAM_FK * L_FK) / grad_accum

                scaler.scale(total).backward()
            scaler.step(opt); scaler.update()

            losses_window.append(float(total.item()) * grad_accum)
            components_window["pos"].append(float(L_pos.item()))
            components_window["rot"].append(float(L_rot.item()))
            components_window["root"].append(float(L_root.item()))
            components_window["fk"].append(float(L_FK.item()))
            step += 1

            if step % log_every == 0 or step == n_steps:
                dt = time.time() - t_window
                step_ms = dt / log_every * 1000
                lwin = float(np.mean(losses_window[-log_every:]))
                cwin = {k: float(np.mean(components_window[k][-log_every:]))
                         for k in components_window}
                eta_s = step_ms / 1000 * (n_steps - step)
                lr_now = float(opt.param_groups[0]["lr"])
                gpu_mem = (torch.cuda.memory_allocated()/1024**3
                            if torch.cuda.is_available() else 0)
                print(f"[{variant}] step {step:>6d}/{n_steps}  total={lwin:.4f}  "
                       f"pos={cwin['pos']:.4f} rot={cwin['rot']:.4f} root={cwin['root']:.4f} fk={cwin['fk']:.4f}  "
                       f"step={step_ms:.0f}ms  eta={eta_s/60:.1f}min")

                if writer is not None:
                    writer.add_scalar("train/loss",       lwin, step)
                    writer.add_scalar("train/loss_pos",   cwin["pos"], step)
                    writer.add_scalar("train/loss_rot",   cwin["rot"], step)
                    writer.add_scalar("train/loss_root",  cwin["root"], step)
                    writer.add_scalar("train/loss_fk",    cwin["fk"], step)
                    writer.add_scalar("train/step_ms",    step_ms, step)
                    writer.add_scalar("train/lr",         lr_now, step)
                    if torch.cuda.is_available():
                        writer.add_scalar("train/gpu_mem_gb", gpu_mem, step)

                loss_history.append({
                    "step":        step,
                    "loss":        round(lwin, 6),
                    "loss_pos":    round(cwin["pos"],  6),
                    "loss_rot":    round(cwin["rot"],  6),
                    "loss_root":   round(cwin["root"], 6),
                    "loss_fk":     round(cwin["fk"],   6),
                    "step_ms":     round(step_ms, 2),
                    "lr":          lr_now,
                    "gpu_mem_gb":  round(float(gpu_mem), 3),
                    "wall_s":      round(time.time() - started_wall, 2),
                    "batch":       cur_batch,
                    "grad_accum":  grad_accum,
                })

                _atomic_write_json(loss_history_path, {
                    "variant":       variant,
                    "mode":          mode,
                    "regime":        "flow_matching_v1",
                    "n_steps_target": n_steps,
                    "log_every":     log_every,
                    "loss_weights":  {"pos": LAM_POS, "rot": LAM_ROT, "root": LAM_ROOT, "fk": LAM_FK},
                    "fields":        ["step", "loss", "loss_pos", "loss_rot", "loss_root",
                                       "loss_fk", "step_ms", "lr", "gpu_mem_gb", "wall_s",
                                       "batch", "grad_accum"],
                    "entries":       loss_history,
                })

                _write_status(
                    status_path,
                    step=step, loss_recent=lwin,
                    loss_pos=round(cwin["pos"], 4), loss_rot=round(cwin["rot"], 4),
                    loss_root=round(cwin["root"], 4), loss_fk=round(cwin["fk"], 4),
                    step_ms=round(step_ms, 1), eta_s=round(eta_s, 1),
                    gpu_mem_gb=round(float(gpu_mem), 2),
                )
                t_window = time.time()

            if step % save_every == 0 or step == n_steps:
                _save_checkpoint(ckpt_dir, model, opt, scaler, step,
                                  loss=losses_window[-1] if losses_window else None,
                                  tag=f"step_{step}")
                _save_checkpoint(ckpt_dir, model, opt, scaler, step,
                                  loss=losses_window[-1] if losses_window else None,
                                  tag="latest")

        except torch.cuda.OutOfMemoryError:
            oom_retries += 1
            torch.cuda.empty_cache()
            new_batch = max(1, cur_batch // 2)
            if new_batch == cur_batch or oom_retries > MAX_OOM_RETRIES:
                _write_status(status_path, status="failed_oom",
                              error=f"OOM at batch={cur_batch} after {oom_retries} retries")
                print(f"[{variant}] FATAL: OOM at batch={cur_batch} exhausted retries; exiting")
                if writer is not None: writer.close()
                return _final_status(status_path, variant, step, n_steps, ckpt_dir, "failed_oom")
            print(f"[{variant}] OOM at batch={cur_batch} → batch={new_batch} grad_accum={grad_accum*2} (retry {oom_retries}/{MAX_OOM_RETRIES})")
            cur_batch  = new_batch
            grad_accum *= 2
            loader_iter = _make_infinite_loader(train_set, cur_batch, seed + step)
            _write_status(status_path, oom_recovered=True,
                          batch=cur_batch, grad_accum=grad_accum, oom_retries=oom_retries)

    _save_checkpoint(ckpt_dir, model, opt, scaler, step,
                      loss=losses_window[-1] if losses_window else None, tag="latest")
    _atomic_write_json(loss_history_path, {
        "variant":       variant,
        "mode":          mode,
        "regime":        "flow_matching_v1",
        "n_steps_target": n_steps,
        "log_every":     log_every,
        "loss_weights":  {"pos": LAM_POS, "rot": LAM_ROT, "root": LAM_ROOT, "fk": LAM_FK},
        "fields":        ["step", "loss", "loss_pos", "loss_rot", "loss_root",
                           "loss_fk", "step_ms", "lr", "gpu_mem_gb", "wall_s",
                           "batch", "grad_accum"],
        "entries":       loss_history,
    })
    if writer is not None:
        writer.close()

    final_state = "interrupted" if interrupted["flag"] else "done"
    return _final_status(status_path, variant, step, n_steps, ckpt_dir, final_state)


def _make_infinite_loader(dataset, batch, seed):
    g = torch.Generator(); g.manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch, shuffle=True,
                         num_workers=4, drop_last=True, generator=g, pin_memory=True)
    while True:
        for b in loader:
            yield b


def _save_checkpoint(ckpt_dir, model, opt, scaler, step, *, loss=None, tag="latest"):
    path = ckpt_dir / f"{tag}.pt"
    tmp  = path.with_suffix(".pt.tmp")
    state = {
        "step":   step,
        "model":  model.state_dict(),
        "optim":  opt.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "loss":   loss,
    }
    torch.save(state, tmp)
    os.replace(tmp, path)


def _final_status(status_path, variant, step, n_steps, ckpt_dir, status):
    final = {
        "variant": variant,
        "status":  status,
        "step":    step,
        "n_steps": n_steps,
        "ckpt_path": str(ckpt_dir / "latest.pt"),
        "finished_at": time.time(),
    }
    _write_status(status_path, **final)
    print(f"[{variant}] {status}: step={step}/{n_steps}  ckpt={final['ckpt_path']}")
    return final


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True)
    p.add_argument("--mode", default="smoke", choices=list(MODE_PRESETS.keys()))
    p.add_argument("--n-steps", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--no-fp16", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    train_variant(
        args.variant,
        mode=args.mode,
        n_steps=args.n_steps,
        batch=args.batch,
        resume=not args.no_resume,
        use_fp16=not args.no_fp16,
        use_compile=args.compile,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
