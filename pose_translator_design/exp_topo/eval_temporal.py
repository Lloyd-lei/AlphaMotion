"""P1 evaluation — the 4 pre-registered ARCH_V4 criteria + the duration-seat probe.

  (i)   mp4:      prefix-1 (and prefix-8 / frame-code floor / GT) decoded windows
                  rendered as human_smpl stick skeletons — the VISUAL verdict is
                  lloyd's, this script only writes the files and prints their paths.
  (ii)  budgets:  OAT vs the unordered control at MATCHED BITS (same registers, same
                  FSQ) across prefix budgets {1,2,4,8}.
  (iii) knn:      body-identifiability of the temporal tokens, kNN probe mirroring
                  R7 (6 bodies, chance 16.7%; frame-level reference 71.6%). PASS
                  requires temporal <= frame-level on the same windows.
  (iv)  recon:    decode reconstructed frame codes through the FROZEN E19 decoder and
                  compare geodesic degrees against GT pose; PASS = within 2 deg of
                  the frame-code floor (true codes through the same decoder).
  duration:       encode the SAME content at 0.5x/1x/2x playback (frame count 30/60/
                  120 at corresponding normalized time) and report which registers'
                  integer codes move — does the learned duration seat capture scale?

P1c: this harness is arch-agnostic. Budgets come from the model's own ladder
(model_budgets): arch="global" (1,2,4,8), arch="conv" the cumulative SCALE boundaries
(2,8), arch="freq" (2,4,8) — band 0 alone is DC, i.e. a static pose by construction, so
a budget of 1 is not a motion budget. --frame_stride runs are read from the run config.
Note that budget_level_acc is NOT the headline metric for P1c; motion energy retention
(scratchpad p1c_motion_metrics.py) is.

Usage (see run_p1.sh / run_p1c.sh; sections via --do, default all):
  eval_temporal.py --run .../runs/P1_oat_z_MMDD --run_unordered .../runs/P1_unord_z_MMDD \
      --sets E19a1k --embs human_smpl,... --probe_embs human_smpl,unitree_g1_29dof,...
"""
from __future__ import annotations
import os, sys, json, argparse, subprocess, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from temporal_common import (load_groups, stream_L, budget_hidden, code_metrics,  # noqa: E402
                             BUDGETS, R7_BODIES, reg_budgets, model_budgets)
from train_temporal import load_run                                               # noqa: E402
from common import rot6d_to_matrix, geodesic_deg                                  # noqa: E402

ROOT = "/var/tmp/alphamotion/topo_exp"


# --------------------------------------------------------------------------- helpers
@torch.no_grad()
def batched_logits(model, w, n_budget, device, bs=16):
    outs = []
    for i in range(0, w.shape[0], bs):
        wb = w[i:i + bs].to(device)
        hid = budget_hidden(wb.shape[0], model.n_reg, n_budget, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
            logits, _ = model(wb, hid)
        outs.append(logits.float().argmax(-1).cpu())
    return torch.cat(outs)                                # long [n,T,D]


def resample_nn(pos_row, T_out):
    """[T] frame indices -> [T_out] at corresponding normalized time (nearest)."""
    T = len(pos_row)
    idx = np.round(np.linspace(0, T - 1, T_out)).astype(np.int64)
    return pos_row[idx]


# --------------------------------------------------------------------------- (ii) budgets
def eval_budgets(models, group, emb, pos, device, f):
    out = {}
    w = group.windows(emb, pos)
    tgt = w.long()
    for tag, model in models.items():
        out[tag] = {}
        for n in model_budgets(model):
            pred = batched_logits(model, w, n, device)
            out[tag][f"n{n}"] = {k: round(v, 4) for k, v in code_metrics(pred, tgt, f).items()}
    return out


# --------------------------------------------------------------------------- (iii) knn
@torch.no_grad()
def eval_knn(model, groups_by_emb, n_per_emb, device, k=10, bs=32):
    """Leave-one-out kNN body probe on temporal register codes, same-clip excluded —
    mirrors R7's frame-level probe (71.6%, 6 bodies, chance 1/6). Also runs the SAME
    kNN on the windows' mean dequantized frame-z as the frame-level reference."""
    feats_t, feats_f, body, clip = [], [], [], []
    for bi, (emb, (gr, pos)) in enumerate(groups_by_emb.items()):
        qs = []
        for i in range(0, len(pos), bs):
            w = gr.windows(emb, pos[i:i + bs]).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                _, q = model.encode_regs(w)
            qs.append(q.float().reshape(q.shape[0], -1).cpu())
            deq = (w.float() - model.deq_off) / model.deq_half
            feats_f.append(deq.mean(1).cpu())              # frame-level reference feature
        feats_t.append(torch.cat(qs))
        body += [bi] * len(pos)
        clip += [f"{emb}?" for _ in pos]                   # placeholder, replaced below
        # same-clip exclusion: clip id = bounds bucket of the window start
        starts = pos[:, 0]
        cid = np.searchsorted(gr.bounds, starts, side="right") - 1
        clip[-len(pos):] = [f"{gr.name}:{c}" for c in cid]
    X_t = torch.cat(feats_t); X_f = torch.cat(feats_f)
    body = torch.as_tensor(body); uclip = {c: i for i, c in enumerate(set(clip))}
    cid = torch.as_tensor([uclip[c] for c in clip])

    def knn_acc(X):
        X = torch.nn.functional.normalize(X, dim=-1).to(device)
        sim = X @ X.t()
        sim.fill_diagonal_(-2.0)
        # NOTE: same clip id but different body = the paired positives — keep those;
        # only exclude same clip AND same body (temporal duplicates of oneself)
        same = (cid[:, None] == cid[None, :]) & (body[:, None] == body[None, :])
        sim = sim.masked_fill(same.to(device), -2.0)
        nb = body.to(device)[sim.topk(k, dim=-1).indices]  # [N,k]
        pred = torch.mode(nb, dim=-1).values
        acc = (pred == body.to(device)).float()
        per = {}
        for bi, emb in enumerate(groups_by_emb):
            per[emb] = round(acc[body.to(device) == bi].mean().item(), 4)
        return round(acc.mean().item(), 4), per

    acc_t, per_t = knn_acc(X_t)
    acc_f, per_f = knn_acc(X_f)
    return {"k": k, "n_per_emb": n_per_emb, "chance": round(1 / len(groups_by_emb), 4),
            "temporal_acc": acc_t, "temporal_per_body": per_t,
            "frame_ref_acc": acc_f, "frame_ref_per_body": per_f,
            "r7_frame_reference": 0.716,
            "pass_pre_registered": bool(acc_t <= 0.716)}


# --------------------------------------------------------------------------- (iv) recon + (i) mp4
def load_frozen(run, cache, emb, device, reach):
    from tokenize_stream import load_tokenizer, load_body, geom_for
    tok, cfg = load_tokenizer(run, device)
    entry, _pose = load_body(cache, emb, device, rows=[0], reach=reach)  # geometry only
    return tok, entry, geom_for


def decode_ints_packed(tok, entry, geom_for, packed, device, K, f, bs=512):
    """packed long [n,D] frame codes -> (rot6d [n,J,6], pos [n,J,3]) via frozen E19."""
    outs_r, outs_p = [], []
    for i in range(0, packed.shape[0], bs):
        p = packed[i:i + bs].to(device)
        b = p.shape[0]
        ip = p[:, :p.shape[1] // 2].reshape(b, 1, K, f).permute(1, 0, 2, 3)
        ir = p[:, p.shape[1] // 2:].reshape(b, 1, K, f).permute(1, 0, 2, 3)
        with torch.no_grad():
            rot, pos = tok.decode_from_ints(ip, ir, geom_for(entry, b))
        outs_r.append(rot.cpu()); outs_p.append(pos.cpu())
    return torch.cat(outs_r), torch.cat(outs_p)


def gt_pose_for(cache, emb, stream, pos, device, reach):
    """GT global pose for packed window positions via rows.npy row map."""
    from tokenize_stream import load_body
    rows = stream["rows"][pos.reshape(-1)]
    entry, pose = load_body(cache, emb, device, rows=rows, reach=reach)
    return pose                                            # [n*T, J, 9] cpu


def eval_recon(model, group, emb, pos, device, tok, entry, geom_for, cache, f, K):
    st = group.streams[emb]
    reach = st["meta"]["reach"]
    gt = gt_pose_for(cache, emb, st, pos, device, reach)
    gtR = rot6d_to_matrix(gt[..., :6])
    w = group.windows(emb, pos)
    out = {}

    def deg_of(packed):
        rot, p = decode_ints_packed(tok, entry, geom_for, packed, device, K, f)
        d = geodesic_deg(rot6d_to_matrix(rot), gtR).mean().item()
        pc = (p - gt[..., 6:9]).norm(dim=-1).mean().item() * reach
        return d, pc

    floor_deg, floor_pos = deg_of(w.long().reshape(-1, w.shape[-1]))
    out["floor"] = {"rot_deg": round(floor_deg, 2), "pos_cm": round(floor_pos, 2)}
    buds = model_budgets(model)
    for n in buds:
        pred = batched_logits(model, w, n, device).reshape(-1, w.shape[-1])
        d, pc = deg_of(pred)
        out[f"n{n}"] = {"rot_deg": round(d, 2), "pos_cm": round(pc, 2),
                        "delta_vs_floor_deg": round(d - floor_deg, 2)}
    out["pass_pre_registered"] = bool(out[f"n{buds[-1]}"]["delta_vs_floor_deg"] <= 2.0)
    return out


# --------------------------------------------------------------------------- (i) mp4
def render_mp4(out_path, panels, titles, spec, fps=30, size=360):
    """panels: list of [T,J,3] cm positions. Stick skeletons, one panel per column."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    T = panels[0].shape[0]
    allp = np.concatenate([p.reshape(-1, 3) for p in panels])
    c = allp.mean(0); r = max(np.abs(allp - c).max(), 1e-3) * 1.1
    fig, axes = plt.subplots(1, len(panels), figsize=(len(panels) * size / 100, size / 100),
                             subplot_kw={"projection": "3d"}, dpi=100)
    if len(panels) == 1:
        axes = [axes]
    frames = []
    for t in range(T):
        for ax, p, ti in zip(axes, panels, titles):
            ax.cla()
            for j, par in enumerate(spec.parents):
                if par >= 0:
                    seg = p[t][[par, j]]
                    ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], "-o", color="#2a78d6",
                            lw=2, ms=2)
            ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r)
            ax.set_zlim(c[2] - r, c[2] + r)
            ax.set_title(ti, fontsize=9); ax.set_axis_off()
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        frames.append(buf.copy())
    plt.close(fig)
    h, w = frames[0].shape[:2]; h -= h % 2; w -= w % 2
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-crf", "26",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fr in frames:
        p.stdin.write(np.ascontiguousarray(fr[:h, :w], dtype=np.uint8).tobytes())
    p.stdin.close(); assert p.wait() == 0, "ffmpeg failed"
    return out_path


def eval_mp4(model, group, emb, device, tok, entry, geom_for, cache, f, K, out_dir,
             n_clips=4):
    os.makedirs(out_dir, exist_ok=True)
    st = group.streams[emb]
    reach = st["meta"]["reach"]
    T = group.T
    paths = []
    picks = group.va_pools[:n_clips] if group.va_pools else group.tr_pools[:n_clips]
    for ci, (s, e) in enumerate(picks):
        start = s + max(0, (e - s - T) // 2)
        pos = np.arange(start, start + T)[None]
        w = group.windows(emb, pos)
        gt = gt_pose_for(cache, emb, st, pos, device, reach)
        panels = [(gt[..., 6:9] * reach).numpy()]
        titles = ["GT pose"]
        specs = [("frame-code floor", w.long().reshape(-1, w.shape[-1]))]
        for n in (model_budgets(model)[-1], model_budgets(model)[0]):
            pred = batched_logits(model, w, n, device).reshape(-1, w.shape[-1])
            specs.append((f"prefix-{n}", pred))
        for ti, packed in specs:
            _, p = decode_ints_packed(tok, entry, geom_for, packed, device, K, f)
            panels.append((p * reach).numpy()); titles.append(ti)
        cid = int(np.searchsorted(group.bounds, start, side="right") - 1)
        name = group.streams[emb]["clips"][cid]["name"][:60].replace("/", "_")
        mp4 = render_mp4(f"{out_dir}/{ci:02d}_{name}.mp4", panels, titles,
                         entry["spec"])
        print("MP4:", mp4, flush=True)
        paths.append(mp4)
    return paths


# --------------------------------------------------------------------------- duration probe
@torch.no_grad()
def eval_duration(model, group, emb, device, n=128, bs=16):
    """Same content at 0.5x/2x playback = frame count 120/30 at corresponding
    normalized time. Reports per-register fraction of the 6 FSQ dims whose integer
    code moved vs the 1x encode. The reserved duration seat is the LAST register."""
    g = np.random.default_rng(11)
    T = group.T
    pos = group.sample_positions(n, g, split="val")        # [n,T]

    def ints_at(T_out):
        outs = []
        for i in range(0, n, bs):
            p = np.stack([resample_nn(row, T_out) for row in pos[i:i + bs]])
            w = group.windows(emb, p).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                outs.append(model.reg_ints(w).cpu())
        return torch.cat(outs)                             # [n,n_reg,f]

    base = ints_at(T)
    out = {}
    for tag, T_out in (("slow_2x", 2 * T), ("fast_05x", T // 2)):
        ch = (ints_at(T_out) != base).float().mean(0).mean(-1)   # [n_reg]
        out[tag] = {f"reg{r}": round(ch[r].item(), 4) for r in range(model.n_reg - 1)}
        out[tag]["duration_seat"] = round(ch[-1].item(), 4)
    return out


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="OAT run dir")
    ap.add_argument("--run_unordered", default="", help="unordered control run dir")
    ap.add_argument("--stream_root", default=f"{ROOT}/e13_stream")
    ap.add_argument("--sets", required=True)
    ap.add_argument("--embs", required=True, help="streams for budget/recon eval")
    ap.add_argument("--probe_embs", default=",".join(R7_BODIES),
                    help="bodies for the kNN probe (needs their streams encoded)")
    ap.add_argument("--recon_emb", default="human_smpl")
    ap.add_argument("--tokenizer_run", default="", help="frozen E19 run (default: stream meta)")
    ap.add_argument("--cache", default="", help="paired cache for GT pose (default: stream meta)")
    ap.add_argument("--do", default="budgets,knn,recon,mp4,duration")
    ap.add_argument("--n_val", type=int, default=256)
    ap.add_argument("--n_knn", type=int, default=300, help="windows per body for the probe")
    ap.add_argument("--n_recon", type=int, default=64)
    ap.add_argument("--viz_clips", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="", help="default {run}/eval_temporal")
    args = ap.parse_args()
    dev = args.device
    do = set(args.do.split(","))
    out_dir = args.out or f"{args.run}/eval_temporal"
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    model, cfg = load_run(args.run, dev)
    T = cfg.get("T", 60)
    results = {"run": args.run, "run_unordered": args.run_unordered}

    groups = load_groups(args.stream_root, args.sets, args.embs, T,
                         stride=cfg.get("frame_stride", 1))
    meta = groups[0].meta
    f, K = meta["f"], meta["K"]
    g0 = groups[0]
    grng = np.random.default_rng(7)
    val_pos = g0.sample_positions(min(args.n_val, 8 * max(len(g0.va_pools), 4)),
                                  grng, split="val")

    if "budgets" in do:
        models = {"oat": model}
        if args.run_unordered:
            models["unordered"], _ = load_run(args.run_unordered, dev)
        results["budgets"] = {}
        for emb in g0.embs():
            results["budgets"][emb] = eval_budgets(models, g0, emb, val_pos, dev, f)
        print("BUDGETS", json.dumps(results["budgets"], indent=1), flush=True)

    if "knn" in do:
        probe = {}
        pe = [e for e in args.probe_embs.split(",") if e]
        pg = load_groups(args.stream_root, args.sets.split(",")[0], ",".join(pe), T,
                         stride=cfg.get("frame_stride", 1))[0]
        for emb in pg.embs():
            kn = np.random.default_rng(13)
            probe[emb] = (pg, pg.sample_positions(args.n_knn, kn, split="val"))
        results["knn"] = eval_knn(model, probe, args.n_knn, dev)
        print("KNN", json.dumps(results["knn"], indent=1), flush=True)

    tok = entry = geom_for = None
    if do & {"recon", "mp4"}:
        run_e19 = args.tokenizer_run or meta["run"]
        cache = args.cache or meta["cache"]
        st = g0.streams[args.recon_emb]
        tok, entry, geom_for = load_frozen(run_e19, cache, args.recon_emb, dev,
                                           st["meta"]["reach"])

    if "recon" in do:
        results["recon"] = eval_recon(model, g0, args.recon_emb, val_pos[:args.n_recon],
                                      dev, tok, entry, geom_for,
                                      args.cache or meta["cache"], f, K)
        print("RECON", json.dumps(results["recon"], indent=1), flush=True)

    if "mp4" in do:
        results["mp4"] = eval_mp4(model, g0, args.recon_emb, dev, tok, entry, geom_for,
                                  args.cache or meta["cache"], f, K,
                                  f"{out_dir}/mp4", n_clips=args.viz_clips)

    if "duration" in do:
        results["duration"] = eval_duration(model, g0, g0.embs()[0], dev)
        print("DURATION", json.dumps(results["duration"], indent=1), flush=True)

    json.dump(results, open(f"{out_dir}/results.json", "w"), indent=2)
    print(f"SAVED {out_dir}/results.json ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
