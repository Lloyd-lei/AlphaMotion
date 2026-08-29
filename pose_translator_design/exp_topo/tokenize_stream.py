"""E13 step 1 — freeze a tokenizer run and encode CONTIGUOUS 30 fps clips.

Unlike tokenize_cache.py (which encodes a paired cache row-for-row and keeps the
cache's own row order), this packs whole clips back-to-back and records exact clip
boundaries, so a temporal model can slice T-frame windows that never straddle two
clips. That is the only thing E13 needs that tokenize_cache.py does not give:
tokenize_cache emits [N,2,K,f] in cache order and copies sequences.json, but the
xwbc caches are already contiguous per clip, so what is actually missing is
(a) a clip subset selection, (b) a packed uint8 array small enough to hold in RAM
for random window sampling, and (c) a frozen `reach` recorded in metadata so a
later partial re-encode reproduces the same codes bit-for-bit.

Output layout under --out:
  codes.npy    uint8  [N, D]     D = 2*S*K*f integer FSQ levels, pos stream then rot
  bounds.npy   int64  [C+1]      packed clip boundaries; clip c = [bounds[c], bounds[c+1])
  rows.npy     int64  [N]        row index back into the source cache (for GT pose)
  clips.json   [{name, start, len}]  start/len in PACKED coordinates
  meta.json    run/cache/emb/K/f/stages/levels/reach/fps/bits + dequantization constants

Dequantization (stages == 1): q = (code - offset) / half, offset = ceil((L-1)/2),
half = (L-1)/2. Those are exactly FSQ.from_ints's grid values, so the continuous
pre-round feature is fully recoverable from the integers up to rounding -- there is
no reason to also store fp16 z, and not storing it saves ~2x disk.

Usage (needs the GPU; run it after the queue sentinel, see run_e13_after_seeds.sh):
  python tokenize_stream.py --run .../runs/E19_K128_0805 \
      --cache .../cache_xwbc_amass --emb human_smpl --out .../e13_stream/E19_amass_human
"""
from __future__ import annotations
import os, sys, json, argparse, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SkeletonSpec                      # noqa: E402
from train_ae import static_cond                     # noqa: E402
from train_global import build_global                # noqa: E402
from dual_common import build_model                  # noqa: E402
from semantics import build_name_embeddings, embed_matrix  # noqa: E402

CACHE_ROOT = "/var/tmp/alphamotion/topo_exp"
DEFAULT_RUN = f"{CACHE_ROOT}/runs/E19_K128_0805"
DEFAULT_CACHE = f"{CACHE_ROOT}/cache_xwbc_amass"
DEFAULT_OUT = f"{CACHE_ROOT}/e13_stream/E19_K128_amass_human"


# --------------------------------------------------------------------------- model / body
def load_tokenizer(run, device):
    cfg = json.load(open(f"{run}/config.json"))
    model = build_model(cfg, device)
    sd = torch.load(f"{run}/model.pt", map_location=device)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"WARN state_dict mismatch: missing={missing} unexpected={unexpected}")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, cfg


def load_body(cache, name, device, rows=None, reach=None):
    """Same recipe as dual_common.load_paired_data but for ONE embodiment and an
    optional row subset. `reach` pins the position normalizer: pass the value from
    meta.json when re-encoding a subset, or the codes will not match the cache."""
    d = np.load(f"{cache}/{name}_spec.npz", allow_pickle=True)
    spec = SkeletonSpec(str(d["name"]), d["parents"], d["rest_offsets"], list(d["joint_names"]))
    local6d = torch.load(f"{cache}/{name}_paired_local6d.pt")
    if rows is not None:
        local6d = local6d[torch.as_tensor(np.asarray(rows), dtype=torch.long)]
    gr6, gp, reach_here = build_global(local6d, spec, device)
    reach = reach_here if reach is None else float(reach)
    import v5_common as _V                       # 0809: match the frozen run's text tower
    lut = build_name_embeddings(list(spec.joint_names), device=device,
                                encoder=_V.frozen_sem_encoder())
    entry = {"spec": spec, "st": static_cond(spec, device),
             "sem": embed_matrix(spec.joint_names, lut).to(device), "reach": reach}
    dof_path = f"{cache}/{name}_dof.npz"
    if os.path.exists(dof_path):
        entry["dof"] = torch.as_tensor(np.load(dof_path)["dof"], dtype=torch.float32, device=device)
    return entry, torch.cat([gr6, gp / reach], -1)          # CPU [N,J,9]


def geom_for(entry, B):
    st = entry["st"]
    g = dict(rest_off=st["rest_off"][None].expand(B, -1, -1),
             bonelen=st["bonelen"][None].expand(B, -1),
             depth=st["depth"][None].expand(B, -1),
             hop=st["hop"][None].expand(B, -1, -1),
             sem=entry["sem"][None].expand(B, -1, -1))
    if "dof" in entry:
        g["dof"] = entry["dof"][None].expand(B, -1, -1)
    return g


@torch.no_grad()
def encode_ints(model, entry, pose_chunk, device):
    """[b,J,9] pose -> uint8 [b, D] packed integer levels (pos stream then rot stream)."""
    x = pose_chunk.to(device)
    fp, fr = model.encode_features(x, geom_for(entry, x.shape[0]))
    ip, ir = model.ints_from_features(fp, fr)               # [S,b,K,f] each
    ip = ip.permute(1, 0, 2, 3).reshape(x.shape[0], -1)     # [b, S*K*f]
    ir = ir.permute(1, 0, 2, 3).reshape(x.shape[0], -1)
    return torch.cat([ip, ir], -1).to(torch.uint8).cpu()


# --------------------------------------------------------------------------- stream metadata
def stream_meta(model, cfg, run, cache, emb, reach, n_frames, n_clips, fps):
    fsq = model.fsqp
    levels = [int(l) for l in fsq._levels.tolist()]
    S = int(getattr(fsq, "n_stages", 1))
    K = int(cfg.get("n_latent", 32))
    f = len(levels)
    half = [(l - 1) / 2 for l in levels]
    return {
        "run": run, "cache": cache, "emb": emb, "reach": float(reach),
        "n_frames": int(n_frames), "n_clips": int(n_clips), "fps": float(fps),
        "K": K, "f": f, "stages": S, "n_streams": 2, "levels": levels,
        "single_codebook": bool(cfg.get("single_codebook", False)),
        "D": 2 * S * K * f,
        # dequantization: q = (code - offset) / half   (per feature dim, tiled over D)
        "deq_half": half, "deq_offset": [float(np.ceil(h)) for h in half],
        "bits_per_dim_nominal": [float(np.log2(l)) for l in levels],
        "bits_per_frame_nominal": float(2 * S * K * sum(np.log2(l) for l in levels)),
    }


def deq_vectors(meta):
    """Per-D-dim (offset, half) float32 vectors so codes -> q is one vectorized op."""
    D, f = meta["D"], meta["f"]
    off = np.tile(np.asarray(meta["deq_offset"], np.float32), D // f)
    half = np.tile(np.asarray(meta["deq_half"], np.float32), D // f)
    return off, half


def load_stream(path, mmap=False):
    """-> dict(codes uint8 [N,D], bounds int64 [C+1], rows, clips, meta, off, half)."""
    meta = json.load(open(f"{path}/meta.json"))
    codes = np.load(f"{path}/codes.npy", mmap_mode="r" if mmap else None)
    off, half = deq_vectors(meta)
    return {"codes": codes, "bounds": np.load(f"{path}/bounds.npy"),
            "rows": np.load(f"{path}/rows.npy"), "clips": json.load(open(f"{path}/clips.json")),
            "meta": meta, "off": off, "half": half, "path": path}


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=DEFAULT_RUN)
    ap.add_argument("--cache", default=DEFAULT_CACHE)
    ap.add_argument("--emb", default="human_smpl")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--max_clips", type=int, default=0, help="0 = all clips in the cache")
    ap.add_argument("--min_len", type=int, default=32, help="drop clips shorter than this")
    ap.add_argument("--seed", type=int, default=0, help="rng for the clip subset (max_clips>0)")
    ap.add_argument("--bs", type=int, default=2048)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    t0 = time.time()
    os.makedirs(args.out, exist_ok=True)
    seqs = json.load(open(f"{args.cache}/sequences.json"))
    man = json.load(open(f"{args.cache}/manifest.json"))
    fps = float(man.get("fps", 30.0))

    keep = [i for i, s in enumerate(seqs) if s["len"] >= args.min_len]
    if args.max_clips and args.max_clips < len(keep):
        sel = np.sort(np.random.default_rng(args.seed).choice(keep, args.max_clips, replace=False))
    else:
        sel = np.asarray(keep)
    lens = np.asarray([seqs[i]["len"] for i in sel], np.int64)
    bounds = np.concatenate([[0], np.cumsum(lens)])
    rows = np.concatenate([np.arange(seqs[i]["start"], seqs[i]["start"] + seqs[i]["len"])
                           for i in sel]).astype(np.int64)
    N = int(bounds[-1])
    print(f"clips {len(sel)}/{len(seqs)} (>= {args.min_len} frames), frames {N}, fps {fps}")

    model, cfg = load_tokenizer(args.run, args.device)
    # reach is computed over the FULL embodiment (not the subset) so the codes are
    # identical to what a full-cache encode would give -- the subset must not move
    # the position normalizer.
    entry, pose = load_body(args.cache, args.emb, args.device)
    print(f"[{args.emb}] J={entry['spec'].J} rows={pose.shape[0]} reach={entry['reach']:.1f}cm "
          f"({time.time()-t0:.0f}s)")
    pose = pose[torch.as_tensor(rows)]

    probe = encode_ints(model, entry, pose[:2], args.device)
    D = probe.shape[1]
    meta = stream_meta(model, cfg, args.run, args.cache, args.emb, entry["reach"], N, len(sel), fps)
    assert D == meta["D"], f"packed D={D} != meta D={meta['D']}"

    codes = np.lib.format.open_memmap(f"{args.out}/codes.npy", mode="w+", dtype=np.uint8, shape=(N, D))
    for i in range(0, N, args.bs):
        j = min(i + args.bs, N)
        codes[i:j] = encode_ints(model, entry, pose[i:j], args.device).numpy()
        if (i // args.bs) % 50 == 0:
            print(f"  {j}/{N}  ({time.time()-t0:.0f}s)", flush=True)
    codes.flush()

    np.save(f"{args.out}/bounds.npy", bounds)
    np.save(f"{args.out}/rows.npy", rows)
    json.dump([{"name": seqs[i]["name"], "start": int(bounds[c]), "len": int(lens[c])}
               for c, i in enumerate(sel)], open(f"{args.out}/clips.json", "w"))
    lo, hi = int(codes[:8192].min()), int(codes[:8192].max())
    meta["observed_code_range"] = [lo, hi]
    json.dump(meta, open(f"{args.out}/meta.json", "w"), indent=2)
    print(f"SAVED {args.out}  codes{codes.shape} uint8 range[{lo},{hi}]  "
          f"{codes.nbytes/1e9:.2f} GB  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
