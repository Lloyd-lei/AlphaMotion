"""Batch side of lloyd's data engine: every GENMO output -> one AlphaMotion cache.

    genmo_batch_ingest.py --roots <GENMO>/outputs --out /var/tmp/alphamotion/genmo_cache

Walks for smpl_params.pt, converts each with the same local->global composition
genmo_ingest.py uses, and writes ONE concatenated tensor plus a sequences.json
index, i.e. the same shape as the paired caches the rest of the stack already
reads.

Two filters, because a data engine that ingests garbage is worse than no engine:
  * geometry gate — FK the result and require a plausible standing human
  * duplicate gate — GENMO's investor_demo re-runs the same source clip under
    many masking variants.  A first-frame hash does NOT catch them (each
    reconstruction differs slightly, so the first pass deduped 0 of 79 while
    66 clips were variants of just 12 materials, some differing by 0.003 mean
    rot6d).  Grouping is by the material name in the path, keeping the longest
    member; without this the corpus would carry those 12 motions at 6x weight.
"""
from __future__ import annotations
import os, sys, json, glob, hashlib, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("V5_AMASS", "/var/tmp/alphamotion/topo_exp/cache_xwbc_amass_full")
import v5_common as V                                            # noqa: E402
from genmo_ingest import aa_to_R, SMPL_PARENTS                    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--out", default="/var/tmp/alphamotion/genmo_cache")
    ap.add_argument("--space", default="global")
    ap.add_argument("--min_frames", type=int, default=30)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    sp = np.load(f"{os.environ['V5_AMASS']}/human_smpl_spec.npz", allow_pickle=True)
    names = [str(x) for x in sp["joint_names"]]

    class S:
        parents = sp["parents"]; rest_offsets = sp["rest_offsets"]
    i_head, i_ank = names.index("Head"), names.index("L_Ankle")
    i_sh, i_el = names.index("L_Shoulder"), names.index("L_Elbow")

    files = sorted({p for r in args.roots
                    for p in glob.glob(f"{r}/**/smpl_params.pt", recursive=True)})
    print(f"找到 {len(files)} 个 GENMO 产出", flush=True)
    seen, keep, rows, seqs, at = {}, {}, [], [], 0
    root_rows = []
    n_dup = n_bad = 0
    for f in files:
        try:
            d = torch.load(f, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"  跳过(读不了) {f}: {e}", flush=True); n_bad += 1; continue
        key = f"body_params_{args.space}"
        if key not in d:
            n_bad += 1; continue
        p = d[key]
        go = p["global_orient"].float().numpy().reshape(-1, 1, 3)
        bp = p["body_pose"].float().numpy().reshape(len(go), 21, 3)
        aa = np.concatenate([go, bp], 1)
        if len(aa) < args.min_frames:
            n_bad += 1; continue
        material = os.path.basename(os.path.dirname(f))
        prev = seen.get(material)
        if prev is not None and prev[0] >= len(aa):
            n_dup += 1; continue
        Rl = aa_to_R(aa)
        Rg = np.zeros_like(Rl)
        for j, par in enumerate(SMPL_PARENTS):
            Rg[:, j] = Rl[:, j] if par < 0 else Rg[:, par] @ Rl[:, j]
        r6 = Rg[..., :, :2].transpose(0, 1, 3, 2).reshape(len(aa), 22, 6)
        pj = V.fk_pos(r6, S)
        ht = float(np.mean(pj[:, i_head, 1] - pj[:, i_ank, 1]))
        ua = float(np.linalg.norm(pj[:, i_el] - pj[:, i_sh], axis=-1).mean())
        if not (90 < ht < 200 and 15 < ua < 40):
            print(f"  几何不过关 头-脚{ht:.0f} 上臂{ua:.0f}: {f}", flush=True)
            n_bad += 1; continue
        tr = p.get("transl", torch.zeros(len(aa), 3)).float().numpy() * 100.0
        if prev is not None:                 # longer member replaces the shorter
            n_dup += 1
            keep.pop(prev[1], None)
        seen[material] = (len(aa), f)
        keep[f] = (torch.from_numpy(r6).float(), torch.from_numpy(tr).float(),
                   material, round(ht, 1))
    if not keep:
        raise SystemExit("没有任何片段通过")
    for f, (r6, tr, material, ht) in keep.items():
        rows.append(r6); root_rows.append(tr)
        seqs.append({"name": material, "start": at, "len": int(len(r6)),
                     "src": f, "height_cm": ht})
        at += len(r6)
    R = torch.cat(rows); RW = torch.cat(root_rows)
    torch.save(R, f"{args.out}/human_smpl_global6d.pt")
    torch.save(RW, f"{args.out}/human_smpl_root_world.pt")
    json.dump(seqs, open(f"{args.out}/sequences.json", "w"), indent=1)
    json.dump({"rows": int(at), "clips": len(seqs), "fps": 30,
               "units": "cm", "rot": "GLOBAL rot6d (already composed)",
               "src": "GENMO", "space": args.space,
               "dropped_duplicate": n_dup, "dropped_bad": n_bad},
              open(f"{args.out}/manifest.json", "w"), indent=1)
    mins = at / 30.0 / 60.0
    print(f"\n入库 {len(seqs)} 段 / {at} 帧 = {mins:.1f} 分钟 "
          f"(去重扔掉 {n_dup}, 几何/格式扔掉 {n_bad})", flush=True)
    print(f"   语料对照: AMASS 缓存 4,733,703 帧 = {4733703/30/60:.0f} 分钟", flush=True)
    print(f"-> {args.out}", flush=True)


if __name__ == "__main__":
    main()
