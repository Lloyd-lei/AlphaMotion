"""AlphaMotion text layer, step 1: build (caption, motion-window) pairs.

    text_pairs_prep.py

Sources:
  HumanML3D texts (scratchpad clone) mapped onto our full-cache clip names via
  index.csv path normalization; mirrored (M*) entries skipped (no mirrored motion
  in our cache).
  bones-seed: 500 cached clips x manifest captions (captions + short + technical),
  split 80/20 by clip (20% = alignment holdout).
Caption vectors: frozen Qwen3-Embedding via semantics._encode_qwen3, cached.
Bones motion codes: soma_body frames through FROZEN Greenwich (C1) — read-only.

Out (/var/tmp/alphamotion/topo_exp/text_align/):
  pairs_train.jsonl / pairs_val_amass.jsonl / pairs_val_bones.jsonl
  caption_emb.pt {caption: [1024] fp16}
  bones_codes.npy [Nrows,256,20] int8 + bones_spans.json
"""
from __future__ import annotations
import os, sys, json, csv
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("V5_AMASS", "/var/tmp/alphamotion/topo_exp/cache_xwbc_amass_full")
import v5_common as V
import eval_newbody as E
import eval_bones_body_e19 as EB
from semantics import _encode_qwen3, build_name_embeddings
from common import CACHE_ROOT

FULL = "/var/tmp/alphamotion/topo_exp/cache_xwbc_amass_full"
BONES = f"{CACHE_ROOT}/cache_bones_body"
HML = ("/tmp/claude-1000/-home-arenalabs-Desktop--be-water--robot-/"
       "eff459e5-3969-4f6d-9ec8-eae021b0a07a/scratchpad/hml3d")
OUT = "/var/tmp/alphamotion/topo_exp/text_align"


def amass_pairs():
    seqs = json.load(open(f"{FULL}/sequences.json"))
    by_name = {s["name"]: i for i, s in enumerate(seqs)}
    rng = np.random.default_rng(V.SPLIT_SEED)
    val = set(rng.permutation(len(seqs))[: len(seqs) // 10].tolist())
    pairs_tr, pairs_va = [], []
    n_map = 0
    with open(f"{HML}/index.csv") as f:
        for row in csv.DictReader(f):
            sp = row["source_path"]
            if "pose_data/" not in sp or "humanact12" in sp:
                continue
            parts = sp.split("pose_data/")[1].replace("_poses.npy", "").split("/")
            if len(parts) < 3:
                continue
            name = "__".join([parts[0], "/".join(parts[1:-1]).replace("/", "_"),
                              parts[-1]]).replace(" ", "_") + "_stageii"
            if name not in by_name:
                continue
            tid = row["new_name"].replace(".npy", "")
            tf = f"{HML}/HumanML3D/texts/{tid}.txt"
            if not os.path.exists(tf):
                continue
            caps = []
            for line in open(tf, errors="ignore"):
                c = line.split("#")[0].strip()
                if len(c) > 8:
                    caps.append(c)
            if not caps:
                continue
            n_map += 1
            i = by_name[name]
            s = seqs[i]
            rec = [{"src": "amass", "name": name, "start": s["start"],
                    "len": s["len"], "caption": c} for c in caps[:4]]
            (pairs_va if i in val else pairs_tr).extend(rec)
    print(f"[hml3d] mapped clips {n_map}  train pairs {len(pairs_tr)} "
          f"val pairs {len(pairs_va)}", flush=True)
    return pairs_tr, pairs_va


def bones_pairs():
    seqs = json.load(open(f"{BONES}/sequences.json"))
    by_name = {s["name"]: s for s in seqs}
    man = {}
    for l in open(f"{BONES}/text_manifest.jsonl"):
        d = json.loads(l)
        if d["name"] in by_name:
            caps = list(d.get("captions", []))[:3]
            if d.get("short"):
                caps.append(d["short"])
            man[d["name"]] = caps
    names = sorted(man)
    rng = np.random.default_rng(7)
    hold = set(rng.permutation(len(names))[: len(names) // 5].tolist())
    tr, va = [], []
    for i, n in enumerate(names):
        s = by_name[n]
        rec = [{"src": "bones", "name": n, "start": s["start"], "len": s["len"],
                "caption": c} for c in man[n] if len(c) > 3]
        (va if i in hold else tr).extend(rec)
    print(f"[bones] clips {len(names)} train pairs {len(tr)} holdout {len(va)}",
          flush=True)
    return tr, va


@torch.no_grad()
def bones_codes(dev="cuda"):
    sp, spcfg = E.load_run(f"{V.RUNS}/PROD_c1_91k_dim20_tail_0810", dev)
    enc = spcfg.get("sem_encoder")
    man = json.load(open(f"{BONES}/manifest.json"))
    spec_b = EB.load_spec(BONES, "soma_body")
    lut = build_name_embeddings(list(spec_b.joint_names), device=dev, encoder=enc)
    entry = EB.entry(BONES, "soma_body", spec_b, dev, lut)
    from dual_common import geom
    g9 = torch.load(f"{BONES}/soma_body_global9d.pt")
    pose = torch.cat([g9[..., :6], g9[..., 6:9] / man["reach_cm"]], -1)
    N = pose.shape[0]
    out = np.zeros((N, 256, 20), np.int8)
    for i in range(0, N, 512):
        b = pose[i:i + 512].to(dev).float()
        fp, fr = sp.encode_features(b, geom({"soma_body": entry}, "soma_body",
                                            b.shape[0]))
        out[i:i + 512, :128] = sp.fsqp.ints(fp)[0].cpu().numpy().astype(np.int8)
        out[i:i + 512, 128:] = sp.fsqr.ints(fr)[0].cpu().numpy().astype(np.int8)
    np.save(f"{OUT}/bones_codes.npy", out)
    print(f"[bones codes] {out.shape} -> {OUT}/bones_codes.npy", flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    tr_a, va_a = amass_pairs()
    tr_b, va_b = bones_pairs()
    json.dump(tr_a + tr_b, open(f"{OUT}/pairs_train.json", "w"))
    json.dump(va_a, open(f"{OUT}/pairs_val_amass.json", "w"))
    json.dump(va_b, open(f"{OUT}/pairs_val_bones.json", "w"))
    caps = sorted({p["caption"] for p in tr_a + tr_b + va_a + va_b})
    print(f"unique captions {len(caps)}", flush=True)
    feats = _encode_qwen3(caps, "cuda", batch=64)
    feats = torch.nn.functional.normalize(feats.float(), dim=-1).half()
    torch.save({c: feats[i] for i, c in enumerate(caps)},
               f"{OUT}/caption_emb.pt")
    print(f"caption embeddings -> {OUT}/caption_emb.pt", flush=True)
    bones_codes()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
