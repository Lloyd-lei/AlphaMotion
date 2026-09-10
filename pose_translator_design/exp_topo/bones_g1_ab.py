"""bones-seed (语料外人体动作) -> G1 零样本对照 (lloyd 0901):
    SOURCE | Y5b (只训 LAFAN, 干净) | C1 Greenwich (LAFAN + 有时间轴 bug 的 xwbc AMASS)
两模型同一段人体动作, 同一管线 (encode -> decode G1 -> 可行性投影 -> MuJoCo 渲染)。
    bones_g1_ab.py <out_dir> <clip_prefix> [clip_prefix ...]
"""
import os, sys, json, re
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import spatial5 as S
from dual_common import geom
RUNS = os.environ.get("AM_RUNS", "Y5b_heightonly_0901,PROD_c1_91k_dim20_tail_0810").split(",")
LABELS = os.environ.get("AM_LABELS", "Y5b (LAFAN only, clean)|C1 Greenwich (LAFAN+buggy AMASS)").split("|")
OUT = sys.argv[1]; PATS = sys.argv[2:]; os.makedirs(OUT, exist_ok=True)
dev = "cuda"; torch.set_grad_enabled(False)
G1 = S.FOUR[0]
man = json.load(open(f"{S.BONES}/manifest.json"))
spec_b = S.EB.load_spec(S.BONES, "soma_body")
g9 = torch.load(f"{S.BONES}/soma_body_global9d.pt"); reach = man["reach_cm"]
pose_all = torch.cat([g9[..., :6], g9[..., 6:9] / reach], -1)
seqs = json.load(open(f"{S.BONES}/sequences.json"))
picks = []
for p in PATS:
    m = [q for q in seqs if q["name"].startswith(p)]
    assert m, p
    picks.append(m[0]); print("[pick]", m[0]["name"], m[0]["len"])
ctxs = []
for r in RUNS:
    c = S.Ctx(r, dev)
    lut = S.build_name_embeddings(list(spec_b.joint_names), device=dev, encoder=c.enc)
    e = S.EB.entry(S.BONES, "soma_body", spec_b, dev, lut); e["reach"] = reach
    c.data = {"soma_body": e}; ctxs.append(c)
for s in picks:
    T = min(s["len"], 150); rows = torch.arange(s["start"], s["start"] + T)
    x = pose_all[rows].to(dev).float()
    streams = []
    for c, lab in zip(ctxs, LABELS):
        zp, zr = c.model.encode(x, geom(c.data, "soma_body", T))
        t = c.target(G1[0], G1[1])
        streams.append((f"G1 <- {lab}", "heldout", t["xml"], t["spec"], t["qnames"], c.decode_to(zp, zr, t)))
    src = S.capsule_tile(g9[rows][..., 6:9].numpy(), spec_b.parents, "SOURCE (bones-seed human)")
    p = f"{OUT}/bonesg1_{re.sub('[^A-Za-z0-9]+','_',s['name'])[:40]}.mp4"
    S.grid_video(ctxs[0], streams, p, f"bones-seed -> G1 zero-shot | {s['name'][:50]}", src_tile=src)
    print("WROTE", p)
