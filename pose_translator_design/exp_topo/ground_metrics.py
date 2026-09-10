"""接地质量的多角度评判 (lloyd 0901: "该有别的角度来判断接地是否最好")。
在全部 24 个地面窗上算 (不是抽 2 个片段), 每臂四个互补口径:
  touch_cm   标签说该踩的关节, 解码后离地 |高度| 的中位/p95 (连续, 不设阈值门槛)
  pen_cm     标签说没踩的关节, 穿到地下深度的 p95 (深浅比个数诚实)
  iou        站点级接触段与标签的 IoU (时序对不对, 不只逐帧)
  tuck       四肢末端到根的平均距离 / GT 同值 (lloyd 肉眼发现的"收拢趋势", <1=收拢)
用法: ground_metrics.py tag=ckpt [tag=ckpt ...]"""
import os, sys, json
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from dual_common import build_model, geom, fk_pos_torch
from common import rot6d_to_matrix
import v2_judge as VJ
import re
RUNS = "/var/tmp/alphamotion/topo_exp/runs"
TGT = "unitree_g1_29dof"
ARMS = [(a.split("=")[0], a.split("=")[1] if "/" in a else f"{RUNS}/{a.split('=')[1]}")
        for a in sys.argv[1:]]

dev = "cuda"
torch.set_grad_enabled(False)
win = json.load(open(VJ.WIN_FILE_GROUND)); T = win["T"]
rows = torch.cat([torch.arange(w["start"], w["start"] + T) for w in win["windows"]])
cfg0 = json.load(open(f"{ARMS[0][1]}/config.json"))
caches = [c for c in str(cfg0["cache"]).split(",") if c]
from semantics import build_name_embeddings
SRC = VJ.SRC
names_all = []
for n in [SRC, TGT]:
    d = np.load(f"{caches[0]}/{n}_spec.npz", allow_pickle=True)
    names_all += [str(x) for x in d["joint_names"]]
lut = build_name_embeddings(names_all, device=dev, encoder=cfg0.get("sem_encoder", "siglip"))
data = {n: VJ.load_emb(n, rows, caches, cfg0, lut, dev) for n in [SRC, TGT]}
spec = data[TGT]["spec"]

z = np.load(f"{caches[0]}/{TGT}_contact.npz", allow_pickle=True)
hdr = json.loads(str(z["header"])); thr = hdr["thr_m"]
ri = rows.numpy()
lab = (z["contact"][ri] | (z["gap"][ri] < 0))
root = int(np.where(np.asarray(spec.parents) < 0)[0][0])
gr = torch.from_numpy(z["gap"][ri][:, root].astype(np.float32)).to(dev)
sid = np.asarray(hdr["site_id"]); ns = int(hdr["n_sites"])
# 脚底偏移 (与损失同一把尺子)
from contact_label import sole_offsets
soff = sole_offsets(TGT, spec)
# 末端关节 (tuck 用): 每个站点里离根最远的关节近似末端
depth = np.zeros(spec.J); 
for j in range(spec.J):
    k=j; d_=0
    while spec.parents[k]>=0: k=spec.parents[k]; d_+=1
    depth[j]=d_
ee = [int(np.flatnonzero(sid==s)[np.argmax(depth[sid==s])]) for s in range(ns) if (sid==s).any()]

def height(r6):
    p = fk_pos_torch(r6, spec)
    h = p[..., 1] / 100.0 + gr[:, None]
    if soff:
        js = sorted(soff)
        R = rot6d_to_matrix(r6[:, js])
        off = torch.stack([torch.as_tensor(soff[j], dtype=torch.float32, device=dev) for j in js])
        h[:, js] = h[:, js] + torch.einsum("bkxy,ky->bkx", R, off)[..., 1]
    return h, p

def sites(c):
    out = np.zeros((c.shape[0], ns), bool)
    for s in range(ns):
        m = sid == s
        if m.any(): out[:, s] = c[:, m].any(1)
    return out

gt6 = data[TGT]["pose"][..., :6]
h_gt, p_gt = height(gt6)
tuck_gt = p_gt[:, ee].norm(dim=-1).mean().item()
streams = {"GT": gt6}
for tag, ck in ARMS:
    cfg = json.load(open(f"{ck}/config.json"))
    m = build_model(cfg, dev); m.load_state_dict(torch.load(f"{ck}/model.pt", map_location=dev)); m.eval()
    B = len(rows)
    con = data[SRC].get("sd") if cfg.get("contact_chan") else None
    streams[tag], _ = m.decode(*m.encode(data[SRC]["pose"], geom(data, SRC, B), con=con),
                               geom(data, TGT, B))
    del m
L = torch.from_numpy(lab).to(dev)
Lsite = sites(lab)
print("%-10s %12s %12s %10s %8s %8s" % ("臂", "该踩|h|中位cm", "该踩|h|p95", "穿深p95cm", "IoU", "tuck"))
for tag, r6 in streams.items():
    h, p = height(r6)
    hc = h[L].abs() * 100
    pen = torch.relu(-h[~L]) * 100
    pred = sites((h < thr).cpu().numpy())
    inter = (pred & Lsite).sum(); union = (pred | Lsite).sum()
    tuck = p[:, ee].norm(dim=-1).mean().item() / tuck_gt
    print("%-10s %12.2f %12.2f %10.2f %8.3f %8.3f" % (
        tag, hc.median(), hc.quantile(.95), pen.quantile(.95), inter / max(union, 1), tuck))
