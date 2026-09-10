"""按家族的 FK 准确率 + 接触准确率总表 (lloyd 0901)。
对每臂: human->TGT 解码, 逐家族报 pos_cm (FK 准确率) 与接触三口径。"""
import os, sys, json, re
import numpy as np, torch
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
from dual_common import build_model, geom, fk_pos_torch
from common import rot6d_to_matrix
from contact_label import family, sole_offsets
import v2_judge as VJ
RUNS="/var/tmp/alphamotion/topo_exp/runs"
CACHE=sys.argv[1]        # lafan|amass
TGT=sys.argv[2]
ARMS=[(a.split("=")[0], a.split("=")[1] if "/" in a else f"{RUNS}/{a.split('=')[1]}") for a in sys.argv[3:]]
C={"lafan":"/var/tmp/alphamotion/topo_exp/cache_xwbc_lafan","amass":"/var/tmp/alphamotion/topo_exp/cache_xwbc_amass_full"}[CACHE]
LBL={"lafan":"/var/tmp/alphamotion/topo_exp/cache_lafan_v3","amass":"/home/arenalabs/tmp/cl_v3/amass"}[CACHE]
dev="cuda"; torch.set_grad_enabled(False)
seqs=json.load(open(f"{C}/sequences.json"))
# 每家族抽最多 4 个片段 x 120 帧
import collections
fam2=collections.defaultdict(list)
for q in seqs:
    if q["len"]>=140: fam2[family(q["name"])].append(q)
rows_l=[]; fams=[]
rng=np.random.default_rng(3)
for f,qs in sorted(fam2.items()):
    take=[qs[i] for i in rng.permutation(len(qs))[:4]]
    for q in take:
        s0=q["start"]+10; rows_l.append(np.arange(s0,s0+120)); fams+= [f]*120
rows=torch.as_tensor(np.concatenate(rows_l)); fams=np.array(fams)
cfg0=json.load(open(f"{ARMS[0][1]}/config.json"))
from semantics import build_name_embeddings
SRC="human_smpl"
names_all=[]
import numpy as _np
for n in [SRC,TGT]:
    d=_np.load(f"{C}/{n}_spec.npz",allow_pickle=True); names_all+=[str(x) for x in d["joint_names"]]
lut=build_name_embeddings(names_all,device=dev,encoder=cfg0.get("sem_encoder","siglip"))
import types
_L=VJ.LAFAN; VJ.LAFAN=C   # load_emb 里写死了 LAFAN 路径, 临时指到本语料
data={n:VJ.load_emb(n,rows,[C],cfg0,lut,dev) for n in [SRC,TGT]}
VJ.LAFAN=_L
spec=data[TGT]["spec"]
z=np.load(f"{LBL}/{TGT}_contact.npz",allow_pickle=True); hdr=json.loads(str(z["header"])); thr=hdr["thr_m"]
ri=rows.numpy(); lab=(z["contact"][ri]|(z["gap"][ri]<0))
root=int(np.where(np.asarray(spec.parents)<0)[0][0])
gr=torch.from_numpy(z["gap"][ri][:,root].astype(np.float32)).to(dev)
tru=z["trust"][ri] if "trust" in z.files else np.ones(len(ri),bool)
soff=sole_offsets(TGT,spec)
zs=np.load(f"{LBL}/{SRC}_contact.npz",allow_pickle=True); hs=json.loads(str(zs["header"]))
s_=hs["thr_m"]-zs["gap"][ri].astype(np.float32)
sd_src=torch.as_tensor((np.sign(s_)*np.log1p(np.abs(s_)/hs["thr_m"])/np.log1p(hs["D_used_m"]/hs["thr_m"])).astype(np.float32),device=dev)
def height(r6):
    h=fk_pos_torch(r6,spec)[...,1]/100.0+gr[:,None]
    if soff:
        js=sorted(soff); R=rot6d_to_matrix(r6[:,js])
        off=torch.stack([torch.as_tensor(soff[j],dtype=torch.float32,device=dev) for j in js])
        h=h.clone(); h[:,js]=h[:,js]+torch.einsum("bkxy,ky->bkx",R,off)[...,1]
    return h
gt6=data[TGT]["pose"][...,:6]
streams={"GT":gt6}
for tag,ck in ARMS:
    cfg=json.load(open(f"{ck}/config.json"))
    m=build_model(cfg,dev); m.load_state_dict(torch.load(f"{ck}/model.pt",map_location=dev)); m.eval()
    B=len(rows); con=sd_src if cfg.get("contact_chan") else None
    streams[tag],_=m.decode(*m.encode(data[SRC]["pose"],geom(data,SRC,B),con=con),geom(data,TGT,B)); del m
L=torch.from_numpy(lab).to(dev); TR=torch.from_numpy(tru).to(dev)
pg=fk_pos_torch(gt6,spec)
print("语料=%s 目标=%s  (pos_cm=FK 全身误差; touch=该踩|h| 中位/p95 cm; 只算 trust 帧)"%(CACHE,TGT))
hdr_pr="%-8s"%"家族"+"".join("%-26s"%t for t in streams)
print(hdr_pr)
for f in sorted(set(fams)):
    m_=torch.from_numpy(fams==f).to(dev)&TR
    if not bool(m_.any()): continue
    line="%-8s"%f[:8]
    for tag,r6 in streams.items():
        pos=float((fk_pos_torch(r6,spec)[m_]-pg[m_]).norm(dim=-1).mean()) if tag!="GT" else 0.0
        h=height(r6); Lm=L[m_]
        hc=h[m_][Lm].abs()*100
        t50=float(hc.median()) if hc.numel() else float("nan")
        t95=float(hc.quantile(.95)) if hc.numel() else float("nan")
        line+="%-26s"%("pos%5.1f 触%5.1f/%5.1f"%(pos,t50,t95))
    print(line)
