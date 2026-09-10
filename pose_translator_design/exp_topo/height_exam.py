"""方言对照考 (0901): 同一模型, 同一批 AMASS 片段, 源人体分别用 raw AMASS 骨架 vs
LAFAN 方言改写后的骨架, 报 FK 误差 + 根高误差 (模型根槽 vs 标签 gap_root)。
用法: height_exam.py <ckpt> <tgt> raw=<cache>:<lbl> conv=<cache>:<lbl> ..."""
import os, sys, json, collections
import numpy as np, torch
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
from dual_common import build_model, geom, fk_pos_torch
from contact_label import family
import v2_judge as VJ
RUNS="/var/tmp/alphamotion/topo_exp/runs"
CK=sys.argv[1] if sys.argv[1].startswith("/") else f"{RUNS}/{sys.argv[1]}"; TGT=sys.argv[2]
SETS=[(a.split("=")[0],a.split("=")[1].split(":")) for a in sys.argv[3:]]
dev="cuda"; torch.set_grad_enabled(False)
C0=SETS[0][1][0]; seqs=json.load(open(f"{C0}/sequences.json"))
fam2=collections.defaultdict(list)
for q in seqs:
    if q["len"]>=140: fam2[family(q["name"])].append(q)
rows_l=[]; fams=[]; rng=np.random.default_rng(3)
for f,qs in sorted(fam2.items()):
    for q in [qs[i] for i in rng.permutation(len(qs))[:6]]:
        s0=q["start"]+10; rows_l.append(np.arange(s0,s0+120)); fams+=[f]*120
rows=torch.as_tensor(np.concatenate(rows_l)); fams=np.array(fams); ri=rows.numpy()
cfg=json.load(open(f"{CK}/config.json"))
m=build_model(cfg,dev); m.load_state_dict(torch.load(f"{CK}/model.pt",map_location=dev)); m.eval()
from semantics import build_name_embeddings
SRC="human_smpl"
res={}
for tag,(C,LBL) in SETS:
    names_all=[]
    for n in [SRC,TGT]:
        d=np.load(f"{C}/{n}_spec.npz",allow_pickle=True); names_all+=[str(x) for x in d["joint_names"]]
    lut=build_name_embeddings(names_all,device=dev,encoder=cfg.get("sem_encoder","siglip"))
    _L=VJ.LAFAN; VJ.LAFAN=C
    data={n:VJ.load_emb(n,rows,[C],cfg,lut,dev) for n in [SRC,TGT]}; VJ.LAFAN=_L
    spec=data[TGT]["spec"]; root=int(np.where(np.asarray(spec.parents)<0)[0][0])
    z=np.load(f"{LBL}/{TGT}_contact.npz",allow_pickle=True); gr=z["gap"][ri][:,root].astype(np.float32)
    tru=z["trust"][ri] if "trust" in z.files else np.ones(len(ri),bool)
    zs=np.load(f"{LBL}/{SRC}_contact.npz",allow_pickle=True); hs=json.loads(str(zs["header"]))
    s_=hs["thr_m"]-zs["gap"][ri].astype(np.float32)
    sd=torch.as_tensor((np.sign(s_)*np.log1p(np.abs(s_)/hs["thr_m"])/np.log1p(hs["D_used_m"]/hs["thr_m"])).astype(np.float32),device=dev)
    B=len(rows); con=sd if cfg.get("contact_chan") else None
    out=[]; hp=[]
    for i in range(0,B,4096):
        sl=slice(i,i+4096); n=sl.stop-sl.start if sl.stop<=B else B-i
        r6,_=m.decode(*m.encode(data[SRC]["pose"][sl],geom(data,SRC,n),con=None if con is None else con[sl]),geom(data,TGT,n))
        out.append(r6); hp.append(m.last_pos[:,root,0].clone())
    r6=torch.cat(out); hp=torch.cat(hp).cpu().numpy()
    gt6=data[TGT]["pose"][...,:6]
    pos=(fk_pos_torch(r6,spec)-fk_pos_torch(gt6,spec)).norm(dim=-1).mean(-1).cpu().numpy()
    herr=(hp-gr)*100
    res[tag]=dict(pos=pos,herr=herr,tru=tru)
print(f"模型={os.path.basename(CK)} 目标={TGT}  每格: FK误差cm | 根高误差 中位/p95 cm (有符号中位)")
print("%-9s"%"家族"+"".join("%-34s"%t for t,_ in SETS))
for f in ["ALL"]+sorted(set(fams)):
    msk=(np.ones(len(fams),bool) if f=="ALL" else fams==f)
    line="%-9s"%f[:9]
    for tag,_ in SETS:
        r=res[tag]; mm=msk&r["tru"]; he=r["herr"][mm]
        line+="%-34s"%("pos%5.1f | h %5.1f/%5.1f (%+6.1f)"%(r["pos"][mm].mean(),np.median(np.abs(he)),np.quantile(np.abs(he),.95),np.median(he)))
    print(line)
