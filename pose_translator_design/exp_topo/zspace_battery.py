"""Greenwich-as-B offline fitness battery (T1-T4). Encoder-side only, frozen production Greenwich.
Feature = pre-quantization continuous: proj_out(tanh(encode_features)) mean-pooled over the
128 latent tokens, pose-stream (256d) ++ root-stream (256d) = 512d per frame."""
import sys, os, json, re, numpy as np, torch
sys.path.insert(0, "/home/arenalabs/Desktop/AlphaMotion/src")
torch.set_grad_enabled(False)
torch.manual_seed(0); np.random.seed(0)
from alphamotion.engine.greenwich import Greenwich
from alphamotion.engine.spatial import SkeletonSpec, build_global, fk_global
from alphamotion.engine.descriptor import geom_batch

C = "/var/tmp/alphamotion/topo_exp/cache_xwbc_lafan/"
OUT = os.path.dirname(os.path.abspath(__file__))
FPS = 30
G = Greenwich.load()
DEV = G.device
print("Greenwich loaded on", DEV, flush=True)

def load_spec(name):
    d = np.load(C+name+"_spec.npz", allow_pickle=True)
    return SkeletonSpec(str(d["name"]), d["parents"], d["rest_offsets"], [str(n) for n in d["joint_names"]])

seqs = json.load(open(C+"sequences.json"))
N_ROWS = 496593
clip_of = np.full(N_ROWS, -1, dtype=np.int32)
for i, s in enumerate(seqs):
    clip_of[s["start"]:s["start"]+s["len"]] = i
LYING = [i for i, s in enumerate(seqs) if re.search(r"fall|ground|getUp", s["name"], re.I)]
print("lying clips", len(LYING), flush=True)

class Emb:
    def __init__(self, name):
        self.name = name
        self.spec = load_spec(name)
        self.dof = np.load(C+name+"_dof.npz", allow_pickle=True)["dof"]
        self.geom = G.geom(self.spec, self.dof)
        self.L = torch.load(C+name+"_paired_local6d.pt", mmap=True)
        # full-data reach, matching training's build_global-over-all-rows max
        mx = 0.0
        for i in range(0, N_ROWS, 8192):
            b = self.L[i:i+8192].to(DEV).float()
            _, gp = fk_global(b, self.spec)
            mx = max(mx, float(gp.norm(dim=-1).max()))
        self.reach = max(mx, 1e-6)
        print(f"[{name}] J={self.spec.J} reach={self.reach:.1f}cm", flush=True)

    def encode(self, idx, return_gp=False, bs=2048):
        """idx: 1-D LongTensor of row indices -> feat [N,512] fp16 cpu (raw, unnormalized)."""
        idx = torch.as_tensor(idx, dtype=torch.long)
        feats = []; gps = []
        for i in range(0, len(idx), bs):
            l6 = self.L[idx[i:i+bs]].float()
            gR, gp = fk_global(l6.to(DEV), self.spec)
            from alphamotion.engine.nets.rotations import matrix_to_rot6d
            gr6 = matrix_to_rot6d(gR)
            pose9 = torch.cat([gr6, gp/self.reach], -1)
            fp, fr = G.model.encode_features(pose9, geom_batch(self.geom, pose9.shape[0]))
            zp = G.model.fsqp.proj_out(torch.tanh(fp)).mean(1)
            zr = G.model.fsqr.proj_out(torch.tanh(fr)).mean(1)
            feats.append(torch.cat([zp, zr], -1).half().cpu())
            if return_gp:
                gps.append(gp.half().cpu())
        f = torch.cat(feats, 0)
        return (f, torch.cat(gps, 0)) if return_gp else f

def cosn(x):
    x = x.float()
    return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-8)

R = {}
g1 = Emb("unitree_g1_29dof")

# ---------------- T1 lying-pose coverage ----------------
ly_idx = np.concatenate([np.arange(seqs[i]["start"], seqs[i]["start"]+seqs[i]["len"], 5) for i in LYING])
other_rows = np.where(~np.isin(clip_of, LYING) & (clip_of >= 0))[0]
st_idx = np.sort(np.random.choice(other_rows, 2000, replace=False))
all_idx = np.concatenate([ly_idx, st_idx])
is_ly = np.zeros(len(all_idx), bool); is_ly[:len(ly_idx)] = True
clips1 = clip_of[all_idx]
F1 = cosn(g1.encode(all_idx)).to(DEV)
# (a) within-lying mean pairwise cosine (subsample 3000) & lying-vs-standing
sub = np.random.choice(len(ly_idx), min(3000, len(ly_idx)), replace=False)
A = F1[sub]
S = A @ A.T
n = len(sub)
within = float((S.sum() - S.diag().sum()) / (n*n - n))
B = F1[len(ly_idx):]
cross = float((A @ B.T).mean())
# (b) NN classification, exclude self-clip
hits = 0
cl_t = torch.as_tensor(clips1, device=DEV)
for i in range(0, len(ly_idx), 1024):
    q = F1[i:i+1024]
    sim = q @ F1.T                                    # [b, M]
    same_clip = cl_t[None, :] == cl_t[i:i+1024, None]
    sim.masked_fill_(same_clip, -2.0)
    nn = sim.argmax(1).cpu().numpy()
    hits += int(is_ly[nn].sum())
nn_rate = hits / len(ly_idx)
R["T1"] = {"n_lying_frames": int(len(ly_idx)), "n_standing": 2000,
           "within_lying_cos": within, "lying_standing_cos": cross,
           "margin": within - cross, "nn_lying_hit_rate": nn_rate,
           "pass": bool(within < 0.995 and (within - cross) > 0.05 and nn_rate > 0.70)}
print("T1", R["T1"], flush=True)
del F1, A, B, S
torch.cuda.empty_cache()

# ---------------- T2 phrase retrieval ----------------
W = 12
valid_starts = []
for i, s in enumerate(seqs):
    valid_starts.append(np.arange(s["start"], s["start"]+s["len"]-W))
valid_starts = np.concatenate(valid_starts)
cand_st = np.random.choice(valid_starts, 5000, replace=False)
qry_st = np.random.choice(valid_starts, 500, replace=False)
def phrase_feats(starts):
    rows = (starts[:, None] + np.arange(W)[None, :]).reshape(-1)
    f = g1.encode(rows).float().view(len(starts), W, -1).mean(1)
    return cosn(f).to(DEV)
FC = phrase_feats(cand_st); FQ = phrase_feats(qry_st)
c_clip = torch.as_tensor(clip_of[cand_st], device=DEV)
q_clip = torch.as_tensor(clip_of[qry_st], device=DEV)
c_st = torch.as_tensor(cand_st, device=DEV); q_st = torch.as_tensor(qry_st, device=DEV)
sim = FQ @ FC.T
near = (q_clip[:, None] == c_clip[None, :]) & ((q_st[:, None] - c_st[None, :]).abs() <= FPS)
sim.masked_fill_(near, -2.0)
nn = sim.argmax(1)
same = (c_clip[nn] == q_clip)
top1 = float(same.float().mean())
ret = c_clip[nn].cpu().numpy()
p = np.bincount(ret, minlength=77).astype(float); p /= p.sum()
ent = float(-(p[p > 0] * np.log2(p[p > 0])).sum())
R["T2"] = {"n_query": 500, "n_cand": 5000, "top1_same_clip": top1,
           "retrieved_clip_entropy_bits": ent, "max_entropy_bits": float(np.log2(77)),
           "chance": 1/77, "pass": bool(top1 >= 0.40)}
print("T2", R["T2"], flush=True)
del FC, FQ, sim
torch.cuda.empty_cache()

# ---------------- T3 smoothness ----------------
WIN = 10 * FPS
big = [i for i, s in enumerate(seqs) if s["len"] > WIN + 2]
picks = np.random.choice(big, 20, replace=False)
fv_all, jv_all = [], []
for ci in picks:
    s = seqs[ci]
    st = s["start"] + np.random.randint(0, s["len"] - WIN)
    f, gp = g1.encode(np.arange(st, st+WIN), return_gp=True)
    f = f.float(); gp = gp.float()
    fv_all.append((f[1:] - f[:-1]).norm(dim=-1).numpy())
    jv_all.append((gp[1:] - gp[:-1]).norm(dim=-1).mean(-1).numpy())
fv = np.concatenate(fv_all); jv = np.concatenate(jv_all)
fm, jm = np.median(fv), np.median(jv)
spur = (fv > 5*fm) & (jv < 2*jm)
R["T3"] = {"n_frames": int(len(fv)), "feat_vel_median": float(fm), "feat_vel_p99": float(np.percentile(fv,99)),
           "joint_vel_median_cm": float(jm), "spurious_jumps": int(spur.sum()),
           "spurious_frac": float(spur.mean()), "pass": bool(spur.mean() < 0.005)}
print("T3", R["T3"], flush=True)

# ---------------- T4 dialect ceiling ----------------
rows4 = np.sort(np.random.choice(N_ROWS, 5000, replace=False))
hum = Emb("human_smpl")
FH = cosn(hum.encode(rows4)).to(DEV)
R["T4"] = {"n_rows": 5000}
for name in ["unitree_g1_29dof", "pnd_adam_lite"]:
    e = g1 if name == "unitree_g1_29dof" else Emb(name)
    FE = cosn(e.encode(rows4)).to(DEV)
    cs = (FH * FE).sum(-1).cpu().numpy()
    R["T4"][name] = {"median": float(np.median(cs)), "p10": float(np.percentile(cs, 10)),
                     "p90": float(np.percentile(cs, 90)),
                     "pass_ceiling": bool(np.median(cs) >= 0.55)}
    print("T4", name, R["T4"][name], flush=True)

json.dump(R, open(os.path.join(OUT, "battery_results.json"), "w"), indent=1)
print("DONE ->", os.path.join(OUT, "battery_results.json"))
