"""VISUAL VERDICT for the 0812 arms — same motion, four codecs, side by side.

    fk_duel_video.py --fam punch

Numbers said rot_only at half the bits is not worse and the FK loss moves arms
by ~2.5%. Those are small deltas, so they have to be looked at: a 2.5% change
that is invisible is a different fact from one you can see.
Panel 1 is the human source (lloyd's rule: every video carries its reference).
"""
import os, sys, json, argparse
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES",
                      "/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
os.environ.setdefault("V5_AMASS", "/var/tmp/alphamotion/topo_exp/cache_xwbc_amass_full")
import v5_common as V, eval_newbody as E, manifold_project as MP
import render_newbody as RN, greenwich_carnival as GC
from dual_common import load_paired_data, split_indices, build_model, geom, fk_pos_torch
from bridge_a import FULL
from train_global import build_global
from spatial5 import capsule_tile, pad_band, font
from common import rot6d_to_matrix
from PIL import Image, ImageDraw

RUNS, dev = "/var/tmp/alphamotion/topo_exp/runs", "cuda"
VIZ = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "R4_VIZ")
MODELS = [("PROD_dim16_13k_0809", "BASELINE  256 slots (128 wasted)"),
          ("FK1_rot128_0812",      "rot-only 128  HALF the bits"),
          ("FK2_rot256_0812",      "rot-only 256  all bits on rotation"),
          ("SMALL_dim16_reenc_0812", "re-encode consistency")]
ap = argparse.ArgumentParser()
ap.add_argument("--fam", default="punch")
ap.add_argument("--T", type=int, default=90)
ap.add_argument("--target", default="unitree_g1_29dof")
ap.add_argument("--seed", type=int, default=3)
args = ap.parse_args()

cfg0 = json.load(open(f"{RUNS}/{MODELS[0][0]}/config.json"))
torch.manual_seed(0)
data, names, N = load_paired_data(cfg0["cache"], dev, sem_encoder=cfg0["sem_encoder"])
val_idx, _ = split_indices(N)
kd = V.KeyData(V.AMASS, ["human_smpl"], device="cpu")
g = np.random.default_rng(args.seed)
s, _ = kd.sample_starts(1, g, "val", span=args.T + 2, families=[args.fam])
r0 = int(s[0]); rows = torch.arange(r0, r0 + args.T)
src = data["human_smpl"]["pose"][rows.to(dev)]
tgt = args.target
spec_t = data[tgt]["spec"]
GMR = "/var/tmp/alphamotion/config_harvest/GMR/assets"   # same table greenwich_carnival uses
XML = {"unitree_g1_29dof": f"{GMR}/unitree_g1/g1_mocap_29dof.xml",
       "fourier_gr3": f"{GMR}/fourier_gr3v2_1_1/mjcf/gr3v2_1_1.xml",
       "booster_t1": f"{GMR}/booster_t1/T1_locomotion.xml"}
spec_x, dof_x, rest_x, qn_x, xml_x = E.get_target(XML[tgt], E.CACHE_AMASS)
g_x = E.target_geom(spec_x, dof_x, dev, sem_encoder=cfg0["sem_encoder"])
dof_d = torch.as_tensor(dof_x, device=dev, dtype=torch.float64)
rest_d = torch.as_tensor(rest_x, device=dev, dtype=torch.float64)

# human reference
sspec = V.load_spec(FULL, "human_smpl")
l6h = torch.load(f"{FULL}/human_smpl_paired_local6d.pt", mmap=True)
g6, _, _ = build_global(l6h[rows], sspec, dev)
srct = capsule_tile(V.fk_pos(g6, sspec), sspec.parents, f"SOURCE human ({args.fam})")

tiles, labs, errs = [srct], ["SOURCE human"], [None]
with torch.no_grad():
    gtr = data[tgt]["pose"][rows.to(dev)][..., :6]
    fg = fk_pos_torch(gtr, spec_t)
    nl = [x.lower() for x in spec_t.joint_names]
    am = torch.tensor([any(k in x for k in ("shoulder","elbow","wrist","arm","hand"))
                       for x in nl], device=dev)
    for mn, lab in MODELS:
        cfg = json.load(open(f"{RUNS}/{mn}/config.json"))
        m = build_model(cfg, dev)
        m.load_state_dict(torch.load(f"{RUNS}/{mn}/model.pt", map_location=dev))
        m.eval()
        zp, zr = m.encode(src, geom(data, "human_smpl", args.T))
        prot, _ = m.decode(zp, zr, geom(data, tgt, args.T))
        fp = fk_pos_torch(prot, spec_t)
        e = (fp - fg).norm(dim=-1)[:, am].mean()
        Rg = rot6d_to_matrix(prot).double()
        _, _, qj = MP.project(Rg, spec_x, dof_d, rest=rest_d, method="global")
        gR, gp = MP.fk_from_angles(qj, spec_x, dof_d, rest=rest_d, root_R=Rg[:, 0])
        tiles.append(GC.render_tile(xml_x, spec_x, qn_x, qj.cpu().numpy(),
                                    gR[:, 0].cpu().numpy(), gp.cpu().numpy(),
                                    qj.shape[0], status="nodataset"))
        labs.append(lab); errs.append(float(e))
        print(f"[{mn}] arm err {float(e):.2f} cm", flush=True)

L = max(len(t) for t in tiles)
frames = []
for t in range(L):
    row = np.concatenate([tiles[0][min(t, len(tiles[0]) - 1)]] +
                         [pad_band(ti[min(t, len(ti) - 1)]) for ti in tiles[1:]], 1)
    im = Image.fromarray(row).convert("RGB")
    dr = ImageDraw.Draw(im)
    for k, lb in enumerate(labs):
        dr.text((k * GC.TILE_W + 8, 6), lb, font=font(11), fill=(255, 235, 170))
        if errs[k] is not None:
            dr.text((k * GC.TILE_W + 8, 24), f"arm err {errs[k]:.2f} cm",
                    font=font(11, False), fill=(150, 230, 255))
    dr.text((8, GC.TILE_H + 24),
            f"same {args.fam} clip -> {tgt}, FK positions from predicted "
            "rotations. all four share one recipe; only the code layout differs",
            font=font(12, False), fill=(210, 210, 215))
    frames.append(np.asarray(im))
p = f"{VIZ}/mp4/bridge_demo/fk_duel_{args.fam}_{tgt}.mp4"
RN.write_mp4(p, np.stack(frames), 20)
print("->", p, flush=True)
