"""Is the punch render static because the MODEL is static, or because the
RENDER cannot show motion? Put the paired GT robot next to the decode.

    punch_probe_video.py

Tiles: human source | GT robot (its own paired rotations, no model involved) |
Greenwich C1 decode | the dim16 baseline. If the GT tile also looks still, the
answer is the renderer (in-place, no world translation). If GT clearly moves and
the decodes do not, the answer is that clip's decode.
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
from dual_common import load_paired_data, build_model, geom, fk_pos_torch
from bridge_a import FULL
from train_global import build_global
from spatial5 import capsule_tile, pad_band, font
from common import rot6d_to_matrix
from PIL import Image, ImageDraw

RUNS, dev = "/var/tmp/alphamotion/topo_exp/runs", "cuda"
VIZ = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "R4_VIZ")
GMR = "/var/tmp/alphamotion/config_harvest/GMR/assets"
ap = argparse.ArgumentParser()
ap.add_argument("--fam", default="punch"); ap.add_argument("--T", type=int, default=90)
ap.add_argument("--seed", type=int, default=3)
args = ap.parse_args()
tgt = "unitree_g1_29dof"
cfg = json.load(open(f"{RUNS}/PROD_c1_91k_dim20_tail_0810/config.json"))
torch.manual_seed(0)
data, names, N = load_paired_data(cfg["cache"], dev, sem_encoder=cfg["sem_encoder"])
kd = V.KeyData(V.AMASS, ["human_smpl"], device="cpu")
g = np.random.default_rng(args.seed)
s, _ = kd.sample_starts(1, g, "val", span=args.T + 2, families=[args.fam])
rows = torch.arange(int(s[0]), int(s[0]) + args.T)
spec = data[tgt]["spec"]
spec_x, dof_x, rest_x, qn_x, xml_x = E.get_target(
    f"{GMR}/unitree_g1/g1_mocap_29dof.xml", E.CACHE_AMASS)
dof_d = torch.as_tensor(dof_x, device=dev, dtype=torch.float64)
rest_d = torch.as_tensor(rest_x, device=dev, dtype=torch.float64)

def render_rot(prot):
    Rg = rot6d_to_matrix(prot).double()
    _, _, qj = MP.project(Rg, spec_x, dof_d, rest=rest_d, method="global")
    gR, gp = MP.fk_from_angles(qj, spec_x, dof_d, rest=rest_d, root_R=Rg[:, 0])
    return GC.render_tile(xml_x, spec_x, qn_x, qj.cpu().numpy(),
                          gR[:, 0].cpu().numpy(), gp.cpu().numpy(),
                          qj.shape[0], status="nodataset")

sspec = V.load_spec(FULL, "human_smpl")
l6h = torch.load(f"{FULL}/human_smpl_paired_local6d.pt", mmap=True)
g6, _, _ = build_global(l6h[rows], sspec, dev)
tiles = [capsule_tile(V.fk_pos(g6, sspec), sspec.parents, "SOURCE human")]
labs, notes = ["SOURCE human"], [""]

gt_rot = data[tgt]["pose"][rows.to(dev)][..., :6]
fg = fk_pos_torch(gt_rot, spec)
gt_amp = float(fg.std(0).norm(dim=-1).mean())
tiles.append(render_rot(gt_rot)); labs.append("GT ROBOT (paired, no model)")
notes.append(f"amp {gt_amp:.2f} cm")
for mn, lab in (("PROD_c1_91k_dim20_tail_0810", "GREENWICH C1"),
                ("PROD_dim16_13k_0809", "dim16 baseline")):
    c = json.load(open(f"{RUNS}/{mn}/config.json"))
    m = build_model(c, dev); m.load_state_dict(torch.load(f"{RUNS}/{mn}/model.pt", map_location=dev)); m.eval()
    with torch.no_grad():
        prot, _ = m.decode(*m.encode(data["human_smpl"]["pose"][rows.to(dev)],
                                     geom(data, "human_smpl", args.T)),
                           geom(data, tgt, args.T))
        fp = fk_pos_torch(prot, spec)
        amp = float(fp.std(0).norm(dim=-1).mean())
        err = float((fp - fg).norm(dim=-1).mean())
    tiles.append(render_rot(prot)); labs.append(lab)
    notes.append(f"amp {amp:.2f} ({100*amp/gt_amp:.0f}% of GT)  err {err:.1f} cm")
    print(f"[{mn}] amp {amp:.2f} vs GT {gt_amp:.2f} ({100*amp/gt_amp:.0f}%)  err {err:.1f}", flush=True)
    del m; torch.cuda.empty_cache()

L = max(len(t) for t in tiles)
frames = []
for t in range(L):
    row = np.concatenate([tiles[0][min(t, len(tiles[0])-1)]] +
                         [pad_band(ti[min(t, len(ti)-1)]) for ti in tiles[1:]], 1)
    im = Image.fromarray(row).convert("RGB"); dr = ImageDraw.Draw(im)
    for k, lb in enumerate(labs):
        dr.text((k*GC.TILE_W+8, 6), lb, font=font(12), fill=(255, 235, 170))
        if notes[k]:
            dr.text((k*GC.TILE_W+8, 24), notes[k], font=font(11, False), fill=(150, 230, 255))
    dr.text((8, GC.TILE_H+24),
            f"same {args.fam} window. tile 2 is the paired GROUND TRUTH robot with no "
            "model in the loop — if it also looks still, the renderer is the reason",
            font=font(12, False), fill=(210, 210, 215))
    frames.append(np.asarray(im))
p = f"{VIZ}/mp4/bridge_demo/punch_probe_gt_vs_greenwich.mp4"
RN.write_mp4(p, np.stack(frames), 20)
print("->", p, flush=True)
