"""Render GT vs reconstructed poses with per-joint ORIENTATION TRIADS so twist is
visible (twist doesn't move joints, so stick figures alone can't show it).
Saves side-by-side PNG frames + a GIF. Paths printed for the user to eyeball.
"""
from __future__ import annotations
import os, sys, glob, json
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from common import SkeletonSpec, rot6d_to_matrix, matrix_to_rot6d, fk_global, spec_from_bvh_skeleton
from model_dual import DualTranslator
from train_ae import static_cond
from train_global import build_global
from semantics import build_name_embeddings, embed_matrix

OUT = os.path.join(os.path.dirname(__file__), "..", "viz")  # in the project dir (Desktop), user-visible
BONES = '/home/arenalabs/Desktop/"be water, robot"/alpha_motion2/wbc_expert/data/raw/bones_seed_full'


def load_seq_soma(n_frames=8):
    from ardy.data_processing.skeleton import load_bvh_animation, Skeleton
    f = sorted(glob.glob(f"{BONES}/soma_uniform/bvh/210531/*.bvh"))[0]
    skl = Skeleton(); skl.load_from_bvh(f)
    spec = spec_from_bvh_skeleton(skl, "soma77"); spec.order_root_first()
    rt, jr = load_bvh_animation(f, skl)
    jr = torch.as_tensor(np.asarray(jr), dtype=torch.float32)
    idx = np.linspace(0, jr.shape[0] - 1, n_frames).round().astype(int)
    return spec, matrix_to_rot6d(jr[idx]), os.path.basename(f)


def draw(ax, pos, R, parents, title, lim):
    ax.set_title(title, fontsize=9)
    axis_len = 0.06 * lim   # triad length scales with skeleton size
    for j, p in enumerate(parents):
        if p >= 0:
            ax.plot([pos[j,0],pos[p,0]],[pos[j,1],pos[p,1]],[pos[j,2],pos[p,2]], c="#888", lw=1.2)
    ax.scatter(pos[:,0], pos[:,1], pos[:,2], c="#222", s=5)
    for j in range(pos.shape[0]):
        o = pos[j]
        for k, col in enumerate(["r","g","b"]):
            d = R[j,:,k] * axis_len
            ax.plot([o[0],o[0]+d[0]],[o[1],o[1]+d[1]],[o[2],o[2]+d[2]], c=col, lw=0.9)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.set_zlim(-lim,lim)
    ax.view_init(elev=8, azim=-70)      # side-front view so a standing figure is visible
    try: ax.set_box_aspect((1,1,1))
    except Exception: pass


def main():
    os.makedirs(OUT, exist_ok=True)
    device = "cuda"
    run = "/var/tmp/alphamotion/topo_exp/runs/FIX_translate"
    model = DualTranslator(n_latent=32, fsq_stages=2).to(device)
    model.load_state_dict(torch.load(f"{run}/model.pt", map_location=device)); model.eval()

    spec, local6d, fname = load_seq_soma(8)
    st = static_cond(spec, device)
    gr6, gp, reach = build_global(local6d, spec, device)
    pose = torch.cat([gr6, gp / reach], -1).to(device)
    lut = build_name_embeddings(list(spec.joint_names), device=device)
    sem = embed_matrix(spec.joint_names, lut).to(device)
    B = pose.shape[0]
    g = dict(rest_off=st["rest_off"][None].expand(B,-1,-1), bonelen=st["bonelen"][None].expand(B,-1),
             depth=st["depth"][None].expand(B,-1), hop=st["hop"][None].expand(B,-1,-1),
             sem=sem[None].expand(B,-1,-1))
    with torch.no_grad():
        zp, zr = model.encode(pose, g); prot, ppos = model.decode(zp, zr, g)
    gt_R = rot6d_to_matrix(gr6).cpu().numpy(); gt_p = gp.cpu().numpy()
    pr_R = rot6d_to_matrix(prot).cpu().numpy(); pr_p = (ppos * reach).cpu().numpy()
    parents = spec.parents
    lim = float(np.abs(gt_p).max()) * 1.05     # auto-scale axes to the skeleton
    frames = []
    for i in range(B):
        fig = plt.figure(figsize=(8,4))
        a1 = fig.add_subplot(121, projection="3d"); draw(a1, gt_p[i], gt_R[i], parents, f"SOMA GT frame{i}", lim)
        a2 = fig.add_subplot(122, projection="3d"); draw(a2, pr_p[i], pr_R[i], parents, "reconstruction", lim)
        fig.suptitle("triads=joint orientation; twist shows as axes rotating about a bone", fontsize=8)
        p = f"{OUT}/soma_recon_{i:02d}.png"; fig.savefig(p, dpi=90, bbox_inches="tight"); plt.close(fig)
        frames.append(p)
    # assemble gif
    try:
        import imageio.v2 as imageio
        imgs = [imageio.imread(p) for p in frames]
        imageio.mimsave(f"{OUT}/soma_recon.gif", imgs, duration=0.6)
        print("GIF:", f"{OUT}/soma_recon.gif")
    except Exception as e:
        print("gif skip:", repr(e)[:80])
    print("PNG frames:", OUT, "(soma_recon_00..07.png)")
    print("source motion:", fname)


if __name__ == "__main__":
    main()
