"""QA video for a production output (no CPU reference): 1920x1080 @ 30 fps, [human SMPL-X surface + object | robot
dense surface points (strict output)], floor grid at z = 0 (calibrated floor).
Colours: robot points within [-0.5, 1] mm of the floor green, of the object orange, penetrating > 0.5 mm red;
human vertices within 1 cm of the floor green / of the object orange.  Text: frame, flags (EXCLUDED), lowest
point, contact point count, max penetration ground / object.  Also writes a 12-frame grid PNG.
Usage: render_prod.py <production npz> [out_dir]
"""
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, '/home/arenalabs/umr_gpu'); sys.path.insert(0, '/home/arenalabs/umr_gpu/eval')
from judge import floor_model
from render_compare import cam, draw_points, floor_grid, W, H, FONT

UMR = '/var/tmp/alphamotion/umr_factory/umr/UMR'


def main():
    npz = Path(sys.argv[1]); outd = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('/var/tmp/alphamotion/umr_factory/umr_gpu/videos/production')
    outd.mkdir(parents=True, exist_ok=True)
    z = np.load(npz, allow_pickle=True)
    prov = json.loads(str(z['provenance']))
    robot = prov['robot']
    Q = z['qpos'].astype(np.float64); flags = z['flags']; T = len(Q)
    # robot points
    m, _, gmap = floor_model(prov['robot_xml'])
    d = mujoco.MjData(m)
    s = np.load(prov['surface_asset'])
    pl = s['points_local'].astype(np.float64); pg = gmap[s['geom'].astype(np.int64)]

    def rpts(q):
        d.qpos[:] = q; mujoco.mj_kinematics(m, d)
        R = d.geom_xmat[pg].reshape(-1, 3, 3)
        return np.einsum('nij,nj->ni', R, pl) + d.geom_xpos[pg]
    # human vertices (same scale / up / floor as the solver)
    sys.path.insert(0, UMR + '/scripts'); sys.path.insert(0, UMR)
    import smpl_surface_retarget_common as common
    src = Path(prov['source_dir'])
    if (src / 'poses.npy').exists():
        poses = np.load(src / 'poses.npy').reshape(-1, 165); tr = np.load(src / ('transl.npy' if (src / 'transl.npy').exists() else 'trans.npy'))
        betas = np.load(src / 'betas.npy').reshape(-1)[:10]; up = str(np.load(src / 'output_up.npy')) if (src / 'output_up.npy').exists() else 'z'
    else:
        zz = np.load(sorted(src.glob('*.npz'))[0], allow_pickle=True)
        poses = zz['poses'].reshape(len(zz['poses']), -1); tr = zz['trans']; betas = zz['betas'].reshape(-1)[:10]; up = str(zz['output_up'])
    fids = z['frame_ids_in_source']
    model = common.build_smplx_model(Path(UMR) / 'smpl', 'neutral', len(fids), device='cpu')
    with torch.no_grad():
        pa = torch.as_tensor(poses[fids], dtype=torch.float32); n = len(fids)
        b = torch.zeros(n, 10); b[:, :len(betas)] = torch.as_tensor(betas, dtype=torch.float32)
        zh = torch.zeros(n, 45); z3 = torch.zeros(n, 3)
        V = model(global_orient=pa[:, :3], body_pose=pa[:, 3:66], betas=b, transl=torch.as_tensor(tr[fids], dtype=torch.float32),
                  left_hand_pose=zh, right_hand_pose=zh, jaw_pose=z3, leye_pose=z3, reye_pose=z3,
                  expression=torch.zeros(n, getattr(model, 'num_expression_coeffs', 10))).vertices.numpy()
    if up == 'y':
        V = V @ np.asarray(common.Y_UP_TO_Z_UP_MATRIX, np.float64).T
    V = V * float(z['human_scale']); V[..., 2] -= float(z['floor_offset_removed'])
    V = V[:, ::2]
    # object
    obj = None
    if 'object_pose' in z.files:
        from surface_retargeting.gpu.hoi import ObjectAsset
        asset = ObjectAsset(str(z['object_xml']), 'cpu')
        sc = float(z['object_scale'])
        P0 = asset.points.numpy() * sc
        op = z['object_pose']
        sdf = asset.sdf

        def obj_world(t):
            Rm = Rot.from_quat([op[t, 4], op[t, 5], op[t, 6], op[t, 3]]).as_matrix()
            return P0 @ Rm.T + op[t, :3]

        def obj_dist(X, t):
            pose = torch.as_tensor(np.concatenate([op[t], [sc]])[None], dtype=torch.float32)
            dd, _ = sdf.sdf(torch.as_tensor(X[None], dtype=torch.float32), pose)
            return dd[0].numpy()
        obj = (obj_world, obj_dist)
    C = cam()
    font = ImageFont.truetype(FONT, 24); small = ImageFont.truetype(FONT, 19)
    W2 = 960

    def frame(t):
        img = Image.new('RGB', (2 * W2, H), (252, 252, 252))
        X = rpts(Q[t])
        center = np.array([Q[t, 0], Q[t, 1], 0.0])
        for k in range(2):
            sub = Image.new('RGB', (W, H), (252, 252, 252)); sd = ImageDraw.Draw(sub)
            scale = 500.0; y0 = H - 130
            floor_grid(sd, C, center, scale, y0)
            if obj is not None:
                Po = obj[0](t)
                sub = draw_points(sub, Po, np.tile(np.array([90, 120, 220], np.uint8), (len(Po), 1)), C, center, scale, y0)
            if k == 0:
                P = V[t]; col = np.tile(np.array([170, 150, 140], np.uint8), (len(P), 1))
                col[P[:, 2] < 0.01] = [20, 170, 40]
                if obj is not None:
                    col[obj[1](P, t) < 0.01] = [255, 140, 0]
                lines = ['human SMPL-X (green: < 1 cm to floor, orange: < 1 cm to object)']
            else:
                P = X; col = np.tile(np.array([150, 150, 160], np.uint8), (len(P), 1))
                zz_ = P[:, 2]
                col[(zz_ >= -0.0005) & (zz_ <= 0.001)] = [20, 170, 40]
                pen_o = 0.0
                if obj is not None:
                    dd = obj[1](P, t)
                    col[(dd >= -0.0005) & (dd <= 0.001)] = [255, 140, 0]
                    col[dd < -0.0005] = [230, 20, 20]; pen_o = max(0.0, -dd.min() * 1000)
                col[zz_ < -0.0005] = [230, 20, 20]
                lines = [f'{robot}  strict output', f'lowest point {zz_.min() * 1000:+.2f} mm   ground contact pts {int(((zz_ >= -0.0005) & (zz_ <= 0.001)).sum())}',
                         f'max pen ground {max(0.0, -zz_.min() * 1000):.2f} mm   object {pen_o:.2f} mm',
                         f'flags {int(flags[t])}' + ('  (EXCLUDED)' if int(flags[t]) & 255 else '') + ('  256 follow human slip' if int(flags[t]) & 256 else '')]
            sub = draw_points(sub, P, col, C, center, scale, y0)
            sd = ImageDraw.Draw(sub)
            for i, ln in enumerate(lines):
                sd.text((14, 14 + 30 * i), ln, fill=(160, 0, 0) if 'EXCLUDED' in ln else (20, 20, 20), font=small if i else font)
            img.paste(sub, (k * W2 + (W2 - W) // 2, 0))
        dr = ImageDraw.Draw(img)
        dr.line([(W2, 0), (W2, H)], fill=(120, 120, 120), width=2)
        dr.text((14, H - 40), f"{prov['source']}/{prov['clip_id']} -> {robot}   frame {t}/{T - 1} @ 30 fps   floor z=0 (calibrated)",
                fill=(0, 0, 0), font=small)
        return img

    name = f"{prov['source']}__{prov['clip_id']}__{robot}"
    w = imageio.get_writer(str(outd / f'{name}.mp4'), fps=30, codec='libx264', quality=5, macro_block_size=8)
    for t in range(T):
        w.append_data(np.asarray(frame(t)))
    w.close()
    fr = np.linspace(0, T - 1, 12).astype(int)
    sheet = Image.new('RGB', (4 * 480, 3 * 270), (255, 255, 255))
    for i, t in enumerate(fr):
        sheet.paste(frame(t).resize((480, 270)), ((i % 4) * 480, (i // 4) * 270))
    sheet.save(outd / f'{name}_grid12.png')
    print('saved', outd / f'{name}.mp4')


if __name__ == '__main__':
    main()
