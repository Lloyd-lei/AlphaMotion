"""12-frame grid render of a converted GRAIL clip: G1-SMPL-X human mesh (blue) + object visual mesh (orange)
+ optional table (grey) + floor plane z=0 (checker), MuJoCo offscreen (EGL). Side camera at low elevation so
floating/sinking is visible; per-panel text: lowest foot gap to support, lowest body point, hand-object distance.
usage: render_grid.py <cat> <key> <out.png> [--src converted|raw] [--dz MM]"""
import argparse, json, os, sys
os.environ.setdefault('MUJOCO_GL', 'egl')
import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, '/var/tmp/alphamotion/umr_factory/umr/UMR/scripts')
from humanoid_retarget_pipeline_hsi_hoi import load_grail_pickle, safe_name  # noqa: E402
from pathlib import Path  # noqa: E402
import trimesh  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from g1smplx import G1SMPLX  # noqa: E402
from floorcal import foot_tracks, support_height, TERRAIN  # noqa: E402

OUT = Path('/home/arenalabs/umr_factory_data/smplx/grail')
RAW = Path('/home/arenalabs/umr_factory_data/raw/grail/data')


def load_obj(p):
    m = trimesh.load(p, force='mesh', process=False)
    return np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64)


def build_scene(hv, hf, ov, of, table, floor_c, hand_pts=None):
    s = mujoco.MjSpec()
    s.visual.global_.offwidth = 640
    s.visual.global_.offheight = 480
    s.visual.quality.shadowsize = 2048
    tex = s.add_texture(name='grid', type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                        rgb1=[0.82, 0.82, 0.82], rgb2=[0.62, 0.62, 0.62], width=512, height=512)
    mat = s.add_material(name='grid', texrepeat=[8, 8], reflectance=0.0)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = 'grid'
    s.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.05], pos=[floor_c[0], floor_c[1], 0], material='grid')
    s.worldbody.add_light(pos=[floor_c[0], floor_c[1] - 1, 4], dir=[0, 0.3, -1], diffuse=[0.7, 0.7, 0.7], castshadow=True)
    s.worldbody.add_light(pos=[floor_c[0] + 2, floor_c[1] + 2, 3], dir=[-0.5, -0.5, -1], diffuse=[0.3, 0.3, 0.3], castshadow=False)
    m = s.add_mesh(name='human'); m.uservert = hv.ravel().tolist(); m.userface = hf.ravel().tolist()
    s.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname='human', rgba=[0.45, 0.6, 0.9, 1])
    if ov is not None:
        m = s.add_mesh(name='obj'); m.uservert = ov.ravel().tolist(); m.userface = of.ravel().tolist()
        s.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname='obj', rgba=[0.95, 0.55, 0.15, 1])
    if table is not None:
        s.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=(np.asarray(table['size']) / 2).tolist(),
                             pos=list(table['pos']), rgba=[0.5, 0.5, 0.5, 0.55])
    if hand_pts is not None:
        for p in hand_pts:
            s.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.012, 0, 0], pos=p.tolist(), rgba=[1, 0, 0, 1],
                                 contype=0, conaffinity=0)
    return s.compile()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cat'); ap.add_argument('key'); ap.add_argument('out')
    ap.add_argument('--src', default='converted')
    ap.add_argument('--dz', type=float, default=0.0, help='raw only: shift human by -dz mm')
    ap.add_argument('--frames', default='')
    ap.add_argument('--azim', type=float, default=None)
    a = ap.parse_args()
    body = G1SMPLX()
    if a.src == 'converted':
        x = load_grail_pickle(OUT / a.cat / 'recon' / f'{a.key}.pkl')
        od = OUT / a.cat / 'object_mjcf' / safe_name(a.key)
        ov_l, of = load_obj(od / 'grail_object.obj')
        meta = json.loads((od / 'metadata.json').read_text()) if (od / 'metadata.json').exists() else {}
        um = x.get('umrf', {})
        fps = float(x['human_data']['mocap_frame_rate'])
    else:
        x = load_grail_pickle(RAW / a.cat / 'recon' / f'{a.key}.pkl')
        sys.path.insert(0, os.path.dirname(__file__))
        from convert_grail import usd_mesh, effective_scale  # noqa
        V, F = usd_mesh(RAW / a.cat / 'object_usd' / f'{a.key}.usd')
        ov_l, of = V * effective_scale(a.cat, x)[None], F
        x['human_data']['trans'] = x['human_data']['trans'] - np.array([0, 0, a.dz / 1000.0])
        um = {}; meta = {}; fps = 24.0
    h, o = x['human_data'], x['obj_data']
    T = len(h['poses'])
    frames = np.linspace(0, T - 1, 12).round().astype(int) if not a.frames else np.array([int(f) for f in a.frames.split(',')])
    v, j = body.verts(h['poses'], h['trans'])
    tr = foot_tracks(body, v)
    V = v.cpu().numpy(); J = j.cpu().numpy()
    R, t = np.asarray(o['obj_R'], np.float64), np.asarray(o['obj_t'], np.float64)
    terrain = a.cat in TERRAIN
    mw0 = trimesh.Trimesh(ov_l @ R[0].T + t[0], of, process=False) if terrain else None
    table = None
    sd = x.get('scene_data') or {}
    if isinstance(sd, dict) and 'table' in sd:
        table = {k: np.asarray(v_, float).tolist() for k, v_ in sd['table'].items()}
    # hand vertex sets: dominant joint = wrists (20,21) or fingers (25..54)
    hand = np.where(np.isin(body.dom, [20, 21] + list(range(25, 55))))[0]
    surf = trimesh.Trimesh(ov_l, of, process=False).sample(20000) if len(of) else ov_l
    kd = cKDTree(surf)
    # camera azimuth: perpendicular to the main horizontal motion direction of the pelvis
    d = J[-1, 0, :2] - J[0, 0, :2]
    az = a.azim if a.azim is not None else (np.degrees(np.arctan2(d[1], d[0])) + 90.0 if np.linalg.norm(d) > 0.3 else 135.0)
    tiles = []
    font = ImageFont.load_default()
    for f in frames:
        ow = ov_l @ R[f].T + t[f]
        hp = V[f][hand]
        loc = (hp - t[f]) @ R[f]  # world -> object local
        dist, _ = kd.query(loc)
        k = int(np.argmin(dist))
        gL = tr['L'][0][f, 2] - support_height(tr['L'][0][f:f + 1, :2], mw0)[0]
        gR = tr['R'][0][f, 2] - support_height(tr['R'][0][f:f + 1, :2], mw0)[0]
        low = V[f][:, 2].min()
        fr2 = x.get('umrf_frame') or {}
        c = J[f, 0]
        model = build_scene(V[f], body.faces, ow, of, table, c, hand_pts=hp[[k]] if dist[k] < 0.05 else None)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        with mujoco.Renderer(model, 480, 640) as r:
            cam = mujoco.MjvCamera()
            cam.lookat[:] = [c[0], c[1], 0.45]
            cam.distance = 2.6 if not terrain else 3.4
            cam.azimuth = az
            cam.elevation = -6.0
            r.update_scene(data, cam)
            img = Image.fromarray(r.render())
        dr = ImageDraw.Draw(img)
        if 'lowest_clearance_m' in fr2:
            lc = float(fr2['lowest_clearance_m'][f]); ok = bool(fr2['src_valid'][f]); dzf = float(fr2['dz_m'][f])
            txt = (f'f{f} t={f / fps:.2f}s  body clearance to support {lc * 1000:+.0f} mm  src_valid={int(ok)}\n'
                   f'dz(t)={dzf * 1000:+.0f} mm  lowest z {low * 1000:+.0f} mm  hand-obj {dist[k] * 1000:.0f} mm')
        else:
            txt = (f'f{f} t={f / fps:.2f}s  footgap L {gL * 1000:+.0f} R {gR * 1000:+.0f} mm\n'
                   f'lowest body z {low * 1000:+.0f} mm  hand-obj {dist[k] * 1000:.0f} mm')
        dr.rectangle([0, 0, 640, 30], fill=(255, 255, 255))
        dr.text((4, 2), txt, fill=(0, 0, 0), font=font)
        tiles.append(np.asarray(img))
    rows = [np.concatenate(tiles[i * 4:(i + 1) * 4], 1) for i in range(3)]
    grid = Image.fromarray(np.concatenate(rows, 0))
    head = Image.new('RGB', (grid.width, 40), (255, 255, 255))
    dh = ImageDraw.Draw(head)
    cal = um.get('floor', {})
    v2s = cal.get('v2', {})
    xyr = cal.get('xy_registration', {})
    dh.text((6, 4), f'{a.cat} / {a.key[:100]}  src={a.src}  fps={fps}  T={T}  dz median={cal.get("dz_median_m", cal.get("dz_m", a.dz / 1000)) * 1000:+.1f} mm '
                    f'(v1 const {cal.get("dz_v1_const_m", float("nan")) * 1000:+.1f})  policy={um.get("shift_policy", "-")}  '
                    f'eff_scale={np.round(um.get("obj_scale_effective", [0])[0], 3) if um else "-"}  terrain xy offset={xyr.get("offset_applied_m", "-")}  '
                    f'src_valid={v2s.get("valid_frac", "-")}',
            fill=(0, 0, 0), font=font)
    dh.text((6, 20), 'blue=G1-SMPL-X human, orange=object visual mesh, grey=table (pickup_table), checker plane=floor z=0; red dot = closest hand vertex if < 5 cm',
            fill=(0, 0, 0), font=font)
    out = Image.new('RGB', (grid.width, grid.height + 40), (255, 255, 255))
    out.paste(head, (0, 0)); out.paste(grid, (0, 40))
    out.save(a.out)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
