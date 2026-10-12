"""对比视频: 人体点云 | CPU 严格版 | GPU 严格版  (同一把尺子独立着色)
颜色(两台机器人用同一规则, 由点云几何独立判定, 不读各自管线的标签):
  红 = 该点在地面以下 >0.5 mm (穿地) ; 绿 = 该点离地 0~1 mm (真接触) ; 灰 = 其它
人体: 绿 = 工厂人体接触标签所在区域的点
用法: python render_cmp.py <clip> <robot> <start> <end> <out.mp4>
"""
import sys, json
from pathlib import Path
sys.path.insert(0, '/home/arenalabs/umr_factory_code')
import numpy as np, mujoco
from PIL import Image, ImageDraw, ImageFont
from umrf import surface as S, geometry as G, pipeline as P
from umrf.paths import OUT

clip, robot, t0, t1, outp = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
o = np.load(f'cpu_{robot}.npz', allow_pickle=True)
g = np.load(f'/var/tmp/alphamotion/umr_factory/lloyd_gpu_eval/{clip}__{robot}.gpu.npz', allow_pickle=True)
prov = json.loads(str(o['provenance']))
Hs = np.load(prov['human_slots'])['source_slots']
umr = np.load(prov['umr_out'], allow_pickle=True)
if len(Hs) != len(o['qpos']):
    Hs = Hs[umr['frame_ids']]
Hs = Hs - np.array([0, 0, float(o['floor_offset_removed'])], np.float32)
hreg, _ = P.human_slot_regions(robot, prov['source_kind'], prov['source_data'], prov['seq_key'],
                               str(Path(prov['umr_out']).with_suffix('.umr.log')))
m = mujoco.MjModel.from_xml_path(str(o['robot_xml'])); surf = S.load_surface(robot)
geo = G.RobotGeometry(m, surf, 0.0)
qg = g['q_out'].copy(); qg[:, 2] -= float(g['floor'][0])          # GPU 世界系 -> 地面 z=0
qc = o['qpos']
W, H, SC = 640, 720, 300.0
try:
    font = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 22)
    small = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 18)
except Exception:
    font = small = None


def robot_pts(q):
    geo.set(q.astype(float)); return geo.dense_world()


def draw(img, X, col, x0, y0):
    u = ((X[:, 0] - x0) * SC).astype(int); v = (y0 - X[:, 2] * SC).astype(int)
    a = np.asarray(img).copy(); ok = (u >= 0) & (u < a.shape[1]) & (v >= 0) & (v < a.shape[0])
    a[v[ok], u[ok]] = col[ok]; return Image.fromarray(a)


def robot_col(X):
    c = np.full((len(X), 3), 150, np.uint8)
    z = X[:, 2]
    c[(z >= -0.0005) & (z <= 0.001)] = [20, 170, 40]
    c[z < -0.0005] = [230, 20, 20]
    return c


def frame(t):
    img = Image.new('RGB', (3 * W, H), (250, 250, 250))
    d = ImageDraw.Draw(img)
    hx = Hs[t][:, 0].mean(); y0 = H - 60
    hc = o['human_ground_contact'][t]
    hcol = np.full((len(Hs[t]), 3), 150, np.uint8); hcol[hc[hreg]] = [20, 170, 40]
    stats = []
    for k, (X, col, title) in enumerate([(Hs[t], hcol, '人体源 (绿=人体接触区域)'),
                                         (robot_pts(qc[t]), None, 'CPU 严格版'),
                                         (robot_pts(qg[t]), None, 'GPU 严格版')]):
        if col is None:
            col = robot_col(X); zmin = X[:, 2].min() * 1000
            ncont = int(((X[:, 2] >= -0.0005) & (X[:, 2] <= 0.001)).sum())
            stats.append(f'最低点 {zmin:+.2f} mm  接触点 {ncont}')
        X = X - [X[:, 0].mean() - hx, 0, 0]
        img = draw(img, X + [k * W / SC, 0, 0], col, hx - W / SC / 2, y0)
        d = ImageDraw.Draw(img)
        d.text((k * W + 10, 10), title, fill=(0, 0, 0), font=font)
        if k > 0:
            d.text((k * W + 10, 40), stats[-1], fill=(0, 0, 0), font=small)
    d.line([(0, y0), (3 * W, y0)], fill=(0, 0, 0), width=2)
    d.text((10, H - 40), f'{clip} -> {robot}   帧 {t}   红=穿地>0.5mm  绿=离地0-1mm真接触   (两台机器人用同一点云规则独立判定)',
           fill=(0, 0, 0), font=small)
    return img


import imageio.v2 as imageio
fps = float(o['fps']); step = max(1, int(round(fps / 30)))
w = imageio.get_writer(outp, fps=fps / step, macro_block_size=8)
for t in range(t0, min(t1, len(qc)), step):
    w.append_data(np.asarray(frame(t)))
w.close()
T = len(qc); fr = np.linspace(0, T - 1, 12).astype(int)
sheet = Image.new('RGB', (4 * 3 * W // 2, 3 * H // 2), (255, 255, 255))
for i, t in enumerate(fr):
    im = frame(t).resize((3 * W // 2, H // 2))
    sheet.paste(im, ((i % 4) * (3 * W // 2) // 2 * 0 + (i % 4) * (3 * W // 2) // 1 // 2 * 0 + (i % 4) * (3 * W // 2) // 1 // 1 // 2 * 2 // 2, (i // 4) * (H // 2)))
sheet = Image.new('RGB', (2 * 3 * W // 2, 6 * H // 2), (255, 255, 255))
for i, t in enumerate(fr):
    im = frame(t).resize((3 * W // 2, H // 2))
    sheet.paste(im, ((i % 2) * (3 * W // 2), (i // 2) * (H // 2)))
sheet.save(outp.replace('.mp4', '_grid12.png'))
# 全片统计(同一规则)
pen_c = []; pen_g = []
for t in range(T):
    pen_c.append(robot_pts(qc[t])[:, 2].min()); pen_g.append(robot_pts(qg[t])[:, 2].min())
pen_c = np.array(pen_c) * 1000; pen_g = np.array(pen_g) * 1000
print(json.dumps({'cpu_min_mm': round(float(pen_c.min()), 3), 'gpu_min_mm': round(float(pen_g.min()), 3),
                  'cpu_frames_pen>0.5mm': int((pen_c < -0.5).sum()), 'gpu_frames_pen>0.5mm': int((pen_g < -0.5).sum()),
                  'frames': T}))
print('saved', outp)
