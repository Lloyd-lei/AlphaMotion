"""接触高亮 + 机器人 mesh 的对比渲染 (lloyd 0831)。

回答的问题: 接地标签到底有没有让接触变好?
每一帧在**标签说该接触**的关节上放一个球, 颜色说的是解码出来的姿态做到没做到:
  绿  = 标签说该踩, 解码后确实在阈值内   -> 接触做到了
  红  = 标签说该踩, 解码后悬空          -> 接触没做到 (这就是"接触做得差")
  黄  = 标签没说, 但解码后穿到地下       -> 多余的穿模
球的半径 = 该机器人自己的接触阈值; 棋盘地面画在标定的 h0 上。
机器人用原生 MJCF 的 mesh 播放, 不是棍状图。

用法: p3_contact_viz.py <TGT> <wset> <out_dir> tag=ckpt [tag=ckpt ...]
"""
import os, sys, json, math, subprocess
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco as mj
from PIL import Image, ImageDraw, ImageFont
from common import rot6d_to_matrix
from dual_common import build_model, geom, fk_pos_torch
from manifold_project import fit_angles, load_dof, fk_from_angles
import v2_judge as VJ
import contact_label as CL

RUNS = "/var/tmp/alphamotion/topo_exp/runs"
TGT, WSET, OUT = sys.argv[1], sys.argv[2].upper(), sys.argv[3]
ARMS = [(a.split("=")[0], a.split("=")[1] if a.split("=")[1].startswith("/")
         else f"{RUNS}/{a.split('=')[1]}") for a in sys.argv[4:]]
ORDER = ["GT"] + [t for t, _ in ARMS]
os.makedirs(OUT, exist_ok=True)
W = H = 560
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def resolve_xml(emb):
    import importlib.util
    alias = {"unitree_g1_29dof": "unitree_g1", "unitree_g1_23dof": "unitree_g1"}
    gp = ('/home/arenalabs/Desktop/"be water, robot"/alpha_motion2/external/GMR'
          '/general_motion_retargeting/params.py')
    s = importlib.util.spec_from_file_location("gp", gp)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return str(m.ROBOT_XML_DICT[alias.get(emb, emb)])


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    wf = VJ.WIN_FILE_GROUND if WSET == "GROUND" else VJ.WIN_FILE
    win = json.load(open(wf)); T = win["T"]
    g = np.random.default_rng(11)
    idx3 = sorted(g.permutation(len(win["windows"]))[:2].tolist())
    picks = {f"w{n}_{win['windows'][i]['clip'][:16]}": win["windows"][i]
             for n, i in enumerate(idx3)}
    rows = torch.cat([torch.arange(w["start"], w["start"] + T) for w in picks.values()])
    seg = {t: slice(i * T, (i + 1) * T) for i, t in enumerate(picks)}
    for t, w in picks.items(): print(f"[pick] {t}: {w['clip']} @ {w['start']}")

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

    # ---- 接触标签 (与训练用的是同一份)
    z = np.load(f"{caches[0]}/{TGT}_contact.npz", allow_pickle=True)
    hdr = json.loads(str(z["header"])); thr = hdr["thr_m"]; h0 = hdr["h0_m"]
    ri = rows.numpy()
    lab = (z["contact"][ri] | (z["gap"][ri] < 0))          # 训练通道
    root = int(np.where(np.asarray(spec.parents) < 0)[0][0])
    gap_root = z["gap"][ri][:, root].astype(np.float32)
    print(f"[label] thr={thr*100:.2f}cm h0={h0*100:.2f}cm  标为接触的关节帧 {lab.mean()*100:.2f}%")

    streams = {"GT": data[TGT]["pose"][..., :6]}
    for tag, ck in ARMS:
        cfg = json.load(open(f"{ck}/config.json"))
        m = build_model(cfg, dev); m.load_state_dict(torch.load(f"{ck}/model.pt", map_location=dev)); m.eval()
        with torch.no_grad():
            B = len(rows)
            streams[tag], _ = m.decode(*m.encode(data[SRC]["pose"], geom(data, SRC, B)),
                                       geom(data, TGT, B))
        del m

    # ---- 每条臂的"接触做到没做到"
    status = {}   # tag -> [N,J] 0=无 1=做到 2=没做到 3=多余穿地
    for tag, r6 in streams.items():
        hgt = (fk_pos_torch(r6, spec)[..., 1] / 100.0
               + torch.from_numpy(gap_root).to(dev)[:, None]).cpu().numpy()
        st = np.zeros_like(hgt, np.int8)
        st[lab & (hgt < thr)] = 1
        st[lab & (hgt >= thr)] = 2
        st[(~lab) & (hgt < 0)] = 3
        status[tag] = st
        n = lab.sum()
        print(f"[contact] {tag:10s} 该踩做到 {100*(st==1).sum()/max(n,1):5.1f}%   "
              f"该踩悬空 {100*(st==2).sum()/max(n,1):5.1f}%   多余穿地 {(st==3).sum()}")

    dof, rest = load_dof(caches[0], TGT, device=dev)
    dofz = np.load(f"{caches[0]}/{TGT}_dof.npz", allow_pickle=True)
    slot_joints = [str(s) for s in dofz["slot_joints"]]
    angles = {}
    for tag in ORDER:
        q, _ = fit_angles(streams[tag], spec, dof, rest=rest, clamp=True, method="global")
        angles[tag] = q.cpu().numpy()

    rw = torch.load(f"{caches[0]}/{TGT}_root_world.pt", mmap=True).float()[rows].numpy() / 100.0
    AX = np.array([[0., 1, 0], [0, 0, 1], [1, 0, 0]])
    root_pos = rw @ AX                                  # ardy -> mj (Z-up)
    root_pos[:, 2] -= h0                                # 地面对齐到 z=0
    Rr = rot6d_to_matrix(streams["GT"])[:, 0].cpu().numpy()
    from scipy.spatial.transform import Rotation
    root_quat = Rotation.from_matrix(AX.T @ Rr @ AX).as_quat()[:, [3, 0, 1, 2]]

    # ---- 场景: 原生 mesh + 棋盘地面 + 一组 mocap 球
    xml = resolve_xml(TGT); print("[xml]", xml)
    sp = mj.MjSpec.from_file(xml)
    tx = sp.add_texture(); tx.name = "_ck"; tx.type = mj.mjtTexture.mjTEXTURE_2D
    tx.builtin = mj.mjtBuiltin.mjBUILTIN_CHECKER; tx.width = tx.height = 300
    tx.rgb1 = [0.22, 0.24, 0.28]; tx.rgb2 = [0.55, 0.58, 0.62]
    mt = sp.add_material(); mt.name = "_ckm"; mt.texrepeat = [12, 12]
    mt.textures[mj.mjtTextureRole.mjTEXROLE_RGB] = "_ck"
    sp.worldbody.add_geom(name="_floor", type=mj.mjtGeom.mjGEOM_PLANE,
                          size=[30, 30, 0.1], material="_ckm")
    sp.worldbody.add_light(name="_sun", pos=[0, -1, 3], dir=[0, .25, -1],
                           diffuse=[.75, .75, .75], castshadow=False)
    # 有些 MJCF (gr3) 自带 <visual><global offheight=480>, 比我们的画布小 -> Renderer 直接拒。
    # 按需放大离屏缓冲, 不改任何物理参数。
    sp.visual.global_.offheight = max(int(sp.visual.global_.offheight), H)
    sp.visual.global_.offwidth = max(int(sp.visual.global_.offwidth), W)
    NM = 8
    for k in range(NM):
        b = sp.worldbody.add_body(name=f"_m{k}", mocap=True, pos=[0, 0, -50])
        b.add_geom(name=f"_ms{k}", type=mj.mjtGeom.mjGEOM_SPHERE, size=[thr, 0, 0],
                   rgba=[1, 0, 0, 0.55], contype=0, conaffinity=0)
    scene = sp.compile()
    mid = [mj.mj_name2id(scene, mj.mjtObj.mjOBJ_BODY, f"_m{k}") for k in range(NM)]
    gid = [mj.mj_name2id(scene, mj.mjtObj.mjOBJ_GEOM, f"_ms{k}") for k in range(NM)]
    madr = [int(scene.body_mocapid[b]) for b in mid]
    adr = {}
    for k, sj in enumerate(slot_joints):
        for s, jn in enumerate([x for x in sj.split(",") if x]):
            jid = mj.mj_name2id(scene, mj.mjtObj.mjOBJ_JOINT, jn)
            if jid >= 0: adr[(k, s)] = int(scene.jnt_qposadr[jid])
    qpos = {}
    for tag, q in angles.items():
        Q = np.zeros((len(rows), scene.nq), np.float32)
        Q[:, 0:3] = root_pos; Q[:, 3:7] = root_quat
        for (k, s), a in adr.items(): Q[:, a] = q[:, k, s]
        qpos[tag] = Q
    # 关节名 -> MJCF body id (放球用)
    jb = [mj.mj_name2id(scene, mj.mjtObj.mjOBJ_BODY, n) for n in spec.joint_names]

    sd = mj.MjData(scene); ren = mj.Renderer(scene, H, W)
    cam = mj.MjvCamera(); cam.distance, cam.elevation, cam.azimuth = 3.2, -12, 135
    COL = {1: [0.15, 0.95, 0.25, 0.55], 2: [1.0, 0.12, 0.12, 0.75], 3: [1.0, 0.85, 0.1, 0.7]}
    f_ = ImageFont.truetype(FONT, 22); f2 = ImageFont.truetype(FONT, 15)

    for vname, sl in seg.items():
        panels = []
        for arm in ORDER:
            F = []
            for t in range(sl.start, sl.stop):
                sd.qpos[:] = qpos[arm][t]
                act = [j for j in range(spec.J) if status[arm][t, j] and jb[j] >= 0][:NM]
                for k in range(NM):
                    if k < len(act):
                        sd.mocap_pos[madr[k]] = 0            # 先归零, 位置下面 forward 后再放
                    else:
                        sd.mocap_pos[madr[k]] = [0, 0, -50]
                mj.mj_forward(scene, sd)
                for k in range(NM):
                    if k < len(act):
                        j = act[k]
                        sd.mocap_pos[madr[k]] = sd.xpos[jb[j]]
                        scene.geom_rgba[gid[k]] = COL[int(status[arm][t, j])]
                mj.mj_forward(scene, sd)
                cam.lookat[:] = sd.qpos[:3] + np.array([0, 0, 0.25])
                ren.update_scene(sd, cam)
                im = Image.fromarray(ren.render().copy()); dr = ImageDraw.Draw(im)
                dr.rectangle([0, 0, W, 50], fill=(0, 0, 0))
                dr.text((10, 3), arm, font=f_, fill=(255, 255, 255))
                n1 = int((status[arm][t] == 1).sum()); n2 = int((status[arm][t] == 2).sum())
                dr.text((10, 28), f"做到{n1}  悬空{n2}", font=f2,
                        fill=(120, 255, 140) if n2 == 0 else (255, 130, 130))
                F.append(np.array(im))
            panels.append(F)
        comp = [np.concatenate([p[t] for p in panels], axis=1) for t in range(T)]
        h_, w_ = comp[0].shape[:2]; h_ -= h_ % 2; w_ -= w_ % 2
        out = f"{OUT}/contact_{vname}_{WSET}_{TGT}.mp4"
        pr = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
                               "-pix_fmt", "rgb24", "-s", f"{w_}x{h_}", "-r", "30", "-i", "-",
                               "-c:v", "libx264", "-crf", "22", "-pix_fmt", "yuv420p", out],
                              stdin=subprocess.PIPE)
        for f in comp + comp:
            pr.stdin.write(np.ascontiguousarray(f[:h_, :w_], np.uint8).tobytes())
        pr.stdin.close(); assert pr.wait() == 0
        print("WROTE", out)
    ren.close()


if __name__ == "__main__":
    main()
