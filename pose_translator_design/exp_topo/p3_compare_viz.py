"""Verification renders after the joint-limit finding (0831).

Mode "official": same 3 windows / same decode+projection pipeline as
pilot_decode_viz.py, but projection AND playback use the OFFICIAL-lineage
g1_29dof.xml (mjlab) instead of GMR's g1_mocap_29dof.xml.  Joint names are
identical between the two XMLs (verified); only the limit boxes differ (hip
pitch/roll/yaw + shoulder + waist_yaw + elbow are wider in the official model).
The cache's {emb}_dof.npz limits == GMR XML limits (verified bit-exact), so we
patch the dof tensor's per-slot (half_range, center) from the official XML by
joint name and run the same fit_angles(clamp=True, method="global").
GT panel: the native retarget rotations are RE-PROJECTED through the official
model exactly like the decode arms, so all four panels share one ruler.

Mode "gr3": same 3 windows decoded human_smpl -> fourier_gr3 (training-set
embodiment, judge direction), projected with the cache dof (== GMR GR3 MJCF
limits) and played on GMR's gr3v2_1_1.xml.  No GT panel: 3 panels base|fk|fkvn.
Root = GR3's own GT world root from the paired cache.

Mode "adam": same 3 windows decoded human_smpl -> pnd_adam_lite (training-set
embodiment), projected with the cache dof and played on GMR's adam_lite.xml
(already has a freejoint).  3 panels base|fk|fkvn.  The cache has no
pnd_adam_lite_root_world.pt, so root POSITION is the G1 GT world root
passthrough (same owner design as ctrl_verify/adam_tracks_lafan.py); root
orientation is Adam's own GT pelvis rotation from the paired cache.
Also prints Adam's cache-dof joint limits vs the XML for the report.

All modes render on a high-contrast checkerboard floor (0831; the old flat
gray plane hid ground contact).

usage: pilot_decode_viz2.py {official|gr3|adam} [scratch_dir]
"""
from __future__ import annotations
import os, sys, json, re, math
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco as mj
import imageio.v2 as iio
from scipy.spatial.transform import Rotation

from common import rot6d_to_matrix, geodesic_deg
from dual_common import build_model, geom
from manifold_project import fit_angles, load_dof, fk_from_angles
import v2_judge as VJ

AX = np.array([[0., 1, 0], [0, 0, 1], [1, 0, 0]])       # v_ardy = AX @ v_mj
RUNS = "/var/tmp/alphamotion/topo_exp/runs"
SRC = VJ.SRC
T = VJ.T
PICKS = None
PANEL_W, PANEL_H = 480, 480
GRID_IDX = np.linspace(0, T - 1, 8).round().astype(int)

# ---------------------------------------------------------------------------
# 0831 通用版: 任意 checkpoint 列表 x 任意窗集 (FIXED / GROUND), 带 GT 面板。
#   用法: p3_compare_viz.py <TGT> <wset> <out_dir> tag=ckpt [tag=ckpt ...]
# 与 pilot_decode_viz2.py 的解码/投影/渲染管线逐字同源, 只把写死的三条臂和窗口
# 换成参数。GT 面板一律渲 —— lloyd 的纪律: 有源动作就必须并排渲真值。
TGT = sys.argv[1]
WSET = sys.argv[2].upper()
SCRATCH = sys.argv[3]
ARMS = []
for a in sys.argv[4:]:
    t, _, c = a.partition("=")
    ARMS.append((t, c if c.startswith("/") else f"{RUNS}/{c}"))
ORDER = ["GT"] + [t for t, _ in ARMS]
os.makedirs(SCRATCH, exist_ok=True)

def _resolve_xml(emb):
    """播放用的 MJCF: 走 GMR 的 params.py, 与缓存同源 (prep_gmr_cache 就是从它建的)。
    g1 的 23/29dof 都映射到 unitree_g1。解析不到就直接报错, 不猜。"""
    import importlib.util
    alias = {"unitree_g1_29dof": "unitree_g1", "unitree_g1_23dof": "unitree_g1"}
    gp = ('/home/arenalabs/Desktop/"be water, robot"/alpha_motion2/external/GMR'
          '/general_motion_retargeting/params.py')
    sp_ = importlib.util.spec_from_file_location("gmr_params", gp)
    m_ = importlib.util.module_from_spec(sp_); sp_.loader.exec_module(m_)
    k = alias.get(emb, emb)
    assert k in m_.ROBOT_XML_DICT, f"GMR 没有 {k} 的 MJCF (可播放的具身: {sorted(m_.ROBOT_XML_DICT)})"
    return str(m_.ROBOT_XML_DICT[k])

XML = _resolve_xml(TGT)
print(f"[xml] {TGT} -> {XML}")
def xml_ranges(path):
    """joint name -> (lo, hi, axis[3]) from raw MJCF text (hinges only here)."""
    out = {}
    for m in re.finditer(r'<joint name="([^"]+)"([^>]*)>', open(path).read()):
        attrs = m.group(2)
        r = re.search(r'(?<![a-zA-Z])range="([^"]+)"', attrs)   # not actuatorfrcrange
        a = re.search(r'axis="([^"]+)"', attrs)
        if r:
            out[m.group(1)] = ([float(x) for x in r.group(1).split()],
                               [float(x) for x in a.group(1).split()] if a else [0, 0, 1])
    return out


def patch_dof_limits(dof, slot_joints, xml_path, old_xml):
    """Overwrite per-slot (limited, half/pi, center/pi) from xml_path by joint name.
    Guard: the slot's current cached limits must equal old_xml's range for that
    joint name (proves the (slot, joint-name) mapping), and old_xml/xml_path must
    agree on the joint axis (hinge angles are then directly comparable scalars;
    the dof tensor's own axis vectors live in the descriptor frame, untouched)."""
    rng, rng_old = xml_ranges(xml_path), xml_ranges(old_xml)
    dof = dof.clone()
    v = dof[:, :18].view(-1, 3, 6)
    n_patch = 0
    for k, sj in enumerate(slot_joints):
        for s, jn in enumerate([x for x in sj.split(",") if x]):
            assert jn in rng and jn in rng_old, f"{jn} missing in XML"
            (lo, hi), ax = rng[jn]
            (olo, ohi), oax = rng_old[jn]
            assert ax == oax, f"axis differs between XMLs for {jn}: {ax} vs {oax}"
            clo = float((v[k, s, 5] - v[k, s, 4]) * math.pi)
            chi = float((v[k, s, 5] + v[k, s, 4]) * math.pi)
            assert abs(clo - olo) < 1e-3 and abs(chi - ohi) < 1e-3, \
                f"cache limits != old XML for {jn}: cache ({clo:.4f},{chi:.4f}) xml ({olo},{ohi})"
            v[k, s, 3] = 1.0
            v[k, s, 4] = (hi - lo) / 2 / math.pi
            v[k, s, 5] = (hi + lo) / 2 / math.pi
            n_patch += 1
    print(f"[dof] patched {n_patch} slots from {os.path.basename(xml_path)}")
    return dof


def print_limits(dof, slot_joints, xml_path):
    """Report the per-joint limits actually used for projection (cache dof)
    next to the playback XML's range, in degrees.  Read-only, no patching."""
    rng = xml_ranges(xml_path)
    v = dof[:, :18].view(-1, 3, 6)
    print(f"[limits] cache dof vs {os.path.basename(xml_path)} (deg):")
    for k, sj in enumerate(slot_joints):
        for s, jn in enumerate([x for x in sj.split(",") if x]):
            lim = float(v[k, s, 3])
            lo = float((v[k, s, 5] - v[k, s, 4]) * 180)
            hi = float((v[k, s, 5] + v[k, s, 4]) * 180)
            if jn in rng:
                (xlo, xhi), _ = rng[jn]
                xs = f"xml=({math.degrees(xlo):8.1f},{math.degrees(xhi):8.1f})"
            else:
                xs = "xml=(no range)"
            print(f"  {jn:24s} limited={lim:.0f} cache=({lo:8.1f},{hi:8.1f}) {xs}")


def add_checker_floor(sp):
    """High-contrast checkerboard floor (0831; flat gray hid ground contact).
    texuniform=True + texrepeat 1/m -> 0.5 m squares regardless of plane size."""
    tx = sp.add_texture()
    tx.name = "_checker"
    tx.type = mj.mjtTexture.mjTEXTURE_2D
    tx.builtin = mj.mjtBuiltin.mjBUILTIN_CHECKER
    tx.rgb1 = [0.2, 0.3, 0.45]
    tx.rgb2 = [0.85, 0.85, 0.9]
    tx.mark = mj.mjtMark.mjMARK_EDGE
    tx.markrgb = [0.1, 0.1, 0.1]
    tx.width = 512
    tx.height = 512
    mt = sp.add_material()
    mt.name = "_checker_mat"
    mt.texrepeat = [1, 1]
    mt.texuniform = True
    mt.reflectance = 0
    mt.textures[mj.mjtTextureRole.mjTEXROLE_RGB] = "_checker"
    planes = [g for g in sp.geoms if g.type == mj.mjtGeom.mjGEOM_PLANE]
    if planes:
        for g in planes:
            g.material = "_checker_mat"
            g.rgba = [1, 1, 1, 1]
    else:
        g = sp.worldbody.add_geom(name="_floor", type=mj.mjtGeom.mjGEOM_PLANE,
                                  size=[20, 20, 0.1])
        g.material = "_checker_mat"
    sky = sp.add_texture()
    sky.name = "_sky"
    sky.type = mj.mjtTexture.mjTEXTURE_SKYBOX
    sky.builtin = mj.mjtBuiltin.mjBUILTIN_GRADIENT
    sky.rgb1 = [0.55, 0.7, 0.9]
    sky.rgb2 = [0.9, 0.93, 0.97]
    sky.width = 512
    sky.height = 512
    sp.visual.headlight.ambient = [0.35, 0.35, 0.35]
    sp.visual.headlight.diffuse = [0.5, 0.5, 0.5]


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    _wf = VJ.WIN_FILE_GROUND if WSET == "GROUND" else VJ.WIN_FILE
    if not os.path.exists(_wf):
        VJ.make_windows_ground() if WSET == "GROUND" else VJ.make_windows()
    win = json.load(open(_wf))
    print(f"[wset] {WSET} <- {_wf}  heldout={win.get('heldout', True)}")
    assert win["T"] == T
    # 从指定窗集里挑 3 个窗 (定种子, 可复现)。GROUND 卷子的窗全是地面/摔倒片段。
    _g = np.random.default_rng(11)
    _idx = sorted(_g.permutation(len(win["windows"]))[:3].tolist())
    picks = {f"w{n}_{win['windows'][i]['clip'][:14]}": win["windows"][i]
             for n, i in enumerate(_idx)}
    for tag, w in picks.items():
        print(f"[pick] {tag}: {w['clip']} @ {w['start']}")
    rows = torch.cat([torch.arange(w["start"], w["start"] + T) for w in picks.values()])
    seg = {tag: slice(i * T, (i + 1) * T) for i, tag in enumerate(picks)}

    cfg0 = json.load(open(f"{ARMS[0][1]}/config.json"))
    caches = [c for c in str(cfg0["cache"]).split(",") if c]
    assert os.path.realpath(os.path.join(caches[0].rstrip("/"), "sequences.json")) == \
           os.path.realpath(os.path.join(VJ.LAFAN, "sequences.json")), caches[0]
    from semantics import build_name_embeddings
    from common import SkeletonSpec
    names_all = []
    for n in [SRC, TGT]:
        d = np.load(f"{caches[0]}/{n}_spec.npz", allow_pickle=True)
        names_all += [str(x) for x in d["joint_names"]]
    lut = build_name_embeddings(names_all, device=dev, encoder=cfg0.get("sem_encoder", "siglip"))
    data = {n: VJ.load_emb(n, rows, caches, cfg0, lut, dev) for n in [SRC, TGT]}
    spec = data[TGT]["spec"]

    streams = {"GT": data[TGT]["pose"][..., :6]}
    for tag, ck in ARMS:
        cfg = json.load(open(f"{ck}/config.json"))
        model = build_model(cfg, dev)
        model.load_state_dict(torch.load(f"{ck}/model.pt", map_location=dev))
        model.eval()
        with torch.no_grad():
            B = len(rows)
            prot, _ = model.decode(*model.encode(data[SRC]["pose"], geom(data, SRC, B)),
                                   geom(data, TGT, B))
        streams[tag] = prot
        del model

    dofz = np.load(f"{caches[0]}/{TGT}_dof.npz", allow_pickle=True)
    slot_joints = [str(s) for s in dofz["slot_joints"]]
    dof, rest = load_dof(caches[0], TGT, device=dev)
    if False:
        dof = patch_dof_limits(dof, slot_joints, XML,
            "/var/tmp/alphamotion/config_harvest/GMR/assets/unitree_g1/g1_mocap_29dof.xml")
    if False:
        print_limits(dof, slot_joints, XML)
    angles = {}
    proj_tags = ORDER if "GT" in ORDER else ["GT"] + ORDER   # GT projected for
    for tag in proj_tags:                                    # numbers, not rendered
        r6 = streams[tag]
        q, active = fit_angles(r6, spec, dof, rest=rest, clamp=True, method="global")
        angles[tag] = q.cpu().numpy()
        with torch.no_grad():
            gR, _ = fk_from_angles(q, spec, dof, rest=rest,
                                   root_R=rot6d_to_matrix(r6)[:, 0])
            gd = geodesic_deg(gR, rot6d_to_matrix(r6))
        print(f"[proj] {tag}: post-projection geodesic mean={gd.mean():.2f} deg "
              f"max={gd.max():.2f} deg")
    np.savez(f"{SCRATCH}/angles_{WSET}_{TGT}.npz", slot_joints=np.array(slot_joints),
             segs=json.dumps({t: [s.start, s.stop] for t, s in seg.items()}),
             **{f"q_{t}": q for t, q in angles.items()})

    root_emb = TGT
    if not os.path.exists(f"{caches[0]}/{TGT}_root_world.pt"):
        root_emb = "unitree_g1_29dof"       # data passthrough, owner design
        print(f"[root] no {TGT}_root_world.pt in cache -> G1 GT world root "
              "passthrough (same as ctrl_verify/adam_tracks_lafan.py)")
    rw = torch.load(f"{caches[0]}/{root_emb}_root_world.pt", mmap=True)[rows].numpy()  # cm ardy Y-up
    root_pos = (rw / 100.0) @ AX
    Rroot = rot6d_to_matrix(streams["GT"][:, 0]).cpu().numpy()
    Rmj = np.einsum("ij,tjk,kl->til", AX.T, Rroot, AX)
    root_quat = Rotation.from_matrix(Rmj).as_quat(scalar_first=True)  # wxyz

    # ---- playback model: freejoint if absent, floor if absent, light ----
    sp = mj.MjSpec.from_file(XML)
    if not any(j.type == mj.mjtJoint.mjJNT_FREE for j in sp.joints):
        sp.bodies[1].add_freejoint()
    add_checker_floor(sp)
    sun = sp.worldbody.add_light(name="_sun", pos=[0, -1, 3], dir=[0, 0.25, -1],
                                 diffuse=[0.7, 0.7, 0.7], castshadow=True)
    sun.type = mj.mjtLightType.mjLIGHT_DIRECTIONAL      # even lighting everywhere
    scene = sp.compile()
    assert scene.jnt_type[0] == mj.mjtJoint.mjJNT_FREE and scene.jnt_qposadr[0] == 0
    adr = {}
    for k, sj in enumerate(slot_joints):
        for s, jn in enumerate([x for x in sj.split(",") if x]):
            jid = mj.mj_name2id(scene, mj.mjtObj.mjOBJ_JOINT, jn)
            assert jid >= 0 and scene.jnt_type[jid] == mj.mjtJoint.mjJNT_HINGE, jn
            adr[(k, s)] = int(scene.jnt_qposadr[jid])
    qpos = {}
    for tag, q in angles.items():
        Q = np.zeros((len(rows), scene.nq), np.float32)
        Q[:, 0:3] = root_pos
        Q[:, 3:7] = root_quat
        for (k, s), a in adr.items():
            Q[:, a] = q[:, k, s]
        qpos[tag] = Q

    sdata = mj.MjData(scene)
    ren = mj.Renderer(scene, PANEL_H, PANEL_W)
    cam = mj.MjvCamera()
    cam.distance, cam.elevation, cam.azimuth = 3.0, -14, 135

    for tag_v, sl in seg.items():
        panels = []
        for arm in ORDER:
            F = []
            for t in range(sl.start, sl.stop):
                sdata.qpos[:] = qpos[arm][t]
                mj.mj_forward(scene, sdata)
                cam.lookat[:] = sdata.qpos[:3] + np.array([0, 0, 0.15])
                ren.update_scene(sdata, cam)
                F.append(ren.render().copy())
            panels.append(F)
        comp = [np.concatenate([p[t] for p in panels], axis=1) for t in range(T)]
        out = f"{SCRATCH}/raw_{tag_v}_{WSET}_{TGT}.mp4"
        # rot 的 imageio 走 PyAV 插件, 既不认 quality= 也不认 output_params=, 而且
        # 奇数边长会在 avcodec_open2(libx264) 里失败。直接把帧喂给 ffmpeg 的 stdin,
        # 绕开插件, 顺带保证宽高为偶数。
        _h, _w = comp[0].shape[:2]
        _h -= _h % 2; _w -= _w % 2
        frames = [f[:_h, :_w] for f in comp] + [f[:_h, :_w] for f in comp for _ in (0, 1)]
        import subprocess
        pr = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{_w}x{_h}", "-r", "30", "-i", "-", "-c:v", "libx264",
             "-crf", "20", "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
        for f in frames:
            pr.stdin.write(np.ascontiguousarray(f, np.uint8).tobytes())
        pr.stdin.close(); assert pr.wait() == 0, "ffmpeg 失败"
        print("WROTE", out, len(comp), "frames (+0.5x replay)")
        grid = np.concatenate(
            [np.concatenate([comp[i] for i in GRID_IDX[r * 2:r * 2 + 2]], axis=1)
             for r in range(4)], axis=0)
        iio.imwrite(f"{SCRATCH}/grid_{tag_v}_{WSET}_{TGT}.png", grid[::2, ::2])
        print("WROTE", f"{SCRATCH}/grid_{tag_v}_{WSET}_{TGT}.png")


if __name__ == "__main__":
    main()
