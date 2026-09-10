"""HPOS 混合解码路的视觉复核 (0831) —— HPOS_KEEP_REPORT.md §6.1 要求的那份视频。

三联面板 (kinematic qpos 回放, GMR g1_mocap_29dof.xml + 高对比棋盘地面):

    GT 原生 retarget | 只旋转解码 (现行) | 混合 (旋转 + hpos 当软位置先验)

三条臂共用同一份 GT 世界根 (位置来自 *_root_world.pt, 朝向来自 GT 全局根旋转) ——
和 pilot_decode_viz.py 一样, 差异纯粹来自关节解码 + 角度投影。

只旋转臂 = 现行部署路 = fit_angles(method="global") = lm_fit(w_rot=1, w_pos=0, iters=25)。
混合臂   = 同一个 lm_fit, 只多传 target_p=hpos*reach 和 w_pos (默认 0.03)。
两条臂共用同一个 greedy 热启动 q0, 同样的 iters/clamp —— 唯一变量就是位置项。

顺带量 (同一批帧, 同一把尺子):
  · 脚穿地深度: MuJoCo 真碰撞 (mj_forward 的 contact.dist < 0), 分"脚"与"全身"
  · 支撑脚滑移: 相邻两帧都与地面接触时脚踝 body 的水平位移 (cm/s)
  · 位置误差 vs GT: 按拓扑分组 (hpos_keep_probe.groups: hand/arm/leg/torso), cm
  · 旋转 geodesic vs GT (deg) + 投影前后的逐帧角跳变 (manifold_project.temporal_report)

usage:
  python hpos_hybrid_viz.py [--windows ground_aiming,crawl_ground1,fallgetup]
                            [--w_pos 0.03] [--iters 25] [--out /home/arenalabs/tmp]
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco as mj
import imageio.v2 as iio
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

from common import rot6d_to_matrix, geodesic_deg
from dual_common import build_model, geom, fk_pos_torch
from manifold_project import (fit_angles, lm_fit, fk_from_angles, load_dof,
                              temporal_report)
from hpos_keep_probe import groups, stats, window_sets
import v2_judge as VJ

AX = np.array([[0., 1, 0], [0, 0, 1], [1, 0, 0]])       # v_ardy = AX @ v_mj
XML = "/var/tmp/alphamotion/config_harvest/GMR/assets/unitree_g1/g1_mocap_29dof.xml"
RUNS = "/var/tmp/alphamotion/topo_exp/runs"
CKPT = f"{RUNS}/PT_c1dec_reenc_0812"
TGT = "unitree_g1_29dof"
SRC = VJ.SRC
T = VJ.T                                                 # 90 帧, 与判官同口径
PANEL_W, PANEL_H = 560, 560
GRID_IDX = np.linspace(0, T - 1, 8).round().astype(int)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# 窗口: tag -> (clip, start row)。
#   ground_aiming = 判官固定窗 #17 (aiming1_subject4 @11023, 骨盆 37-42cm 全窗贴地作业,
#                   留出片段) —— pilot_decode_viz 的 "ground" pick, 之前用过的那一个。
#   crawl_ground1 = 真趴地 (ground1_subject1, 骨盆 10.7-12.5cm) —— 训练行, 判官卷里没有。
#   fallgetup     = 摔倒+爬起 (fallAndGetUp1_subject1, 骨盆 5 -> 82cm) —— 训练行。
WINDOWS = {
    "ground_aiming": ("aiming1_subject4", 11023),
    "crawl_ground1": ("ground1_subject1", 154517),
    "fallgetup": ("fallAndGetUp1_subject1", 90800),
}


# ------------------------------------------------------------------ 场景
def add_checker_floor(sp):
    """高对比棋盘地面 (照抄 pilot_decode_viz2.py; GMR 那张 XML 的地面是纯白棋盘,
    渲染出来是一片灰雾, 看不出脚有没有贴地)。"""
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


def label(img, text, sub=""):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype(FONT, 22)
    f2 = ImageFont.truetype(FONT, 15)
    d.rectangle([0, 0, im.width, 30 + (20 if sub else 0)], fill=(15, 15, 20))
    d.text((8, 4), text, font=f, fill=(255, 255, 255))
    if sub:
        d.text((8, 30), sub, font=f2, fill=(190, 200, 210))
    return np.asarray(im)


# ------------------------------------------------- MuJoCo 接触 / 穿地 / 滑移
def foot_body_set(model):
    """脚 = ankle_roll_link 及其子孙 body (toe_link)。"""
    names = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, b) or "" for b in range(model.nbody)]
    foot = np.zeros(model.nbody, bool)
    for b in range(model.nbody):
        k = b
        while k > 0:
            if "ankle_roll" in names[k]:
                foot[b] = True
                break
            k = int(model.body_parentid[k])
    return foot, names


def contact_probe(model, data, qpos, floor_gid, foot_body, ankle_bid):
    """逐帧跑 mj_forward, 收集与地面的真碰撞。

    返回 dict:
      pen_foot / pen_all [Tf]  该帧最大穿地深度 (cm, >=0; 无接触则 0)
      touch [Tf,2]  左右脚该帧是否与地面接触
      ank_xy [Tf,2,2]  左右脚踝 body 的水平位置 (cm)
      ank_z [Tf,2]  左右脚踝 body 高度 (cm)
    """
    Tf = len(qpos)
    pen_foot = np.zeros(Tf); pen_all = np.zeros(Tf)
    touch = np.zeros((Tf, 2), bool)
    ank_xy = np.zeros((Tf, 2, 2)); ank_z = np.zeros((Tf, 2))
    for t in range(Tf):
        data.qpos[:] = qpos[t]
        data.qvel[:] = 0
        mj.mj_forward(model, data)
        for i in range(data.ncon):
            c = data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if floor_gid not in (g1, g2):
                continue
            other = g2 if g1 == floor_gid else g1
            b = int(model.geom_bodyid[other])
            d = max(-float(c.dist), 0.0) * 100.0          # cm
            pen_all[t] = max(pen_all[t], d)
            if foot_body[b]:
                pen_foot[t] = max(pen_foot[t], d)
                for s, bid in enumerate(ankle_bid):
                    k = b
                    while k > 0:
                        if k == bid:
                            touch[t, s] = True
                            break
                        k = int(model.body_parentid[k])
        for s, bid in enumerate(ankle_bid):
            ank_xy[t, s] = data.xpos[bid][:2] * 100.0
            ank_z[t, s] = data.xpos[bid][2] * 100.0
    return dict(pen_foot=pen_foot, pen_all=pen_all, touch=touch,
                ank_xy=ank_xy, ank_z=ank_z)


def contact_stats(pr, fps=30.0):
    pf, pa, tc, xy = pr["pen_foot"], pr["pen_all"], pr["touch"], pr["ank_xy"]
    sk = []
    for s in range(2):
        m = tc[:-1, s] & tc[1:, s]
        if m.any():
            d = np.linalg.norm(xy[1:, s] - xy[:-1, s], axis=-1)[m] * fps
            sk.append(d)
    sk = np.concatenate(sk) if sk else np.zeros(1)
    return {
        "pen_foot_max": round(float(pf.max()), 2),
        "pen_foot_p95": round(float(np.quantile(pf, .95)), 2),
        "pen_foot_mean": round(float(pf.mean()), 2),
        "pen_all_max": round(float(pa.max()), 2),
        "pen_all_p95": round(float(np.quantile(pa, .95)), 2),
        "foot_contact_rate": round(float(tc.any(-1).mean()), 3),
        "skate_mean_cm_s": round(float(sk.mean()), 2),
        "skate_p95_cm_s": round(float(np.quantile(sk, .95)), 2),
        "ank_z_min_cm": round(float(pr["ank_z"].min()), 2),
    }


# ------------------------------------------------------------------ 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default="ground_aiming,crawl_ground1,fallgetup")
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--w_pos", type=float, default=0.03)
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--out", default="/home/arenalabs/tmp")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sweep", default="", help="额外的 w_pos 扫描, 逗号分隔 (只出数字不渲染)")
    ap.add_argument("--ctrl_fk", action="store_true",
                    help="对照臂: 位置目标换成模型自己的 FK(hrot) 位置 (不含 hpos 新信息)")
    ap.add_argument("--no_render", action="store_true")
    ap.add_argument("--wset", default="", help="改用 hpos_keep_probe 的整卷窗口 "
                    "(FIXED|GROUND|CTRL, 24 窗) 做统计, 自动关渲染")
    args = ap.parse_args()
    dev = args.device
    torch.manual_seed(0)
    os.makedirs(args.out, exist_ok=True)

    if args.wset:
        ws = window_sets()[0][args.wset]
        picks = {f"{args.wset}_{i:02d}_{w['clip']}": (w["clip"], w["start"])
                 for i, w in enumerate(ws)}
        args.no_render = True
    else:
        picks = {k: WINDOWS[k] for k in args.windows.split(",") if k}
    for tag, (clip, st) in picks.items():
        print(f"[pick] {tag}: {clip} @ {st}")
    rows = torch.cat([torch.arange(st, st + T) for _, st in picks.values()])
    seg = {tag: slice(i * T, (i + 1) * T) for i, tag in enumerate(picks)}

    ck = args.ckpt
    cfg = json.load(open(f"{ck}/config.json"))
    caches = [c for c in str(cfg["cache"]).split(",") if c]
    from semantics import build_name_embeddings
    names_all = []
    for n in [SRC, TGT]:
        d = np.load(f"{caches[0]}/{n}_spec.npz", allow_pickle=True)
        names_all += [str(x) for x in d["joint_names"]]
    lut = build_name_embeddings(names_all, device=dev,
                                encoder=cfg.get("sem_encoder", "siglip"))
    data = {n: VJ.load_emb(n, rows, caches, cfg, lut, dev) for n in [SRC, TGT]}
    spec, reach = data[TGT]["spec"], data[TGT]["reach"]

    model = build_model(cfg, dev)
    model.load_state_dict(torch.load(f"{ck}/model.pt", map_location=dev))
    model.eval()
    B = len(rows)
    with torch.no_grad():
        prot, ppos = model.decode(*model.encode(data[SRC]["pose"], geom(data, SRC, B)),
                                  geom(data, TGT, B))
    del model
    gt = data[TGT]["pose"]
    gtp = gt[..., 6:9] * reach                    # GT 位置 (cm, 根在原点)
    hp = ppos * reach                             # 位置头直出 (cm)
    Rg, Rgt = rot6d_to_matrix(prot), rot6d_to_matrix(gt[..., :6])
    root = int(np.where(np.asarray(spec.parents) < 0)[0][0])
    print(f"[pre-proj] geodesic(prot,GT) = {float(geodesic_deg(Rg, Rgt).mean()):.2f} deg | "
          f"hpos {float((hp - gtp).norm(dim=-1).mean()):.2f} cm | "
          f"FK(hrot) {float((fk_pos_torch(prot, spec) - gtp).norm(dim=-1).mean()):.2f} cm")

    dofz = np.load(f"{caches[0]}/{TGT}_dof.npz", allow_pickle=True)
    slot_joints = [str(s) for s in dofz["slot_joints"]]
    dof, rest = load_dof(caches[0], TGT, device=dev)
    gm = groups(spec, dev)

    # ---- 三条臂的角度 -------------------------------------------------
    q_gt, _ = fit_angles(gt[..., :6], spec, dof, rest=rest, clamp=True, method="global",
                         lm_iters=args.iters)
    q0, _ = fit_angles(prot, spec, dof, rest=rest, clamp=True, method="greedy")
    q_rot = lm_fit(Rg, spec, dof, q0, rest=rest, root_R=Rg[:, root], target_p=None,
                   w_rot=1.0, w_pos=0.0, clamp=True, iters=args.iters)
    q_hyb = lm_fit(Rg, spec, dof, q0, rest=rest, root_R=Rg[:, root], target_p=hp,
                   w_rot=1.0, w_pos=args.w_pos, clamp=True, iters=args.iters)
    angles = {"GT": q_gt, "rot": q_rot, "hybrid": q_hyb}
    for w in [float(x) for x in args.sweep.split(",") if x]:
        angles[f"hyb_w{w:g}"] = lm_fit(Rg, spec, dof, q0, rest=rest, root_R=Rg[:, root],
                                       target_p=hp, w_rot=1.0, w_pos=w, clamp=True,
                                       iters=args.iters)
    if args.ctrl_fk:
        fkp = fk_pos_torch(prot, spec)          # 模型自己的 FK 位置, 没有 hpos 的新信息
        angles[f"ctrlFK_w{args.w_pos:g}"] = lm_fit(
            Rg, spec, dof, q0, rest=rest, root_R=Rg[:, root], target_p=fkp,
            w_rot=1.0, w_pos=args.w_pos, clamp=True, iters=args.iters)
    ARMS_DEC = [a for a in angles if a != "GT"]

    kin = {}
    for tag, q in angles.items():
        with torch.no_grad():
            gR, gp = fk_from_angles(q, spec, dof, rest=rest,
                                    root_R=(Rgt if tag == "GT" else Rg)[:, root])
        kin[tag] = (gR, gp)

    # ---- 数值 (逐窗) --------------------------------------------------
    res = {"ckpt": os.path.basename(ck), "w_pos": args.w_pos, "iters": args.iters,
           "windows": {k: {"clip": v[0], "start": v[1]} for k, v in picks.items()},
           "per_window": {}}
    for tag_v, sl in seg.items():
        d = {}
        for arm in ARMS_DEC:
            gR, gp = kin[arm]
            e = (gp[sl] - gtp[sl]).norm(dim=-1).cpu()
            o = {g: stats(e, m) for g, m in gm.items()}
            o["all"] = stats(e, torch.ones_like(gm["leg"]))
            d[arm] = {"pos_cm": {k: v for k, v in o.items()},
                      "geo_deg": round(float(geodesic_deg(gR[sl], Rgt[sl]).mean()), 2),
                      "resid_vs_hpos_cm": round(float((gp[sl] - hp[sl]).norm(dim=-1).mean()), 2)}
        # 逐帧角跳变: before=只旋转臂, after=混合臂 (查混合会不会引入 LM 分支抖动)
        d["jitter_rot_vs_hybrid_deg"] = temporal_report(q_rot[sl], q_hyb[sl])
        res["per_window"][tag_v] = d

    # ---- 世界根 (三条臂共用 GT) ---------------------------------------
    rw = torch.load(f"{caches[0]}/{TGT}_root_world.pt", mmap=True)[rows].numpy()  # cm ardy Y-up
    root_pos = (rw / 100.0) @ AX
    Rmj = np.einsum("ij,tjk,kl->til", AX.T, Rgt[:, root].cpu().numpy(), AX)
    root_quat = Rotation.from_matrix(Rmj).as_quat(scalar_first=True)

    # ---- 场景 + qpos ---------------------------------------------------
    sp = mj.MjSpec.from_file(XML)
    add_checker_floor(sp)
    sun = sp.worldbody.add_light(name="_sun", pos=[0, -1, 3], dir=[0, 0.25, -1],
                                 diffuse=[0.7, 0.7, 0.7], castshadow=True)
    sun.type = mj.mjtLightType.mjLIGHT_DIRECTIONAL
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
        qn = q.cpu().numpy()
        Q = np.zeros((B, scene.nq), np.float32)
        Q[:, 0:3] = root_pos
        Q[:, 3:7] = root_quat
        for (k, s), a in adr.items():
            Q[:, a] = qn[:, k, s]
        qpos[tag] = Q

    floor_gid = mj.mj_name2id(scene, mj.mjtObj.mjOBJ_GEOM, "floor")
    assert floor_gid >= 0
    foot_body, bnames = foot_body_set(scene)
    ankle_bid = [mj.mj_name2id(scene, mj.mjtObj.mjOBJ_BODY, f"{s}_ankle_roll_link")
                 for s in ("left", "right")]
    assert all(b >= 0 for b in ankle_bid)
    print(f"[mj] floor geom={floor_gid} foot bodies={[bnames[i] for i in np.where(foot_body)[0]]}")

    sdata = mj.MjData(scene)
    probe = {}
    for tag_v, sl in seg.items():
        for arm in ["GT"] + ARMS_DEC:
            pr = contact_probe(scene, sdata, qpos[arm][sl], floor_gid, foot_body, ankle_bid)
            probe[(tag_v, arm)] = pr
            cs = contact_stats(pr)
            res["per_window"][tag_v].setdefault(arm, {})["contact"] = cs
            print(f"[contact {tag_v:14s} {arm:6s}] {cs}")

    # ---- 渲染 ----------------------------------------------------------
    if args.no_render:
        jp = f"{args.out}/hpos_hybrid_metrics{'_' + args.wset if args.wset else ''}.json"
        json.dump(res, open(jp, "w"), indent=1, ensure_ascii=False)
        print("WROTE", jp, "(no render)")
        return
    ren = mj.Renderer(scene, PANEL_H, PANEL_W)
    cam = mj.MjvCamera()
    cam.distance, cam.elevation, cam.azimuth = 2.2, -6, 135
    heads = {"GT": ("GT native retarget", "unitree_g1_29dof paired cache"),
             "rot": ("rotation-only decode", "lm_fit w_pos=0 (current path)"),
             "hybrid": ("hybrid: rot + hpos prior", f"lm_fit target_p=hpos, w_pos={args.w_pos}")}
    for tag_v, sl in seg.items():
        panels = []
        for arm in ["GT", "rot", "hybrid"]:
            F = []
            pr = probe[(tag_v, arm)]
            for i, t in enumerate(range(sl.start, sl.stop)):
                sdata.qpos[:] = qpos[arm][t]
                mj.mj_forward(scene, sdata)
                cam.lookat[:] = sdata.qpos[:3] + np.array([0, 0, 0.15])
                ren.update_scene(sdata, cam)
                h0, h1 = heads[arm]
                h1 = (f"{h1} | foot below floor {pr['pen_foot'][i]:5.1f} cm"
                      f" | ankle z {pr['ank_z'][i].min():5.1f} cm")
                F.append(label(ren.render().copy(), h0, h1))
            panels.append(F)
        comp = [np.concatenate([p[t] for p in panels], axis=1) for t in range(T)]
        clip, st = picks[tag_v]
        out = f"{args.out}/hpos_hybrid_{tag_v}.mp4"
        w = iio.get_writer(out, fps=30, codec="libx264", quality=8)
        for f in comp:
            w.append_data(f)
        for f in comp:                                  # 0.5x 慢放重播
            w.append_data(f); w.append_data(f)
        w.close()
        print("WROTE", out, len(comp), "frames (+0.5x replay)", clip, st)
        grid = np.concatenate(
            [np.concatenate([comp[i] for i in GRID_IDX[r * 2:r * 2 + 2]], axis=1)
             for r in range(4)], axis=0)
        gpath = f"{args.out}/hpos_hybrid_{tag_v}_grid.png"
        iio.imwrite(gpath, grid[::2, ::2])
        print("WROTE", gpath)

    jpath = f"{args.out}/hpos_hybrid_metrics.json"
    json.dump(res, open(jpath, "w"), indent=1, ensure_ascii=False)
    print("WROTE", jpath)


if __name__ == "__main__":
    main()
