#!/usr/bin/env python3
"""Generate the 4 schematic SVG figures for the pose-translator design appendix.
Plain hand-authored SVG via string templates -- no external plotting libs,
so output stays clean vector line-art suitable for a paper/slide figure.
"""
import math
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

FONT = "Noto Sans CJK SC, Noto Sans, Arial, sans-serif"

# ---------- shared style ----------
COL_BONE = "#4b5563"       # slate gray - skeleton lines
COL_JOINT = "#1f2937"      # dark - joint dots
COL_JOINT_FREE = "#2563eb" # blue - free joint
COL_JOINT_PIN = "#dc2626"  # red - pinned joint
COL_BOX = "#eef2ff"        # light indigo - module box fill
COL_BOX_STROKE = "#4338ca" # indigo - module box stroke
COL_BOX2 = "#ecfeff"       # light cyan - alt module fill
COL_BOX2_STROKE = "#0e7490"
COL_ARROW = "#374151"
COL_TOKEN = "#7c3aed"      # purple - abstract token
COL_TEXT = "#111827"
COL_SUBTEXT = "#4b5563"
COL_SWING = "#059669"      # green - swing (observable)
COL_TWIST = "#ea580c"      # orange - twist (ambiguous)


def svg_header(w, h):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}" font-family="{FONT}">\n'
            f'<rect x="0" y="0" width="{w}" height="{h}" fill="white"/>\n'
            f'<defs>\n'
            f'<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{COL_ARROW}"/></marker>\n'
            f'<marker id="arrowThin" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">'
            f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{COL_SUBTEXT}"/></marker>\n'
            f'</defs>\n')


def svg_footer():
    return "</svg>\n"


def text(x, y, s, size=16, anchor="middle", color=COL_TEXT, weight="normal", style="normal"):
    return (f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" '
            f'fill="{color}" font-weight="{weight}" font-style="{style}">{s}</text>\n')


def multiline(x, y, lines, size=13, anchor="middle", color=COL_SUBTEXT, lh=17):
    out = ""
    for i, ln in enumerate(lines):
        out += text(x, y + i * lh, ln, size=size, anchor=anchor, color=color)
    return out


def rrect(x, y, w, h, rx=14, fill=COL_BOX, stroke=COL_BOX_STROKE, sw=2, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" ry="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>\n')


def arrow(x1, y1, x2, y2, color=COL_ARROW, sw=2.5, marker="arrow", dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
            f'stroke-width="{sw}" marker-end="url(#{marker})"{d}/>\n')


def curved_arrow(x1, y1, x2, y2, bulge, color=COL_ARROW, sw=2.5, dash="6,4"):
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2 - bulge
    return (f'<path d="M {x1} {y1} Q {mx} {my} {x2} {y2}" fill="none" stroke="{color}" '
            f'stroke-width="{sw}" stroke-dasharray="{dash}" marker-end="url(#arrow)"/>\n')


def bone(x1, y1, x2, y2, color=COL_BONE, sw=4):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}" stroke-linecap="round"/>\n'


def joint(x, y, r=6, color=COL_JOINT, stroke="white", sw=1.5):
    return f'<circle cx="{x}" cy="{y}" r="{r}" fill="{color}" stroke="{stroke}" stroke-width="{sw}"/>\n'


# ---------- simple stick skeletons ----------

def human_skeleton(cx, cy, s=1.0, joint_colors=None, label_joints=False):
    """Return svg for a simple humanoid stick figure centered at (cx,cy). scale s."""
    pts = {
        "head": (cx, cy - 90 * s),
        "neck": (cx, cy - 62 * s),
        "chest": (cx, cy - 30 * s),
        "pelvis": (cx, cy + 6 * s),
        "l_shoulder": (cx - 26 * s, cy - 56 * s),
        "r_shoulder": (cx + 26 * s, cy - 56 * s),
        "l_elbow": (cx - 44 * s, cy - 24 * s),
        "r_elbow": (cx + 44 * s, cy - 24 * s),
        "l_wrist": (cx - 52 * s, cy + 10 * s),
        "r_wrist": (cx + 52 * s, cy + 10 * s),
        "l_hip": (cx - 16 * s, cy + 8 * s),
        "r_hip": (cx + 16 * s, cy + 8 * s),
        "l_knee": (cx - 20 * s, cy + 52 * s),
        "r_knee": (cx + 20 * s, cy + 52 * s),
        "l_ankle": (cx - 22 * s, cy + 96 * s),
        "r_ankle": (cx + 22 * s, cy + 96 * s),
    }
    bones = [
        ("neck", "chest"), ("chest", "pelvis"),
        ("neck", "l_shoulder"), ("neck", "r_shoulder"),
        ("l_shoulder", "l_elbow"), ("l_elbow", "l_wrist"),
        ("r_shoulder", "r_elbow"), ("r_elbow", "r_wrist"),
        ("pelvis", "l_hip"), ("pelvis", "r_hip"),
        ("l_hip", "l_knee"), ("l_knee", "l_ankle"),
        ("r_hip", "r_knee"), ("r_knee", "r_ankle"),
    ]
    out = f'<circle cx="{pts["head"][0]}" cy="{pts["head"][1]}" r="{16*s}" fill="none" stroke="{COL_BONE}" stroke-width="{4*s}"/>\n'
    for a, b in bones:
        out += bone(*pts[a], *pts[b], sw=4 * s)
    for name, (x, y) in pts.items():
        if name == "head":
            continue
        c = COL_JOINT
        if joint_colors and name in joint_colors:
            c = joint_colors[name]
        out += joint(x, y, r=6 * s, color=c)
    return out


def biped_robot_skeleton(cx, cy, s=1.0, joint_colors=None):
    """A humanoid robot with a 3-segment shoulder chain (pitch/roll/yaw) to visually
    contrast joint count against the human arm (1 shoulder joint)."""
    pts = {
        "head": (cx, cy - 92 * s),
        "neck": (cx, cy - 64 * s),
        "chest": (cx, cy - 30 * s),
        "pelvis": (cx, cy + 6 * s),
        "l_sh_pitch": (cx - 18 * s, cy - 58 * s),
        "l_sh_roll": (cx - 30 * s, cy - 50 * s),
        "l_sh_yaw": (cx - 36 * s, cy - 36 * s),
        "l_elbow": (cx - 42 * s, cy - 8 * s),
        "l_wrist": (cx - 46 * s, cy + 24 * s),
        "r_sh_pitch": (cx + 18 * s, cy - 58 * s),
        "r_sh_roll": (cx + 30 * s, cy - 50 * s),
        "r_sh_yaw": (cx + 36 * s, cy - 36 * s),
        "r_elbow": (cx + 42 * s, cy - 8 * s),
        "r_wrist": (cx + 46 * s, cy + 24 * s),
        "l_hip": (cx - 16 * s, cy + 8 * s),
        "r_hip": (cx + 16 * s, cy + 8 * s),
        "l_knee": (cx - 20 * s, cy + 52 * s),
        "r_knee": (cx + 20 * s, cy + 52 * s),
        "l_ankle": (cx - 22 * s, cy + 96 * s),
        "r_ankle": (cx + 22 * s, cy + 96 * s),
    }
    bones = [
        ("neck", "chest"), ("chest", "pelvis"),
        ("neck", "l_sh_pitch"), ("l_sh_pitch", "l_sh_roll"), ("l_sh_roll", "l_sh_yaw"),
        ("l_sh_yaw", "l_elbow"), ("l_elbow", "l_wrist"),
        ("neck", "r_sh_pitch"), ("r_sh_pitch", "r_sh_roll"), ("r_sh_roll", "r_sh_yaw"),
        ("r_sh_yaw", "r_elbow"), ("r_elbow", "r_wrist"),
        ("pelvis", "l_hip"), ("pelvis", "r_hip"),
        ("l_hip", "l_knee"), ("l_knee", "l_ankle"),
        ("r_hip", "r_knee"), ("r_knee", "r_ankle"),
    ]
    out = f'<rect x="{pts["head"][0]-13*s}" y="{pts["head"][1]-13*s}" width="{26*s}" height="{26*s}" rx="4" fill="none" stroke="{COL_BONE}" stroke-width="{4*s}"/>\n'
    for a, b in bones:
        out += bone(*pts[a], *pts[b], sw=4 * s)
    for name, (x, y) in pts.items():
        if name == "head":
            continue
        c = COL_JOINT
        if joint_colors and name in joint_colors:
            c = joint_colors[name]
        r = 5.5 * s if "sh_" in name else 6 * s
        out += joint(x, y, r=r, color=c)
    return out


def bird_skeleton(cx, cy, s=1.0):
    pts = {
        "head": (cx - 60 * s, cy - 40 * s),
        "neck": (cx - 30 * s, cy - 20 * s),
        "chest": (cx + 6 * s, cy - 6 * s),
        "tail": (cx + 62 * s, cy + 10 * s),
        "l_sh": (cx - 4 * s, cy - 16 * s),
        "l_wing1": (cx - 30 * s, cy - 42 * s),
        "l_wing2": (cx - 58 * s, cy - 60 * s),
        "r_sh": (cx - 4 * s, cy + 4 * s),
        "r_wing1": (cx - 26 * s, cy + 30 * s),
        "r_wing2": (cx - 50 * s, cy + 52 * s),
        "hip": (cx + 18 * s, cy + 14 * s),
        "l_leg_knee": (cx + 20 * s, cy + 46 * s),
        "l_foot": (cx + 12 * s, cy + 78 * s),
    }
    bones = [
        ("head", "neck"), ("neck", "chest"), ("chest", "tail"),
        ("chest", "l_sh"), ("l_sh", "l_wing1"), ("l_wing1", "l_wing2"),
        ("chest", "r_sh"), ("r_sh", "r_wing1"), ("r_wing1", "r_wing2"),
        ("chest", "hip"), ("hip", "l_leg_knee"), ("l_leg_knee", "l_foot"),
    ]
    out = ""
    for a, b in bones:
        out += bone(*pts[a], *pts[b], sw=4 * s)
    for name, (x, y) in pts.items():
        out += joint(x, y, r=6 * s, color=COL_JOINT)
    return out


# =====================================================================
# FIGURE 1 -- overall pipeline
# =====================================================================

def fig1():
    W, H = 1440, 620
    s = svg_header(W, H)

    # --- source skeleton panel ---
    s += rrect(30, 60, 190, 420, rx=16, fill="#f9fafb", stroke="#d1d5db", sw=1.5)
    s += text(125, 92, "输入：源骨架的一帧姿态", size=15, weight="bold")
    s += human_skeleton(125, 300, s=1.05)
    s += multiline(125, 430, ["任意关节数 J_src", "（人 / 动物 / 任意拓扑）"], size=12.5)

    x = 260
    s += arrow(230, 300, x, 300)

    # --- encoder box ---
    ew = 210
    s += rrect(x, 210, ew, 180, fill=COL_BOX, stroke=COL_BOX_STROKE)
    s += text(x + ew / 2, 245, "① 共享编码器", size=17, weight="bold", color=COL_BOX_STROKE)
    s += text(x + ew / 2, 268, "Graph Transformer", size=12.5, color=COL_SUBTEXT, style="italic")
    s += multiline(x + ew / 2, 300, [
        "每个关节看自己的方向、",
        "骨长比例、名字语义，",
        "还会参考骨架的连接",
        "关系（谁是谁的父节点）",
    ], size=12.5)
    s += text(x + ew / 2, 372, "→ 关节数可变，不挑骨架", size=12, color=COL_BOX_STROKE, weight="bold")

    x2 = x + ew + 40
    s += arrow(x + ew, 300, x2, 300)

    # --- bottleneck box ---
    bw = 230
    s += rrect(x2, 190, bw, 220, fill="#f5f3ff", stroke=COL_TOKEN)
    s += text(x2 + bw / 2, 220, "② 摘要成 K=8 个 token", size=16, weight="bold", color=COL_TOKEN)
    s += text(x2 + bw / 2, 240, "跟骨架种类完全无关", size=12, color=COL_SUBTEXT)
    # draw 8 tokens 4x2
    tok_r = 13
    gx0, gy0 = x2 + bw / 2 - 3 * 34, 285
    for i in range(8):
        row, col = divmod(i, 4)
        tx = gx0 + col * 34 + 17
        ty = gy0 + row * 40
        s += f'<circle cx="{tx}" cy="{ty}" r="{tok_r}" fill="{COL_TOKEN}" opacity="0.85"/>\n'
        s += text(tx, ty + 5, str(i + 1), size=12, color="white", weight="bold")
    s += multiline(x2 + bw / 2, 380, ["FSQ 量化压缩成离散码", "（这一层不装几何量，只装", "“这一帧大致在做什么”）"], size=11.8)

    x3 = x2 + bw + 40
    s += arrow(x2 + bw, 300, x3, 300)

    # --- iterative decoder box ---
    dw = 340
    s += rrect(x3, 150, dw, 320, fill="#ecfeff", stroke=COL_BOX2_STROKE)
    s += text(x3 + dw / 2, 182, "③ 迭代解码器", size=17, weight="bold", color=COL_BOX2_STROKE)
    s += text(x3 + dw / 2, 203, "（类似 AlphaFold 的做法）", size=12, color=COL_SUBTEXT, style="italic")

    inner_w = dw - 50
    ix = x3 + 25
    s += rrect(ix, 222, inner_w, 70, rx=10, fill="white", stroke=COL_BOX2_STROKE, sw=1.5)
    s += multiline(ix + inner_w / 2, 245, ["每一步：目标骨架的关节互相", "“看一眼”，一起把姿态往前调一点"], size=12.3)

    s += curved_arrow(ix + inner_w - 10, 300, ix + 10, 300, bulge=-46, color=COL_BOX2_STROKE)
    s += text(x3 + dw / 2, 340, "重复 L 次（约 4～8 步）", size=12, color=COL_BOX2_STROKE, weight="bold")

    s += rrect(ix, 362, inner_w, 82, rx=10, fill="#fff7ed", stroke="#c2410c", sw=1.5)
    s += multiline(ix + inner_w / 2, 386, ["如果某个关节被指定了精确位置", "或角度：每一步都强制把它按", "目标数值覆写，再让别的关节响应"], size=12, color="#9a3412")

    s += text(x3 + dw / 2, 462, "旋转是唯一学习量，位置由骨长+旋转", size=11.5, color=COL_SUBTEXT)
    s += text(x3 + dw / 2, 478, "算出来（骨长永远对，不用单独约束）", size=11.5, color=COL_SUBTEXT)

    x4 = x3 + dw + 40
    s += arrow(x3 + dw, 300, x4, 300)

    # --- target skeleton panel ---
    s += rrect(x4, 60, 190, 420, rx=16, fill="#f9fafb", stroke="#d1d5db", sw=1.5)
    s += text(x4 + 95, 92, "输出：目标骨架的一帧姿态", size=15, weight="bold")
    jc = {"l_sh_pitch": COL_JOINT_PIN, "r_wrist": COL_JOINT_PIN}
    s += biped_robot_skeleton(x4 + 95, 300, s=1.05, joint_colors=jc)
    s += multiline(x4 + 95, 430, ["任意关节数 J_tgt", "（人形机器人 / 动物 / …）"], size=12.5)

    # legend
    ly = 560
    s += f'<circle cx="60" cy="{ly}" r="7" fill="{COL_JOINT_PIN}"/>\n'
    s += text(78, ly + 5, "被钉住的关节（精确 SE3 约束）", size=12.5, anchor="start")
    s += f'<circle cx="360" cy="{ly}" r="7" fill="{COL_JOINT}"/>\n'
    s += text(378, ly + 5, "自由关节（网络自己决定）", size=12.5, anchor="start")

    s += svg_footer()
    with open(os.path.join(OUT_DIR, "fig1_pipeline.svg"), "w") as f:
        f.write(s)


# =====================================================================
# FIGURE 2 -- swing / twist decomposition
# =====================================================================

def fig2():
    W, H = 1300, 460
    s = svg_header(W, H)
    s += text(W / 2, 40, "一根骨头的旋转 = Swing（摆动） × Twist（自转）", size=19, weight="bold")

    panel_w = 400
    centers = [(80 + panel_w / 2, 250), (80 + panel_w + 60 + panel_w / 2, 250), (80 + 2 * (panel_w + 60) + panel_w / 2, 250)]
    titles = ["静止姿态（参考骨方向）", "Swing：转到观测方向", "Twist：绕骨头自己转"]
    subtitles = [
        "骨骼没有任何旋转时的样子",
        "2 个自由度，由关节位置\n完全决定，可以直接算出来",
        "1 个自由度，子关节位置\n完全不变，位置信息看不出它",
    ]
    colors = [COL_BONE, COL_SWING, COL_TWIST]

    for i, (cx, cy) in enumerate(centers):
        px = 80 + i * (panel_w + 60)
        s += rrect(px, 90, panel_w, 330, fill="#fafafa", stroke="#d1d5db", sw=1.5)
        s += text(cx, 122, titles[i], size=15, weight="bold", color=colors[i] if i else COL_TEXT)

        top = (cx, cy - 90)
        if i == 0:
            bot = (cx, cy + 90)
        elif i == 1:
            bot = (cx + 55, cy + 78)
        else:
            bot = (cx + 55, cy + 78)

        s += bone(*top, *bot, color=COL_BONE, sw=7)
        s += joint(*top, r=9, color=COL_JOINT)
        s += joint(*bot, r=9, color=COL_JOINT)

        if i == 1:
            # swing arc from vertical to tilted, on the sphere of possible directions
            r = 90
            s += (f'<path d="M {cx} {cy-90+r} A {r} {r} 0 0 1 {bot[0]} {bot[1]}" '
                  f'fill="none" stroke="{COL_SWING}" stroke-width="2.5" stroke-dasharray="5,4" marker-end="url(#arrowThin)"/>\n')
        if i == 2:
            # twist: circular arrow around the bone's own axis, drawn as an ellipse "collar" partway down the bone
            mx = (top[0] + bot[0]) / 2 + 18
            my = (top[1] + bot[1]) / 2
            s += (f'<ellipse cx="{mx}" cy="{my}" rx="22" ry="9" fill="none" stroke="{COL_TWIST}" '
                  f'stroke-width="2.8" transform="rotate(-35 {mx} {my})"/>\n')
            s += (f'<path d="M {mx+20} {my-6} l 10 -3 l -2 10 z" fill="{COL_TWIST}" '
                  f'transform="rotate(-35 {mx} {my})"/>\n')
            s += text(mx, my - 34, "转多少度都一样", size=11.5, color=COL_TWIST)

        lines = subtitles[i].split("\n")
        s += multiline(cx, 360, lines, size=12.3, color=COL_SUBTEXT)

    # equals / times signs between panels
    s += text(80 + panel_w + 30, 260, "×", size=30, color=COL_SUBTEXT, weight="bold")
    s += text(80 + 2 * panel_w + 90, 260, "=", size=30, color=COL_SUBTEXT, weight="bold")
    s += text(W / 2, 430, "实际做法：Swing 用公式直接算；Twist 才是真正需要网络去学（或建模成一个分布）的部分", size=13.5, color=COL_TEXT)

    s += svg_footer()
    with open(os.path.join(OUT_DIR, "fig2_swing_twist.svg"), "w") as f:
        f.write(s)


# =====================================================================
# FIGURE 3 -- no direct bone correspondence across skeletons
# =====================================================================

def fig3():
    W, H = 1360, 640
    s = svg_header(W, H)
    s += text(W / 2, 42, "不同骨架之间，关节没有一一对应关系", size=19, weight="bold")
    s += text(W / 2, 66, "翻译要经过一层跟骨架无关的“摘要”，不能逐骨头去对", size=13.5, color=COL_SUBTEXT)

    # left: human
    lx, ly = 230, 320
    s += rrect(60, 110, 340, 460, rx=16, fill="#f9fafb", stroke="#d1d5db")
    s += text(230, 145, "源：人体骨架", size=16, weight="bold")
    jc = {"l_shoulder": "#0891b2", "l_elbow": "#0891b2", "l_wrist": "#0891b2"}
    s += human_skeleton(lx, ly, s=1.35, joint_colors=jc)
    s += multiline(lx, 500, ["青色关节：LeftShoulder → LeftArm", "人体这一侧，肩到肘只有 1 段“上臂”"], size=12, color="#0891b2")

    # middle: bottleneck tokens
    mx = 680
    s += rrect(mx - 130, 250, 260, 140, fill="#f5f3ff", stroke=COL_TOKEN)
    s += text(mx, 280, "K 个抽象 token", size=15, weight="bold", color=COL_TOKEN)
    for i in range(6):
        col = i % 3
        row = i // 3
        tx = mx - 60 + col * 60
        ty = 320 + row * 40
        s += f'<circle cx="{tx}" cy="{ty}" r="13" fill="{COL_TOKEN}" opacity="0.85"/>\n'
    s += text(mx, 372, "（跟“上臂”“肩pitch”这些", size=11, color=COL_SUBTEXT)
    s += text(mx, 386, "具体骨头名字完全无关）", size=11, color=COL_SUBTEXT)

    # arrows many-to-few and few-to-many
    s += arrow(400, 260, mx - 130, 300, color="#0891b2")
    s += arrow(400, 300, mx - 130, 320, color="#0891b2")
    s += arrow(400, 340, mx - 130, 340, color="#0891b2")
    s += arrow(mx + 130, 300, 940, 260, color="#c2410c")
    s += arrow(mx + 130, 320, 940, 300, color="#c2410c")
    s += arrow(mx + 130, 340, 940, 340, color="#c2410c")
    s += arrow(mx + 130, 360, 940, 380, color="#c2410c")

    # right: robot with 3-segment shoulder
    rx_, ry = 1130, 320
    s += rrect(960, 110, 340, 460, rx=16, fill="#f9fafb", stroke="#d1d5db")
    s += text(1130, 145, "目标：人形机器人骨架", size=16, weight="bold")
    jc2 = {"l_sh_pitch": "#c2410c", "l_sh_roll": "#c2410c", "l_sh_yaw": "#c2410c"}
    s += biped_robot_skeleton(rx_, ry, s=1.35, joint_colors=jc2)
    s += multiline(rx_, 500, ["橙色关节：shoulder_pitch / roll / yaw", "机器人这一侧，同一个物理肩关节拆成 3 段"], size=12, color="#c2410c")

    # crossed-out direct mapping line at the bottom to make the point explicit
    s += text(W / 2, 600, "✗  “LeftArm → shoulder_pitch” 这种逐骨头直接对应关系是不存在的", size=13.5, color="#b91c1c", weight="bold")

    s += svg_footer()
    with open(os.path.join(OUT_DIR, "fig3_cross_topology.svg"), "w") as f:
        f.write(s)


# =====================================================================
# FIGURE 4 -- constraint mechanism (rotation-pin vs position-pin)
# =====================================================================

def fig4():
    W, H = 1360, 560
    s = svg_header(W, H)
    s += text(W / 2, 42, "两种“精确控制”，机制不一样", size=19, weight="bold")

    # left panel: rotation-pin
    s += rrect(60, 90, 610, 420, fill="#fff7ed", stroke="#c2410c")
    s += text(365, 125, "① rotation-pin：钉住某个关节的角度", size=16, weight="bold", color="#9a3412")
    jc = {"l_sh_pitch": COL_JOINT_PIN}
    s += biped_robot_skeleton(230, 320, s=1.25, joint_colors=jc)
    s += rrect(430, 240, 200, 170, rx=10, fill="white", stroke="#c2410c", sw=1.5)
    s += multiline(530, 268, ["每一步直接把它的角度", "覆写成指定值", "（跟 RFdiffusion 钉住", "蛋白质片段是同一个思路）"], size=12.3, color="#9a3412")
    s += text(530, 380, "精确度：浮点精度，", size=12.5, color="#9a3412", weight="bold")
    s += text(530, 398, "数学上就是对的", size=12.5, color="#9a3412", weight="bold")

    # right panel: position-pin
    s += rrect(690, 90, 610, 420, fill="#eff6ff", stroke="#1d4ed8")
    s += text(995, 125, "② position-pin：钉住某个关节要到的位置", size=16, weight="bold", color="#1e3a8a")
    jc2 = {"l_wrist": COL_JOINT_PIN}
    s += biped_robot_skeleton(870, 320, s=1.25, joint_colors=jc2)
    tx, ty = 870 - 46 * 1.25 - 46, 320 + 24 * 1.25 + 10
    s += f'<path d="M {tx-14} {ty-14} L {tx+14} {ty+14} M {tx+14} {ty-14} L {tx-14} {ty+14}" stroke="#1d4ed8" stroke-width="3"/>\n'
    s += text(tx, ty - 24, "目标点", size=11.5, color="#1d4ed8")
    s += rrect(1060, 240, 200, 170, rx=10, fill="white", stroke="#1d4ed8", sw=1.5)
    s += multiline(1160, 268, ["每一步算出当前位置和", "目标的差距，把这个差距", "反传给上游关节，带动", "整条链条一起调整"], size=12.3, color="#1e3a8a")
    s += text(1160, 380, "精确度：多步收敛，", size=12.5, color="#1e3a8a", weight="bold")
    s += text(1160, 398, "步数够多可以任意接近", size=12.5, color="#1e3a8a", weight="bold")

    s += text(W / 2, 540, "两种约束混合训练：大部分时间不加约束（直接转换姿势），少数时候钉一两个关节做局部精修", size=13, color=COL_SUBTEXT)

    s += svg_footer()
    with open(os.path.join(OUT_DIR, "fig4_constraints.svg"), "w") as f:
        f.write(s)


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
    print("done:", os.listdir(OUT_DIR))
