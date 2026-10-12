#!/usr/bin/env python3
"""HiPHI diversity survey (read-only over metadata + converted tree).

Maps every HiPHI clip (Frame-LU + text_annotation) into the unified taxonomy
(multi-label + one primary label), infers contact types and object usage,
and joins per-part download / SMPL-X conversion / retarget status.

Outputs (all under /home/arenalabs/umr_factory_data/diversity/hiphi/):
  hiphi_clips_labeled.csv   one row per ORIGINAL clip (mirror = same labels)
  hiphi_lu_mapping.csv      one row per Frame-LU (214)
  hiphi_diversity_summary.json
  (markdown report written separately from the JSON)
"""
from surface_retargeting.config import CFG as SRCFG  # machine paths: configs/sr.toml
import json, os, re, glob, collections
import pandas as pd

ROOT = f'{SRCFG.data_root}'
RAW = f"{ROOT}/raw/hiphi"
OUT = f"{ROOT}/diversity/hiphi"

CLASSES = ["locomotion", "dynamic", "ground_contact", "sitting_on_object",
           "upper_body_manipulation", "whole_body_object", "gesture_social",
           "idle_posture", "other"]
# primary-label priority (first hit wins)
PRIORITY = ["ground_contact", "dynamic", "sitting_on_object", "whole_body_object",
            "upper_body_manipulation", "locomotion", "gesture_social",
            "idle_posture", "other"]

def R(p):
    return re.compile(p, re.I)

# ---------------------------------------------------------------- text rules
DANCE = R(r"\b(dance|dancing|danceza|dancea|ballet|samba|kpop|k-pop|tiktok|jazz|hip-hop|latin|disco|reggaeton)\b")
MARTIAL = R(r"(boxing|muay thai|sanda|^combat$|tai chi|drunken fist|mantis fist|sword|broadsword|spear|staff|cudgel|dagger|changquan|wubuquan|\bpunch|barehand play|^flag$)")
KICK = R(r"(?<!heel-)\b(kick|kicks|kicking|punt|punts|punting)\b")
ACRO = R(r"(flip|somersault|handspring|cartwheel|handstand|headstand|aerial|flair|round-?off|butterfly (twist|kick)|split leap|tiger leap|dive fall|dragon coil|right angle support|scale pose|back-arch landing|timber fall|sideward roll|^shoulder roll$|rolling breakfall|slap kick|ground slap|duozi)")
JUMP = R(r"\b(jump|jumps|jumping|hop|hops|hopping|leap|leaps|leaping|pounce|pounces|pouncing|skip|skips|skipping|skip-steps)\b|spring (straight )?upward|spring lightly upward")
FALL = R(r"(breakfall|\bfall(s|ing)?\b[^.;]{0,30}(floor|ground|knees|seated|backward|flat|diagonally)|topple[^.;]{0,30}(floor|knees|kneeling|seated)|kneeling fall|plunge (hard )?to the floor|plunge to the floor|throw yourself down|flop down|drop onto your back|seated drop)")
GETUP_PAT = R(r"(get up|get back up|gets up|stand up|stand again|to standing|rise again|rise to standing|stand back up|back up to standing|come back up|get back up|return upright|rise from|push up and come to standing|press up)")

KNEES = R(r"(kneel|kneeling|knee steps|\bon (both|one|your) knees?\b|\bonto (both|one|your) knees?\b|\bto (both|one|your) knees?\b|on one knee|onto one knee|hands and knees|forearms and knees|half-kneeling|onto your heels)")
HANDS_GROUND = R(r"(hands and knees|hands and feet|forearms and knees|(hands?|palms?) (flat |pressed )?on the (floor|ground)|push off the floor|push-?ups?|press up|push up|handstand|headstand|handspring|cartwheel|round-?off|flair|breakfall|ground slap|tiger leap|dive fall|right angle support|dragon coil|\bcrawl|scramble|sideward roll|^shoulder roll$)")
LIE_BACK = R(r"(on your back|onto your back|lie flat|lying flat|lie on your back|lying on your back|lie down|lying down|lying position|back fall|backward breakfall|fall backward to the floor|sprawl on your back|lie back down|stay(ing)? reclined|start lying)")
LIE_FRONT = R(r"(face-down|face down|on your stomach|onto your stomach|prone|on your belly|forward breakfall|dive fall)")
LIE_SIDE = R(r"(on (one|your) side\b|onto your side|side breakfall|lying fully on your side)")
SIT_GROUND = R(r"(^sit down and|sit(ting)? (down )?(on|onto|to) the floor|sit upright on the floor|seated position on the floor|cross-legged|sit in a (tight|loose|small)|sit(ting)? with your legs|seated (position|drop)|into a seated position|legs straight out in front on the floor|sit back down quickly|sprawled sit|sit down onto your heels|sit down from standing)")
FLOOR_GENERIC = R(r"(topple|plunge|fall|drop|throw yourself|flop|let yourself)[^.;]{0,40}(to|onto|on) the (floor|ground)|floor sprawl|flop down|close to the floor on your")
ROLL_GROUND = R(r"(forward / backward / sideward roll|^shoulder roll$|rolling breakfall|roll onto your side|roll up and stand)")

SIT_OBJ = R(r"(^sit\b|(you|your|and|to|then|,|as you|while you) sit\b|\bsit (on|down|astride|sideways|deep|diagonally|farther|tall|upright|squarely|perched|with|back)|sitting|seated|\bperch|astride|straddle|on your buttocks|sit up|lower onto|onto it for a perched|sideways seat|folded seat|back onto the seat|sprawled sit|drop onto the chair|sitting position)")
STEP_ON = R(r"(step(s|ping)? (up )?(onto|on to)|step over the bench|(step|steps|stepping|footfalls)[^.;]{0,30}(along|on) the top of the bench|along the top of the bench|onto one end of the bench|step onto|stand on (the|top)|climb)")
OBJ_FOOT = R(r"(kick|punt|side-foot|with (your|one|the) (foot|feet|toe|instep|sole)|top of your foot|foot pulls|one foot\b[^.;]{0,30}(ball|box)|lower leg around|hook one (lower )?leg)")
OBJ_BODY = R(r"(against your (hip|hips|body|chest|ribs|side|stomach|torso|thigh|thighs)|onto your (hip|shoulder)|on (your|one) shoulder|shoulder it|shoulder area|your back against|crook of your arm|\bhug|cradle|tight against|braced against|wedged|pinned|on your forearm|forearms? (on|resting on)|elbow|into your chest|between your (legs|thighs)|against your)")
HANDS_WORDS = R(r"(hand|palm|grab|grip|hold|lift|pick|carry|catch|throw|toss|push|pull|drag|haul|seize|clasp|squeeze|stroke|touch|nudge|hook|arm|tip|shake|rock|turn|rotate|spin|swing|guide|slide)")

LOCO = R(r"(\bwalk|\brun\b(?! of)|\bruns\b|running|\bjog|sprint|side-?step|\bsidestep|backpedal|\bstrides?\b|figure-eight|figure eight|\bloops?\b|looping|curved path|curving|\bcurves?\b|s-curves|shuttle|shuttling|\btravel(ing|ling)?\s+(forward|backward|back|sideways|in|along|across|around)|\bsteps\b|\bstepping\b|\bmarch(?!\s+in place)|marching steps|tiptoe|\blunge|shuffle (sideways|forward|backward|back)|from a to b|between a and b|between center and corner|(move|moving|step|go|hurry|carry it|carry the|hop|jump|shuffle|stomp)\s+(\w+\s+){0,3}(forward|backward|back and forth|forward and back|sideways|ahead|around|along|in (a|small|large|wide|tight|loose|broad|fast|quick|smooth|rounded|wider|larger|bigger)))")
IN_PLACE_ONLY = R(r"\bin place\b")
GROUND_GAIT = R(r"(\bcrawl|burrow[^.;]{0,40}(knees|forearms)|scramble|knee steps|hands and knees|hands and feet|forearms and knees)")
STUMBLE = R(r"(stumble|trip\b|teeter|wobbly|near-falling|nearly tip|overbalanc|off-balance|lurch|totter|past balance)")

GESTURE = R(r"(gesture|gesticulat|speech|beckon|salute|\bclap|curtsy|\bshrug|\bnod\b|ushering|for someone to)")
IDLE = R(r"(balance on one foot|single-leg balance|stand(ing)? (upright|tall)|standing posture|stretch|reach both arms|slouched standing)")

# ---------------------------------------------------------------- LU defaults
LOCO_LU_FRAMES = {"Change_direction", "Patrolling", "Dodging"}
SELF_MOTION_NOT_LOCO = {"Self_motion-crawl", "Self_motion-burrow", "Self_motion-scramble",
                        "Self_motion-vault", "Self_motion-spring", "Self_motion-sashay",
                        "Self_motion-hop", "Self_motion-jump", "Self_motion-step"}
GESTURE_LU = {"Gesture-gesticulate", "Gesture-gesture", "Body_movement-clap", "Body_movement-nod",
              "Body_movement-shrug", "Body_movement-curtsy", "Body_movement-wave",
              "Body_movement-shiver", "Posture-cower"}
IDLE_LU = {"Posture-stand", "Posture-bend", "Posture-slouch", "Posture-crouch", "Posture-squat",
           "Posture-huddle", "Change_posture-stoop", "Change_posture-bend", "Change_posture-hunch_up",
           "Change_posture-crouch_down", "Change_posture-squat_down", "Body_movement-stretch",
           "Body_movement-reach", "Body_movement-arch", "Body_movement-bend", "Body_movement-flex"}
ISOLATION_LU = {"Body_movement-bob", "Body_movement-crane", "Body_movement-cross", "Body_movement-drop",
                "Body_movement-duck", "Body_movement-flap", "Body_movement-fling", "Body_movement-flutter",
                "Body_movement-hang", "Body_movement-lift", "Body_movement-move", "Body_movement-roll",
                "Body_movement-shake", "Body_movement-swing", "Body_movement-toss", "Body_movement-twist",
                "Body_movement-twitch", "Body_movement-wiggle", "Body_movement-wriggle",
                "Body_movement-shuffle", "Body_movement-stamp", "Manipulation-paw",
                "Scouring-rifle", "Scouring-scour", "Scouring-sweep",
                "Motion_directional-dip", "Motion_directional-rise", "Motion_directional-drop",
                "Body_movement-kneel"}
# HOI objects that count as "big" (whole-body) when transported / pushed / tipped
BIG_OBJ = {"chair", "table", "box", "stepstool", "bench", "clothrack", "trashbin"}
TRANSPORT_LU_FRAMES = {"Bringing"}
TRANSPORT_LU = {"Cause_motion-move", "Cause_motion-push", "Cause_motion-pull", "Cause_motion-lift",
                "Cause_motion-drag", "Cause_motion-draw", "Cause_motion-haul", "Cause_motion-force",
                "Cause_motion-shove", "Cause_motion-propel", "Cause_motion-thrust", "Cause_motion-punt",
                "Taking-grab", "Taking-seize", "Taking-take", "Placing-shoulder", "Placing-pile",
                "Placing-situate", "Cause_to_move_in_place-roll",
                "Cause_motion-drop", "Self_motion-step"}
TRANSPORT_TXT = R(r"(\bcarry|\blift|hoist|\blug\b|\bhump|\bhaul|\bdrag|\bpush|\bpull|\bshove|\bforce\b|\bforcing|\bthrust|\bpropel|\bmove the|\bslide it|sliding the|guide the|\btip\b|end over end|whole body|full-body|whole-body|deep squat|take the (chair|table|box|stepstool)|take it with you|with you as you|tilt the bench)")

AMBIGUOUS_LU_NOTES = {
    "Body_movement-swing": "混合(162 条): 43 条是武术/器械套路名(Tornado kick/Tai Chi/剑/枪/棍…, 0.62 h)→dynamic(martial); 116 条是单腿/单臂左右摆(2.0 h)→other(原地局部动作)",
    "Self_motion-sashay": "混合(156 条): 46 条是具名舞蹈(Samba/KPOP/古典/民族/爵士…, 2.94 h, 单条最长 309 s)→dynamic(dance); 110 条是扭胯走步(1.95 h)→locomotion",
    "Self_motion-vault": "70 条全是体操/特技/武术动作名(后空翻/侧手翻/倒立/滚翻/倒地/旋子): 67 条 acrobatic, 其中 46 条有手撑地/倒地→主类 ground_contact; 3 条单腿平衡→idle_posture",
    "Cause_motion-hit": "混合(48 条): 15 条徒手格斗(Boxing/Muay Thai/Sanda/Combat, 无物体, 0.82 h)→dynamic(martial); 15 条用脚背踢箱子→whole_body_object+kick; 18 条手掌/前臂拍箱子→upper_body_manipulation",
    "Self_motion-step": "混合(86 条): 20 条踏上/跨过/沿长凳顶面走(物体-脚接触, 类台阶 0.34 h); 19 条原地踏步; 其余推桌/推箱/扶凳走",
    "Self_motion-trip": "绊脚踉跄(扰动步态)→归 locomotion, 子标签 stumble; 不算 fall",
    "Self_motion-totter": "摇晃欲倒步态→locomotion+stumble",
    "Self_motion-lurch": "前扑踉跄步态→locomotion+stumble",
    "Motion_directional-fall": "混合(37 条): 28 条真倒地/跪倒/坐倒/护身倒法(ground_contact+fall); 9 条只是落入低蹲(idle_posture/locomotion)",
    "Motion_directional-topple": "混合(39 条): 20 条倒地/跪倒; 19 条踉跄后落入低蹲",
    "Change_posture-sit_down": "混合(29 条): 14 条坐到地上/跪坐(ground_contact) vs 15 条坐到椅/箱/桌(sitting_on_object)",
    "Change_posture-stand_up": "混合(36 条): 15 条从地上/跪/趴起身(ground+getup) vs 21 条从椅子/箱子起身(sitting_on_object)",
    "Change_posture-sprawl": "混合: 扑倒在地 vs 瘫坐/趴在椅子上",
    "Posture-sit": "56 条: 53 条坐椅/凳/箱(sitting_on_object); 3 条坐地上抱着衣架/垃圾桶/水桶(ground_contact)",
    "Manipulation-paw": "框架叫 Manipulation 但无物体: 是手在自己腿上抓挠→other(身体局部)",
    "Scouring-rifle": "无物体的'搜寻'哑剧(下蹲伸手)→other/locomotion",
    "Scouring-scour": "无物体的环顾/转身搜寻→locomotion/other",
    "Scouring-sweep": "无物体的'扫视/伸手扫地面'→other",
    "Body_movement-wave": "是手臂来回挥摆, 不一定是社交挥手; 归 gesture_social(wave)",
    "Body_movement-kneel": "跪地后做上身动作→ground_contact",
    "Body_movement-shiver": "发抖/寒颤(情绪)→gesture_social(emote); 多数边走边抖→也算 locomotion",
    "Posture-cower": "蜷缩畏惧姿态→gesture_social(emote)+idle_posture",
    "Posture-shrink": "抱膝坐地蜷缩→ground_contact(hips)",
    "Attaching-hook": "用手臂弯/脚背/小腿勾衣架: 一半是勾住后全身拉/用脚抬架子(whole_body_object, 物体-脚接触 18 条), 一半是手臂勾挂(upper_body_manipulation)",
    "Placing-lean": "把椅子斜靠在髋部→upper_body_manipulation + object-body",
    "Change_posture-lean": "HOI(32 条): 坐在椅/箱上前倾(sitting_on_object 40%), 全身压/靠在椅子或箱子上(whole_body_object 41%), 蹲/跪在椅边(ground 16%)",
    "Cause_motion-punt": "踢箱子/球(object-feet, 140 条): 多标签 whole_body_object(kick ball)+dynamic(kick), 主类按优先级记为 dynamic",
    "Cause_motion-force": "推/顶衣架、长凳、垃圾桶、球, 多为全身发力",
    "Cause_motion-shove": "同上",
    "Cause_to_move_in_place-turn": "混合: 77% 原地扭转/旋转物体(upper_body_manipulation), 23% 把长凳/箱子立起翻倒(whole_body_object)",
    "Self_motion-press": "'press forward' 是前倾推进步态, 不是按压物体",
    "Self_motion-edge": "'edge' 是沿边缘踮脚走, 不是物体",
    "Self_motion-shrink": "缩身怯步走→locomotion",
    "Self_motion-slip": "'slip' 是低身滑步走, 不是滑倒",
    "Self_motion-back": "全是后退步态(walk/march backward)",
}


def classify(row):
    lu = row.frame_lu
    frame = row.frame
    t = str(row.text_annotation).replace("step stool", "stepstool").replace("Step stool", "stepstool")
    tl = t.lower().replace("_", " ")
    tl = re.sub(r"(heels kick|kick them apart)", "flick", tl)
    # object-as-subject "sit" (how it sits / legs to sit evenly) and push-back-upright are not sitting / transport
    tl = re.sub(r"\b(how it|where it|it|legs|the legs|they) (to )?sits?\b", "it rests", tl)
    tl = re.sub(r"(push|press|pull) (yourself )?back (up|upright)", "return upright", tl)
    hoi = bool(row.is_hoi)
    objs = [] if pd.isna(row.object_categories) else str(row.object_categories).split(";")
    labels, sub = set(), set()
    contacts = set()

    # ---------------- dynamic
    if DANCE.search(tl):
        labels.add("dynamic"); sub.add("dance")
    if (MARTIAL.search(tl.strip()) and lu != "Body_movement-stretch") or (lu == "Body_movement-swing" and len(t) < 70 and not tl.startswith("swing one")
                                      and "balance" not in tl):
        labels.add("dynamic"); sub.add("martial")
    if lu == "Cause_motion-hit" and not hoi:
        labels.add("dynamic"); sub.add("martial")
    if KICK.search(tl):
        labels.add("dynamic"); sub.add("kick")
    if ACRO.search(tl) or (lu == "Self_motion-vault" and "balance" not in tl):
        labels.add("dynamic"); sub.add("acrobatic")
    if JUMP.search(tl) or lu in ("Self_motion-spring", "Self_motion-pounce"):
        labels.add("dynamic"); sub.add("jump_hop")
    fall = bool(FALL.search(tl)) and not hoi
    if fall:
        labels.add("dynamic"); sub.add("fall")

    # ---------------- ground contact (body on the floor)
    g = set()
    if KNEES.search(tl): g.add("knees")
    if HANDS_GROUND.search(tl): g.add("hands_ground")
    if LIE_BACK.search(tl): g.add("back")
    if LIE_FRONT.search(tl): g.add("front")
    if LIE_SIDE.search(tl) and ("lie" in tl or "lying" in tl or "sprawl" in tl or "breakfall" in tl): g.add("side")
    sitg = SIT_GROUND.search(tl)
    if sitg and (not hoi or "floor" in tl):
        g.add("hips_ground")
    if lu in ("Posture-shrink",) or (lu == "Posture-sprawl" and tl.startswith("sit")):
        g.add("hips_ground")
    if not hoi and (FLOOR_GENERIC.search(tl) or ROLL_GROUND.search(tl)) and not g:
        g.add("body_ground_unspecified")
    if lu == "Self_motion-burrow" and "deep crouch" in tl:
        g.discard("hands_ground")  # burrow in crouch = feet only
    if g:
        labels.add("ground_contact"); contacts |= g
        if GETUP_PAT.search(tl) or lu == "Getting_up-get_up":
            labels.add("dynamic"); sub.add("getup")

    # ---------------- HOI families
    big = any(o in BIG_OBJ for o in objs)
    if hoi:
        sit_obj = bool(SIT_OBJ.search(tl)) and not ("floor" in tl and "sit" in tl and lu != "Posture-sit"
                                                    and "chair" not in tl and "box" not in tl)
        if lu in ("Posture-sit", "Change_posture-sit_up") and "floor" not in tl:
            sit_obj = True
        if lu == "Posture-sit" and ("floor" in tl or any(x in objs for x in ("trashbin", "bucket", "clothrack"))):
            sit_obj = False  # sitting on the floor holding an object between the legs
            labels.add("ground_contact"); contacts.add("hips_ground")
        if sit_obj:
            labels.add("sitting_on_object"); contacts.add("object_hips")
        step_on = bool(STEP_ON.search(tl))
        if step_on:
            contacts.add("object_feet"); sub.add("step_onto_object")
            labels.add("locomotion"); sub.add("platform_step")
        foot = bool(OBJ_FOOT.search(tl))
        if foot:
            contacts.add("object_feet")
        if OBJ_BODY.search(tl):
            contacts.add("object_body")
        hands = bool(HANDS_WORDS.search(tl)) or not (foot or sit_obj or step_on)
        # pure foot actions (kick / foot tap / roll with foot) without hand words
        if foot and not re.search(r"(hand|palm|grab|hold|lift|pick|carry|catch|both hands)", tl):
            hands = False
        if hands:
            contacts.add("object_hands")
        kick_obj = "kick" in sub or (foot and not hands and any(x in objs for x in ("ball", "soccerball", "box")))
        if kick_obj and "kick" not in sub and not re.search(r"foot pulls|keep the ball rolling", tl):
            labels.add("dynamic"); sub.add("kick")
        transport = (frame in TRANSPORT_LU_FRAMES or lu in TRANSPORT_LU or bool(TRANSPORT_TXT.search(tl)))
        ball_whole = any(o in ("ball", "soccerball") for o in objs) and re.search(
            r"(full-body|whole-body|whole body|deep lean|along the floor with quick heavy|drive it away)", tl)
        whole = (big and transport and not sit_obj) or bool(kick_obj) or bool(ball_whole)
        if lu in ("Self_motion-step",) and step_on and not re.search(r"(push|pull|slid|guide|nudg)", tl):
            whole = False
        if whole:
            labels.add("whole_body_object")
            if kick_obj:
                sub.add("kick_object")
        if hands and not whole and not (sit_obj and not re.search(r"(lift|carry|push|pull|rock|turn|spin|tip)", tl)):
            labels.add("upper_body_manipulation")
        if hands and whole is False and sit_obj:
            pass

    # ---------------- locomotion
    ground_gait = bool(GROUND_GAIT.search(tl))
    loco = False
    if frame == "Self_motion" and lu not in SELF_MOTION_NOT_LOCO:
        loco = True
    if frame in LOCO_LU_FRAMES:
        loco = True
    if LOCO.search(tl):
        loco = True
    march_in_place = bool(re.search(r"march(ing)? in place", tl))
    if IN_PLACE_ONLY.search(tl) and not march_in_place and not re.search(r"(walk|run|loop|figure|forward|backward|sideways|back and forth|side-step|travel)", tl):
        loco = False
    if lu == "Self_motion-sashay" and "dance" in sub:
        loco = False
    if ground_gait:
        loco = False
    if march_in_place and not ground_gait:
        loco = True; sub.add("in_place_stepping")
    if loco:
        labels.add("locomotion")
        if STUMBLE.search(tl) or lu in ("Self_motion-trip", "Self_motion-totter", "Self_motion-lurch"):
            sub.add("stumble")
        if re.search(r"backward|walk back|step back|retreat|backpedal", tl) or lu == "Self_motion-back":
            sub.add("backward")
        if re.search(r"side-?step|sideways|sidestep|\bsidle", tl):
            sub.add("sidestep")
        if re.search(r"\brun\b|running|sprint|\bjog|\blope|bounding strides", tl) or lu in ("Self_motion-run",):
            sub.add("run")
        if frame == "Change_direction" or re.search(r"\bturn|figure-eight|loop|curve|slalom|veer|cut into", tl):
            sub.add("turn_curve")

    # ---------------- gesture / idle
    if lu in GESTURE_LU or GESTURE.search(tl):
        labels.add("gesture_social")
        if lu in ("Body_movement-shiver", "Posture-cower"):
            sub.add("emote")
    if lu in IDLE_LU or IDLE.search(tl) or lu == "Posture-cower":
        if not hoi:
            labels.add("idle_posture")
    if lu in ISOLATION_LU and not (labels & {"dynamic", "ground_contact"}):
        sub.add("in_place_body_isolation")
    if lu == "Motion_directional-fall" or lu == "Motion_directional-topple":
        if not g and ("crouch" in tl or "squat" in tl):
            labels.add("idle_posture"); sub.add("drop_to_crouch")
    if not labels:
        labels.add("other")
        if lu in ISOLATION_LU:
            sub.add("in_place_body_isolation")

    # ---------------- contact summary
    if not (contacts & {"knees", "hands_ground", "back", "front", "side", "hips_ground",
                        "body_ground_unspecified"}):
        contacts.add("feet_ground")  # supported by feet only (w.r.t. floor)
    primary = next(c for c in PRIORITY if c in labels)
    return primary, sorted(labels), sorted(sub), sorted(contacts), objs


def main():
    m = pd.read_csv(f"{RAW}/metadata/hiphi_metadata.csv")
    act = pd.read_csv(f"{RAW}/metadata/hiphi_actor_metadata.csv").set_index("actor_id")
    m2p = pd.read_csv(f"{RAW}/data/motion_to_part.csv")
    m2p["part"] = m2p.archive_name.str.extract(r"part_(\d{4})")[0]
    part_of = dict(zip(m2p.motion_id, m2p.part))
    status = json.load(open(f"{ROOT}/logs/hiphi_streamc_status.json"))
    prio = pd.read_csv(f"{ROOT}/logs/hiphi_part_priority.csv", dtype={"part": str})

    # parts physically complete on disk (tar size == manifest compressed size, no .aria2)
    man = pd.read_csv(f"{RAW}/data/archive_manifest.csv")
    tar_state = {}
    for _, r in man.iterrows():
        p = re.search(r"part_(\d{4})", r.archive_name).group(1)
        f = f"{RAW}/data/{r.archive_name}"
        if os.path.exists(f):
            sz = os.path.getsize(f)
            partial = os.path.exists(f + ".aria2") or sz != r.compressed_bytes
            tar_state[p] = ("partial" if partial else "complete", sz, int(r.compressed_bytes))
        else:
            tar_state[p] = ("absent", 0, int(r.compressed_bytes))
    extracted_parts = sorted(re.search(r"part_(\d{4})", d).group(1)
                             for d in glob.glob(f"{RAW}/extracted/part_*") if os.path.isdir(d))

    # converted SMPL-X (CPU pipeline) + GPU bench
    conv = set(os.path.basename(os.path.dirname(f)) for f in
               glob.glob(f"{ROOT}/smplx/hiphi/data/*/*/*/motion_actor_smplx.npz"))
    gpu = set(os.path.basename(os.path.dirname(f)) for f in
              glob.glob(f'{SRCFG.convert_work}/runs/bench4/hiphi/data/*/*/*/motion_actor_smplx.npz'))
    conv_log = [json.loads(l) for l in open(f"{ROOT}/logs/hiphi_convert.jsonl")]
    conv_ok = {os.path.basename(r["clip_id"]) for r in conv_log if r.get("ok")}
    # retargeted: look for any robot output referencing hiphi
    retargeted = set()
    for d in glob.glob(f"{ROOT}/out/*") + glob.glob(f"{ROOT}/runs/*"):
        if "hiphi" in os.path.basename(d).lower():
            retargeted.add(os.path.basename(d))

    o = m[m.mirrored == False].copy()
    mir = m[m.mirrored == True]
    res = o.apply(classify, axis=1, result_type="expand")
    o[["primary", "labels", "subtags", "contacts", "objs"]] = res
    o["hours"] = o.duration_sec / 3600.0
    o["part"] = o.motion_id.map(part_of)
    o["height_cm"] = o.actor_id.map(act.height_cm)
    o["gender"] = o.actor_id.map(act.gender)
    o["tar_state"] = o.part.map(lambda p: tar_state[p][0])
    o["extracted"] = o.part.isin(extracted_parts)
    o["converted_smplx"] = o.motion_id.isin(conv)
    o["converted_ok_log"] = o.motion_id.isin(conv_ok)
    o["gpu_bench"] = o.motion_id.isin(gpu)
    mir_conv = sum(1 for x in conv if x.endswith("__mirror"))

    def status_of(r):
        if r.converted_smplx:
            return "converted_smplx"
        if r.tar_state == "complete":
            return "on_disk_raw"
        if r.tar_state == "partial":
            return "downloading"
        return "not_downloaded"
    o["status"] = o.apply(status_of, axis=1)

    # mirror hours check
    mh = mir.duration_sec.sum() / 3600
    oh = o.hours.sum()

    # ---------------- per-class aggregation
    def agg(df):
        hs = df.hours.sum()
        out = {
            "n_clips_orig": int(len(df)),
            "hours_orig": round(hs, 2),
            "hours_with_mirror": round(2 * hs, 2),
            "share_of_orig_hours_pct": round(100 * hs / oh, 1),
            "n_actors": int(df.actor_id.nunique()),
            "actor_height_cm_range": [int(df.height_cm.min()), int(df.height_cm.max())] if len(df) else None,
            "hours_by_gender": {k: round(v, 2) for k, v in df.groupby("gender").hours.sum().items()},
            "hoi_hours_orig": round(df[df.is_hoi].hours.sum(), 2),
            "status_hours_orig": {k: round(v, 2) for k, v in df.groupby("status").hours.sum().items()},
            "status_clips_orig": {k: int(v) for k, v in df.groupby("status").size().items()},
            "retargeted_robots": 0,
            "top_frame_lus_by_hours": [[k, round(v, 2)] for k, v in
                                       df.groupby("frame_lu").hours.sum().sort_values(ascending=False).head(8).items()],
        }
        return out

    multi = {}
    for c in CLASSES:
        df = o[o.labels.apply(lambda L: c in L)]
        multi[c] = agg(df)
    prim = {}
    for c in CLASSES:
        prim[c] = agg(o[o.primary == c])

    # sub-tags
    subtag_h = collections.defaultdict(float); subtag_n = collections.Counter()
    for _, r in o.iterrows():
        for s in r.subtags:
            subtag_h[s] += r.hours; subtag_n[s] += 1
    subtags = {s: {"hours_orig": round(subtag_h[s], 2), "hours_with_mirror": round(2 * subtag_h[s], 2),
                   "n_clips_orig": subtag_n[s]} for s in sorted(subtag_h, key=lambda k: -subtag_h[k])}

    # contacts
    con_h = collections.defaultdict(float); con_n = collections.Counter()
    for _, r in o.iterrows():
        for s in r.contacts:
            con_h[s] += r.hours; con_n[s] += 1
    contacts = {s: {"hours_orig": round(con_h[s], 2), "hours_with_mirror": round(2 * con_h[s], 2),
                    "n_clips_orig": con_n[s]} for s in sorted(con_h, key=lambda k: -con_h[k])}
    feet_only = o[o.contacts.apply(lambda C: C == ["feet_ground"])]
    contacts_feet_only_strict = {"hours_orig": round(feet_only.hours.sum(), 2),
                                 "note": "只有脚着地且不碰任何物体"}

    # HOI by object
    hoi = o[o.is_hoi]
    hoi_obj = {}
    for cat, d in hoi.groupby(hoi.object_categories):
        hoi_obj[cat] = {
            "n_clips_orig": int(len(d)), "hours_orig": round(d.hours.sum(), 2),
            "hours_with_mirror": round(2 * d.hours.sum(), 2),
            "n_actors": int(d.actor_id.nunique()),
            "primary_hours": {k: round(v, 2) for k, v in d.groupby("primary").hours.sum().sort_values(ascending=False).items()},
            "n_frame_lus": int(d.frame_lu.nunique()),
            "converted_hours_orig": round(d[d.converted_smplx].hours.sum(), 2),
            "on_disk_or_converted_hours_orig": round(d[d.status.isin(["on_disk_raw", "converted_smplx"])].hours.sum(), 2),
        }
    hoi_obj = dict(sorted(hoi_obj.items(), key=lambda kv: -kv[1]["hours_orig"]))

    # ground contact detail
    gc = o[o.labels.apply(lambda L: "ground_contact" in L)]
    gc_detail = {
        "hours_orig": round(gc.hours.sum(), 2), "hours_with_mirror": round(2 * gc.hours.sum(), 2),
        "n_clips_orig": int(len(gc)), "n_actors": int(gc.actor_id.nunique()),
        "by_contact_hours_orig": {k: round(gc[gc.contacts.apply(lambda C: k in C)].hours.sum(), 2)
                                  for k in ["knees", "hands_ground", "hips_ground", "back", "front", "side",
                                            "body_ground_unspecified"]},
        "top_frame_lus": [[k, round(v, 2)] for k, v in gc.groupby("frame_lu").hours.sum().sort_values(ascending=False).head(25).items()],
        "status_hours_orig": {k: round(v, 2) for k, v in gc.groupby("status").hours.sum().items()},
    }

    # actor diversity
    a = act.reset_index()
    actor = {
        "n_actors": int(len(a)), "gender": a.gender.value_counts().to_dict(),
        "height_cm": {"min": int(a.height_cm.min()), "p10": float(a.height_cm.quantile(.1)),
                      "median": float(a.height_cm.median()), "p90": float(a.height_cm.quantile(.9)),
                      "max": int(a.height_cm.max())},
        "weight_kg": {"min": int(a.weight_kg.min()), "median": float(a.weight_kg.median()), "max": int(a.weight_kg.max())},
        "height_by_gender_median": a.groupby("gender").height_cm.median().to_dict(),
        "hours_by_gender_orig": {k: round(v, 2) for k, v in o.groupby("gender").hours.sum().items()},
        "hours_per_actor_orig": {"min": round(o.groupby("actor_id").hours.sum().min(), 2),
                                 "median": round(o.groupby("actor_id").hours.sum().median(), 2),
                                 "max": round(o.groupby("actor_id").hours.sum().max(), 2)},
        "top10_actor_share_pct": round(100 * o.groupby("actor_id").hours.sum().sort_values(ascending=False).head(10).sum() / oh, 1),
        "median_actors_per_frame_lu": float(o.groupby("frame_lu").actor_id.nunique().median()),
        "frame_lus_with_ge10_actors": int((o.groupby("frame_lu").actor_id.nunique() >= 10).sum()),
        "frame_lus_with_le3_actors": int((o.groupby("frame_lu").actor_id.nunique() <= 3).sum()),
    }

    # per-part status table
    parts = []
    st_parts = {p["part"]: p for p in status["parts"]}
    for _, r in prio.iterrows():
        p = r.part
        d = o[o.part == p]
        parts.append({
            "rank": int(r["rank"]), "part": p, "streamc_state": st_parts.get(p, {}).get("state"),
            "tar_on_disk": tar_state[p][0], "tar_GB_have": round(tar_state[p][1] / 1e9, 2),
            "tar_GB_total": round(tar_state[p][2] / 1e9, 2),
            "extracted": p in extracted_parts,
            "n_orig": int(len(d)), "orig_h": round(d.hours.sum(), 2), "hoi_h": round(d[d.is_hoi].hours.sum(), 2),
            "ground_h": round(d[d.labels.apply(lambda L: "ground_contact" in L)].hours.sum(), 2),
            "converted_orig": int(d.converted_smplx.sum()),
            "converted_h": round(d[d.converted_smplx].hours.sum(), 2),
        })

    # per-LU mapping table
    rows = []
    for lu, d in o.groupby("frame_lu"):
        ph = d.groupby("primary").hours.sum().sort_values(ascending=False)
        lab = collections.Counter(l for L in d.labels for l in L)
        rows.append({
            "frame_lu": lu, "n_orig": len(d), "hours_orig": round(d.hours.sum(), 2),
            "n_actors": d.actor_id.nunique(), "hoi_frac": round(d.is_hoi.mean(), 2),
            "objects": ";".join(f"{k}:{v}" for k, v in d.object_categories.fillna("-").value_counts().items()),
            "primary_main": ph.index[0], "primary_main_frac": round(ph.iloc[0] / d.hours.sum(), 2),
            "primary_mix": ";".join(f"{k}:{v/d.hours.sum():.2f}" for k, v in ph.items()),
            "multi_labels": ";".join(f"{k}:{v/len(d):.2f}" for k, v in lab.most_common()),
            "subtags": ";".join(f"{k}:{v}" for k, v in collections.Counter(s for S in d.subtags for s in S).most_common(4)),
            "contacts": ";".join(f"{k}:{v}" for k, v in collections.Counter(s for S in d.contacts for s in S).most_common(4)),
            "converted_orig": int(d.converted_smplx.sum()),
            "on_disk_or_converted_orig": int(d.status.isin(["on_disk_raw", "converted_smplx"]).sum()),
            "ambiguous_note": AMBIGUOUS_LU_NOTES.get(lu, ""),
        })
    lu_df = pd.DataFrame(rows).sort_values("hours_orig", ascending=False)
    lu_df.to_csv(f"{OUT}/hiphi_lu_mapping.csv", index=False)

    clip_cols = ["motion_id", "frame", "lu", "frame_lu", "duration_sec", "actor_id", "gender", "height_cm",
                 "is_hoi", "object_categories", "primary", "labels", "subtags", "contacts", "part",
                 "tar_state", "converted_smplx", "gpu_bench", "status", "text_annotation"]
    oc = o[clip_cols].copy()
    for c in ["labels", "subtags", "contacts"]:
        oc[c] = oc[c].apply(lambda L: "|".join(L))
    oc.to_csv(f"{OUT}/hiphi_clips_labeled.csv", index=False)

    summary = {
        "source": "HiPHI (Noitom Robotics, CoRL 2026, arXiv 2608.16222)",
        "generated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "license": "ModalityNet Open Research License v1.0 — NON-COMMERCIAL ONLY; no redistribution/public mirrors; no competing dataset or commercial-substitute motion generator; commercial license via licensing@modalitynet.com",
        "fps": {"nominal": 90.0, "measured_min": round(float((o.frame_count / o.duration_sec).min()), 4),
                "measured_max": round(float((o.frame_count / o.duration_sec).max()), 4)},
        "format": "55-joint BVH (Y-up, cm, ZXY Euler) + HOI object tracks (CSV, m, xyzw) + 40 OBJ meshes (+40 mirrored)",
        "totals": {
            "n_clips_orig": int(len(o)), "n_clips_mirror": int(len(mir)),
            "hours_orig": round(oh, 2), "hours_mirror": round(mh, 2), "hours_with_mirror": round(oh + mh, 2),
            "hoi_hours_orig": round(o[o.is_hoi].hours.sum(), 2), "hoi_clips_orig": int(o.is_hoi.sum()),
            "body_only_hours_orig": round(o[~o.is_hoi].hours.sum(), 2),
            "n_frames": 22, "n_frame_lus": int(o.frame_lu.nunique()),
            "unique_text_annotations": int(o.text_annotation.nunique()),
            "clip_duration_sec": {"min": round(o.duration_sec.min(), 1), "median": round(o.duration_sec.median(), 1),
                                  "max": round(o.duration_sec.max(), 1)},
            "objects_per_hoi_clip": "always exactly 1 (no multi-object rows)",
        },
        "taxonomy_rule": "multi-label from Frame-LU defaults + keyword rules on text_annotation; primary = first hit in priority " + " > ".join(PRIORITY),
        "classes_multilabel": multi,
        "classes_primary_partition": prim,
        "subtags": subtags,
        "contacts": contacts,
        "contacts_feet_only_no_object": contacts_feet_only_strict,
        "hoi_by_object": hoi_obj,
        "ground_contact": gc_detail,
        "actors": actor,
        "status": {
            "parts_total": 32,
            "tar_complete_on_disk": sorted(p for p, v in tar_state.items() if v[0] == "complete"),
            "tar_partial": {p: f"{v[1]/1e9:.2f}/{v[2]/1e9:.2f} GB" for p, v in tar_state.items() if v[0] == "partial"},
            "extracted_parts": extracted_parts,
            "streamc_snapshot": {k: status[k] for k in ["t", "current", "our_rate_MBps_10m", "remaining_download_GB",
                                                       "eta_all_parts_h", "deadline_in_h", "n_todo", "net_up"]},
            "converted_smplx_orig_clips": int(o.converted_smplx.sum()),
            "converted_smplx_mirror_clips": mir_conv,
            "converted_smplx_hours_orig": round(o[o.converted_smplx].hours.sum(), 2),
            "converted_ok_in_log": len(conv_ok),
            "gpu_bench4_clips_same_set": int(o.gpu_bench.sum()),
            "converted_dir": f"{ROOT}/smplx/hiphi (3.8 GB; smplx_gpu/hiphi does not exist; GPU re-run lives in /home/arenalabs/gpu_convert/runs/bench4/hiphi)",
            "retargeted_clips": len(retargeted), "retargeted_robots": 0,
            "status_hours_orig": {k: round(v, 2) for k, v in o.groupby("status").hours.sum().items()},
            "status_clips_orig": {k: int(v) for k, v in o.groupby("status").size().items()},
            "parts": parts,
        },
        "ambiguous_frame_lus": AMBIGUOUS_LU_NOTES,
        "outputs": {"summary_json": f"{OUT}/hiphi_diversity_summary.json",
                    "report_md": f"{OUT}/hiphi_diversity_report.md",
                    "lu_mapping_csv": f"{OUT}/hiphi_lu_mapping.csv",
                    "clips_csv": f"{OUT}/hiphi_clips_labeled.csv",
                    "script": f"{OUT}/hiphi_diversity.py"},
    }
    json.dump(summary, open(f"{OUT}/hiphi_diversity_summary.json", "w"), indent=1, ensure_ascii=False, default=str)
    print("ok", round(oh, 2), round(mh, 2))


if __name__ == "__main__":
    main()
