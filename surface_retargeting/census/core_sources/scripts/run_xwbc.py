from surface_retargeting.config import CFG as SRCFG  # machine paths: configs/sr.toml
import sys, json, os, re, collections, glob
import pandas as pd, numpy as np
from surface_retargeting.census.core_sources.scripts import taxo

OUT = f'{SRCFG.data_root}/diversity/xwbc_amass_lafan'
os.makedirs(OUT, exist_ok=True)
SCR = os.path.dirname(__file__)

# ---------------- AMASS (xwbc human stream) ----------------
seqs = json.load(open("/var/tmp/alphamotion/topo_exp/cache_xwbc_amass_full/sequences.json"))
members = [l.strip() for l in open(os.path.join(SCR, "amass_human_members.txt")) if l.strip().endswith(".npz")]
member_names = {os.path.basename(m)[:-4] for m in members}

# HumanML3D-style captions already aligned to these clip names
cap = collections.defaultdict(list)
for f in ["pairs_train.json", "pairs_val_amass.json"]:
    for r in json.load(open("/var/tmp/alphamotion/topo_exp/text_align/" + f)):
        if r["src"] == "amass":
            cap[r["name"]].append(r["caption"])

# BABEL v1.0 (raw release on disk) -> key (dataset, subject, stem)
BABEL_DS = {"ACCAD": "ACCAD", "BMLmovi": "BMLmovi", "CMU": "CMU", "DFaust_67": "DFaust", "DanceDB": "DanceDB",
            "EKUT": "EKUT", "Eyes_Japan_Dataset": "EyesJapanDataset", "HumanEva": "HumanEva", "KIT": "KIT",
            "MPI_HDM05": "HDM05", "MPI_Limits": "PosePrior", "MPI_mosh": "MoSh", "SFU": "SFU",
            "SSM_synced": "SSM", "TCD_handMocap": "TCDHands", "TotalCapture": "TotalCapture",
            "Transitions_mocap": "Transitions", "BioMotionLab_NTroje": "BMLrub", "BMLhandball": "BMLhandball"}


def nk(s):
    s = re.sub(r"(_poses|_stageii)$", "", s.lower())
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")


babel = {}
BD = f'{SRCFG.research_root}/data/babel/babel_v1.0_release'
for f in ["train.json", "val.json", "test.json", "extra_train.json", "extra_val.json"]:
    for k, v in json.load(open(os.path.join(BD, f))).items():
        parts = v["feat_p"].replace("\\", "/").split("/")
        ds = BABEL_DS.get(parts[1], parts[1])
        key = (ds, nk(parts[-2]), nk(parts[-1][:-4]))
        labs = babel.setdefault(key, set())
        for ann in ("seq_ann", "frame_ann"):
            a = v.get(ann)
            if a and a.get("labels"):
                for l in a["labels"]:
                    for c in (l.get("act_cat") or []):
                        labs.add(c)
                    if l.get("proc_label"):
                        labs.add(l["proc_label"])

SUBSET_DEFAULT = {"WEIZMANN": ["locomotion"], "CNRS": ["locomotion"], "PosePrior": ["idle_posture"],
                  "GRAB": ["upper_body_manipulation"], "DanceDB": ["dynamic"]}
SUBSET_LICENSE = "AMASS 非商业研究许可, 禁止再分发 (各子集另有原始许可, 如 CMU 免费/GRAB 非商业)"

rows = []
nb = 0
for s in seqs:
    name = s["name"]; parts = name.split("__")
    ds = parts[0]
    subj = parts[-2] if len(parts) >= 3 else ""
    stem = parts[-1]
    key = (ds, nk(subj), nk(stem))
    bl = babel.get(key)
    if bl: nb += 1
    tokens = " ".join(parts[1:]).replace("_stageii", "")
    text = " | ".join([tokens] + cap.get(name, []) + sorted(bl or []))
    extra = []
    if ds in SUBSET_DEFAULT and not (bl or cap.get(name)):
        extra = SUBSET_DEFAULT[ds]
    if ds == "GRAB":
        extra = ["upper_body_manipulation"]
    cls, con, objs = taxo.classify(text, extra)
    if ds == "GRAB":
        con.add("object_hands"); con.discard("feet_only")
        if not objs & set(taxo.MANIP_OBJ_CATS):
            objs.add("small_geometric_toy")
    has_obj = ("object_hands" in con) or ds == "GRAB"
    # text-only mentions without a real prop: keep as 'implied'
    rows.append(dict(clip=name, subset=ds, subject=f"{ds}/{subj}" if ds not in ("KIT", "CMU") else f"{ds}/{parts[1]}",
                     frames_30fps=s["len"], hours=s["len"] / 30 / 3600, fps=30, classes="|".join(sorted(cls)),
                     primary=taxo.primary(cls), contacts="|".join(sorted(con)), has_object_implied=bool(has_obj),
                     object_cats="|".join(sorted(objs)), babel_labelled=bool(bl), humanml3d_captioned=name in cap,
                     in_human_tar=name in member_names))
A = pd.DataFrame(rows)
A.to_csv(os.path.join(OUT, "amass_clip_labels.csv.gz"), index=False, compression="gzip")
print("AMASS clips", len(A), "hours", A.hours.sum(), "babel matched", nb, "captioned", A.humanml3d_captioned.sum(),
      "in tar", A.in_human_tar.sum(), "tar members", len(member_names))
print(A.groupby("primary").hours.sum().round(2).sort_values(ascending=False))
for c in taxo.CLASSES:
    g = A[A.primary == c]
    if len(g):
        smp = g.sample(min(12, len(g)), random_state=0)
        print("##", c, "|", "; ".join(smp["clip"].str.replace("_stageii", "")))

# ---------------- LAFAN (xwbc human stream) ----------------
LDIR = "/var/tmp/alphamotion/xwbc_lafan/extracted/human_motion"
LAFAN_EXTRA = {"aiming": ["locomotion", "gesture_social"], "dance": ["dynamic"], "fallAndGetUp": ["dynamic", "ground_contact"],
               "fight": ["dynamic"], "fightAndSports": ["dynamic"], "ground": ["ground_contact"], "jumps": ["dynamic"],
               "multipleActions": ["locomotion", "dynamic", "idle_posture"], "obstacles": ["locomotion", "dynamic"],
               "push": ["dynamic"], "pushAndFall": ["dynamic", "ground_contact"], "pushAndStumble": ["dynamic"],
               "run": ["locomotion"], "sprint": ["locomotion"], "walk": ["locomotion"]}
LAFAN_CONTACT = {"fallAndGetUp": ["hands_ground", "knees", "hips", "back"], "ground": ["hands_ground", "knees", "hips", "back"],
                 "pushAndFall": ["hands_ground", "hips", "back"]}
lrows = []
for f in sorted(glob.glob(LDIR + "/*.npz")):
    d = np.load(f, allow_pickle=True)
    T = d["body_pos_w"].shape[0]; fps = int(d["fps"][0])
    stem = os.path.basename(f)[:-4]
    act = re.sub(r"\d+$", "", stem.split("_subject")[0])
    subj = "subject" + stem.split("_subject")[1]
    cls = set(LAFAN_EXTRA.get(act, ["other"]))
    con = set(LAFAN_CONTACT.get(act, ["feet_only"]))
    lrows.append(dict(clip=stem, action=act, subject=subj, frames=T, fps=fps, hours=T / fps / 3600,
                      classes="|".join(sorted(cls)), primary=taxo.primary(cls), contacts="|".join(sorted(con)),
                      has_object_implied=False))
Lf = pd.DataFrame(lrows)
Lf.to_csv(os.path.join(OUT, "lafan_clip_labels.csv"), index=False)
print("LAFAN", len(Lf), Lf.hours.sum(), Lf.fps.unique(), Lf.subject.nunique())
print(Lf.groupby("primary").hours.sum().round(3))
