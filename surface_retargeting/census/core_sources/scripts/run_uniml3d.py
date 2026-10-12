import sys, os, re, json
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd
import taxo

OUT = "/home/arenalabs/umr_factory_data/diversity/uniml3d_mixamo"
os.makedirs(OUT, exist_ok=True)
REM = "/var/tmp/alphamotion/umr_factory/uniml3d_reload/remote"
c = pd.read_csv(f"{REM}/clips.csv")
c["hours"] = c.duration_sec / 3600
S = os.path.expanduser("~/.cache/huggingface/hub/datasets--Linzhan--UniML3D/snapshots/053e8b4b06b6a8ba717a030e675dd5b7fd5be92e")
local_mix = {f[:-8] for f in os.listdir(f"{S}/features/mixamo/motions")} if os.path.isdir(f"{S}/features/mixamo/motions") else set()
# UniMate training-set skip list (stage-4 filter) = metadata.txt total 2162
PROP_RX = re.compile(r"(?<![a-z])(w briefcase|briefcase|rifle|pistol|gun|sword|shield|box|bat|ball|axe|torch|"
                     r"bow|spear|staff|katana|chair|barrel|carry|carrying|pick(ing)? up|lift(ing)?|throw\w*|"
                     r"push(ing)?|pull(ing)?|mutant|cpr|zombie|drink\w*|phone|guitar|drum\w*|sit\w*)(?![a-z])")
rows = []
for r in c.itertuples(index=False):
    if r.dataset == "truebones":
        continue
    stem = r.clip if r.dataset == "mixamo" else r.clip.split("-", 1)[1]
    text = f"{stem} | {r.caption}" if r.dataset == "mixamo" else f"{r.caption}"
    cls, con, objs = taxo.classify(text)
    nm = taxo.norm(stem)
    prop_in_name = bool(re.search(r"(?<![a-z])(w briefcase|briefcase|rifle|pistol|gun|sword|shield|box|ball|axe|torch|"
                                  r"bow|spear|katana|barrel|guitar|drum|phone)(?![a-z])", nm)) if r.dataset == "mixamo" else False
    rows.append(dict(dataset=r.dataset, clip=r.clip, object_type=r.object_type, category=r.category, frames=r.frames,
                     fps=r.fps, hours=r.hours, num_joints=r.num_joints, classes="|".join(sorted(cls)),
                     primary=taxo.primary(cls), contacts="|".join(sorted(con)), object_cats="|".join(sorted(objs)),
                     prop_in_name=prop_in_name, caption=r.caption,
                     local_feature_npz=(r.dataset == "mixamo" and f"{r.clip}" in local_mix)))
U = pd.DataFrame(rows)
U.to_csv(os.path.join(OUT, "uniml3d_clip_labels.csv.gz"), index=False, compression="gzip")
mix = U[U.dataset == "mixamo"]
obj_b = U[(U.dataset == "objaverse") & (U.category == "bipedal")]
print("mixamo", len(mix), mix.hours.sum(), "local feature npz", mix.local_feature_npz.sum(), "prop_in_name", mix.prop_in_name.sum(),
      mix[mix.prop_in_name].hours.sum())
print(mix.groupby("primary").hours.sum().round(3).sort_values(ascending=False))
print("objaverse bipedal", len(obj_b), obj_b.hours.sum())
print(obj_b.groupby("primary").hours.sum().round(3).sort_values(ascending=False))
for cl in taxo.CLASSES:
    g = mix[mix.primary == cl]
    if len(g):
        print("##", cl, "|", "; ".join(g.sample(min(12, len(g)), random_state=0)["clip"]))
