from surface_retargeting.config import CFG as SRCFG  # machine paths: configs/sr.toml
import sys, json, os, re, collections
import pandas as pd, pyarrow.parquet as pq
from surface_retargeting.census.core_sources.scripts import taxo

OUT = f'{SRCFG.data_root}/diversity/bones_seed'
os.makedirs(OUT, exist_ok=True)
df = pq.read_table(f'{SRCFG.factory_root}/sources/bones_meta_v004.parquet').to_pandas()
df["hours"] = df.move_duration_frames / 120 / 3600

PKG_ALWAYS = {("Dances", "Dancing"): ["dynamic"], ("Sport", "Martial Arts"): ["dynamic"],
              ("Locomotion", "Stunts"): ["dynamic"]}
PKG_FALLBACK = {"Communication": ["gesture_social"], "Dances": ["dynamic"], "Sport": ["dynamic"],
                "Locomotion": ["locomotion"], "Interactions": ["upper_body_manipulation"],
                "Everyday": ["upper_body_manipulation"], "Gaming": ["upper_body_manipulation"], "Other": ["idle_posture"]}
WBO_PROPS = re.compile(r"big|crate|platform|ladder|^heavy object$|(^|, ?)heavy object")
WBO_NAME = re.compile(r"climb|come up|come down|lift crate|push obstacle|jump (on|off|from)|box jump|ladder|kick ball|"
                      r"dribbl|on the edge|walking on edge|jump over obstacle")
cache = {}
rows = []
for r in df.itertuples(index=False):
    props = (r.content_props or "0").strip()
    props = "" if props == "0" else props
    bp = r.content_body_position or ""
    bp = ", ".join(x for x in re.split(r",\s*", bp) if x.strip() not in ("standing", ""))
    name = re.sub(r"_M$", "", r.content_name or "")
    text = " | ".join(x for x in [name, r.content_type_of_movement or "", bp, props] if x)
    key = (text, r.package, r.category)
    if key not in cache:
        cls, con, _ = taxo.classify(text, PKG_ALWAYS.get((r.package, r.category), []))
        nt = taxo.norm(name + " " + (r.content_type_of_movement or ""))
        pn = taxo.norm(props).strip()
        cls.discard("whole_body_object")
        if WBO_PROPS.search(pn) or WBO_NAME.search(nt):
            cls.add("whole_body_object"); cls.discard("other")
        pobj = {k for k, rx in taxo.OBJ_CATS.items() if rx.search(taxo.norm(props))} if props else set()
        manip = sorted(pobj & set(taxo.MANIP_OBJ_CATS))
        if re.search(r"sitting on (chair|box)|on a box|sitting on bench", bp) and not re.search(r"kneel|crawl|floor", name):
            cls.discard("ground_contact"); con.discard("knees"); cls.add("sitting_on_object")
        if r.category == "Gestures" and not props:
            cls.discard("upper_body_manipulation")    # pantomime gestures without a prop
        if cls <= {"other"} or cls == {"idle_posture"} and r.package in ("Communication",):
            cls = (cls - {"other"}) | set(PKG_FALLBACK.get(r.package, ["other"]))
        if not props:
            con.discard("object_hands")
        if manip and ({"upper_body_manipulation", "whole_body_object"} & cls):
            con.add("object_hands")
        if len(con) > 1: con.discard("feet_only")
        if not con: con.add("feet_only")
        cache[key] = (cls, con, pobj, manip)
    cls, con, pobj, manip = cache[key]
    rows.append(dict(move_name=r.move_name, is_mirror=bool(r.is_mirror), package=r.package, category=r.category,
                     hours=r.hours, fps=120, classes="|".join(sorted(cls)), primary=taxo.primary(cls),
                     contacts="|".join(sorted(con)), props=props, has_object=bool(props),
                     object_manipulated=bool(manip), object_cats="|".join(sorted(pobj)),
                     actor=r.actor_uid, gender=r.actor_gender, height_cm=r.actor_height_cm,
                     style=r.content_uniform_style, body_position=r.content_body_position,
                     type_of_movement=r.content_type_of_movement))
L = pd.DataFrame(rows)
L.to_csv(os.path.join(OUT, "bones_clip_labels.csv.gz"), index=False, compression="gzip")
print(L.shape)
o = L[~L.is_mirror]; m = L[L.is_mirror]
print("orig h", o.hours.sum(), "mirror h", m.hours.sum())
print("primary (orig h):"); print(o.groupby("primary").hours.sum().round(2).sort_values(ascending=False))
ml = {c: o[o.classes.str.contains(c)].hours.sum() for c in taxo.CLASSES}
print("multilabel (orig h):", {k: round(v, 2) for k, v in ml.items()})
import random
random.seed(1)
for c in taxo.CLASSES:
    g = o[o.primary == c]
    if len(g):
        s = g.sample(min(12, len(g)), random_state=0)
        print("##", c, "|", "; ".join(s["move_name"].str.replace(r"__A\d+$", "", regex=True)))
