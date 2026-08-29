"""JOB 2a: BONES-SEED text-pair manifest for P2 (text-conditioned generator).

One JSONL row per soma_uniform BVH clip on disk:

  name        clip id (filename stem, e.g. reaching_far_R_001__A033)
  bvh         absolute BVH path (soma_uniform)
  g1_csv      absolute GMR G1 CSV path if the twin exists, else null
  session     BVH session dir (capture date)
  caption_mech  mechanical caption from the filename (underscores -> words,
                L/R -> left/right, actor suffix + take index stripped)
  family      R7 6-family action regex on the FILENAME (deterministic; "other" if none)
  family_text same regex, filename + short description (fewer "other")
  captions    natural-language descriptions from seed_metadata_v004.csv (1-4 per clip)
  short / technical / category / package / is_mirror  straight from the metadata
  events      temporal segments from seed_metadata_v002_temporal_labels.jsonl
  duration_frames  from the metadata (BVH native fps)

The metadata CSV covers the full corpus, so nearly every clip gets real captions;
the filename caption is the fallback and a second view for retrieval.
"""
from __future__ import annotations
import os, csv, json, re, glob, argparse, sys, collections

BONES = ('/home/arenalabs/Desktop/"be water, robot"/alpha_motion2/wbc_expert/'
         'data/raw/bones_seed_full')
CACHE_OUT = "/var/tmp/alphamotion/topo_exp/cache_bones_body"

# --- R7 action-family regex (scratchpad r7/r7_common.py, verbatim) -----------
FAMILIES = ["dance", "jump", "run", "walk", "sit", "reach"]
_PATS = {
    "dance": r"danc|salsa|cha_?cha|tango|ballet|jazz|samba|swing_danc",
    "jump": r"jump|hop|leap|skip",
    "run": r"run|jog|sprint",
    "walk": r"walk|stroll|march|stride",
    "sit": r"sit|chair|stool",
    "reach": r"reach|pick|grasp|grab|place",
}


def action_family(text):
    low = text.lower()
    for f in FAMILIES:
        if re.search(_PATS[f], low):
            return f
    return "other"


# --- mechanical caption from the filename ------------------------------------
_WORD = {"l": "left", "r": "right", "ff": "", "bb": ""}


def mech_caption(name):
    """reaching_far_R_001__A033 -> 'reaching far right';
    come_up_50cm_box_001__A359 -> 'come up 50cm box'."""
    s = re.sub(r"__A\d+.*$", "", name)          # actor suffix (+ _M mirror marker)
    s = re.sub(r"_\d+$", "", s)                 # take index
    toks = []
    for t in s.split("_"):
        if not t:
            continue
        t = _WORD.get(t.lower(), t) if len(t) <= 2 else t
        if t:
            toks.append(t.lower())
    return " ".join(toks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{CACHE_OUT}/text_manifest.jsonl")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # metadata CSV: filename -> row
    meta = {}
    csv.field_size_limit(1 << 24)
    with open(f"{BONES}/metadata/seed_metadata_v004.csv") as f:
        for row in csv.DictReader(f):
            meta[row["filename"]] = row

    # temporal event labels: filename -> [events]
    events = {}
    with open(f"{BONES}/metadata/seed_metadata_v002_temporal_labels.jsonl") as f:
        for line in f:
            d = json.loads(line)
            events[d["filename"]] = d["events"]

    n = n_meta = n_ev = 0
    fam_count = collections.Counter()
    with open(args.out, "w") as out:
        for sess in sorted(os.listdir(f"{BONES}/soma_uniform/bvh")):
            for bvh in sorted(glob.glob(f"{BONES}/soma_uniform/bvh/{sess}/*.bvh")):
                name = os.path.basename(bvh)[:-4]
                m = meta.get(name)
                csvp = f"{BONES}/g1/csv/{sess}/{name}.csv"
                caps = []
                if m:
                    caps = [m[f"content_natural_desc_{i}"].strip()
                            for i in range(1, 5) if m.get(f"content_natural_desc_{i}", "").strip()]
                mech = mech_caption(name)
                short = (m or {}).get("content_short_description", "").strip()
                fam = action_family(name)
                rec = {
                    "name": name, "bvh": bvh,
                    "g1_csv": csvp if os.path.exists(csvp) else None,
                    "session": sess,
                    "caption_mech": mech,
                    "family": fam,
                    "family_text": action_family(name + " " + short),
                    "captions": caps,
                    "short": short or None,
                    "technical": (m or {}).get("content_technical_description", "").strip() or None,
                    "category": (m or {}).get("category") or None,
                    "package": (m or {}).get("package") or None,
                    "is_mirror": (m or {}).get("is_mirror") == "True",
                    "duration_frames": int(float(m["move_duration_frames"])) if m else None,
                    "events": events.get(name),
                }
                out.write(json.dumps(rec) + "\n")
                n += 1; n_meta += bool(caps); n_ev += name in events
                fam_count[fam] += 1
    print(f"[manifest] {n} clips -> {args.out}")
    print(f"  with natural captions: {n_meta}  with temporal events: {n_ev}")
    print("  family (filename regex):", dict(fam_count))


if __name__ == "__main__":
    main()
