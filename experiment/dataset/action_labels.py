"""Extract coarse action-family labels from clip metadata.

Sources we see in the mixed dataset:

  Kimodo      — rich captions ("A person sprints forward") — keyword regex
  HY Motion   — descriptive filenames ("front_kick__xxxxx") — keyword regex
  AMASS       — generic subject IDs (no action info in sample_name). We fall
                back to a per-subset-file heuristic (e.g. dancedb → dance,
                kit → generic_locomotion).

Labels cover a 12-family taxonomy that captures the motion FAMILIES most
likely to form visually distinct clusters in a pair-tensor PCA:

  walk        — walking / strolling / pacing
  run         — running / sprinting / jogging
  jump        — jumping / hopping / leaping
  fight       — kicks, punches, martial arts, boxing
  dance       — all dance styles
  sit         — sitting / kneeling
  gesture     — pointing, waving, clapping (mostly upper body)
  manipulate  — picking up / throwing / handling objects
  stretch     — bowing / rotating / bending / stretching
  sport       — sport-specific (golf, tennis, climbing) — rare
  locomotion  — generic movement (used when we know "moves around" but
                don't know run vs walk — weak AMASS label)
  other       — we could not infer anything useful

We keep the taxonomy coarse so clusters have at least 50+ samples each.
"""

from __future__ import annotations

import re


# ---- Keyword map (priority-ordered; first match wins) -------------------
KEYWORD_MAP = [
    # fight first to catch "kick" before "walk"-ish words
    ("fight",      r"kick|punch|box|fight|martial|karate|combat|strike|bruce\s*lee|kung"),
    ("dance",      r"danc|ballet|waltz|tango|salsa|hip.?hop|charleston|capoeira|breakdance"),
    ("run",        r"run|sprint|jog|dash"),
    ("jump",       r"jump|hop|leap|skip(?!\s+up)|bound|vault"),
    ("sit",        r"sit|kneel|crouch|squat|lie\s+down|lay\s+down"),
    ("manipulate", r"pick\s+up|lift|throw|catch|carry|hold|grab|handle|place|put\s+down|drink|eat"),
    ("gesture",    r"wave|point|clap|nod|bow|shake|scratch|cross|fold\s+arm|salute"),
    ("stretch",    r"stretch|bend|rotat|twist|reach|yawn|arch|tilt"),
    ("sport",      r"golf|tennis|basketball|soccer|football|baseball|climb|swim|skate|ski|surf|row|bowling|split"),
    ("walk",       r"walk|stroll|pace|march|step|stride|amble|wander|cross"),
    ("locomotion", r"move|turn|circle|zig.?zag|navigate"),
]

# Pre-compile
KEYWORD_RE = [(lbl, re.compile(p, re.IGNORECASE)) for lbl, p in KEYWORD_MAP]


# ---- AMASS subset file → coarse label fallback --------------------------
# Keys are the `_soma77.npz` file stems, values are a default family.
AMASS_SUBSET_DEFAULT = {
    "dancedb":     "dance",
    "hdm05":       "other",           # wide variety
    "kit":         "locomotion",      # heavy walk+run mix
    "ekut":        "walk",            # dominated by walking studies
    "bmlmovi":     "other",
    "bmlrub":      "other",
    "cmu":         "other",           # too many actions, per-subject lookup would be needed
    "totalcapture": "other",
    "humaneva":    "walk",
    "mpi":         "other",
    "tcdhands":    "gesture",
    "ssm":         "other",
    "acccad":      "other",
}


def label_from_text(text: str) -> str | None:
    """Run keyword regex against a free-form text field (caption, filename).
    Returns a family label, or None if nothing matched."""
    if not text:
        return None
    for lbl, rex in KEYWORD_RE:
        if rex.search(text):
            return lbl
    return None


def label_kimodo(sample_name: str) -> str:
    """Kimodo samples have caption-style names."""
    lbl = label_from_text(sample_name)
    return lbl if lbl is not None else "other"


def label_hy_motion(sample_name: str) -> str:
    """HY Motion samples have descriptive filenames with underscores."""
    norm = sample_name.replace("_", " ")
    lbl = label_from_text(norm)
    return lbl if lbl is not None else "other"


def label_amass(subset_file_stem: str, subset: str, sample_name: str) -> str:
    """AMASS: first try the sample_name / subset strings (rarely hits),
    then fall back to the subset_file default.
    """
    # Rare hits: some KIT samples have action words in the filename.
    for txt in (sample_name, subset):
        lbl = label_from_text(str(txt))
        if lbl is not None:
            return lbl
    stem = subset_file_stem.replace("_soma77", "").lower()
    return AMASS_SUBSET_DEFAULT.get(stem, "other")


def label_row(source_name: str, subset_file_stem: str,
              subset: str, sample_name: str) -> str:
    source_name = str(source_name).lower()
    if source_name == "kimodo":
        return label_kimodo(str(sample_name))
    if source_name == "hy_motion":
        return label_hy_motion(str(sample_name))
    if source_name == "amass":
        return label_amass(subset_file_stem, str(subset), str(sample_name))
    if source_name == "100style":
        return label_from_text(str(sample_name)) or "other"
    return "other"


# Expose labels for diagnostics
ALL_LABELS = (
    "walk", "run", "jump", "fight", "dance", "sit",
    "gesture", "manipulate", "stretch", "sport", "locomotion", "other",
)


if __name__ == "__main__":
    # Self-test on representative strings
    cases = [
        ("kimodo", "A person sprints forward",       "run"),
        ("kimodo", "A person performs a karate kick", "fight"),
        ("kimodo", "A person jumps forward",          "jump"),
        ("kimodo", "A person dances slowly",          "dance"),
        ("kimodo", "A person drinks from a cup",      "manipulate"),
        ("kimodo", "A person waves hello with both hands", "gesture"),
        ("kimodo", "A person bows forward",           "gesture"),
        ("kimodo", "A person rotates the upper body", "stretch"),
        ("kimodo", "A person sits down",              "sit"),
        ("kimodo", "A person walks briskly",          "walk"),
        ("hy_motion", "boxing_combo__00000001_000_c0", "fight"),
        ("hy_motion", "front_kick__00000001_000_c0",   "fight"),
        ("hy_motion", "fold_arms__00000001_000_c0",    "gesture"),
        ("hy_motion", "fight_like_bruce_lee__00000001_000_c0", "fight"),
        ("hy_motion", "do_a_split__00000001_000_c0",   "sport"),
    ]
    fails = 0
    for src, name, expected in cases:
        got = label_row(src, "dummy", "dummy", name)
        tag = "OK" if got == expected else "FAIL"
        if tag == "FAIL":
            fails += 1
        print(f"  [{tag}] {src:10s} {name[:50]:50s} → got={got:<10s} expected={expected}")
    print(f"\n{'PASS' if fails == 0 else f'FAIL ({fails} wrong)'}")
