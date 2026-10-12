"""Collect per-clip qa.json into /home/arenalabs/umr_factory_data/manifest_sources.json."""
import glob
import json
import time
from pathlib import Path

ROOT = Path("/home/arenalabs/umr_factory_data")
LIC = {
    "OmniContact": "CC BY-NC 4.0 (lightcone02/OmniContact-Dataset); non-commercial, attribution",
    "HiPHI": "ModalityNet Open Research License v1.0 (Noitom); non-commercial only, no redistribution of raw data, "
             "internal R&D supporting a commercial offering counts as commercial (sec.1), no competing dataset",
}
SMPLX_LIC = "SMPL-X model licence (MPI, non-commercial) applies to the SMPL-X parameterisation"
clips, failed = [], []
for q in sorted(glob.glob(str(ROOT / "smplx/omnicontact/*/*/*/qa.json")) + glob.glob(str(ROOT / "smplx/hiphi/data/*/*/*/qa.json"))):
    r = json.loads(Path(q).read_text())
    if not r.get("ok"):
        failed.append({"clip_id": r.get("clip_id"), "error": r.get("error")})
        continue
    objs = r.get("objects", [])
    clips.append({
        "clip_id": r["clip_id"], "source": r["source"], "path": str(Path(q).parent),
        "frames": r["frames"], "fps": r["fps"], "duration_s": round(r["frames"] / r["fps"], 2),
        "has_object": bool(objs), "object_ids": [o["object_id"] for o in objs],
        "object_categories": [o.get("category") for o in objs] if r["source"] == "HiPHI" else None,
        "floor_estimate_y_m": r["floor"]["floor_y_m"], "up_axis": "y",
        "floor_qa": {k: r["floor"][k] for k in ("static_frac", "sink_gt1cm_frac", "sink_max_cm", "above_gt3cm_frac",
                                                  "static_lowest_p10_rel_median_cm")},
        "object_rest_rel_floor_cm": [o.get("obj_resting_rel_floor_cm_median") for o in objs],
        "bvh_layout": r.get("bvh_layout"), "ik_residual_cm": r.get("ik_residual_cm"),
        "actor_id": r.get("actor_id"), "gender_meta": (r.get("actor_metadata") or {}).get("gender"),
        "smplx_gender_used": "neutral", "betas": r.get("betas"),
        "problems": r.get("problems", []),
        "license": LIC[r["source"]] + "; " + SMPLX_LIC,
    })
m = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "n_clips": len(clips), "n_failed": len(failed),
     "conventions": "SMPL-X neutral, flat_hand_mean=True, fingers/jaw/eyes zero (hiphi2smplx body-only); world Y-up metres; "
                    "object OBJ in cm, v_world = R(q_xyzw) @ (0.01 v) + p; floor_estimate = median lowest-vertex height "
                    "over frames whose lowest vertex moves < 10 cm/s",
     "failed": failed, "clips": clips}
(ROOT / "manifest_sources.json").write_text(json.dumps(m, indent=1))
print(len(clips), "clips", len(failed), "failed")
