"""Keep only OmniContact's own labels from the processed G1 npz (for cross-check), then
the caller deletes npz/.  Out: raw/omnicontact/g1_labels/<domain>/<case>/<capture_id>.npz with
contact_info [T,4,1] (left_ankle,right_ankle,left_wrist,right_wrist), fps, object_pos_w,
object_quat_w (wxyz), base_pos_w, and the frame count, plus a summary json."""
from surface_retargeting.config import CFG as SRCFG  # machine paths: configs/sr.toml
import json
import sys
from pathlib import Path

import numpy as np

R = Path(f'{SRCFG.data_root}/raw/omnicontact')
out = R / "g1_labels"
summ = {}
for f in sorted((R / "npz").glob("*/*/*.npz")):
    dom, case = f.parts[-3], f.parts[-2]
    cid = f.stem.replace("_with_contact", "")
    try:
        with np.load(f, allow_pickle=False) as z:
            keep = {k: z[k] for k in ("contact_info", "fps", "object_pos_w", "object_quat_w", "base_pos_w") if k in z.files}
            keys = list(z.files)
        dst = out / dom / case / f"{cid}.npz"
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dst, **keep)
        c = keep.get("contact_info")
        summ[f"{dom}/{case}/{cid}"] = {"frames": int(len(c)) if c is not None else None,
                                       "fps": float(np.asarray(keep["fps"]).reshape(-1)[0]) if "fps" in keep else None,
                                       "contact_frac_lA_rA_lW_rW": [round(float(x), 3) for x in c.reshape(len(c), -1).mean(0)] if c is not None else None,
                                       "keys": keys}
    except Exception as e:
        summ[f"{dom}/{case}/{cid}"] = {"error": str(e)}
(out / "summary.json").write_text(json.dumps(summ, indent=1))
print(len(summ), "files;", sum("error" in v for v in summ.values()), "errors")
