"""Fetch the OmniContact processed npz files whose labels are not yet in g1_labels/, one by one
(retrying), keep only the label arrays, delete each npz right away."""
from surface_retargeting.config import CFG as SRCFG  # machine paths: configs/sr.toml
import os
os.environ.pop("HF_HUB_OFFLINE", None)
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from huggingface_hub import HfApi, hf_hub_download

R = Path(f'{SRCFG.data_root}/raw/omnicontact')
TMP = R / "_npz_tmp"
api = HfApi()
files = [f for f in api.list_repo_files("lightcone02/OmniContact-Dataset", repo_type="dataset")
         if f.startswith("npz/") and f.endswith(".npz")]
todo = []
for f in files:
    p = Path(f)
    dom, case = p.parts[1], p.parts[2]
    cid = p.stem.replace("_with_contact", "")
    if not (R / "g1_labels" / dom / case / f"{cid}.npz").exists():
        todo.append((f, dom, case, cid))
print("npz in repo", len(files), "todo", len(todo), flush=True)


def one(item):
    f, dom, case, cid = item
    for attempt in range(5):
        try:
            lp = hf_hub_download("lightcone02/OmniContact-Dataset", f, repo_type="dataset", local_dir=str(TMP),
                                 cache_dir=f'{SRCFG.data_root}/hf_cache')
            with np.load(lp, allow_pickle=False) as z:
                keep = {k: z[k] for k in ("contact_info", "fps", "object_pos_w", "object_quat_w", "base_pos_w") if k in z.files}
            dst = R / "g1_labels" / dom / case / f"{cid}.npz"
            dst.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dst, **keep)
            os.remove(lp)
            return f, "ok"
        except Exception as e:
            err = repr(e)
            time.sleep(5 * (attempt + 1))
    return f, "FAIL " + err


with ThreadPoolExecutor(6) as ex:
    for f, s in ex.map(one, todo):
        if s != "ok":
            print(f, s, flush=True)
print("DONE", flush=True)
