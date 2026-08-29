"""Export SMPL-X params for the speed-token demo windows (same seed/order).

    export_smpl_params.py          (env: ardy)

The cache's human_smpl local6d IS the SMPL skeleton's per-joint local rotation
(order matches SMPL-X body 22 exactly: Pelvis..R_Wrist), so no fitting needed:
local6d -> axis-angle, root_world -> transl. Renders happen in the genmo-gem env
(render_smpl_source.py) because ardy has no smplx.

Output: scratchpad smpl_params_{fam}.npz {global_orient[T,3], body_pose[T,63],
transl[T,3], fam, clip}
"""
from __future__ import annotations
import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v5_common as V                                                # noqa: E402
from common import rot6d_to_matrix                                   # noqa: E402
from speed_token_rollout import EMB, FAMS                            # noqa: E402
from scipy.spatial.transform import Rotation                         # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else "."


def main():
    T = V.T_WIN
    kd = V.KeyData(V.AMASS, [EMB], device="cpu")
    l6 = torch.load(f"{V.AMASS}/{EMB}_paired_local6d.pt", map_location="cpu")
    rw = torch.load(f"{V.AMASS}/{EMB}_root_world.pt", map_location="cpu")
    g = np.random.default_rng(23)                       # SAME seed/order as the demo
    for fam in FAMS:
        st, cid = kd.sample_starts(1, g, "val", span=T, families=[fam])
        s0 = int(st[0]); cname = kd.clips[int(cid[0])]["name"]
        R = rot6d_to_matrix(l6[s0:s0 + T].double()).numpy()          # [T,22,3,3]
        aa = Rotation.from_matrix(R.reshape(-1, 3, 3)).as_rotvec().reshape(T, 22, 3)
        np.savez(f"{OUT}/smpl_params_{fam}.npz",
                 global_orient=aa[:, 0].astype(np.float32),
                 body_pose=aa[:, 1:22].reshape(T, 63).astype(np.float32),
                 transl=(rw[s0:s0 + T].numpy() / 100.0).astype(np.float32),
                 fam=fam, clip=cname)
        print(f"[{fam}] {cname} -> {OUT}/smpl_params_{fam}.npz", flush=True)


if __name__ == "__main__":
    main()
