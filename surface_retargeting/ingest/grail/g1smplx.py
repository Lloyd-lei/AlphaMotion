"""GPU batched G1-SMPL-X body (GRAIL overlay) — same template/posedirs scaling as UMR's smplx_model_loader."""
import sys
from pathlib import Path
import numpy as np
import torch

UMR = Path('/var/tmp/alphamotion/umr_factory/umr/UMR')
OVERLAY = UMR / 'sample_data/grail/g1_smplx_model'
sys.path.insert(0, str(UMR / 'scripts'))


class G1SMPLX:
    def __init__(self, device='cuda'):
        from smplx_model_loader import build_smplx_model
        from smplx.lbs import lbs  # noqa: F401
        m = build_smplx_model(OVERLAY, 'neutral', 1)
        self.device = device
        g = lambda t: t.detach().to(device=device, dtype=torch.float32)  # noqa: E731
        self.v_template = g(m.v_template)
        self.shapedirs = g(m.shapedirs)
        self.posedirs = g(m.posedirs)
        self.J_regressor = g(m.J_regressor)
        self.parents = m.parents.to(device)
        self.lbs_weights = g(m.lbs_weights)
        self.faces = np.asarray(m.faces, dtype=np.int64)
        w = m.lbs_weights.numpy()
        dom = w.argmax(1)
        # SMPL-X joints: 7 L_ankle, 8 R_ankle, 10 L_foot, 11 R_foot
        self.foot_idx = {'L': np.where(np.isin(dom, [7, 10]))[0], 'R': np.where(np.isin(dom, [8, 11]))[0]}
        self.dom = dom

    @torch.no_grad()
    def verts(self, poses, trans):
        """poses [N,165] axis-angle (SMPL-X order), trans [N,3] -> verts [N,V,3] (torch, device)."""
        from smplx.lbs import lbs
        p = torch.as_tensor(np.asarray(poses, np.float32), device=self.device).reshape(-1, 165)
        t = torch.as_tensor(np.asarray(trans, np.float32), device=self.device).reshape(-1, 3)
        n = p.shape[0]
        betas = torch.zeros(n, self.shapedirs.shape[-1], device=self.device)
        v, j = lbs(betas, p, self.v_template, self.shapedirs, self.posedirs, self.J_regressor, self.parents,
                   self.lbs_weights, pose2rot=True)
        return v + t[:, None, :], j + t[:, None, :]
