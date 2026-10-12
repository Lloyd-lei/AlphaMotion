"""Human-object interaction inputs on the GPU, per clip with UMR's formulas (load_object_contact_source):

  object asset   unit-scale object (body frame) from the clip's MJCF; UMR's 1024 first-hit surface samples;
                 'scaled' mode -> the clip object is smpl_scale x asset (samples, SDF, collision all scale)
  object motion  UMR read_prop_motion (y-up -> z-up conversion, positions x smpl_scale)
  contact map    per frame and source slot: nearest object sample, offset vector, distance (< 5 mm snapped to 0);
                 active slots: distance <= 0.1 m, 128 closest (stable by slot index), as UMR's solve function
                 selects them.  Slots farther than the object's bounding radius + 0.1 m from its centre cannot be
                 active and are skipped (exactly equivalent for the solver).
"""
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import torch
import warp as wp

from surface_retargeting.gpu.geometry import ObjectSDF, mesh_verts

UMR = Path('/var/tmp/alphamotion/umr_factory/umr/UMR')
wp.init()


def _umr():
    for p in (str(UMR / 'scripts'), str(UMR)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import retarget_smpl_to_humanoid_surface_vector as M
    return M


def object_mesh_from_xml(xml_path, scale=1.0):
    """Body-frame mesh (V, F) of the object MJCF (all mesh geoms), meshes scaled by `scale` (UMR helpers)."""
    M = _umr()
    root = ET.parse(xml_path).getroot()
    M.make_mjcf_mesh_paths_absolute(root, Path(xml_path))
    M.scale_mjcf_mesh_assets(root, scale)
    m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    from scipy.spatial.transform import Rotation as Rot
    V, F, off = [], [], 0
    for g in range(m.ngeom):
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        v, f = mesh_verts(m, g)
        q = m.geom_quat[g]; Rg = Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        V.append(v @ Rg.T + m.geom_pos[g]); F.append(f + off); off += len(v)
    return np.concatenate(V), np.concatenate(F)


class ObjectAsset:
    def __init__(self, xml_path, device='cuda', samples=1024, seed=0, grid=96):
        M = _umr()
        self.xml = str(xml_path)
        self.name = Path(xml_path).stem
        V, F = object_mesh_from_xml(xml_path, 1.0)
        self.V, self.F = V, F
        self.sdf = ObjectSDF(V, F, device, grid=grid)
        self.points = torch.as_tensor(M.object_points_from_mjcf(Path(xml_path), samples, seed, 1.0), device=device, dtype=torch.float32)
        self.center = self.points.mean(0)
        self.radius = float((self.points - self.center).norm(dim=-1).max())

    @staticmethod
    def motion(prop_csv, frame_ids, scale, output_up='y', convert_y_up=True):
        M = _umr()
        mo = M.read_prop_motion(Path(prop_csv), np.asarray(frame_ids), output_up, convert_y_up, False, 0.0, 0.0, float(scale))
        return np.concatenate([mo['positions'], mo['quats_wxyz']], 1).astype(np.float64)       # [F,7]


@wp.kernel
def nn_kernel(src: wp.array2d(dtype=wp.vec3), opts: wp.array2d(dtype=wp.vec3), oc: wp.array(dtype=wp.vec3),
              orad: wp.array(dtype=float), act: float, dist: wp.array2d(dtype=float), ids: wp.array2d(dtype=wp.int32)):
    f, i = wp.tid()
    p = src[f, i]
    if wp.length(p - oc[f]) > orad[f] + act:
        dist[f, i] = 1.0e30
        ids[f, i] = -1
        return
    best = float(1.0e30)
    bi = int(-1)
    n = opts.shape[1]
    for k in range(n):
        d = wp.length(p - opts[f, k])
        if d < best:
            best = d
            bi = k
    dist[f, i] = best
    ids[f, i] = bi


def object_contact_maps(asset, src, pose8, args, kmax=128):
    """src [F,N,3] source slots (retarget frame, scaled), pose8 [F,8] (pos, quat wxyz, scale) ->
    dict(oc_slot [F,K] (-1 pad), oc_d [F,K], oc_vec [F,K,3], oc_pt [F,K,3]) as UMR selects them."""
    from surface_retargeting.gpu.kin import quat_to_mat
    dev = src.device
    F, N, _ = src.shape
    R = quat_to_mat(pose8[:, 3:7].float()); x = pose8[:, :3].float(); s = pose8[:, 7].float()
    opts = (R[:, None] @ (asset.points[None] * s[:, None, None])[..., None])[..., 0] + x[:, None]      # [F,P,3]
    oc = (R @ (asset.center[None] * s[:, None])[..., None])[..., 0] + x
    orad = asset.radius * s
    thr = float(args['object_contact_map_threshold']); snap = float(args['object_contact_map_snap_threshold'])
    dist = torch.empty(F, N, dtype=torch.float32, device=dev); ids = torch.empty(F, N, dtype=torch.int32, device=dev)
    st = wp.stream_from_torch(torch.cuda.current_stream()); wd = 'cuda:%d' % (dev.index or 0)
    wp.launch(nn_kernel, dim=(F, N), inputs=[wp.from_torch(src.float().contiguous(), dtype=wp.vec3),
              wp.from_torch(opts.contiguous(), dtype=wp.vec3), wp.from_torch(oc.contiguous(), dtype=wp.vec3),
              wp.from_torch(orad.contiguous()), thr + 0.01, wp.from_torch(dist), wp.from_torch(ids)], device=wd, stream=st)
    snapm = dist < snap
    dist = torch.where(snapm, torch.zeros_like(dist), dist)
    act = dist <= thr
    k = min(kmax, N)
    key = torch.where(act, dist.double() + torch.arange(N, device=dev, dtype=torch.float64)[None] * 1e-12, torch.full_like(dist, float('inf'), dtype=torch.float64))
    kv, ki = torch.topk(key, k, dim=1, largest=False)
    ok = torch.isfinite(kv)
    sl = torch.where(ok, ki, torch.full_like(ki, -1))
    pid = torch.gather(ids, 1, ki).long().clamp_min(0)
    pt = torch.gather(opts, 1, pid[..., None].expand(-1, -1, 3))
    sp = torch.gather(src.float(), 1, ki[..., None].expand(-1, -1, 3))
    vec = torch.where(torch.gather(snapm, 1, ki)[..., None], torch.zeros_like(sp), sp - pt)
    od = torch.gather(dist, 1, ki)
    z3 = torch.zeros_like(pt)
    return dict(oc_slot=sl, oc_d=torch.where(ok, od, torch.zeros_like(od)).double(), oc_vec=torch.where(ok[..., None], vec, z3).double(),
                oc_pt=torch.where(ok[..., None], pt, z3).double())
