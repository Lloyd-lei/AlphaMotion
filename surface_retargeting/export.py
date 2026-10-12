"""Per (clip, robot) output in the DATA_SPEC_v2 layout (computed on the GPU from the final qpos).

  qpos [T,nq] float32           root pose + joints, floor-relative (calibrated ground = z 0), fps stored
  floor_offset_removed          calibrated floor height in the solver's raw world (metres)
  dense-point contact (sparse, CSR over frames, indices into the robot's static dense surface asset):
    contact_ptr [T+1], contact_idx uint16, contact_partner uint8 (0 ground, 1 object)   points within [-0.5, 1] mm
    near_ptr [T+1], near_idx uint16, near_partner uint8, near_dist_q uint8                 points within 2 cm,
        distance quantised d = (q * 0.1 mm) - 0.5 mm   (q = 0..205)
  region level [T,13]: ground_contact / object_contact (hysteresis enter < 1 mm, leave > 2 mm, >= 3 frames),
    ground_slip_speed [T,R] float16 m/s: robot region slip speed between t-1 and t (umr_gpu/slip.py ruler: min
        tangential speed over the region's dense points within [-0.5, 1] mm at both frames = the contact / pivot
        point; nan = not in contact at both frames).  Rolling heel->toe and pivoting about the contact point are 0.
    human_ground_slip_speed [T,R] float16: the same ruler on the human SMPL-X support points (when given)
    ground_slip [T,R]: robot region slides (> 1 cm/s) while the human region does not (or much faster) -> bit 32
    ground_slip_follow [T,R]: robot region slides together with the human region (lloyd 10-11) -> bit 256 (kept)
    ground_pen_mm / object_pen_mm (float16, max(0, -min clearance) over dense + hull points)
  human (same ruler: SMPL-X surface slots, same floor): human_ground_contact, human_object_contact, human clearances
  flags [T] (bitmask, FLAG_BITS; exclude a frame iff flags & DROP_MASK != 0 -- bit 256 is informational and NOT
  excluded) + QC summary json (frames, flagged = excluded frames, kept_frames, follow_human_slip_frames, flags_hist,
  ground_slip_speed_max / p99 (m/s, kept frames), slip_rule thresholds, flag_bits, drop_mask); with qref (the UMR solution the strict pass started from) the QC
  also reports temporal continuity: root step excess over UMR (cm/frame at 30 fps), joint step excess (rad/frame at
  30 fps), root offset from UMR (cm).
"""
from __future__ import annotations

import json

import numpy as np
import torch

from surface_retargeting.gpu import labels as L
from surface_retargeting.strict.strict import seg_min
from surface_retargeting.strict.strict_fp import hysteresis_t, drop_short_t
from surface_retargeting.gpu import slip as SL
from surface_retargeting.gpu.slip import DROP_MASK

FLAG_BITS = {1: 'ground penetration > 0.5 mm', 2: 'object penetration > 0.5 mm', 4: 'self penetration',
             8: 'required ground contact not solid', 16: 'required object contact not solid',
             32: 'robot contact slip: a robot region in contact slides > 1 cm/s (slip.py ruler, contact/pivot point) while the same human region does not slide (<= 2 cm/s) or the robot slides faster than max(2 cm/s, 1.5 x human); object latch moved',
             64: 'lift: human region > 2 cm off but robot region <= 5 mm',
             128: 'temporal continuity: root / joint step exceeds the UMR step by > 1 cm / 0.05 rad / 0.1 rad per frame (30 fps)',
             256: 'follow_human_slip (informational, frame KEPT): a robot region slides > 1 cm/s together with the same human region (human > 2 cm/s, robot <= max(2 cm/s, 1.5 x human))'}

Q_STEP = 0.0001
Q_OFF = 0.0005


def _csr(mask, vals=None):
    """mask [T,N] bool -> ptr [T+1], idx, (vals)."""
    t, i = torch.nonzero(mask, as_tuple=True)
    cnt = torch.bincount(t, minlength=mask.shape[0])
    ptr = torch.cat([torch.zeros(1, dtype=torch.long, device=mask.device), torch.cumsum(cnt, 0)])
    return ptr, i, (None if vals is None else vals[t, i])


@torch.no_grad()
def export_clip(proj, qpos_world, floor, fps, flags, pose=None, human=None, chunk=4096, qref=None):
    """qpos_world [T,nq] (solver raw world), floor (float), pose [T,7] object (raw world) or None.
    human: dict with ground/object contact [T,R] and clearances (numpy).  Returns dict of numpy arrays."""
    dev = qpos_world.device
    T = qpos_world.shape[0]
    pts = proj.pts
    nd = pts.ndense
    R = L.NR
    zs, ds = [], []
    gpen = torch.zeros(T, R, device=dev); open_ = torch.zeros(T, R, device=dev)
    for s in range(0, T, chunk):
        q = qpos_world[s:s + chunk]
        n = q.shape[0]
        K = proj.kt.fk(q)
        fl = torch.full((n,), float(floor), device=dev)
        z, _, _ = proj.gauge.ground(K, fl) if proj.gauge is not None else (proj.pts.world(K)[..., 2] - float(floor), None, None)
        zr, _ = seg_min(z, pts.region)
        gpen[s:s + n] = torch.relu(-zr)
        zs.append(z[:, :nd])
        if proj.obj is not None and pose is not None:
            d, _, _ = proj.gauge.object(K, proj.obj, pose[s:s + n])
            dr, _ = seg_min(d, pts.region)
            open_[s:s + n] = torch.where(torch.isfinite(dr), torch.relu(-dr), torch.zeros_like(dr))
            ds.append(d[:, :nd])
    z = torch.cat(zs)
    d = torch.cat(ds) if ds else None
    # contact points / near points (sparse)
    cg = (z >= -0.0005) & (z <= 0.001)
    ng = z < 0.02
    out = {}
    if d is not None:
        co = (d >= -0.0005) & (d <= 0.001)
        no = d < 0.02
        cmask = torch.cat([cg, co], 1); nmask = torch.cat([ng, no], 1)
        dist = torch.cat([z, d], 1)
    else:
        cmask, nmask, dist = cg, ng, z
    ptr, idx, _ = _csr(cmask)
    out['contact_ptr'] = ptr.cpu().numpy().astype(np.int64)
    out['contact_idx'] = (idx % nd).cpu().numpy().astype(np.uint16)
    out['contact_partner'] = (idx // nd).cpu().numpy().astype(np.uint8)
    ptr, idx, val = _csr(nmask, dist)
    out['near_ptr'] = ptr.cpu().numpy().astype(np.int64)
    out['near_idx'] = (idx % nd).cpu().numpy().astype(np.uint16)
    out['near_partner'] = (idx // nd).cpu().numpy().astype(np.uint8)
    out['near_dist_q'] = torch.clamp(torch.round((val + Q_OFF) / Q_STEP), 0, 255).cpu().numpy().astype(np.uint8)
    # region-level labels from dense clearances
    reg = pts.region[:nd]
    zr, _ = seg_min(z, reg)
    reset = torch.zeros(1, T, dtype=torch.bool, device=dev); reset[0, 0] = True
    gc = drop_short_t(hysteresis_t(zr[None], 0.001, 0.002, reset), 3, reset)[0]
    K_all = None
    # slip (slip.py ruler, backward difference t-1 -> t, same as the strict pass's output gate); the old gauge took the
    # forward difference median over whatever points were in the band at t and t+1 (1-2 points of a foot sweeping
    # through the band, or root jumps of the frame-parallel engine -> 14-34 m/s)
    rs = SL.robot_slip_speed(proj, qpos_world, float(floor), fps, chunk=chunk)
    hs = None
    if human is not None and human.get('ground_slip_speed') is not None:
        hs = torch.as_tensor(np.asarray(human['ground_slip_speed'], np.float32), device=dev)
    gslip, gfollow = SL.slip_rule(rs, hs if hs is not None else torch.full_like(rs, float('nan')))
    speed = rs
    out.update(ground_contact=gc.cpu().numpy(), ground_slip=gslip.cpu().numpy(), ground_slip_follow=gfollow.cpu().numpy(),
               ground_slip_speed=speed.cpu().numpy().astype(np.float16),
               ground_pen_mm=(gpen * 1000).cpu().numpy().astype(np.float16), ground_clearance_mm=(zr * 1000).cpu().numpy().astype(np.float16))
    if d is not None:
        dr, _ = seg_min(d, reg)
        oc = drop_short_t(hysteresis_t(dr[None], 0.001, 0.002, reset), 3, reset)[0]
        out.update(object_contact=oc.cpu().numpy(), object_pen_mm=(open_ * 1000).cpu().numpy().astype(np.float16),
                   object_clearance_mm=(torch.clamp(dr, max=10.0) * 1000).cpu().numpy().astype(np.float16))
    if human is not None:
        for k, v_ in human.items():
            out['human_' + k] = np.asarray(v_).astype(np.float16) if k == 'ground_slip_speed' else np.asarray(v_)
    fl_np = flags.cpu().numpy() if torch.is_tensor(flags) else np.asarray(flags)
    out['flags'] = fl_np.astype(np.int32)
    qp = qpos_world.clone(); qp[:, 2] -= float(floor)
    out['qpos'] = qp.cpu().numpy().astype(np.float32)
    out['floor_offset_removed'] = np.array(float(floor), np.float32)
    out['floor_height'] = np.array(0.0, np.float32)
    out['fps'] = np.array(float(fps), np.float32)
    # QC summary
    hg = human.get('ground_contact') if human else None
    kept = (fl_np & DROP_MASK) == 0
    sp_np = speed.cpu().numpy(); spk = sp_np[kept]; spk = spk[np.isfinite(spk)]
    qc = dict(frames=T, flagged=int((~kept).sum()), kept_frames=int(kept.sum()),
              follow_human_slip_frames=int(((fl_np & SL.FLAG_FOLLOW) > 0).sum()),
              flags_hist={str(b): int(((fl_np & b) > 0).sum()) for b in (1, 2, 4, 8, 16, 32, 64, 128, 256)},
              ground_pen_max_mm=float(gpen.max() * 1000), object_pen_max_mm=float(open_.max() * 1000),
              ground_slip_frac_of_contact=float(gslip.sum() / max(int(gc.sum()), 1)),
              ground_follow_frac_of_contact=float(gfollow.sum() / max(int(gc.sum()), 1)),
              ground_slip_speed_kept_max=float(spk.max()) if spk.size else None,
              ground_slip_speed_kept_p99=float(np.percentile(spk, 99)) if spk.size else None,
              ground_slip_speed_max=float(np.nanmax(sp_np)) if np.isfinite(sp_np).any() else None,
              slip_rule=dict(v_slip=SL.V_SLIP, v_human=SL.V_HUMAN, v_abs_tol=SL.V_ABS_TOL, k_rel=SL.K_REL))
    if hg is not None:
        hgt = torch.as_tensor(hg, device=dev)
        solid = (zr >= -0.0005) & (zr <= 0.001)
        qc['human_contact_robot_solid_frac'] = float((hgt & solid).sum() / max(int(hgt.sum()), 1))
    if qref is not None:
        qs = qpos_world.double(); qr = torch.as_tensor(qref, device=qs.device).double()
        s30 = float(fps) / 30.0
        if T > 1:
            ep = (torch.diff(qs[:, :3], dim=0) - torch.diff(qr[:, :3], dim=0)).norm(dim=-1) * s30 * 100
            ej = (torch.diff(qs[:, 7:], dim=0) - torch.diff(qr[:, 7:], dim=0)).abs().amax(-1) * s30
            st = torch.diff(qs[:, :3], dim=0).norm(dim=-1) * s30 * 100
            q_ = lambda x, p: float(torch.quantile(x.float(), p))
            qc['continuity'] = dict(root_excess_cm_p50=round(q_(ep, 0.5), 3), root_excess_cm_p99=round(q_(ep, 0.99), 3),
                                    root_excess_cm_max=round(float(ep.max()), 3), joint_excess_rad_max=round(float(ej.max()), 4),
                                    root_step_cm_max=round(float(st.max()), 2),
                                    root_offset_vs_umr_cm_max=round(float((qs[:, :3] - qr[:, :3]).norm(dim=-1).max() * 100), 2))
    qc['flag_bits'] = {str(k): v for k, v in FLAG_BITS.items()}
    qc['drop_mask'] = DROP_MASK
    out['qc_json'] = np.array(json.dumps(qc))
    return out, qc
