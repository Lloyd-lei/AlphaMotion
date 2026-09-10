"""通用精修器 (对照臂, 独立于 codex 那套): manifold_project.lm_fit 的扩展版.
新增 (1) 逐轴位置权重 w_pos [J,3] (平面约束只钉高度), (2) 方向约束 (脚底法线 / 脚前向轴, 绕轴扭转自由).
脚约束四档: point = 位置; plane = 高度 + 脚底法线; axis = 位置 + 前向轴; full = 位置 + 完整旋转.
坐标: 缓存约定 ardy Y-up (x 左 y 上 z 前), 根相对, 厘米. 根旋转透传 (root_R), 不解."""
import os, sys, json, numpy as np, torch
E = '/home/arenalabs/Desktop/"be water, robot"/pose_translator_design/exp_topo'; sys.path.insert(0, E)
import manifold_project as MP
from manifold_project import (_fk_axes, _skew, _slot_axes, _limits_open, _ancestor_mask, fit_angles, fk_from_angles)


def lm_fit2(target_R, spec, dof, q_init, rest=None, root_R=None, target_p=None,
            w_rot=None, w_pos=None, q_ref=None, w_ref=0.0, dirs=None, clamp=True,
            soft_margin=1.0, iters=25, lam0=1e-3, tol=1e-12, return_info=False):
    """= MP.lm_fit, 加: w_pos 可为 [J,3] (逐轴); dirs = [(j, a_local[3], t_world[B,3], w)] 方向残差 sqrt(w)(gR_j a - t)."""
    B, J = q_init.shape[0], q_init.shape[1]
    dev, dt = q_init.device, q_init.dtype
    parents = np.asarray(spec.parents)
    off = torch.as_tensor(np.asarray(spec.rest_offsets), dtype=dt, device=dev)
    A, active = _slot_axes(dof.to(dev, dt))
    if rest is None: rest = torch.eye(3, device=dev, dtype=dt).expand(J, 3, 3)
    rest = rest.to(dev, dt)
    root = int(np.where(parents < 0)[0][0])
    root_R = (target_R[:, root] if root_R is None else root_R).to(dev, dt)
    par = active.clone(); par[root] = False
    pj, ps = par.nonzero(as_tuple=True); P = int(pj.numel())
    q = q_init.clone()
    if P == 0: return (q, {"iters": 0, "cost": 0.0}) if return_info else q
    lo, hi, _ = _limits_open(dof.to(dev, dt), soft_margin); lo_p, hi_p = lo[pj, ps], hi[pj, ps]
    if not clamp: lo_p = torch.full_like(lo_p, -float("inf")); hi_p = torch.full_like(hi_p, float("inf"))
    sup = torch.as_tensor(_ancestor_mask(parents), device=dev)[:, pj]          # [J,P]
    ones = torch.ones(J, device=dev, dtype=dt)
    w_rot = ones if w_rot is None else (ones * w_rot if np.isscalar(w_rot) else w_rot.to(dev, dt))
    if w_pos is None: w_pos = torch.zeros(J, 3, device=dev, dtype=dt)
    elif np.isscalar(w_pos): w_pos = torch.full((J, 3), float(w_pos), device=dev, dtype=dt)
    else:
        w_pos = w_pos.to(dev, dt)
        if w_pos.dim() == 1: w_pos = w_pos[:, None].expand(J, 3)
    use_pos = bool((w_pos > 0).any()) and target_p is not None
    sr = w_rot.sqrt()[None, :, None, None]; srj = w_rot.sqrt().view(1, J, 1, 1, 1)
    sp = w_pos.sqrt()[None]                                                     # [1,J,3]
    use_ref = w_ref > 0 and q_ref is not None
    if use_ref: qref_p = q_ref[:, pj, ps]; sref = float(np.sqrt(w_ref))
    dirs = dirs or []
    D = [(int(j), torch.as_tensor(a, device=dev, dtype=dt), torch.as_tensor(t, device=dev, dtype=dt).expand(B, 3), float(np.sqrt(w)))
         for (j, a, t, w) in dirs]

    def cost_of(qq):
        gR, gp, _ = _fk_axes(qq, A, rest, parents, off, root_R)
        c = ((sr * (gR - target_R)) ** 2).flatten(1).sum(-1)
        if use_pos: c = c + ((sp * (gp - target_p)) ** 2).flatten(1).sum(-1)
        if use_ref: c = c + w_ref * ((qq[:, pj, ps] - qref_p) ** 2).sum(-1)
        for (j, a, t, sw) in D: c = c + ((sw * ((gR[:, j] @ a) - t)) ** 2).sum(-1)
        return c, gR, gp

    lam = torch.full((B,), float(lam0), device=dev, dtype=dt)
    cost, _, _ = cost_of(q)
    eye_p = torch.eye(P, device=dev, dtype=dt)[None]
    hist = [float(cost.mean())]
    for _ in range(int(iters)):
        gR, gp, W = _fk_axes(q, A, rest, parents, off, root_R)
        w = W[:, pj, ps]; S = _skew(w)                                           # [B,P,3] / [B,P,3,3]
        dR = torch.einsum("bpik,bjkl->bjpil", S, gR) * sup[None, :, :, None, None] * srj
        Jm = dR.reshape(B, J, P, 9).permute(0, 1, 3, 2).reshape(B, J * 9, P)
        r = (sr * (gR - target_R)).reshape(B, J * 9)
        if use_pos:
            rel = gp[:, :, None, :] - gp[:, pj][:, None, :, :]
            dp = torch.cross(w[:, None, :, :].expand_as(rel), rel, dim=-1) * sup[None, :, :, None] * sp[:, :, None, :]
            Jm = torch.cat([Jm, dp.permute(0, 1, 3, 2).reshape(B, J * 3, P)], 1)
            r = torch.cat([r, (sp * (gp - target_p)).reshape(B, J * 3)], 1)
        for (j, a, t, sw) in D:
            u = gR[:, j] @ a                                                    # [B,3]
            du = torch.cross(w, u[:, None, :].expand_as(w), dim=-1) * sup[None, j, :, None] * sw   # [B,P,3]
            Jm = torch.cat([Jm, du.permute(0, 2, 1)], 1)
            r = torch.cat([r, sw * (u - t)], 1)
        if use_ref:
            Jm = torch.cat([Jm, sref * eye_p.expand(B, P, P)], 1)
            r = torch.cat([r, sref * (q[:, pj, ps] - qref_p)], 1)
        JtJ = Jm.transpose(1, 2) @ Jm; Jtr = (Jm.transpose(1, 2) @ r[..., None]).squeeze(-1)
        d = JtJ.diagonal(dim1=-2, dim2=-1)
        for _try in range(6):
            H = JtJ + (lam[:, None] * d.clamp_min(1e-9))[:, :, None] * eye_p + 1e-12 * eye_p
            try: step = torch.linalg.solve(H, -Jtr[..., None]).squeeze(-1)
            except Exception: step = torch.linalg.lstsq(H, -Jtr[..., None]).solution.squeeze(-1)
            qn = q.clone(); qn[:, pj, ps] = (q[:, pj, ps] + step).clamp(lo_p[None], hi_p[None])
            cn, _, _ = cost_of(qn); ok = cn < cost
            if bool(ok.any()):
                q = torch.where(ok[:, None, None], qn, q); cost = torch.where(ok, cn, cost)
                lam = torch.where(ok, (lam / 3).clamp_min(1e-9), lam * 4); break
            lam = lam * 4
        hist.append(float(cost.mean()))
        if len(hist) > 2 and abs(hist[-2] - hist[-1]) < tol * max(hist[-1], 1e-9): break
    if return_info: return q, {"iters": len(hist) - 1, "cost": hist, "n_par": P}
    return q


def refine(q0, spec, dof, rest, root_R, pins, w_ref=1e-2, lm_iters=60, q_ref=None):
    """q0 [B,J,3] 起点 (模型姿态经 fit_angles, 或零姿态); root_R [B,3,3] 透传.
    pins: 列表, 每项 dict(j, pos[B,3] 或 None, rot[B,3,3] 或 None, mode in {point,plane,axis,full},
          w_pos, w_rot, w_dir, sole_normal_local[3], forward_local[3]).
    返回 q, (gR, gp)."""
    B, J = q0.shape[0], q0.shape[1]; dt = q0.dtype
    TR = torch.eye(3, dtype=dt).expand(B, J, 3, 3).clone(); TP = torch.zeros(B, J, 3, dtype=dt)
    wr = torch.zeros(J, dtype=dt); wp = torch.zeros(J, 3, dtype=dt); dirs = []
    for c in pins:
        j = int(c["j"]); m = c.get("mode", "full")
        if c.get("pos") is not None:
            TP[:, j] = c["pos"]
            if m == "plane": wp[j, 1] = c["w_pos"]
            else: wp[j] = c["w_pos"]
        if c.get("rot") is not None:
            R = c["rot"]
            if m == "full": TR[:, j] = R; wr[j] = c["w_rot"]
            elif m == "plane":
                a = torch.as_tensor(c["sole_normal_local"], dtype=dt); dirs.append((j, a, R @ a, c["w_dir"]))
            elif m == "axis":
                a = torch.as_tensor(c["forward_local"], dtype=dt); dirs.append((j, a, R @ a, c["w_dir"]))
    q = lm_fit2(TR, spec, dof, q0, rest=rest, root_R=root_R, target_p=TP, w_rot=wr, w_pos=wp,
                q_ref=q0 if q_ref is None else q_ref, w_ref=w_ref, dirs=dirs, clamp=True, iters=lm_iters)
    gR, gp = fk_from_angles(q, spec, dof, rest=rest, root_R=root_R)
    return q, gR, gp


def geo_deg(Ra, Rb):
    tr = torch.einsum("...ii->...", Ra.transpose(-1, -2) @ Rb)
    return torch.rad2deg(torch.arccos(((tr - 1) / 2).clamp(-1, 1)))
