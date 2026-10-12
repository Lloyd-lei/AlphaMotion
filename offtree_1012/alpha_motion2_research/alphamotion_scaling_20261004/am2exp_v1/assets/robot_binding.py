"""Robot kinematic adapters and verified link -> AM2-joint binding of every visual mesh geom.

Adapters (imported read-only from surface/build_verified_patches.py = BVP):
  * 14 XWBC robots: BVP.source_adapter(body)  (vendor MJCF via exp_topo prep_xwbc_cache.resolve_xml,
    build_merge keep/frame, names/parents/offsets checked against the cached spec).  We additionally check the
    adapter against the exact AM2 Body (names, parents, non-root offsets) from dump_am2_bodies.
  * adroit_pen / relay_franka_full_v2: BVP.source_adapter(body, data/payloads/<body>.json) (rest-rebased frames).
  * unitree_h2: the AM2 tree is the anonymous data tree body00..body31 (Isaac-Lab BFS order).  Node k>0 is the
    child body of data joint k-1, so the vendor MJCF body is found by the data joint NAME
    (/var/tmp/alphamotion/xwbc_joint_names.json); the parent relation is checked against the MJCF tree.  Frames are
    the MJCF link frames (checked against stored data body quaternions: <0.1 deg).  AM2 offsets differ from the
    MJCF by up to several cm (documented in prep_xwbc_cache DATA_TREE), so H2 gets an APPROXIMATE binding: the
    per-geom transform in its anchor frame is least-squares fitted over the verification states using AM2 FK
    positions, and the residual is reported (pure geometry: MJCF + AM2 skeleton, no motion data used to fit).

AM2 convention: R_j(q) = AX R_mj(frame_j, q) [Rrest_j^T] AX^T, positions root-relative canonical Y-up cm,
P_child = P_parent + R_parent (offset_child [+ axis*(q-q_ref)]).  Identity rotations = AM2 rest pose.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from . import _env as E

AX = np.array([[0., 1, 0], [0, 0, 1], [1, 0, 0]], np.float64)

_BVP = None


def bvp():
    global _BVP
    if _BVP is None:
        E.sealed_surface_path()
        import build_verified_patches as m  # sealed, read-only
        _BVP = m
    return _BVP


def am2_bodies():
    p = E.CACHE_DIR / 'am2_bodies.json'
    if not p.exists():
        raise FileNotFoundError(f'{p} missing: run dump_am2_bodies.py (build_surface_v2 does this first)')
    return E.read_json(p)['bodies']


@dataclass
class Kinematics:
    body: str
    xml: str
    model: object
    keep: np.ndarray            # [J] MuJoCo body id carrying AM2 joint j (node map)
    frame: np.ndarray           # [J] MuJoCo body id whose xmat is joint j's global rotation
    names: list
    parents: np.ndarray
    offsets_cm: np.ndarray      # AM2 offsets (cm)
    rest_rebase: np.ndarray | None
    unit: float                 # MuJoCo length unit -> cm
    kind: str
    slide_axis: np.ndarray | None = None      # [J,3] canonical, unit
    slide_qadr: np.ndarray | None = None      # [J] int, -1 none
    slide_ref_cm: np.ndarray | None = None    # [J]
    slide_upper_cm: np.ndarray | None = None
    notes: dict = field(default_factory=dict)

    @property
    def J(self):
        return len(self.parents)

    # ------------------------------------------------------------------------------------------------------
    def forward(self, qpos):
        """MuJoCo state -> (R [J,3,3], P_fk [J,3] cm AM2 FK, P_mj [J,3] cm MuJoCo, gp [ngeom,3] cm, gr [ngeom,3,3])."""
        import mujoco
        m = self.model
        d = mujoco.MjData(m)
        d.qpos[:] = qpos
        mujoco.mj_forward(m, d)
        r = d.xmat[self.frame].reshape(-1, 3, 3).copy()
        if self.rest_rebase is not None:
            r = r @ self.rest_rebase.transpose(0, 2, 1)
        R = AX[None] @ r @ AX.T[None]
        root = d.xpos[self.keep[0]].copy()
        Pmj = (d.xpos[self.keep] - root) @ AX.T * self.unit
        gp = (d.geom_xpos - root) @ AX.T * self.unit
        gr = AX[None] @ d.geom_xmat.reshape(-1, 3, 3) @ AX.T[None]
        slide = np.zeros(self.J)
        if self.slide_qadr is not None:
            msk = self.slide_qadr >= 0
            slide[msk] = qpos[self.slide_qadr[msk]] * self.unit
        Pfk = self.fk(R, slide)
        return R, Pfk, Pmj, gp, gr

    def fk(self, R, slide_cm=None):
        P = np.zeros((self.J, 3))
        for j, p in enumerate(self.parents):
            if p < 0:
                continue
            off = self.offsets_cm[j].copy()
            if self.slide_axis is not None and slide_cm is not None and self.slide_qadr[j] >= 0:
                off = off + self.slide_axis[j] * (slide_cm[j] - self.slide_ref_cm[j])
            P[j] = P[p] + R[p] @ off
        return P

    def rest_positions(self, slide_cm=None):
        R = np.tile(np.eye(3), (self.J, 1, 1))
        if slide_cm is None and self.slide_ref_cm is not None:
            slide_cm = self.slide_ref_cm.copy()
        return self.fk(R, slide_cm)

    def random_states(self, count, seed=None):
        """BVP.random_states (seed 20261004): qpos0 followed by `count` uniform-in-limit states."""
        if seed is None:
            return bvp().random_states(self.model, count)
        import mujoco
        rng = np.random.default_rng(seed)
        m = self.model
        states = [m.qpos0.copy()]
        for _ in range(count):
            q = m.qpos0.copy()
            for j in range(m.njnt):
                adr = int(m.jnt_qposadr[j]); typ = int(m.jnt_type[j])
                if typ == int(mujoco.mjtJoint.mjJNT_HINGE):
                    if m.jnt_limited[j]:
                        lo, hi = m.jnt_range[j]; q[adr] = rng.uniform(lo + .08 * (hi - lo), hi - .08 * (hi - lo))
                    else:
                        q[adr] = rng.uniform(-.9, .9)
                elif typ == int(mujoco.mjtJoint.mjJNT_SLIDE):
                    lo, hi = m.jnt_range[j] if m.jnt_limited[j] else (-.03, .03)
                    q[adr] = rng.uniform(lo, hi)
                elif typ == int(mujoco.mjtJoint.mjJNT_FREE):
                    q[adr:adr + 3] += rng.normal(0, .1, 3); quat = rng.normal(size=4); q[adr + 3:adr + 7] = quat / np.linalg.norm(quat)
                elif typ == int(mujoco.mjtJoint.mjJNT_BALL):
                    quat = rng.normal(size=4); q[adr:adr + 4] = quat / np.linalg.norm(quat)
            states.append(q)
        return states

    def open_slides_qpos(self):
        """qpos0 with every kept slide at its upper limit (Franka: fully open gripper)."""
        q = self.model.qpos0.copy()
        if self.slide_qadr is not None:
            for j in np.flatnonzero(self.slide_qadr >= 0):
                q[self.slide_qadr[j]] = self.slide_upper_cm[j] / self.unit
        return q


def _check_against_am2(kin, am2, tol_cm=1e-3):
    if list(kin.names) != list(am2['names']):
        raise ValueError(f'{kin.body}: adapter names differ from AM2 Body')
    if [int(x) for x in kin.parents] != [int(x) for x in am2['parents']]:
        raise ValueError(f'{kin.body}: adapter parents differ from AM2 Body')
    off = np.asarray(am2['offsets_cm'], np.float64)
    return float(np.abs(off[1:] - kin.offsets_cm[1:]).max())


def load_kinematics(body) -> Kinematics:
    am2 = am2_bodies()[body]
    B = bvp()
    if body == 'unitree_h2':
        return _h2_kinematics(am2)
    if body in E.BODIES_EXTERNAL:
        meta = E.ROOT / 'data' / 'payloads' / f'{body}.json'
        a = B.source_adapter(body, str(meta))
        unit = 100.0
        kin = Kinematics(body=body, xml=str(a['xml']), model=a['model'], keep=np.asarray(a['keep'], int),
                         frame=np.asarray(a['frame'], int), names=list(a['names']), parents=np.asarray(a['parents'], int),
                         offsets_cm=np.asarray(a['offsets'], np.float64) * unit, rest_rebase=np.asarray(a['rest_rebase']),
                         unit=unit, kind='external_rest_rebased(BVP)')
        if 'slide_qpos_addresses' in a and (np.asarray(a['slide_qpos_addresses']) >= 0).any():
            ax = np.asarray(a['prismatic_axis'], np.float64)
            kin.slide_axis = ax
            kin.slide_qadr = np.asarray(a['slide_qpos_addresses'], int)
            kin.slide_ref_cm = np.asarray(a['slide_reference'], np.float64) * unit
            kin.slide_upper_cm = np.asarray(a['slide_upper'], np.float64) * unit
            ta = np.asarray(am2.get('typed_prismatic_axis', ax))
            kin.notes['slide_axis_vs_am2_max_abs'] = float(np.abs(ta - ax).max())
        err = _check_against_am2(kin, am2)
        kin.notes['adapter_vs_am2_offset_max_abs_cm'] = err
        kin.offsets_cm = np.asarray(am2['offsets_cm'], np.float64)
        return kin
    a = B.source_adapter(body)
    kin = Kinematics(body=body, xml=str(a['xml']), model=a['model'], keep=np.asarray(a['keep'], int),
                     frame=np.asarray(a['frame'], int), names=list(a['names']), parents=np.asarray(a['parents'], int),
                     offsets_cm=np.asarray(a['offsets'], np.float64), rest_rebase=None, unit=100.0,
                     kind='xwbc_merged_frame(BVP)')
    err = _check_against_am2(kin, am2)
    kin.notes['adapter_vs_am2_offset_max_abs_cm'] = err
    kin.offsets_cm = np.asarray(am2['offsets_cm'], np.float64)
    return kin


def _h2_kinematics(am2) -> Kinematics:
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(E.H2_XML))
    jn = E.read_json('/var/tmp/alphamotion/xwbc_joint_names.json')['unitree_h2']['joint_names']
    if len(jn) != am2['J'] - 1:
        raise ValueError('H2 data joint count mismatch')
    keep = [int(m.body_rootid[1]) if False else 1]
    for n in jn:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        if jid < 0:
            raise ValueError('H2 joint not in MJCF: ' + n)
        keep.append(int(m.jnt_bodyid[jid]))
    keep = np.asarray(keep, int)
    if len(set(keep.tolist())) != len(keep):
        raise ValueError('H2 node map not injective')
    parents = np.asarray(am2['parents'], int)
    # tree check: MJCF ancestor chain of keep[k] must reach keep[parent[k]] before any other mapped body
    mapped = set(keep.tolist())
    for k in range(1, len(keep)):
        a = int(m.body_parentid[keep[k]])
        while a not in mapped and a > 0:
            a = int(m.body_parentid[a])
        if a != keep[parents[k]]:
            raise ValueError(f'H2 tree mismatch at node {k}')
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(b)) for b in keep]
    kin = Kinematics(body='unitree_h2', xml=str(E.H2_XML), model=m, keep=keep, frame=keep.copy(), names=list(am2['names']),
                     parents=parents, offsets_cm=np.asarray(am2['offsets_cm'], np.float64), rest_rebase=None, unit=100.0,
                     kind='h2_data_tree_by_joint_name(approx)')
    kin.notes['vendor_body_names'] = names
    kin.notes['node_map_rule'] = 'AM2 node k>0 = MJCF body of data joint k-1 (xwbc_joint_names.json, Isaac BFS order); node 0 = pelvis'
    return kin


# ------------------------------------------------------------------------------------------------------------
def visual_geoms(kin):
    """BVP rule: mesh geoms with contype==conaffinity==0 under the robot subtree (fallback: every mesh geom)."""
    import mujoco
    m = kin.model
    vis = [g for g in range(m.ngeom) if m.geom_bodyid[g] > 0 and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
           and m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0]
    scope = 'visual_mesh'
    if not vis:
        vis = [g for g in range(m.ngeom) if m.geom_bodyid[g] > 0 and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]
        scope = 'declared_mesh_geometry_no_visual_collision_split'
    keepset = set(kin.keep.tolist())
    out, omitted = [], []
    for g in vis:
        b = int(m.geom_bodyid[g]); a = b
        while a not in keepset and a > 0:
            a = int(m.body_parentid[a])
        if a == 0:
            omitted.append({'geom': g, 'reason': 'outside_robot_subtree'})
            continue
        if m.geom_rgba[g][3] == 0:
            omitted.append({'geom': g, 'reason': 'transparent'})
            continue
        out.append(g)
    return out, omitted, scope


def mesh_arrays(kin, g):
    m = kin.model
    mid = int(m.geom_dataid[g])
    va, nv = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
    fa, nf = int(m.mesh_faceadr[mid]), int(m.mesh_facenum[mid])
    V = m.mesh_vert[va:va + nv].astype(np.float64)
    F = m.mesh_face[fa:fa + nf].astype(np.int64)
    return V, F


def _orthonormalize(M):
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1; R = U @ Vt
    return R


def _rot_angle_deg(R):
    c = np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(c))


def _fit_anchor(snaps, a, g, sub):
    """Least-squares rigid transform of geom g's vertices `sub` (canonical cm, geom frame) in AM2 frame a."""
    mats = np.stack([s[0][a].T @ s[4][g] for s in snaps])
    mat = _orthonormalize(mats.mean(0))
    tr = []
    for s in snaps:
        xt = sub @ s[4][g].T + s[3][g]
        tr.append(((xt - s[1][a]) @ s[0][a]) - sub @ mat.T)
    return mat, np.concatenate(tr).mean(0)


def _vertex_residual(snaps, a, g, Vc, mat, trans):
    """Max over states of |AM2-posed - MuJoCo-posed| per vertex (cm)."""
    res = np.zeros(len(Vc))
    for s in snaps:
        xt = Vc @ s[4][g].T + s[3][g]
        xp = (Vc @ mat.T + trans) @ s[0][a].T + s[1][a]
        res = np.maximum(res, np.linalg.norm(xp - xt, axis=-1))
    return res


def bind_geoms(kin, n_random=7, rigid_rot_tol=1e-6, rigid_pos_tol_cm=1e-4, piecewise=False):
    """Bind every visual geom to AM2 joints; verify over qpos0 + n_random BVP states (seed 20261004).

    Exact rule (BVP): a geom is rigid in AM2 frame a when R_a^T R_geom and R_a^T (x_geom - P_a) are constant over
    all verification states (tol 1e-6 / 1e-4 cm).  Preference: the anchor whose keep/frame body carries the geom,
    then the nearest.  Geoms with no rigid anchor (links between the two MuJoCo bodies of a merged AM2 joint, H2
    whose AM2 offsets differ from the MJCF, ...) get an APPROXIMATE binding: for every candidate anchor a
    restricted candidate set (nearest kept ancestor node and its parent) with either the transform that is exact at
    qpos0 or the least-squares one over the states, whichever has the smaller typical (median over states) worst
    vertex error; the reported residual is still the max over all states; piecewise=True
    (off by default: it picked unrelated anchors that fit random states but not data) lets every FACE choose.

    Returns (parts, audit).  A part = one (geom, anchor) piece: {geom, body, body_name, mesh_name, anchor, mat [3,3],
    trans [3] cm (anchor frame), rigid, residual_cm (max vertex error over the states), V (native), F (subset of the
    geom's faces), area_native}.
    """
    import mujoco
    m = kin.model
    states = kin.random_states(n_random)
    snaps = [kin.forward(q) for q in states]
    fk_err = max(float(np.abs(s[1] - s[2]).max()) for s in snaps)  # AM2 FK vs MuJoCo joint positions (cm)
    rest_dev = float(_rot_angle_deg(snaps[0][0]).max())             # AM2 rotation at qpos0 vs identity
    geoms, omitted, scope = visual_geoms(kin)
    keep_list = kin.keep.tolist()
    parts, approx = [], []
    for g in geoms:
        b = int(m.geom_bodyid[g])
        V, F = mesh_arrays(kin, g)
        Vc = (V @ AX.T) * kin.unit  # geom-frame vertices, canonical axes, cm
        mid = int(m.geom_dataid[g])
        meta = dict(geom=g, body=b, body_name=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b),
                    mesh_name=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, mid), V=V)
        cands = []
        for a in range(kin.J):
            mats = np.stack([s[0][a].T @ s[4][g] for s in snaps])
            trans = np.stack([s[0][a].T @ (s[3][g] - s[1][a]) for s in snaps])
            er = float(np.abs(mats - mats[0]).max()); ep = float(np.linalg.norm(trans - trans[0], axis=-1).max())
            if er < rigid_rot_tol and ep < rigid_pos_tol_cm:
                pref = 0 if (keep_list[a] == b or kin.frame[a] == b) else 1
                cands.append((pref, float(np.linalg.norm(trans[0])), a, mats[0], trans[0]))
        if cands:
            cands.sort(key=lambda c: (c[0], c[1]))
            _, _, a, mat, trans = cands[0]
            res = float(_vertex_residual(snaps, a, g, Vc, mat, trans).max())
            parts.append(dict(meta, anchor=int(a), mat=mat, trans=trans, rigid=True, residual_cm=res, F=F,
                              n_faces_geom=len(F)))
            continue
        # APPROXIMATE geom.  Candidates: the AM2 node whose kept body is the nearest MuJoCo ancestor of the geom's
        # body (or whose frame body carries it) and that node's parent.  Two transform rules per candidate:
        # 'qpos0' = exact at the vendor neutral pose (error grows with the hidden DOF's deviation from qpos0) and
        # 'ls' = least squares over the verification states.  The (candidate, rule) with the smallest max vertex
        # error over the verification states wins (single anchor per geom; piecewise only if requested).
        chain, a_ = [], b
        while a_ > 0:
            chain.append(a_); a_ = int(m.body_parentid[a_])
        near = next((x for x in chain if x in keep_list), None)
        cand = set()
        if near is not None:
            a1 = keep_list.index(near); cand.add(a1)
            if kin.parents[a1] >= 0:
                cand.add(int(kin.parents[a1]))
        cand |= {a for a in range(kin.J) if int(kin.frame[a]) == b}
        if not cand:
            cand = set(range(kin.J))
        sub = Vc[np.linspace(0, len(Vc) - 1, min(len(Vc), 400)).astype(int)]
        fits, vres = {}, {}
        for a in sorted(cand):
            s0 = snaps[0]
            mat0 = s0[0][a].T @ s0[4][g]; t0 = s0[0][a].T @ (s0[3][g] - s0[1][a])
            for rule, (mat, t) in (('qpos0', (mat0, t0)), ('ls', _fit_anchor(snaps, a, g, sub))):
                fits[(a, rule)] = (mat, t)
                vres[(a, rule)] = _vertex_residual(snaps, a, g, Vc, mat, t)
        whole = {k: float(r.max()) for k, r in vres.items()}
        typical = {k: float(np.median([_vertex_residual([sn], k[0], g, sub, *fits[k]).max() for sn in snaps]))
                   for k in vres}
        k_best = min(typical, key=lambda k: (typical[k], whole[k]))   # typical-case (median over states) error
        if piecewise:
            keys = sorted(vres)
            fr = np.stack([vres[k][F].max(1) for k in keys])
            choice_i = np.argmin(fr, axis=0)
        else:
            keys = [k_best]; choice_i = np.zeros(len(F), int)
        pieces = []
        for i in np.unique(choice_i):
            k = keys[int(i)]; sel = choice_i == i
            Fa = F[sel]
            res = float(vres[k][np.unique(Fa)].max())
            parts.append(dict(meta, anchor=int(k[0]), mat=fits[k][0], trans=fits[k][1], rigid=False, fit_rule=k[1],
                              residual_cm=res, F=Fa, n_faces_geom=len(F)))
            pieces.append({'anchor': kin.names[int(k[0])], 'rule': k[1], 'faces': int(sel.sum()), 'residual_cm': res})
        approx.append({'geom': g, 'body': meta['body_name'], 'candidates': [kin.names[a] for a in sorted(cand)],
                       'best_anchor': kin.names[k_best[0]], 'best_rule': k_best[1], 'best_residual_cm': whole[k_best],
                       'selection': 'min over candidates of the MEDIAN over verification states of the max vertex error',
                       'best_median_state_residual_cm': typical[k_best],
                       'all_candidates_residual_cm': {f'{kin.names[k[0]]}/{k[1]}': v for k, v in whole.items()},
                       'piecewise_residual_cm': max(p['residual_cm'] for p in pieces), 'pieces': pieces})
    for p in parts:
        tri = p['V'][p['F']]
        p['area_native'] = float(0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum())
    audit = dict(states=len(states), random_state_seed=20261004, fk_am2_vs_mujoco_max_cm=fk_err,
                 am2_rotation_at_qpos0_vs_identity_max_deg=rest_dev, mesh_scope=scope, omitted_geoms=omitted,
                 n_geoms=len(geoms), n_parts=len(parts), n_rigid_geoms=len(geoms) - len(approx),
                 max_residual_cm=max(p['residual_cm'] for p in parts),
                 max_residual_rigid_cm=max([p['residual_cm'] for p in parts if p['rigid']] or [0.0]),
                 approx_geoms=approx, piecewise=bool(piecewise))
    return parts, audit


def assembled_mesh(kin, parts, R=None, P=None):
    """Whole-body mesh (cm, canonical) from bound parts at given AM2 frames (default: AM2 rest pose).

    Every part keeps its own copy of the vertices it uses, so a point's rest position is exactly
    R_a(rest)=I @ local + P_a(rest).  Returns V [N,3] cm, F [M,3], face_part [M] (index into `parts`)."""
    if R is None:
        R = np.tile(np.eye(3), (kin.J, 1, 1))
    if P is None:
        P = kin.rest_positions()
    Vs, Fs, fp = [], [], []
    off = 0
    for i, x in enumerate(parts):
        used, inv = np.unique(x['F'].reshape(-1), return_inverse=True)
        Vc = (x['V'][used] @ AX.T) * kin.unit
        loc = Vc @ x['mat'].T + x['trans']
        Vw = loc @ R[x['anchor']].T + P[x['anchor']]
        Vs.append(Vw); Fs.append(inv.reshape(-1, 3) + off); fp.append(np.full(len(x['F']), i)); off += len(Vw)
    return np.concatenate(Vs), np.concatenate(Fs), np.concatenate(fp)


def data_states(kin, clips=('walk1_subject1', 'fallAndGetUp1_subject1', 'ground1_subject1'), frames_per_clip=60):
    """MuJoCo qpos from the xwbc extracted LAFAN robot trajectories (root at qpos0; joints by name)."""
    import mujoco
    names = E.read_json('/var/tmp/alphamotion/xwbc_joint_names.json').get(kin.body, {}).get('joint_names')
    if not names:
        return [], 'no joint-name table'
    m = kin.model
    qadr = []
    for n in names:
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        qadr.append(int(m.jnt_qposadr[j]) if j >= 0 else -1)
    qadr = np.asarray(qadr)
    states, used = [], []
    for c in clips:
        f = E.XWBC_EXTRACTED / kin.body / f'{c}.npz'
        if not f.exists():
            continue
        with np.load(f) as z:
            jp = z['joint_pos']
        if jp.shape[1] != len(names):
            return [], f'joint count mismatch {jp.shape[1]} vs {len(names)}'
        for t in np.linspace(0, len(jp) - 1, frames_per_clip).astype(int):
            q = m.qpos0.copy()
            ok = qadr >= 0
            q[qadr[ok]] = jp[t, ok]
            states.append(q)
        used.append(c)
    return states, f'{len(states)} frames from {used}; {int((qadr < 0).sum())} data joints absent from MJCF'


def data_residual(kin, parts, states):
    """Per-part max vertex error (cm) of the AM2-rigid posing vs MuJoCo over real data joint states."""
    if not states:
        return None
    snaps = [kin.forward(q) for q in states]
    out = []
    for p in parts:
        Vc = (p['V'][np.unique(p['F'])] @ AX.T) * kin.unit
        r = np.zeros(len(snaps))
        for i, s in enumerate(snaps):
            xt = Vc @ s[4][p['geom']].T + s[3][p['geom']]
            xp = (Vc @ p['mat'].T + p['trans']) @ s[0][p['anchor']].T + s[1][p['anchor']]
            r[i] = np.linalg.norm(xp - xt, axis=-1).max()
        out.append(r)
    out = np.stack(out)  # [parts, frames]
    worst = out.max(0)
    return dict(frames=int(out.shape[1]), max_cm=float(out.max()), p95_of_frame_max_cm=float(np.percentile(worst, 95)),
                p50_of_frame_max_cm=float(np.median(worst)),
                per_part_max_cm=[float(x) for x in out.max(1)],
                per_part_p95_cm=[float(x) for x in np.percentile(out, 95, axis=1)])


def h2_data_verification(kin, bound, clip='walk1_subject1', frames=120):
    """Verification only (never used to fit): stored xwbc H2 body poses vs MJCF at the same joint angles."""
    import mujoco
    f = E.XWBC_EXTRACTED / 'unitree_h2' / f'{clip}.npz'
    z = np.load(f)
    jn = [str(x) for x in z['joint_names']]
    m = kin.model
    d = mujoco.MjData(m)
    qadr = {n: int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in jn}
    T = len(z['joint_pos']); idx = np.linspace(0, T - 1, frames).astype(int)

    def q2m(q):
        w, x, y, zq = q
        return np.array([[1 - 2 * (y * y + zq * zq), 2 * (x * y - w * zq), 2 * (x * zq + w * y)],
                         [2 * (x * y + w * zq), 1 - 2 * (x * x + zq * zq), 2 * (y * zq - w * x)],
                         [2 * (x * zq - w * y), 2 * (y * zq + w * x), 1 - 2 * (x * x + y * y)]])
    rot_err, pos_err, vert_err = [], [], []
    for t in idx:
        d.qpos[:] = m.qpos0; d.qpos[:3] = z['body_pos_w'][t, 0]; d.qpos[3:7] = z['body_quat_w'][t, 0]
        for i, n in enumerate(jn):
            d.qpos[qadr[n]] = z['joint_pos'][t, i]
        mujoco.mj_kinematics(m, d)
        Rd = np.stack([AX @ q2m(q) @ AX.T for q in z['body_quat_w'][t]])
        Pd = (z['body_pos_w'][t] - z['body_pos_w'][t, 0]) @ AX.T * 100
        root = d.xpos[kin.keep[0]]
        for k in range(kin.J):
            Rm = AX @ d.xmat[kin.keep[k]].reshape(3, 3) @ AX.T
            c = np.clip((np.trace(Rd[k].T @ Rm) - 1) / 2, -1, 1); rot_err.append(np.degrees(np.arccos(c)))
            pos_err.append(np.linalg.norm(Pd[k] - (d.xpos[kin.keep[k]] - root) @ AX.T * 100))
        gp = (d.geom_xpos - root) @ AX.T * 100; gr = AX[None] @ d.geom_xmat.reshape(-1, 3, 3) @ AX.T[None]
        for x in bound:
            used = np.unique(x['F'])
            Vc = (x['V'][used[::max(1, len(used) // 100)]] @ AX.T) * 100
            xt = Vc @ gr[x['geom']].T + gp[x['geom']]
            xp = (Vc @ x['mat'].T + x['trans']) @ Rd[x['anchor']].T + Pd[x['anchor']]
            vert_err.append(float(np.linalg.norm(xp - xt, axis=-1).max()))
    rot_err, pos_err, vert_err = map(np.asarray, (rot_err, pos_err, vert_err))
    return dict(clip=clip, frames=int(len(idx)), body_rotation_err_deg_max=float(rot_err.max()),
                body_origin_err_cm_p50=float(np.median(pos_err)), body_origin_err_cm_max=float(pos_err.max()),
                surface_vertex_err_cm_p50=float(np.median(vert_err)), surface_vertex_err_cm_p95=float(np.percentile(vert_err, 95)),
                surface_vertex_err_cm_max=float(vert_err.max()),
                note='data-frame posed asset geometry vs vendor MJCF posed at the same joint angles (verification only)')
