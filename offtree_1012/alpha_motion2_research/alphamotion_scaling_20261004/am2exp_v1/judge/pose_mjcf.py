"""Judge J1 geometry: vendor-MJCF COLLISION geoms rigidly re-posed from AM2 global rotations/positions.

Independence contract (final_plan.md "Judge J1"): this module never touches the dense visual surface
(CODE/assets), patches, the sole table, any training loss or the model's contact head.

AM2 kinematic convention for the XWBC robots (verified here at random q for every robot):
    R_am2[j] = AX @ R_world(frame_body[j]) @ AX.T        (global, NOT rest-relative: non-identity at
    P_am2[j] = AX @ (x_world(keep_body[j]) - x_root) * 100   qpos0 for tilted MJCF bodies)
with AX = [[0,1,0],[0,0,1],[1,0,0]] (MuJoCo Z-up -> canonical Y-up), keep/frame from the cache builder's
`prep_gmr_cache.build_merge` (imported read-only, only at build time).  A collision geom g is bound to the
AM2 node j for which (R_j^T Rg, R_j^T (pg - P_j)) is constant over qpos0 + 15 seeded random q
(`rigid`), or, when no node frame carries it rigidly (bodies merged into a co-located chain with their
own hinge), to the node with the smallest re-pose residual (`approx`, residual recorded in cm).

unitree_h2: the AM2 H2 tree (anonymous body00..31) was built from the retarget data, not from the
vendor MJCF.  Nodes are mapped to vendor bodies by geometry/topology (H2_NODE_MAP below; the data
rotations were verified to rotate about exactly the vendor joint axes with the vendor rest frames,
see README) and vendor geoms are attached to the mapped node; the vendor-vs-data bone offset
mismatch is recorded per node (`h2_offset_residual_cm`).

Binding files: OUT/judge/bindings_v1/<body>.npz + .json (build with --build; judge-time loading
never imports the exp_topo cache code).
"""
from __future__ import annotations
import argparse
import functools
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

try:
    from am2exp_v1.judge.regions import REGIONS, ROBOTS, region_of, N_REGIONS
except ImportError:  # pragma: no cover - executed as a file
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from am2exp_v1.judge.regions import REGIONS, ROBOTS, region_of, N_REGIONS

ROOT = Path('/home/arenalabs/Desktop/"be water, robot"/alpha_motion2/research/alphamotion_scaling_20261004')
OUT = Path('/var/tmp/alphamotion/am2exp')
BINDING_DIR = OUT / 'judge' / 'bindings_v1'
EXP_TOPO = Path('/home/arenalabs/Desktop/"be water, robot"/pose_translator_design/exp_topo')
H2_XML = Path('/var/tmp/alphamotion/robot_models/unitree_ros/robots/h2_description/unitree_h2_mjcf.xml')
AX = np.array([[0., 1, 0], [0, 0, 1], [1, 0, 0]], np.float64)
N_RANDOM_STATES = 15
RIGID_ROT_TOL = 1e-6
RIGID_POS_TOL_CM = 1e-4
FK_TOL_CM = 0.1  # cache builder FK round-trip gate (prep_gmr_cache.FK_GATE_CM); GR3 AM2 offsets differ by 0.5 mm
SCHEMA = 'am2exp_j1_collision_binding_v1'

# AM2 unitree_h2 node -> vendor MJCF body.  Topology: AM2 parents
# [-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,10,11,12,13,14,15,16,18,19,22,23,24,25,26,27,28,29].
# Node 9 is the zero-offset torso (vendor torso_link, merged by build_merge into waist_roll).
H2_NODE_MAP = {0: 'pelvis', 1: 'left_hip_pitch_link', 2: 'right_hip_pitch_link', 3: 'waist_yaw_link',
               4: 'left_hip_roll_link', 5: 'right_hip_roll_link', 6: 'waist_roll_link',
               7: 'left_hip_yaw_link', 8: 'right_hip_yaw_link', 9: 'torso_link', 10: 'left_knee_link',
               11: 'right_knee_link', 12: 'head_pitch_link', 13: 'left_shoulder_pitch_link',
               14: 'right_shoulder_pitch_link', 15: 'left_ankle_roll_link', 16: 'right_ankle_roll_link',
               17: 'head_yaw_link', 18: 'left_shoulder_roll_link', 19: 'right_shoulder_roll_link',
               20: 'left_ankle_pitch_link', 21: 'right_ankle_pitch_link', 22: 'left_shoulder_yaw_link',
               23: 'right_shoulder_yaw_link', 24: 'left_elbow_link', 25: 'right_elbow_link',
               26: 'left_wrist_roll_link', 27: 'right_wrist_roll_link', 28: 'left_wrist_pitch_link',
               29: 'right_wrist_pitch_link', 30: 'left_wrist_yaw_link', 31: 'right_wrist_yaw_link'}


def sha256_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def hop_matrix(parents):
    J = len(parents)
    adj = [[] for _ in range(J)]
    for j, p in enumerate(parents):
        if p >= 0:
            adj[j].append(int(p)); adj[int(p)].append(j)
    H = np.full((J, J), 10 ** 6, np.int64)
    for s in range(J):
        H[s, s] = 0; frontier = [s]
        while frontier:
            nxt = []
            for a in frontier:
                for b in adj[a]:
                    if H[s, b] > H[s, a] + 1:
                        H[s, b] = H[s, a] + 1; nxt.append(b)
            frontier = nxt
    return H


# ----------------------------------------------------------------------------------------------
# build time (CPU, run once per body; imports exp_topo cache code read-only)
# ----------------------------------------------------------------------------------------------
def _exp_topo():
    """Load the cache builders read-only. They do `from common import ...` (exp_topo/common.py); only
    the build CLI calls this, never a process that also imports the sealed AM2 legacy `common`."""
    sys.dont_write_bytecode = True
    if str(EXP_TOPO) not in sys.path:
        sys.path.insert(0, str(EXP_TOPO))
    import prep_xwbc_cache as PX  # noqa: E402
    import prep_gmr_cache as PG   # noqa: E402
    return PX, PG


def resolve_xml(body: str) -> str:
    PX, _ = _exp_topo()
    return str(Path(PX.resolve_xml(body)))


def am2_body_spec(body: str):
    """names/parents/offsets(cm) exactly as the AM2 data manifest + static spec define them."""
    man = json.loads((ROOT / 'data/data_manifest_v2.json').read_text())
    b = next(b for b in man['xwbc']['bodies'] if b['body'] == body)
    spec = Path(b['static_spec'])
    if sha256_file(spec) != b['static_spec_sha256']:
        raise ValueError('static spec hash changed')
    with np.load(spec, allow_pickle=True) as s:
        parents = np.asarray(s['parents'], np.int64); off = np.asarray(s['rest_offsets'], np.float64)
        names = [str(x) for x in s['joint_names']]
    if names != list(b['joint_names']) or parents.tolist() != list(b['parents']):
        raise ValueError('manifest/spec disagreement')
    return names, parents, off, str(spec), b['static_spec_sha256']


def random_states(model, count=N_RANDOM_STATES, seed=20261004):
    """qpos0 + seeded random q inside 8%-trimmed limits (same rule as surface/build_verified_patches)."""
    import mujoco
    rng = np.random.default_rng(seed); states = [model.qpos0.copy()]
    for _ in range(count):
        q = model.qpos0.copy()
        for j in range(model.njnt):
            adr = int(model.jnt_qposadr[j]); typ = int(model.jnt_type[j])
            if typ == int(mujoco.mjtJoint.mjJNT_HINGE):
                if model.jnt_limited[j]:
                    lo, hi = model.jnt_range[j]; q[adr] = rng.uniform(lo + .08 * (hi - lo), hi - .08 * (hi - lo))
                else:
                    q[adr] = rng.uniform(-.9, .9)
            elif typ == int(mujoco.mjtJoint.mjJNT_SLIDE):
                lo, hi = model.jnt_range[j] if model.jnt_limited[j] else (-.03, .03); q[adr] = rng.uniform(lo, hi)
            elif typ == int(mujoco.mjtJoint.mjJNT_FREE):
                q[adr:adr + 3] += rng.normal(0, .1, 3); quat = rng.normal(size=4); q[adr + 3:adr + 7] = quat / np.linalg.norm(quat)
            elif typ == int(mujoco.mjtJoint.mjJNT_BALL):
                quat = rng.normal(size=4); q[adr:adr + 4] = quat / np.linalg.norm(quat)
        states.append(q)
    return states


def _rot_angle(M):
    return float(np.arccos(np.clip((np.trace(M) - 1) / 2, -1, 1)))


def collision_geoms(model):
    import mujoco
    bad = {int(mujoco.mjtGeom.mjGEOM_PLANE), int(mujoco.mjtGeom.mjGEOM_HFIELD)}
    return [g for g in range(model.ngeom) if model.geom_bodyid[g] > 0 and int(model.geom_type[g]) not in bad
            and (model.geom_contype[g] or model.geom_conaffinity[g])]


def hull_vertices(model, g):
    import mujoco
    if int(model.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_MESH):
        return np.zeros((0, 3))
    mid = int(model.geom_dataid[g]); a = int(model.mesh_vertadr[mid]); n = int(model.mesh_vertnum[mid])
    v = model.mesh_vert[a:a + n].astype(np.float64)
    try:
        from scipy.spatial import ConvexHull
        v = v[ConvexHull(v).vertices]
    except Exception:
        pass
    return v


MAX_CHAIN = 4


def rodrigues(axis, theta):
    """axis [...,3] unit, theta [...] -> rotation [...,3,3]."""
    axis = np.asarray(axis, np.float64); theta = np.asarray(theta, np.float64)
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    c, s = np.cos(theta), np.sin(theta); C = 1 - c
    R = np.stack([np.stack([c + x * x * C, x * y * C - z * s, x * z * C + y * s], -1),
                  np.stack([y * x * C + z * s, c + y * y * C, y * z * C - x * s], -1),
                  np.stack([z * x * C - y * s, z * y * C + x * s, c + z * z * C], -1)], -2)
    return R


def _chain_products(Q, axes, theta):
    """Q [n,3,3], axes [n,3] (zero = no joint), theta [T,n] -> cumulative rotations [T,n,3,3] and per-body
    (pre-joint, post-joint) rotations, relative to the parent node frame."""
    T, n = theta.shape
    cur = np.broadcast_to(np.eye(3), (T, 3, 3)).copy(); pre, post = [], []
    for i in range(n):
        Rp = cur @ Q[i]
        Rj = rodrigues(np.broadcast_to(axes[i], (T, 3)), theta[:, i]) if np.any(axes[i]) else np.broadcast_to(np.eye(3), (T, 3, 3))
        cur = Rp @ Rj; pre.append(Rp); post.append(cur)
    return np.stack(pre, 1), np.stack(post, 1)


def solve_chain(Q, axes, lo, hi, Rrel, iters=25):
    """Least-squares hinge angles theta [T,n] with prod_i Q_i Rot(a_i, theta_i) ~= Rrel [T,3,3]. Multi-start
    Gauss-Newton (numeric Jacobian); among near-equal residual solutions the one inside the joint range
    with the smallest |theta| wins (resolves Euler branch ambiguity)."""
    n = len(Q); T = len(Rrel)
    active = np.array([bool(np.any(a)) for a in axes])
    starts = [0.0, np.pi / 2, -np.pi / 2, np.pi]
    best_th = np.zeros((T, n)); best_key = np.full((T, 3), np.inf)
    for s0 in starts:
        th = np.zeros((T, n)); th[:, active] = s0
        for _ in range(iters):
            r0 = (_chain_products(Q, axes, th)[1][:, -1] - Rrel).reshape(T, 9)
            Jm = np.zeros((T, 9, n)); eps = 1e-6
            for i in np.flatnonzero(active):
                tp = th.copy(); tp[:, i] += eps
                Jm[:, :, i] = ((_chain_products(Q, axes, tp)[1][:, -1] - Rrel).reshape(T, 9) - r0) / eps
            JtJ = np.einsum('tki,tkj->tij', Jm, Jm) + 1e-9 * np.eye(n)
            step = np.linalg.solve(JtJ, -np.einsum('tki,tk->ti', Jm, r0)[..., None])[..., 0]
            th = th + np.clip(step, -0.5, 0.5)
        th = (th + np.pi) % (2 * np.pi) - np.pi
        res = np.linalg.norm((_chain_products(Q, axes, th)[1][:, -1] - Rrel).reshape(T, 9), axis=-1)
        viol = np.clip(lo - th, 0, None).sum(1) + np.clip(th - hi, 0, None).sum(1)
        key = np.stack([np.round(res, 6), viol, np.abs(th).sum(1)], 1)
        better = (key[:, 0] < best_key[:, 0]) | ((key[:, 0] == best_key[:, 0]) & (
            (key[:, 1] < best_key[:, 1]) | ((key[:, 1] == best_key[:, 1]) & (key[:, 2] < best_key[:, 2]))))
        best_key[better] = key[better]; best_th[better] = th[better]
    return best_th, best_key[:, 0]


def _merged_chain(model, data, parents, node_body, node_frame, root_body, gb, g, snaps, k):
    """Geom on a body merged inside a co-located chain: express it through the hinge angles that connect
    the parent node frame to the node frame (solved per frame).  Returns None when not applicable."""
    import mujoco
    J = len(parents)
    for j in range(1, J):
        p = int(parents[j]); top = int(node_frame[p]); path = []; cur = int(node_frame[j])
        while cur not in (top, 0):
            path.append(cur); cur = int(model.body_parentid[cur])
        if cur != top or gb not in path:
            continue
        path = path[::-1]
        if len(path) > MAX_CHAIN:
            return None
        Q, axes, bp, jp, lo, hi = [], [], [], [], [], []
        for b in path:
            jids = [jj for jj in range(model.njnt) if model.jnt_bodyid[jj] == b]
            if len(jids) > 1 or any(int(model.jnt_type[jj]) != int(mujoco.mjtJoint.mjJNT_HINGE) for jj in jids):
                return None
            qm = np.zeros(9); mujoco.mju_quat2Mat(qm, model.body_quat[b])
            Q.append(AX @ qm.reshape(3, 3) @ AX.T); bp.append(AX @ model.body_pos[b] * 100)
            if jids:
                jj = jids[0]; axes.append(AX @ model.jnt_axis[jj]); jp.append(AX @ model.jnt_pos[jj] * 100)
                lo.append(model.jnt_range[jj][0] if model.jnt_limited[jj] else -np.pi)
                hi.append(model.jnt_range[jj][1] if model.jnt_limited[jj] else np.pi)
            else:
                axes.append(np.zeros(3)); jp.append(np.zeros(3)); lo.append(0.0); hi.append(0.0)
        ch = dict(node=j, parent=p, bodies=path, gidx=path.index(gb), Q=np.array(Q), axes=np.array(axes),
                  bp=np.array(bp), jp=np.array(jp), lo=np.array(lo), hi=np.array(hi))
        # geom pose relative to its own body (rigid), from state 0
        data.qpos[:] = model.qpos0; mujoco.mj_forward(model, data)
        Rb = AX @ data.xmat[gb].reshape(3, 3) @ AX.T; pb = AX @ (data.xpos[gb] - data.xpos[root_body]) * 100
        Rg = AX @ data.geom_xmat[g].reshape(3, 3) @ AX.T; pg = AX @ (data.geom_xpos[g] - data.xpos[root_body]) * 100
        ch['mat'] = Rb.T @ Rg; ch['trans'] = Rb.T @ (pg - pb)
        # verification: solve on the random states and compare with MuJoCo
        rot = np.stack([s['r'] for s in snaps]); pos = np.stack([s['p'] for s in snaps])
        gr, gp, res = chain_pose(ch, rot, pos)
        Rt = np.stack([s['gr'][k] for s in snaps]); pt = np.stack([s['gp'][k] for s in snaps])
        rb = float(model.geom_rbound[g]) * 100
        err = [float(np.linalg.norm(gp[t] - pt[t])) + _rot_angle(gr[t] @ Rt[t].T) * rb for t in range(len(snaps))]
        ch['verify_residual_cm'] = float(max(err))
        return ch
    return None


def chain_pose(ch, rot, pos):
    """Pose of a merged-chain geom from AM2 rotations/positions (canonical): returns gr [T,3,3], gp [T,3] cm,
    and the per-frame rotation-fit residual (Frobenius)."""
    Rp = rot[:, ch['parent']]; Rj = rot[:, ch['node']]
    Rrel = np.einsum('tba,tbc->tac', Rp, Rj)
    th, res = solve_chain(ch['Q'], ch['axes'], ch['lo'], ch['hi'], Rrel)
    pre, post = _chain_products(ch['Q'], ch['axes'], th)
    pre = Rp[:, None] @ pre; post = Rp[:, None] @ post
    # positions: start at the parent node position, walk the (near co-located) chain
    pcur = pos[:, ch['parent']].copy(); Rprev = Rp
    for i in range(ch['gidx'] + 1):
        anchor = pcur + np.einsum('tab,b->ta', Rprev, ch['bp'][i]) + np.einsum('tab,b->ta', pre[:, i], ch['jp'][i])
        pcur = anchor - np.einsum('tab,b->ta', post[:, i], ch['jp'][i]); Rprev = post[:, i]
    Rb = post[:, ch['gidx']]
    return Rb @ ch['mat'], np.einsum('tab,b->ta', Rb, ch['trans']) + pcur, res


def _pack_chains(chains, G):
    """Flatten chain definitions into fixed-size arrays for the npz (chain_* keys)."""
    n = len(chains); M = MAX_CHAIN
    out = dict(chain_geom=np.zeros(n, np.int32), chain_node=np.zeros(n, np.int32), chain_parent=np.zeros(n, np.int32),
               chain_len=np.zeros(n, np.int32), chain_gidx=np.zeros(n, np.int32), chain_Q=np.tile(np.eye(3), (n, M, 1, 1)),
               chain_axes=np.zeros((n, M, 3)), chain_bp=np.zeros((n, M, 3)), chain_jp=np.zeros((n, M, 3)),
               chain_lo=np.zeros((n, M)), chain_hi=np.zeros((n, M)), chain_mat=np.zeros((n, 3, 3)), chain_trans=np.zeros((n, 3)),
               chain_verify_residual_cm=np.zeros(n))
    for i, (k, c) in enumerate(sorted(chains.items())):
        L = len(c['Q'])
        out['chain_geom'][i] = k; out['chain_node'][i] = c['node']; out['chain_parent'][i] = c['parent']
        out['chain_len'][i] = L; out['chain_gidx'][i] = c['gidx']
        for key in ('Q', 'axes', 'bp', 'jp', 'lo', 'hi'):
            out['chain_' + key][i, :L] = c[key]
        out['chain_mat'][i] = c['mat']; out['chain_trans'][i] = c['trans']; out['chain_verify_residual_cm'][i] = c['verify_residual_cm']
    return out


def _unpack_chain(b, i):
    get = (lambda k: b[k]) if isinstance(b, dict) else (lambda k: getattr(b, k))
    L = int(get('chain_len')[i])
    return dict(node=int(get('chain_node')[i]), parent=int(get('chain_parent')[i]), gidx=int(get('chain_gidx')[i]),
                Q=get('chain_Q')[i, :L], axes=get('chain_axes')[i, :L], bp=get('chain_bp')[i, :L], jp=get('chain_jp')[i, :L],
                lo=get('chain_lo')[i, :L], hi=get('chain_hi')[i, :L], mat=get('chain_mat')[i], trans=get('chain_trans')[i])


def build_binding(body: str) -> tuple[dict, dict]:
    import mujoco
    names, parents, off, spec_path, spec_sha = am2_body_spec(body)
    J = len(names)
    PX, PG = _exp_topo()
    xml = str(H2_XML) if body == 'unitree_h2' else resolve_xml(body)
    model = mujoco.MjModel.from_xml_path(xml); data = mujoco.MjData(model)
    bname = lambda i: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(i))
    if body == 'unitree_h2':
        node_body = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, H2_NODE_MAP[j]) for j in range(J)])
        if (node_body < 0).any():
            raise ValueError('H2 node map names missing in vendor MJCF')
        node_frame = node_body.copy(); root_body = int(node_body[0])
    else:
        keep, frame = PG.build_merge(model)
        if len(keep) != J or [bname(k) for k in keep] != names:
            raise ValueError(f'{body}: vendor merged skeleton != AM2 joint names')
        node_body = np.asarray(keep, np.int64); node_frame = np.asarray([frame[k] for k in keep], np.int64)
        root_body = int(keep[0])
    states = random_states(model)
    geoms = collision_geoms(model)
    snaps = []
    fk_err = 0.0
    for q in states:
        data.qpos[:] = q; mujoco.mj_forward(model, data)
        r = AX[None] @ data.xmat[node_frame].reshape(-1, 3, 3) @ AX.T[None]
        p = (data.xpos[node_body] - data.xpos[root_body]) @ AX.T * 100.0
        gr = AX[None] @ data.geom_xmat[geoms].reshape(-1, 3, 3) @ AX.T[None]
        gp = (data.geom_xpos[geoms] - data.xpos[root_body]) @ AX.T * 100.0
        if body != 'unitree_h2':
            fk = np.zeros_like(p)
            for j, pa in enumerate(parents):
                if pa >= 0:
                    fk[j] = fk[pa] + r[pa] @ off[j]
            fk_err = max(fk_err, float(np.linalg.norm(fk - p, axis=-1).max()))
        snaps.append(dict(r=r, p=p, gr=gr, gp=gp))
    if body != 'unitree_h2' and fk_err > FK_TOL_CM:
        raise ValueError(f'{body}: AM2 offsets do not reproduce vendor FK at random q ({fk_err} cm)')
    # H2: vendor-equivalent bone offsets in the parent frame vs the AM2 data-tree offsets
    h2_res = None
    if body == 'unitree_h2':
        s0 = snaps[0]; h2_res = {}
        for j, pa in enumerate(parents):
            if pa >= 0:
                ov = s0['r'][pa].T @ (s0['p'][j] - s0['p'][pa])
                h2_res[int(j)] = float(np.linalg.norm(ov - off[j]))
    anchors, mats, trans, approx, resid, gnames = [], [], [], [], [], []
    chains = {}
    for k, g in enumerate(geoms):
        gb = int(model.geom_bodyid[g]); rb = float(model.geom_rbound[g]) * 100
        best_rigid = []; best_approx = (np.inf, None)
        for j in range(J):
            Ms = [s['r'][j].T @ s['gr'][k] for s in snaps]
            ts = [s['r'][j].T @ (s['gp'][k] - s['p'][j]) for s in snaps]
            er = max(float(np.abs(M - Ms[0]).max()) for M in Ms)
            ep = max(float(np.linalg.norm(t - ts[0])) for t in ts)
            if er < RIGID_ROT_TOL and ep < RIGID_POS_TOL_CM:
                pref = 0 if gb in (int(node_body[j]), int(node_frame[j])) else 1
                best_rigid.append((pref, float(np.linalg.norm(ts[0])), j, Ms[0], ts[0]))
            res = max(float(np.linalg.norm(t - ts[0])) + _rot_angle(M @ Ms[0].T) * rb for M, t in zip(Ms, ts))
            if res < best_approx[0]:
                best_approx = (res, (j, Ms[0], ts[0]))
        gnames.append(bname(gb))
        if best_rigid:
            best_rigid.sort(key=lambda t: (t[0], t[1])); _, _, j, M0, t0 = best_rigid[0]
            anchors.append(j); mats.append(M0); trans.append(t0); approx.append(False); resid.append(0.0)
            continue
        ch = _merged_chain(model, data, parents, node_body, node_frame, root_body, gb, g, snaps, k)
        if ch is not None:
            chains[k] = ch
            anchors.append(ch['node']); mats.append(np.eye(3)); trans.append(np.zeros(3)); approx.append(False)
            resid.append(ch['verify_residual_cm'])
            continue
        j, M0, t0 = best_approx[1]
        anchors.append(j); mats.append(M0); trans.append(t0); approx.append(True); resid.append(best_approx[0])
    G = len(geoms)
    region = np.array([region_of(body, gnames[k], int(model.geom_bodyid[geoms[k]]) == root_body, float(snaps[0]['gp'][k][1]))
                       for k in range(G)], np.int16)
    hulls = [hull_vertices(model, g) for g in geoms]
    hull_num = np.array([len(h) for h in hulls], np.int32); hull_adr = np.concatenate([[0], np.cumsum(hull_num)[:-1]]).astype(np.int32)
    hull = np.concatenate([h for h in hulls if len(h)] or [np.zeros((0, 3))], 0)
    arrays = dict(geom_id=np.asarray(geoms, np.int32), anchor=np.asarray(anchors, np.int32),
                  mat=np.asarray(mats, np.float64), trans_cm=np.asarray(trans, np.float64),
                  approx=np.asarray(approx, bool), residual_cm=np.asarray(resid, np.float64),
                  geom_type=np.asarray([int(model.geom_type[g]) for g in geoms], np.int32),
                  geom_size=np.asarray([model.geom_size[g] for g in geoms], np.float64),
                  geom_rbound=np.asarray([model.geom_rbound[g] for g in geoms], np.float64),
                  region=region, hull=hull, hull_adr=hull_adr, hull_num=hull_num,
                  parents=parents, hop=hop_matrix(parents),
                  rest_rot=snaps[0]['r'], rest_pos_cm=snaps[0]['p'])  # vendor qpos0 in AM2 convention (diagnostics only)
    arrays.update(_pack_chains(chains, G))
    # self-penetration pair filter (geometry only): non-adjacent nodes (hop>=2), not interpenetrating at
    # qpos0 (> -0.5 cm), and not penetrating (< -1 cm) in more than half of 32 random states.
    pairs = []
    hop = arrays['hop']
    cand = [(a, b) for a in range(G) for b in range(a + 1, G) if hop[anchors[a], anchors[b]] >= 2]
    st = random_states(model, 32, seed=7)
    dist = np.zeros((len(st), len(cand)))
    for si, q in enumerate(st):
        data.qpos[:] = q; mujoco.mj_forward(model, data)
        for ci, (a, b) in enumerate(cand):
            dist[si, ci] = mujoco.mj_geomDistance(model, data, geoms[a], geoms[b], 0.05, None)
    excluded = []
    for ci, (a, b) in enumerate(cand):
        rest, frac = dist[0, ci], float((dist[1:, ci] < -0.01).mean())
        if rest > -0.005 and frac <= 0.5:
            pairs.append((a, b))
        else:
            excluded.append({'a': gnames[a], 'b': gnames[b], 'rest_dist_cm': float(rest * 100), 'random_pen_frac': frac})
    arrays['pairs'] = np.asarray(pairs, np.int32).reshape(-1, 2)
    # rest-pose geometry table for review
    data.qpos[:] = model.qpos0; mujoco.mj_forward(model, data)
    rest_lowest = geom_lowest_cm_from(arrays, *pose_geoms_arrays(arrays, snaps[0]['r'][None], snaps[0]['p'][None]))[0]
    manifest = dict(schema=SCHEMA, body=body, xml=xml, xml_sha256=sha256_file(xml), mujoco_version=mujoco.__version__,
                    am2_static_spec=spec_path, am2_static_spec_sha256=spec_sha, am2_joint_names=names,
                    node_body=[bname(i) for i in node_body], node_frame_body=[bname(i) for i in node_frame],
                    random_states=len(states), fk_max_error_cm=None if body == 'unitree_h2' else fk_err,
                    h2_node_map=H2_NODE_MAP if body == 'unitree_h2' else None,
                    h2_offset_residual_cm=h2_res, h2_offset_residual_max_cm=None if h2_res is None else max(h2_res.values()),
                    n_geoms=G, n_rigid=int((~arrays['approx']).sum()) - len(chains), n_chain=len(chains),
                    n_approx=int(arrays['approx'].sum()),
                    chain_geoms=[dict(geom=int(geoms[k]), body=gnames[k], node=names[c['node']], parent_node=names[c['parent']],
                                      chain=[bname(b) for b in c['bodies']], verify_residual_cm=c['verify_residual_cm'])
                                 for k, c in chains.items()],
                    approx_residual_max_cm=float(arrays['residual_cm'].max()) if G else 0.0,
                    geoms=[dict(geom=int(g), body=gnames[k], node=names[anchors[k]], region=REGIONS[region[k]],
                                type=int(arrays['geom_type'][k]), approx=bool(approx[k]), chain=k in chains,
                                residual_cm=float(resid[k]),
                                rest_lowest_cm=float(rest_lowest[k]), rest_centre_cm=snaps[0]['gp'][k].tolist())
                           for k, g in enumerate(geoms)],
                    regions_present=sorted({REGIONS[i] for i in region.tolist()}),
                    regions_missing=[r for i, r in enumerate(REGIONS) if i not in set(region.tolist())],
                    selfpen_pairs=len(pairs), selfpen_candidate_pairs=len(cand), selfpen_excluded=excluded,
                    selfpen_rule='hop(anchor)>=2, rest distance > -0.5 cm at qpos0, penetrating (<-1cm) in <=50% of 32 random q',
                    independence='vendor MJCF collision geometry only; no visual surface, sole table, labels or loss')
    return arrays, manifest


def build_all(bodies, out_dir=BINDING_DIR):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    for body in bodies:
        arrays, man = build_binding(body)
        np.savez(out_dir / f'{body}.npz', **arrays)
        man['npz_sha256'] = sha256_file(out_dir / f'{body}.npz')
        (out_dir / f'{body}.json').write_text(json.dumps(man, indent=1))
        report[body] = {k: man[k] for k in ('n_geoms', 'n_rigid', 'n_approx', 'approx_residual_max_cm', 'fk_max_error_cm',
                                              'h2_offset_residual_max_cm', 'selfpen_pairs', 'regions_missing')}
        print(body, json.dumps(report[body]), flush=True)
    return report


# ----------------------------------------------------------------------------------------------
# judge time (numpy; no exp_topo import)
# ----------------------------------------------------------------------------------------------
class Binding:
    def __init__(self, body, directory=BINDING_DIR):
        directory = Path(directory)
        self.body = body
        self.manifest = json.loads((directory / f'{body}.json').read_text())
        if sha256_file(directory / f'{body}.npz') != self.manifest['npz_sha256']:
            raise ValueError(f'{body}: binding npz hash mismatch')
        with np.load(directory / f'{body}.npz') as z:
            for k in z.files:
                setattr(self, k, z[k])
        self.G = len(self.geom_id)
        self.J = len(self.parents)
        self.region_mask = np.zeros((N_REGIONS, self.G), bool)
        for k, r in enumerate(self.region):
            self.region_mask[int(r), k] = True
        self.regions_present = self.region_mask.any(1)

    @functools.cached_property
    def mj(self):
        import mujoco
        xml = self.manifest['xml']
        if sha256_file(xml) != self.manifest['xml_sha256']:
            raise ValueError(f'{self.body}: MJCF changed since binding was built')
        m = mujoco.MjModel.from_xml_path(xml)
        return m, mujoco.MjData(m)


@functools.lru_cache(maxsize=32)
def load_binding(body, directory=str(BINDING_DIR)) -> Binding:
    return Binding(body, directory)


def pose_geoms_arrays(b, rot, pos):
    """rot [T,J,3,3] AM2 global, pos [T,J,3] cm root-relative -> geom canonical rot [T,G,3,3], centre [T,G,3] cm."""
    anchor = b['anchor'] if isinstance(b, dict) else b.anchor
    mat = b['mat'] if isinstance(b, dict) else b.mat
    trans = b['trans_cm'] if isinstance(b, dict) else b.trans_cm
    Ra = rot[:, anchor]
    gr = Ra @ mat[None]
    gp = np.einsum('tgab,gb->tga', Ra, trans) + pos[:, anchor]
    cg = b['chain_geom'] if isinstance(b, dict) else getattr(b, 'chain_geom', np.zeros(0, np.int32))
    for i, k in enumerate(cg):
        gr[:, k], gp[:, k], _ = chain_pose(_unpack_chain(b, i), rot, pos)
    return gr, gp


def geom_lowest_cm_from(b, gr, gp):
    """Lowest canonical-y (= world z) point of every collision geom, cm, root-relative frame."""
    get = (lambda k: b[k]) if isinstance(b, dict) else (lambda k: getattr(b, k if k != 'trans_cm' else 'trans_cm'))
    gtype, size, hull, adr, num = get('geom_type'), get('geom_size'), get('hull'), get('hull_adr'), get('hull_num')
    Rw = np.einsum('ab,tgbc,cd->tgad', AX.T, gr, AX)        # world rotation of each geom
    w = Rw[..., 2, :]                                        # world z expressed in geom-local axes [T,G,3]
    T, G = w.shape[:2]
    s = np.empty((T, G))
    for g in range(G):
        t = int(gtype[g]); sz = size[g]; wg = w[:, g]
        if t == 7:  # mesh (convex hull vertices, geom-local metres)
            V = hull[adr[g]:adr[g] + num[g]]
            s[:, g] = (wg @ V.T).min(1)
        elif t == 6:  # box
            s[:, g] = -(np.abs(wg) @ sz)
        elif t == 2:  # sphere
            s[:, g] = -sz[0]
        elif t == 3:  # capsule (radius, half-length along local z)
            s[:, g] = -(np.abs(wg[:, 2]) * sz[1] + sz[0])
        elif t == 5:  # cylinder
            s[:, g] = -(np.abs(wg[:, 2]) * sz[1] + sz[0] * np.sqrt(np.clip(1 - wg[:, 2] ** 2, 0, 1)))
        elif t == 4:  # ellipsoid
            s[:, g] = -np.sqrt(((wg * sz[None]) ** 2).sum(-1))
        else:
            raise ValueError(f'unsupported collision geom type {t}')
    return gp[..., 1] + 100.0 * s


def geom_lowest_cm(b: Binding, rot, pos):
    gr, gp = pose_geoms_arrays(b, rot, pos)
    return geom_lowest_cm_from(b, gr, gp), gr, gp


def selfpen_depth_cm(b: Binding, gr, gp, report_threshold_cm=0.5):
    """Max penetration depth (cm, >=0) over the allowed non-adjacent geom pairs, per frame, via
    mujoco.mj_geomDistance on the re-posed geoms.  Pairs whose bounding spheres cannot overlap by more
    than `report_threshold_cm` are skipped (so depths below that value read as 0)."""
    import mujoco
    m, d = b.mj
    T = gr.shape[0]
    out = np.zeros(T)
    if not len(b.pairs):
        return out
    Rw = np.einsum('ab,tgbc,cd->tgad', AX.T, gr, AX)
    pw = gp @ AX / 100.0          # canonical cm -> world m (AX.T @ x  ==  x @ AX)
    rb = b.geom_rbound
    a, c = b.pairs[:, 0], b.pairs[:, 1]
    cd = np.linalg.norm(pw[:, a] - pw[:, c], axis=-1)                     # [T,K]
    near = cd < (rb[a] + rb[c] - report_threshold_cm / 100.0)
    gid = b.geom_id
    for t in np.flatnonzero(near.any(1)):
        d.geom_xpos[gid] = pw[t]
        d.geom_xmat[gid] = Rw[t].reshape(len(gid), 9)
        best = 0.0
        for k in np.flatnonzero(near[t]):
            dist = mujoco.mj_geomDistance(m, d, int(gid[a[k]]), int(gid[c[k]]), 0.0, None)
            if dist < -best / 100.0:
                best = -dist * 100.0
        out[t] = best
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--build', action='store_true'); ap.add_argument('--bodies', default=','.join(ROBOTS))
    ap.add_argument('--out', default=str(BINDING_DIR)); ap.add_argument('--regions', action='store_true')
    a = ap.parse_args()
    bodies = [x for x in a.bodies.split(',') if x]
    if a.build:
        rep = build_all(bodies, a.out)
        (Path(a.out) / 'build_report.json').write_text(json.dumps(rep, indent=1))
    if a.regions:
        for body in bodies:
            man = json.loads((Path(a.out) / f'{body}.json').read_text())
            print(f'== {body}  missing={man["regions_missing"]}')
            for gg in man['geoms']:
                c = gg['rest_centre_cm']
                print(f"  {gg['body']:32s} node={gg['node']:28s} {gg['region']:12s} type={gg['type']} "
                      f"centre=({c[0]:7.1f},{c[1]:7.1f},{c[2]:7.1f}) low={gg['rest_lowest_cm']:7.1f}"
                      + (f" APPROX {gg['residual_cm']:.2f}cm" if gg['approx'] else ''))


if __name__ == '__main__':
    main()
