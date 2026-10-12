"""E2 hand data (eval-only): D4RL hammer/relocate human-v2 + pen object side-file.

Three steps, each runnable on its own (CPU only):

1. ``download``   (any env, curl) -> OUT/data/e2/d4rl/<ds>/human-v2/data/{main_data.hdf5,metadata.json}
   pinned HF revision, sha256 (LFS oid) / git-blob-sha1 verified.  Never touches the /home HF cache.
2. ``export-raw`` (rot env: h5py)  -> OUT/data/e2/raw/<ds>_raw.npz   (also the local pen hdf5)
3. ``convert``    (ardy env: mujoco) -> OUT/data/e2/hand_<ds>.npz + hand_data_manifest.json + pen_object.npz

``all`` runs 1-3 (step 2 through a PYROT subprocess).

Conventions (identical to data/mujoco_dataset_adapter.py and orchestration/scaling_data.external_body):
canonical = AX @ world(z-up), AX=[[0,1,0],[0,0,1],[1,0,0]]; positions root(forearm)-relative,
training unit cm (metres x100); rotations own-joint global, rest relative: AX (R R_rest^T) AX^T.
Hammer/relocate hand qpos (24 hinges, same names/order as pen) are decoded on the PEN model
(forearm fixed) with the pen adapter ``decode_q``.  Object poses are moved into the pen forearm
frame:  p_can = AX R_fa_pen R_fa_src^T (x_src - P_fa_src),  R_can = AX R_fa_pen R_fa_src^T R_src.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from e2._env import E2_DATA, ROOT, PYROT, PEN_XML, HAMMER_XML, RELOCATE_XML, PEN_HDF5, ADROIT_PAYLOAD, ADROIT_PAYLOAD_JSON, sealed  # noqa: E402
else:
    from ._env import E2_DATA, ROOT, PYROT, PEN_XML, HAMMER_XML, RELOCATE_XML, PEN_HDF5, ADROIT_PAYLOAD, ADROIT_PAYLOAD_JSON, sealed

AX = np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
HF_REPO = 'farama-minari/D4RL'
HF_REVISION = 'a0fd465d1d9d253b075d532b270e03d8fb4f46e6'  # same revision as the local pen/kitchen files
FILES = {
    'hammer': {'main_data.hdf5': {'sha256': 'dba97a4abf115142a303fe425219c607527412b9df6f6e24882afb267f4eea56', 'bytes': 6159280},
               'metadata.json': {'git_sha1': '70f49969eb26685a3c154e09db026adecc9f159f', 'bytes': 2571}},
    'relocate': {'main_data.hdf5': {'sha256': '3a3ef46b93e8ed86daf1593264b16d9d136b13013a7e78a72ad624540b4a545e', 'bytes': 5001616},
                 'metadata.json': {'git_sha1': 'f4f6db0b17eef7da49d8675c0b8f6b96563881b4', 'bytes': 2480}},
}
SPLIT_SEED = 20261004
N_DEV = 10
N_TEST = 15
# Object geometry (from the MJCFs; verified at convert time against the compiled model)
OBJECT_GEOM = {
    'hammer': {'geom': 'handle', 'type': 'capsule', 'radius_m': 0.025, 'half_length_m': 0.05, 'diameter_mm': 50.0},
    'relocate': {'geom': 'sphere', 'type': 'sphere', 'radius_m': 0.035, 'half_length_m': 0.0, 'diameter_mm': 70.0},
    'pen': {'geom': 'pen', 'type': 'cylinder', 'radius_m': 0.015, 'half_length_m': 0.065, 'diameter_mm': 30.0},
}
SOURCE_XML = {'hammer': HAMMER_XML, 'relocate': RELOCATE_XML, 'pen': PEN_XML}
HAND_JOINTS = ['WRJ1', 'WRJ0', 'FFJ3', 'FFJ2', 'FFJ1', 'FFJ0', 'MFJ3', 'MFJ2', 'MFJ1', 'MFJ0', 'RFJ3', 'RFJ2', 'RFJ1', 'RFJ0',
               'LFJ4', 'LFJ3', 'LFJ2', 'LFJ1', 'LFJ0', 'THJ4', 'THJ3', 'THJ2', 'THJ1', 'THJ0']


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def git_blob_sha1(path):
    data = Path(path).read_bytes()
    return hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=True) + '\n')
    tmp.replace(path)


# ----------------------------------------------------------------------------------------------
# 1. download
# ----------------------------------------------------------------------------------------------
def d4rl_path(dataset, name='main_data.hdf5', base=None):
    base = Path(base) if base is not None else E2_DATA / 'd4rl'
    return base / dataset / 'human-v2' / 'data' / name


def _verify(path, spec):
    if not path.exists() or path.stat().st_size != spec['bytes']:
        return False
    if 'sha256' in spec:
        return sha256_file(path) == spec['sha256']
    return git_blob_sha1(path) == spec['git_sha1']


def download(base=None, timeout_s=600):
    """curl from the pinned HF revision into OUT/data/e2/d4rl (NOT the /home HF cache)."""
    base = Path(base) if base is not None else E2_DATA / 'd4rl'
    if str(base.resolve()).startswith('/home/arenalabs/.cache'):
        raise RuntimeError('refusing to write into the /home HF cache')
    env = dict(os.environ)
    env.pop('HF_HUB_OFFLINE', None)  # explicit download step only
    receipt = {'repo': HF_REPO, 'revision': HF_REVISION, 'files': [], 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
               'method': 'curl https://huggingface.co/datasets/<repo>/resolve/<revision>/<path>', 'license': 'CC-BY-4.0',
               'hf_cache_used': False}
    for ds, files in FILES.items():
        for name, spec in files.items():
            dst = d4rl_path(ds, name, base)
            dst.parent.mkdir(parents=True, exist_ok=True)
            rel = f'{ds}/human-v2/data/{name}'
            status = 'present_verified'
            if not _verify(dst, spec):
                url = f'https://huggingface.co/datasets/{HF_REPO}/resolve/{HF_REVISION}/{rel}'
                tmp = dst.with_suffix(dst.suffix + '.part')
                subprocess.run(['curl', '-sS', '-L', '--fail', '-m', str(timeout_s), '-o', str(tmp), url], check=True, env=env,
                               timeout=timeout_s + 30)
                if not _verify(tmp, spec):
                    tmp.unlink(missing_ok=True)
                    raise RuntimeError(f'checksum mismatch for {rel}')
                tmp.replace(dst)
                status = 'downloaded_verified'
            receipt['files'].append({'rfilename': rel, 'path': str(dst), 'bytes': dst.stat().st_size, **{k: v for k, v in spec.items() if k != 'bytes'},
                                     'sha256_actual': sha256_file(dst), 'status': status})
    write_json(base.parent / 'acquisition.json', receipt)
    return receipt


# ----------------------------------------------------------------------------------------------
# 2. export raw (rot env, h5py)
# ----------------------------------------------------------------------------------------------
def _episode_keys(f):
    return sorted(f, key=lambda x: int(x.split('_')[-1]))


def export_raw(base=None, out_dir=None, pen_hdf5=None):
    """Read hdf5 with h5py (rot env) and write flat npz files.  No geometry here."""
    import h5py  # rot env only
    base = Path(base) if base is not None else E2_DATA / 'd4rl'
    out_dir = Path(out_dir) if out_dir is not None else E2_DATA / 'raw'
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = {'hammer': d4rl_path('hammer', base=base), 'relocate': d4rl_path('relocate', base=base),
               'pen': Path(pen_hdf5) if pen_hdf5 is not None else PEN_HDF5}
    report = {}
    for ds, src in sources.items():
        obs, acts, ep_index, frame_index, names, init = [], [], [], [], [], {}
        with h5py.File(src, 'r') as f:
            for e_i, ep in enumerate(_episode_keys(f)):
                o = np.asarray(f[ep]['observations'][...], np.float64)
                a = np.asarray(f[ep]['actions'][...], np.float64)
                if len(o) != len(a) + 1:
                    raise ValueError(f'{ds}/{ep}: observations must be actions+1')
                obs.append(o)
                acts.append(np.concatenate([a, np.full((1, a.shape[1]), np.nan)], 0))
                ep_index.append(np.full(len(o), e_i, np.int32))
                frame_index.append(np.arange(len(o), dtype=np.int32))
                names.append(ep)
                for k, v in f[ep].attrs.items():
                    if k.startswith('options/initial_state_dict/'):
                        init.setdefault(k.split('/')[-1], []).append(np.asarray(v, np.float64))
        payload = {'observations': np.concatenate(obs), 'actions': np.concatenate(acts), 'episode_index': np.concatenate(ep_index),
                   'frame_index': np.concatenate(frame_index), 'episode_names': np.array(names, dtype='U32'),
                   'source_sha256': np.array(sha256_file(src))}
        for k, vals in init.items():
            payload['init_' + k] = np.stack(vals)
        dst = out_dir / f'{ds}_raw.npz'
        np.savez_compressed(dst, **payload)
        report[ds] = {'path': str(dst), 'episodes': len(names), 'frames': int(len(payload['observations'])),
                      'obs_dim': int(payload['observations'].shape[1]), 'init_keys': sorted(init), 'source': str(src)}
    write_json(out_dir / 'raw_manifest.json', report)
    return report


# ----------------------------------------------------------------------------------------------
# 3. convert (ardy env, mujoco)
# ----------------------------------------------------------------------------------------------
def episode_split(dataset, n_episodes, seed=SPLIT_SEED, n_dev=N_DEV):
    """Seeded, content-free split: rank episodes by sha256(seed|dataset|episode_index)."""
    order = sorted(range(n_episodes), key=lambda i: hashlib.sha256(f'{seed}|{dataset}|{i}'.encode()).hexdigest())
    split = np.empty(n_episodes, dtype='U4')
    for r, i in enumerate(order):
        split[i] = 'dev' if r < n_dev else 'test'
    return split


def euler2mat_gym(euler):
    """gymnasium_robotics.utils.rotations.euler2mat (the inverse of the quat2euler used in obs)."""
    euler = np.asarray(euler, dtype=np.float64)
    ai, aj, ak = -euler[..., 2], -euler[..., 1], -euler[..., 0]
    si, sj, sk = np.sin(ai), np.sin(aj), np.sin(ak)
    ci, cj, ck = np.cos(ai), np.cos(aj), np.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk
    mat = np.empty(euler.shape[:-1] + (3, 3), dtype=np.float64)
    mat[..., 2, 2] = cj * ck
    mat[..., 2, 1] = sj * sc - cs
    mat[..., 2, 0] = sj * cc + ss
    mat[..., 1, 2] = cj * sk
    mat[..., 1, 1] = sj * ss + cc
    mat[..., 1, 0] = sj * cs - sc
    mat[..., 0, 2] = -sj
    mat[..., 0, 1] = cj * si
    mat[..., 0, 0] = cj * ci
    return mat


def mat2euler_gym(mat):
    """gymnasium_robotics.utils.rotations.mat2euler."""
    mat = np.asarray(mat, dtype=np.float64)
    eps = np.finfo(np.float64).eps * 4.0
    cy = np.sqrt(mat[..., 2, 2] * mat[..., 2, 2] + mat[..., 1, 2] * mat[..., 1, 2])
    condition = cy > eps
    euler = np.empty(mat.shape[:-1], dtype=np.float64)
    euler[..., 2] = np.where(condition, -np.arctan2(mat[..., 0, 1], mat[..., 0, 0]), -np.arctan2(-mat[..., 1, 0], mat[..., 1, 1]))
    euler[..., 1] = np.where(condition, -np.arctan2(-mat[..., 0, 2], cy), -np.arctan2(-mat[..., 0, 2], cy))
    euler[..., 0] = np.where(condition, -np.arctan2(mat[..., 1, 2], mat[..., 2, 2]), 0.0)
    return euler


SUBSTEP_LAG = 0.2  # one 2 ms physics sub-step of a 10 ms (frame_skip=5) control frame


def sync_to_obs_time(q, episode, lag=SUBSTEP_LAG):
    """q(t) -> q(t - lag frames) by linear interpolation with the previous frame of the same episode.

    The first frame of each episode is the reset observation (computed after mj_forward), so it is exact.
    """
    q = np.asarray(q, np.float64)
    out = q.copy()
    same = np.r_[False, episode[1:] == episode[:-1]]
    out[same] = q[same] - lag * (q[same] - q[np.flatnonzero(same) - 1])
    return out


def extrapolate_from_obs_time(x, episode, lag=SUBSTEP_LAG):
    """Object quantity observed at t - lag -> estimate at t (linear, previous frame of the same episode)."""
    x = np.asarray(x, np.float64)
    out = x.copy()
    same = np.r_[False, episode[1:] == episode[:-1]]
    out[same] = x[same] + lag * (x[same] - x[np.flatnonzero(same) - 1])
    return out


def _named(m, obj, name):
    import mujoco
    i = mujoco.mj_name2id(m, obj, name)
    if i < 0:
        raise KeyError(name)
    return i


def hand_joint_layout(xml):
    """qpos addresses + names of the 24 hand hinges (wrist+fingers) in a model."""
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(xml))
    out = []
    for k in range(m.njnt):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, k)
        if name in HAND_JOINTS:
            out.append((name, int(m.jnt_qposadr[k]), m.jnt_axis[k].tolist(), m.jnt_range[k].tolist()))
    return m, out


def verify_hand_joint_order():
    """Hand joint names/order/axes/ranges identical across pen/relocate/hammer; hand subtree identical."""
    import mujoco
    layouts = {}
    for ds, xml in SOURCE_XML.items():
        m, lay = hand_joint_layout(xml)
        layouts[ds] = {'names': [x[0] for x in lay], 'qposadr': [x[1] for x in lay], 'axes': [x[2] for x in lay], 'ranges': [x[3] for x in lay],
                       'nq': int(m.nq)}
    ref = layouts['pen']
    ok = all(layouts[d]['names'] == ref['names'] == HAND_JOINTS and layouts[d]['axes'] == ref['axes'] and layouts[d]['ranges'] == ref['ranges']
             for d in layouts)
    contiguous = {d: layouts[d]['qposadr'] == list(range(layouts[d]['qposadr'][0], layouts[d]['qposadr'][0] + 24)) for d in layouts}
    # Hand subtree geometry in forearm frame: random hand q, compare body poses relative to forearm.
    rng = np.random.default_rng(0)
    models = {d: mujoco.MjModel.from_xml_path(str(x)) for d, x in SOURCE_XML.items()}
    datas = {d: mujoco.MjData(models[d]) for d in models}
    lo = np.array([r[0] for r in ref['ranges']])
    hi = np.array([r[1] for r in ref['ranges']])
    max_err = 0.0
    hand_bodies = None
    for _ in range(5):
        qh = lo + rng.random(24) * (hi - lo)
        rel = {}
        for d, m in models.items():
            dd = datas[d]
            dd.qpos[:] = m.qpos0
            a0 = layouts[d]['qposadr'][0]
            dd.qpos[a0:a0 + 24] = qh
            mujoco.mj_forward(m, dd)
            fa = _named(m, mujoco.mjtObj.mjOBJ_BODY, 'forearm')
            Rf = dd.xmat[fa].reshape(3, 3)
            Pf = dd.xpos[fa]
            if hand_bodies is None:
                hand_bodies = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(m.nbody)
                               if b != fa and _is_descendant(m, b, fa)]
            ids = [_named(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in hand_bodies]
            rel[d] = np.concatenate([(dd.xpos[ids] - Pf) @ Rf, np.einsum('ji,bjk->bik', Rf, dd.xmat[ids].reshape(-1, 3, 3)).reshape(len(ids), 9)], 1)
        for d in rel:
            max_err = max(max_err, float(np.abs(rel[d] - rel['pen']).max()))
    return {'identical_names_axes_ranges': bool(ok), 'contiguous_qpos': contiguous, 'layouts': layouts,
            'hand_subtree_forearm_frame_max_abs_diff': max_err, 'hand_bodies': hand_bodies, 'passed': bool(ok and all(contiguous.values()) and max_err < 1e-9)}


def _is_descendant(m, b, root):
    while b > 0:
        if b == root:
            return True
        b = int(m.body_parentid[b])
    return False


class SourceScene:
    """Source MJCF (hammer/relocate/pen) used only for FK of forearm, palm site and object geom."""

    def __init__(self, dataset):
        import mujoco
        self.mj = mujoco
        self.dataset = dataset
        self.m = mujoco.MjModel.from_xml_path(str(SOURCE_XML[dataset]))
        self.d = mujoco.MjData(self.m)
        self.fa = _named(self.m, mujoco.mjtObj.mjOBJ_BODY, 'forearm')
        self.obj = _named(self.m, mujoco.mjtObj.mjOBJ_BODY, 'Object')
        self.palm_site = _named(self.m, mujoco.mjtObj.mjOBJ_SITE, 'S_grasp')
        self.geom = _named(self.m, mujoco.mjtObj.mjOBJ_GEOM, OBJECT_GEOM[dataset]['geom'])
        _, lay = hand_joint_layout(SOURCE_XML[dataset])
        self.hand_adr = lay[0][1]
        # geom pose in the Object body frame (static)
        self.geom_pos_body = self.m.geom_pos[self.geom].copy()
        q = self.m.geom_quat[self.geom]
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, q)
        self.geom_R_body = R.reshape(3, 3)
        size = self.m.geom_size[self.geom]
        self.geom_size = size.copy()

    def forward(self, qpos):
        self.d.qpos[:] = qpos
        self.mj.mj_forward(self.m, self.d)
        return self.d


def _object_geometry_world(R_obj, p_obj, scene):
    """World centre + axis (geom z) of the object collision geom from the Object body pose."""
    centre = p_obj + np.einsum('...ij,j->...i', R_obj, scene.geom_pos_body)
    Rg = np.einsum('...ij,jk->...ik', R_obj, scene.geom_R_body)
    return centre, Rg


def convert(base_raw=None, out_dir=None):
    """Convert raw npz to AM2 adroit_pen J25 frames + objects in pen-forearm canonical (cm)."""
    sealed()
    import mujoco
    sys.path.insert(0, str(ROOT / 'data'))
    from mujoco_dataset_adapter import MujocoMotionAdapter, fk_global, static_reach, rot6  # sealed, read-only
    base_raw = Path(base_raw) if base_raw is not None else E2_DATA / 'raw'
    out_dir = Path(out_dir) if out_dir is not None else E2_DATA
    out_dir.mkdir(parents=True, exist_ok=True)
    order = verify_hand_joint_order()
    if not order['passed']:
        raise RuntimeError('hand joint order/subtree differs across XMLs: ' + json.dumps(order)[:2000])
    pen = MujocoMotionAdapter(PEN_XML, 'forearm')
    assert len(pen.names) == 25
    off_cm = (pen.offsets * 100).astype(np.float32)
    reach_cm = static_reach(pen.parents, off_cm.astype(np.float64))
    pen_scene = SourceScene('pen')
    pen_scene.forward(pen_scene.m.qpos0)
    R_fa_pen = pen_scene.d.xmat[pen_scene.fa].reshape(3, 3).copy()
    P_fa_pen = pen_scene.d.xpos[pen_scene.fa].copy()
    manifest = {'schema': 'am2exp_e2_hand_v1', 'joint_order_check': {k: v for k, v in order.items() if k != 'layouts'},
                'pen_adapter': {'xml': str(PEN_XML), 'joint_names': pen.names, 'parents': pen.parents.tolist(), 'reach_cm': reach_cm},
                'canonical': 'AX @ (pen-world - pen forearm origin), cm; rotations own-joint global rest-relative', 'datasets': {}}
    for ds in ['hammer', 'relocate']:
        raw = dict(np.load(base_raw / f'{ds}_raw.npz'))
        obs = raw['observations']
        ep = raw['episode_index']
        n_ep = int(ep.max()) + 1
        split_ep = episode_split(ds, n_ep)
        scene = SourceScene(ds)
        a0 = scene.hand_adr
        nq = scene.m.nq
        N = len(obs)
        # Hand qpos in obs: hammer obs[:, 2:26] (qpos[:-6] = 2 arm + 24 hand + nail), relocate obs[:, 6:30]
        nq_obs = a0 + 24 + (1 if ds == 'hammer' else 0)
        q_rec = obs[:, :nq_obs]
        # TIME SYNC: D4RL/Gymnasium obs take qpos AFTER mj_step but xpos/site_xpos (palm, object) from the
        # forward pass BEFORE the last 2 ms sub-step (frame_skip=5).  Frame time := observation time, so the
        # recorded qpos is moved back by one sub-step (linear, within the episode): q(t-0.2 frame).
        q_obs_time = sync_to_obs_time(q_rec, ep, SUBSTEP_LAG)
        hand_recorded = q_rec[:, a0:a0 + 24]
        hand = q_obs_time[:, a0:a0 + 24]
        q_arm = q_obs_time[:, :a0]
        # Pen-model decode (forearm fixed) of the 24 hand joints.
        qpen = np.repeat(pen.model.qpos0[None], N, 0)
        qpen[:, :24] = hand
        dec = pen.decode_q(qpen)
        G = dec['global_matrices']
        P_cm = dec['position_native'] * 100.0
        fk_err = float(np.abs(fk_global(G, pen.offsets, pen.parents) * 100 - P_cm).max())
        # Source-model forearm pose + palm site per frame (object qpos irrelevant for these).
        Pfa = np.zeros((N, 3))
        Rfa = np.zeros((N, 3, 3))
        palm_w = np.zeros((N, 3))
        qsrc = np.repeat(scene.m.qpos0[None], N, 0)
        qsrc[:, :a0] = q_arm
        qsrc[:, a0:a0 + 24] = hand
        if ds == 'hammer':
            qsrc[:, a0 + 24] = q_obs_time[:, a0 + 24]  # nail_dir slide (qpos[26])
        for i in range(N):
            d = scene.forward(qsrc[i])
            Pfa[i] = d.xpos[scene.fa]
            Rfa[i] = d.xmat[scene.fa].reshape(3, 3)
            palm_w[i] = d.site_xpos[scene.palm_site]
        checks = {}
        if ds == 'hammer':
            # obs = [qpos[:-6](27), qvel[-6:](6), palm(3), obj_pos(3), obj_rot euler(3), target(3), nail_impact(1)] = 46
            assert obs.shape[1] == 46, obs.shape
            palm_obs = obs[:, 33:36]
            p_obj = obs[:, 36:39]
            R_obj = euler2mat_gym(obs[:, 39:42])
            checks['palm_site_vs_obs_max_mm'] = float(np.linalg.norm(palm_w - palm_obs, axis=1).max() * 1000)
        else:
            # obs = [qpos[:-6](30), palm-obj(3), palm-target(3), obj-target(3)] = 39
            assert obs.shape[1] == 39, obs.shape
            if 'init_target_pos' not in raw:
                raise RuntimeError('relocate target position missing from initial_state_dict')
            target = raw['init_target_pos'][ep]
            p_obj = target + obs[:, 36:39]
            R_obj = np.repeat(np.eye(3)[None], N, 0)  # sphere: orientation irrelevant
            palm_minus_obj = palm_w - p_obj
            checks['palm_minus_obj_vs_obs_max_mm'] = float(np.linalg.norm(palm_minus_obj - obs[:, 30:33], axis=1).max() * 1000)
            checks['palm_minus_target_vs_obs_max_mm'] = float(np.linalg.norm(palm_w - target - obs[:, 33:36], axis=1).max() * 1000)
            checks['obs_internal_consistency_max_mm'] = float(np.linalg.norm(obs[:, 30:33] - obs[:, 33:36] + obs[:, 36:39], axis=1).max() * 1000)
        # Initial-frame check: object pose from init qpos (full state) vs reconstructed obs-based pose.
        init_q = raw['init_qpos']
        e_pos, e_rot = [], []
        body_pos0 = scene.m.body_pos[scene.obj].copy()
        for e in range(n_ep):
            i0 = int(np.flatnonzero(ep == e)[0])
            if 'init_obj_pos' in raw:  # relocate reset moves model.body_pos of Object (not qpos)
                scene.m.body_pos[scene.obj] = raw['init_obj_pos'][e]
            d = scene.forward(init_q[e])
            scene.m.body_pos[scene.obj] = body_pos0
            e_pos.append(np.linalg.norm(d.xpos[scene.obj] - p_obj[i0]))
            if ds == 'hammer':
                e_rot.append(np.abs(d.xmat[scene.obj].reshape(3, 3) - R_obj[i0]).max())
            e_hand = np.abs(init_q[e][a0:a0 + 24] - hand[i0]).max()
            if e_hand > 1e-5:
                raise RuntimeError(f'{ds} ep{e}: obs hand qpos != initial state qpos ({e_hand})')
        checks['init_object_pos_max_mm'] = float(max(e_pos) * 1000)
        if e_rot:
            checks['init_object_rot_max_abs'] = float(max(e_rot))
        # Object geometry in world, then pen-forearm canonical.
        centre_w, Rg_w = _object_geometry_world(R_obj, p_obj, scene)
        Rrel = np.einsum('ij,njk->nik', R_fa_pen, np.transpose(Rfa, (0, 2, 1)))   # R_fa_pen R_fa_src^T
        T = np.einsum('ij,njk->nik', AX, Rrel)                                    # AX R_fa_pen R_fa_src^T
        obj_centre_cm = np.einsum('nij,nj->ni', T, centre_w - Pfa) * 100
        obj_body_cm = np.einsum('nij,nj->ni', T, p_obj - Pfa) * 100
        obj_R_can = np.einsum('nij,njk->nik', T, R_obj)
        geom_R_can = np.einsum('nij,njk->nik', T, Rg_w)
        palm_cm = np.einsum('nij,nj->ni', T, palm_w - Pfa) * 100
        # Cross-check: palm site via the PEN model at the same hand q must equal transformed source palm.
        pen_palm = np.zeros((N, 3))
        for i in range(N):
            d = pen_scene.forward(qpen[i])
            pen_palm[i] = AX @ (d.site_xpos[pen_scene.palm_site] - P_fa_pen) * 100
        checks['palm_pen_model_vs_transformed_source_max_mm'] = float(np.linalg.norm(pen_palm - palm_cm, axis=1).max() * 10)
        info = OBJECT_GEOM[ds]
        if not np.allclose(scene.geom_size[0], info['radius_m']) or (info['type'] == 'capsule' and not np.allclose(scene.geom_size[1], info['half_length_m'])):
            raise RuntimeError(f'{ds}: object geom size {scene.geom_size} disagrees with constants {info}')
        axis_can = geom_R_can[:, :, 2]
        seg = np.stack([obj_centre_cm - axis_can * info['half_length_m'] * 100, obj_centre_cm + axis_can * info['half_length_m'] * 100], 1)
        rot6v = rot6(G).astype(np.float32)
        pose9 = np.concatenate([rot6v, (P_cm / reach_cm).astype(np.float32)], -1)
        split = split_ep[ep]
        checks['substep_lag_frames'] = SUBSTEP_LAG
        checks['hand_shift_obs_time_vs_recorded_max_rad'] = float(np.abs(hand - hand_recorded).max())
        payload = {'hand_qpos': hand.astype(np.float64), 'hand_qpos_recorded': hand_recorded.astype(np.float64), 'global_rot6': rot6v, 'position_cm': P_cm.astype(np.float32), 'pose9': pose9,
                   'episode_index': ep.astype(np.int32), 'frame_index': raw['frame_index'].astype(np.int32), 'split': split,
                   'episode_split': split_ep, 'obj_centre_cm': obj_centre_cm.astype(np.float32), 'obj_body_cm': obj_body_cm.astype(np.float32),
                   'obj_R_can': obj_R_can.astype(np.float32), 'obj_geom_R_can': geom_R_can.astype(np.float32), 'obj_segment_cm': seg.astype(np.float32),
                   'obj_radius_cm': np.array(info['radius_m'] * 100, np.float32), 'obj_half_length_cm': np.array(info['half_length_m'] * 100, np.float32),
                   'obj_type': np.array(info['type']), 'obj_diameter_mm': np.array(info['diameter_mm'], np.float32), 'palm_site_cm': palm_cm.astype(np.float32),
                   'src_forearm_pos_w': Pfa.astype(np.float32), 'src_forearm_R_w': Rfa.astype(np.float32), 'reach_cm': np.array(reach_cm, np.float32),
                   'joint_names': np.array(pen.names), 'parents': pen.parents.astype(np.int64), 'rest_offsets_cm': off_cm}
        dst = out_dir / f'hand_{ds}.npz'
        np.savez_compressed(dst, **payload)
        manifest['datasets'][ds] = {
            'path': str(dst), 'sha256': sha256_file(dst), 'episodes': n_ep, 'frames': N, 'fps': 100.0,
            'episode_split': {'dev': [int(i) for i in np.flatnonzero(split_ep == 'dev')], 'test': [int(i) for i in np.flatnonzero(split_ep == 'test')]},
            'frames_by_split': {s: int((split == s).sum()) for s in ['dev', 'test']}, 'split_rule': f'sha256({SPLIT_SEED}|{ds}|episode_index) rank; first {N_DEV} dev',
            'hand_qpos_obs_columns': [a0, a0 + 24], 'source_xml': str(SOURCE_XML[ds]), 'source_nq': int(nq), 'fk_max_err_cm': fk_err,
            'object': info, 'checks': checks}
    # Pen object side-file in payload row order.
    manifest['pen'] = convert_pen_side_file(base_raw, out_dir, pen_scene, P_fa_pen)
    write_json(out_dir / 'hand_data_manifest.json', manifest)
    return manifest


def convert_pen_side_file(base_raw, out_dir, pen_scene, P_fa_pen):
    raw = dict(np.load(Path(base_raw) / 'pen_raw.npz'))
    obs = raw['observations']
    with np.load(ADROIT_PAYLOAD) as z:
        q_payload = z['recorded_robot_qpos']
        split = z['split']
        fidx = z['frame_index']
        group = z['source_group_index']
    if len(q_payload) != len(obs):
        raise RuntimeError('pen payload/hdf5 row count mismatch')
    qerr = float(np.abs(q_payload - obs[:, :24]).max())
    if qerr > 1e-6:
        raise RuntimeError(f'pen payload row order differs from hdf5 ({qerr})')
    # obs = [qpos[:-6](24), obj_pos(3), obj_vel(6), obj_orien(3), desired_orien(3), pos-desired(3), orien-desired(3)] = 45
    p_obj_obs = obs[:, 24:27]
    axis_obs = obs[:, 33:36]
    ep_raw = raw['episode_index']
    # Pen hand rows are the RECORDED qpos (training payload, untouchable); the obs pen pose lags by one
    # sub-step, so the object is extrapolated forward to the qpos time instead (see sync_to_obs_time).
    p_obj = extrapolate_from_obs_time(p_obj_obs, ep_raw)
    axis_w = extrapolate_from_obs_time(axis_obs, ep_raw)
    axis_w = axis_w / np.linalg.norm(axis_w, axis=1, keepdims=True)
    centre_cm = (p_obj - P_fa_pen) @ AX.T * 100
    axis_can = axis_w @ AX.T
    hl = OBJECT_GEOM['pen']['half_length_m'] * 100
    seg = np.stack([centre_cm - axis_can * hl, centre_cm + axis_can * hl], 1)
    # Verify: pen centre/axis from init qpos FK equals obs at episode starts.
    init_q = raw['init_qpos']
    ep = raw['episode_index']
    errs = []
    obj = pen_scene.obj
    for e in range(int(ep.max()) + 1):
        i0 = int(np.flatnonzero(ep == e)[0])
        d = pen_scene.forward(init_q[e])
        errs.append(np.linalg.norm(d.xpos[obj] - p_obj_obs[i0]))
    payload = {'obj_centre_cm': centre_cm.astype(np.float32), 'obj_axis_can': axis_can.astype(np.float32), 'obj_segment_cm': seg.astype(np.float32),
               'obj_centre_cm_obs_time': ((p_obj_obs - P_fa_pen) @ AX.T * 100).astype(np.float32),
               'obj_radius_cm': np.array(OBJECT_GEOM['pen']['radius_m'] * 100, np.float32), 'obj_half_length_cm': np.array(hl, np.float32),
               'obj_type': np.array('cylinder'), 'obj_diameter_mm': np.array(OBJECT_GEOM['pen']['diameter_mm'], np.float32),
               'payload_split': split, 'payload_frame_index': fidx, 'payload_group': group, 'episode_index': raw['episode_index'].astype(np.int32),
               'hand_qpos': q_payload}
    dst = Path(out_dir) / 'pen_object.npz'
    np.savez_compressed(dst, **payload)
    return {'path': str(dst), 'sha256': sha256_file(dst), 'frames': int(len(obs)), 'payload_qpos_max_abs_diff': qerr,
            'init_object_pos_max_mm': float(max(errs) * 1000), 'object_time_sync': 'obs pen pose extrapolated +0.2 frame to the recorded-qpos time',
            'note': 'pen geom is a cylinder r=15mm, half-length 65mm along the Object z (site axis)'}


def load_hand(dataset, out_dir=None):
    out_dir = Path(out_dir) if out_dir is not None else E2_DATA
    with np.load(out_dir / f'hand_{dataset}.npz') as z:
        return {k: z[k] for k in z.files}


def load_pen(out_dir=None):
    """Pen eval frames from the sealed payload + object side-file, in the same layout as load_hand()."""
    out_dir = Path(out_dir) if out_dir is not None else E2_DATA
    with np.load(out_dir / 'pen_object.npz') as z:
        side = {k: z[k] for k in z.files}
    with np.load(ADROIT_PAYLOAD) as z:
        pay = {k: z[k] for k in ['global_rot6', 'position_native', 'parents', 'rest_offsets', 'joint_names', 'split', 'frame_index', 'recorded_robot_qpos']}
    sys.path.insert(0, str(ROOT / 'data'))
    from mujoco_dataset_adapter import static_reach
    off_cm = pay['rest_offsets'].astype(np.float32) * 100
    reach = static_reach(pay['parents'], off_cm.astype(np.float64))
    P_cm = pay['position_native'] * 100
    pose9 = np.concatenate([pay['global_rot6'], (P_cm / reach)], -1).astype(np.float32)
    split = np.where(pay['split'] == 0, 'train', 'eval')
    return {'hand_qpos': pay['recorded_robot_qpos'], 'global_rot6': pay['global_rot6'], 'position_cm': P_cm.astype(np.float32), 'pose9': pose9,
            'episode_index': side['episode_index'], 'frame_index': pay['frame_index'].astype(np.int32), 'split': split,
            'obj_centre_cm': side['obj_centre_cm'], 'obj_segment_cm': side['obj_segment_cm'], 'obj_radius_cm': side['obj_radius_cm'],
            'obj_half_length_cm': side['obj_half_length_cm'], 'obj_type': side['obj_type'], 'obj_diameter_mm': side['obj_diameter_mm'],
            'reach_cm': np.array(reach, np.float32), 'joint_names': pay['joint_names'], 'parents': pay['parents']}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('step', choices=['download', 'export-raw', 'convert', 'all'])
    ap.add_argument('--out', type=Path, default=E2_DATA)
    args = ap.parse_args(argv)
    if args.step in ('download', 'all'):
        print(json.dumps(download(args.out / 'd4rl'), indent=1)[:3000])
    if args.step == 'export-raw':
        print(json.dumps(export_raw(args.out / 'd4rl', args.out / 'raw'), indent=1))
    if args.step == 'all':
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', CUDA_VISIBLE_DEVICES='')
        subprocess.run([PYROT, str(Path(__file__).resolve()), 'export-raw', '--out', str(args.out)], check=True, env=env, timeout=1200)
    if args.step in ('convert', 'all'):
        man = convert(args.out / 'raw', args.out)
        print(json.dumps({k: v for k, v in man['datasets'].items()}, indent=1, default=str)[:6000])
        print(json.dumps(man['pen'], indent=1))


if __name__ == '__main__':
    main()
