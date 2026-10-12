"""Human-source resolution for a factory output (provenance -> SMPL-X params exactly as UMR consumed them).

Mirrors UMR d1cdaed loaders (read-only replica, verified on the factory outputs):
  * kind 'smplx': smpl_surface_retarget_common.load_motion_collection(source_data)[seq_key]
                  (npz keys poses|pose_aa|root_orient+pose_body, trans|trans_orig, betas|beta, fps keys, output_up)
  * kind 'hoi'  : humanoid_retarget_pipeline_hsi_hoi.load_samp_motion(seq_dir) with seq_dir from
                  sequence_dir_candidates(source_data, seq_key): GRAIL pickle (via UMR import), packed
                  motion_actor_smplx.npz (HiPHI), or the flat *.npy layout (OmniContact).
"""
from __future__ import annotations
from surface_retargeting.config import CFG as SRCFG  # machine paths: configs/sr.toml

import json
import shutil
import sys
from pathlib import Path

import numpy as np

UMR_SCRIPTS = Path(f'{SRCFG.umr_root}/scripts')


def _scalar(z, key, default=None):
    if key not in z:
        return default
    v = np.asarray(z[key])
    return v.item() if v.shape == () else v.reshape(-1)[0]


def _scalar_npy(path: Path, default=None):
    if not path.exists():
        return default
    v = np.load(path, allow_pickle=True)
    return v.item() if v.shape == () else v.reshape(-1)[0]


def _load_smplx_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as z:
        if 'poses' in z:
            pose = np.asarray(z['poses'], np.float32).reshape(len(z['poses']), -1)
        elif 'pose_aa' in z:
            pose = np.asarray(z['pose_aa'], np.float32).reshape(len(z['pose_aa']), -1)
        elif 'root_orient' in z and 'pose_body' in z:
            ro = np.asarray(z['root_orient'], np.float32).reshape(-1, 3)
            pose = np.concatenate([ro, np.asarray(z['pose_body'], np.float32).reshape(len(ro), -1)], 1)
        else:
            raise KeyError(f'{path}: no poses')
        tk = 'trans' if 'trans' in z else ('trans_orig' if 'trans_orig' in z else None)
        trans = np.asarray(z[tk], np.float32).reshape(len(pose), 3) if tk else np.zeros((len(pose), 3), np.float32)
        bk = 'betas' if 'betas' in z else ('beta' if 'beta' in z else None)
        betas = np.asarray(z[bk], np.float32).reshape(-1) if bk else np.zeros(10, np.float32)
        fps = float(_scalar(z, 'mocap_frame_rate', _scalar(z, 'mocap_framerate', _scalar(z, 'fps', 30.0))))
        up = str(_scalar(z, 'samp_output_up', _scalar(z, 'output_up', 'z'))).lower()
        return dict(poses=pose, trans=trans, betas=betas, gender=str(_scalar(z, 'gender', 'neutral')).lower(),
                    fps=fps, up=up, human_scale=float(_scalar(z, 'human_scale', 1.0)), file=str(path), layout='smplx_npz')


def _hoi_seq_dir(data: Path, seq_key: str) -> Path:
    if data.is_file():
        return data
    if _is_hoi_dir(data) and (not seq_key or data.name == seq_key):
        return data
    cands = [data / seq_key, data / 'train_and_test' / seq_key, data / 'train' / seq_key, data / 'test' / seq_key]
    for c in cands:
        if _is_hoi_dir(c):
            return c
    if (data / 'recon' / f'{seq_key}.pkl').is_file():
        return data / 'recon' / f'{seq_key}.pkl'
    raise FileNotFoundError(f'HOI sequence {seq_key!r} not found under {data}')


def _is_hoi_dir(p: Path) -> bool:
    if p.is_dir() and (p / 'motion_actor_smplx.npz').is_file():
        return True
    has_pose = (p / 'poses.npy').exists() or (p / 'smpl_pose_axis_angle.npy').exists()
    has_trans = (p / 'transl.npy').exists() or (p / 'trans.npy').exists()
    return bool(p.is_dir() and has_pose and has_trans and (p / 'betas.npy').exists())


def _load_hoi(seq: Path) -> dict:
    if seq.is_file() and seq.suffix == '.pkl':                       # GRAIL: defer to UMR's own pickle loader
        if str(UMR_SCRIPTS) not in sys.path:
            sys.path.insert(0, str(UMR_SCRIPTS))
        import humanoid_retarget_pipeline_hsi_hoi as H               # noqa
        m = H.load_samp_motion(seq)
        return dict(poses=m['poses'], trans=m['trans'], betas=m['betas'], gender=m['gender'], fps=m['fps'],
                    up=m['output_up'], human_scale=m.get('human_scale', 1.0), file=str(seq), layout='grail_pkl')
    packed = seq / 'motion_actor_smplx.npz'
    if packed.is_file() and not (seq / 'poses.npy').exists() and not (seq / 'smpl_pose_axis_angle.npy').exists():
        with np.load(packed, allow_pickle=True) as z:
            pk = 'poses' if 'poses' in z.files else 'smpl_pose_axis_angle'
            tk = 'transl' if 'transl' in z.files else 'trans'
            poses = np.asarray(z[pk], np.float32).reshape(len(z[pk]), -1)
            trans = np.asarray(z[tk], np.float32).reshape(len(poses), 3)
            betas = np.asarray(z['betas'], np.float32).reshape(-1)
            gender = str(_scalar(z, 'gender', 'neutral')).lower()
            fps = float(_scalar(z, 'mocap_framerate', _scalar(z, 'fps', 30.0)))
            up = str(_scalar(z, 'output_up', _scalar(z, 'coordinate_up', 'y'))).lower()
        return dict(poses=poses, trans=trans, betas=betas, gender=gender, fps=fps, up=up, human_scale=1.0,
                    file=str(packed), layout='hiphi_packed')
    pp = seq / 'poses.npy'
    if not pp.exists():
        pp = seq / 'smpl_pose_axis_angle.npy'
    tp = seq / 'transl.npy'
    if not tp.exists():
        tp = seq / 'trans.npy'
    raw = np.load(pp)
    poses = np.asarray(raw, np.float32).reshape(-1, raw.shape[-1])
    trans = np.asarray(np.load(tp), np.float32).reshape(len(poses), 3)
    betas = np.asarray(np.load(seq / 'betas.npy'), np.float32).reshape(-1)
    return dict(poses=poses, trans=trans, betas=betas, gender=str(_scalar_npy(seq / 'gender.npy', 'neutral')).lower(),
                fps=float(_scalar_npy(seq / 'mocap_framerate.npy', _scalar_npy(seq / 'fps.npy', 30.0))),
                up=str(_scalar_npy(seq / 'output_up.npy', 'z')).lower(), human_scale=1.0, file=str(seq), layout='flat_npy')


def load_human(prov: dict) -> dict:
    """SMPL-X params of the clip as UMR saw them. Returns poses [T,165|..], trans [T,3], betas, gender, fps, up, ..."""
    kind, data, key = prov['source_kind'], Path(prov['source_data']), str(prov['seq_key'])
    if kind == 'smplx':
        if data.is_dir():
            cand = [data / f'{key}.npz', data / key]
            f = next((c for c in cand if c.is_file()), None)
            if f is None:
                raise FileNotFoundError(f'{key} not in {data}')
        else:
            f = data
        h = _load_smplx_npz(f)
        seq_dir = f.parent
    else:
        seq = _hoi_seq_dir(data, key)
        h = _load_hoi(seq)
        seq_dir = seq if seq.is_dir() else seq.parent
    h['seq_dir'] = str(seq_dir)
    md = seq_dir / 'metadata.json'                                     # HiPHI: text, actor, frame/LU labels
    if md.is_file():
        try:
            m = json.load(open(md))
            h['metadata'] = {k: m[k] for k in ('dataset', 'actor_id', 'actor_metadata', 'frame', 'lu', 'frame_lu',
                                               'motion_id', 'is_hoi', 'mirrored', 'text_annotation', 'duration_sec')
                             if k in m}
            h['metadata']['objects'] = [{k: o.get(k) for k in ('object_id', 'mesh_id', 'object_category')}
                                        for o in m.get('objects', [])]
        except Exception as e:                                          # pragma: no cover
            h['metadata'] = {'error': repr(e)}
    return h


def dataset_tag(prov: dict) -> str:
    s = (prov.get('source_data', '') + '/' + str(prov.get('seq_key', ''))).lower()
    for k in ('hiphi', 'omnicontact', 'grail', 'lafan', 'omomo', 'bones', 'amass'):
        if k in s:
            return 'lafan1' if k == 'lafan' else k
    return 'other'


def copy_object_assets(object_xml: str, dst_root: Path) -> dict:
    """Copy the object MJCF and the mesh files it references into dst_root/<stem>/ (idempotent)."""
    import re
    src = Path(object_xml)
    if not src.is_file():
        return dict(error=f'missing {object_xml}')
    d = dst_root / src.stem
    d.mkdir(parents=True, exist_ok=True)
    txt = src.read_text()
    files = [src]
    for m in re.finditer(r'file="([^"]+)"', txt):
        p = Path(m.group(1))
        p = p if p.is_absolute() else src.parent / p
        if p.is_file():
            files.append(p)
    out = []
    for f in files:
        t = d / f.name
        if not t.exists() or t.stat().st_size != f.stat().st_size:
            shutil.copy2(f, t)
        out.append(str(t))
    return dict(dir=str(d), files=out)
