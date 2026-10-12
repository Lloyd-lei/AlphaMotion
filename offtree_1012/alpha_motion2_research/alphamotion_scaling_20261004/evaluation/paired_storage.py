"""Lossless paired-array I/O with immutable shared reference arrays.

This changes storage only: no inference, metrics, selection or trajectory math.
The reader also accepts the original self-contained v1 NPZ files.
"""
from pathlib import Path
import hashlib,os
import numpy as np

REFERENCE_KEYS=frozenset(['frame_ids','timestamps_seconds','source_joint_names','target_joint_names',
    'source_pose9_reference_not_necessarily_input','reference_global_rot6','reference_position_native',
    'reference_global_rotation','observed_endpoint_indices','target_parents','target_rest_offsets_native',
    'target_static_reach','condition_values_pose9','reference_world_positions_only',
    'surface_patch_ids','surface_valid','surface_area_native_squared',
    'reference_surface_position_native','reference_surface_normal'])
META=('__reference_file__','__reference_sha256__','__reference_values_sha256__')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def array_digest(values):
    h=hashlib.sha256()
    for key in sorted(values):
        a=np.asarray(values[key])
        if a.dtype.hasobject:raise ValueError('Object arrays are not permitted')
        for v in [key,a.dtype.str,repr(a.shape)]:h.update(v.encode());h.update(b'\0')
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()

def atomic_npz(path,values):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.stem+f'.{os.getpid()}.tmp.npz')
    np.savez_compressed(temp,**values);temp.replace(path)

def save_paired_arrays(directory,arrays,*,reference_store=None):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    if any(k.startswith('__reference_') for k in arrays):raise ValueError('Reserved storage metadata key')
    reference={k:v for k,v in arrays.items() if k in REFERENCE_KEYS}
    dynamic={k:v for k,v in arrays.items() if k not in REFERENCE_KEYS}
    required={'frame_ids','timestamps_seconds','reference_position_native','reference_global_rotation'}
    if not required<=set(reference):raise ValueError('Reference identity incomplete')
    values_sha=array_digest(reference)
    store=Path(reference_store) if reference_store is not None else directory.parent/'_references'
    if store.name!='_references':raise ValueError('Reference store must be explicitly named _references')
    target=store/(values_sha+'.npz')
    if target.exists():
        with np.load(target,allow_pickle=False) as z:
            if array_digest(dict(z))!=values_sha:raise ValueError('Existing immutable reference arrays changed')
    else:atomic_npz(target,reference)
    file_sha=sha(target);relative=os.path.relpath(target,directory)
    dynamic.update(dict(zip(META,[np.asarray(relative),np.asarray(file_sha),np.asarray(values_sha)])))
    atomic_npz(directory/'paired_arrays.npz',dynamic)
    return {'schema':'paired-arrays-v2-shared-reference','dynamic_file':'paired_arrays.npz',
        'reference_file':relative,'reference_sha256':file_sha,'reference_values_sha256':values_sha,
        'reference_keys':sorted(reference),'reader':'evaluation.paired_storage.load_paired_arrays',
        'semantics':'Exact original keys, shapes, dtypes and values reconstructed by the reader; no metric or trajectory changes'}

def load_paired_arrays(path):
    path=Path(path)
    with np.load(path,allow_pickle=False) as z:values=dict(z)
    present=[k in values for k in META]
    if not any(present):return values
    if not all(present):raise ValueError('Incomplete paired-array reference metadata')
    relative,file_sha,values_sha=[str(values.pop(k).item()) for k in META]
    reference=(path.parent/relative).resolve()
    if Path(relative).is_absolute() or reference.parent.name!='_references' or reference.name!=values_sha+'.npz':
        raise ValueError('Invalid content-addressed reference location')
    if sha(reference)!=file_sha:raise ValueError('Shared reference file hash changed')
    with np.load(reference,allow_pickle=False) as z:common=dict(z)
    if array_digest(common)!=values_sha or set(common)&set(values):raise ValueError('Shared reference numeric identity changed')
    return {**common,**values}
