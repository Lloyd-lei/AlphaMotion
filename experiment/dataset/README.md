# Stage 1.5 Mixed Mocap Dataset

Target: ~10k samples across multiple sources, unified to SOMA77 skeleton (or
SMPL-X 22 body-only, depending on final design decision).

## Sources

| source | registration | license | local status | estimated size |
|--------|:---:|:---:|:---:|---|
| Kimodo (self-generated) | — | ours | ✅ 2k generating | ~1 GB |
| Gen2Humanoid HY Motion | — | lab internal | ✅ 54 copied | ~40 MB |
| AMASS | required | CC BY-NC | ⬜ manual | ~20 GB compressed |
| HumanML3D | required (via AMASS) | CC BY-NC | ⬜ manual | ~5 GB |
| 100STYLE | free | CC | ⬜ auto | ~500 MB |
| Motion-X | required | non-commercial | ⬜ manual | ~30 GB |

## Manual download steps

### AMASS (biggest free real-mocap collection, ~40 h motion)

1. Register at https://amass.is.tue.mpg.de (free, ~1 day approval for non-edu)
2. Download the SMPL-H body subset (not SMPL-X, smaller and sufficient):
   - Main subsets: `CMU`, `BMLrub`, `BMLhandball`, `HDM05`, `KIT`, `TotalCapture`
3. Place `*.npz` files under `experiment/dataset/amass/` preserving subset folders
4. Run `python prepare_amass.py` to convert to our canonical format

### HumanML3D (AMASS subset with text labels)

1. Follow https://github.com/EricGuo5513/HumanML3D (requires AMASS access)
2. Download their preprocessed `new_joint_vecs/` and `texts/` folders
3. Place under `experiment/dataset/humanml3d/`

### 100STYLE

1. `wget https://www.ianxmason.com/downloads/100STYLE.zip -P experiment/dataset/100style/`
2. Unzip in place
3. BVH format - need converter (use `bvhio` from Kimodo's deps)

### Motion-X (optional for Stage 1.5)

Very large. Register at https://motion-x-dataset.github.io. Only if AMASS + HumanML3D + 100STYLE insufficient.

## Canonical format

All sources end up as a dict of numpy arrays per sample:

```python
{
  "joints_world":    [T, 77, 3] float32   # SOMA77 joints in world frame
  "local_rot_mats":  [T, 77, 3, 3] float32
  "global_rot_mats": [T, 77, 3, 3] float32
  "root_positions":  [T, 3] float32
  "source_id":       int              # 0=kimodo 1=hy 2=amass 3=humanml3d 4=100style
  "prompt":          string (optional) # if source has text annotation
  "action_label":    string (optional) # coarse category
  "subject_id":      string (optional) # for AMASS etc. for held-out test split
}
```

Converters needed:
- `SMPL-H (52 joints) → SOMA77`: uses Kimodo's `kimodo.skeleton.SOMASkeleton30.to_SOMASkeleton77()` or similar
- `SMPL body (22) → SOMA77`: upper-body subset mapping, fingers T-pose
- `BVH → SOMA77`: use bvhio then joint re-order

See `prepare_{amass,humanml3d,100style,hy_motion}.py` for each source.

## Split strategy (4-way)

| split | description | used for |
|---|---|---|
| `train` | mixed sources, 80% | training |
| `val_id` | mixed sources, 10% held out randomly | in-distribution val |
| `val_subject` | all samples from held-out AMASS subjects | subject OOD |
| `val_source` | entire Tencent HY Motion set | cross-source OOD |

`build_split.py` produces `splits.json` that enumerates sample indices per split.
