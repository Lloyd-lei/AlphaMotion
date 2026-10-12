# umr_dataset: compact paired human to 21-robot motion dataset

This package takes the factory outputs (one `out/<clip>/<robot>.npz` per clip and robot) and turns them into compact,
immutable, checksummed, memory-mapped shards (UMRS v1). It also provides a PyTorch loader that serves paired windows:
the same source clip and the same frames for the human (SMPL-X) and for any of the 21 robots, with strict
contact and slip labels.

Code: `/home/arenalabs/umr_dataset/umr_dataset/`. Dataset root: `/home/arenalabs/umr_dataset/data/`.
Python: `/var/tmp/alphamotion/umr_factory/umr/env/bin/python`, used read-only. The converter needs numpy, torch and
mujoco. mujoco is only needed once per robot, to build its spec. The loader needs only numpy and torch.

## Layout
```
data/manifest.json             every shard: whole-file sha256, size
data/catalog.npz               flat arrays the loader uses: entries (clip, body, shard, row0, n, k0), valid runs, per-clip fps/stride/betas
data/catalog_meta.json         body / clip / shard names, robot joint names, totals
data/clips.jsonl               one line per clip: SMPL-X betas/gender/up axis, dataset, text annotation, objects, source file + sha256
data/robots/<r>/spec.json      joint names, per-joint uint16 range (MJCF limits +/- 1 mrad), MJCF sha256
data/robots/<r>/kin.npz        torch-FK tables (bodies, geoms, 4096 dense surface points, hull vertices, regions)
data/robots/<r>/bounds.json    static decode-displacement bound per region (for the strictness guard)
data/robots/<r>/*.umrs         robot shards: <robot>-{body|hoi}-<batch>-NNN.umrs
data/human/*.umrs              SMPL-X shards (one row per stored frame; object poses for HOI clips)
data/objects/<obj>/            object MJCF + meshes, copied from the source
```

**Shard file** (`container.py`). It holds a magic number, a JSON header and 4096-aligned C-contiguous blobs.
- The header lists each blob's dtype, shape, offset and sha256, the per-shard metadata, and one entry per (clip, body)
  segment. An entry carries `row0`, `n`, `k0`, the valid runs, provenance, the source path/size/mtime/sha256,
  `src_row0`/`src_row_step` (which factory-npz rows hold the stored frames), flag histogram, and guard statistics.
- Shards are written to a `.tmp` file and then hard-linked into place. A shard is never overwritten.

**Frame grid (pairing).** Row `k` of every body of a clip is source frame `k * stride`, with
`stride = floor(fps_src / 30)`. HiPHI and OmniContact go from 90 to 30 fps and LAFAN stays at 30. GRAIL has no integer
stride to 30, so it stays at 25 Hz; `batch['fps']` tells you which.
- The factory may run UMR on decimated frames: the `pilot_*` outputs have `frame_ids` 0, 3, 6, ...
- The converter reads UMR's `frame_ids`, so `k` always refers to the same source instant for all 21 robots and the
  human. This is tested frame-exactly.
- A robot frame dropped by the decimation that carried a factory flag sets bit 128 on its kept neighbour.

## Record formats and measured decode error
Measured on the 7 live entries (18,207 robot frames, 3 human clips). The full report is in `qa/roundtrip_7entries.json`.

| field | storage | error after decode |
|---|---|---|
| root position | float32, exact copy | 0 |
| root orientation | uint64 smallest-three (3 x 20 bit + index + sign) | max 2.8e-6 rad (codec bound 1e-5) |
| joints | uint16 over the MJCF limit range, fixed per robot | max 4.5e-5 rad (step/2) |
| **robot surface (FK, 4096 dense points + hull)** | | **max 0.062 mm**; static bound 0.05-0.20 mm per robot, holds on every frame |
| contact / slip, ground + object, robot + human, 13 regions | uint16 bitmask each | bit-exact |
| flags | uint8: factory bits 1/2/4/8/16, + 64 decode guard, + 128 dropped-frame | factory bits exact |
| clearance (standard tier) | int8 signed log, D0 = 0.5 mm | <= 0.034 mm at 0, <= 0.68 mm within +/-2 cm, 3.5 % relative beyond |
| slip speed (standard tier) | uint8 log | <= 3.5 % relative |
| human SMPL-X | transl float32 exact; 22 x 3 body axis-angle int16 (per-shard scale); hands/jaw/eyes only if non-zero | pose 4.8e-5 rad, **SMPL-X vertices max 0.21 mm** |
| object pose (HOI) | float32 pos + quat, once per clip (human shard), human metric Z-up frame | 6e-4 mm / 1e-7 rad after mapping to a robot frame |

Robot frame = `scale * (human metric Z-up frame) - floor_offset * z`. The loader returns `scale` and `floor_offset`
for each robot slot, and the object pose already mapped into each robot's frame.

## Strictness survives quantisation
`convert.decode_guard` sets flag bit 64 on any frame whose **decoded** pose might break the factory's strict rules:
penetration <= 0.5 mm against ground, object and self, and every labelled contact region within [-0.5, 1] mm.
- **Stage 1 (every frame, numpy).** A configuration-independent bound on how far any point can move under decode
  (`kinematics.static_disp_bound`: chain lengths x per-joint half step, plus geom bounding spheres) proves more than
  99 % of frames safe from the factory's own stored gauges.
- **Stage 2 (GPU torch FK, only the borderline frames, about 0.9 %).**
  - The visual ground gauge is recomputed exactly on the decoded pose.
  - The collision part of the factory's union gauge, plus self and object penetration, are bounded with a per-frame
    displacement bound.
- **Result:** 19 of 18,207 frames newly flagged (0.1 %).
- **End-to-end check (`scripts/strict_recheck.py`).** The factory's own `RobotGeometry` re-measures every frame the
  loader can serve, on both the decoded and the source pose.
  - Union-gauge output: 0 violations, 0 introduced by decode.
  - Visual-gauge outputs under their own gauge: 0 violations over 17,393 frames.
- The loader only ever serves frames with `flags == 0`.

**Gauge versions.** Factory outputs written before 2026-10-04 22:47 were verified with the visual gauge only. Under
the current visual+collision gauge, their sources penetrate on up to 1,745 of 3,940 frames (GR3, up to 6.3 mm).
`--min-gauge union` (the default) therefore skips them until the factory re-runs them.

## Converter: incremental, safe while the factory writes
```
python -m umr_dataset.convert --root data [--tier core|standard|full] [--workers 8] [--settle 120]
python -m umr_dataset.convert --root data --watch --interval 300 --lifetime 36000 --compact-every 12
python -m umr_dataset.compact --root data [--grace-hours 6]     # merge small shards (byte copy), drop dead ones
python -m umr_dataset.verify  --root data [--quick]             # every file + blob sha256
python -m umr_dataset.convert --root data --list-ingested --verify-sha   # factory npz that are safe to prune
```
- An npz is (re)ingested when its (size, mtime) changed **and** its sha256 changed, and it has not been modified for
  `--settle` seconds. The factory writes npz in place.
- A re-ingested (clip, robot) supersedes its old entry. The newest ingest wins, and `compact` reclaims the old rows.
  Replaced files stay at their path for `--grace-hours`, so a running trainer that still holds the old catalog keeps
  working. Reload the catalog each epoch.
- Unreadable or incomplete files are logged and retried on the next cycle.
- The human SMPL-X is resolved through the same loaders UMR used (`sources.py`) and written once per clip.
- **Throughput (measured):** 3,000 entries (8.80 M stored robot frames + 1,000 human clips) in 25.9-41.8 s with 8 workers
  on the shared, loaded box, about 210-340 k stored frames/s.
  - Peak RSS 6.2 GB; GPU at most 2.1 GB of torch cache (FK batch 8192) plus the CUDA context, released after each run.
  - The factory target (50 h x 20 robots per hour) is about 30 k frames/s, so the converter has about 7x headroom.

## Loader
```python
from umr_dataset.dataset import PairedWindows, collate
ds = PairedWindows('/home/arenalabs/umr_dataset/data', window=64, pair_mode='human2robot',   # robot2human | robot2robot | any | multi
                   n_bodies=2, robots=None, source_robots=None, target_robots=None, split='train', val_permille=50,
                   datasets=None, fields=('labels',), seed=0)      # fields += 'clearance' (standard tier), 'exact'/'points' (full tier)
dl = torch.utils.data.DataLoader(ds, batch_size=256, num_workers=8, collate_fn=collate, pin_memory=True,
                                 persistent_workers=True, prefetch_factor=4)
```
- Each sample is one clip and one window `[k, k+W)` on the clip grid, for `n_bodies` bodies.
  - Clips are drawn in proportion to their length.
  - The window is drawn uniformly over the frames where **every** chosen robot is strictly valid.
  - Sampling is a pure function of (seed, epoch, index). Call `ds.set_epoch`.
- Batches are gathered and decoded vectorised in the worker (`__getitems__`). Each shard is mmapped once per worker.
- The val split is a hash of the clip name, the same for all bodies.
- `batch['slotJ']` holds:
  - **Robot:** `qpos [B,W,Qmax]` (MuJoCo qpos, zero-padded) + `q_mask`; `contact / slip / human_contact /
    human_slip / object_* [B,W,13]`; `flags`; `scale`; `floor_offset`; `object_pos/quat` in that robot's frame;
    optional `clearance / human_clearance / slip_speed`.
  - **Human:** SMPL-X `transl, global_orient, body_pose[21], jaw/eyes, hands[15+15], betas[16], gender, up_y,
    floor_z`.
- `batch` also holds `clip, k, fps, bodies`.
- **Throughput (measured,** 1,000 clips / 1.9 GB, B = 256, W = 64, shared box at load average 34**):**

| mode | 0 workers | 4 workers | 8 workers | 16 workers |
|---|---|---|---|---|
| human2robot | 1.1 k | **14.0 k samples/s** (0.90 M frames/s per slot) | 12.9 k | 10.8 k |
| robot2robot | 1.1 k | 16.7 k | 14.1 k | 14.4 k |
| multi | 0.5 k | 6.8 k | 7.0 k | 6.6 k |

  - One process decodes 16 k samples/s.
  - With workers the limit is IPC plus the pin-memory thread, at about 0.9-1.3 GB/s of decoded tensors.
  - With 0 workers, `pin_memory=True` in the main thread costs 5x (2.9 k vs 14.7 k samples/s). Use 4 or more workers.
  - With the page cache dropped (`--cold`): 13.6 k samples/s.
- GPU helpers (`torch_fk.BatchFK`, run in the trainer):
  - body poses / `rot6` (AM2 convention, optional Z-up to Y-up);
  - the 4,096-point surface;
  - strict per-point ground contact recomputed from `qpos`: about 67 k frames/s on the RTX PRO 6000, 3.8 GB at
    256 x 64 frames. On sampled windows it is a subset of the factory's hysteresis point bits, with 0 exceptions.

## Storage (exact record sizes, 21 robots + human, including headers; `scripts/storage_estimate.py`)
**Per robot-hour** (30 fps, mean over 21 robots, without / with object labels):

| tier | MB per robot-hour |
|---|---|
| core | 8.8 / 9.6 |
| standard | 13.0 / 16.6 |
| full (+512 B ground and +512 B object per-point contact bits, fp16 clearances) | 77 / 142 |

Factory npz files (compressed, native fps) take 45-151 MB per robot-hour.

**Corpus totals** (GB):

| corpus | core | standard | full |
|---|---|---|---|
| HiPHI 617 h (308.7 h originals + 308.7 h mirrors) | 143.5 | 212.6 | 1,362 |
| HiPHI 309 h originals (what the factory converts) | 71.7 | 106.3 | 681 |
| GRAIL 62 h (25 Hz, all HOI) | 12.9 | 20.5 | 157 |
| OmniContact 10 h | 3.0 | 4.4 | 31 |
| **617 h + 62 h + 10 h** | **159** | **238** | **1,549** |
| **309 h + 62 h + 10 h** | **88** | **131** | **868** |

**What fits the ~300 GB /home budget.** About 59 GB of that is already used by `umr_factory_data`, plus the
factory's working set.
- The **full tier cannot fit.** Drop the per-point contact bits; `BatchFK` recomputes strict point contact on the GPU.
- Also dropped: `qpos_umr` (the raw UMR pose before projection), `flags_initial`, float clearances (int8 in
  standard), and the visual/collision split of the gauges.
- **Recommended: standard tier on originals only, 131 GB.**
- Using the 617 h total doubles HiPHI with mirrors the factory does not retarget. If mirrors are ever retargeted:
  - standard for originals + core for mirrors = 203 GB;
  - or all core = 159 GB.
- Factory npz must be pruned after ingest (`--list-ingested`): 21 robots x 381 h at 45-151 MB per robot-hour is
  0.36-1.2 TB.

## Tests / QA
- `python tests/test_umr_dataset.py` runs these tests (all pass):
  - codecs;
  - container checksum and corruption detection;
  - 3-round incremental ingest;
  - unchanged files skipped, and same-content touch skipped;
  - a changed file supersedes its entry;
  - compaction keeps every decoded window identical (3,240 windows);
  - 256 paired windows equal the factory npz and the SMPL-X source frame-exactly.
- Other QA scripts:
  - `scripts/roundtrip_check.py` (quantisation, label bit-exactness, SMPL-X vertex error);
  - `scripts/strict_recheck.py` (factory gauge on decoded frames);
  - `scripts/bench_loader.py`;
  - `scripts/storage_estimate.py`.
