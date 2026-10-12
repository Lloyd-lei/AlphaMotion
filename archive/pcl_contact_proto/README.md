# Point-cloud contact / penetration / slip labeller — spec v0.1 + prototype

Code: `pcl_contact/` (`core.py` labeller, `sources.py` world placement + human hand fix, `floor.py` session floor).
Runners: `run_floor.py`, `run_prototype.py`, `render_grid.py`. Tests: `tests/test_core.py` (6 synthetic tests pass, 0.5 s).
Outputs: `out/report_<body>__<clip>.json`, `out/labels_<body>__<clip>_window.npz`, `out/session_floor.json`, `png/grid_*.png`.
Diagnostics scripts (one-off) in `diag/`. Total disk 31 MB. CPU only; no sealed file modified (assets and trajectory reader imported by path, read-only).

```
PY=/home/arenalabs/miniconda3/envs/ardy/bin/python; export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=
$PY run_floor.py        # ~40 s: session floor for G1 + human, LAFAN subject4/5
$PY run_prototype.py    # ~90 s: 2 bodies x 2 clips, labels on full clip, report full clip + window
$PY render_grid.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH= $PY -m pytest -q -p no:cacheprovider tests/
```

## Spec (defaults in `core.DEFAULTS`)
Input: world-frame posed dense surface `X [T,P,3]` cm, Y-up, every point a material point (rigid link binding, or bone_revolution for human); partner = plane `n·x = c` or a trimesh object mesh plus per-frame pose `T_obj [T,4,4]`.

| label | definition |
|---|---|
| signed distance `d [T,P]` | plane: `n·x − c`. Object: outside-positive (trimesh closest point + `contains`); points more than 15 cm from the bounding sphere are culled (NaN). |
| point contact | hysteresis: enter `d < 1.0 cm`, leave `d > 2.0 cm`, then AND-ed with the owning link's state |
| link / region contact | hysteresis on the group clearance = 3rd-smallest point `d` (robust to one stray point). Despeckle: fill gaps < 2 frames, drop contacts < 3 frames (0.1 s @30 fps). Groups are every anchor link, every region, plus any custom union. |
| penetration | `max(0, −d)`; reported per point, per group max, per frame max. |
| slip speed | tangential velocity (normal component removed) of material points. Positions are Savitzky–Golay(5,2) smoothed, then central-differenced. Slip is measured on the **support layer**: points in contact at t−1, t and t+1 with `d ≤ group min + 0.5 cm`. Group speed = p25 of those points (rolling-invariant). For objects, the velocity is taken **relative to the object**: the finite difference of the point expressed in the object frame. |
| slip judged | only while clearance `< enter` (truly touching) and the support layer is non-empty; NaN otherwise. |
| slip flag | judged and `speed > 4 cm/s`, min 2 frames. **sustained** = excluding the first/last 3 frames of each contact segment; **transition** = the rest. Per-segment slip distance (cm), total and sustained. |
| floor | per capture **session**: median of the lowest foot point on quasi-static stance frames (foot speed < 5 cm/s, foot lowest within 0.5 cm of the body's lowest point), pooled over every clip of the session. `world0` (y = 0) is the control arm. |

Verified by synthetic tests:
- A cylinder rolling at 30 cm/s without slip gives a support-layer slip median < 2 cm/s. Its centre (a joint-centre rule) moves at 30 cm/s.
- Pure sliding at 20 cm/s is measured as 20 ± 0.5 cm/s.
- Penetration depth is exact.
- Hand pads carried by an object that translates (15 cm/s) and yaws (0.5 rad/s) give relative slip < 1 cm/s. Sliding 10 cm/s across its top is measured as 10 ± 0.5 cm/s, and lifting the pads ends contact.
- Object contact is tested **only on synthetic data**: LAFAN has no objects. The object path is ready for UMR object meshes. MuJoCo `mj_geomDistance` on a convex decomposition is a drop-in alternative to trimesh for `object_signed_distance`; it is not implemented.

## World trajectories available (verified)
- `data/trajectory_reader.load_xwbc_trajectory` returns `world_root` in cm, Y-up, from `<cache>/<body>_root_world.pt`.
- These files exist for all 18 bodies in `topo_exp/cache_lafan_v3` (LAFAN, 77 clips) and in `cache_amass_lconv_tfix` (AMASS, 13,281 clips, time-base fixed; `cache_amass_v3` is the pre-fix version). Use tfix, per the timebase-bug memory.
- `world_root_y − 100·root_height_m` is constant: G1 1.25 cm, human 1.44 cm (±0.03), which equals the legacy `h0`. So the source floor is world y = 0, and the legacy floor sits 1.25 / 1.44 cm above it.
- Session floors from the point cloud (`out/session_floor.json`):

  | session | G1 floor | human floor |
  |---|---|---|
  | subject4 | +2.28 cm | −0.99 cm |
  | subject5 | +2.10 cm | −0.86 cm |

  The G1 GMR retarget stands about 2.1–2.3 cm above y = 0. **With `world0`, strict 1 cm contact finds G1 walk foot contact on only 5–10 % of frames, against 70 % with the session floor.** Floor estimation is therefore mandatory, and its value must ship with the labels.

## Prototype results — GMR G1 (`unitree_g1_29dof`, cache_lafan_v3); labels on the full clip
Windows: fall = `fallAndGetUp1_subject4 [4136,4264)`; walk = `walk1_subject5 [1000,1300)`. Legacy = `contact_hard` (height + 10 cm/s joint-speed gate, hysteresis) and `contact_low` (height only), from `cache_lafan_v3/<body>_contact.npz`.

**Walk, full clip (7839 frames)**
- PCL foot contact is 70 % / 71 % (L/R). Legacy `contact_hard` is 51 % / 50 %; legacy `contact_low` is 77 % / 78 %.
- Agreement with legacy_hard is 79 % / 76 % (κ 0.58 / 0.53). Almost all disagreements are PCL-only frames (1536 / 1758 frames; legacy-only frames are 78 / 89).
- On PCL-only frames, the legacy sole gap is p50 3.1 cm: legacy drops them through its speed gate.
- Against legacy_low: agreement 90 % / 91 %, κ 0.75 / 0.77.
- Timing: PCL touchdown is 2–3 frames before legacy, and PCL lift-off is 3 frames after.
- Penetration: 12.6 % of frames exceed 1 cm, maximum 3.7 cm.
- Slip:
  - 86–88 % of contact frames are judged.
  - Sustained slip is on 35 % / 40 % of judged interior frames; speed p50 2.7 / 3.2 cm/s, p90 10 / 12 cm/s.
  - Slip by stance-phase decile (R foot): mid-stance 1.4–3 cm/s, late stance 6.5–10.8 cm/s.
  - At late stance the support layer is at the toe end (0.89–0.93 of foot length) with vertical velocity ≈ 0. This is G1 toe drag at push-off, consistent with the known GMR skating.

**Fall window (128 frames)**
- Penetration is on every frame: max per frame p50 7.2 cm, p95 13.2 cm, max 15.1 cm. Over the full clip, 68 % of frames exceed 1 cm and 51 % exceed 3 cm, with a maximum of 24 cm (R thigh). This is GMR data violating the floor, not a labeller artefact: joint centres themselves reach −10.8 cm (p1, world y).
- Body-contact disagreement with legacy:

  | region | PCL contact | legacy contact | legacy gap on PCL-only frames |
  |---|---|---|---|
  | trunk + pelvis | 75 % | 0 % | p50 10 cm (the joint centre is 10 cm up while the surface lies on the floor) |
  | forearms | 30 % / 44 % | 0 % | 15–21 cm |
  | shins | 75 % / 84 % | 7 % / 29 % | 9–30 cm |

  Overall κ ranges from 0.05 to 0.26 on these regions. Legacy cannot see surface contact far from a joint centre. On the full clip, trunk + pelvis is PCL 27 % vs legacy 1 %.
- Legacy-only frames are rare: mostly the head (158 frames, PCL clearance p50 2.9 cm), plus hand frames where the PCL surface is 1.2–3 cm up.

**human_smpl source, same clips (hand fix on)**
- Walk feet: PCL 70 % vs legacy 57 %, κ 0.65–0.75.
- Sustained foot slip is 27–29 % of judged frames at 4 cm/s, but mid-stance is 0.5–1.3 cm/s. The late-stance excess comes from the support points moving **down** (−5 to −12 cm/s) to −1.5 to −2.4 cm while sliding: the rigid toe digs in as the heel rises.
- LAFAN toes are rigid: the toe rotation is a constant 21.5° from the foot on every frame, so this cannot be fixed from the data.
- **Read human late-stance slip as an asset/source artefact.** Use sustained-slip numbers or the mid-stance deciles as the human noise floor (≈ 1–2 cm/s).

## human_smpl surface sinking (~22–25 cm): diagnosis and fix
The cause is **not** root height or `beta_vis`. `ground_clearance` = world y + 0.14 cm (root-height offset 1.44, beta −1.58), and walking soles sit at −1.45 cm p50. Decomposition (fall clip; worst 5 % frames):

1. **Hand ellipsoid along the forearm — 100 % of the worst frames, ~10 cm.** The leaf wrist bone takes its direction from the forearm. When the palm is planted the forearm points down (y ≈ −0.9), so the 20 cm hand ellipsoid goes straight into the floor. The data's wrist **rotation** does carry the hand direction: in the LAFAN dialect, bones run along local +Z, and `R_elbow @ unit(off_wrist)` reproduces the measured forearm to 0.02° p99. On planted frames, `R_wrist @ unit(off_wrist)` is horizontal (y p50 ≈ 0) with 65–73° wrist flexion.
   - **Fix:** `sources.pose_human_handfix(hand_fix='rot_axis')`. Clip p1 goes from −25.3 to −17.2 cm and window min from −23.7 to −13.5 cm. Hand-min minus wrist height on planted frames goes from p50 −18 to −8.1 / −6.8 cm.
   - `rot_frame` (twist also from rotation, calibrated on walk frames) is not better (p50 −9.5 / −7.7).
   - Residual: the ellipsoid is 9.8 cm thick, against ~3 cm for a real hand. Next step: flatten the semi-axis w to about 1.5–2 cm, or bind MANO-flat-hand vertices to the wrist rotation.
2. **Source penetration, ~4–8 cm.** In the window the wrist and knee joints themselves are at −4 to −10 cm (legacy session stats: subject4 floor p2 = −8.1 cm). This is a property of the data, labelled as penetration and not hidden.
3. **Body girth, ~3 cm, plus dialect.** The fitted betas (beta0 = 1.56, height 187 cm) give pelvis-joint→back 18.3 cm and chest depth 31.6 cm. With betas = 0 these are 15.7 cm and 26.0 cm. Pelvis surface points (radial up to 22 cm) reach −15 cm while the LAFAN pelvis joint is +7 cm.
   - **Fix:** fit bone lengths through the skeleton but keep girth betas near 0. For example, regularise beta0 separately, or fit only the length-related PCs. This needs an asset rebuild (WP-ASSETS), not done here.

After fix 1 the fall window min is −12.4 cm, coming from the non-hand pelvis/shin, which is explained by 2 and 3.

## Assumed / not verified
- The xwbc LAFAN robot streams are GMR retargets ("all robots share GMR sampling" in `LCONV_meta.json`). This was not re-derived here.
- Floor is a flat plane per session. Uneven terrain and stairs (subject5 static-foot p90 is 17 cm, so some clips stand on raised objects) need a terrain heightfield partner. The plane partner is only valid for flat clips; flag the others.
- The thresholds (1 / 2 cm, 4 cm/s, 0.5 cm support layer, p25, 3-frame trim) are first choices. Sensitivity sweeps have not been run.
- Points with `point_data_residual_p95_cm > 3` are masked out of link groups (kuavo / H1_2); G1 has none.
- The AMASS path (`cache_amass_lconv_tfix`) has world roots but was not run in this prototype.
