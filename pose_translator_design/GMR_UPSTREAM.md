# GMR Upstream Sync Check + Hand Retargeting Ecosystem Notes

Date: 2026-08-03. Upstream clone: `/var/tmp/alphamotion/gmr_upstream` (sparse, source-only; assets excluded).
Vendored copy: `alpha_motion2/external/GMR`. Upstream: https://github.com/YanjieZe/GMR, master HEAD = `bb1bbe4` (2026-04-02).

## 1. GMR upstream sync

**Bottom line: our vendored copy IS upstream master HEAD. Byte-identical. Nothing to merge.**

Verified with `diff -rq` over `general_motion_retargeting/`, `scripts/`, `setup.py`, `README.md`, `DOC.md` — the only differences are our local `__pycache__` dirs. Upstream has had no commits since 2026-04-02 (last activity: Xsens MVN live-streaming PR #158).

### (a) ik_limits/safety_break positional-arg bug — NOT fixed upstream

Both our copy and upstream `motion_retarget.py` (lines 181/190/200/210) still call:

```python
mink.solve_ik(self.configuration, self.tasks1, dt, self.solver, self.damping, self.ik_limits)
```

mink's signature is `solve_ik(configuration, tasks, dt, solver, damping=1e-12, safety_break=False, limits=None, ...)`, so `self.ik_limits` lands in the `safety_break` slot. Effect: joint/velocity limits are silently never enforced (`limits=None`), and the truthy list turns `safety_break` on (raises if the configuration exceeds limits). Our production pipeline already works around this with the `install_mink_shim()` monkeypatch in `/var/tmp/alphamotion/gmr_spike/prod_worker.py` (routes a list/tuple from `safety_break` back into `limits`). Keep the shim. Worth filing an upstream issue/PR — it's a one-line keyword-arg fix.

### (b) New robots since our copy — none

`ROBOT_XML_DICT` in `params.py` is identical: 19 entries (unitree_g1, unitree_g1_with_hands, unitree_h1, unitree_h1_2, booster_t1, booster_t1_29dof, stanford_toddy, fourier_n1, engineai_pm01, kuavo_s45, hightorque_hi, galaxea_r1pro, berkeley_humanoid_lite, booster_k1, pnd_adam_lite, tienkung, pal_talos, fourier_gr3).

### (c) Elbow/forearm-offset changes in ik_configs — none pending

The most recent upstream arm-IK change ("fix arm IK quaternion offsets", commit `1919269`, 2026-03-27) only touched `xsens_mvn_to_g1.json` — and our copy already contains it. No changes to `smplx_to_*` or any config our SOMA work derives from. Our tuned elbow/forearm quaternion offsets live only in our own `/var/tmp/alphamotion/gmr_spike/ik_configs/bvh_soma_to_*.json` files and are unaffected.

### (d) Breaking API changes affecting SOMA configs — none

Config schema (`robot_root_name`, `human_root_name`, `ground_height`, `human_height_assumption`, `use_ik_match_table1/2`, `human_scale_table`, `ik_match_table1/2`) and the 5-tuple match-table entry format `[human_body, pos_weight, rot_weight, pos_offset, rot_offset_wxyz]` are unchanged. All 17 SOMA configs in `/var/tmp/alphamotion/gmr_spike/ik_configs/` remain valid as-is.

### Recommendation: NO MERGE

We are already at upstream HEAD; there is nothing to pull. Action items instead:
1. Keep the `solve_ik` shim in `prod_worker.py` (upstream bug still live).
2. Optionally upstream the fix as a PR (one-line: pass `limits=self.ik_limits` as keyword).
3. Re-check upstream in a few months; the repo is low-traffic (4 commits in 2026).

## 2. dex-retargeting quick eval (github.com/dexsuite/dex-retargeting)

Cloned at `/var/tmp/alphamotion/dex-retargeting`. MIT license, pip-installable, CPU-only friendly, robot-agnostic (any URDF via pinocchio + NLopt).

- **Supported hands**: Allegro, Shadow, Schunk SVH, LEAP, Ability, Inspire (left/right each) + Panda gripper. **Unitree Dex3-1 is NOT supported** — no unitree/dex3/g1 hit anywhere in the repo, nor in the sister dex-urdf repo.
- **Input format**: 21 human hand keypoints, (21,3) array in MediaPipe/MANO joint ordering, wrist-local frame. Three modes: `position` (offline; only the 5 fingertip positions matter), `vector` (teleop; wrist-to-fingertip vectors), `dexpilot` (vector + pinch priors).
- **Can it eat SOMA finger joints?** Yes, with a small adapter. Position mode needs just 5 fingertip positions per hand. Catch: SOMA/SMPL-X chains end at the distal joint (index3 etc.), no fingertip point — either extrapolate tips along the distal segment, or edit `target_link_names`/`target_link_human_indices` in the yaml to target distal links instead. Plus: wrist-frame transform (OPERATOR2MANO convention) and a scaling factor. Roughly a 100-line adapter.
- **G1 + Dex3-1 integration effort**: no existing config, but Unitree publishes the Dex3-1 URDF (unitree_ros / unitree_mujoco), so it's config-writing, not authoring: extract hand subtree, one ~15-line yaml per hand (Dex3-1 is 3-fingered, 7 DOF → map thumb/index/middle tips only), the SOMA adapter, then tune + eyeball in a viewer. **Estimate: 2–4 person-days.**

## 3. OmniRetarget

OmniRetarget (arXiv:2509.26633, Sept 2025; Amazon FAR + MIT/Berkeley/Stanford/CMU) is a kinematic data-generation engine for humanoid loco-manipulation, not a policy. It builds an "interaction mesh" spanning the robot, manipulated objects, and terrain, then retargets human MoCap by minimizing the mesh's Laplacian deformation under hard constraints (joint limits, non-penetration, contact preservation) — so hand-object and foot-terrain contacts survive retargeting. Inputs: human motion clips (OMOMO, LAFAN1, in-house MoCap; SMPL-X-style), plus **object meshes and terrain geometry for interaction tasks** — contacts are derived from the mesh formulation, no dense manual contact annotations needed; plain locomotion needs no objects. Output: kinematically feasible whole-body Unitree G1 trajectories with augmentation over object pose/size, terrain, and embodiment. **It is open-source now**: code lives inside Amazon FAR's holosoma repo (github.com/amazon-far/holosoma), with an ~8–9 h public dataset on Hugging Face (omniretarget/OmniRetarget_Dataset), project page omniretarget.github.io. Relevance: complementary to dex-retargeting — OmniRetarget does the whole body incl. object interaction on G1; dex-retargeting does only the hand.
