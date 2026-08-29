"""xsource_ref — lloyd 0824: four heterogeneous source types -> Greenwich codes ->
X2 reference (feet-defined root, token condition) packed for the WBC generalist.

  A  robot     unitree_g1_29dof real retarget stream (running-turn val clip)
  B  ego-proxy human_smpl pick-up-box clip (stand-in for lifted ego motion)
  C  arm       RBY1 dual-arm platform, scripted in-limit reach-wave (real descriptor)
  D  UMI-proxy same human clip, ONLY R_Wrist enters the encoder (joint-mask sketch)

Per lane: codes -> realize() (X2 qpos30 + token variants + stance/contact root)
-> pack() -> motions/xsource/<lane>.npz, plus kinematic wrist deviation of the
X2 decode vs the source (both reach-normalised, reported at source scale, cm).

Run (ardy):  PYTHONPATH=/home/arenalabs/alphamotion_ctrl \
  /home/arenalabs/miniconda3/envs/ardy/bin/python xsource_ref.py
Then roll each npz with scripts/diag_terms.py (.venv) on the hz3 r12 checkpoint.
"""
from __future__ import annotations
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from amctrl.horizon.am_env import AMEnv, CACHE
from amctrl.horizon.realize import realize
from amctrl.horizon.pack import pack

RBY1 = ("/var/tmp/alphamotion/robot_models/menagerie/rainbow_robotics_rby1/"
        "rby1a_1.2_no_gripper.xml")
CLIP_LOCO = "ACCAD__Female1Running_c3d__C14_-__run_turn_right__(90)_stageii"
CLIP_MANIP = "ACCAD__Male1General_c3d__General_A5_-_Pick_Up_Box_stageii"
OUT = "/home/arenalabs/alphamotion_ctrl/motions/xsource"
T = 150


def rows_of(am, clip, n):
  s = am.seqs[clip]
  return s["start"], min(s["len"], n)


def cached_pose(am, emb, clip, n):
  """Any cached embodiment -> (pose9 on device, spec, dof, gp cm, reach)."""
  from alphamotion.engine.spatial import SkeletonSpec, build_global
  torch = am.torch
  st, n = rows_of(am, clip, n)
  d = np.load(f"{CACHE}/{emb}_spec.npz", allow_pickle=True)
  spec = SkeletonSpec(str(d["name"]), d["parents"], d["rest_offsets"],
                      [str(x) for x in d["joint_names"]])
  dof = np.load(f"{CACHE}/{emb}_dof.npz", allow_pickle=True)["dof"]
  l6 = torch.load(f"{CACHE}/{emb}_paired_local6d.pt", mmap=True)[st:st + n].float()
  g6, gp, reach = build_global(l6, spec, "cpu")
  pose = torch.cat([g6, gp / reach], -1).to(am.device)
  return pose, spec, dof, gp.numpy(), float(reach)


def synth_rby1(am, n):
  """Scripted dual-arm reach-wave inside RBY1's own limit boxes, 30 fps."""
  from scipy.spatial.transform import Rotation as Rot
  from alphamotion.embodiment import registry
  from alphamotion.engine.spatial import build_global
  torch = am.torch
  emb = registry.load_from_xml(RBY1, "rby1")
  spec, dof, rest = emb.spec, emb.dof, emb.rest
  v = np.asarray(dof[:, :18], dtype=np.float64).reshape(-1, 3, 6)
  half, ctr = v[:, :, 4] * np.pi, v[:, :, 5] * np.pi
  ax = v[:, :, :3]
  ax = ax / np.maximum(np.linalg.norm(ax, axis=-1, keepdims=True), 1e-9)
  q = np.tile(ctr, (n, 1, 1))
  t = np.linspace(0, 4 * np.pi, n)
  rng = np.random.default_rng(7)
  for j, nm in enumerate(spec.joint_names):
    if "arm" not in nm.lower():
      continue
    for k in range(3):
      if half[j, k] < 1e-3:
        continue
      ph, f = rng.uniform(0, 2 * np.pi), rng.choice([0.5, 1.0])
      q[:, j, k] = ctr[j, k] + 0.6 * half[j, k] * np.sin(f * t + ph)
  J = spec.J
  L = np.empty((n, J, 3, 3))
  for j in range(J):
    R = (Rot.from_rotvec(ax[j, 0][None] * q[:, j, 0, None]).as_matrix()
         @ Rot.from_rotvec(ax[j, 1][None] * q[:, j, 1, None]).as_matrix()
         @ Rot.from_rotvec(ax[j, 2][None] * q[:, j, 2, None]).as_matrix())
    L[:, j] = R @ np.asarray(rest[j], dtype=np.float64)[None]
  l6 = torch.as_tensor(np.concatenate([L[..., :, 0], L[..., :, 1]], -1),
                       dtype=torch.float32)
  g6, gp, reach = build_global(l6, spec, "cpu")
  pose = torch.cat([g6, gp / reach], -1).to(am.device)
  return pose, spec, dof, gp.numpy(), float(reach)


def sketch_codes(am, pose, keep_names):
  """Encode with ONLY keep_names visible (m4 v2 joint-mask sketch on human)."""
  from alphamotion.engine.descriptor import geom_batch
  torch = am.torch
  n = pose.shape[0]
  keep = [am.hs.joint_names.index(k) for k in keep_names]
  jm = torch.ones(n, am.hs.J, dtype=torch.bool)
  jm[:, keep] = False
  gb = geom_batch(am.G.geom(am.hs, am.hd), n)
  fp, fr = am.G.model.encode_features(pose, gb, joint_mask=jm.to(am.device))
  ip, ir = am.G.model.ints_from_features(fp, fr)
  return torch.cat([ip[0], ir[0]], 1).long()


def wrist_dev(am, codes, src_gp, src_reach, src_spec, src_kws=("wrist", "hand")):
  """X2 decode of the codes vs source wrist path, both reach-normalised, cm at
  source scale."""
  _, pos = am.G.decode_full(codes, am.xs, am.xd)          # [T,J,3] reach-normalised
  x = pos.cpu().numpy()
  def right(spec, kws):
    ids = [i for i, nm in enumerate(spec.joint_names)
           if any(k in nm.lower() for k in kws)]
    r = [i for i in ids if "r" == spec.joint_names[i].lower()[:1]
         or "right" in spec.joint_names[i].lower()]
    return (r or ids)[0]
  si = right(src_spec, src_kws); ti = right(am.xs, ("wrist", "hand"))
  d = np.linalg.norm(src_gp[:, si] / src_reach - x[:, ti], axis=-1) * src_reach
  return (float(d.mean()), float(np.percentile(d, 95)),
          src_spec.joint_names[si], am.xs.joint_names[ti])


def main():
  os.makedirs(OUT, exist_ok=True)
  am = AMEnv.get()
  lanes = []
  pose, spec, dof, gp, reach = cached_pose(am, "unitree_g1_29dof", CLIP_LOCO, T)
  lanes.append(("A_g1_robot", am.G.encode(pose, spec, dof), gp, reach, spec,
                ("wrist", "hand")))
  pose, spec, dof, gp, reach = cached_pose(am, "human_smpl", CLIP_MANIP, T)
  lanes.append(("B_human_ego_proxy", am.G.encode(pose, spec, dof), gp, reach, spec,
                ("wrist", "hand")))
  poseD = pose                                            # same clip for the sketch
  pose, spec, dof, gp, reach = synth_rby1(am, T)
  lanes.append(("C_rby1_arm", am.G.encode(pose, spec, dof), gp, reach, spec,
                ("arm_4",)))
  gpD, reachD = lanes[1][2], lanes[1][3]
  lanes.append(("D_umi_wrist_only", sketch_codes(am, poseD, ["R_Wrist"]),
                gpD, reachD, lanes[1][4], ("wrist", "hand")))
  report = {}
  for tag, codes, gp, reach, sspec, kws in lanes:
    m, p95, sj, tj = wrist_dev(am, codes, gp, reach, sspec, kws)
    qp, tok, stance, dg = realize(codes, seed=0)
    out = f"{OUT}/{tag}.npz"
    pack([(tag, qp, tok)], out)
    report[tag] = dict(frames=int(codes.shape[0]), wrist_mean_cm=round(m, 2),
                       wrist_p95_cm=round(p95, 2), src_joint=sj, x2_joint=tj,
                       skate_p90=dg.get("skate_p90"), flight_frac=dg.get("flight_frac"))
    print(f"[{tag}] wrist dev ({sj}->{tj}): mean {m:.2f}cm p95 {p95:.2f}cm "
          f"| {out}", flush=True)
  json.dump(report, open(f"{OUT}/report.json", "w"), indent=1)
  print("XSOURCE_DONE")


if __name__ == "__main__":
  main()
