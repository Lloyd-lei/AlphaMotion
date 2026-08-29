"""Render the ORIGINAL human motion behind a horizon window (lloyd 0828: "this
doesn't look like a kick").  The green reference in every rollout video is the
Greenwich decode onto X2; this shows what the human actually did, so a family
label that came from a regex on the clip NAME can be checked against the clip.

  ardy: PYTHONPATH=/home/arenalabs/alphamotion_ctrl python render_source.py --window amass_00492
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MUJOCO_GL", "egl")
from amctrl.horizon.am_env import AMEnv
import spatial5 as S5
import render_newbody as RN

OUT = ('/home/arenalabs/Desktop/"be water, robot"/pose_translator_design/'
       'R4_VIZ/mp4/ctrl_verify/x2/source')


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--store", default="/home/arenalabs/alphamotion_artifacts/horizon_x2_hz4")
  ap.add_argument("--window", required=True); ap.add_argument("--fps", type=int, default=30)
  a = ap.parse_args()
  man = {w["id"]: w for w in json.load(open(f"{a.store}/manifest.json"))["windows"]}
  w = man[a.window]
  am = AMEnv.get()
  _, gp, reach = am.human_pose9(w["clip"], int(w["start"]), int(w["frames"]))
  pos = gp.numpy()                                        # [T,J,3] cm, root-relative, Y-up
  os.makedirs(OUT, exist_ok=True)
  frames = S5.capsule_tile(pos, am.hs.parents, f"HUMAN SOURCE: {w['clip'][:44]}")
  p = f"{OUT}/{a.window}_human.mp4"
  RN.write_mp4(p, np.stack(frames), a.fps)
  # crude motion signature so the label can be checked numerically too
  n = am.hs.joint_names
  iL, iR = n.index("L_Ankle"), n.index("R_Ankle")
  y = pos[:, [iL, iR], 1]; ground = np.percentile(y, 10)
  lift = (y - ground).max(1)
  print(f"{a.window}  clip={w['clip']}")
  print(f"  最高抬脚 {lift.max():.0f} cm | 抬脚>15cm 的帧占比 {100*(lift>15).mean():.0f}%")
  print(f"  骨盆水平位移 {np.linalg.norm(pos[-1,0,[0,2]]-pos[0,0,[0,2]]):.0f} cm")
  print(f"  -> {p}")


if __name__ == "__main__":
  main()
