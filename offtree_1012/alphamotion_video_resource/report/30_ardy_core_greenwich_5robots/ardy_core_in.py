"""ARDY cskel27 (Core human, 27 joints) npz -> SkeletonSpec in our canonical frame + global rotations, with an FK gauge check."""
import numpy as np, torch
from alphamotion.engine.nets.rotations import SkeletonSpec, matrix_to_rot6d
from alphamotion.embodiment.dof_features import ball_dof
from alphamotion.engine.spatial import fk_pos
ARDY = '/home/arenalabs/Desktop/"be water, robot"/experiment/ardy'
NAMES = ['Hips', 'Spine', 'Spine1', 'Spine2', 'Spine3', 'Neck', 'Head', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'RightHandEnd', 'RightHandThumb1',
         'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'LeftHandEnd', 'LeftHandThumb1', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase']
PAR = np.array([-1, 0, 1, 2, 3, 4, 5, 4, 7, 8, 9, 10, 10, 4, 13, 14, 15, 16, 16, 0, 19, 20, 21, 0, 23, 24, 25])
neutral = torch.load(f'{ARDY}/ardy/assets/skeletons/cskel27/joints.p', map_location='cpu', weights_only=False).squeeze().numpy().astype(np.float64)   # [27,3] m, Y-up (ARDY frame == our canonical: x left, y up, z forward)
OFF = np.zeros((27, 3), np.float32)
for j in range(27):
    if PAR[j] >= 0: OFF[j] = (neutral[j] - neutral[PAR[j]]) * 100
SPEC = SkeletonSpec('ardy_core27', PAR, OFF, NAMES)
_D, _R = ball_dof(27); DOF = torch.as_tensor(_D).float(); REST = torch.as_tensor(_R).float()
IDX = {n: i for i, n in enumerate(NAMES)}
FEET_IDX = {'left': [IDX['LeftFoot'], IDX['LeftToeBase']], 'right': [IDX['RightFoot'], IDX['RightToeBase']]}
LEG_CM = float(np.linalg.norm(neutral[IDX['LeftFoot']] - neutral[IDX['LeftUpLeg']]) * 100)      # hip joint -> ankle joint, neutral pose (robot profiles use hip -> foot)
HEIGHT_CM = float((neutral[:, 1].max() - neutral[:, 1].min()) * 100)

def load_ardy(path):
    d = np.load(path, allow_pickle=True)
    an = [str(n) for n in d['joint_names']]; assert an == NAMES, an
    assert d['joint_parents'].tolist() == PAR.tolist()
    G = d['global_rot_mats'].astype(np.float32); P = d['posed_joints'].astype(np.float32) * 100
    T = len(G); root_cm = P[:, 0].copy()
    gp = fk_pos(matrix_to_rot6d(torch.as_tensor(G)), SPEC)                       # root-relative cm via our FK on our offsets
    err = np.linalg.norm(gp - (P - root_cm[:, None]), axis=-1)                    # [T,27]
    contacts = d['foot_contacts'].astype(np.float32)                              # [T,4] = L heel, L toe, R heel, R toe (ardy/motion_rep/feet.py)
    fps = float(d['fps'])
    # source energy / skate (human joints, no robot model): mean joint speed per second; foot speed while labeled in contact
    v = np.linalg.norm(np.diff(P, axis=0), axis=-1) / 100 * fps                    # m/s [T-1,27]
    per_sec = [float(v[int(s * fps):int((s + 1) * fps)].mean()) for s in range(int(T / fps))]
    foot_v = np.stack([np.linalg.norm(np.diff(P[:, j, [0, 2]], axis=0), axis=-1) * fps for j in FEET_IDX['left'] + FEET_IDX['right']], 1)   # cm/s
    lab = contacts[1:] > 0.5
    skate = float(np.percentile(foot_v[lab], 95)) if lab.any() else 0.0
    return dict(gR=torch.as_tensor(G), gp=torch.as_tensor(gp).float(), root_R=G[:, 0], root_cm=root_cm, fps=fps, text=str(d['text']), fk_err_cm=float(err.mean()), fk_err_max_cm=float(err.max()),
                contacts=contacts, ardy_P=P, joint_speed_per_sec=per_sec, src_skate_p95_cm_s=skate, src_root_h_cm=[float(P[:, 0, 1].min()), float(P[:, 0, 1].max())], src_min_joint_y_cm=float(P[:, :, 1].min()))

if __name__ == '__main__':
    import glob, sys
    print('core27 leg', round(LEG_CM, 1), 'cm; height', round(HEIGHT_CM, 1), 'cm; offsets |.| max', np.linalg.norm(OFF, axis=1).max().round(1))
    for f in sorted(glob.glob(sys.argv[1] + '/*.npz')):
        r = load_ardy(f); print(f.split('/')[-1], 'T', len(r['gR']), 'fps', r['fps'], 'FK err mean', round(r['fk_err_cm'], 3), 'max', round(r['fk_err_max_cm'], 3), 'root y', np.round(r['src_root_h_cm'], 1), 'min joint y', round(r['src_min_joint_y_cm'], 1), 'skate95', round(r['src_skate_p95_cm_s'], 1), 'contacts', r['contacts'].mean(0).round(2), '\n   speed/s', np.round(r['joint_speed_per_sec'], 2))
