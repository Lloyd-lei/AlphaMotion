"""ARDY g1skel34 npz -> our unitree_g1_29dof (36-node) global rotations + joint angles, with an FK gauge check."""
import numpy as np, torch
from alphamotion.embodiment import registry
from alphamotion.engine.descriptor import build_from_mjcf
from alphamotion.engine.constraints import fk_from_angles, project
from alphamotion.engine.nets.rotations import matrix_to_rot6d, rot6d_to_matrix, geodesic_deg

eg = registry.load('unitree_g1_29dof'); SPEC, DOF, REST = eg.spec, torch.as_tensor(eg.dof).float(), torch.as_tensor(eg.rest).float()
_, _, _, QN, _ = build_from_mjcf(eg.xml, 'g1x'); GMAP = {n: (k, s) for k, names in enumerate(QN) for s, n in enumerate(names)}
NAMES = list(SPEC.joint_names); PAR = np.asarray(SPEC.parents)
def ardy_name(our):
    if our == 'pelvis': return 'pelvis_skel'
    if our.endswith('_toe_link'): return our.replace('_toe_link', '_toe_base')
    if our.endswith('_rubber_hand'): return our.replace('_rubber_hand', '_hand_roll_skel')
    if our == 'waist_roll_link': return 'waist_pitch_skel'          # our waist_roll_link carries roll+pitch = torso frame
    if our in ('head_link', 'head_mocap', 'imu_in_torso'): return None
    return our.replace('_link', '_skel')

def load_ardy(path):
    d = np.load(path, allow_pickle=True)
    if 'joint_names' in d.files and len(d['joint_names']) > 0: an = [str(n) for n in d['joint_names']]
    else:                                                            # scripts/generate.py omits names; g1skel34 order is fixed (benchmark npz carries it)
        an = [str(n) for n in np.load('/home/arenalabs/Desktop/mingrun/outputs_ardy/original/motions/itching_left_buttock_002__A030__gen00.npz', allow_pickle=True)['joint_names']]
        assert d['global_rot_mats'].shape[1] == len(an)
    idx = {n: i for i, n in enumerate(an)}
    G = d['global_rot_mats'].astype(np.float32); P = d['posed_joints'].astype(np.float32) * 100   # canonical cm, Y-up, x left, z forward
    T = len(G); gR = np.zeros((T, len(NAMES), 3, 3), np.float32); have = np.zeros(len(NAMES), bool)
    for j, n in enumerate(NAMES):
        a = ardy_name(n)
        if a is not None: gR[:, j] = G[:, idx[a]]; have[j] = True
    rest = REST.numpy()
    for j, n in enumerate(NAMES):                                   # DOF-less nodes: parent @ own rest (zero residual for the fit)
        if not have[j]: gR[:, j] = np.einsum('tab,bc->tac', gR[:, PAR[j]], rest[j])
    r6p, gp, q = project(torch.as_tensor(matrix_to_rot6d(torch.as_tensor(gR))), SPEC, DOF, rest=REST)
    q = q.numpy(); root_R = G[:, idx['pelvis_skel']]; root_cm = P[:, idx['pelvis_skel']]
    gRf, gpf = fk_from_angles(torch.as_tensor(q), SPEC, DOF, rest=REST, root_R=torch.as_tensor(root_R))
    # gauge: our FK (root-relative) vs ARDY posed joints (root-relative)
    errs = {}
    for j, n in enumerate(NAMES):
        a = ardy_name(n)
        if a is None: continue
        errs[n] = float(np.linalg.norm(gpf[:, j].numpy() - (P[:, idx[a]] - root_cm), axis=1).mean())
    rot_err = float(geodesic_deg(gRf[:, have], torch.as_tensor(gR)[:, have]).mean())
    contacts = d['foot_contacts'].astype(np.float32)                 # [T,4]: order unknown -> map by name below
    return dict(q=q, root_R=root_R, root_cm=root_cm, gR=gRf, gp=gpf, fps=float(d['fps']), text=str(d['text']),
                fk_err_cm=errs, rot_fit_deg=rot_err, contacts=contacts, ardy_P=P, ardy_names=an)

if __name__ == '__main__':
    import glob, sys
    for f in sorted(glob.glob('/home/arenalabs/Desktop/mingrun/outputs_ardy/original/motions/*.npz')):
        r = load_ardy(f); e = r['fk_err_cm']
        print(f.split('/')[-1], 'rot fit', round(r['rot_fit_deg'], 2), 'deg; FK err cm max', round(max(e.values()), 2), 'mean', round(np.mean(list(e.values())), 2),
              'worst', sorted(e.items(), key=lambda x: -x[1])[:3])
