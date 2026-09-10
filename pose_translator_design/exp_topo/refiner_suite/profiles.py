"""Auto-derive per-robot EE/foot profile from rest FK (no hand-written tables).
For each robot: root, left/right hand joint, left/right foot joint, sole-normal & forward axis in the foot's local frame,
arm length (shoulder->hand at rest), leg length. Self-checks print; failures raise.
Frame: cache convention ardy Y-up (x left, y up, z forward); positions cm, root at origin."""
import os, sys, json, numpy as np, torch
E='/home/arenalabs/Desktop/"be water, robot"/pose_translator_design/exp_topo'; sys.path.insert(0,E)
sys.path.insert(0, os.path.expanduser('~/Desktop/AlphaMotion/src'))
from common import SkeletonSpec
from manifold_project import fk_from_angles, load_dof
C='/var/tmp/alphamotion/topo_exp/cache_lafan_v3'
XML={'agibot_x2_ultra':'/var/tmp/alphamotion/robot_models/agibot_x2_urdf/X2_URDF-v1.4.0/agibot_x2_ultra_mjcf.xml',
     'agibot_a3_ultra_t3d0':'/var/tmp/alphamotion/robot_models/agibot_a3/a3_ultra_t3d0/mjcf/agibot_a3_ultra_t3d0_mjcf.xml',
     'agibot_a3_t3d0':'/var/tmp/alphamotion/robot_models/agibot_a3/a3_t3d0/mjcf/agibot_a3_t3d0_mjcf.xml'}
CACHED=[n for n in sorted(x[:-9] for x in os.listdir(C) if x.endswith('_spec.npz')) if n not in ('human_smpl','pal_talos','engineai_pm01')]

def load(name):
    if name in XML:
        from alphamotion.embodiment import registry as reg
        e=reg.load_from_xml(XML[name], name)
        spec=SkeletonSpec(e.spec.name, np.asarray(e.spec.parents), np.asarray(e.spec.rest_offsets), list(e.spec.joint_names))
        dof=torch.as_tensor(np.asarray(e.dof),dtype=torch.float64); rest=torch.as_tensor(np.asarray(e.rest),dtype=torch.float64)
        return spec,dof,rest,'xml'
    d=np.load(f'{C}/{name}_spec.npz',allow_pickle=True)
    spec=SkeletonSpec(str(d['name']),d['parents'],d['rest_offsets'],[str(x) for x in d['joint_names']])
    dof,rest=load_dof(C,name,device='cpu'); return spec,dof.double(),rest.double(),'cache'

def profile(name, verbose=True):
    spec,dof,rest,src=load(name); J=spec.J; jn=spec.joint_names; par=np.asarray(spec.parents)
    root=int(np.where(par<0)[0][0])
    q0=torch.zeros(1,J,3,dtype=torch.float64)
    gR,gp=fk_from_angles(q0,spec,dof,rest=rest); gR=gR[0].numpy(); gp=gp[0].numpy()   # rest pose, root at origin, Y-up
    children={j:[k for k in range(J) if par[k]==j] for j in range(J)}
    leaves=[j for j in range(J) if not children[j]]
    # depth-first chain classification: feet = lowest leaves (one per side), hands = leaves that are lateral & not feet & not head
    ys=gp[:,1]; xs=gp[:,0]
    feet={}
    for side,sgn in (('left',1),('right',-1)):
        cand=[j for j in range(J) if sgn*xs[j]>0 and j!=root]
        f=min(cand,key=lambda j:ys[j])
        # parallel-linkage leaves (r1 ankle_constraint_*): lowest joint sits <2cm under a parent that fans out -> use the parent
        p=int(par[f])
        if p>=0 and ys[p]-ys[f]<2.0 and len(children[p])>=2: f=p
        feet[side]=f
    heads=[j for j in leaves if abs(xs[j])<5 and ys[j]>10]
    head=max(heads,key=lambda j:ys[j]) if heads else None
    hands={}
    for side,sgn in (('left',1),('right',-1)):
        cand=[j for j in leaves if sgn*xs[j]>0 and j!=feet[side] and j!=head and ys[j]>ys[feet[side]]+15]
        # most lateral leaf on that side (arms hang beside the body at rest for most robots; if arm forward, still most lateral)
        h=max(cand,key=lambda j:abs(xs[j])) if cand else None
        assert h is not None, f'{name}: no hand leaf on {side}'
        # fingered hands (A3 ultra): leaf is a fingertip; ascend to the palm = lateral junction fanning >=3 children
        pth=[h]
        while par[pth[-1]]>=0 and len(children[int(par[pth[-1]])])<2: pth.append(int(par[pth[-1]]))
        junc=int(par[pth[-1]]) if par[pth[-1]]>=0 else -1
        if junc>=0 and len(children[junc])>=3 and abs(xs[junc])>8: h=junc
        hands[side]=h
    # shoulder/hip = top joint of the lateral limb chain: ascend while the parent is lateral (|x|>3cm) or a pure chain link;
    # stop at the root or at a midline junction (torso / waist). Limb length = sum of segment lengths (pose-invariant).
    def branch_root(j):
        path=[int(j)]
        while True:
            p=int(par[path[-1]])
            if p<0 or p==root: break
            if len(children[p])>=2 and abs(xs[p])<3: break
            path.append(p)
        return path
    def chain_len(path): return float(sum(np.linalg.norm(gp[path[i]]-gp[path[i+1]]) for i in range(len(path)-1)))
    out={'name':name,'source':src,'J':int(J),'root':int(root),'root_name':jn[root],'joint_names':list(jn)}
    for side in ('left','right'):
        h=hands[side]; f=feet[side]
        hp=branch_root(h); fp=branch_root(f)
        sh=hp[-1]; hip=fp[-1]
        arm_len=chain_len(hp); leg_len=chain_len(fp)
        Rf=gR[f]; Rh=gR[h]
        out[side]={'hand':int(h),'hand_name':jn[h],'shoulder':int(sh),'shoulder_name':jn[sh],'arm_len_cm':round(arm_len,1),
                   'foot':int(f),'foot_name':jn[f],'hip':int(hip),'hip_name':jn[hip],'leg_len_cm':round(leg_len,1),
                   'foot_rest_pos':[round(float(v),1) for v in gp[f]],'hand_rest_pos':[round(float(v),1) for v in gp[h]],
                   # local-frame axes: world down (-y) and world forward (+z) expressed in the foot's rest frame
                   'sole_normal_local':[round(float(v),4) for v in Rf.T@np.array([0,-1.,0])],
                   'foot_forward_local':[round(float(v),4) for v in Rf.T@np.array([0,0,1.])],
                   'hand_forward_local':[round(float(v),4) for v in Rh.T@np.array([0,0,1.])]}
    out['pelvis_height_rest_cm']=round(float(-min(gp[feet['left'],1],gp[feet['right'],1])),1)
    # self-checks
    chk=[]
    L,R=out['left'],out['right']
    if not (L['foot_rest_pos'][0]>0 and R['foot_rest_pos'][0]<0): chk.append('feet side mismatch')
    if abs(L['arm_len_cm']-R['arm_len_cm'])>2: chk.append(f"arm len asym {L['arm_len_cm']} vs {R['arm_len_cm']}")
    if abs(L['leg_len_cm']-R['leg_len_cm'])>2: chk.append('leg len asym')
    if L['foot_rest_pos'][1]>-20: chk.append('feet not low enough')
    if L['hand']==L['foot'] or L['hand']==R['hand']: chk.append('hand==foot')
    out['checks']=chk or ['ok']
    if verbose:
        print(f"== {name} ({src}) J={J} root={jn[root]}  pelvisH {out['pelvis_height_rest_cm']}cm  checks={out['checks']}")
        for side in ('left','right'):
            s=out[side]; print(f"   {side:5s} hand {s['hand_name']:28s} @ {s['hand_rest_pos']} arm {s['arm_len_cm']}cm (sh {s['shoulder_name']}) | foot {s['foot_name']:24s} @ {s['foot_rest_pos']} leg {s['leg_len_cm']}cm sole_n {s['sole_normal_local']} fwd {s['foot_forward_local']}")
    return out

if __name__=='__main__':
    names=sys.argv[1:] or CACHED+list(XML)
    P={}
    for n in names:
        try: P[n]=profile(n)
        except Exception as ex: print('!! fail',n,repr(ex))
    json.dump(P,open(os.path.expanduser('~/tmp/refiner_suite/profiles.json'),'w'),indent=1)
