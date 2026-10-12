import sys, json, numpy as np
sys.path.insert(0, '/home/arenalabs/Desktop/"be water, robot"/alpha_motion2/research/alphamotion_scaling_20261004/am2exp_v1')
from assets import surface_v2 as S
BODIES=['human_smpl','booster_k1','booster_t1','booster_t1_29dof','fourier_n1','hightorque_hi','pnd_adam_lite','tienkung','unitree_g1_23dof','unitree_g1_29dof','unitree_h1','unitree_h1_2','fourier_gr3','kuavo_s45','unitree_h2','unitree_r1','adroit_pen','relay_franka_full_v2']
def hops(par):
    J=len(par); D=np.full((J,J),99)
    adj=[[] for _ in range(J)]
    for j,p in enumerate(par):
        if p>=0: adj[j].append(p); adj[p].append(j)
    for s in range(J):
        D[s,s]=0; q=[s]
        while q:
            x=q.pop(0)
            for y in adj[x]:
                if D[s,y]>D[s,x]+1: D[s,y]=D[s,x]+1; q.append(y)
    return D
RN=S.REGIONS
out={}
for b in BODIES:
    z=np.load(f'/var/tmp/alphamotion/am2exp/assets/surface_v2/{b}.npz')
    par=z['joint_parents']; D=hops(par)
    anc=z['anchor']; reg=z['region']; rp=z['rest_points']
    res={}
    for tag,gc,gm,ga,gr in (('g256',z['group_centre'],z['group_members'],z['group_anchor'],z['group_region']),('g128',z['group128_centre'],z['group128_members'],z['group128_anchor'],z['group128_region'])):
        cov=len(np.unique(gm))/4096
        dup=int(sum(len(np.unique(r))<len(r) for r in gm))
        rad=np.linalg.norm(rp[gm]-rp[gc][:,None],axis=-1).max(1)
        mix_hop=np.array([D[anc[r]][:,anc[r]].max() for r in gm])
        mix_reg=np.array([len(np.unique(reg[r])) for r in gm])
        # cross-limb mixing: group with members whose anchors are >=3 hops apart AND belong to different limbs
        bad=[]
        for gi,r in enumerate(gm):
            rr=set(RN[x] for x in reg[r])
            if (any('hand' in x or 'forearm' in x for x in rr) and any('thigh' in x or 'pelvis' in x or 'trunk' in x for x in rr)) or ({'L_foot','R_foot'}<=rr) or ({'L_shin','R_shin'}<=rr) or ({'L_thigh','R_thigh'}<=rr):
                bad.append(gi)
        gmaj_ok=float(np.mean([np.bincount(anc[r]).argmax()==a for r,a in zip(gm,ga)]))
        res[tag]=dict(coverage=round(cov,3),dup_groups=dup,radius_p50=round(float(np.median(rad)),1),radius_max=round(float(rad.max()),1),
                      groups_hop_ge3=int((mix_hop>=3).sum()),cross_limb_groups=len(bad),cross_limb_examples=[sorted(set(RN[x] for x in reg[gm[g]])) for g in bad[:3]],
                      anchor_is_majority=gmaj_ok, region_counts_zero=[RN[i] for i in range(15) if b not in ('adroit_pen','relay_franka_full_v2') and not (gr==i).any()])
    sub=z['sub1024']; subreg={RN[r]:int((reg[sub]==r).sum()) for r in np.unique(reg)}
    ap=z['allowed_anchor_pairs']
    # sanity pairs by region-representative anchors
    def anc_of(rn):
        m=reg==RN.index(rn) if rn in RN else None
        return set(np.unique(anc[m]).tolist()) if m is not None and m.any() else set()
    chk={}
    for a,c in (('L_hand','R_hand'),('L_foot','R_foot'),('L_hand','L_thigh'),('L_hand','head'),('L_shin','R_shin'),('L_foot','R_shin'),('L_upper_arm','trunk'),('L_thigh','pelvis')):
        A1,A2=anc_of(a),anc_of(c)
        if A1 and A2:
            chk[f'{a}-{c}']=round(float(np.mean([ap[i,j] for i in A1 for j in A2])),2)
    out[b]=dict(groups=res,sub1024_min_region=min(subreg.values()),sub1024_regions=len(subreg),allowed_pairs=int(ap.sum()//2),pair_checks=chk)
    print(b,json.dumps(out[b]))
json.dump(out,open('groups_review.json','w'),indent=1)
