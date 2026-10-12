import json, sys, numpy as np
from pathlib import Path
ROOT=Path('/home/arenalabs/Desktop/"be water, robot"/alpha_motion2/research/alphamotion_scaling_20261004')
cfg=json.loads((ROOT/'orchestration/configs/A_S_seed0.json').read_text())
xd=Path(cfg['pilot_data_root'])/'data_xwbc'
xm=json.loads((xd/'manifest.json').read_text())
groups=json.loads((ROOT/'data/data_manifest_v2.json').read_text())['xwbc']['groups']
ent=next(e for e in xm['body_payloads'] if e.get('body')=='human_smpl')
z=np.load(xd/ent['payload'],allow_pickle=False)
print(sorted(z.files)[:60])
gi=z['group_index'].astype(int); trust=z['contact_trust'].astype(bool)
ci=z['source_cache_index'] if 'source_cache_index' in z.files else None
fi=z['frame_index'] if 'frame_index' in z.files else None
bad=0
for i,g in enumerate(groups):
    m=gi==i
    if not m.any(): continue
    if ci is not None and not np.all(ci[m]==g['cache_index']): bad+=1
    if fi is not None and not np.all(np.isin(fi[m], g['frame_indices'])): bad+=1
print('groups',len(groups),'mismatched',bad)
df=[i for i,g in enumerate(groups) if g['source_dataset']=='DFaust']
pp=[i for i,g in enumerate(groups) if g['source_dataset']=='PosePrior']
print('DFaust rows',int(np.isin(gi,df).sum()),'groundvalid among them',float(z['ground_valid'][np.isin(gi,df)].mean()) if 'ground_valid' in z.files else None)
for i in pp: 
    m=gi==i; print('PP',i,groups[i]['clip'][:40],'rows',int(m.sum()),'trust_any',bool(trust[m].any()) if m.any() else None,'split',groups[i]['split'])
sys.path.insert(0,str(ROOT))
from am2exp_v1.model import ground
ex=ground.excluded_ground_groups()
print('excluded',len(ex),'DFaust all in',set(df)<=ex,'PosePrior in',sorted(set(pp)&ex))
