from surface_retargeting.config import CFG as SRCFG  # machine paths: configs/sr.toml
# metadata-only listing of GRAIL subfolders not mirrored locally (robot/objects/video/bps)
import os, json, time
from huggingface_hub import HfApi
tok = os.path.expanduser('~/.cache/huggingface/token')
api = HfApi(token=open(tok).read().strip() if os.path.exists(tok) else None)
REPO = 'nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL'
out = {}
cats = ['curb','slope','sitting','pickup_ground','pickup_table','stair_p1','stair_p2']
for c in cats:
    subs = [x.path.split('/')[-1] for x in api.list_repo_tree(REPO, path_in_repo=f'data/{c}', repo_type='dataset')]
    for sub in subs:
        if sub in ('recon','meta','object_usd'): continue
        t0=time.time()
        items = list(api.list_repo_tree(REPO, path_in_repo=f'data/{c}/{sub}', repo_type='dataset', recursive=False))
        files = [(x.path.split('/')[-1], x.size) for x in items if getattr(x,'size',None) is not None]
        out[f'{c}/{sub}'] = {'n': len(files), 'bytes': sum(s for _,s in files), 'names': [f for f,_ in files]}
        print(c, sub, len(files), round(out[f'{c}/{sub}']['bytes']/1e9,3),'GB', f'{time.time()-t0:.1f}s', flush=True)
json.dump(out, open(f'{SRCFG.data_root}/diversity/omnicontact_grail/_hf_remaining_tree.json','w'))
