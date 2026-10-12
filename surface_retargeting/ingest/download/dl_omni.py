import os, sys, time
os.environ.pop('HF_HUB_OFFLINE', None)
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = os.environ.get('HF_HUB_ENABLE_HF_TRANSFER', '0')
from huggingface_hub import snapshot_download
pats = sys.argv[1].split(',')
t = time.time()
p = snapshot_download('lightcone02/OmniContact-Dataset', repo_type='dataset',
    local_dir='/home/arenalabs/umr_factory_data/raw/omnicontact',
    cache_dir='/home/arenalabs/umr_factory_data/hf_cache',
    allow_patterns=pats, max_workers=8)
print('DONE', pats, f'{time.time()-t:.0f}s', flush=True)
