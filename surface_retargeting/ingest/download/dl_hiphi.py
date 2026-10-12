from surface_retargeting.config import CFG as SRCFG  # machine paths: configs/sr.toml
import os, sys, time
os.environ.pop('HF_HUB_OFFLINE', None)
from huggingface_hub import hf_hub_download
t = time.time()
for part in sys.argv[1:]:
    fn = f'data/HiPHI_data_part_{part}_of_0032.tar.zst'
    p = hf_hub_download('noitomrobotics/HiPHI', fn, repo_type='dataset',
        local_dir=f'{SRCFG.data_root}/raw/hiphi',
        cache_dir=f'{SRCFG.data_root}/hf_cache')
    print('DONE', p, f'{time.time()-t:.0f}s', flush=True)
