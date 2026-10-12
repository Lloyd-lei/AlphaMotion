"""BONES-SEED 下载（lloyd 1011 同意）：只下 SOMA Proportional（按演员真实身材的 BVH，45.5 GB）+ 每演员体型参数 + 元数据。
不下 soma_uniform（同动作的统一身材版）和 g1.tar.gz（官方 G1 重定向）。
落盘 raw/bones_seed/；HF 缓存放 /var/tmp/hf_home_bones，绝不写 ~/.cache/huggingface。断点续传：重跑即可。"""
from surface_retargeting.config import CFG as SRCFG  # machine paths: configs/sr.toml
import os, sys, time
from huggingface_hub import snapshot_download
t = time.time()
for attempt in range(1, 10000):  # WiFi 会掉线：失败就等 60 秒重试，断点续传
    try:
        p = snapshot_download(
            repo_id="bones-studio/seed", repo_type="dataset",
            local_dir=f'{SRCFG.data_root}/raw/bones_seed',
            allow_patterns=["soma_proportional.tar.gz", "soma_shapes/**", "metadata/**", "README.md", "LICENSE.md"],
            max_workers=2,
        )
        break
    except Exception as e:
        print(f"attempt {attempt} failed: {e!r}"[:300], flush=True); time.sleep(60)
print(f"DONE {p} in {(time.time()-t)/3600:.2f} h", flush=True)
