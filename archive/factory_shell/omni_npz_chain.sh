#!/bin/bash
cd /home/arenalabs/umr_factory_data
# convert continuously while raw downloads, then npz labels, then a final sweep
until ! ps -p $(cat logs/omni_batch1.pid) >/dev/null; do sleep 20; done
while ps -p $(cat logs/dl_omni_raw.pid) >/dev/null; do timeout 4h scripts/run_omni_batch.sh; sleep 60; done
echo "raw done $(date)"
timeout 4h scripts/run_omni_batch.sh &
B=$!
timeout 2h /home/arenalabs/miniconda3/bin/python -u scripts/dl_omni.py 'npz/*' > logs/dl_omni_npz.log 2>&1
du -sh raw/omnicontact/npz
timeout 1h /home/arenalabs/miniconda3/envs/genmo-gem/bin/python scripts/omni_labels_extract.py && du -sh raw/omnicontact/g1_labels && rm -rf raw/omnicontact/npz
wait $B
timeout 4h scripts/run_omni_batch.sh
echo OMNI_CHAIN_DONE $(date)
