#!/bin/bash
cd /home/arenalabs/umr_factory_data
for P in 0029 0002; do
  F=raw/hiphi/data/HiPHI_data_part_${P}_of_0032.tar.zst
  until [ -f $F ] && [ ! -f $F.aria2 ]; do sleep 30; done
  sz=$(stat -c %s $F); echo "part $P downloaded $sz bytes $(date)"
  scripts/run_hiphi_part.sh $P
done
echo CHAIN_DONE
