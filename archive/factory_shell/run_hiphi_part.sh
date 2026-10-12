#!/bin/bash
# usage: run_hiphi_part.sh 0029   -> extract originals of that part, convert all, 14 workers, 40 min cap per clip
set -u
P=$1
cd /home/arenalabs/umr_factory_data
PY=/home/arenalabs/miniconda3/envs/genmo-gem/bin/python
AR=raw/hiphi/data/HiPHI_data_part_${P}_of_0032.tar.zst
EX=raw/hiphi/extracted/part_${P}
mkdir -p $EX
if [ ! -f $EX/.extracted ]; then
  t0=$(date +%s)
  timeout 2h tar --use-compress-program=zstd -xf $AR -C $EX --exclude='*__mirror*' && touch $EX/.extracted
  echo "extract part $P: $(( $(date +%s)-t0 )) s, $(du -sh $EX | cut -f1)"
fi
find $EX -name metadata.json -printf '%h\n' | sort > logs/hiphi_${P}_todo.txt
echo "todo $(wc -l < logs/hiphi_${P}_todo.txt)"
cat logs/hiphi_${P}_todo.txt | xargs -P 14 -I{} timeout 2400 $PY scripts/convert_hiphi.py {} >> logs/hiphi_convert.jsonl 2>> logs/hiphi_convert.err
echo "part $P done $(date)"
