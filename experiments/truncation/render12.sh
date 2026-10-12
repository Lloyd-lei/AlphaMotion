#!/bin/bash
# renders the 12 inspection clips sequentially (single process at a time)
cd /home/arenalabs/umr_factory_data/experiments/truncation
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
for m in Cause_to_move_in_place-rotate_0202 Cause_motion-push_0212 Self_motion-step_0001 Change_posture-sit_down_0024 \
         Change_posture-sit_down_0021 Cause_motion-chuck_0032 Placing-shoulder_0010 Bringing-hump_0144 \
         Change_posture-lean_0005 Cause_motion-punt_0002 Posture-sit_0005 Bringing-lug_0291; do
  timeout 600 taskset -c 0-15 nice -n 10 ~/miniconda3/bin/python render_grid.py $m 20 || echo "FAIL $m"
done
echo RENDER_DONE
