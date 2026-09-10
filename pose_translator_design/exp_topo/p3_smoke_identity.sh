#!/bin/bash
# Greenwich-v2 P3 冒烟: 证明「接触层的每个开关都关掉时, 训练与打补丁之前逐位相同」。
#
#   bash p3_smoke_identity.sh            # CPU, 几十秒, 不占 GPU
#
# 做法: 把打补丁**之前**的 4 个文件 (train_dual/dual_common/model_dual/masking) 放进
# 一个影子目录, 其余模块符号链接过去, 用完全相同的参数各跑一次, 比 model.pt 的 sha256。
# 为什么比 model.pt 而不是比 loss 打印: 权重是所有随机流 + 所有梯度累加顺序的联合指纹,
# 少消耗一次 RNG 或换一次加法结合序都会立刻体现出来。
set -e
cd "$(dirname "$0")"
EXP=$PWD
PY=${PY:-$HOME/miniconda3/envs/rot/bin/python}
# 补丁前的 4 个文件存在这里 (不能用 git 重建: 打补丁时 dual_common.py 本来就有未提交改动,
# HEAD 版本不等于"今天的行为")。
BASE=${BASE:-/var/tmp/alphamotion/topo_exp/v2_prepatch_0831}
[ -d "$BASE" ] || { echo "缺基线目录 $BASE"; exit 1; }
W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
mkdir -p "$W/base"
for f in "$EXP"/*; do ln -sf "$f" "$W/base/$(basename "$f")"; done
for f in train_dual.py dual_common.py model_dual.py masking.py; do
  rm -f "$W/base/$f"; cp "$BASE/$f.bak" "$W/base/$f" 2>/dev/null || cp "$BASE/$f" "$W/base/$f"
done

ARGS=(--cache /var/tmp/alphamotion/topo_exp/cache_xwbc_lafan
      --device cpu --max_rows 3000 --steps 6 --bs 16 --eval_every 6
      --n_latent 8 --fsq_stages 1 --fsq_levels 8,8,8,8
      --d_model 32 --n_heads 4 --enc_layers 1 --dec_layers 1
      --mask_prob 0.3 --mask_mix rand:0.3,ee:0.25,root:0.15,sparse:0.3
      --w_pos 5.0 --w_contrast 0.5 --combo_sample 3 --contrast_sample 3
      --use_dof --judge_mode grad --slip_pairs --slip_w 0.2 --seed 0)

echo "== 补丁前 (base)";  $PY "$W/base/train_dual.py" "${ARGS[@]}" --out "$W/o_base" >"$W/base.log" 2>&1 || { tail -25 "$W/base.log"; exit 1; }
echo "== 补丁后 (new, 接触开关全关)"; $PY "$EXP/train_dual.py" "${ARGS[@]}" --out "$W/o_new" >"$W/new.log" 2>&1 || { tail -25 "$W/new.log"; exit 1; }

A=$(sha256sum "$W/o_base/model.pt" | cut -d' ' -f1)
B=$(sha256sum "$W/o_new/model.pt"  | cut -d' ' -f1)
echo "base model.pt sha256 $A"
echo "new  model.pt sha256 $B"
diff <(grep EVAL "$W/base.log") <(grep EVAL "$W/new.log") >/dev/null && EV=same || EV=DIFF
echo "EVAL 行: $EV"
if [ "$A" = "$B" ] && [ "$EV" = same ]; then echo "PASS: 开关全关 = 逐位一致"; else echo "FAIL: 出现行为漂移"; exit 1; fi

# 反向对照: 打开接触开关后必须**不同** (否则说明开关根本没接上线)
echo "== 反向对照: --contact_chan --w_pen 100 --w_cbce 0.1"
$PY "$EXP/train_dual.py" "${ARGS[@]}" --contact_chan --w_pen 100 --w_contact 10 \
    --w_cbce 0.1 --w_slip_c 0.2 --out "$W/o_on" >"$W/on.log" 2>&1 || { tail -25 "$W/on.log"; exit 1; }
C=$(sha256sum "$W/o_on/model.pt" | cut -d' ' -f1)
echo "on   model.pt sha256 $C"
[ "$A" != "$C" ] && echo "PASS: 开关打开后确实改变了训练" || { echo "FAIL: 开关是摆设"; exit 1; }
grep -o '"c_pen":[^,]*\|"c_cmaint":[^,]*\|"c_cbce":[^,]*\|"slipc_dec_ms":[^,]*' "$W/on.log" | tail -4
