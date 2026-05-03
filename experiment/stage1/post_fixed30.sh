#!/usr/bin/env bash
# Post-processing for axis-fixed 30-epoch runs (mixed_fixed30).
# Run after training completes. Produces:
#   runs/<model>__mixed_fixed30/diag/        (9 plots + summary.json)
#   runs/<model>__mixed_fixed30/amass_heldout.json
#
# The key figures to compare against mixed_full30:
#   diag/08_style_tsne_source.png   — should show amass/kimodo no longer
#                                     forming two giant separated blobs
#   amass_heldout.json              — pos should match or beat
#                                     mixed_full30's full=0.096 / opm=0.107

set -euo pipefail
cd "$(dirname "$0")"

if pgrep -f "train.py.*run-tag mixed_fixed30" > /dev/null; then
    echo "WARNING: training still running. Exiting. Rerun when done."
    exit 1
fi

for m in opm_only full; do
    ck="runs/${m}__mixed_fixed30/final.pt"
    if [ ! -f "$ck" ]; then
        echo "[skip] $ck missing"
        continue
    fi

    echo "==== [$m] diagnose_stage15 ===="
    python diagnose_stage15.py --checkpoint "$ck" --model "$m"

    echo "==== [$m] eval_on_amass_heldout ===="
    python eval_on_amass_heldout.py --checkpoint "$ck" --model "$m" \
        --subsets cmu bmlrub kit hdm05 \
        --out-json "runs/${m}__mixed_fixed30/amass_heldout.json"
done

echo ""
echo "[done] compare against mixed_full30:"
for m in opm_only full; do
    old="runs/${m}__mixed_full30/amass_heldout.json"
    new="runs/${m}__mixed_fixed30/amass_heldout.json"
    [ -f "$old" ] && [ -f "$new" ] || continue
    echo "  $m:"
    python - <<PY
import json
a = json.load(open("$old")); b = json.load(open("$new"))
print(f'    mixed_full30 (axis-bug):  pos={a["pos"]:.4f}  rot_deg={a["rot_degrees"]:.3f}')
print(f'    mixed_fixed30 (axis-ok):  pos={b["pos"]:.4f}  rot_deg={b["rot_degrees"]:.3f}')
print(f'    delta:                    pos={b["pos"]-a["pos"]:+.4f}  rot_deg={b["rot_degrees"]-a["rot_degrees"]:+.3f}')
PY
done
echo ""
echo "Visual check:"
echo "  diff runs/full__mixed_full30/diag/08_style_tsne_source.png"
echo "       runs/full__mixed_fixed30/diag/08_style_tsne_source.png"
