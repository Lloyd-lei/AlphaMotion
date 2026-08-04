"""E14 morning report: the cross-embodiment make-or-break readout.

For each E14 run (judge-grad / judge-observe):
  1. LAFAN G1 zero-shot (from its own eval_log holdout entries)
  2. BONES external exam: soma77->g1_30 translation — BOTH embodiments unseen
     (double zero-shot; SOMA J=78 with fingers never in training)
  3. Judge matrix: per DATA embodiment (18 xwbc + 2 BONES), mean sigmoid of every
     judge, using the TRAIN-TIME name->index mapping (read from the xwbc manifest).
     The G1/talos judges trained negatives-only: darkness expected by construction;
     the informative rows are which OTHER judges light on G1 data.

Usage: python e14_report.py --runs runs/E14_grad_0804 runs/E14_obs_0804
Writes {run}/e14_exam.json + prints a compact comparison.
"""
from __future__ import annotations
import os, sys, json, argparse, itertools
import torch
sys.path.insert(0, os.path.dirname(__file__))
from dual_common import load_paired_data, split_indices, build_model, evaluate, geom

XWBC_CACHE = "/var/tmp/alphamotion/topo_exp/cache_xwbc_lafan"
BONES_CACHE = "/var/tmp/alphamotion/topo_exp/cache_paired_fixed"


def judge_matrix(model, data, names, train_names, idx):
    out = {}
    with torch.no_grad():
        for n in names:
            zp, zr = model.encode(data[n]["pose"][idx], geom(data, n, len(idx)))
            act = torch.sigmoid(model.judge(zp, zr)).mean(0)
            row = {train_names[i]: round(float(act[i]), 3) for i in range(len(train_names))}
            top = max(row, key=row.get)
            out[n] = {"top": [top, row[top]], "row": row}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    args = ap.parse_args()
    device = "cuda"

    xman = json.load(open(f"{XWBC_CACHE}/manifest.json"))
    train_names_full = xman["embodiments"]                 # judge index order == load order

    for run in args.runs:
        cfg = json.load(open(f"{run}/config.json"))
        report = {"run": run, "judge_mode": cfg.get("judge_mode")}

        # 1. LAFAN G1 zero-shot from training eval log (last entry)
        log = json.load(open(f"{run}/eval_log.json"))[-1]
        report["lafan_g1_zeroshot"] = {k: v for k, v in log.items()
                                       if isinstance(v, dict) and "g1" in k}
        # 2+3. per cache: translations + judge matrix
        for tag, cache in [("xwbc", XWBC_CACHE), ("bones", BONES_CACHE)]:
            torch.manual_seed(0)
            data, names, N = load_paired_data(cache, device)
            val_idx, _ = split_indices(N)
            model = build_model(cfg, device)
            missing, unexpected = model.load_state_dict(
                torch.load(f"{run}/model.pt", map_location=device), strict=False)
            model.eval()
            if tag == "bones":
                combos = list(itertools.product(names, names))
                ev = evaluate(model, data, names, combos, val_idx)
                report["bones_exam"] = {k: v for k, v in ev.items() if "->" in k}
            if model.judge is not None:
                report[f"judge_matrix_{tag}"] = judge_matrix(model, data, names,
                                                             train_names_full, val_idx[:1024])
            del data, model; torch.cuda.empty_cache()
        json.dump(report, open(f"{run}/e14_exam.json", "w"), indent=2)
        print(f"\n===== {run} ({report['judge_mode']}) =====")
        print("LAFAN G1 零样本:", json.dumps(report["lafan_g1_zeroshot"], ensure_ascii=False))
        print("BONES 双端零样本:", json.dumps(report.get("bones_exam", {}), ensure_ascii=False))
        jm = report.get("judge_matrix_xwbc", {})
        for g in ("unitree_g1_29dof", "unitree_g1_23dof"):
            if g in jm:
                print(f"G1数据({g}) 最亮裁判:", jm[g]["top"])
        jb = report.get("judge_matrix_bones", {})
        for n, v in jb.items():
            print(f"BONES {n} 最亮裁判:", v["top"])


if __name__ == "__main__":
    main()
