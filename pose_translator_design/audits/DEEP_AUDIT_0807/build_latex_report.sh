#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

xelatex -interaction=nonstopmode -halt-on-error AlphaMotion_Mechanism_Audit_CN.tex
xelatex -interaction=nonstopmode -halt-on-error AlphaMotion_Mechanism_Audit_CN.tex

rm -f AlphaMotion_Mechanism_Audit_CN.aux \
      AlphaMotion_Mechanism_Audit_CN.log \
      AlphaMotion_Mechanism_Audit_CN.out

echo "$HERE/AlphaMotion_Mechanism_Audit_CN.pdf"
