"""Clip truncation rules (from experiments/truncation, finalized 2026-10-07; see its PROGRESS.md).

final_rules_v1.json / final_rules_v2.json are the finalized tables (v1 ruler: mean joint distance < 5 cm;
v2 ruler: max joint distance < 12 cm).  Rule: clips not marked "repeatedly" are kept whole; "repeatedly" clips keep
an adaptive head (>= 15 s, until 90 % of the remaining 1 s windows already appeared in the head) plus the last 8 s.
The analysis code (rules.py, headtail.py, cov_all.py, common.py ...) stays in experiments/truncation/.
"""
