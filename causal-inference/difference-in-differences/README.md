# Difference-in-Differences

Two case studies in causal inference, built slowly and verified against answer keys.

## 1. The classic 2×2 — [did-case-study.html](did-case-study.html)

Did the program work? State A launches a job-training program in 2020, State B
doesn't. The 2×2 table of means, the regression version, the parallel-trends
assumption, placebo testing. Intuitive first, formal names in brackets, with a
30-second version for non-technical bosses.

## 2. Staggered timing — [did-staggered-case-study.html](did-staggered-case-study.html)

Treatment arrives in four waves (Mar/May/Jul/Sep 2022). Event-time relabeling, one
honest DiD per cohort (reusing the code bank's `did_2x2`) then averaged — and a
demonstration of why the naive one-big-regression invents phantom pre-trends
([Goodman-Bacon], [Sun & Abraham] contamination, shown side by side).

## Reproduce

```bash
# from the repo root, with the venv + deps installed (see root README)
python3 causal-inference/difference-in-differences/scripts/did_case_study.py
python3 causal-inference/difference-in-differences/scripts/did_staggered.py
python3 causal-inference/difference-in-differences/scripts/verify_did_bank.py
```

All data is synthetic with planted truths (`data/`), so every claim is checkable.
The code-bank module this study earned: `toolkit/did.py`.
