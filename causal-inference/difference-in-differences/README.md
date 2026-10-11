# Difference-in-Differences

Two case studies in causal inference, built slowly and verified against answer keys.

## 1. The classic 2×2 — [did-case-study.html](did-case-study.html)

Did the program work? State A launches a job-training program in 2020, State B
doesn't. The 2×2 table of means, the regression version, the parallel-trends
assumption, placebo testing. Intuitive first, formal names in brackets, with a
30-second version for non-technical bosses.

## 2. Staggered timing — [did-staggered-case-study.html](did-staggered-case-study.html)

100 patients, one weight-loss drug, three different start months. Line everyone up
on "months since starting" [event time], run one honest DiD per start-month group
against the never-treated, then average. Includes a hand-worked 6-patient example
and a demo of how using already-treated patients as controls can flip the sign.

## Reproduce

```bash
# from the repo root, with the venv + deps installed (see root README)
python3 causal-inference/difference-in-differences/scripts/did_case_study.py
python3 causal-inference/difference-in-differences/scripts/did_staggered.py
python3 causal-inference/difference-in-differences/scripts/verify_did_bank.py
```

All data is synthetic with planted truths (`data/`), so every claim is checkable.
The code-bank module this study earned: `toolkit/did.py`.
