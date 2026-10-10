# Applied Data Science Lab

Public case studies in applied statistics, experimentation, causal inference, and
machine learning — built slowly, on purpose. Each one runs on data with a known
answer key (real data or synthetic data with a planted truth), shows the full code,
and includes checks a human can verify. No trust-me statistics.

## Read the case studies

Rendered as pages via GitHub Pages (see `docs/`):

- **Causal inference → [difference-in-differences](causal-inference/difference-in-differences/)**
  — the classic 2×2 and the staggered-timing version
- **Stats & prob → [confidence-intervals-t-tests](stats-and-prob/confidence-intervals-t-tests/)**
  — CIs and the two-tailed t-test, verified three ways
- **Stats & prob → [biomarkers-hypertension](stats-and-prob/biomarkers-hypertension/)**
  — advanced reference on real NHANES data

## Layout

- `stats-and-prob/` — confidence intervals, t-tests, real-data reference studies
- `experimentation/` — A/B testing lives here next
- `causal-inference/` — difference-in-differences case studies
- `machine-learning/` — saved for later in the track
- `toolkit/` — the code bank: reusable, verified recipes. Currently `did.py`
  (`did_2x2`, `did_regression`, `placebo_did`, `plot_group_trends`). Nothing enters
  the bank until it has survived a case study's checks.

## Reproduce everything

```bash
python3 -m venv .venv
.venv/bin/pip install pandas numpy scipy matplotlib statsmodels
# DiD case studies (from the repo root):
.venv/bin/python causal-inference/difference-in-differences/scripts/did_case_study.py
.venv/bin/python causal-inference/difference-in-differences/scripts/did_staggered.py
.venv/bin/python causal-inference/difference-in-differences/scripts/verify_did_bank.py
```

Built by an AI employee, reviewed by a human boss.
