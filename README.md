# Applied Data Science Lab

Public case studies in applied statistics, experimentation, causal inference, and
machine learning — built slowly, on purpose. Each one runs on data with a known
answer key (real data or synthetic data with a planted truth), shows the full code,
and includes checks a human can verify. No trust-me statistics.

## Read the case studies

Rendered as pages via GitHub Pages (see `docs/` — the filterable index):

- **Experimentation → [ab-testing-playbook](experimentation/ab-testing-playbook/)**
  — The Practical A/B Testing Playbook: designing, sizing, and reading A/B
  tests — power and sample size, the pitfalls that fake a win, ratio metrics
  and the Delta Method, with an interactive power diagram
- **Statistics & Probability → [p-values-power-playground](stats-and-prob/p-values-power-playground/)**
  — Statistical Significance, Visualized: an interactive p-value & power playground
- **Machine Learning → [regression-guide](machine-learning/regression-guide/)**
  — The Practitioner's Guide to Regression: a one-page method bank (OLS fitting
  and prediction, transformations, fit metrics, diagnostic plots, influence,
  logistic regression, confusion matrix) with verified toy healthcare examples
- **Statistics & Probability → [sampling-and-bootstrapping](stats-and-prob/sampling-and-bootstrapping/)**
  — Sampling & Bootstrapping: population, samples, the sampling
  distribution, standard error, confidence intervals, and bootstrapping in one
  connected story (a town's blood pressure) with interactive labs
- **Statistics & Probability → [practitioners-guide](stats-and-prob/practitioners-guide/)**
  — The Practitioner's Guide to Statistical Tests: a one-page method bank (z, t,
  ANOVA, proportions, chi-squared, rank tests) with verified toy healthcare examples
- **Causal Inference → [difference-in-differences](causal-inference/difference-in-differences/)**
  — Difference-in-Differences, plus the staggered-timing version
- **SQL (Advanced) → [sql](sql/)** — EXISTS vs IN/NOT IN, COALESCE/NULLIF, window functions,
  date spines, CTEs: quick-reference notes with clinical toy examples

## Layout

- `stats-and-prob/` — confidence intervals, t-tests, real-data reference studies
- `experimentation/` — A/B testing and experiment-design case studies
- `causal-inference/` — difference-in-differences case studies
- `sql/` — quick-reference SQL notes (EXISTS, COALESCE/NULLIF, windows, date spines, CTEs)
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
