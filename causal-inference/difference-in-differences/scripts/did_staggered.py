"""Staggered DiD case study, gentle version.

Story: 100 patients, weight-loss drug. Nobody designed an experiment — some
patients started the drug in January, some in February, some in March, and
40 never took it. We observe everyone's weight monthly. Did the drug work?

SYNTHETIC with planted effects. Verifiable: we know the truth.
  - Jan starters (20 patients): true effect -4.0 kg
  - Feb starters (20 patients): true effect -3.0 kg
  - Mar starters (20 patients): true effect -2.0 kg
  - Never-treated (40 patients): controls
  - True overall ATT = -3.0 kg (planted)

Estimators:
  1. One honest DiD per start-month cohort vs never-treated [cohort-specific
     DiD], then averaged [group-time aggregation / Callaway-Sant'Anna intuition].
     Reuses the bank's did_2x2 per cohort.
  2. Naive pooled TWFE (one big regression) -- for comparison.
  3. Cohort event study: per-cohort DiD at each relative month vs never-treated,
     averaged. Pre-launch estimates should hover near zero.
  4. Placebo: fake-treated among the never-treated -> should find nothing.
"""
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

STUDY = Path(__file__).resolve().parent.parent  # this case study's folder
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))  # repo root
from toolkit.did import did_2x2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)

COHORTS = {"jan": (4, -4.0, 20), "feb": (5, -3.0, 20), "mar": (6, -2.0, 20)}
NEVER_N = 40
MONTHS = np.arange(1, 10)  # Oct=1 ... Jun=9 ; Jan=4, Feb=5, Mar=6
OUT = STUDY / "output"
FIG = OUT / "figures"
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

rows = []
pid = 0
for name, (start, eff, n) in COHORTS.items():
    for _ in range(n):
        pid += 1
        fe = rng.normal(0, 3)  # patient baseline differences
        for t in MONTHS:
            w = 88 + fe + 0.2 * t + (eff if t >= start else 0.0) + rng.normal(0, 0.7)
            rows.append((pid, name, start, t, w))
never_ids = []
for _ in range(NEVER_N):
    pid += 1
    never_ids.append(pid)
    fe = rng.normal(0, 3)
    for t in MONTHS:
        w = 88 + fe + 0.2 * t + rng.normal(0, 0.7)
        rows.append((pid, "never", 0, t, w))
df = pd.DataFrame(rows, columns=["patient", "cohort", "start", "month", "weight"])
df.to_csv(STUDY / "data" / "did_staggered.csv", index=False)

# 1. cohort DiDs vs never-treated -------------------------------------------
cohort_est = {}
for name, (start, eff, n) in COHORTS.items():
    d = df[((df.cohort == name) | (df.cohort == "never"))].copy()
    d["treated"] = (d.cohort == name)
    d["post"] = (d.month >= start)
    r = did_2x2(d, "treated", "post", "weight")
    cohort_est[name] = {"estimate": r["estimate"], "truth": eff, "n": n}
    print(f"cohort {name}: est {r['estimate']:.2f} vs truth {eff}")

att = float(np.mean([cohort_est[k]["estimate"] for k in COHORTS]))
print(f"ATT (avg of cohorts): {att:.2f} vs truth -3.00")

# 2. naive pooled TWFE --------------------------------------------------------
import statsmodels.formula.api as smf
d = df.copy()
d["treated_now"] = ((d.cohort != "never") & (d.month >= d.start)).astype(int)
fit = smf.ols("weight ~ treated_now + C(patient) + C(month)", data=d).fit()
twfe = float(fit.params["treated_now"])
print(f"naive TWFE: {twfe:.2f} vs truth -3.00")

# 3. cohort event study -------------------------------------------------------
ev = {-1: 0.0}  # reference month: zero by construction
for k in [k for k in range(-3, 4) if k != -1]:
    ests = []
    for name, (start, eff, n) in COHORTS.items():
        t1, t0 = start + k, start - 1
        if t1 < 1 or t1 > 9:
            continue
        d = df[((df.cohort == name) | (df.cohort == "never"))
               & (df.month.isin([t0, t1]))].copy()
        d["treated"] = (d.cohort == name)
        d["post"] = (d.month == t1)
        ests.append(did_2x2(d, "treated", "post", "weight")["estimate"])
    ev[k] = float(np.mean(ests))
print("event study:", {k: round(v, 2) for k, v in ev.items()})

fig, ax = plt.subplots(figsize=(8, 4.5))
ks = sorted(ev)
ax.plot(ks, [ev[k] for k in ks], marker="o", color="#1f6feb")
ax.axhline(0, color="gray", linestyle="--")
ax.axvline(-0.5, color="gray", linestyle=":")
ax.set(xlabel="Months since starting the drug (event time)",
       ylabel="Estimated weight change (kg)",
       title="Cohort event study: lined up by start month, vs never-treated")
fig.tight_layout()
fig.savefig(FIG / "did_event_cohort.png", dpi=110)
plt.close(fig)

# 4. placebo: fake-treated among the never-treated ------------------------------
placebos = []
for rep in range(30):
    fake = rng.choice(never_ids, 20, replace=False)
    d = df[df.cohort == "never"].copy()
    d["treated"] = d.patient.isin(fake)
    d["post"] = (d.month >= 4)
    placebos.append(did_2x2(d, "treated", "post", "weight")["estimate"])
placebos = np.array(placebos)
print(f"placebo: mean {placebos.mean():.2f}, sd {placebos.std():.2f} (truth 0)")

results = {
    "cohort_est": {k: {"estimate": round(v["estimate"], 2), "truth": v["truth"], "n": v["n"]}
                   for k, v in cohort_est.items()},
    "att": round(att, 2), "att_truth": -3.0,
    "twfe": round(twfe, 2),
    "event_study": {str(k): round(v, 2) for k, v in ev.items()},
    "placebo_mean": round(float(placebos.mean()), 2),
    "placebo_sd": round(float(placebos.std()), 2),
}
with open(OUT / "did_staggered_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("saved results + figure")
