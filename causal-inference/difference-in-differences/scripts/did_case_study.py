"""DiD case study: did State A's job-training program work?

Story: State A launches a job-training program in 2020. State B does not.
We observe county employment rates, 2018-2022.

SYNTHETIC DATA with a PLANTED true effect of +5 points. That is the point:
we know the answer key, so every check is verifiable.

NOTE: this code lives in the case-study folder for review. It is NOT in the
code bank (toolkit/) -- the bank stays empty until Daniel designs it.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

STUDY = Path(__file__).resolve().parent.parent  # this case study's folder
OUT = STUDY / "output"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

TRUE_EFFECT = 5.0
rng = np.random.default_rng(123)

rows = []
for state, base in [("A", 60.0), ("B", 64.0)]:  # different baselines on purpose
    for county in range(40):
        county_base = base + rng.normal(0, 3)
        for year in range(2018, 2023):
            trend = 1.0 * (year - 2018)            # both states rise 1pt/yr (parallel)
            treated = (state == "A") and (year >= 2020)
            y = county_base + trend + rng.normal(0, 1.5) + (TRUE_EFFECT if treated else 0)
            rows.append({"state": state, "county": f"{state}{county:02d}",
                         "year": year, "treated_state": state == "A",
                         "post": year >= 2020, "employment": y})
df = pd.DataFrame(rows)
df.to_csv(STUDY / "data" / "did_synthetic.csv", index=False)

# ---- 1. The 2x2 table of means ----
t = df.groupby(["treated_state", "post"])["employment"].mean().unstack()
a_pre, a_post = t.loc[True, False], t.loc[True, True]
b_pre, b_post = t.loc[False, False], t.loc[False, True]
did_2x2 = (a_post - a_pre) - (b_post - b_pre)

# ---- 2. The regression version: Y ~ Treat + Post + Treat:Post ----
fit = smf.ols("employment ~ treated_state + post + treated_state:post", data=df).fit()
coef = fit.params["treated_state[T.True]:post[T.True]"]
se = fit.bse["treated_state[T.True]:post[T.True]"]
ci = fit.conf_int().loc["treated_state[T.True]:post[T.True]"].tolist()

# ---- 3. Check: pre-trends plot (the eyeball test) ----
yearly = df.groupby(["state", "year"])["employment"].mean().unstack("state")
fig, ax = plt.subplots(figsize=(7.5, 4.2))
for s, color in [("A", "#d1242f"), ("B", "#1f6feb")]:
    ax.plot(yearly.index, yearly[s], marker="o", label=f"State {s}", color=color)
ax.axvline(2019.5, color="gray", linestyle="--", label="Program starts (2020)")
ax.set_xlabel("Year"); ax.set_ylabel("Mean employment rate (%)")
ax.set_title("Employment trends: parallel before 2020, diverge after")
ax.legend(); ax.grid(alpha=0.25)
fig.savefig(FIG / "did_trends.png")
plt.close(fig)

# ---- 4. Check: placebo test (pretend treatment started in 2019, use 2018-2019 only) ----
pre = df[df["year"] <= 2019].copy()
pre["fake_post"] = pre["year"] >= 2019
pfit = smf.ols("employment ~ treated_state + fake_post + treated_state:fake_post", data=pre).fit()
placebo = pfit.params["treated_state[T.True]:fake_post[T.True]"]
placebo_p = pfit.pvalues["treated_state[T.True]:fake_post[T.True]"]

results = {
    "true_effect": TRUE_EFFECT,
    "means": {"A_pre": round(a_pre, 2), "A_post": round(a_post, 2),
              "B_pre": round(b_pre, 2), "B_post": round(b_post, 2)},
    "did_2x2": round(did_2x2, 2),
    "did_reg": round(coef, 2), "did_se": round(se, 2),
    "did_ci": [round(ci[0], 2), round(ci[1], 2)],
    "did_p": float(fit.pvalues["treated_state[T.True]:post[T.True]"]),
    "placebo_effect": round(placebo, 2), "placebo_p": round(float(placebo_p), 3),
}
(OUT / "did_results.json").write_text(json.dumps(results, indent=1))
print(json.dumps(results, indent=1))
