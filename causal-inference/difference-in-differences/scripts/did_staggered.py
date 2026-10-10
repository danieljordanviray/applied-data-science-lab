"""Staggered DiD case study: treatment arrives at different times.

Story: a retail chain rolls out a new staff-scheduling system store by store.
  - 15 stores adopt Mar 2022 (true effect +8), 15 May 2022 (+6),
    15 Jul 2022 (+4), 15 Sep 2022 (+2). True overall ATT = 5.0 (planted).
  - 40 stores never adopt (controls).
  - Monthly sales ($k), Oct 2021 - Dec 2022.

SYNTHETIC with planted heterogeneous effects. Verifiable: we know the truth.

Estimators compared:
  1. Cohort DiDs vs never-treated, averaged [Callaway-Sant'Anna intuition].
     The bank's did_2x2 is reused per cohort -- the flywheel in action.
  2. Naive pooled TWFE (one big regression) -- for comparison.
  3. Naive TWFE event study (one big regression with lead/lag dummies).
  4. Cohort event study: per-cohort DiD at each event month vs never-treated,
     averaged. The honest event-study.

Finding: (3) is contaminated even though the design is full-rank
[Sun & Abraham 2021]; (1)/(4) recover the planted truth.
New staggered machinery lives HERE for review, not in the bank.
"""
import sys, json
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))  # repo root
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from toolkit.did import did_2x2

STUDY = Path(__file__).resolve().parent.parent  # this case study's folder
OUT = STUDY / "output"; FIG = OUT / "figures"

rng = np.random.default_rng(2026)
MONTHS = pd.period_range("2021-10", "2022-12", freq="M")
M = {str(m): i for i, m in enumerate(MONTHS)}
COHORTS = {"2022-03": 8.0, "2022-05": 6.0, "2022-07": 4.0, "2022-09": 2.0}
TRUE_ATT = float(np.mean(list(COHORTS.values())))

rows = []
sid = 0
for adopt, effect in COHORTS.items():
    ai = M[adopt]
    for s in range(15):
        base = 100 + rng.normal(0, 8)
        for mi, m in enumerate(MONTHS):
            y = base + 0.8 * mi + rng.normal(0, 2.5) + (effect if mi >= ai else 0)
            rows.append({"store": sid, "month": str(m), "mi": mi, "cohort": adopt,
                         "adopt_mi": ai, "ever_treated": True,
                         "treated_now": mi >= ai, "sales": y})
        sid += 1
for s in range(40):
    base = 100 + rng.normal(0, 8)
    for mi, m in enumerate(MONTHS):
        y = base + 0.8 * mi + rng.normal(0, 2.5)
        rows.append({"store": sid, "month": str(m), "mi": mi, "cohort": "never",
                     "adopt_mi": np.nan, "ever_treated": False,
                     "treated_now": False, "sales": y})
    sid += 1
df = pd.DataFrame(rows)
df.to_csv(STUDY / "data" / "did_staggered.csv", index=False)

# ---- 1. Per-cohort DiD vs never-treated (bank's did_2x2, reused) ----
def se_did(sub, tcol, pcol, y):
    g = sub.groupby([tcol, pcol])[y]
    v = g.var(ddof=1); n = g.count()
    return float(np.sqrt((v / n).sum()))

cohort_res = []
for adopt, effect in COHORTS.items():
    ai = M[adopt]
    win = df[((df["cohort"] == adopt) | (df["cohort"] == "never")) &
             (df["mi"] >= ai - 3) & (df["mi"] <= ai + 2)].copy()
    win["post"] = win["mi"] >= ai
    r = did_2x2(win, "ever_treated", "post", "sales")
    cohort_res.append({"cohort": adopt, "n": 15, "true_effect": effect,
                       "did": round(r["estimate"], 2),
                       "se": round(se_did(win, "ever_treated", "post", "sales"), 2)})
att = float(np.mean([c["did"] for c in cohort_res]))
att_se = float(np.sqrt(np.mean([c["se"] ** 2 for c in cohort_res])) / 2)

# ---- 2. Naive pooled TWFE ----
d = pd.get_dummies(df["store"], prefix="s", drop_first=True).astype(float)
t = pd.get_dummies(df["mi"], prefix="m", drop_first=True).astype(float)
X = pd.concat([d, t, df[["treated_now"]].astype(float)], axis=1)
twfe = float(sm.OLS(df["sales"], sm.add_constant(X)).fit().params["treated_now"])

# ---- 3. Naive TWFE event study (the cautionary tale) ----
df["event"] = df["mi"] - df["adopt_mi"]
ev = pd.DataFrame(index=df.index)
for k in [-3, -2, 0, 1, 2, 3]:
    ev[f"E{k}"] = (df["event"] == k).astype(float)
fe = sm.OLS(df["sales"], sm.add_constant(pd.concat([d, t, ev], axis=1))).fit()
twfe_event = {k: (float(fe.params[f"E{k}"]), float(fe.bse[f"E{k}"]))
              for k in [-3, -2, 0, 1, 2, 3]}

# ---- 4. Honest cohort event study: per-cohort DiD at each event month ----
event_res = {}
for k in [-3, -2, 0, 1, 2, 3]:
    ests, ses = [], []
    for adopt in COHORTS:
        ai = M[adopt]
        win = df[((df["cohort"] == adopt) | (df["cohort"] == "never")) &
                 (df["mi"].isin([ai - 1, ai + k]))].copy()
        win["post"] = win["mi"] == ai + k
        r = did_2x2(win, "ever_treated", "post", "sales")
        ests.append(r["estimate"]); ses.append(se_did(win, "ever_treated", "post", "sales"))
    event_res[k] = {"est": float(np.mean(ests)),
                    "se": float(np.sqrt(np.mean(np.square(ses))) / 2)}
event_res[-1] = {"est": 0.0, "se": 0.0}

def event_plot(data, path, title):
    # data: {k: (est, se)}
    ks = sorted(data)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.errorbar(ks, [data[k][0] for k in ks],
                 yerr=[1.96 * data[k][1] for k in ks],
                 fmt="o", capsize=4, color="#1f6feb")
    ax.axhline(0, color="black", linewidth=1)
    ax.axvline(-0.5, color="gray", linestyle="--", label="Adoption month")
    ax.set_xlabel("Months since adoption (event time)")
    ax.set_ylabel("Estimated effect on sales ($k)")
    ax.set_title(title); ax.legend(); ax.grid(alpha=0.25)
    fig.savefig(path); plt.close(fig)

event_plot({k: (twfe_event[k][0], twfe_event[k][1]) for k in twfe_event},
           FIG / "did_event_twfe.png", "Naive TWFE event study: phantom pre-trends, muted effects")
event_plot({k: (event_res[k]["est"], event_res[k]["se"]) for k in event_res},
           FIG / "did_event_cohort.png",
           "Cohort event study: flat before, clean lift after")

# ---- 5. Placebo: 30 repetitions of fake 4-cohort adoption among never-treated ----
# Mirrors the real estimator exactly (4 cohorts x 5 stores -> aggregate).
never_ids_all = df.loc[df["cohort"] == "never", "store"].unique()
placebo_atts = []
for rep in range(30):
    r = np.random.default_rng(10_000 + rep)
    fids = r.choice(never_ids_all, 20, replace=False).reshape(4, 5)
    ctrl_ids = np.setdiff1d(never_ids_all, fids.ravel())
    att_r = []
    for ci, adopt in enumerate(COHORTS):
        ai = M[adopt]
        fstores = fids[ci]
        win = df[((df["store"].isin(fstores)) | (df["store"].isin(ctrl_ids))) &
                 (df["mi"] >= ai - 3) & (df["mi"] <= ai + 2)].copy()
        win["post"] = win["mi"] >= ai
        win["fake_t"] = win["store"].isin(fstores)
        att_r.append(did_2x2(win, "fake_t", "post", "sales")["estimate"])
    placebo_atts.append(float(np.mean(att_r)))
placebo_est = float(np.mean(placebo_atts))
placebo_se = float(np.std(placebo_atts, ddof=1) / np.sqrt(len(placebo_atts)))

results = {"true_att": TRUE_ATT, "cohort_dids": cohort_res,
           "att_aggregated": round(att, 2), "att_se": round(att_se, 2),
           "twfe_pooled": round(twfe, 2),
           "twfe_event_study": {str(k): round(v[0], 2) for k, v in twfe_event.items()},
           "cohort_event_study": {str(k): round(event_res[k]["est"], 2) for k in sorted(event_res)},
           "placebo_effect": round(placebo_est, 2), "placebo_se": round(placebo_se, 2)}
(OUT / "did_staggered_results.json").write_text(json.dumps(results, indent=1))
print(json.dumps(results, indent=1))
