"""Phase 1 analysis: which biomarkers associate with hypertension?

Steps:
  1. Weighted prevalence of hypertension (survey weights matter).
  2. Per-biomarker two-group comparison: weighted t-test + Mann-Whitney + Cohen's d.
  3. Multiple-testing correction: Bonferroni + Benjamini-Hochberg.
  4. Confounding check: age/sex-adjusted logistic regression + age-stratified means.
"""
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))  # repo root
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm

from toolkit.weighted import weighted_ttest_ind, cohens_d_weighted, weighted_prevalence, weighted_mean
from toolkit.multitest import bonferroni, benjamini_hochberg
from toolkit.plots import set_house_style, forest_plot

STUDY = Path(__file__).resolve().parent.parent  # this case study's folder
df = pd.read_pickle(STUDY / "data" / "analysis.pkl")
FIG = STUDY / "output" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
set_house_style()

BIOMARKERS = {
    "LBXTC": "Total cholesterol", "LBDHDD": "HDL cholesterol",
    "LBXTR": "Triglycerides", "LBDLDL": "LDL cholesterol",
    "LBXGH": "HbA1c", "LBXGLU": "Fasting glucose",
    "LBXHSCRP": "C-reactive protein", "BMXBMI": "BMI",
    "BMXWAIST": "Waist circumference",
}
WCOL = {"LBXTR": "weight_fast", "LBDLDL": "weight_fast", "LBXGLU": "weight_fast"}

prev = weighted_prevalence(df["hypertension"], df["weight_mec"])
print(f"Weighted hypertension prevalence (adults): {prev:.1%}  (n={len(df)})")

rows = []
for col, label in BIOMARKERS.items():
    wcol = WCOL.get(col, "weight_mec")
    sub = df[["hypertension", col, wcol, "age", "sex"]].dropna()
    g1 = sub[sub["hypertension"] == 1]
    g0 = sub[sub["hypertension"] == 0]
    t = weighted_ttest_ind(g1[col], g1[wcol], g0[col], g0[wcol])
    mw = stats.mannwhitneyu(g1[col], g0[col], alternative="two-sided")
    d = cohens_d_weighted(g1[col], g1[wcol], g0[col], g0[wcol])
    rows.append({"biomarker": col, "label": label, "n": len(sub),
                 "mean_hyp": t["mean_a"], "mean_no": t["mean_b"],
                 "diff": t["diff"], "p_raw": t["p"], "p_mw": mw.pvalue,
                 "cohens_d": d})
res = pd.DataFrame(rows)
bonf = bonferroni(res["p_raw"])
bh = benjamini_hochberg(res["p_raw"])
res["p_bonf"] = bonf["adjusted"]
res["p_bh"] = bh["adjusted"]
res["sig_bonf"] = bonf["reject"]
res["sig_bh"] = bh["reject"]

# Age/sex-adjusted logistic regression per biomarker (standardized)
adj = []
for col, label in BIOMARKERS.items():
    sub = df[["hypertension", col, "weight_mec", "age", "sex"]].dropna().copy()
    sub["z"] = (sub[col] - sub[col].mean()) / sub[col].std()
    sub["male"] = (sub["sex"] == "male").astype(int)
    X = sm.add_constant(sub[["z", "age", "male"]])
    # freq_weights are treated as case counts: rescale so they sum to n,
    # otherwise the model thinks n = 200 million and SEs collapse.
    fw = sub["weight_mec"] * len(sub) / sub["weight_mec"].sum()
    try:
        fit = sm.GLM(sub["hypertension"], X, family=sm.families.Binomial(),
                     freq_weights=fw).fit(disp=0)
        adj.append({"biomarker": col, "or_per_sd": float(np.exp(fit.params["z"])),
                    "p_adj": float(fit.pvalues["z"])})
    except Exception as e:
        adj.append({"biomarker": col, "or_per_sd": np.nan, "p_adj": np.nan})
res = res.merge(pd.DataFrame(adj), on="biomarker")
res.to_csv(STUDY / "output" / "results.csv", index=False)
print(res[["label", "n", "p_raw", "sig_bonf", "sig_bh", "cohens_d", "p_adj"]].to_string())

# ---- Figure 1: p-value plot with correction thresholds ----
fig, ax = plt.subplots(figsize=(8, 4.5))
order = np.argsort(res["p_raw"])
xs = np.arange(len(res))
ax.scatter(xs, -np.log10(res["p_raw"].values[order]), s=60, color="#1f6feb", zorder=3)
ax.axhline(-np.log10(bonf["threshold"]), color="#d1242f", linestyle="--",
           label=f"Bonferroni (p={bonf['threshold']:.2e})")
ax.axhline(-np.log10(0.05), color="gray", linestyle=":", label="Naive p=0.05")
ax.set_xticks(xs)
ax.set_xticklabels(res["label"].values[order], rotation=35, ha="right")
ax.set_ylabel("-log10(p-value)")
ax.set_title("Nine biomarkers tested: raw p-values vs. correction thresholds")
ax.legend()
fig.savefig(FIG / "fig1_pvalues.png")
plt.close(fig)

# ---- Figure 2: forest plot of effect sizes ----
d = res.sort_values("cohens_d")
se_d = 1.96 * np.sqrt(1 / d["n"] + d["cohens_d"]**2 / (2 * d["n"]))
forest_plot(list(d["label"]), list(d["cohens_d"]),
            list(d["cohens_d"] - se_d), list(d["cohens_d"] + se_d),
            FIG / "fig2_forest.png", xlabel="Cohen's d (hypertension vs not)")

# ---- Figure 3: age confounding demo (CRP or strongest crude signal) ----
demo_col = res.sort_values("p_raw").iloc[0]["biomarker"]
demo_label = BIOMARKERS[demo_col]
bands = [(18, 39, "18-39"), (40, 59, "40-59"), (60, 120, "60+")]
fig, ax = plt.subplots(figsize=(7, 4.2))
x = np.arange(len(bands)); wdt = 0.35
mh, mn = [], []
for lo, hi, name in bands:
    s = df[(df["age"] >= lo) & (df["age"] < hi)]
    s = s[["hypertension", demo_col, "weight_mec"]].dropna()
    mh.append(weighted_mean(s.loc[s["hypertension"] == 1, demo_col],
                            s.loc[s["hypertension"] == 1, "weight_mec"]))
    mn.append(weighted_mean(s.loc[s["hypertension"] == 0, demo_col],
                            s.loc[s["hypertension"] == 0, "weight_mec"]))
ax.bar(x - wdt/2, mh, wdt, label="Hypertension", color="#d1242f")
ax.bar(x + wdt/2, mn, wdt, label="No hypertension", color="#1f6feb")
ax.set_xticks(x); ax.set_xticklabels([b[2] for b in bands])
ax.set_ylabel(f"Weighted mean {demo_label}")
ax.set_title(f"{demo_label} by hypertension status within age bands")
ax.legend()
fig.savefig(FIG / "fig3_age_confounding.png")
plt.close(fig)

# ---- Figure 4: SBP distribution with cutoff ----
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(df["sbp"].dropna(), bins=60, color="#1f6feb", alpha=0.8)
ax.axvline(130, color="#d1242f", linestyle="--", label="130 mmHg cutoff")
ax.set_xlabel("Mean systolic BP (mmHg)")
ax.set_ylabel("Adults")
ax.set_title("Distribution of systolic blood pressure, NHANES 2017-2018")
ax.legend()
fig.savefig(FIG / "fig4_sbp_dist.png")
plt.close(fig)
print("figures saved:", demo_col, "used for fig3")
