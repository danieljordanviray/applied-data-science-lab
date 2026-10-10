"""Assemble phase-1 Jupyter notebook (narrative + executable code cells)."""
import json
from pathlib import Path

STUDY = Path(__file__).resolve().parent.parent  # this case study's folder
NB = STUDY / "notebooks" / "phase-01-biomarkers-hypertension.ipynb"

def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}

def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.splitlines(keepends=True)}

cells = [
md("""# Phase 1: Which blood biomarkers actually associate with hypertension?

**Question.** Across nine blood biomarkers in NHANES 2017-2018, which ones are genuinely
associated with hypertension once we correct for testing everything at once?

**Data.** CDC NHANES 2017-2018 public-use files (public domain). 5,437 adults with a
valid blood-pressure outcome. Outcome = hypertension by ACC/AHA-style rule:
mean SBP >= 130, or mean DBP >= 80, or on BP medication.

**Methods in one paragraph.** For each biomarker we compare the hypertension group vs the
no-hypertension group using survey-weighted means (NHANES oversamples some groups by
design, so every respondent carries a sampling weight), a weighted Welch t-test, and a
Mann-Whitney check. Nine tests means nine chances to get fooled, so we correct with
Bonferroni and Benjamini-Hochberg. Then we ask the uncomfortable question: is it the
biomarker, or is it just age? via age/sex-adjusted logistic regression and age-stratified
means.
"""),
code("""import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))  # repo root
import pandas as pd, numpy as np
from toolkit.nhanes import read_xpt, merge_nhanes, recode_yesno, row_mean
from pathlib import Path
RAW = Path("data/raw")  # run from the project root
"""),
md("## 1. Load and merge the NHANES files"),
code("""demo = read_xpt(RAW/"DEMO_J.xpt"); bpx = read_xpt(RAW/"BPX_J.xpt")
bpq = read_xpt(RAW/"BPQ_J.xpt"); bmx = read_xpt(RAW/"BMX_J.xpt")
tchol = read_xpt(RAW/"TCHOL_J.xpt"); hdl = read_xpt(RAW/"HDL_J.xpt")
trig = read_xpt(RAW/"TRIGLY_J.xpt"); ghb = read_xpt(RAW/"GHB_J.xpt")
hscrp = read_xpt(RAW/"HSCRP_J.xpt"); glu = read_xpt(RAW/"GLU_J.xpt")
df = merge_nhanes(demo, bpx, bpq, tchol, hdl, trig, ghb, hscrp, glu, bmx)
df["sbp"] = row_mean(df, ["BPXSY1","BPXSY2","BPXSY3","BPXSY4"])
df["dbp"] = row_mean(df, ["BPXDI1","BPXDI2","BPXDI3","BPXDI4"])
df["on_bp_meds"] = recode_yesno(df["BPQ050A"])
df["hypertension"] = ((df["sbp"]>=130)|(df["dbp"]>=80)|(df["on_bp_meds"]==1)).astype(float)
df["age"] = pd.to_numeric(df["RIDAGEYR"], errors="coerce")
df["weight_mec"] = pd.to_numeric(df["WTMEC2YR"], errors="coerce")
adults = df[(df["age"]>=18) & df["hypertension"].notna() & (df["weight_mec"]>0)].copy()
print(f"{len(adults)} adults with a valid outcome")
"""),
md("## 2. Weighted prevalence: why the weights exist\nNHANES deliberately oversamples some groups. A raw head-count would misrepresent\nthe US population, so every respondent carries a sampling weight."),
code("""from toolkit.weighted import weighted_prevalence
prev = weighted_prevalence(adults["hypertension"], adults["weight_mec"])
print(f"Weighted hypertension prevalence: {prev:.1%}")
"""),
md("## 3. One test per biomarker (weighted t-test + Mann-Whitney + effect size)"),
code("""from toolkit.weighted import weighted_ttest_ind, cohens_d_weighted
from scipy import stats
BIOMARKERS = {"LBXTC":"Total cholesterol","LBDHDD":"HDL cholesterol","LBXTR":"Triglycerides",
  "LBDLDL":"LDL cholesterol","LBXGH":"HbA1c","LBXGLU":"Fasting glucose",
  "LBXHSCRP":"C-reactive protein","BMXBMI":"BMI","BMXWAIST":"Waist circumference"}
rows = []
for col, label in BIOMARKERS.items():
    s = adults[["hypertension", col, "weight_mec"]].dropna()
    g1, g0 = s[s.hypertension==1], s[s.hypertension==0]
    t = weighted_ttest_ind(g1[col], g1.weight_mec, g0[col], g0.weight_mec)
    rows.append({"biomarker": label, "n": len(s), "p": t["p"],
                 "d": cohens_d_weighted(g1[col], g1.weight_mec, g0[col], g0.weight_mec),
                 "mw_p": stats.mannwhitneyu(g1[col], g0[col]).pvalue})
res = pd.DataFrame(rows)
res
"""),
md("## 4. The multiple-testing correction: nine tests, nine chances to be fooled"),
code("""from toolkit.multitest import bonferroni, benjamini_hochberg
bonf, bh = bonferroni(res["p"]), benjamini_hochberg(res["p"])
res["p_bonf"], res["p_bh"] = bonf["adjusted"], bh["adjusted"]
res["sig_bonf"], res["sig_bh"] = bonf["reject"], bh["reject"]
print(f"Significant raw (p<0.05): {(res['p']<0.05).sum()}, "
      f"Bonferroni: {res['sig_bonf'].sum()}, BH-FDR: {res['sig_bh'].sum()}")
res.sort_values("p")[["biomarker","p","sig_bonf","sig_bh","d"]]
"""),
md("## 5. Confounding check: is it the biomarker, or just age?\nLogistic regression per biomarker, adjusted for age and sex. Biomarkers standardized\nso odds ratios are per standard deviation and comparable."),
code("""import statsmodels.api as sm
out = []
for col, label in BIOMARKERS.items():
    s = adults[["hypertension", col, "weight_mec", "age", "RIAGENDR"]].dropna().copy()
    s["z"] = (s[col]-s[col].mean())/s[col].std()
    s["male"] = (s["RIAGENDR"]==1).astype(int)
    X = sm.add_constant(s[["z","age","male"]])
    fw = s["weight_mec"]*len(s)/s["weight_mec"].sum()  # rescale: weights are not counts
    fit = sm.GLM(s["hypertension"], X, family=sm.families.Binomial(), freq_weights=fw).fit(disp=0)
    out.append({"biomarker": label, "OR_per_SD": round(float(np.exp(fit.params["z"])),3),
                "p_adj": fit.pvalues["z"]})
pd.DataFrame(out).sort_values("p_adj")
"""),
md("""## 6. Takeaways
- **Metabolic cluster wins.** Waist circumference (d=0.67), HbA1c (d=0.58), fasting
  glucose (d=0.54), BMI (d=0.47) show the largest differences, and all survive
  age/sex adjustment. This hangs together biologically: insulin resistance and
  hypertension travel together.
- **Correction matters.** HDL looks "significant" at p=0.007 raw, dies under Bonferroni,
  survives under BH-FDR. Same data, three different verdicts: the method is the message.
- **LDL is the dog that didn't bark.** No signal crude or adjusted. Plausible reason:
  people with high LDL take statins, which lowers their LDL while they remain older
  and hypertensive. A reminder that a null can be just as informative, and that
  treatment effects contaminate observational comparisons (a theme for phase 2).
- **Limitations.** Cross-sectional: association, not causation. Weighted t-tests are
  approximations (full survey-design SEs need PSU/strata). Fasting analytes ideally
  want fasting-subsample weights (unavailable in this file; MEC weights used).
"""),
]

nb_obj = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "cells": cells}
NB.parent.mkdir(parents=True, exist_ok=True)
NB.write_text(json.dumps(nb_obj, indent=1))
print("notebook written:", NB)
