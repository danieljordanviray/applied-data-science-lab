"""Phase 1: build the analysis dataset from NHANES 2017-2018 public files.

Outcome: hypertension (ACC/AHA-style: mean SBP>=130 or DBP>=80, or on BP meds).
Population: adults (age>=18) with a valid BP outcome and exam weight.
Biomarkers: lipids, HbA1c, glucose, CRP, BMI, waist.
"""
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))  # repo root
from pathlib import Path
import pandas as pd
import numpy as np
from toolkit.nhanes import read_xpt, merge_nhanes, recode_yesno, row_mean

STUDY = Path(__file__).resolve().parent.parent  # this study's folder
RAW = STUDY / "data" / "raw"
OUT = STUDY / "data"

demo = read_xpt(RAW / "DEMO_J.xpt")
bpx = read_xpt(RAW / "BPX_J.xpt")
bpq = read_xpt(RAW / "BPQ_J.xpt")
tchol = read_xpt(RAW / "TCHOL_J.xpt")
hdl = read_xpt(RAW / "HDL_J.xpt")
trig = read_xpt(RAW / "TRIGLY_J.xpt")
ghb = read_xpt(RAW / "GHB_J.xpt")
hscrp = read_xpt(RAW / "HSCRP_J.xpt")
glu = read_xpt(RAW / "GLU_J.xpt")
bmx = read_xpt(RAW / "BMX_J.xpt")

print("WTSAF2YR present:", "WTSAF2YR" in demo.columns)

df = merge_nhanes(demo, bpx, bpq, tchol, hdl, trig, ghb, hscrp, glu, bmx)

df["sbp"] = row_mean(df, ["BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4"])
df["dbp"] = row_mean(df, ["BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4"])
df["on_bp_meds"] = recode_yesno(df["BPQ050A"])
df["hypertension"] = ((df["sbp"] >= 130) | (df["dbp"] >= 80) | (df["on_bp_meds"] == 1)).astype(float)
df.loc[df["sbp"].isna() & df["dbp"].isna() & df["on_bp_meds"].isna(), "hypertension"] = np.nan

df["age"] = pd.to_numeric(df["RIDAGEYR"], errors="coerce")
df["sex"] = df["RIAGENDR"].map({1: "male", 2: "female"})
df["weight_mec"] = pd.to_numeric(df["WTMEC2YR"], errors="coerce")
wfast = pd.to_numeric(df["WTSAF2YR"], errors="coerce") if "WTSAF2YR" in df.columns else df["weight_mec"]
df["weight_fast"] = wfast

# biomarker registry: (column, label, weight column, unit)
BIOMARKERS = {
    "LBXTC":    ("Total cholesterol", "weight_mec", "mg/dL"),
    "LBDHDD":   ("HDL cholesterol", "weight_mec", "mg/dL"),
    "LBXTR":    ("Triglycerides", "weight_fast", "mg/dL"),
    "LBDLDL":   ("LDL cholesterol", "weight_fast", "mg/dL"),
    "LBXGH":    ("HbA1c", "weight_mec", "%"),
    "LBXGLU":   ("Fasting glucose", "weight_fast", "mg/dL"),
    "LBXHSCRP": ("C-reactive protein", "weight_mec", "mg/L"),
    "BMXBMI":   ("BMI", "weight_mec", "kg/m^2"),
    "BMXWAIST": ("Waist circumference", "weight_mec", "cm"),
}
for col in BIOMARKERS:
    df[col] = pd.to_numeric(df[col], errors="coerce")

adults = df[(df["age"] >= 18) & df["hypertension"].notna() & (df["weight_mec"] > 0)].copy()
print(f"adults with outcome: {len(adults)} of {len(df)}")

adults.to_pickle(OUT / "analysis.pkl")

with open(OUT / "dictionary.md", "w") as f:
    f.write("# Phase 1 data dictionary (NHANES 2017-2018)\n\n")
    f.write(f"Rows: {len(adults)} adults with a valid hypertension outcome.\n\n")
    f.write("Outcome `hypertension` = 1 if mean SBP>=130 or mean DBP>=80 or on BP meds (BPQ050A).\n")
    f.write("Weights: `weight_mec` = WTMEC2YR (exam); `weight_fast` = WTSAF2YR fasting subsample\n"
            "(falls back to MEC weight if absent).\n\n")
    f.write("| column | label | unit |\n|---|---|---|\n")
    for col, (label, w, unit) in BIOMARKERS.items():
        f.write(f"| {col} | {label} | {unit} |\n")
print("saved analysis.pkl")
