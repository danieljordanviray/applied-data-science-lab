# Phase 1 data dictionary (NHANES 2017-2018)

Rows: 5437 adults with a valid hypertension outcome.

Outcome `hypertension` = 1 if mean SBP>=130 or mean DBP>=80 or on BP meds (BPQ050A).
Weights: `weight_mec` = WTMEC2YR (exam); `weight_fast` = WTSAF2YR fasting subsample
(falls back to MEC weight if absent).

| column | label | unit |
|---|---|---|
| LBXTC | Total cholesterol | mg/dL |
| LBDHDD | HDL cholesterol | mg/dL |
| LBXTR | Triglycerides | mg/dL |
| LBDLDL | LDL cholesterol | mg/dL |
| LBXGH | HbA1c | % |
| LBXGLU | Fasting glucose | mg/dL |
| LBXHSCRP | C-reactive protein | mg/L |
| BMXBMI | BMI | kg/m^2 |
| BMXWAIST | Waist circumference | cm |
