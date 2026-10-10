"""Verify the DiD code bank against the case-study answer key."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))  # repo root
import pandas as pd
from toolkit.did import did_2x2, did_regression, placebo_did

STUDY = Path(__file__).resolve().parent.parent  # this case study's folder
df = pd.read_csv(STUDY / "data" / "did_synthetic.csv")
key = json.load(open(STUDY / "output" / "did_results.json"))

r1 = did_2x2(df, "treated_state", "post", "employment")
assert abs(r1["estimate"] - key["did_2x2"]) < 1e-2, r1  # JSON stores 2-decimal rounding
print(f"did_2x2: {r1['estimate']:.4f} == case study {key['did_2x2']}  PASS")

r2 = did_regression(df, "treated_state", "post", "employment")
assert abs(r2["estimate"] - key["did_reg"]) < 1e-2, r2
print(f"did_regression: {r2['estimate']:.4f} == case study {key['did_reg']}  PASS")

pre = df[df.year <= 2019].copy()
pre["fake_post"] = pre.year >= 2019
r3 = placebo_did(pre, "treated_state", "fake_post", "employment")
assert abs(r3["placebo_estimate"] - key["placebo_effect"]) < 1e-2 and r3["p"] > 0.05
print(f"placebo_did: {r3['placebo_estimate']} (p={r3['p']:.3f}) finds nothing  PASS")
print("Code bank verified: it reproduces the case study exactly.")
