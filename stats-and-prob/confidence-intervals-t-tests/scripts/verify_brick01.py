"""Brick 1 prove-it: verify every claim before it touches the lesson.

1. Hand-check: CI on 5 numbers, arithmetic shown, must match mean_ci and scipy.
2. Planted answer: fake groups with a KNOWN difference of 5 -> method must find ~5.
   Fake groups with NO difference -> must not cry wolf.
3. Cross-check: two_sample_ttest must equal scipy.stats.ttest_ind on real data.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))  # repo root
import numpy as np
import pandas as pd
from scipy import stats
from toolkit.stats_basics import mean_ci, two_sample_ttest

print("=== 1. Hand-check: CI on [22.1, 24.3, 21.8, 25.0, 23.2] ===")
x = np.array([22.1, 24.3, 21.8, 25.0, 23.2])
n = 5
mean = x.sum() / n                                   # 23.28
s = np.sqrt(((x - mean) ** 2).sum() / (n - 1))        # sample sd
se = s / np.sqrt(n)
t_star = 2.776  # t_{0.975, df=4}, look it up in any t-table (3-decimal rounding)
hand = (mean - t_star * se, mean + t_star * se)
got = mean_ci(x)
sp = stats.t.interval(0.95, n - 1, loc=mean, scale=se)
print(f"hand:      ({hand[0]:.4f}, {hand[1]:.4f})")
print(f"mean_ci:   ({got['ci_low']:.4f}, {got['ci_high']:.4f})")
print(f"scipy:     ({sp[0]:.4f}, {sp[1]:.4f})")
# tolerance 5e-4: the hand calc uses a 3-decimal t-table value, the code uses full precision
assert abs(hand[0] - got["ci_low"]) < 5e-4 and abs(hand[1] - got["ci_high"]) < 5e-4
assert abs(sp[0] - got["ci_low"]) < 1e-9
print("PASS: hand arithmetic == mean_ci == scipy\n")

print("=== 2. Planted answer: true difference is exactly 5 ===")
rng = np.random.default_rng(7)
A = rng.normal(0, 1, 500)
B = rng.normal(5, 1, 500)   # we PLANTED this: B is 5 higher
r = two_sample_ttest(B, A)
print(f"recovered diff: {r['diff']:.3f} (truth: 5.0), p = {r['p']:.2e}, reject H0: {r['reject_h0']}")
assert abs(r["diff"] - 5) < 0.5 and r["p"] < 1e-10 and r["reject_h0"]
C = rng.normal(0, 1, 500)
D = rng.normal(0, 1, 500)   # no difference planted
r0 = two_sample_ttest(C, D)
print(f"no-difference case: diff = {r0['diff']:.3f}, p = {r0['p']:.3f}, reject H0: {r0['reject_h0']}")
assert not r0["reject_h0"]
print("PASS: finds planted differences, stays quiet when there's nothing\n")

print("=== 3. Cross-check on REAL data: BMI by sex ===")
df = pd.read_csv(str(Path(__file__).resolve().parent.parent / "data" / "brick01_clean.csv"))
m = df.loc[df.sex == "male", "bmi"]
f = df.loc[df.sex == "female", "bmi"]
mine = two_sample_ttest(m, f)
ref = stats.ttest_ind(m, f, equal_var=False)
print(f"mine:  t={mine['t']:.6f} p={mine['p']:.6f}")
print(f"scipy: t={ref.statistic:.6f} p={ref.pvalue:.6f}")
assert abs(mine["t"] - ref.statistic) < 1e-9 and abs(mine["p"] - ref.pvalue) < 1e-12
print("PASS: identical to scipy on real data\n")

print("=== Real-data results for the lesson ===")
ci = mean_ci(df["bmi"])
print(f"Mean BMI 95% CI: ({ci['ci_low']:.2f}, {ci['ci_high']:.2f}), n={ci['n']}")
print(f"BMI men vs women: diff={mine['diff']:.3f}, 95% CI ({mine['ci_low']:.3f}, {mine['ci_high']:.3f}), "
      f"d={mine['cohens_d']:.3f}, p={mine['p']:.4f}, reject H0: {mine['reject_h0']}")
sbp = two_sample_ttest(df.loc[df.sex == "male", "sbp"], df.loc[df.sex == "female", "sbp"])
print(f"SBP men vs women: diff={sbp['diff']:.3f}, 95% CI ({sbp['ci_low']:.3f}, {sbp['ci_high']:.3f}), "
      f"d={sbp['cohens_d']:.3f}, p={sbp['p']:.2e}")
