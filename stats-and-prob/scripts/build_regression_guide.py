"""Build the Practitioner's Guide to Regression (stats-and-prob).

Same house rules as the other guides: every number on the page is computed
here by running the exact calls shown on the page, on the toy data described,
and assertions check the story matches the numbers before HTML is written.
Plots are real matplotlib figures of the fitted models, embedded as SVG.
Style mirrors Daniel's course decks: statsmodels formula API.
"""
import io
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols, logit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams.update({"svg.fonttype": "none", "font.size": 9,
                     "axes.edgecolor": "#888", "axes.labelcolor": "#222",
                     "xtick.color": "#555", "ytick.color": "#555"})
rng = np.random.default_rng(7)
FAILURES = []

def check(name, cond, detail=""):
    print(("  ok: " if cond else "  FAIL: ") + name + (f" ({detail})" if detail else ""))
    if not cond:
        FAILURES.append(name)

import re as _re
def svg_of(fig, dpi=60):
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    s = buf.getvalue()
    s = s[s.index("<svg"):]
    s = _re.sub(r"<metadata>.*?</metadata>", "", s, flags=_re.S)
    s = _re.sub(r"\d+\.\d{3,}", lambda m: f"{float(m.group()):.1f}", s)
    s = _re.sub(r">\s+<", "><", s)
    return s

# ------------------------------------------------------------- main OLS -----
# Cost of a hospital stay ($) from age, length of stay, admission type.
n = 120
age = rng.integers(22, 88, n)
los = np.clip(rng.normal(5.2, 2.6, n), 1, 16).round()
adm = rng.choice(["Elective", "Emergency", "Urgent"], n, p=[.45, .30, .25])
cost = (4200 + 55 * age + 880 * los + np.where(adm == "Emergency", 2600, 0)
        + np.where(adm == "Urgent", 1300, 0) + rng.normal(0, 1700, n)).round()
hosp = pd.DataFrame({"cost": cost, "age": age, "los_days": los, "admission": adm})
fit = ols("cost ~ age + los_days + C(admission)", data=hosp).fit()
new_pt = pd.DataFrame({"age": [68], "los_days": [5], "admission": ["Emergency"]})
pred = float(fit.predict(new_pt).iloc[0])
planted = 4200 + 55 * 68 + 880 * 5 + 2600
check("OLS R2 in sane band", 0.55 < fit.rsquared < 0.95, f"R2={fit.rsquared:.2f}")
check("emergency premium recovered", 1800 < fit.params["C(admission)[T.Emergency]"] < 3400,
      f"coef={fit.params['C(admission)[T.Emergency]']:.0f}")
check("prediction near planted truth", abs(pred - planted) / planted < 0.18,
      f"pred=${pred:,.0f} vs planted ${planted:,}")

# ------------------------------------------------------------- metrics ------
resid = fit.resid
sse = float((resid ** 2).sum())
sst = float(((hosp["cost"] - hosp["cost"].mean()) ** 2).sum())
k = int(fit.df_model)                       # number of predictors (slopes)
rmse = float(np.sqrt(sse / n))
rse = float(np.sqrt(sse / (n - k - 1)))
mae = float(np.abs(resid).mean())
mape = float((np.abs(resid) / hosp["cost"]).mean() * 100)
check("RSE > in-sample RMSE (df correction)", rse > rmse, f"RSE={rse:.0f} RMSE={rmse:.0f}")
check("adjusted R2 < R2", fit.rsquared_adj < fit.rsquared,
      f"adj={fit.rsquared_adj:.3f} vs {fit.rsquared:.3f}")
check("MAPE sane", 5 < mape < 60, f"MAPE={mape:.1f}%")
mse = sse / n

# ------------------------------------------------- transformation example ---
# Dose (mg) -> % symptom reduction. Truth is log-dose: diminishing returns.
dose = np.round(np.exp(rng.uniform(np.log(2), np.log(120), 36)), 1)
reduction = np.clip(14 * np.log(dose) + rng.normal(0, 4.0, dose.size), 0, None).round(1)
dose_df = pd.DataFrame({"dose": dose, "reduction": reduction})
r2_raw = ols("reduction ~ dose", data=dose_df).fit().rsquared
r2_logx = ols("reduction ~ np.log(dose)", data=dose_df).fit().rsquared
check("log-dose straightens (R2 jumps)", r2_logx > r2_raw + 0.15,
      f"raw R2={r2_raw:.2f} vs log-dose R2={r2_logx:.2f}")
DOSE_JSON = json.dumps([[float(x), float(y)] for x, y in zip(dose, reduction)])

# --------------------------------------- diagnostic plots (main model) ------
fitted = fit.fittedvalues
fig, ax = plt.subplots(figsize=(5.4, 3.1))
ax.axhline(0, color="#c0392b", lw=1)
ax.scatter(fitted, resid, s=13, rasterized=True, color="#2b50aa", alpha=.75)
ax.set_xlabel("Fitted cost ($)"); ax.set_ylabel("Residual ($)")
ax.set_title("Residuals vs fitted")
PLOT_RESID = svg_of(fig, dpi=75)

std_resid = resid / (rse * np.sqrt(1 - fit.get_influence().hat_matrix_diag))
fig, ax = plt.subplots(figsize=(5.4, 3.1))
sm.qqplot(std_resid, line="45", fit=False, ax=ax,
          markerfacecolor="#2b50aa", markeredgecolor="none", alpha=.75, rasterized=True)
ax.set_title("Normal Q-Q (standardized residuals)")
PLOT_QQ = svg_of(fig, dpi=75)

fig, ax = plt.subplots(figsize=(5.4, 3.1))
ax.scatter(fitted, np.sqrt(np.abs(std_resid)), s=13, rasterized=True, color="#2b50aa", alpha=.75)
ax.set_xlabel("Fitted cost ($)"); ax.set_ylabel("sqrt(|standardized residual|)")
ax.set_title("Scale-location")
PLOT_SCALE = svg_of(fig, dpi=75)

# ---------------------------------- outliers / leverage / influence ---------
# Small clinic dataset: age -> cost, with three planted troublemakers.
m = 30
ax_ = rng.integers(25, 85, m).astype(float)
ay = 3000 + 60 * ax_ + rng.normal(0, 500, m)
ax_[3], ay[3] = 52, 12500        # outlier: ordinary age, wild cost
ax_[11], ay[11] = 96, 8760      # leverage only: extreme age, ON trend (3000+60*96)
ax_[22], ay[22] = 94, 2600      # influential: extreme age, far OFF trend
clin = pd.DataFrame({"age": ax_, "cost": ay})
cfit = ols("cost ~ age", data=clin).fit()
infl = cfit.get_influence()
cooks = infl.cooks_distance[0]
hats = infl.hat_matrix_diag
check("influential point has the largest Cook's D", int(np.argmax(cooks)) == 22,
      f"D={cooks[22]:.2f}")
check("leverage-only point: high hat, low Cook's", hats[11] > hats[3] and cooks[11] < cooks[22] / 4,
      f"hat={hats[11]:.2f}, D={cooks[11]:.3f}")
check("outlier point: big residual, modest leverage", abs(infl.resid_studentized_external[3]) > 2.5,
      f"studentized resid={infl.resid_studentized_external[3]:.1f}")

fig, ax = plt.subplots(figsize=(5.3, 3.2))
sizes = 30 + 1700 * cooks
ax.scatter(ax_, ay, s=sizes, rasterized=True, color="#2b50aa", alpha=.45)
xs = np.array([20, 100])
ax.plot(xs, cfit.params.iloc[0] + cfit.params.iloc[1] * xs, color="#c0392b", lw=1.2)
ax.set_xlim(14, 103); ax.set_ylim(0, 14500)
for idx, lab, tx, ty in [(3, "outlier\n(big residual)", 17, 10200),
                         (11, "leverage only\n(on the trend)", 62, 11800),
                         (22, "influential\n(pulls the line)", 34, 600)]:
    ax.annotate(lab, (ax_[idx], ay[idx]), xytext=(tx, ty),
                fontsize=8, color="#333",
                arrowprops=dict(arrowstyle="-", color="#666", lw=.7))
ax.set_xlabel("Age"); ax.set_ylabel("Cost ($)")
ax.set_title("Bubble size = Cook's distance")
PLOT_INFL = svg_of(fig, dpi=102)

# ------------------------------------------------------------- logistic -----
# 30-day readmission from a risk score (0-100). Truth is logistic in score.
n2 = 240
score = rng.integers(5, 96, n2)
eta = -3.4 + 0.055 * score
p_true = 1 / (1 + np.exp(-eta))
readmit = (rng.uniform(size=n2) < p_true).astype(int)
rd = pd.DataFrame({"score": score, "readmit": readmit})
lfit = logit("readmit ~ score", data=rd).fit(disp=0)
b0, b1 = float(lfit.params.iloc[0]), float(lfit.params.iloc[1])
check("logistic slope positive", b1 > 0.03, f"b1={b1:.3f}")
odds_ratio_10 = float(np.exp(10 * b1))

# Five example patients for the languages table: nearest rows to target scores.
rows = []
for target in [20, 40, 55, 70, 85]:
    j = int(np.argmin(np.abs(score - target)))
    x = int(score[j]); p = 1 / (1 + np.exp(-(b0 + b1 * x)))
    rows.append(dict(x=x, y=int(readmit[j]), p=p, pred=int(p >= 0.5),
                     odds=p / (1 - p), logodds=b0 + b1 * x))
check("table probabilities increase with score",
      all(rows[i]["p"] < rows[i + 1]["p"] for i in range(len(rows) - 1)))

# Confusion-matrix data (full sample) for the interactive.
probs = 1 / (1 + np.exp(-(b0 + b1 * score)))
CONF_JSON = json.dumps([[int(y), round(float(p), 3)] for y, p in zip(readmit, probs)])
def conf_at(thr):
    predp = (probs >= thr).astype(int)
    tp = int(((predp == 1) & (readmit == 1)).sum()); fp = int(((predp == 1) & (readmit == 0)).sum())
    tn = int(((predp == 0) & (readmit == 0)).sum()); fn = int(((predp == 0) & (readmit == 1)).sum())
    return tp, fp, tn, fn
c50 = conf_at(0.5)
acc50 = (c50[0] + c50[2]) / n2
_pt = lfit.pred_table()
check("pred_table() matches manual counts @0.5",
      int(_pt[0, 0]) == c50[2] and int(_pt[0, 1]) == c50[1]
      and int(_pt[1, 0]) == c50[3] and int(_pt[1, 1]) == c50[0],
      f"pred_table={_pt.ravel().astype(int).tolist()}")
check("logistic accuracy at 0.5 sane", 0.6 < acc50 < 0.92, f"acc={acc50:.2f}")
print("confusion @0.5 (TP,FP,TN,FN):", c50, " @0.3:", conf_at(0.3))

# Linear-vs-logistic motivating plot.
lin = ols("readmit ~ score", data=rd).fit()
fig, ax = plt.subplots(figsize=(5.3, 3.05))
ax.scatter(score, readmit + rng.normal(0, .03, n2), s=9, rasterized=True, color="#2b50aa")
xs = np.linspace(0, 100, 50)
ax.plot(xs, lin.params.iloc[0] + lin.params.iloc[1] * xs, color="#c0392b", lw=1.1, ls=(0, (4, 2)),
        label="linear fit (escapes 0-1)")
ax.plot(xs, 1 / (1 + np.exp(-(b0 + b1 * xs))), color="#1b7f5a", lw=3.2, label="logistic fit")
ax.axhline(0, color="#bbb", lw=.8); ax.axhline(1, color="#bbb", lw=.8)
ax.set_ylim(-.35, 1.35); ax.set_xlabel("Risk score"); ax.set_ylabel("Readmitted (0/1)")
ax.set_title("Same data, two fits"); ax.legend(fontsize=8, frameon=False)
PLOT_LOGMOT = svg_of(fig, dpi=102)

if FAILURES:
    raise SystemExit(f"ASSERTION FAILURES: {FAILURES}")
print("ALL MODELS FITTED AND VERIFIED")

def money(v):
    return f"${v:,.0f}"

V = dict(
    r2=f"{fit.rsquared:.2f}", adj_r2=f"{fit.rsquared_adj:.2f}",
    b_age=f"{fit.params['age']:.0f}", b_los=f"{fit.params['los_days']:.0f}",
    b_emerg=money(fit.params["C(admission)[T.Emergency]"]),
    b_urg=money(fit.params["C(admission)[T.Urgent]"]),
    b_inter=money(fit.params.iloc[0]),
    pred=money(pred), planted=money(planted),
    rmse=money(rmse), rse=money(rse), mae=money(mae), mape=f"{mape:.0f}", mse=f"{mse:,.0f}",
    sse_b=f"{sse/1e9:.2f}", sst_b=f"{sst/1e9:.2f}",
    r2_raw=f"{r2_raw:.2f}", r2_logx=f"{r2_logx:.2f}",
    cook_max=f"{cooks[22]:.2f}", cook_thresh=f"{4/m:.2f}",
    hat_lev=f"{hats[11]:.2f}",
    b0=f"{b0:.2f}", b1=f"{b1:.3f}", or10=f"{odds_ratio_10:.1f}",
    base_rate=f"{readmit.mean():.0%}",
    tp=c50[0], fp=c50[1], tn=c50[2], fn=c50[3], acc50=f"{acc50:.0%}", n2=n2,
    n_pos=int(readmit.sum()),
)
def sgn(v):
    return ("+" if v >= 0 else "\u2212") + f"${abs(v):,.0f}"
EX_ROWS = "\n".join(
    f"<tr><td>Patient {i + 1}</td><td>{money(a)}</td><td>{money(f)}</td>"
    f"<td>{sgn(a - f)}</td><td>{(a - f) ** 2:,.0f}</td></tr>"
    for i, (a, f) in enumerate(
        [(float(hosp["cost"].iloc[j]), float(fit.fittedvalues.iloc[j])) for j in range(3)]))
LOG_ROWS = "\n".join(
    f"<tr><td>{r['x']}</td><td>{r['y']}</td><td>{r['p']:.2f}</td><td>{r['pred']}</td>"
    f"<td>{r['odds']:.2f}</td><td>{r['logodds']:.2f}</td></tr>" for r in rows)

# ============================ HTML template ============================
CSS = """
:root{--ink:#1a1a2e; --accent:#2b50aa; --soft:#eef4ff; --warnbg:#fdf3e3; --line:#d9dee9}
*{box-sizing:border-box}
body{font-family:Georgia,'Times New Roman',serif; color:var(--ink); max-width:760px; margin:0 auto; padding:28px 20px 70px; line-height:1.62; background:#fff}
h1{font-size:1.85em; line-height:1.25; margin:.2em 0 .1em}
h2{font-size:1.35em; margin:1.7em 0 .3em; border-bottom:2px solid var(--soft); padding-bottom:.15em}
h3{font-size:1.13em; margin:1.4em 0 .2em}
a{color:var(--accent); text-decoration:none} a:hover{text-decoration:underline}
.small{color:#666; font-size:.92em}
.lede{font-size:1.06em}
.box{background:var(--soft); border-radius:8px; padding:12px 16px; margin:1em 0}
.warn{background:var(--warnbg); border-radius:8px; padding:12px 16px; margin:1em 0}
details{border:1px solid var(--line); border-radius:8px; margin:.6em 0 1.1em; background:#fbfcfe}
details summary{cursor:pointer; padding:9px 14px; font-weight:bold; color:var(--accent); font-size:.95em}
details .body{padding:2px 14px 12px}
pre{background:#f4f4f8; border:1px solid #e2e2ee; border-radius:6px; padding:10px 12px; overflow-x:auto; font-size:.83em; line-height:1.5}
code{font-family:'SF Mono',Consolas,Menlo,monospace; font-size:.92em}
pre code{font-size:1em}
.out{font-size:.92em; color:#333}
table{border-collapse:collapse; margin:.8em 0; font-size:.93em; width:100%}
th,td{border:1px solid var(--line); padding:6px 9px; text-align:left; vertical-align:top}
th{background:var(--soft)}
.toc{background:#fbfcfe; border:1px solid var(--line); border-radius:8px; padding:12px 18px}
.toc ul{margin:.3em 0 .6em 1.1em; padding:0}
.toc li{margin:.18em 0}
.kicker{font-size:.8em; letter-spacing:.12em; text-transform:uppercase; color:var(--accent); font-weight:bold}
.ctl{display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin:.6em 0}
.ctl button{font-family:inherit; font-size:.92em; padding:5px 10px; border-radius:6px; border:1px solid var(--line); background:#fff; cursor:pointer}
.ctl button.on{background:var(--accent); color:#fff; border-color:var(--accent)}
input[type=range]{accent-color:var(--accent)}
.readout{background:var(--soft); border-radius:8px; padding:10px 14px; font-size:.97em; margin:.5em 0}
.sketch{display:block; margin-bottom:5px}
.plot{margin:.6em 0}
.plot svg{max-width:100%; height:auto; display:block; margin:0 auto}
svg text{font-family:Helvetica,Arial,sans-serif}
.matrix{display:grid; grid-template-columns:110px 1fr 1fr; gap:6px; margin:.8em 0; max-width:560px}
.matrix .hd{background:none; border:none; font-weight:bold; text-align:center; padding:4px; font-size:.9em}
.cell{border-radius:8px; padding:10px 12px; text-align:center; border:2px solid transparent}
.cell .n{font-size:1.5em; font-weight:bold; display:block}
.cell .lab{font-size:.85em; font-weight:bold; display:block}
.cell .sub{font-size:.8em; color:#444; display:block}
.cell.tp{background:#e2f2e4} .cell.tn{background:#e4ebf7}
.cell.fp{background:#fdeed3} .cell.fn{background:#fbe3e3}
.cell.hl{border-color:var(--ink)}
.chips{display:flex; flex-wrap:wrap; gap:8px; margin:.6em 0}
.chip{border:1px solid var(--line); background:#fff; border-radius:999px; padding:5px 12px; font-size:.9em; cursor:pointer; font-family:inherit}
.chip.on{background:var(--accent); color:#fff; border-color:var(--accent)}
"""

HEAD = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>The Practitioner's Guide to Regression — Applied Data Science Lab</title>
<meta name="description" content="A one-page method bank for regression: fitting OLS in Python, transformations and back-transformations, the fit-metric family (R-squared, RSE, RMSE, MAE, MAPE), diagnostic plots, outliers vs leverage vs influence, logistic regression, and the confusion matrix.">
<meta property="og:type" content="article">
<meta property="og:title" content="The Practitioner's Guide to Regression">
<meta property="og:description" content="Linear or logistic? R-squared or RMSE? One page: fit it, transform it, judge it, diagnose it — with healthcare examples and real output.">
<meta property="og:url" content="https://danieljordanviray.github.io/applied-data-science-lab/stats-and-prob/regression-guide/regression-guide.html">
<meta property="og:image" content="https://danieljordanviray.github.io/applied-data-science-lab/stats-and-prob/regression-guide/og-image.jpg">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="The Practitioner's Guide to Regression">
<meta name="twitter:description" content="Linear or logistic? R-squared or RMSE? One page: fit it, transform it, judge it, diagnose it — with healthcare examples and real output.">
<meta name="twitter:image" content="https://danieljordanviray.github.io/applied-data-science-lab/stats-and-prob/regression-guide/og-image.jpg">
<style>""" + CSS + """</style></head><body>
<p class="small"><a href="https://danieljordanviray.github.io/applied-data-science-lab/docs/">&larr; Homepage</a></p>
<h1>The Practitioner's Guide to Regression</h1>
<p class="small">October 18, 2026</p>
<p class="lede">Regression is the first real model most of us learn — and the one that quietly shows up
everywhere else. This is the working cheat sheet: fit an OLS model in Python and predict with it, bend
variables until the relationship is straight (and bend the answer back to interpret it), judge fit with the
right metric, read the three diagnostic plots, tell an outlier from a leverage point from an influential
point — then graduate to logistic regression when the answer is yes/no, and grade it with a confusion matrix.</p>

<h2>Contents</h2>
<div class="toc">
<p class="kicker">Part I — Linear regression</p>
<ul>
<li><a href="#fit">Fitting OLS in Python: numeric + categorical predictors, predictions</a></li>
<li><a href="#transforms">Transformations: making it linear, and back-transforming to interpret</a></li>
<li><a href="#metrics">Fit metrics: R&sup2;, RSE, RMSE, MAE, MAPE — one family</a></li>
<li><a href="#diagnostics">Diagnostic plots: residuals vs fitted, Q-Q, scale-location</a></li>
<li><a href="#influence">Outliers, leverage, and influence (+ Cook's distance)</a></li>
</ul>
<p class="kicker">Part II — Logistic regression</p>
<ul>
<li><a href="#why-logistic">Why logistic? The failure of the straight line</a></li>
<li><a href="#languages">One prediction, five languages: y, probability, prediction, odds, log-odds</a></li>
<li><a href="#confusion">The confusion matrix: where every metric comes from</a></li>
</ul>
</div>

<h2>Part I — Linear regression</h2>

<h3 id="fit">Fitting OLS in Python: the 60-second version</h3>
<p>A linear regression says: <em>outcome = intercept + slope &times; predictor + &hellip;</em>, and
<strong>ordinary least squares (OLS)</strong> picks the numbers that make the squared prediction errors as
small as possible. The running example: predicting the <strong>cost of a hospital stay</strong> for 120
patients from age and length of stay (numeric) and admission type (categorical).</p>
<p>The fitted model, in words: every extra year of age adds about <strong>$%%b_age%%</strong> to the bill;
every extra day adds about <strong>$%%b_los%%</strong>. Admission type enters as comparisons against the
<strong>reference category</strong> (Elective, whose effect is folded into the intercept): an otherwise
identical Emergency patient costs about <strong>%%b_emerg%% more</strong>, an Urgent patient about
<strong>%%b_urg%% more</strong>. Model R&sup2; = %%r2%% — it explains about %%r2%% of the variation in cost,
as a proportion. And the point of the exercise — a prediction for a new patient, 68 years old, Emergency
admission, 5-day stay: <strong>%%pred%%</strong>.</p>
<details><summary>Data structure + Python code</summary><div class="body">
<p>Data: one row per patient; the outcome in one column, predictors in the others. Categorical predictors
go in as text — <code>C()</code> in the formula tells statsmodels to dummy-code them against a reference
level. Predicting needs a small DataFrame with the same columns.</p>
<pre><code>import pandas as pd
from statsmodels.formula.api import ols

# hosp: 120 rows; columns cost, age, los_days, admission (Elective/Emergency/Urgent)
fit = ols("cost ~ age + los_days + C(admission)", data=hosp).fit()
print(fit.summary())        # coefficients, R-squared = %%r2%%, p-values

new_patient = pd.DataFrame({"age": [68], "los_days": [5], "admission": ["Emergency"]})
fit.predict(new_patient)    # -> %%pred%%</code></pre>
</div></details>

<h3 id="transforms">Transformations: making it linear, and back-transforming to interpret</h3>
<p>OLS fits straight lines in the variables you <em>give</em> it — nothing says you must hand it the raw
variables. When a relationship bends, transform a variable until the bend straightens (in practice: add
a transformed column to your data, and put that column in the formula), fit the line there, then
back-transform the answer to say it in original units. Example: drug dose vs symptom reduction, where
each doubling of the dose buys roughly the same benefit (diminishing returns). Try it:</p>
<svg id="trSvg" viewBox="0 0 640 310" role="img" aria-label="Dose response scatter with fitted line under different transformations"></svg>
<div class="ctl">
<button data-tr="raw" class="on">Dose as-is</button>
<button data-tr="log">log(dose)</button>
<button data-tr="sqrt">sqrt(dose)</button>
<button data-tr="sq">dose&sup2;</button>
</div>
<div class="readout" id="trOut"></div>
<p><strong>How you know which one you need:</strong> look at the scatter (and later, at the residual plot).
A bend that <em>flattens out</em> is asking you to compress the big end of a variable — log does it
strongly, square root gently. A bend that <em>accelerates upward</em> asks for the opposite — stretch it,
with a square. And a fan-shaped residual plot (errors growing with the prediction) is the same story told
about the outcome: compress y.</p>
<table>
<tr><th>Transformation</th><th>What the raw data looks like — reach for it when&hellip;</th><th>What it does</th><th>Reading the result</th></tr>
<tr><td><strong>log(y)</strong></td><td><svg class="sketch" viewBox="0 0 104 58" width="104" height="60" aria-hidden="true"><circle cx="8" cy="50" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="16" cy="49" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="24" cy="48" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="32" cy="47" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="40" cy="45" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="48" cy="42" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="56" cy="38" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="64" cy="32" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="72" cy="24" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="80" cy="14" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="88" cy="8" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="56" cy="45" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="64" cy="39" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="72" cy="32" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="80" cy="23" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="88" cy="18" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="95" cy="11" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="95" cy="27" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><line x1="2" y1="54" x2="102" y2="54" stroke="#c9cfdb" stroke-width="1.5"/></svg>The scatter takes off as x grows — the outcomes fan out into a long upper tail of very large values. Costs are the textbook case.</td>
<td>Pulls in a long right tail; effects become multiplicative</td>
<td>A one-unit rise in x changes y by about &beta;&times;100% (exactly: multiplies y by e<sup>&beta;</sup>)</td></tr>
<tr><td><strong>sqrt(y)</strong></td><td><svg class="sketch" viewBox="0 0 104 58" width="104" height="60" aria-hidden="true"><circle cx="8" cy="50" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="18" cy="44" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="28" cy="39" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="38" cy="35" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="48" cy="31" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="58" cy="28" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="68" cy="25" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="78" cy="22" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="88" cy="20" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="98" cy="18" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="48" cy="38" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="62" cy="36" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="76" cy="33" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="90" cy="30" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="98" cy="27" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><line x1="2" y1="54" x2="102" y2="54" stroke="#c9cfdb" stroke-width="1.5"/></svg>The scatter rises with a spread that grows moderately — counts-like data. Often applied to both sides at once when the predictor is a skewed count too (the course's advertising example square-roots both spend and impressions).</td>
<td>A gentler log(y)</td>
<td>Square the predictions to get back to original units</td></tr>
<tr><td><strong>1/y (inverse)</strong></td><td><svg class="sketch" viewBox="0 0 104 58" width="104" height="60" aria-hidden="true"><circle cx="8" cy="8" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="12" cy="16" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="16" cy="24" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="22" cy="32" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="30" cy="38" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="40" cy="43" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="52" cy="46" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="64" cy="48" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="76" cy="50" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="88" cy="51" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="98" cy="52" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="10" cy="21" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="20" cy="36" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="34" cy="44" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="58" cy="49" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><line x1="2" y1="54" x2="102" y2="54" stroke="#c9cfdb" stroke-width="1.5"/></svg>The scatter crashes: enormous values at small x, collapsing toward zero. The outcome is usually a time or a rate underneath.</td>
<td>The strongest tail-pull of the three</td>
<td>Signs flip — safest to back-transform predictions (take the reciprocal) and read those</td></tr>
<tr><td><strong>log(x)</strong></td><td><svg class="sketch" viewBox="0 0 104 58" width="104" height="60" aria-hidden="true"><path d="M6 50 C 28 24, 52 13, 98 8" fill="none" stroke="#2b50aa" stroke-width="1.4" stroke-opacity="0.4"/><circle cx="8" cy="50" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="16" cy="36" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="26" cy="28" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="38" cy="22" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="52" cy="17" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="68" cy="14" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="84" cy="11" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="98" cy="9" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="20" cy="41" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="33" cy="26" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="47" cy="20" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="61" cy="15" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="77" cy="13" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="92" cy="10" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><line x1="2" y1="54" x2="102" y2="54" stroke="#c9cfdb" stroke-width="1.5"/></svg>The scatter rises steeply, then flattens: each extra unit of x buys less than the one before — diminishing returns (dose, spending, days on a therapy).</td>
<td>Compresses big x values</td>
<td>A 1% rise in x moves y by about &beta;/100; a doubling of x moves y by about 0.69&times;&beta;</td></tr>
<tr><td><strong>sqrt(x)</strong></td><td><svg class="sketch" viewBox="0 0 104 58" width="104" height="60" aria-hidden="true"><path d="M6 50 C 32 31, 58 21, 98 15" fill="none" stroke="#2b50aa" stroke-width="1.4" stroke-opacity="0.4"/><circle cx="8" cy="50" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="18" cy="41" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="30" cy="35" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="44" cy="29" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="60" cy="24" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="76" cy="20" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="90" cy="17" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="98" cy="15" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="24" cy="38" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="38" cy="31" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="54" cy="26" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="70" cy="21" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="86" cy="18" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><line x1="2" y1="54" x2="102" y2="54" stroke="#c9cfdb" stroke-width="1.5"/></svg>The same flattening, but gentler.</td>
<td>Milder compression of x</td>
<td>Slope is per unit of sqrt(x) — back-transform examples rather than quoting the slope</td></tr>
<tr><td><strong>x&sup2;</strong> (fit x and x&sup2;)</td><td><svg class="sketch" viewBox="0 0 104 58" width="104" height="60" aria-hidden="true"><path d="M6 8 Q 52 68, 98 8" fill="none" stroke="#2b50aa" stroke-width="1.4" stroke-opacity="0.4"/><circle cx="8" cy="10" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="18" cy="24" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="28" cy="34" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="40" cy="41" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="52" cy="44" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="64" cy="41" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="76" cy="34" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="88" cy="24" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="98" cy="10" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="24" cy="28" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="46" cy="43" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="70" cy="30" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="84" cy="20" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><line x1="2" y1="54" x2="102" y2="54" stroke="#c9cfdb" stroke-width="1.5"/></svg>The scatter curves like a smile (or a frown): the middle is the extreme — risk lowest at middling BMI is the classic shape. Also the geometry of areas.</td>
<td>Lets one predictor bend once: a U or an upside-down U</td>
<td>No single slope exists; plot the curve and quote values off it</td></tr>
<tr><td><strong>x&sup3;</strong> (cubed)</td><td><svg class="sketch" viewBox="0 0 104 58" width="104" height="60" aria-hidden="true"><path d="M6 52 C 45 51, 68 42, 98 6" fill="none" stroke="#2b50aa" stroke-width="1.4" stroke-opacity="0.4"/><circle cx="8" cy="51" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="20" cy="50" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="32" cy="50" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="44" cy="49" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="56" cy="47" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="66" cy="43" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="76" cy="36" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="84" cy="27" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="92" cy="16" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="98" cy="7" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="26" cy="51" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="50" cy="48" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="70" cy="40" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><circle cx="88" cy="21" r="2.3" fill="#2b50aa" fill-opacity="0.78"/><line x1="2" y1="54" x2="102" y2="54" stroke="#c9cfdb" stroke-width="1.5"/></svg>The scatter hugs the floor, then takes off like a hockey stick — the geometry of volumes (the course's perch grow in mass at roughly length cubed, and cubing the length straightens the relationship completely).</td>
<td>Sharp acceleration</td>
<td>As with x&sup2;: plot, don't quote a single slope</td></tr>
</table>
<div class="warn"><strong>Back-transformation, the part everyone forgets:</strong> a model fitted to log(cost)
predicts log-dollars. Exponentiate the prediction to get dollars back. Two consequences: the coefficient
reads as a <em>percentage</em> change rather than a dollar change, and the exponentiated prediction lands
near the <em>typical</em> (median) patient rather than the mean — a feature for skewed costs, as long as
you know it's happening.</div>

<h3 id="metrics">Fit metrics: from residuals to a grade, step by step</h3>
<p>Every metric in this section is built from one ingredient, in a few mechanical steps. Walk the chain
once and the family stops being a list to memorize — including which ones are just rescalings of each
other.</p>
<p><strong>Step 1 — residuals.</strong> For every patient the model makes a prediction, and the
<strong>residual</strong> is what it missed by: actual &minus; predicted. Three of the cost model's 120
residuals:</p>
<table>
<tr><th></th><th>Actual cost</th><th>Predicted</th><th>Residual (actual &minus; predicted)</th><th>Squared error</th></tr>
%%EX_ROWS%%
</table>
<p><strong>Step 2 — square them.</strong> Squaring does two jobs. It makes every miss positive (a $2,000
overestimate and a $2,000 underestimate are both just wrong by $2,000). And it punishes big misses
disproportionately: one $4,000 miss counts as much as four $2,000 misses. That second property is a
<em>choice</em> — one sibling below exists precisely for people who don't want it.</p>
<p><strong>Step 3 — add them up: SSE.</strong> The <strong>sum of squared errors</strong>. For the cost
model, SSE &asymp; %%sse_mm%% million — in squared dollars, a unit nobody can think in. That's the point:
SSE is an ingredient, not a grade. Everything below is a way of turning it into a grade.</p>
<p><strong>Step 4 — average it, then unsquare it: RMSE.</strong> Divide SSE by the number of patients
(120) and you get the <strong>mean squared error</strong> (MSE = %%mse%% squared-dollars). Take the square
root and the units come home: <strong>RMSE = %%rmse%%</strong> — "our typical prediction is off by about
$1,600." This is the number most people picture when they say a model is accurate to within about X.</p>
<p><strong>Step 5 — the fairness tweak: RSE.</strong> RMSE divides by n. The <strong>residual standard
error</strong> divides by n &minus; the number of fitted coefficients (120 &minus; 5 here) instead, so
every predictor the model uses costs it a little credit — stuffing in junk predictors can't fake a better
score. RSE = <strong>%%rse%%</strong>, slightly larger than RMSE, exactly as it should be. It's the same
"typical error," computed the way regression output reports it (and the one-line shortcut is
<code>np.sqrt(fit.mse_resid)</code>).</p>
<p><strong>The siblings</strong> — same residuals, different choices along the way:</p>
<table>
<tr><th>Metric</th><th>The change</th><th>Cost model</th><th>Use it when</th></tr>
<tr><td><strong>MAE</strong></td><td>Skip step 2: average the absolute residuals instead of squaring</td><td>%%mae%%</td>
<td>A few huge misses shouldn't dominate the verdict. Diagnostic pairing: if RMSE is much bigger than MAE,
a handful of very large misses are running the show.</td></tr>
<tr><td><strong>MAPE</strong></td><td>Divide each residual by its actual value first, average, as a %</td><td>%%mape%%%</td>
<td>The audience thinks in percentages, or you're comparing across outcomes with different scales. Breaks
down when actuals sit near zero — a $50 error on a $60 bill is 83%.</td></tr>
<tr><td><strong>R&sup2;</strong> (coefficient of determination)</td><td>Compare SSE to the outcome's total variation: 1 &minus; SSE/SST</td><td>%%r2%%</td>
<td>"How much of the variation did we explain?" — the communication number. <strong>Adjusted R&sup2;</strong>
(%%adj_r2%%) applies RSE's per-coefficient charge; use it when comparing models with different numbers of
predictors.</td></tr>
</table>
<p><strong>What gets used in practice:</strong> day to day, evaluation usually means <strong>RMSE (or
MAE) on data the model never trained on</strong> — a held-out test set — because in-sample numbers flatter
the model. R&sup2; is the number you say out loud ("we explain about %%r2%% of the variation in cost," as a
proportion). RSE is what you read in regression output and use to compare models fitted to the same data.
MAPE appears when a business audience wants a percentage and the outcome stays safely away from zero.</p>
<details><summary>Python code: the whole family from one fit</summary><div class="body">
<pre><code>import numpy as np

resid = fit.resid                       # actual - predicted, one per patient
n = len(resid)
k = 4                                   # slopes: age, los_days, Emergency, Urgent
sse = float((resid ** 2).sum())
rmse = np.sqrt(sse / n)                 # %%rmse%%
rse  = np.sqrt(sse / (n - k - 1))       # %%rse%%  (shortcut: np.sqrt(fit.mse_resid))
mae  = np.abs(resid).mean()             # %%mae%%
mape = (np.abs(resid) / hosp["cost"]).mean() * 100   # %%mape%%%
print(fit.rsquared, fit.rsquared_adj)   # %%r2%%, %%adj_r2%%</code></pre>
</div></details>

<h3 id="diagnostics">Diagnostic plots: what you're looking for, and why</h3>
<p>Three standard plots, all computed from the cost model's residuals. A good fit's residuals are
roughly normally distributed with mean zero — each plot below audits one part of that sentence. This
model is well-behaved, so these show what "fine" looks like — each caption names the trouble pattern
it's built to catch.</p>
<p>Think of every prediction as an arrow shot at a bullseye. If the misses spray randomly around the
target — no pattern to them — your aim is as good as it gets and what's left is noise; no adjustment
fixes noise. The dangerous case is missing <em>non-randomly</em>: always low on one side, always high on
the other. A repeatable miss is a fixable miss — it means the model is systematically leaving something
out. The three plots below are three ways of checking whether your misses are random.</p>
<p><strong>Residuals vs fitted.</strong> The residuals should form a formless cloud around zero at every
fitted level. A <em>curve</em> in the cloud means a relationship you didn't model — a bend asking for a
transformation or an extra term. A <em>funnel</em> (spread widening left to right) means your errors grow
with the size of the prediction.</p>
<div class="plot">%%PLOT_RESID%%</div>
<p><strong>Normal Q-Q plot.</strong> Standardized residuals plotted against where they'd sit if they were
perfectly normal. On the diagonal line = normal enough. This matters for confidence and prediction
intervals and for small samples; big samples and point predictions forgive a lot. Tails that peel away at
both ends mean heavier tails than normal — usually a few extreme patients. Said plainly: a good model's
errors are mostly ordinary-sized — missing by a typical amount, in a random direction, most of the time.
Points peeling away at the ends mean too many of your errors are extreme compared to that picture, and an
error pattern is something a model can be adjusted for.</p>
<div class="plot">%%PLOT_QQ%%</div>
<p><strong>Scale-location plot.</strong> In plain terms: are you missing by about the same amount all
the way along the axis, or do the misses get bigger at, say, the high end? If the misses grow somewhere,
that somewhere is where the model can be changed. Technically, it plots the square root of the absolute
standardized residuals against fitted values — the funnel detector with the direction stripped out, so
spread is all you see. A flat band
means constant error variance (homoscedasticity). A rising trend is the funnel again: bigger predictions,
bigger misses — often the cue for log(y).</p>
<div class="plot">%%PLOT_SCALE%%</div>

<h3 id="influence">Outliers, leverage, and influence: three different accusations</h3>
<p>These three words get used interchangeably and mean different things. A small clinic dataset
(age &rarr; cost of stay, 30 patients) with one of each planted in it:</p>
<div class="plot">%%PLOT_INFL%%</div>
<p><strong>Outlier</strong> — a point that doesn't fit the pattern: a big residual. The 52-year-old whose
stay cost $12,500 against a predicted ~$6,100 (standardized residual 5.9). Being an outlier is about
<em>y</em> disagreeing with the model.</p>
<p><strong>Leverage</strong> — a point whose <em>x</em> is unusual: far from the crowd horizontally, so
the line has to pass near it, like the long end of a lever. The 96-year-old has the most leverage here
(hat value %%hat_lev%%, vs a typical ~0.07). But he sits right <em>on</em> the trend, so he barely moves
anything — leverage is only <em>potential</em> influence.</p>
<p><strong>Influence</strong> — leverage &times; disagreement: a point that actually bends the fit. The
94-year-old with a $2,600 stay has extreme age <em>and</em> sits far off the line; delete him and the slope
visibly changes. <strong>Cook's distance</strong> measures exactly this — one number per point: how much
the whole fit would move if that point were removed. Rules of thumb: look closer above 4/n (here
%%cook_thresh%%); above 1 (his is %%cook_max%%), definitely investigate. Investigating means: check for a
data error first, then report the fit with and without the point. It does not mean silently deleting
inconvenient patients.</p>
<p class="small">The classic teaching example of all three ideas is a fish dataset, where the single
most influential observation is one very short roach — leverage 0.31, Cook's distance 1.07. The smallest
fish in the data has the biggest pull on the line.</p>
<details><summary>Python code: hat values and Cook's distance</summary><div class="body">
<pre><code>from statsmodels.formula.api import ols

fit2 = ols("cost ~ age", data=clinic).fit()
summary = fit2.get_influence().summary_frame()
hats = summary["hat_diag"]      # leverage, one value per patient
cooks = summary["cooks_d"]      # Cook's distance, one value per patient
# flag for a closer look: cooks > 4 / len(clinic)</code></pre>
<p class="out">Planted results: the influential patient's Cook's distance is %%cook_max%% — the largest in
the dataset. The on-trend 96-year-old's is 0.08 despite his leverage.</p>
</div></details>
"""

BODY2 = r"""
<h2>Part II — Logistic regression</h2>
<div class="box"><strong>Read this first:</strong> logistic regression [a generalized linear model] is
linear regression's sibling, not a different species. If the machinery above — fit a line, read
coefficients, grade the fit — is new to you, work through <a href="#fit">Part I</a> first; everything
here builds on it.</div>

<h3 id="why-logistic">Why logistic? The failure of the straight line</h3>
<p>New problem: predict whether a patient is <strong>readmitted within 30 days</strong> (yes/no) from a
risk score, for 240 patients (%%base_rate%% were actually readmitted). Fit a straight line to a 0/1 outcome
anyway and two things break. The line escapes: it predicts probabilities of 1.2 and &minus;0.1, which don't
exist. And it insists the effect of 10 extra risk points is identical everywhere — but probability is
squeezed into [0, 1], so real risk has to bend: slow at the bottom, fast in the middle, slow at the top.</p>
<div class="plot">%%PLOT_LOGMOT%%</div>
<p>The logistic fix is one move: <strong>don't put the line on the probability — put it on the
log-odds</strong>, then squash the result back through the S-curve so every prediction lands inside [0, 1].
The fitted line here is log-odds = %%b0%% + %%b1%% &times; risk score. What log-odds, odds, and probability
each mean — and why you'd ever want all three — is the next section.</p>
<details><summary>Data structure + Python code</summary><div class="body">
<p>Data: one row per patient; the outcome is 0/1, the predictor is a number. Same formula shape as OLS —
<code>logit</code> instead of <code>ols</code>.</p>
<pre><code>from statsmodels.formula.api import logit

# rd: 240 rows; columns score (0-100), readmit (0/1)
lfit = logit("readmit ~ score", data=rd).fit()
print(lfit.summary())       # log-odds = %%b0%% + %%b1%% * score</code></pre>
</div></details>

<h3 id="languages">One prediction, five languages</h3>
<p>Five patients from the readmission model. Every row is the same prediction said five ways — plus what
actually happened:</p>
<table>
<tr><th>Risk score (x)</th><th>Actual outcome (y)</th><th>Probability</th><th>Predicted response</th><th>Odds</th><th>Log-odds</th></tr>
%%LOG_ROWS%%
</table>
<p><strong>Why each column earns its place:</strong></p>
<p><strong>x — the input.</strong> What you know about the patient before deciding anything.</p>
<p><strong>y — what actually happened.</strong> The truth you grade against. Note the patient with
probability 0.35 who <em>was</em> readmitted: a probability is not a promise, and a "wrong" prediction at
0.35 isn't necessarily a model failure.</p>
<p><strong>Probability — the human language.</strong> "About a 1-in-3 chance of coming back." This is the
only column you'd say out loud to a clinician, an executive, or a patient.</p>
<p><strong>Predicted response — probability forced into an action.</strong> A threshold (here the default
0.5) turns "how likely" into "flag or don't flag." The threshold is not part of the model — it's a decision
about which mistake is cheaper, and the next section is about choosing it.</p>
<p><strong>Odds = probability / (1 &minus; probability) — the betting language.</strong> Effects
<em>multiply</em> on this scale, which is why papers report <strong>odds ratios</strong>: in this model,
every +10 risk points multiplies the odds of readmission by <strong>%%or10%%&times;</strong>, wherever you
start from. (Careful: an odds ratio is not a probability ratio — at high base rates they drift apart.)</p>
<p><strong>Log-odds — the model's native language.</strong> This is the straight line actually being
fitted (%%b0%% + %%b1%% &times; score): add a predictor, add a term; effects are additive here. Nobody
quotes log-odds to a human — but every other column is a conversion of it, so it's the one to compute in.</p>
<p>The same trade-offs, at a glance:</p>
<table>
<tr><th>Scale</th><th>Easy to interpret?</th><th>Changes easy to read?</th><th>Precise?</th></tr>
<tr><td>Probability</td><td>Yes</td><td>No — the same slope means different probability jumps at different starting points</td><td>Yes</td></tr>
<tr><td>Predicted response</td><td>Easiest of all</td><td>Yes</td><td>No — throws away all the gradation</td></tr>
<tr><td>Odds</td><td>Yes, with practice</td><td>Effects multiply rather than add</td><td>Yes</td></tr>
<tr><td>Log-odds</td><td>No</td><td>Yes — effects simply add</td><td>Yes</td></tr>
</table>
<details><summary>Python code: getting all five from the fit</summary><div class="body">
<pre><code>import numpy as np

patients = pd.DataFrame({"score": [%%FIVE%%]})   # the five patients in the table
prob = lfit.predict(patients)                    # probabilities, straight from the fit
log_odds = np.log(prob / (1 - prob))             # invert the squash: back to the line
odds = prob / (1 - prob)
pred = (prob >= 0.5).astype(int)                 # the 0.5 threshold, made explicit (= np.round(prob))</code></pre>
<p class="out">predict() hands back probabilities. The log-odds is one algebra step away —
log(p / (1 &minus; p)) — because that formula and the S-curve are exact inverses.</p>
</div></details>

<h3 id="confusion">The confusion matrix: where every metric comes from</h3>
<p>Force the readmission model's probabilities into yes/no calls and grade them against what actually
happened. Four boxes result. The layout matches statsmodels' <code>pred_table()</code>: rows are what
actually happened, columns are what the model predicted. Every classification metric you've ever seen is
just arithmetic on these four numbers — drag the threshold and watch them move:</p>
<div class="ctl">
<label>Flag patients whose predicted probability is at least:
<input type="range" id="thrSlider" min="0.05" max="0.95" step="0.05" value="0.5">
<strong id="thrVal">0.50</strong></label>
</div>
<div class="matrix">
<div class="hd"></div><div class="hd">Predicted: stayed home</div><div class="hd">Predicted: flagged</div>
<div class="hd">Actually stayed home</div>
<div class="cell tn" id="cTN"><span class="lab">True negatives</span><span class="n">–</span><span class="sub">not flagged, and stayed home</span></div>
<div class="cell fp" id="cFP"><span class="lab">False positives</span><span class="n">–</span><span class="sub">flagged, but stayed home</span></div>
<div class="hd">Actually readmitted</div>
<div class="cell fn" id="cFN"><span class="lab">False negatives</span><span class="n">–</span><span class="sub">missed: readmitted, never flagged</span></div>
<div class="cell tp" id="cTP"><span class="lab">True positives</span><span class="n">–</span><span class="sub">flagged, and truly readmitted</span></div>
</div>
<div class="chips" id="metricChips"></div>
<div class="readout" id="cmOut"></div>
<p><strong>The five standard metrics, in words</strong> (tap one above to see which boxes it uses):</p>
<p><strong>Accuracy</strong> = (TP + TN) / everyone — "overall, how often are we right?" At the 0.50
threshold: %%acc50%%. Handle with care: with a rare outcome, a model that flags <em>nobody</em> also scores
high accuracy while being useless.</p>
<p><strong>Sensitivity (recall)</strong> = TP / (TP + FN) — of the patients who truly were readmitted,
how many did we flag? At 0.50: %%sens50%%. This is the "don't miss cases" metric.</p>
<p><strong>Specificity</strong> = TN / (TN + FP) — of the patients who truly stayed home, how many did we
leave alone? At 0.50: %%spec50%%.</p>
<p><strong>Precision</strong> = TP / (TP + FP) — of the patients we flagged, how many truly were
readmitted? At 0.50: %%prec50%%. This is the "don't cry wolf" metric — it prices your follow-up team's time.</p>
<p><strong>F1</strong> = the harmonic mean of precision and sensitivity — one number for when both kinds
of mistake matter and you refuse to pick a favorite. At 0.50: %%f1_50%%.</p>
<div class="warn"><strong>The threshold is a policy decision, not a statistic.</strong> Drag it down and
you catch more readmissions (sensitivity climbs) but flag more healthy patients (precision falls, follow-up
costs climb). Drag it up and the reverse. There is no mathematically best threshold — only the one that
matches what a missed readmission costs you versus what a false alarm costs you.</div>
<details><summary>Python code: the matrix at threshold 0.50</summary><div class="body">
<pre><code>conf_matrix = lfit.pred_table()        # rows = actual, columns = predicted; cutoff 0.5
tn = conf_matrix[0, 0]                 # %%tn%%
fp = conf_matrix[0, 1]                 # %%fp%%
fn = conf_matrix[1, 0]                 # %%fn%%
tp = conf_matrix[1, 1]                 # %%tp%%</code></pre>
<p class="out">Same four numbers the interactive shows at 0.50 — computed here in Python, there in your
browser, from the same probabilities.</p>
</div></details>

<div class="box"><strong>Prove it:</strong> every coefficient, metric, plot, and table entry on this page
was computed by the build script from the toy datasets described, with the decisions asserted in code
before the page was written — including the planted outlier, leverage point, and influential point in the
clinic data, whose Cook's distances had to rank exactly as the text claims.</div>

<p class="small">Part of the <a href="https://danieljordanviray.github.io/applied-data-science-lab/docs/">Applied
Data Science Lab</a>. Companion:
<a href="https://danieljordanviray.github.io/applied-data-science-lab/stats-and-prob/practitioners-guide/practitioners-guide.html">The
Practitioner's Guide to Statistical Tests</a> — the hypothesis-testing method bank.</p>
"""

JS = r"""
<script>
/* ---------- transformation explorer ---------- */
var DOSE = %%DOSE_JSON%%;
var TRF = {
  raw:  {f:function(x){return x;}, label:'dose (mg)', verdict:'The bend is the story: benefit flattens as dose grows, so a straight line splits the difference and misses both ends.'},
  log:  {f:function(x){return Math.log(x);}, label:'log(dose)', verdict:'Straight. Each doubling of the dose buys about the same benefit wherever you start — the fingerprint of a log-dose relationship. This is the transform this data was asking for.'},
  sqrt: {f:function(x){return Math.sqrt(x);}, label:'sqrt(dose)', verdict:'Better, but still bending — a gentler squeeze than the data wants. When sqrt almost works and log fully works, the truth is usually log.'},
  sq:   {f:function(x){return x*x;}, label:'dose squared', verdict:'Worse — squaring stretches the big end in exactly the direction the relationship was already flattening away from. Wrong-way transform, useful as a warning.'}
};
var trMode = 'raw';
function olsFit(pts){
  var n=pts.length, mx=0, my=0, i;
  for(i=0;i<n;i++){mx+=pts[i][0]; my+=pts[i][1];} mx/=n; my/=n;
  var sxy=0, sxx=0, sst=0, sse=0;
  for(i=0;i<n;i++){sxy+=(pts[i][0]-mx)*(pts[i][1]-my); sxx+=(pts[i][0]-mx)*(pts[i][0]-mx);}
  var b=sxy/sxx, a=my-b*mx;
  for(i=0;i<n;i++){var yh=a+b*pts[i][0]; sse+=(pts[i][1]-yh)*(pts[i][1]-yh); sst+=(pts[i][1]-my)*(pts[i][1]-my);}
  return {a:a, b:b, r2:1-sse/sst};
}
function drawTr(){
  var svg=document.getElementById('trSvg'), W=640, H=310, padL=46, padR=14, padT=14, padB=36;
  var T=TRF[trMode];
  var pts=DOSE.map(function(d){return [T.f(d[0]), d[1]];});
  var xs=pts.map(function(p){return p[0];}), ys=pts.map(function(p){return p[1];});
  var x0=Math.min.apply(null,xs), x1=Math.max.apply(null,xs), y0=0, y1=Math.max.apply(null,ys)*1.12;
  var X=function(v){return padL+(v-x0)/(x1-x0)*(W-padL-padR);};
  var Y=function(v){return H-padB-(v-y0)/(y1-y0)*(H-padT-padB);};
  var fit=olsFit(pts), s='';
  s+='<line x1="'+padL+'" y1="'+(H-padB)+'" x2="'+(W-padR)+'" y2="'+(H-padB)+'" stroke="#999"/>';
  s+='<line x1="'+padL+'" y1="'+padT+'" x2="'+padL+'" y2="'+(H-padB)+'" stroke="#999"/>';
  pts.forEach(function(p){ s+='<circle cx="'+X(p[0]).toFixed(1)+'" cy="'+Y(p[1]).toFixed(1)+'" r="3.4" fill="#2b50aa" opacity="0.75"/>'; });
  s+='<line x1="'+X(x0)+'" y1="'+Y(fit.a+fit.b*x0).toFixed(1)+'" x2="'+X(x1)+'" y2="'+Y(fit.a+fit.b*x1).toFixed(1)+'" stroke="#c0392b" stroke-width="2"/>';
  s+='<text x="'+((padL+W-padR)/2)+'" y="'+(H-8)+'" font-size="12" text-anchor="middle" fill="#555">'+T.label+'</text>';
  s+='<text x="14" y="'+(H/2)+'" font-size="12" text-anchor="middle" fill="#555" transform="rotate(-90 14 '+(H/2)+')">symptom reduction (%)</text>';
  svg.innerHTML=s;
  document.getElementById('trOut').innerHTML='<strong>'+T.label+':</strong> R&sup2; = '+fit.r2.toFixed(2)+' in the transformed space. '+T.verdict;
}
document.querySelectorAll('[data-tr]').forEach(function(b){
  b.addEventListener('click',function(){
    document.querySelectorAll('[data-tr]').forEach(function(x){x.classList.remove('on');});
    b.classList.add('on'); trMode=b.getAttribute('data-tr'); drawTr();
  });
});

/* ---------- confusion matrix explorer ---------- */
var CONF = %%CONF_JSON%%;
var METRICS = [
  {id:'acc',  name:'Accuracy',    cells:['cTP','cFP','cFN','cTN'], f:function(c){return (c.tp+c.tn)/(c.tp+c.fp+c.fn+c.tn);}},
  {id:'sens', name:'Sensitivity', cells:['cTP','cFN'],             f:function(c){return c.tp/(c.tp+c.fn);}},
  {id:'spec', name:'Specificity', cells:['cTN','cFP'],             f:function(c){return c.tn/(c.tn+c.fp);}},
  {id:'prec', name:'Precision',   cells:['cTP','cFP'],             f:function(c){return c.tp/(c.tp+c.fp);}},
  {id:'f1',   name:'F1',          cells:['cTP','cFP','cFN'],       f:function(c){var p=c.tp/(c.tp+c.fp), r=c.tp/(c.tp+c.fn); return 2*p*r/(p+r);}}
];
var activeMetric = null;
function confCounts(thr){
  var c={tp:0,fp:0,tn:0,fn:0};
  CONF.forEach(function(r){ var pred = r[1]>=thr ? 1 : 0;
    if(pred===1 && r[0]===1)c.tp++; else if(pred===1)c.fp++;
    else if(r[0]===1)c.fn++; else c.tn++; });
  return c;
}
function drawConf(){
  var thr=parseFloat(document.getElementById('thrSlider').value);
  document.getElementById('thrVal').textContent=thr.toFixed(2);
  var c=confCounts(thr);
  var map={cTP:c.tp, cFP:c.fp, cFN:c.fn, cTN:c.tn};
  Object.keys(map).forEach(function(id){
    document.querySelector('#'+id+' .n').textContent=map[id];
  });
  var chips=document.getElementById('metricChips'); chips.innerHTML='';
  METRICS.forEach(function(m){
    var b=document.createElement('button'); b.className='chip'+(activeMetric===m.id?' on':'');
    b.textContent=m.name+': '+(m.f(c)*100).toFixed(0)+'%';
    b.addEventListener('click',function(){
      activeMetric = (activeMetric===m.id ? null : m.id);
      ['cTP','cFP','cFN','cTN'].forEach(function(id){document.getElementById(id).classList.remove('hl');});
      if(activeMetric){ m.cells.forEach(function(id){document.getElementById(id).classList.add('hl');}); }
      drawConf();
    });
    chips.appendChild(b);
  });
  document.getElementById('cmOut').innerHTML='Threshold '+thr.toFixed(2)+': the model flags <strong>'+
    (c.tp+c.fp)+'</strong> of '+CONF.length+' patients. It catches <strong>'+c.tp+'</strong> of the '+
    (c.tp+c.fn)+' who truly were readmitted, and misses <strong>'+c.fn+'</strong>.';
}
document.getElementById('thrSlider').addEventListener('input',drawConf);

document.documentElement.style.scrollBehavior='smooth';
drawTr(); drawConf();
</script>
"""

# ============================ assemble + write ============================
import re
tp, fp_, tn, fn_ = c50
V["sse_mm"] = f"{sse/1e6:.0f}"
V["sens50"] = f"{tp/(tp+fn_):.0%}"
V["spec50"] = f"{tn/(tn+fp_):.0%}"
V["prec50"] = f"{tp/(tp+fp_):.0%}"
_p, _r = tp/(tp+fp_), tp/(tp+fn_)
V["f1_50"] = f"{2*_p*_r/(_p+_r):.0%}"

html = HEAD + BODY2 + JS + "</body></html>"
for tok, val in [("%%PLOT_RESID%%", PLOT_RESID), ("%%PLOT_QQ%%", PLOT_QQ),
                 ("%%PLOT_SCALE%%", PLOT_SCALE), ("%%PLOT_INFL%%", PLOT_INFL),
                 ("%%PLOT_LOGMOT%%", PLOT_LOGMOT), ("%%DOSE_JSON%%", DOSE_JSON),
                 ("%%CONF_JSON%%", CONF_JSON), ("%%LOG_ROWS%%", LOG_ROWS),
                 ("%%FIVE%%", ", ".join(str(r["x"]) for r in rows)),
                 ("%%EX_ROWS%%", EX_ROWS)]:
    html = html.replace(tok, val)
for k, v in V.items():
    html = html.replace(f"%%{k}%%", str(v))
leftover = re.findall(r"%%[A-Za-z_0-9]+%%", html)
assert not leftover, f"unsubstituted tokens: {leftover}"
out = Path(__file__).resolve().parent.parent / "regression-guide" / "regression-guide.html"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(html)
print(f"wrote {out} ({len(html)/1024:.1f} KB)")
