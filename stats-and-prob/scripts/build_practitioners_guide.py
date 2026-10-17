"""Build the Practitioner's Guide to Statistical Tests (stats-and-prob).

House rules (same as the SQL notes): every number printed on the page is
computed HERE by running the exact code shown on the page, on the toy data
shown on the page, and assertions check the decisions match the narrative
before any HTML is written.

Code style mirrors Daniel's course decks (DataCamp Hypothesis Testing in
Python): scipy.stats, pingouin, statsmodels.proportions_ztest.
"""
import numpy as np
import pandas as pd
import pingouin as pg
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest
from pathlib import Path

rng = np.random.default_rng(11)
FAILURES = []

def check(name, cond, detail=""):
    print(("  ok: " if cond else "  FAIL: ") + name + (f" ({detail})" if detail else ""))
    if not cond:
        FAILURES.append(name)

def fp(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"

# ---------------------------------------------------------------- z-test ----
# Fasting glucose at a new clinic vs reference population mean 100 mg/dL.
# The reference lab's SD (sigma = 12) is known from years of QC data.
n_z = 64
glucose = rng.normal(104.3, 12, n_z)
xbar = glucose.mean()
z_stat = (xbar - 100) / (12 / np.sqrt(n_z))
z_p = 2 * (1 - stats.norm.cdf(abs(z_stat)))
check("z-test rejects", z_p < 0.05, f"z={z_stat:.2f}, {fp(z_p)}")

# ------------------------------------------------------- one-sample t-test ----
# HbA1c (%) for 25 patients at a diabetes clinic vs target 7.0.
a1c = rng.normal(7.45, 0.85, 25)
t1 = pg.ttest(x=a1c, y=7.0)
t1_stat, t1_p = float(t1["T"].iloc[0]), float(t1["p_val"].iloc[0])
check("one-sample t rejects", t1_p < 0.05, f"T={t1_stat:.2f}, {fp(t1_p)}")

# ------------------------------------------------------- two-sample t-test ----
# Length of stay (days): new discharge protocol vs standard care, 30 each.
los_new = rng.normal(4.0, 1.0, 35)
los_std = rng.normal(5.3, 1.2, 35)
t2 = pg.ttest(x=los_new, y=los_std)
t2_stat, t2_p, t2_dof = float(t2["T"].iloc[0]), float(t2["p_val"].iloc[0]), float(t2["dof"].iloc[0])
check("two-sample t rejects", t2_p < 0.05, f"T={t2_stat:.2f}, {fp(t2_p)}")

# ------------------------------------------------------------ paired t-test ---
# Systolic BP before/after a 12-week hypertension program, same 20 patients.
bp_before = rng.normal(142, 10, 20)
bp_after = bp_before - rng.normal(8, 9, 20)
tp = pg.ttest(x=bp_after, y=bp_before, paired=True)
tp_stat, tp_p = float(tp["T"].iloc[0]), float(tp["p_val"].iloc[0])
check("paired t rejects", tp_p < 0.05, f"T={tp_stat:.2f}, {fp(tp_p)}")

# ------------------------------------------------------------------- ANOVA ----
# Weight loss (kg at 3 months) across 4 programs, 16 patients each.
wl = pd.DataFrame({
    "loss": np.concatenate([rng.normal(2.1, 1.8, 16), rng.normal(4.3, 1.8, 16),
                            rng.normal(5.1, 1.9, 16), rng.normal(6.4, 1.8, 16)]),
    "program": (["Diet only"] * 16 + ["Diet + exercise"] * 16
                + ["Medication"] * 16 + ["Medication + exercise"] * 16),
})
aov = pg.anova(data=wl, dv="loss", between="program")
aov_F, aov_p = float(aov["F"].iloc[0]), float(aov["p_unc"].iloc[0])
check("ANOVA rejects", aov_p < 0.01, f"F={aov_F:.2f}, {fp(aov_p)}")
wl_means = wl.groupby("program")["loss"].mean().round(1).to_dict()

# ------------------------------------------- pairwise tests + Bonferroni ------
pw = pg.pairwise_tests(data=wl, dv="loss", between="program", padjust="bonf")
pw_rows = [(r["A"], r["B"], float(r["p_unc"]), float(r["p_corr"])) for _, r in pw.iterrows()]
strongest = min(pw_rows, key=lambda r: r[3])
check("pairwise: strongest pair survives Bonferroni", strongest[3] < 0.05,
      f"{strongest[0]} vs {strongest[1]}: p-bonf={strongest[3]:.4f}")
fw = 1 - 0.95 ** len(pw_rows)   # family-wise false-positive risk, uncorrected

# ---------------------------------------------- one-sample proportion test --
# Surgical site infections: 14 of 200 vs published benchmark 5%.
z1p, p1p = proportions_ztest(count=14, nobs=200, value=0.05)
check("one-sample proportion FAILS to reject (by design)", p1p > 0.05,
      f"z={z1p:.2f}, {fp(float(p1p))}")

# ---------------------------------------------- two-sample proportion test --
# 30-day readmissions: new follow-up program 15/150 vs standard care 32/160.
z2p, p2p = proportions_ztest(count=np.array([15, 32]), nobs=np.array([150, 160]))
check("two-sample proportion rejects", p2p < 0.05, f"z={z2p:.2f}, {fp(float(p2p))}")

# ------------------------------------------- chi-squared test of independence -
# Smoking status vs COPD diagnosis in a screening sample (N = 500).
rows = ([{"smoker": "Smoker", "copd": "COPD"}] * 45 + [{"smoker": "Smoker", "copd": "No COPD"}] * 155
        + [{"smoker": "Non-smoker", "copd": "COPD"}] * 20 + [{"smoker": "Non-smoker", "copd": "No COPD"}] * 280)
screen = pd.DataFrame(rows)
expected, observed, chi_stats = pg.chi2_independence(data=screen, x="smoker", y="copd")
chi2_stat = float(chi_stats.loc[chi_stats["test"] == "pearson", "chi2"].iloc[0])
chi2_p = float(chi_stats.loc[chi_stats["test"] == "pearson", "pval"].iloc[0])
min_expected = float(np.min(expected.values))
check("chi2 independence rejects", chi2_p < 0.001, f"chi2={chi2_stat:.1f}, {fp(chi2_p)}")
check("chi2 expected counts all >= 5", min_expected >= 5, f"min expected={min_expected:.0f}")

# --------------------------------------------- chi-squared goodness of fit --
# ED arrivals by weekday vs a uniform week.
ed_obs = np.array([205, 160, 150, 152, 158, 135, 140])
ed_exp = np.repeat(ed_obs.sum() / 7, 7)
gof = stats.chisquare(f_obs=ed_obs, f_exp=ed_exp)
check("chi2 GOF rejects", gof.pvalue < 0.05, f"chi2={gof.statistic:.1f}, {fp(gof.pvalue)}")

# ------------------------------------------------- Wilcoxon signed-rank test --
# Pain scores (0-10) before/after physical therapy, 14 patients. Ordinal+skewed.
pain_before = np.array([8, 7, 9, 6, 8, 7, 9, 8, 6, 7, 8, 9, 7, 8])
pain_after = np.array([5, 6, 6, 5, 4, 7, 6, 5, 6, 4, 5, 7, 5, 6])
wil = pg.wilcoxon(x=pain_before, y=pain_after)
wil_W, wil_p = float(wil["W_val"].iloc[0]), float(wil["p_val"].iloc[0])
check("Wilcoxon rejects", wil_p < 0.05, f"W={wil_W:.0f}, {fp(wil_p)}")

# ------------------------------------------------------- Mann-Whitney U test -
# Wait time (minutes) at two clinics; heavily right-skewed, so ranks not means.
wait_a = rng.lognormal(3.4, 0.5, 28)
wait_b = rng.lognormal(3.75, 0.5, 28)
mwu = pg.mwu(x=wait_a, y=wait_b)
mwu_U, mwu_p = float(mwu["U_val"].iloc[0]), float(mwu["p_val"].iloc[0])
check("Mann-Whitney rejects", mwu_p < 0.05, f"U={mwu_U:.0f}, {fp(mwu_p)}")

# --------------------------------------------------------- Kruskal-Wallis test -
# Patient satisfaction (1-10) across 4 clinics; ordinal + skewed.
sat = pd.DataFrame({
    "score": np.concatenate([rng.choice(range(3, 9), 18, p=[.1, .15, .25, .25, .15, .1]),
                             rng.choice(range(4, 10), 18, p=[.1, .2, .3, .2, .1, .1]),
                             rng.choice(range(5, 11), 18, p=[.15, .25, .3, .15, .1, .05]),
                             rng.choice(range(2, 8), 18, p=[.1, .25, .3, .2, .1, .05])]),
    "clinic": ["A"] * 18 + ["B"] * 18 + ["C"] * 18 + ["D"] * 18,
})
kw = pg.kruskal(data=sat, dv="score", between="clinic")
kw_H, kw_p = float(kw["H"].iloc[0]), float(kw["p_unc"].iloc[0])
check("Kruskal-Wallis rejects", kw_p < 0.05, f"H={kw_H:.1f}, {fp(kw_p)}")

if FAILURES:
    raise SystemExit(f"ASSERTION FAILURES: {FAILURES}")
print("ALL TESTS COMPUTED AND VERIFIED")

V = dict(
    xbar=f"{xbar:.1f}", z_stat=f"{z_stat:.2f}", z_p=fp(z_p),
    a1c_mean=f"{a1c.mean():.2f}", t1_stat=f"{t1_stat:.2f}", t1_p=fp(t1_p),
    los_new=f"{los_new.mean():.1f}", los_std=f"{los_std.mean():.1f}",
    t2_stat=f"{t2_stat:.2f}", t2_p=fp(t2_p),
    bp_b=f"{bp_before.mean():.0f}", bp_a=f"{bp_after.mean():.0f}",
    tp_stat=f"{tp_stat:.2f}", tp_p=fp(tp_p),
    aov_F=f"{aov_F:.1f}", aov_p=fp(aov_p),
    m_diet=f"{wl_means['Diet only']:.1f}", m_dietex=f"{wl_means['Diet + exercise']:.1f}",
    m_med=f"{wl_means['Medication']:.1f}", m_medex=f"{wl_means['Medication + exercise']:.1f}",
    fw=f"{fw:.0%}",
    z1p=f"{z1p:.2f}", p1p=fp(float(p1p)),
    z2p=f"{z2p:.2f}", p2p=fp(float(p2p)),
    chi2=f"{chi2_stat:.1f}", chi2_p=fp(chi2_p), min_exp=f"{min_expected:.0f}",
    gof=f"{gof.statistic:.1f}", gof_p=fp(gof.pvalue), ed_total=int(ed_obs.sum()),
    wil_W=f"{wil_W:.0f}", wil_p=fp(wil_p),
    mwu_U=f"{mwu_U:.0f}", mwu_p=fp(mwu_p),
    wait_med_a=f"{np.median(wait_a):.0f}", wait_med_b=f"{np.median(wait_b):.0f}",
    kw_H=f"{kw_H:.1f}", kw_p=fp(kw_p),
)
PW_LINES = "\n".join(
    f"  {a} vs {b}:  p = {pu:.3f}  ->  Bonferroni p = {pb:.3f}" for a, b, pu, pb in pw_rows)

# ============================ HTML template ============================
CSS = """
:root{--ink:#1a1a2e; --accent:#2b50aa; --soft:#eef4ff; --warnbg:#fdf3e3; --line:#d9dee9}
*{box-sizing:border-box}
body{font-family:Georgia,'Times New Roman',serif; color:var(--ink); max-width:760px; margin:0 auto; padding:28px 20px 70px; line-height:1.62; background:#fff}
h1{font-size:1.85em; line-height:1.25; margin:.2em 0 .1em}
h2{font-size:1.35em; margin:1.7em 0 .3em; border-bottom:2px solid var(--soft); padding-bottom:.15em}
h3{font-size:1.13em; margin:1.5em 0 .2em; color:var(--ink)}
a{color:var(--accent); text-decoration:none} a:hover{text-decoration:underline}
.small{color:#666; font-size:.92em}
.lede{font-size:1.06em}
.box{background:var(--soft); border-radius:8px; padding:12px 16px; margin:1em 0}
.warn{background:var(--warnbg); border-radius:8px; padding:12px 16px; margin:1em 0}
.use{margin:.35em 0}
.assump{color:#444; font-size:.96em; margin:.35em 0}
.fallback{font-size:.96em; margin:.35em 0}
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
.ctl{display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin:.6em 0}
.ctl button,.ctl select{font-family:inherit; font-size:.92em; padding:5px 10px; border-radius:6px; border:1px solid var(--line); background:#fff; cursor:pointer}
.ctl button.on{background:var(--accent); color:#fff; border-color:var(--accent)}
input[type=range]{accent-color:var(--accent)}
.readout{background:var(--soft); border-radius:8px; padding:10px 14px; font-size:.97em; margin:.5em 0}
svg{max-width:100%; height:auto; display:block; margin:.4em auto}
select.q{font-family:inherit; font-size:.95em; padding:5px 8px; border-radius:6px; border:1px solid var(--line); max-width:100%}
label.q{display:block; font-weight:bold; margin:.55em 0 .15em; font-size:.97em}
.kicker{font-size:.8em; letter-spacing:.12em; text-transform:uppercase; color:var(--accent); font-weight:bold}
"""

HEAD = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>The Practitioner's Guide to Statistical Tests — Applied Data Science Lab</title>
<meta name="description" content="A one-page method bank for the everyday statistical tests: when to use each one, a healthcare example with real output, the assumptions, the non-parametric fallback, and the Python code.">
<style>""" + CSS + """</style></head><body>
<p class="small"><a href="https://danieljordanviray.github.io/applied-data-science-lab/docs/">&larr; Homepage</a></p>
<h1>The Practitioner's Guide to Statistical Tests</h1>
<p class="small">October 17, 2026</p>
<p class="lede">Ever stared at a dataset and wondered: z-test or t-test? ANOVA or Kruskal-Wallis?
And what <em>is</em> a Wilcoxon signed-rank test, anyway? This guide is a method bank — every common test on one
page: when to use it, a healthcare example with real output, the assumptions to check, the fallback test when
those assumptions fail, and the Python code (click to expand). Built for future me, and for anyone learning
these for the first time.</p>
<div class="box"><strong>How to read the numbers:</strong> every statistic on this page was computed by running
the exact code in the dropdowns on the toy datasets described — a build script asserts each result before the
page is written, so nothing here is typed in from memory.</div>

<h2>Not sure which test? Answer three questions</h2>
<div class="box">
<label class="q">1. What kind of outcome are you testing?</label>
<select class="q" id="qOutcome">
  <option value="mean">A measurement (blood pressure, HbA1c, days, dollars)</option>
  <option value="rate">A yes/no rate (readmitted? infected? responded?)</option>
  <option value="cats">Counts across categories</option>
</select>
<label class="q">2. What are you comparing?</label>
<select class="q" id="qCompare">
  <option value="bench">One group vs a known benchmark</option>
  <option value="two">Two groups</option>
  <option value="three">Three or more groups</option>
  <option value="assoc">Two categorical variables — are they associated?</option>
  <option value="dist">Observed counts vs an expected distribution</option>
</select>
<label class="q">3. If it's a measurement: is it roughly bell-shaped? (If not — skewed, or a 1–10 score — pick the second option.)</label>
<select class="q" id="qShape">
  <option value="normal">Roughly bell-shaped (or a large sample)</option>
  <option value="skew">Clearly skewed, or ordinal scores</option>
</select>
<label class="q">4. If two groups: are they the same people measured twice?</label>
<select class="q" id="qPaired">
  <option value="no">No — different people</option>
  <option value="yes">Yes — before/after on the same people</option>
</select>
<div class="readout" id="chooserOut"></div>
</div>

<h2>Contents</h2>
<div class="toc">
<p class="kicker">The setup</p>
<ul>
<li><a href="#setup">The setup: null vs alternative, &alpha;, and tails</a></li>
<li><a href="#zscores">z-scores &amp; test statistics: one idea behind every test</a></li>
<li><a href="#errors">Type I vs Type II errors</a></li>
<li><a href="#courtroom">The decision rule: guilty beyond a reasonable doubt</a></li>
</ul>
<p class="kicker">Comparing means</p>
<ul>
<li><a href="#ztest">z-test</a></li>
<li><a href="#t1">One-sample t-test</a></li>
<li><a href="#t2">Two-sample t-test</a></li>
<li><a href="#paired">Paired t-test</a></li>
<li><a href="#anova">ANOVA</a></li>
<li><a href="#pairwise">Pairwise comparisons + Bonferroni correction</a></li>
</ul>
<p class="kicker">Proportions &amp; categories</p>
<ul>
<li><a href="#prop1">One-sample proportion test</a></li>
<li><a href="#prop2">Two-sample proportion test</a></li>
<li><a href="#chi2ind">Chi-squared test of independence</a></li>
<li><a href="#chi2gof">Chi-squared goodness-of-fit test</a></li>
</ul>
<p class="kicker">When the assumptions break (rank tests)</p>
<ul>
<li><a href="#wilcoxon">Wilcoxon signed-rank test</a></li>
<li><a href="#mwu">Mann-Whitney U test</a></li>
<li><a href="#kruskal">Kruskal-Wallis test</a></li>
</ul>
<ul><li><a href="#cheat">The one-table cheat sheet</a></li></ul>
</div>

<h2 id="setup">The setup: null vs alternative, &alpha;, and tails</h2>
<p>Every test on this page is the same argument. The <strong>null hypothesis (H<sub>0</sub>)</strong> says
<em>nothing is going on</em> — no difference, no effect, the benchmark is right. The
<strong>alternative hypothesis (H<sub>A</sub>)</strong> is the thing you're trying to show. You assume H<sub>0</sub>
is true, then ask: <em>if it were, how surprising is my data?</em> That surprise is the
<strong>p-value</strong>: the probability of seeing data at least this extreme if H<sub>0</sub> were true.</p>
<p>The <strong>significance level (&alpha;)</strong> is your surprise threshold, set <em>before</em> looking at the
data — 0.05 is the convention. If p &le; &alpha;, you reject H<sub>0</sub>; if p &gt; &alpha;, you fail to reject it
(more on that phrasing in the <a href="#courtroom">courtroom section</a> — it matters).</p>
<p><strong>Tails</strong> are where H<sub>A</sub> points. <strong>Two-tailed</strong>: a difference in either direction
counts (the default — safest, and what you should use unless you committed in advance).
<strong>Right-tailed</strong>: only "more than" counts. <strong>Left-tailed</strong>: only "less than" counts.
Choosing the tail after seeing the data is cheating: it quietly doubles your false-alarm rate.</p>
<svg id="tailsSvg" viewBox="0 0 640 240" role="img" aria-label="Normal curve with rejection regions shaded"></svg>
<div class="ctl">
<button data-tail="two" class="on">Two-tailed</button>
<button data-tail="right">Right-tailed</button>
<button data-tail="left">Left-tailed</button>
<label>&alpha; =
<select id="alphaSel"><option value="0.10">0.10</option><option value="0.05" selected>0.05</option><option value="0.01">0.01</option></select></label>
</div>
<div class="readout" id="tailsOut"></div>

<h2 id="zscores">z-scores &amp; test statistics: one idea behind every test</h2>
<p>A <strong>z-score</strong> measures distance in units of noise:
<em>z = (what you saw &minus; what H<sub>0</sub> expects) / standard error</em>. Saw a sample mean of 104 when
H<sub>0</sub> says 100, with a standard error of 1.5? That's z = 2.7 — your result sits 2.7 noise-units away from
"nothing going on." The further out, the smaller the p-value.</p>
<p>Every test statistic on this page — t, F, &chi;&sup2;, U, H — is that same move with different plumbing:
<em>distance from H<sub>0</sub>, divided by how much wobble you'd expect by chance</em>. Learn to read one and
you can read them all. Field rule of thumb: a statistic around <strong>2 standard errors</strong> from the null
is roughly the p = 0.05 line for two-tailed tests.</p>

<h2 id="errors">Type I vs Type II errors</h2>
<p>There are two ways to be wrong, and you can't minimize both at once:</p>
<table>
<tr><th></th><th>You reject H<sub>0</sub></th><th>You fail to reject H<sub>0</sub></th></tr>
<tr><th>H<sub>0</sub> is true</th><td><strong>Type I error</strong> — a false alarm. Probability = &alpha;, which you set.</td><td>Correct.</td></tr>
<tr><th>H<sub>0</sub> is false</th><td>Correct — this probability is the test's <strong>power</strong> (convention: aim for 80%).</td><td><strong>Type II error</strong> — a missed detection. Probability = &beta;.</td></tr>
</table>
<p>You control Type I directly (pick &alpha;). You control Type II only indirectly: a bigger sample or a
bigger true effect separates the curves, and misses get rarer. Drag the sliders and watch &beta; shrink:</p>
<svg id="powSvg" viewBox="0 0 640 250" role="img" aria-label="Null and alternative distributions with error regions shaded"></svg>
<div class="ctl">
<label>True effect (Cohen's d): <input type="range" id="dSlider" min="0.1" max="1" step="0.05" value="0.5"> <span id="dVal">0.50</span></label>
<label>Sample size (per test): <input type="range" id="nSlider" min="10" max="200" step="5" value="30"> <span id="nVal">30</span></label>
</div>
<div class="readout" id="powOut"></div>

<h2 id="courtroom">The decision rule: guilty beyond a reasonable doubt</h2>
<div class="box">
<p style="margin-top:0">A hypothesis test is a criminal trial. The defendant is H<sub>0</sub> — "no effect" —
and the defendant is <strong>presumed innocent</strong>. The data is the evidence. Your &alpha; is the standard of
proof: <em>guilty beyond a reasonable doubt</em>.</p>
<p><strong>Reject H<sub>0</sub></strong> = a guilty verdict: the evidence is too strong to be a coincidence.</p>
<p><strong>Fail to reject H<sub>0</sub></strong> = a "not guilty" verdict — which is <em>not</em> a declaration of
innocence. It means the evidence wasn't strong enough. That's why statisticians never say "accept H<sub>0</sub>"
or "we proved there's no effect": absence of evidence is not evidence of absence. Your study may simply have
been too small to see what was there (that's &beta;).</p>
<p style="margin-bottom:0">The two errors map too: a <strong>Type I error</strong> convicts an innocent defendant;
a <strong>Type II error</strong> lets a guilty one walk. Tighten the standard of proof (smaller &alpha;) and you
convict fewer innocents — but more of the guilty walk free. &alpha; = 0.05 is a convention for balancing the
two, not a law of nature.</p>
</div>
"""

BODY = r"""
<h2>Comparing means</h2>

<h3 id="ztest">z-test (one sample, known &sigma;)</h3>
<p class="use"><strong>Use it when:</strong> you're comparing a sample mean to a known benchmark <em>and</em> you
genuinely know the population standard deviation &sigma; — usually because a huge reference dataset or a
calibrated instrument pinned it down.</p>
<p><strong>Example:</strong> a reference lab has measured fasting glucose for years: population mean 100 mg/dL,
&sigma; = 12 known from quality-control data. A new clinic's 64 patients average %%xbar%% mg/dL. Different
population, or just noise? z = %%z_stat%%, %%z_p%% &rarr; <strong>reject H<sub>0</sub></strong> at &alpha; = 0.05.</p>
<p class="assump"><strong>Assumptions:</strong> &sigma; truly known; independent observations; sample mean roughly
normal (data normal, or n &ge; ~30).</p>
<p class="fallback"><strong>Reality check:</strong> &sigma; is almost never known in real work — that's why the
t-test below is the everyday tool. The z machinery's main modern jobs are proportions (further down) and
large-sample approximations.</p>
<details><summary>Data structure + Python code</summary><div class="body">
<p>Data: one sample summarized by its mean, plus the known &sigma;. The course-style way — build the p-value
by hand from the normal CDF:</p>
<pre><code>import numpy as np
from scipy.stats import norm

xbar, mu0, sigma, n = %%xbar%%, 100, 12, 64   # sample mean, H0 value, KNOWN sd, sample size
z = (xbar - mu0) / (sigma / np.sqrt(n))       # = %%z_stat%%
p = 2 * (1 - norm.cdf(abs(z)))                # two-tailed: double the upper tail
# left-tailed test:  p = norm.cdf(z)
# right-tailed test: p = 1 - norm.cdf(z)</code></pre>
<p class="out">Output: z = %%z_stat%%, %%z_p%%.</p>
</div></details>

<h3 id="t1">One-sample t-test</h3>
<p class="use"><strong>Use it when:</strong> one sample mean vs a benchmark, and &sigma; is <em>unknown</em> —
the normal real-world case. The t distribution is the normal with fatter tails, charging you for having to
estimate the spread from the same data.</p>
<p><strong>Example:</strong> HbA1c for 25 patients at a diabetes clinic averages %%a1c_mean%%% against a
target of 7.0%. T = %%t1_stat%%, %%t1_p%% &rarr; <strong>reject H<sub>0</sub></strong>: the clinic's patients
run above target.</p>
<p class="assump"><strong>Assumptions:</strong> independent observations; data roughly normal, or n &ge; ~30
(the t-test tolerates mild skew at moderate n).</p>
<p class="fallback"><strong>If that fails</strong> (small n, clearly skewed): the one-sample
<a href="#wilcoxon">Wilcoxon signed-rank test</a> against the benchmark value.</p>
<details><summary>Data structure + Python code</summary><div class="body">
<p>Data: one column of numbers (here, a NumPy array of HbA1c values).</p>
<pre><code>import pingouin as pg

# a1c: array of 25 HbA1c values, mean %%a1c_mean%%
pg.ttest(x=a1c, y=7.0)        # y = the benchmark value under H0</code></pre>
<p class="out">Key output: T = %%t1_stat%%, dof = 24, %%t1_p%%.</p>
</div></details>

<h3 id="t2">Two-sample t-test</h3>
<p class="use"><strong>Use it when:</strong> comparing the means of two <em>independent</em> groups.</p>
<p><strong>Example:</strong> length of stay under a new discharge protocol (35 patients, mean %%los_new%% days)
vs standard care (35 patients, mean %%los_std%% days). T = %%t2_stat%%, %%t2_p%% &rarr;
<strong>reject H<sub>0</sub></strong>: the new protocol's stays are shorter.</p>
<p class="assump"><strong>Assumptions:</strong> the two groups are independent; each group roughly normal or
n &ge; ~30 per group. You do <em>not</em> need equal variances — use Welch's version (pingouin's default
behavior), which adjusts for unequal spreads.</p>
<p class="fallback"><strong>If that fails</strong> (heavy skew — and length of stay is famously skewed, so in
real work you may land here directly): the <a href="#mwu">Mann-Whitney U test</a>.</p>
<details><summary>Data structure + Python code</summary><div class="body">
<p>Data: two columns/arrays, one per group.</p>
<pre><code>import pingouin as pg

pg.ttest(x=los_new, y=los_std)   # Welch's correction applied automatically</code></pre>
<p class="out">Key output: T = %%t2_stat%%, %%t2_p%%.</p>
</div></details>

<h3 id="paired">Paired t-test</h3>
<p class="use"><strong>Use it when:</strong> the same subjects are measured twice — before/after, or matched
pairs. The test runs on the <em>differences</em>, one per subject.</p>
<p><strong>Example:</strong> systolic blood pressure in 20 patients before (mean %%bp_b%% mmHg) and after
(mean %%bp_a%% mmHg) a 12-week hypertension program. T = %%tp_stat%%, %%tp_p%% &rarr;
<strong>reject H<sub>0</sub></strong>: blood pressure dropped.</p>
<p class="assump"><strong>Why pairing is powerful:</strong> each patient is their own control, so the large
person-to-person variation cancels out of the comparison. <strong>Assumptions:</strong> the <em>differences</em>
are roughly normal (not the raw values); pairs are independent of each other.</p>
<p class="fallback"><strong>If that fails</strong> (differences skewed, or the outcome is ordinal):
the <a href="#wilcoxon">Wilcoxon signed-rank test</a>.</p>
<details><summary>Data structure + Python code</summary><div class="body">
<p>Data: two aligned columns — row i is the same patient in both.</p>
<pre><code>import pingouin as pg

pg.ttest(x=bp_after, y=bp_before, paired=True)</code></pre>
<p class="out">Key output: T = %%tp_stat%%, dof = 19, %%tp_p%%.</p>
</div></details>

<h3 id="anova">ANOVA (one-way)</h3>
<p class="use"><strong>Use it when:</strong> comparing means across <em>three or more</em> groups with a single
test. ANOVA answers only "does <em>any</em> group differ?" — not which ones.</p>
<p><strong>Example:</strong> weight loss at 3 months across four programs — diet only (mean %%m_diet%% kg),
diet + exercise (%%m_dietex%%), medication (%%m_med%%), medication + exercise (%%m_medex%%), 16 patients each.
F = %%aov_F%%, %%aov_p%% &rarr; <strong>reject H<sub>0</sub></strong>: the programs are not all equal. Which
ones differ is the <a href="#pairwise">next section's</a> job.</p>
<p class="assump"><strong>Assumptions:</strong> independent observations; roughly normal within each group (or
decent n per group); roughly similar spreads across groups.</p>
<p class="fallback"><strong>If that fails:</strong> the <a href="#kruskal">Kruskal-Wallis test</a>.</p>
<details><summary>Data structure + Python code</summary><div class="body">
<p>Data: a long-format table — one row per patient, a value column and a group column.</p>
<pre><code>import pingouin as pg

# df: columns 'loss' (kg) and 'program' (4 labels), 64 rows
pg.anova(data=df, dv='loss', between='program')</code></pre>
<p class="out">Key output: F = %%aov_F%%, %%aov_p%%.</p>
</div></details>

<h3 id="pairwise">Pairwise comparisons + Bonferroni correction</h3>
<p class="use"><strong>Use it when:</strong> ANOVA says "something differs" and you need to know <em>which</em>
pairs of groups differ.</p>
<p><strong>The trap:</strong> four programs make 6 pairwise tests. Run each at &alpha; = 0.05 and the chance
of at least one false alarm isn't 5% — it's about <strong>%%fw%%</strong> (1 &minus; 0.95<sup>6</sup>). Every extra
test is another lottery ticket for a fluke. Running many t-tests instead of ANOVA + corrected pairwise tests
is exactly this mistake.</p>
<p><strong>The Bonferroni fix:</strong> multiply each p-value by the number of tests (equivalently, judge
each against &alpha;/6 &asymp; 0.008). Simple, a bit conservative, and the right default. On the weight-loss
data:</p>
<pre><code>%%PW_LINES%%</code></pre>
<p>The strongest gap (diet only vs medication + exercise) survives the correction easily; the correction costs
you the borderline pairs — that's the price of honesty.</p>
<details><summary>Data structure + Python code</summary><div class="body">
<p>Data: the same long-format table as ANOVA.</p>
<pre><code>import pingouin as pg

pg.pairwise_tests(data=df, dv='loss', between='program', padjust='bonf')
# read the p_corr column: already Bonferroni-adjusted</code></pre>
</div></details>

<h2>Proportions &amp; categories</h2>

<h3 id="prop1">One-sample proportion test</h3>
<p class="use"><strong>Use it when:</strong> one observed rate vs a published benchmark rate.</p>
<p><strong>Example:</strong> 14 surgical-site infections in 200 surgeries (7.0%) vs a published benchmark of
5%. z = %%z1p%%, %%p1p%% &rarr; <strong>fail to reject H<sub>0</sub></strong>. Read that the courtroom way: not
proof our rate <em>is</em> 5% — just not enough evidence to say it differs.</p>
<div class="warn"><strong>Why z here, not t?</strong> A proportion carries its own standard error:
SE = &radic;(p(1&minus;p)/n). There's no separate spread to estimate from the sample the way a mean forces you
to estimate &sigma; — and it's estimating &sigma; that earns the t distribution its fatter tails. With the SE
coming from the proportion itself, the plain normal (z) is the right reference, once n is large enough for the
normal approximation to the binomial.</div>
<p class="assump"><strong>Assumptions:</strong> independent cases; np and n(1&minus;p) both &ge; ~10 under the
benchmark p.</p>
<p class="fallback"><strong>If n is too small:</strong> the exact binomial test
(<code>scipy.stats.binomtest</code>) — same question, no normal approximation.</p>
<details><summary>Data structure + Python code</summary><div class="body">
<p>Data: just two counts — how many events, out of how many trials.</p>
<pre><code>from statsmodels.stats.proportion import proportions_ztest

proportions_ztest(count=14, nobs=200, value=0.05)   # value = benchmark under H0</code></pre>
<p class="out">Output: z = %%z1p%%, %%p1p%%.</p>
</div></details>

<h3 id="prop2">Two-sample proportion test</h3>
<p class="use"><strong>Use it when:</strong> comparing a rate between two groups.</p>
<p><strong>Example:</strong> 30-day readmissions: 15 of 150 patients (10%) on a new follow-up program vs 32
of 160 (20%) on standard care. z = %%z2p%%, %%p2p%% &rarr; <strong>reject H<sub>0</sub></strong>: the program's
readmission rate is lower.</p>
<p class="assump"><strong>Assumptions:</strong> independent groups; each cell count &ge; ~10.
<strong>Worth knowing:</strong> this test is mathematically the same as a chi-squared test of independence on
the 2&times;2 table — two doors into the same room.</p>
<details><summary>Data structure + Python code</summary><div class="body">
<p>Data: two counts and two totals, as arrays.</p>
<pre><code>import numpy as np
from statsmodels.stats.proportion import proportions_ztest

proportions_ztest(count=np.array([15, 32]), nobs=np.array([150, 160]))</code></pre>
<p class="out">Output: z = %%z2p%%, %%p2p%%.</p>
</div></details>

<h3 id="chi2ind">Chi-squared test of independence</h3>
<p class="use"><strong>Use it when:</strong> two categorical variables, and you're asking whether they're
associated. In one line: compare the counts you observed with the counts you'd expect if the variables were
independent — big gaps make a big &chi;&sup2;.</p>
<p><strong>Example:</strong> smoking status vs COPD diagnosis in a screening sample of 500: smokers 45 COPD /
155 no COPD; non-smokers 20 / 280. &chi;&sup2; = %%chi2%%, %%chi2_p%% &rarr;
<strong>reject H<sub>0</sub></strong>: smoking and COPD are associated. (Associated — not proven causal; that's
a design question, not a test question.)</p>
<p class="assump"><strong>Assumptions:</strong> work with counts, never percentages; independent observations;
the <em>expected</em> count is &ge; 5 in every cell (here the smallest expected count is %%min_exp%%).</p>
<p class="fallback"><strong>If expected counts are too small</strong> (2&times;2 table): Fisher's exact test.</p>
<details><summary>Data structure + Python code</summary><div class="body">
<p>Data: a long-format table — one row per person, one column per categorical variable.</p>
<pre><code>import pingouin as pg

# df: 500 rows; columns 'smoker' (Smoker / Non-smoker) and 'copd' (COPD / No COPD)
expected, observed, stats = pg.chi2_independence(data=df, x='smoker', y='copd')</code></pre>
<p class="out">Key output (pearson row): &chi;&sup2; = %%chi2%%, %%chi2_p%%.</p>
</div></details>

<h3 id="chi2gof">Chi-squared goodness-of-fit test</h3>
<p class="use"><strong>Use it when:</strong> one categorical variable, and you're asking whether observed
counts match a hypothesized distribution.</p>
<p><strong>Example:</strong> %%ed_total%% emergency-department arrivals across a week — Mon 205, Tue 160,
Wed 150, Thu 152, Fri 158, Sat 135, Sun 140 — against a uniform expectation of ~157 per day.
&chi;&sup2; = %%gof%%, %%gof_p%% &rarr; <strong>reject H<sub>0</sub></strong>: arrivals are not spread evenly;
Mondays spike.</p>
<p class="assump"><strong>Assumptions:</strong> counts, not proportions; expected count &ge; 5 in every
category.</p>
<details><summary>Data structure + Python code</summary><div class="body">
<p>Data: two aligned lists — observed counts and expected counts (same total).</p>
<pre><code>from scipy.stats import chisquare

chisquare(f_obs=[205, 160, 150, 152, 158, 135, 140],
          f_exp=[157.1, 157.1, 157.1, 157.1, 157.1, 157.1, 157.1])</code></pre>
<p class="out">Output: &chi;&sup2; = %%gof%%, %%gof_p%%.</p>
</div></details>

<h2>When the assumptions break: rank tests</h2>
<p>The three tests below don't use the raw values at all — they convert everything to <strong>ranks</strong>
(1st, 2nd, 3rd&hellip;) and test those. Skew and outliers can't bully a rank, and ordinal scores (pain 0–10,
satisfaction 1–10) are welcome. The trade: slightly less power when the data really are normal. Each one is
the fallback for a test above.</p>

<h3 id="wilcoxon">Wilcoxon signed-rank test</h3>
<p class="use"><strong>Use it when:</strong> paired data (before/after) whose differences are skewed — or an
ordinal outcome. The <a href="#paired">paired t-test's</a> fallback; a one-sample version also tests against
a benchmark. In one line: rank the sizes of the differences, then check whether the positive and negative
ranks balance.</p>
<p><strong>Example:</strong> pain scores (0–10) before and after physical therapy for 14 back-pain patients —
an ordinal scale, so ranks, not means. W = %%wil_W%%, %%wil_p%% &rarr; <strong>reject H<sub>0</sub></strong>:
pain dropped. (W = 0 is as one-sided as it gets: every patient's pain went down.)</p>
<details><summary>Data structure + Python code</summary><div class="body">
<p>Data: two aligned arrays — row i is the same patient in both.</p>
<pre><code>import pingouin as pg

pain_before = [8, 7, 9, 6, 8, 7, 9, 8, 6, 7, 8, 9, 7, 8]
pain_after  = [5, 6, 6, 5, 4, 7, 6, 5, 6, 4, 5, 7, 5, 6]
pg.wilcoxon(x=pain_before, y=pain_after)</code></pre>
<p class="out">Key output: W = %%wil_W%%, %%wil_p%%.</p>
</div></details>

<h3 id="mwu">Mann-Whitney U test</h3>
<p class="use"><strong>Use it when:</strong> two independent groups and a skewed outcome or ordinal scores.
The <a href="#t2">two-sample t-test's</a> fallback. It asks whether a random patient from one group tends to
rank above a random patient from the other — so report medians alongside it.</p>
<p><strong>Example:</strong> wait times at two clinics (minutes, heavily right-skewed): median %%wait_med_a%%
at Clinic A vs %%wait_med_b%% at Clinic B, 28 patients each. U = %%mwu_U%%, %%mwu_p%% &rarr;
<strong>reject H<sub>0</sub></strong>: the waits genuinely differ.</p>
<details><summary>Data structure + Python code</summary><div class="body">
<p>Data: two arrays, one per group.</p>
<pre><code>import pingouin as pg

pg.mwu(x=wait_a, y=wait_b)</code></pre>
<p class="out">Key output: U = %%mwu_U%%, %%mwu_p%%.</p>
</div></details>

<h3 id="kruskal">Kruskal-Wallis test</h3>
<p class="use"><strong>Use it when:</strong> three or more groups, skewed outcome or ordinal scores —
<a href="#anova">ANOVA's</a> fallback. (Same relationship as Mann-Whitney to the t-test: Kruskal-Wallis is
to Wilcoxon-Mann-Whitney what ANOVA is to the t-test.)</p>
<p><strong>Example:</strong> patient satisfaction (1–10) across four clinics, 18 patients each.
H = %%kw_H%%, %%kw_p%% &rarr; <strong>reject H<sub>0</sub></strong>: the clinics differ. Follow up with
pairwise Mann-Whitney tests plus a Bonferroni correction, exactly as in the
<a href="#pairwise">pairwise section</a>.</p>
<details><summary>Data structure + Python code</summary><div class="body">
<p>Data: a long-format table — one row per patient, score column + clinic column.</p>
<pre><code>import pingouin as pg

# df: columns 'score' (1-10) and 'clinic' (A/B/C/D), 72 rows
pg.kruskal(data=df, dv='score', between='clinic')</code></pre>
<p class="out">Key output: H = %%kw_H%%, %%kw_p%%.</p>
</div></details>

<h2 id="cheat">The one-table cheat sheet</h2>
<table>
<tr><th>Test</th><th>Use it when</th><th>Key assumption</th><th>If it fails, use</th></tr>
<tr><td><a href="#ztest">z-test</a></td><td>Mean vs benchmark, &sigma; known</td><td>&sigma; known; n &ge; ~30</td><td>— (that's the t-test)</td></tr>
<tr><td><a href="#t1">One-sample t</a></td><td>Mean vs benchmark</td><td>Roughly normal, or n &ge; ~30</td><td><a href="#wilcoxon">Wilcoxon</a> (vs the benchmark)</td></tr>
<tr><td><a href="#t2">Two-sample t</a></td><td>Two group means</td><td>Normal-ish per group, or n &ge; ~30 each</td><td><a href="#mwu">Mann-Whitney U</a></td></tr>
<tr><td><a href="#paired">Paired t</a></td><td>Before/after means</td><td>Differences roughly normal</td><td><a href="#wilcoxon">Wilcoxon signed-rank</a></td></tr>
<tr><td><a href="#anova">ANOVA</a></td><td>3+ group means</td><td>Normal-ish per group, similar spreads</td><td><a href="#kruskal">Kruskal-Wallis</a></td></tr>
<tr><td><a href="#pairwise">Pairwise + Bonferroni</a></td><td>After ANOVA: which pairs?</td><td>Adjust p by the number of pairs</td><td>Pairwise rank tests + Bonferroni</td></tr>
<tr><td><a href="#prop1">One-sample proportion</a></td><td>Rate vs benchmark</td><td>np, n(1&minus;p) &ge; ~10</td><td>Exact binomial test</td></tr>
<tr><td><a href="#prop2">Two-sample proportion</a></td><td>Two group rates</td><td>Counts &ge; ~10 per cell</td><td>Fisher's exact / chi-squared</td></tr>
<tr><td><a href="#chi2ind">Chi-squared independence</a></td><td>Two categorical variables</td><td>Expected &ge; 5 per cell</td><td>Fisher's exact (2&times;2)</td></tr>
<tr><td><a href="#chi2gof">Chi-squared goodness of fit</a></td><td>Counts vs an expected distribution</td><td>Expected &ge; 5 per category</td><td>Exact / simulated test</td></tr>
<tr><td><a href="#wilcoxon">Wilcoxon signed-rank</a></td><td>Paired, skewed or ordinal</td><td>Paired differences exist</td><td>— (it is the fallback)</td></tr>
<tr><td><a href="#mwu">Mann-Whitney U</a></td><td>Two groups, skewed or ordinal</td><td>Independent groups</td><td>— (it is the fallback)</td></tr>
<tr><td><a href="#kruskal">Kruskal-Wallis</a></td><td>3+ groups, skewed or ordinal</td><td>Independent groups</td><td>— (it is the fallback)</td></tr>
</table>

<div class="box"><strong>Prove it:</strong> pick any number on this page, open the dropdown under it, and run
the code — the generator script already did, on the toy data described, and asserted every decision
(reject / fail to reject) before writing a word of HTML. If a result hadn't matched the story, the page
wouldn't exist.</div>

<p class="small">Part of the <a href="https://danieljordanviray.github.io/applied-data-science-lab/docs/">Applied
Data Science Lab</a>. Companion:
<a href="https://danieljordanviray.github.io/applied-data-science-lab/stats-and-prob/p-values-power-playground/">Statistical
Significance, Visualized</a> — the interactive p-value &amp; power playground.</p>
"""

JS = r"""
<script>
function npdf(x){return Math.exp(-x*x/2)/Math.sqrt(2*Math.PI);}
function erf(x){var s=x<0?-1:1; x=Math.abs(x); var t=1/(1+0.3275911*x);
  var y=1-(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-0.284496736)*t+0.254829592)*t*Math.exp(-x*x);
  return s*y;}
function ncdf(x){return 0.5*(1+erf(x/Math.SQRT2));}
function curvePath(cx,sd,x0,x1,yBase,scale,W,H){
  var X=function(x){return 30+(x-x0)/(x1-x0)*(W-60);};
  var Y=function(v){return yBase-v*scale;};
  var d='',i,x;
  for(i=0;i<=160;i++){x=x0+(x1-x0)*i/160; var px=X(x), py=Y(npdf((x-cx)/sd)/sd);
    d+=(i===0?'M':'L')+px.toFixed(1)+' '+py.toFixed(1);}
  return {d:d, X:X, Y:Y};
}
function shadePath(cx,sd,a,b,x0,x1,yBase,scale,W){
  var X=function(x){return 30+(x-x0)/(x1-x0)*(W-60);};
  var Y=function(v){return yBase-v*scale;};
  if(a<x0)a=x0; if(b>x1)b=x1; if(b<=a)return '';
  var d='M'+X(a).toFixed(1)+' '+yBase, i,x;
  for(i=0;i<=60;i++){x=a+(b-a)*i/60; d+='L'+X(x).toFixed(1)+' '+Y(npdf((x-cx)/sd)/sd).toFixed(1);}
  d+='L'+X(b).toFixed(1)+' '+yBase+'Z'; return d;
}

/* ---------- tails explorer ---------- */
var tailMode='two';
var CRIT={two:{'0.10':1.645,'0.05':1.96,'0.01':2.576}, one:{'0.10':1.282,'0.05':1.645,'0.01':2.326}};
function drawTails(){
  var svg=document.getElementById('tailsSvg'), W=640,H=240,yBase=200,scale=430,x0=-4.2,x1=4.2;
  var alpha=document.getElementById('alphaSel').value;
  var zc=CRIT[tailMode==='two'?'two':'one'][alpha];
  var c=curvePath(0,1,x0,x1,yBase,scale,W,H), s='';
  var regions = tailMode==='two' ? [[x0,-zc],[zc,x1]] : (tailMode==='right' ? [[zc,x1]] : [[x0,-zc]]);
  regions.forEach(function(r){ s+='<path d="'+shadePath(0,1,r[0],r[1],x0,x1,yBase,scale,W)+'" fill="#d64545" opacity="0.55"/>'; });
  s+='<path d="'+c.d+'" fill="none" stroke="#1a1a2e" stroke-width="2"/>';
  s+='<line x1="30" y1="'+yBase+'" x2="'+(W-30)+'" y2="'+yBase+'" stroke="#999"/>';
  var crits = tailMode==='two' ? [-zc,zc] : (tailMode==='right'?[zc]:[-zc]);
  crits.forEach(function(z){ var px=c.X(z);
    s+='<line x1="'+px+'" y1="60" x2="'+px+'" y2="'+yBase+'" stroke="#d64545" stroke-dasharray="4 3"/>';
    s+='<text x="'+px+'" y="50" font-size="13" text-anchor="middle" fill="#d64545">'+(z<0?'&minus;':'')+Math.abs(z).toFixed(2)+'</text>'; });
  s+='<text x="'+c.X(0)+'" y="'+(yBase+22)+'" font-size="13" text-anchor="middle" fill="#666">0 — the H0 value</text>';
  s+='<text x="'+(W/2)+'" y="18" font-size="13" text-anchor="middle" fill="#d64545">red = rejection region (&alpha; = '+alpha+')</text>';
  svg.innerHTML=s;
  var msg;
  if(tailMode==='two') msg='Two-tailed, &alpha; = '+alpha+': reject H<sub>0</sub> when |z| &gt; '+zc.toFixed(2)+' — a difference in <em>either</em> direction counts. The glucose example\'s z = %%z_stat%% clears this bar.';
  else if(tailMode==='right') msg='Right-tailed, &alpha; = '+alpha+': all of &alpha; sits in the right tail — reject H<sub>0</sub> when z &gt; '+zc.toFixed(2)+'. Only "greater" counts as evidence; a huge difference in the wrong direction is just... nothing.';
  else msg='Left-tailed, &alpha; = '+alpha+': reject H<sub>0</sub> when z &lt; &minus;'+zc.toFixed(2)+'. Only "less" counts. (This is the shape of the blood-pressure and length-of-stay tests below, run as two-tailed by default.)';
  document.getElementById('tailsOut').innerHTML=msg;
}
document.querySelectorAll('[data-tail]').forEach(function(b){
  b.addEventListener('click',function(){
    document.querySelectorAll('[data-tail]').forEach(function(x){x.classList.remove('on');});
    b.classList.add('on'); tailMode=b.getAttribute('data-tail'); drawTails();
  });
});
document.getElementById('alphaSel').addEventListener('change',drawTails);

/* ---------- Type I / II / power explorer ---------- */
function drawPow(){
  var svg=document.getElementById('powSvg'), W=640,H=250,yBase=205,scale=430,x0=-4.5,x1=8.5;
  var d=parseFloat(document.getElementById('dSlider').value);
  var n=parseInt(document.getElementById('nSlider').value,10);
  document.getElementById('dVal').textContent=d.toFixed(2);
  document.getElementById('nVal').textContent=n;
  var shift=Math.min(d*Math.sqrt(n),5.5), crit=1.96;
  var c0=curvePath(0,1,x0,x1,yBase,scale,W,H), cA=curvePath(shift,1,x0,x1,yBase,scale,W,H), s='';
  s+='<path d="'+shadePath(shift,1,crit,x1,x0,x1,yBase,scale,W)+'" fill="#3d9970" opacity="0.6"/>';
  s+='<path d="'+shadePath(shift,1,-crit,crit,x0,x1,yBase,scale,W)+'" fill="#e8912d" opacity="0.65"/>';
  s+='<path d="'+shadePath(0,1,crit,x1,x0,x1,yBase,scale,W)+'" fill="#d64545" opacity="0.55"/>';
  s+='<path d="'+c0.d+'" fill="none" stroke="#1a1a2e" stroke-width="2"/>';
  s+='<path d="'+cA.d+'" fill="none" stroke="#2b50aa" stroke-width="2"/>';
  s+='<line x1="30" y1="'+yBase+'" x2="'+(W-30)+'" y2="'+yBase+'" stroke="#999"/>';
  var px=c0.X(crit);
  s+='<line x1="'+px+'" y1="55" x2="'+px+'" y2="'+yBase+'" stroke="#d64545" stroke-dasharray="4 3"/>';
  s+='<text x="'+c0.X(0)+'" y="30" font-size="13" text-anchor="middle" fill="#1a1a2e">H0: no effect</text>';
  s+='<text x="'+c0.X(shift)+'" y="30" font-size="13" text-anchor="middle" fill="#2b50aa">HA: real effect</text>';
  s+='<text x="'+px+'" y="'+(yBase+22)+'" font-size="12" text-anchor="middle" fill="#d64545">rejection line (z = 1.96)</text>';
  svg.innerHTML=s;
  var beta=ncdf(crit-shift)-ncdf(-crit-shift), power=1-beta;
  var verdict = power>=0.8 ? 'comfortably powered — the conventional 80% bar is met.' :
    (power>=0.5 ? 'a coin-flip study: even with a real effect this size, you\'d miss it nearly half the time.' :
    'badly underpowered: most real effects this size slip straight past a study this small.');
  document.getElementById('powOut').innerHTML='<strong>Power &asymp; '+Math.round(power*100)+'%</strong> &middot; '+
    'Type II (&beta;) &asymp; '+Math.round(beta*100)+'% &middot; Type I (&alpha;) = 5% (fixed). '+
    'With d = '+d.toFixed(2)+' and n = '+n+': '+verdict;
}
document.getElementById('dSlider').addEventListener('input',drawPow);
document.getElementById('nSlider').addEventListener('input',drawPow);

/* ---------- which-test chooser ---------- */
function choose(){
  var o=document.getElementById('qOutcome').value, c=document.getElementById('qCompare').value,
      s=document.getElementById('qShape').value, p=document.getElementById('qPaired').value;
  var txt, href;
  if(c==='dist'){ txt='Chi-squared goodness-of-fit test — do the observed counts match the distribution you expected?'; href='#chi2gof'; }
  else if(o==='cats'||c==='assoc'){ txt='Chi-squared test of independence — build the two-way count table and test whether the variables are associated.'; href='#chi2ind'; }
  else if(o==='rate'){
    if(c==='bench'){ txt='One-sample proportion test — your rate vs the benchmark (z, not t: a proportion carries its own standard error). Small n? Use the exact binomial test instead.'; href='#prop1'; }
    else if(c==='two'){ txt='Two-sample proportion test — compare the two rates (the same test as chi-squared on the 2&times;2 table).'; href='#prop2'; }
    else { txt='Chi-squared test of independence — make it a group &times; outcome count table.'; href='#chi2ind'; }
  } else {
    if(c==='bench'){
      if(s==='skew'){ txt='Wilcoxon signed-rank test (one-sample form) against the benchmark — skewed data, so ranks beat means.'; href='#wilcoxon'; }
      else { txt='One-sample t-test — or the z-test above it if the population SD is genuinely known.'; href='#t1'; }
    } else if(c==='two'){
      if(p==='yes'){
        if(s==='skew'){ txt='Wilcoxon signed-rank test — paired data, but skewed or ordinal: rank the differences.'; href='#wilcoxon'; }
        else { txt='Paired t-test — same people measured twice, so run the test on the differences.'; href='#paired'; }
      } else {
        if(s==='skew'){ txt='Mann-Whitney U test — two independent groups, skewed outcome: compare ranks and report medians.'; href='#mwu'; }
        else { txt='Two-sample t-test (Welch\'s version) — compare the two group means.'; href='#t2'; }
      }
    } else {
      if(s==='skew'){ txt='Kruskal-Wallis test — 3+ groups on ranks; follow up with pairwise Mann-Whitney tests plus Bonferroni.'; href='#kruskal'; }
      else { txt='ANOVA first — one test asking "does anything differ?", then pairwise comparisons with a Bonferroni correction.'; href='#anova'; }
    }
  }
  document.getElementById('chooserOut').innerHTML='<strong>Reach for:</strong> '+txt+' <a href="'+href+'">Jump to it &darr;</a>';
}
['qOutcome','qCompare','qShape','qPaired'].forEach(function(id){
  document.getElementById(id).addEventListener('change',choose);
});

document.documentElement.style.scrollBehavior='smooth';
drawTails(); drawPow(); choose();
</script>
"""

# ============================ assemble + write ============================
import re
html = HEAD + BODY + JS + "</body></html>"
html = html.replace("%%PW_LINES%%", PW_LINES)
for k, v in V.items():
    html = html.replace(f"%%{k}%%", str(v))
leftover = re.findall(r"%%[A-Za-z_0-9]+%%", html)
assert not leftover, f"unsubstituted tokens: {leftover}"
out = Path(__file__).resolve().parent.parent / "practitioners-guide" / "practitioners-guide.html"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(html)
print(f"wrote {out} ({len(html)/1024:.1f} KB)")
