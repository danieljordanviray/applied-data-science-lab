"""Generate the self-contained Phase 1 HTML report for Daniel."""
import base64
from pathlib import Path
import pandas as pd

STUDY = Path(__file__).resolve().parent.parent  # this case study's folder
res = pd.read_csv(STUDY / "output" / "results.csv")

def b64(name):
    p = STUDY / "output" / "figures" / name
    return base64.b64encode(p.read_bytes()).decode()

def fmt_p(p):
    return f"{p:.2e}" if p < 0.001 else f"{p:.3f}"

def verdict(r):
    if r["sig_bonf"]:
        return "Yes — survives both"
    if r["sig_bh"]:
        return "Borderline — FDR only"
    return "No"

rows = []
for _, r in res.sort_values("p_raw").iterrows():
    rows.append(
        f"<tr><td>{r['label']}</td><td>{int(r['n'])}</td>"
        f"<td>{fmt_p(r['p_raw'])}</td><td>{verdict(r)}</td>"
        f"<td>{r['cohens_d']:.2f}</td><td>{fmt_p(r['p_adj'])}</td></tr>"
    )
table = "\n".join(rows)

html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Phase 1 Report — Biomarkers and Hypertension</title>
<style>
body {{ font-family: -apple-system, Georgia, serif; max-width: 760px; margin: 0 auto;
       padding: 24px 18px; line-height: 1.65; color: #1a1a1a; }}
h1 {{ font-size: 1.7em; }} h2 {{ margin-top: 2em; border-bottom: 1px solid #ddd; padding-bottom: 6px; }}
.lesson {{ background: #f4f7ff; border-left: 4px solid #1f6feb; padding: 12px 16px; margin: 1em 0; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.85em; }}
th, td {{ border: 1px solid #ccc; padding: 6px 8px; text-align: left; }}
th {{ background: #f0f0f0; }}
img {{ max-width: 100%; height: auto; margin: 1em 0; }}
.small {{ color: #555; font-size: 0.9em; }}
code {{ background: #f0f0f0; padding: 1px 5px; border-radius: 4px; font-size: 0.9em; }}
</style></head><body>

<h1>Phase 1 Report: Which blood biomarkers actually associate with hypertension?</h1>
<p class="small">Bio-ML Library · Phase 1 · NHANES 2017–2018 (CDC, public domain) · n = 5,437 adults</p>

<h2>The question</h2>
<p>Across nine blood biomarkers measured in a real national health survey, which ones are
genuinely associated with having hypertension — once we correct for the fact that we
tested nine things at once?</p>

<h2>How I went about it</h2>
<ol>
<li><b>Download.</b> Pulled ten public files from the CDC: demographics, blood-pressure
exam and questionnaire, cholesterol labs, HbA1c, glucose, CRP, and body measures.</li>
<li><b>Clean and merge.</b> Joined everything on the respondent ID, averaged the
repeated blood-pressure readings, recoded questionnaire answers, and mapped lab
columns to clean names. Real-world mess, handled explicitly in code.</li>
<li><b>Define the outcome.</b> Hypertension = mean systolic ≥ 130, or mean diastolic ≥ 80,
or currently on blood-pressure medication (the ACC/AHA-style rule).</li>
<li><b>Compare groups.</b> For each biomarker: survey-weighted means, a weighted
two-sample t-test, a Mann-Whitney robustness check, and an effect size.</li>
<li><b>Correct for nine tests.</b> Bonferroni and Benjamini–Hochberg.</li>
<li><b>Interrogate confounding.</b> Age/sex-adjusted logistic regression plus
age-stratified means: is it the biomarker, or just age?</li>
<li><b>Extract the bricks.</b> Every reusable piece went into the <code>toolkit/</code>
code bank for future phases.</li>
</ol>

<h2>Results</h2>
<p><b>Headline:</b> 48% of US adults meet the hypertension definition (survey-weighted).
The metabolic cluster wins clearly: waist circumference, HbA1c, fasting glucose, and
BMI show the largest differences and all survive age/sex adjustment.</p>
<table>
<tr><th>Biomarker</th><th>n</th><th>Raw p</th><th>Significant?</th><th>Cohen's d</th><th>Age-adjusted p</th></tr>
{table}
</table>
<p class="small">Cohen's d = standardized difference (hypertension vs not). Age-adjusted p comes
from a logistic regression controlling for age and sex.</p>

<img src="data:image/png;base64,{b64('fig1_pvalues.png')}" alt="p-values vs correction thresholds">
<p class="small"><b>Figure 1.</b> Raw p-values for all nine biomarkers against the naive 0.05 line
and the Bonferroni line. Seven clear the strict bar; HDL sits in between — significant
raw, dead under Bonferroni, alive under FDR.</p>

<img src="data:image/png;base64,{b64('fig2_forest.png')}" alt="forest plot of effect sizes">
<p class="small"><b>Figure 2.</b> Effect sizes with 95% intervals. Waist circumference (d = 0.67)
and HbA1c (d = 0.58) lead; the ranking by <i>size</i> differs from the ranking by
<i>p-value</i> — that difference is lesson 4 below.</p>

<img src="data:image/png;base64,{b64('fig3_age_confounding.png')}" alt="waist by hypertension within age bands">
<p class="small"><b>Figure 3.</b> The waist-circumference gap persists inside every age band, so it
is not just age in disguise. Note the gap is biggest in the youngest band — a hint
worth following up.</p>

<img src="data:image/png;base64,{b64('fig4_sbp_dist.png')}" alt="systolic BP distribution">
<p class="small"><b>Figure 4.</b> The systolic blood-pressure distribution with the 130 mmHg
cutoff. Outcomes are choices; this one follows the ACC/AHA rule and is documented
in the code.</p>

<h2>The walkthrough: five lessons from this phase</h2>

<div class="lesson"><b>Lesson 1 — What a p-value actually is.</b><br>
Plain English: a p-value is the probability of seeing a difference <i>at least this
big</i> if there were really no difference at all. Small p = "this would be a weird
coincidence." It is <i>not</i> the probability that the effect is real, and it says
nothing about how big the effect is.<br>
<span class="small">Formal: p = P(data this extreme | H<sub>0</sub>), where H<sub>0</sub>
is "equal means." Our test statistic was Welch's t on survey-weighted means.</span></div>

<div class="lesson"><b>Lesson 2 — Nine tests, nine chances to fool yourself.</b><br>
Plain English: if you test 20 useless biomarkers at the 5% level, you'll "discover"
about one fake association on average. That's not fraud, it's arithmetic. Bonferroni
fixes it by demanding p &lt; 0.05/9 — strict, sometimes too strict. Benjamini–Hochberg
instead controls the <i>false discovery rate</i>: among the things you call significant,
at most 5% are false alarms. Watch HDL in our table: raw p = 0.007 looks exciting,
Bonferroni kills it, FDR keeps it. Same data, three verdicts — the method is the message.<br>
<span class="small">Formal: Bonferroni controls the family-wise error rate at α;
BH controls E[V/R] ≤ α. With m = 9 tests, Bonferroni's bar is p &lt; 0.0056.</span></div>

<div class="lesson"><b>Lesson 3 — Why the survey weights exist.</b><br>
Plain English: NHANES deliberately oversamples some groups (older adults, some
minorities) so it can study them precisely. A plain head-count would then
misrepresent America. Every respondent carries a weight saying "I stand for this many
people." Our 48% prevalence number uses those weights; the unweighted number would be
wrong in a boring, silent way — the most dangerous kind.<br>
<span class="small">Formal: we used the MEC exam weight WTMEC2YR; weighted means and a
Welch-style t with Kish effective sample sizes. Full design-based standard errors
would also use the PSU/strata variables — noted as a refinement.</span></div>

<div class="lesson"><b>Lesson 4 — Effect size vs. significance.</b><br>
Plain English: with 5,000 people, even a trivial difference can be "highly
significant." The p-value answers "is there anything?" Cohen's d answers "how much?"
Waist circumference doesn't just have the smallest p — it has the biggest <i>d</i>
(0.67, a medium-to-large gap). Always report both; never let a tiny p-value sell you
a tiny effect.<br>
<span class="small">Formal: d = (μ₁ − μ₂)/s_pooled on weighted moments; 0.2/0.5/0.8 are
the conventional small/medium/large benchmarks.</span></div>

<div class="lesson"><b>Lesson 5 — Confounding: is it the biomarker, or just age?</b><br>
Plain English: older people have higher blood pressure <i>and</i> worse biomarkers.
So a biomarker can "predict" hypertension while merely being a proxy for age. We
checked two ways: a regression that holds age and sex fixed, and Figure 3, which
compares groups <i>within</i> age bands. The metabolic signals survive both. And the
fun one: LDL cholesterol shows <i>no</i> signal at all — plausibly because people with
high LDL take statins, which lowers their LDL while they stay older and hypertensive.
A null result can be the most informative one in the room, and treatment effects
poisoning observational comparisons is exactly what phase 2 (causal inference) is for.<br>
<span class="small">Formal: per-biomarker logistic regression,
logit(P) = β₀ + β₁·z_biomarker + β₂·age + β₃·male, survey-weighted; biomarkers
standardized so odds ratios are per-SD and comparable.</span></div>

<h2>Toolkit bricks gained</h2>
<ul>
<li><code>toolkit/nhanes.py</code> — read XPT files, merge on respondent ID, recode answers</li>
<li><code>toolkit/weighted.py</code> — weighted means, variances, t-tests, effect sizes, prevalence</li>
<li><code>toolkit/multitest.py</code> — Bonferroni and Benjamini–Hochberg corrections</li>
<li><code>toolkit/plots.py</code> — house plotting style + forest plots</li>
<li><code>toolkit/eda.py</code> — one-call data-quality report (missingness, descriptives)</li>
</ul>
<p>Phase 2 will import these instead of rewriting them. That's the flywheel.</p>

<h2>Limitations (honestly)</h2>
<ul>
<li>Cross-sectional: association, not causation. Nothing here says what <i>causes</i> hypertension.</li>
<li>Weighted t-tests are approximations; full survey-design standard errors are a refinement.</li>
<li>Fasting analytes ideally want fasting-subsample weights; the MEC weight was used and documented.</li>
<li>Medication use (statins especially) contaminates the lipid comparisons — see the LDL story.</li>
</ul>

<h2>What's next</h2>
<p>Phase 2: causal inference. We take the confounding lesson seriously — propensity
methods on NHANES to estimate effects rather than associations, then possibly CMS
synthetic claims data, which looks like the real-world data of your day job.</p>

<p class="small">All code, the notebook, and the toolkit live in the project workspace and will
move to GitHub whenever you're ready. Data: CDC NHANES 2017–2018, public domain.</p>

</body></html>"""

out = STUDY / "phase-01-report.html"
out.write_text(html)
print("report written:", out, f"({out.stat().st_size/1024:.0f} KB)")
