"""Generate the DiD case-study report (intuitive + [formal names])."""
import base64
from pathlib import Path

STUDY = Path(__file__).resolve().parent.parent  # this case study's folder
img = base64.b64encode((STUDY / "output" / "figures" / "did_trends.png").read_bytes()).decode()

html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Case study: Difference-in-Differences</title>
<style>
body {{ font-family: -apple-system, Georgia, serif; max-width: 720px; margin: 0 auto;
       padding: 24px 18px; line-height: 1.7; color: #1a1a1a; }}
h1 {{ font-size: 1.6em; }} h2 {{ margin-top: 2em; border-bottom: 1px solid #ddd; padding-bottom: 6px; }}
.boss {{ background: #eef4ff; border-left: 4px solid #1f6feb; padding: 12px 16px; margin: 1em 0; }}
.proveit {{ background: #eef7ee; border-left: 4px solid #2a7d2a; padding: 12px 16px; margin: 1em 0; }}
table {{ border-collapse: collapse; font-size: 0.9em; margin: 1em 0; }}
th, td {{ border: 1px solid #ccc; padding: 6px 12px; text-align: center; }}
th {{ background: #f0f0f0; }}
img {{ max-width: 100%; height: auto; margin: 1em 0; }}
.small {{ color: #555; font-size: 0.9em; }}
code {{ background: #f0f0f0; padding: 1px 5px; border-radius: 4px; font-size: 0.88em; }}
pre {{ background: #f4f4f4; padding: 12px; border-radius: 6px; overflow-x: auto; font-size: 0.82em; }}
.formal {{ color: #6a5acd; }}
.topnav {{ font-size: .85rem; margin: 0 0 1.2em; }}
.topnav a {{ color: #1f6feb; text-decoration: none; }}
</style></head><body>
<div class="topnav"><a href="../../docs/index.html">← Homepage</a></div>


<h1>Case study: Difference-in-Differences — did the program work?</h1>
<p class="small">October 10, 2026</p>
<p class="small">Methods case study · synthetic data with a planted true effect of +5 points,
so every claim below is checkable against an answer key we wrote ourselves.</p>

<h2>The story</h2>
<p>State A launches a job-training program in 2020. State B does nothing. We observe
county employment rates in both states, every year 2018–2022. Question: did the program
raise employment? This is the bread-and-butter setup: one group gets the treatment,
one doesn't, and we watch both before and after.</p>

<h2>Why the two obvious comparisons fail</h2>
<p><b>Obvious idea 1:</b> look at State A before vs. after. It went 61.59 → 68.83, up
7.24 points. Program works, right? Not so fast — State B <i>also</i> rose, 64.34 →
67.07. Employment was climbing everywhere. Crediting all of A's rise to the program
confuses the program with the passing of time <span class="formal">[confounding by time trends]</span>.</p>
<p><b>Obvious idea 2:</b> compare A vs. B after the program: 68.83 vs. 67.07, A wins by
1.76. But they started at different levels (61.59 vs. 64.34) — maybe A was always
going to differ <span class="formal">[baseline differences / selection bias]</span>.</p>
<p>Each naive comparison has exactly one hole. Difference-in-differences patches both
at once.</p>

<h2>The idea: subtract the changes</h2>
<table>
<tr><th></th><th>Before (2018–19)</th><th>After (2020–22)</th><th>Change</th></tr>
<tr><th>State A (program)</th><td>61.59</td><td>68.83</td><td>+7.24</td></tr>
<tr><th>State B (no program)</th><td>64.34</td><td>67.07</td><td>+2.73</td></tr>
</table>
<p>State A's change minus State B's change: <b>7.24 − 2.73 = 4.51 points</b>. That's the
estimate.</p>
<p><b>The intuition.</b> State B's change (+2.73) is our best guess for <i>what would
have happened in State A without the program</i> <span class="formal">[the counterfactual]</span>.
We can't observe that world directly — it's the road not taken — so we borrow B's
journey as the stand-in. Whatever is left over after subtracting it, we attribute to
the program. Two differences, hence the name.</p>

<h2>The regression version (same thing, fancier clothes)</h2>
<pre>employment ~ treated_state + post + treated_state:post</pre>
<p>Plain English: start from a baseline, add "are you in State A?", add "is it after
2020?", and then the star of the show — the <i>interaction</i>
<span class="formal">[Treat × Post coefficient, β₃]</span>: "how much <i>extra</i>
change did the treated group get, after?" That coefficient <i>is</i> the 4.51 from the
table. The regression just packages it with standard errors: 4.51, 95% CI (3.18, 5.84),
p ≈ 0.0000000001. The planted truth was 5.0 — inside the interval.</p>

<h2>The one assumption that matters</h2>
<p><b>Intuition:</b> this whole trick works only if, <i>without the program, State A
would have kept moving in parallel with State B</i> — the gap between them would have
stayed roughly constant <span class="formal">[parallel trends assumption]</span>. Look
at the plot: 2018–2019, both lines climb about 1 point per year, side by side. That's
the assumption, made visible. If A had already been accelerating before 2020, we'd be
crediting the program for a trend that was already there.</p>
<img src="data:image/png;base64,{img}" alt="parallel pre-trends diverging after 2020">
<p><b>Supporting assumptions</b> (the fine print): the program didn't leak into State B
<span class="formal">[no spillover / SUTVA]</span> — if B's workers joined A's program,
B stops being a clean stand-in. The counties are the same kind of places before and
after <span class="formal">[stable composition]</span>. And the start date is sharp —
everyone knows when 2020 began.</p>

<h2>Checks along the way</h2>
<div class="proveit">
<b>Check 1 — eyeball the pre-trends.</b> The plot above: parallel before 2020, sharp
divergence after. If the pre-lines had fanned out, we'd stop right here.<br><br>
<b>Check 2 — the placebo test</b> <span class="formal">[falsification test]</span>.
Pretend the program started in 2019 and run the whole machine on 2018–2019 data only —
a period where we <i>know</i> nothing happened. Result: estimated effect 0.27, p = 0.80.
Correctly finds nothing. If the method hallucinated effects where none exist, we
couldn't trust it where we need it.<br><br>
<b>Check 3 — the planted truth.</b> We wrote the answer key: the true effect is 5.0.
The method recovered 4.51 with CI (3.18, 5.84). It found what we hid.
</div>

<h2>Limitations, honestly</h2>
<ul>
<li><b>Parallel trends can't be proven.</b> We verified it <i>before</i> 2020, but the
assumption is about what would have happened <i>after</i> 2020 without the program —
a world we never observe. It's an assumption, not a finding. The pre-trend check makes
it plausible, never certain.</li>
<li><b>Everything hinges on the control group.</b> A bad stand-in (a state on a totally
different trajectory) gives a confidently wrong answer. Choosing B is the real skill.</li>
<li><b>With two states, uncertainty is understated</b> <span class="formal">[clustering]</span> —
proper inference clusters at the level treatment was assigned; with few clusters the
standard errors deserve skepticism. Noted, not solved, in this brick.</li>
<li><b>Synthetic is clean; reality isn't.</b> Real data brings messy timing, partial
compliance, and spillovers. The machinery is the same; the vigilance has to be higher.</li>
</ul>

<div class="boss"><b>The 30-second version for your boss.</b><br>
"We wanted to know if State A's training program worked. We couldn't just look at
State A before and after, because employment was already rising everywhere — and we
couldn't just compare the states afterward, because they started at different levels.
So we measured how much each state <i>changed</i>, and subtracted: State B's change
is our best guess for what would have happened in A without the program. The leftover
difference, about four and a half points, is our estimate of the program's effect.
The key judgment call is that the two states were moving in parallel before the
program started, which we verified in the earlier years."</div>

<h2>The code (now in the code bank)</h2>
<p>This study earned its place in the bank: the reusable pieces live in
<code>toolkit/did.py</code> — <code>did_2x2</code>, <code>did_regression</code>,
<code>placebo_did</code>, <code>plot_group_trends</code> — each verified to reproduce
the numbers on this page. The working version is shown below:</p>
<pre>import statsmodels.formula.api as smf

# the 2x2 table of means
t = df.groupby(["treated_state", "post"])["employment"].mean().unstack()
did = (t.loc[True, True] - t.loc[True, False]) \\
    - (t.loc[False, True] - t.loc[False, False])   # 4.51

# the regression version: the interaction IS the DiD estimate
fit = smf.ols("employment ~ treated_state + post + treated_state:post",
              data=df).fit()
fit.params["treated_state[T.True]:post[T.True]"]   # 4.51

# placebo: pretend treatment began in 2019, pre-period data only
pre = df[df.year <= 2019].copy()
pre["fake_post"] = pre.year >= 2019
smf.ols("employment ~ treated_state + fake_post + treated_state:fake_post",
        data=pre).fit().params["treated_state[T.True]:fake_post[T.True]"]
# 0.27, p = 0.80 -> correctly finds nothing</pre>
<p class="small">Full script: <code>scripts/did_case_study.py</code> in the project folder.
Data: <code>data/did_synthetic.csv</code> (80 counties × 5 years, planted effect +5).</p>

<h2>What's next</h2>
<p>Your call. Natural follow-ups: staggered timing (states adopt in different years),
the event-study plot (effect year by year instead of one number), or applying DiD to a
real dataset where the answer key doesn't exist and the checks have to carry the weight.</p>
</body></html>"""

out = STUDY / "did-case-study.html"
out.write_text(html)
print("written:", out, f"({out.stat().st_size/1024:.0f} KB)")
