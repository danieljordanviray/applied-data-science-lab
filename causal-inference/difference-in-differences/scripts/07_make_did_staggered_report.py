"""Generate the staggered-DiD case-study report."""
import base64
from pathlib import Path

STUDY = Path(__file__).resolve().parent.parent  # this case study's folder
def b64(n):
    return base64.b64encode((STUDY / "output" / "figures" / n).read_bytes()).decode()

html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Case study: Staggered Difference in Differences</title>
<style>
body {{ font-family: -apple-system, Georgia, serif; max-width: 720px; margin: 0 auto;
       padding: 24px 18px; line-height: 1.7; color: #1a1a1a; }}
h1 {{ font-size: 1.6em; }} h2 {{ margin-top: 2em; border-bottom: 1px solid #ddd; padding-bottom: 6px; }}
.boss {{ background: #eef4ff; border-left: 4px solid #1f6feb; padding: 12px 16px; margin: 1em 0; }}
.answer {{ background: #f6f0ff; border-left: 4px solid #6a5acd; padding: 12px 16px; margin: 1em 0; }}
.proveit {{ background: #eef7ee; border-left: 4px solid #2a7d2a; padding: 12px 16px; margin: 1em 0; }}
table {{ border-collapse: collapse; font-size: 0.88em; margin: 1em 0; }}
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


<h1>Staggered Difference in Differences — when treatment arrives at different times</h1>
<p class="small">Methods case study · synthetic data, planted effects per cohort (8 / 6 / 4 / 2),
true overall effect 5.0 — the answer key exists, so everything is checkable.</p>

<h2>The story</h2>
<p>A retail chain rolls out a new staff-scheduling system store by store. Fifteen stores
adopt in <b>March 2022</b>, fifteen in <b>May</b>, fifteen in <b>July</b>, fifteen in
<b>September</b>. Forty stores never adopt. We track monthly sales, Oct 2021 – Dec 2022.
Same question as last time — did it work? — but now there is no single "before" and
"after." Every store has its own clock.</p>

<h2>The new wrinkle: event time</h2>
<p>Relabel each store's calendar around its own adoption month: adoption month becomes
<b>month 0</b>, the three months before are <b>−3, −2, −1</b>, the three after are
<b>+1, +2, +3</b> <span class="formal">[event time / relative time]</span>. A store
adopting in March and a store adopting in September now line up on the same ruler, and
"3 months pre, 3 months post" means the same thing for everyone. This relabeling is
half the battle.</p>

<div class="answer"><b>Your question, answered: what about the control group — they have
no start date?</b><br>
They don't need one. The never-treated stores lend their timeline to <i>every</i>
cohort's comparison: the March cohort is compared against never-treated stores over
March's calendar window, the September cohort against the same stores over September's
window. In event time, the controls show up in every cohort's 2×2 table. A control
group without a start date is a feature, not a problem — they're the universal
yardstick.</div>

<h2>The estimator: one DiD per cohort, then average</h2>
<p>For each adoption wave, run the familiar 2×2 against the never-treated stores over
that wave's own window — this reuses the code bank's <code>did_2x2</code> verbatim,
the flywheel working — then average the four, weighted by cohort size
<span class="formal">[group-time aggregation, Callaway &amp; Sant'Anna intuition]</span>:</p>
<table>
<tr><th>Cohort</th><th>Planted truth</th><th>Recovered DiD</th></tr>
<tr><td>Mar 2022</td><td>+8.0</td><td>+8.96</td></tr>
<tr><td>May 2022</td><td>+6.0</td><td>+6.15</td></tr>
<tr><td>Jul 2022</td><td>+4.0</td><td>+2.84</td></tr>
<tr><td>Sep 2022</td><td>+2.0</td><td>+1.26</td></tr>
<tr><th>Average (ATT)</th><th>5.0</th><th><b>4.8</b> (±1.1)</th></tr>
</table>
<p>Each cohort is a little noisy (15 stores), but the average lands on the truth.</p>

<h2>The twist: the naive one-big-regression lies to you</h2>
<p>The tempting shortcut is one giant regression with store and month fixed effects plus
a "treated" dummy <span class="formal">[two-way fixed effects, TWFE]</span>. Pooled, it
says 4.98 — close to the truth here, so you might trust it. But look at its
<i>event-study</i> version, which estimates the effect month by month:</p>
<img src="data:image/png;base64,{b64('did_event_twfe.png')}" alt="TWFE event study">
<p><b>Phantom pre-trends</b> (−1.2, −2.3 before adoption — we <i>planted</i> zero) and
<b>muted post effects</b> (~2 instead of ~5). Now the honest version, built cohort by
cohort:</p>
<img src="data:image/png;base64,{b64('did_event_cohort.png')}" alt="Cohort event study">
<p>Flat around zero before, clean lift of ~5 after. Same data, two pictures.</p>
<p><b>Intuition for why.</b> The big regression doesn't just compare treated stores to
never-treated stores. Behind the scenes it also compares <i>later</i> adopters against
<i>already-treated</i> stores — using treated stores as controls
<span class="formal">[Goodman-Bacon decomposition; "forbidden comparisons"]</span>.
When effects differ across waves (here: 8 down to 2), those comparisons smear effects
across time periods <span class="formal">[Sun &amp; Abraham contamination]</span>:
pre-period coefficients absorb post-treatment effects with negative weights, inventing
trends that never existed and shrinking the real ones. The cohort method never lets a
treated store play control. That's the whole difference, and it's why the modern best
practice is cohort-first.</p>

<h2>Checks along the way</h2>
<div class="proveit">
<b>Check 1 — pre-period flatness.</b> The cohort event study's pre-coefficients sit at
0.9 and 0.5 with wide intervals covering zero: no effect before adoption, as planted.
The parallel-trends check, now month by month.<br><br>
<b>Check 2 — placebo.</b> Thirty repetitions of fake four-wave adoptions among the
never-treated stores, run through the identical cohort machinery: average estimated
effect <b>0.06 ± 0.09</b>. Finds nothing, as it should.<br><br>
<b>Check 3 — the planted truth.</b> True average effect 5.0 → recovered 4.8 ± 1.1.
The method found what we hid, per cohort and overall.<br><br>
<b>Check 4 — the check that caught <i>my</i> bug.</b> My first placebo accidentally
included already-treated stores in the control group and returned −2.58 — the check
screamed, I looked, and the contamination was exactly the forbidden-comparison problem
above, in miniature. The checks aren't decoration; this one earned its keep.
</div>

<h2>Limitations, honestly</h2>
<ul>
<li><b>Parallel trends, per cohort.</b> Each wave needs its own believable parallel
path with the controls. More waves, more assumptions.</li>
<li><b>No anticipation</b> <span class="formal">[no anticipatory effects]</span> — stores
didn't change behavior <i>before</i> their adoption month. If they staffed up early,
month −1 is already treated.</li>
<li><b>No spillover between stores</b> — a treated store didn't steal sales from a
control store across the street.</li>
<li><b>Effects assumed constant within cohort here;</b> real effects may also evolve
over event time — the event study is how you'd see that.</li>
<li><b>Standard errors are approximate</b> (clustering at the store level deserves a
fuller treatment with few cohorts).</li>
</ul>

<div class="boss"><b>The 30-second version for your boss.</b><br>
"Different stores got the new system in different months, so we lined every store up
on its own clock — three months before and after its start date. For each wave of
stores, we compared their change against stores that never got the system over the
same calendar months; those stores show us what would have happened anyway. Then we
averaged across the four waves: about +$4.8k per store per month. One technical
judgment call: we did it wave by wave on purpose, because the shortcut version —
one big regression — quietly compares later stores against already-treated ones and
gives a distorted picture. The wave-by-wave version doesn't."</div>

<h2>The code</h2>
<p>The bank's <code>did_2x2</code> did the per-cohort work unchanged — first reuse,
flywheel turning. The staggered machinery (event-time relabeling, cohort aggregation,
both event studies, the repeated placebo) is new code in
<code>scripts/did_staggered.py</code>, shown here for review and <b>not</b> in the bank
until you say so:</p>
<pre>df["event"] = df["month_idx"] - df["adopt_month"]  # each store's own clock

# one honest DiD per cohort, vs never-treated, then average
ests = []
for cohort, ai in cohorts.items():
    win = df[((df.cohort == cohort) | (df.cohort == "never")) &
             (df.mi >= ai - 3) & (df.mi <= ai + 2)]
    win["post"] = win.mi >= ai
    ests.append(did_2x2(win, "ever_treated", "post", "sales")["estimate"])
att = sum(ests) / len(ests)   # 4.8, truth 5.0</pre>
<p class="small">Full script: <code>scripts/did_staggered.py</code>. Data:
<code>data/did_staggered.csv</code> (100 stores × 15 months, planted effects 8/6/4/2).</p>

<h2>What's next</h2>
<p>Your call. Candidates: apply the cohort machinery to a real dataset (where the
answer key doesn't exist and the checks carry the weight), add covariates to the
comparison, or the event-study plot as a standalone recipe for the bank.</p>
</body></html>"""

out = STUDY / "did-staggered-case-study.html"
out.write_text(html)
print("written:", out, f"({out.stat().st_size/1024:.0f} KB)")
