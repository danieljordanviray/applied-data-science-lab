"""Generate the staggered-DiD case-study report (gentle version)."""
import base64
import json
from pathlib import Path

STUDY = Path(__file__).resolve().parent.parent  # this case study's folder

def b64(n):
    return base64.b64encode((STUDY / "output" / "figures" / n).read_bytes()).decode()

res = json.load(open(STUDY / "output" / "did_staggered_results.json"))
ce = res["cohort_est"]
ev = res["event_study"]

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
.trap {{ background: #fef2f2; border-left: 4px solid #dc2626; padding: 12px 16px; margin: 1em 0; }}
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
<div class="topnav"><a href="../../docs/index.html">\u2190 Homepage</a></div>

<h1>Staggered Difference in Differences — when everyone starts at different times</h1>
<p class="small">Methods case study · synthetic data, planted effects per start month
(&minus;4 / &minus;3 / &minus;2 kg), true overall effect &minus;3.0 kg — the answer key
exists, so everything is checkable. Assumes you know plain difference-in-differences.</p>

<h2>The setup</h2>
<p>100 patients. A weight-loss drug — but nobody designed an experiment. Some patients
started it in <strong>January</strong>, some in <strong>February</strong>, some in
<strong>March</strong>, and 40 never took it at all. We can see everyone's weight,
every month. Question: <strong>did the drug work?</strong></p>
<p>This is the situation observational data usually comes in: treatment happened,
but on nobody's schedule but reality's.</p>

<h2>Why plain DiD gets confused</h2>
<p>In the classic setup there's one "before" and one "after." Here, <em>"after"</em>
means January for some patients, February for others, March for the rest. If you
lump everyone together and compare "before vs after," you're averaging people whose
treatment started months apart — and worse, at any given calendar month, some of
your "control" patients already started the drug months ago. Their weight already
reflects the treatment, so they stop being a clean stand-in for "what would have
happened."</p>

<h2>The fix, in one sentence</h2>
<div class="answer">Line everyone up on <strong>"months since <em>they</em> started"</strong>
instead of calendar month <span class="formal">[event time]</span>; do one small,
honest DiD per start-month group against the never-treated patients
<span class="formal">[cohort-specific DiD]</span>; then average the answers
<span class="formal">[aggregation]</span>.</div>
<p>That's the whole machinery. Everything below is just doing it carefully.</p>

<h2>By hand: 6 patients</h2>
<p>Ana and Ben started in January. Cara and Dan started in February. Eli and Fay never
took it. Weights in kg (Oct&ndash;Mar):</p>
<table>
<tr><th></th><th>Oct</th><th>Nov</th><th>Dec</th><th>Jan</th><th>Feb</th><th>Mar</th></tr>
<tr><td>Ana (Jan)</td><td>90</td><td>90</td><td>90</td><td>87</td><td>87</td><td>87</td></tr>
<tr><td>Ben (Jan)</td><td>92</td><td>92</td><td>92</td><td>89</td><td>89</td><td>89</td></tr>
<tr><td>Cara (Feb)</td><td>90</td><td>90</td><td>90</td><td>90</td><td>88</td><td>88</td></tr>
<tr><td>Dan (Feb)</td><td>91</td><td>91</td><td>91</td><td>91</td><td>89</td><td>89</td></tr>
<tr><td>Eli (never)</td><td>90</td><td>90</td><td>90</td><td>90</td><td>90</td><td>90</td></tr>
<tr><td>Fay (never)</td><td>89</td><td>89</td><td>89</td><td>89</td><td>89</td><td>89</td></tr>
</table>
<p><strong>January group:</strong> their "before" is Oct&ndash;Dec, their "after" is
Jan&ndash;Mar. Treated change: 88.0 &minus; 91.0 = <strong>&minus;3.0</strong>.
Never-treated change over the same months: 89.5 &minus; 89.5 = <strong>0</strong>.
DiD = <strong>&minus;3.0 kg</strong> — the planted truth.</p>
<p><strong>February group:</strong> their "before" is Oct&ndash;Jan, their "after" is
Feb&ndash;Mar. Treated change: 88.5 &minus; 90.5 = <strong>&minus;2.0</strong>.
Never-treated change over the same months: <strong>0</strong>. DiD =
<strong>&minus;2.0 kg</strong> — the planted truth.</p>
<p><strong>Average:</strong> (&minus;3.0 + &minus;2.0) / 2 = <strong>&minus;2.5
kg</strong>. Each group compared against the never-treated over <em>its own</em>
calendar window. Nothing fancy — just the plain DiD you already know, run twice.</p>

<h2>The trap: already-treated patients as controls</h2>
<div class="trap"><strong>Watch what happens with one lazy shortcut.</strong> For the
February group, suppose we use the <em>January</em> patients as the "control"
instead of the never-treated — after all, in February the January group isn't
"starting" anything.<br><br>
January change over Feb's window (Oct&ndash;Jan vs Feb&ndash;Mar):
90.25 &rarr; 88.0 = <strong>&minus;2.25</strong> (they're still losing weight
<em>from the drug</em>).<br>
February change: <strong>&minus;2.0</strong>.<br>
"DiD" = &minus;2.0 &minus; (&minus;2.25) = <strong>+0.25 kg</strong> — the drug looks
like it causes weight <em>gain</em>.<br><br>
Using already-treated patients as the control subtracts the drug's own effect and
can even flip the sign <span class="formal">[forbidden comparisons]</span>. This is
the mistake the naive one-big-regression makes quietly, inside its machinery.</div>

<h2>Does the machinery work? 100 patients, planted answers</h2>
<p>Same idea, bigger: 20 January starters (truth &minus;4), 20 February starters
(truth &minus;3), 20 March starters (truth &minus;2), 40 never-treated. One honest
DiD per start month, reusing the code bank's <code>did_2x2</code>:</p>
<table>
<tr><th>Start month</th><th>Estimated effect</th><th>Planted truth</th></tr>
<tr><td>January</td><td>{ce['jan']['estimate']:.2f} kg</td><td>&minus;4.00 kg</td></tr>
<tr><td>February</td><td>{ce['feb']['estimate']:.2f} kg</td><td>&minus;3.00 kg</td></tr>
<tr><td>March</td><td>{ce['mar']['estimate']:.2f} kg</td><td>&minus;2.00 kg</td></tr>
</table>
<p><strong>Average effect: {res['att']:.2f} kg</strong> vs planted truth
&minus;3.00 kg. <span class="formal">[The average treatment effect on the
treated, ATT.]</span></p>

<h2>The event-study plot: lined up by "months since starting"</h2>
<p>Instead of one number, estimate the effect <em>month by month</em> relative to
each patient's start — month &minus;1 is the reference (zero by construction).
Before starting, the estimates should hover near zero (no effect before the drug
exists); after starting, they should sit near the true effect:</p>
<img src="data:image/png;base64,{b64('did_event_cohort.png')}" alt="Cohort event study">
<p>Pre-start months (&minus;3, &minus;2): {ev['-3']:.2f}, {ev['-2']:.2f} &mdash; essentially
zero, as they should be. Post-start months: {ev['0']:.2f}, {ev['1']:.2f},
{ev['2']:.2f}, {ev['3']:.2f} &mdash; hovering around the true &minus;3.0.
<span class="formal">[Event-study plot.]</span> One caution from the fine print:
flat pre-trends are <em>reassuring</em>, not <em>proof</em> &mdash; they can't show
you what would have happened after.</p>

<h2>Placebo: the method must stay quiet when nothing happened</h2>
<p>Take the 40 never-treated patients, randomly label 20 of them "fake treated,"
and run the whole machine. Repeat 30 times. Average fake effect:
<strong>{res['placebo_mean']:.2f} kg</strong> (truth: 0). It finds nothing, as it
should. <span class="formal">[Placebo / falsification test.]</span></p>

<h2>What about one big regression?</h2>
<p>The naive approach &mdash; one regression with patient and month adjustments and a
single "on the drug" switch <span class="formal">[two-way fixed effects,
TWFE]</span> &mdash; gives <strong>{res['twfe']:.2f} kg</strong> here, close to the
honest answer because our planted effects are steady. Don't let that comfort you:
inside, it still mixes already-treated patients into its comparisons (the trap
above), and when effects grow or fade over time it can distort badly. The
cohort-by-cohort version never lets that happen, because its control group is
always genuinely untreated.</p>

<h2>When even the good version fails</h2>
<p>Suppose doctors prescribed the drug to the patients whose weight was
<em>spiking</em> — the worst-off got treated first. Then the treated groups were
already on a different trajectory than the never-treated, and no estimator, however
careful, can fix that: the comparison was broken at assignment.
<span class="formal">[Selection on the outcome; the parallel-trends assumption
fails.]</span> Same if patients started dieting the month <em>before</em> their
prescription (the "baseline" is already affected)
<span class="formal">[anticipation effects]</span>. Staggered DiD fixes the
<em>timing</em> problem; it cannot fix a <em>selection</em> problem. Always ask how
people ended up treated.</p>

<h2>Limitations, honestly</h2>
<ul>
<li><strong>You need never-treated (or not-yet-treated) people.</strong> If everyone
eventually takes the drug, the comparison runs out — you can only study horizons
where a valid control group still exists.</li>
<li><strong>Keep the comparison window common.</strong> Don't compare month 6
(only January starters observed that long) against month 1 (everyone observed) and
call the difference "the effect growing" — the <em>mix of patients</em> changed,
not necessarily the effect.</li>
<li><strong>Synthetic is clean; reality isn't.</strong> Real prescribing has messy
timing, partial adherence, and spillovers (partners dieting together). The machinery
is the same; the vigilance has to be higher.</li>
</ul>

<div class="boss"><strong>The 30-second version for your boss.</strong><br><br>
"We wanted to know if the weight-loss drug worked, but patients started it in
different months, so there's no single 'before and after.' We lined everyone up on
<em>months since they started</em> instead of calendar month, compared each
start-month group against patients who never took it, and averaged the results:
about 3 kg of weight loss. The key discipline: never use already-treated patients
as the comparison group — their weight already reflects the drug, and doing so can
make a working drug look harmful. The one judgment call, as always, is that the
groups would have trended together without the drug — which is about <em>why</em>
people started it when they did, not something a chart can prove."</div>

<div class="proveit"><strong>Prove it.</strong> The simulation script
(<code>scripts/did_staggered.py</code>) plants the effects, so every number above
is checkable against the answer key in
<code>output/did_staggered_results.json</code>. The hand-worked 6-patient tables
can be verified with a calculator.</div>
</body></html>
"""

out = STUDY / "did-staggered-case-study.html"
out.write_text(html)
print(f"written: {out} ({len(html)//1024} KB)")
