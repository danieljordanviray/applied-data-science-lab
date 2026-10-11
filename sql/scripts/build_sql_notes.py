#!/usr/bin/env python3
"""Build the SQL notes pages.

Every query on every page is executed against SQLite and its output asserted
before the HTML is written. The tables readers see are actual query output.
"""
import sqlite3
from pathlib import Path
from html import escape

SQL = Path(__file__).resolve().parent.parent  # sql/
con = sqlite3.connect(":memory:")
cur = con.cursor()

STYLE = """<style>
body { font-family: -apple-system, Georgia, serif; max-width: 720px; margin: 0 auto;
       padding: 24px 18px; line-height: 1.7; color: #1a1a1a; }
h1 { font-size: 1.6em; } h2 { margin-top: 2em; border-bottom: 1px solid #ddd; padding-bottom: 6px; }
h3 { margin-top: 1.6em; }
.topnav { font-size: .85rem; margin: 0 0 1.2em; }
.topnav a { color: #1f6feb; text-decoration: none; }
.watchout { background: #fffbeb; border-left: 4px solid #d97706; padding: 12px 16px; margin: 1.5em 0; }
.watchout strong { color: #92400e; }
.proveit { background: #eef7ee; border-left: 4px solid #2a7d2a; padding: 12px 16px; margin: 1.5em 0; }
table { border-collapse: collapse; font-size: 0.88em; margin: 1em 0; }
th, td { border: 1px solid #ccc; padding: 6px 12px; text-align: left; }
th { background: #f0f0f0; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
.small { color: #555; font-size: 0.9em; }
code { background: #f0f0f0; padding: 1px 5px; border-radius: 4px; font-size: 0.88em; }
pre { background: #f4f4f4; padding: 12px; border-radius: 6px; overflow-x: auto; font-size: 0.82em; }
.null { color: #999; font-style: italic; }
</style>"""

NAV = '<div class="topnav"><a href="../../docs/index.html">\u2190 Homepage</a></div>'


def run(sql, params=()):
    cur.execute(sql, params)
    cols = [d[0] for d in cur.description]
    return cols, cur.fetchall()


def cell(v):
    if v is None:
        return '<td class="null">NULL</td>'
    if isinstance(v, float):
        return f'<td class="num">{v}</td>'
    return f'<td>{escape(str(v))}</td>'


def table(cols, rows, caption=None):
    h = ""
    if caption:
        h += f'<p class="small">{caption}</p>\n'
    h += "<table>\n<tr>" + "".join(f"<th>{escape(c)}</th>" for c in cols) + "</tr>\n"
    for r in rows:
        h += "<tr>" + "".join(cell(v) for v in r) + "</tr>\n"
    return h + "</table>\n"


def watchout(items):
    lis = "".join(f"<li>{it}</li>" for it in items)
    return f'<div class="watchout"><strong>Watch out.</strong><ul>{lis}</ul></div>\n'


def page(slug, title, subtitle, body):
    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(title)} — SQL notes</title>
{STYLE}</head><body>
{NAV}
<h1>{escape(title)}</h1>
<p class="small">{subtitle}</p>
{body}
<div class="proveit"><strong>Prove it.</strong> Every query on this page was run
against SQLite and the tables show actual output — not hand-typed guesses.</div>
</body></html>
"""
    out = SQL / slug / f"{slug}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"written: {out} ({len(html)//1024} KB)")


def check(name, got, want):
    assert got == want, f"{name}: got {got}, want {want}"
    print(f"  ok: {name}")


# ---------------------------------------------------------------- toy data
cur.executescript("""
CREATE TABLE patients (patient_id INTEGER PRIMARY KEY, name TEXT);
INSERT INTO patients VALUES
  (1,'Alice'),(2,'Bob'),(3,'Carol'),(4,'Dave');

CREATE TABLE prescriptions (rx_id INTEGER PRIMARY KEY, patient_id INTEGER, drug TEXT);
INSERT INTO prescriptions VALUES
  (101, 1, 'atorvastatin'),
  (102, 2, 'metformin'),
  (103, NULL, 'atorvastatin'),
  (104, 1, 'lisinopril');

CREATE TABLE labs (patient_id INTEGER, test_date TEXT, hba1c REAL);
INSERT INTO labs VALUES
  (1,'2024-01-05',8.2),
  (2,'2024-02-11',NULL),
  (3,'2024-03-01',7.1);

CREATE TABLE doses (patient_id INTEGER, dose_mg REAL, weight_kg REAL);
INSERT INTO doses VALUES
  (1,500,80),
  (2,850,0),
  (3,1000,95);

CREATE TABLE a1c_readings (patient_id INTEGER, reading_date TEXT, hba1c REAL);
INSERT INTO a1c_readings VALUES
  (1,'2024-01-05',8.2),(1,'2024-04-02',7.9),(1,'2024-07-10',7.9),(1,'2024-10-01',7.4),
  (2,'2024-02-11',9.1),(2,'2024-05-20',8.8);

CREATE TABLE med_admin (patient_id INTEGER, admin_date TEXT, dose_mg REAL);
INSERT INTO med_admin VALUES
  (1,'2024-01-05',500),(1,'2024-02-05',500),(1,'2024-03-05',250),
  (2,'2024-02-11',850);

CREATE TABLE ed_visits (visit_date TEXT, department TEXT);
INSERT INTO ed_visits VALUES
  ('2024-01-02','ER'),('2024-01-02','ER'),
  ('2024-01-04','Cardiology'),
  ('2024-01-07','ER');
""")

# ================================================================ page 1
cols, rows = run("SELECT patient_id, name FROM patients ORDER BY patient_id")
t_patients = table(cols, rows, "patients")
cols, rows = run("SELECT rx_id, patient_id, drug FROM prescriptions ORDER BY rx_id")
t_rx = table(cols, rows, "prescriptions — note rx 103: paper script, patient never linked (NULL)")

q1 = """SELECT name FROM patients p
WHERE EXISTS (SELECT 1 FROM prescriptions r WHERE r.patient_id = p.patient_id)
ORDER BY name;"""
cols, r1 = run(q1)
check("exists", r1, [("Alice",), ("Bob",)])

q2 = """SELECT name FROM patients
WHERE patient_id IN (SELECT patient_id FROM prescriptions)
ORDER BY name;"""
cols, r2 = run(q2)
check("in", r2, [("Alice",), ("Bob",)])

q3 = """SELECT name FROM patients
WHERE patient_id NOT IN (SELECT patient_id FROM prescriptions)
ORDER BY name;"""
cols, r3 = run(q3)
check("not-in trap (empty!)", r3, [])

q4 = """SELECT name FROM patients
WHERE patient_id NOT IN (SELECT patient_id FROM prescriptions WHERE patient_id IS NOT NULL)
ORDER BY name;"""
cols, r4 = run(q4)
check("not-in fixed", r4, [("Carol",), ("Dave",)])

q5 = """SELECT name FROM patients p
WHERE NOT EXISTS (SELECT 1 FROM prescriptions r WHERE r.patient_id = p.patient_id)
ORDER BY name;"""
cols, r5 = run(q5)
check("not exists", r5, [("Carol",), ("Dave",)])

page("exists-vs-in-not-in", "EXISTS vs IN / NOT IN",
     "SQL notes · clinical toy data · the NULL trap that empties your result",
f"""
<p><strong>The idea.</strong> Three ways to ask "which patients have a matching row
in another table?" <code>EXISTS</code> and <code>IN</code> usually agree. But
<code>NOT IN</code> has a trap: a single <code>NULL</code> in the subquery list
silently wipes out the <em>entire</em> result — no error, no warning.</p>

<h2>Toy data</h2>
{t_patients}
{t_rx}

<h2>Who has at least one prescription?</h2>
<h3>With EXISTS — "is there at least one matching row?"</h3>
<pre>{escape(q1)}</pre>
{table(*run(q1))}
<p>Read it as a per-patient question: for Alice, is there <em>at least one</em>
prescription row pointing at her? Yes → keep her. Correlated, but the optimizer
handles it.</p>

<h3>With IN — "is the id in this list?"</h3>
<pre>{escape(q2)}</pre>
{table(*run(q2))}
<p>Same answer. The <code>NULL</code> in the list doesn't hurt here: Alice and Bob
match real values, and matching is all <code>IN</code> needs.</p>

<h2>Who has <em>no</em> prescription? — the trap</h2>
<pre>{escape(q3)}</pre>
{table(*run(q3), "empty — but Carol and Dave have no prescriptions!")}
<p><strong>Why empty?</strong> For Carol the database evaluates
<code>3 NOT IN (1, 2, NULL, 1)</code>, i.e.
<code>3&lt;&gt;1 AND 3&lt;&gt;2 AND 3&lt;&gt;NULL AND 3&lt;&gt;1</code> →
<code>TRUE AND TRUE AND UNKNOWN AND TRUE</code> = <code>UNKNOWN</code> — and
<code>WHERE</code> keeps only <code>TRUE</code> rows. One <code>NULL</code>
poisons <em>every</em> candidate. In real pipelines the NULL arrives via ETL gaps,
optional foreign keys, or half-linked records like rx 103.</p>

<h3>Fix 1: filter the NULLs out of the list</h3>
<pre>{escape(q4)}</pre>
{table(*run(q4))}

<h3>Fix 2: NOT EXISTS — NULL-proof by construction</h3>
<pre>{escape(q5)}</pre>
{table(*run(q5))}
<p><code>NOT EXISTS</code> never compares against the <code>NULL</code> as a value;
it only asks whether a matching row exists. When the column is nullable, prefer it.</p>
""" + watchout([
  "<code>NOT IN</code> + any <code>NULL</code> in the subquery = silently empty result. The #1 real-world gotcha.",
  "<code>IN</code> with a <code>NULL</code> in the list is safe for <em>matches</em> — the asymmetry between <code>IN</code> and <code>NOT IN</code> is the lesson.",
  "If a <code>NOT IN</code> query returns nothing and you expected rows, check the subquery for <code>NULL</code>s first.",
]))

# ================================================================ page 2
q1 = """SELECT patient_id,
       COALESCE(CAST(hba1c AS TEXT), 'pending') AS result
FROM labs ORDER BY patient_id;"""
cols, r1 = run(q1)
check("coalesce display", r1, [(1, "8.2"), (2, "pending"), (3, "7.1")])

q2 = """SELECT ROUND(AVG(hba1c), 2) AS avg_observed,
       ROUND(AVG(COALESCE(hba1c, 0)), 2) AS avg_zero_filled
FROM labs;"""
cols, r2 = run(q2)
check("coalesce avg", r2, [(7.65, 5.1)])

q3 = """SELECT patient_id,
       ROUND(dose_mg / NULLIF(weight_kg, 0), 2) AS mg_per_kg
FROM doses ORDER BY patient_id;"""
cols, r3 = run(q3)
check("nullif", r3, [(1, 6.25), (2, None), (3, 10.53)])

cols, rl = run("SELECT patient_id, test_date, hba1c FROM labs ORDER BY patient_id")
t_labs = table(cols, rl, "labs — patient 2's result never came back (NULL)")
cols, rd = run("SELECT patient_id, dose_mg, weight_kg FROM doses ORDER BY patient_id")
t_doses = table(cols, rd, "doses — patient 2's weight recorded as 0 (missing-data placeholder)")

page("coalesce-nullif", "COALESCE and NULLIF",
     "SQL notes · clinical toy data · defaults for NULLs, and taming placeholder zeros",
f"""
<p><strong>The idea.</strong> <code>COALESCE</code> returns the first non-<code>NULL</code>
value in its list — for defaults and display. <code>NULLIF(a, b)</code> returns
<code>NULL</code> when <code>a = b</code> — for turning placeholder values (like a
<code>0</code> that means "unknown") back into real <code>NULL</code>s so they don't
corrupt your math.</p>

<h2>Toy data</h2>
{t_labs}
{t_doses}

<h2>COALESCE: fill the gap for display</h2>
<pre>{escape(q1)}</pre>
{table(*run(q1))}
<p>Reads left to right, stops at the first non-<code>NULL</code>. The lab value if
present, the word "pending" otherwise.</p>

<h2>COALESCE changes your averages — know which question you're answering</h2>
<pre>{escape(q2)}</pre>
{table(*run(q2))}
<p><code>AVG(hba1c)</code> ignores <code>NULL</code>s (denominator 2 → 7.65: the mean
of <em>observed</em> results). <code>AVG(COALESCE(hba1c, 0))</code> fills the gap
first (denominator 3 → 5.10: the mean <em>assuming missing = 0</em>). Neither is
"right" — but they answer different questions, and filling blindly can manufacture
a worse number than the honest one.</p>

<h2>NULLIF: turn a placeholder zero back into unknown</h2>
<pre>{escape(q3)}</pre>
{table(*run(q3))}
<p>Patient 2's weight of <code>0</code> isn't a real weight — it's how "unknown" got
typed in. Without <code>NULLIF</code> this query divides by zero (error, or worse,
an <code>Infinity</code> that flows downstream). <code>NULLIF(weight_kg, 0)</code>
converts the placeholder back into what it means: unknown → the result is
<code>NULL</code>, honestly.</p>
""" + watchout([
  "Don't <code>COALESCE</code> away <code>NULL</code>s before you understand <em>why</em> they're <code>NULL</code> — a missing lab and a lab of 0 are different facts.",
  "<code>COALESCE(a, b)</code> is standard SQL (<code>IFNULL</code>/<code>NVL</code> are dialect spellings of the two-argument case).",
  "<code>NULLIF</code> also guards rate calculations: <code>events / NULLIF(days, 0)</code> never divides by zero.",
]))


# ================================================================ page 3
SYNTAX = """SELECT patient_id, reading_date, hba1c,
  FUNC(...) OVER (
    PARTITION BY patient_id   -- restart per patient ("per category")
    ORDER BY reading_date     -- the sequence within each group
    ROWS BETWEEN 2 PRECEDING AND CURRENT ROW  -- the frame: which rows count
  )
FROM a1c_readings;"""

q1 = """SELECT patient_id, reading_date,
       ROW_NUMBER() OVER (PARTITION BY patient_id ORDER BY reading_date) AS visit_n
FROM a1c_readings ORDER BY patient_id, reading_date;"""
cols, r1 = run(q1)
check("row_number", [r[2] for r in r1], [1, 2, 3, 4, 1, 2])

q2 = """SELECT patient_id, hba1c,
       RANK() OVER (PARTITION BY patient_id ORDER BY hba1c DESC) AS rnk,
       DENSE_RANK() OVER (PARTITION BY patient_id ORDER BY hba1c DESC) AS dense_rnk
FROM a1c_readings ORDER BY patient_id, hba1c DESC;"""
cols, r2 = run(q2)
check("rank", [(r[1], r[2], r[3]) for r in r2],
      [(8.2, 1, 1), (7.9, 2, 2), (7.9, 2, 2), (7.4, 4, 3), (9.1, 1, 1), (8.8, 2, 2)])

q3 = """SELECT patient_id, reading_date, hba1c,
       ROUND(hba1c - LAG(hba1c) OVER (PARTITION BY patient_id ORDER BY reading_date), 2)
         AS change_since_last
FROM a1c_readings ORDER BY patient_id, reading_date;"""
cols, r3 = run(q3)
check("lag", [r[3] for r in r3], [None, -0.3, 0.0, -0.5, None, -0.3])

q4 = """SELECT patient_id, reading_date, hba1c,
       ROUND(AVG(hba1c) OVER (PARTITION BY patient_id ORDER BY reading_date
                              ROWS BETWEEN 2 PRECEDING AND CURRENT ROW), 2)
         AS rolling_avg_3,
       COUNT(*) OVER (PARTITION BY patient_id ORDER BY reading_date
                      ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS n_in_avg
FROM a1c_readings ORDER BY patient_id, reading_date;"""
cols, r4 = run(q4)
check("rolling avg", [(r[3], r[4]) for r in r4],
      [(8.2, 1), (8.05, 2), (8.0, 3), (7.73, 3), (9.1, 1), (8.95, 2)])

q5 = """SELECT patient_id, admin_date,
       SUM(dose_mg) OVER (PARTITION BY patient_id ORDER BY admin_date
                          ROWS UNBOUNDED PRECEDING) AS running_total_mg
FROM med_admin ORDER BY patient_id, admin_date;"""
cols, r5 = run(q5)
check("running total", [r[2] for r in r5], [500.0, 1000.0, 1250.0, 850.0])

cols, rr = run("SELECT patient_id, reading_date, hba1c FROM a1c_readings ORDER BY patient_id, reading_date")
t_a1c = table(cols, rr, "a1c_readings — HbA1c lab readings per patient over time")
cols, rm = run("SELECT patient_id, admin_date, dose_mg FROM med_admin ORDER BY patient_id, admin_date")
t_med = table(cols, rm, "med_admin — medication doses administered")

page("window-functions", "Window Functions",
     "SQL notes · clinical toy data · per-group calculations without collapsing rows",
f"""
<p><strong>The idea.</strong> Aggregates like <code>AVG</code> collapse rows; window
functions don't. Each row keeps its identity while the function peers at a
<em>window</em> of related rows — the patient's other visits, the previous reading,
a running total. One query, row-level detail <em>and</em> group math.</p>

<h2>The shape (learn this once, reuse forever)</h2>
<pre>{escape(SYNTAX)}</pre>
<p><code>PARTITION BY</code> is "per category": restart the calculation for each
patient (or ward, or drug). <code>ORDER BY</code> defines the sequence. The frame
(<code>ROWS BETWEEN …</code>) says which rows count — omit it and you get the
default, which is rarely what you want for rolling math.</p>

<h2>Toy data</h2>
{t_a1c}
{t_med}

<h2>Visit number per patient — ROW_NUMBER()</h2>
<pre>{escape(q1)}</pre>
{table(*run(q1))}

<h2>Rank readings within each patient — RANK() vs DENSE_RANK()</h2>
<pre>{escape(q2)}</pre>
{table(*run(q2))}
<p>Patient 1 has two readings of 7.9 (a tie). <code>RANK()</code> gives both rank 2
then <em>skips</em> to 4; <code>DENSE_RANK()</code> doesn't skip (2, 2, 3). Pick
based on whether the skipped numbers matter downstream.</p>

<h2>Change since last visit — LAG() (and LEAD() looks forward)</h2>
<pre>{escape(q3)}</pre>
{table(*run(q3))}
<p><code>LAG(hba1c)</code> is "the previous row's value, within this patient's
sequence." The first row per patient has no previous row → <code>NULL</code>,
honestly. <code>LEAD()</code> is the mirror image (next row).</p>

<h2>Rolling 3-reading average, per patient</h2>
<pre>{escape(q4)}</pre>
{table(*run(q4))}
<p>The frame <code>ROWS BETWEEN 2 PRECEDING AND CURRENT ROW</code> is the whole
trick: "this reading and the two before it." The <code>hba1c</code> column shows
exactly which values went into each average, and <code>n_in_avg</code> makes the
edge effect visible: the first reading averages just 1 value, the second averages
2 — only from the third row on is it a true 3-reading average. The frame simply
has fewer rows to work with early on; no error, but worth knowing when you read
the first few values.</p>

<h2>Running total of dose, per patient</h2>
<pre>{escape(q5)}</pre>
{table(*run(q5))}
<p><code>ROWS UNBOUNDED PRECEDING</code>: "everything from the patient's first row
up to this one." Cumulative dose, cumulative cost, cumulative events — same shape.</p>
""" + watchout([
  "<code>ORDER BY</code> inside <code>OVER</code> defines row order — without it, <code>ROW_NUMBER()</code> is arbitrary.",
  "<code>ROWS</code> counts physical rows; <code>RANGE</code> groups tied values. Say <code>ROWS</code> when you mean rows.",
  "Forget <code>PARTITION BY</code> and your 'per-patient' running total silently runs across <em>all</em> patients.",
  "Window functions run <em>after</em> <code>WHERE</code>/<code>GROUP BY</code> — you can't put one in a <code>WHERE</code> clause; wrap it in a subquery or CTE first.",
]))

# ================================================================ page 4
SPINE = """WITH RECURSIVE spine(d) AS (
  SELECT date('2024-01-01')
  UNION ALL
  SELECT date(d, '+1 day') FROM spine WHERE d < '2024-01-31'
)"""

q1 = """SELECT visit_date AS day, COUNT(*) AS visits
FROM ed_visits GROUP BY visit_date ORDER BY day;"""
cols, r1 = run(q1)
check("naive group-by", r1, [("2024-01-02", 2), ("2024-01-04", 1), ("2024-01-07", 1)])

cols, r2 = run(SPINE + "\nSELECT d FROM spine;")
check("spine 31 days", (len(r2), r2[0][0], r2[-1][0]), (31, "2024-01-01", "2024-01-31"))

q3 = SPINE + """,
depts AS (SELECT DISTINCT department AS dept FROM ed_visits)
SELECT s.d, x.dept FROM spine s CROSS JOIN depts x ORDER BY s.d, x.dept;"""
cols, r3 = run(q3)
check("grid 62 rows", len(r3), 62)

q4 = SPINE + """,
depts AS (SELECT DISTINCT department AS dept FROM ed_visits)
SELECT s.d AS day, x.dept AS department, COUNT(v.visit_date) AS visits
FROM spine s CROSS JOIN depts x
LEFT JOIN ed_visits v ON v.visit_date = s.d AND v.department = x.dept
GROUP BY s.d, x.dept ORDER BY s.d, x.dept;"""
cols, r4 = run(q4)
check("spine rows", len(r4), 62)
nonzero = [r for r in r4 if r[2] > 0]
check("nonzero cells", nonzero,
      [("2024-01-02", "ER", 2), ("2024-01-04", "Cardiology", 1), ("2024-01-07", "ER", 1)])
check("a zero day", [r for r in r4 if r[0] == "2024-01-03"],
      [("2024-01-03", "Cardiology", 0), ("2024-01-03", "ER", 0)])

cols, rv = run("SELECT visit_date, department FROM ed_visits ORDER BY visit_date")
t_vis = table(cols, rv, "ed_visits — January has 31 days; only 3 of them have visits")

# trimmed display: first 8 rows + the 3 nonzero rows, with an omission note
shown = [r for r in r4 if r[0] <= "2024-01-05"]
trim_note = '<p class="small">… 54 quiet rows omitted …</p>'
t4 = table(cols, shown) + trim_note + table(cols, nonzero, "the 3 days with visits")

page("date-spines", "Date Spines and CROSS JOINs",
     "SQL notes · clinical toy data · every day on the calendar, zeros included",
f"""
<p><strong>The idea.</strong> Real event tables have gaps — no row exists for days
nothing happened. A plain <code>GROUP BY day</code> then <em>lies by omission</em>:
quiet days vanish instead of showing zero. The fix is a <strong>date spine</strong>
(a complete calendar you build yourself), optionally crossed with your categories,
with the events <code>LEFT JOIN</code>ed on.</p>

<h2>Toy data</h2>
{t_vis}

<h2>The naive version — and what it hides</h2>
<pre>{escape(q1)}</pre>
{table(*run(q1))}
<p>Three rows. The other 28 days of January simply don't exist here — any chart drawn
from this is missing most of the month.</p>

<h2>Step 1: build the spine (recursive CTE)</h2>
<pre>{escape(SPINE + "\\nSELECT d FROM spine;  -- 31 rows")}</pre>
<p>Start at Jan 1, keep adding a day until Jan 31. (Dialect note: BigQuery spells this
<code>GENERATE_DATE_ARRAY</code>; the idea is identical.)</p>

<h2>Step 2: CROSS JOIN the grid — every day × every department</h2>
<pre>{escape(q3)}</pre>
<p><code>CROSS JOIN</code> is the full multiplication: 31 days × 2 departments =
62 rows. This grid is the skeleton your facts hang on.</p>

<h2>Step 3: LEFT JOIN the events onto the grid</h2>
<pre>{escape(q4)}</pre>
{t4}
""" + watchout([
  "<code>COUNT(v.visit_date)</code>, not <code>COUNT(*)</code> — after a <code>LEFT JOIN</code>, <code>COUNT(*)</code> counts the empty joined rows as 1 and your zeros become ones.",
  "Put the category match (<code>v.department = x.dept</code>) in the <code>ON</code> clause, not <code>WHERE</code> — <code>WHERE</code> filters <em>after</em> the join and kills your zero rows.",
  "<code>CROSS JOIN</code> multiplies: sanity-check the grid size (days × categories) before joining facts onto it.",
]))

# ================================================================ page 5
CTE = """WITH latest AS (
  SELECT patient_id, MAX(reading_date) AS d
  FROM a1c_readings GROUP BY patient_id
),
latest_vals AS (
  SELECT r.patient_id, r.hba1c
  FROM a1c_readings r
  JOIN latest l ON r.patient_id = l.patient_id AND r.reading_date = l.d
)
SELECT patient_id, hba1c FROM latest_vals WHERE hba1c > 8.0 ORDER BY patient_id;"""
cols, r_cte = run(CTE)
check("cte", r_cte, [(2, 8.8)])

SUB = """SELECT r.patient_id, r.hba1c
FROM a1c_readings r
JOIN (SELECT patient_id, MAX(reading_date) AS d
      FROM a1c_readings GROUP BY patient_id) l
  ON r.patient_id = l.patient_id AND r.reading_date = l.d
WHERE r.hba1c > 8.0 ORDER BY r.patient_id;"""
cols, r_sub = run(SUB)
check("subquery matches cte", r_sub, r_cte)

page("ctes-subqueries", "CTEs and Subqueries",
     "SQL notes · clinical toy data · same question, two shapes — readability is the difference",
f"""
<p><strong>The idea.</strong> A <strong>CTE</strong> (<code>WITH …</code>) names an
intermediate result up front; a <strong>subquery</strong> nests it inline. Same
question, same answer, usually the same query plan — the difference is how a human
reads it six months later.</p>

<p><strong>The question</strong> (reusing the HbA1c readings from the window-functions
page): which patients' <em>most recent</em> HbA1c exceeds 8.0?</p>

<h2>As CTEs — read top to bottom</h2>
<pre>{escape(CTE)}</pre>
{table(*run(CTE))}
<p><code>latest</code>: each patient's most recent reading date.
<code>latest_vals</code>: the actual values on those dates. Then filter.
Each step has a name; the query reads like a recipe.</p>

<h2>As a subquery — read inside out</h2>
<pre>{escape(SUB)}</pre>
{table(*run(SUB))}
<p>Identical result. But you parse it backwards: find the innermost query first,
then work outward. Fine for one nesting — miserable at three.</p>
""" + watchout([
  "Name CTEs after what they contain (<code>latest</code>, not <code>cte1</code>) — future-you will thank present-you.",
  "A CTE is a <em>named subquery</em>, not a temp table: some engines re-evaluate it per reference. Fine at notes scale.",
  "Reach for a CTE when a step deserves a name or gets reused; a one-off inline filter is fine as a subquery.",
  "In old Postgres (&lt;12) CTEs were optimization fences — the planner couldn't flatten them. Rarely matters, good to know.",
]))

print("ALL PAGES BUILT AND VERIFIED")
