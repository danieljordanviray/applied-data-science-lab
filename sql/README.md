# SQL

Quick-reference notes: one technique per page, each with a small clinical toy
example and queries verified by actually running them (SQLite). Written for
future-me first — the version I want to find when I'm stuck — and for anyone
learning these for the first time.

- [EXISTS vs IN / NOT IN](exists-vs-in-not-in/exists-vs-in-not-in.html) — and the NULL trap
- [COALESCE and NULLIF](coalesce-nullif/coalesce-nullif.html)
- [Window Functions](window-functions/window-functions.html) — rolling averages, running totals
- [Date Spines and CROSS JOINs](date-spines/date-spines.html)
- [CTEs and Subqueries](ctes-subqueries/ctes-subqueries.html)

Rebuild the pages from verified queries: `python3 sql/scripts/build_sql_notes.py`
(requires only Python's standard library).
