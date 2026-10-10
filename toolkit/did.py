"""Code bank: difference-in-differences recipes.

Added Oct 10, 2026 -- authorized by Daniel after reviewing the case study.
Each function is one checklist item from the DiD workflow.

Verified in scripts/verify_did_bank.py: did_2x2 and did_regression reproduce
the case-study numbers exactly; placebo_did finds nothing on pre-period data.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


def did_2x2(df, treated_col, post_col, y_col):
    """The 2x2 table of means and the difference-in-differences estimate.

    Checklist: group the data into treated/control x before/after,
    take each group's change, subtract the control change from the
    treated change. The control change is the stand-in for what would
    have happened to the treated group without treatment [counterfactual].
    """
    t = df.groupby([treated_col, post_col])[y_col].mean().unstack()
    a_pre, a_post = t.loc[True, False], t.loc[True, True]
    b_pre, b_post = t.loc[False, False], t.loc[False, True]
    return {"means": {"treated_pre": float(a_pre), "treated_post": float(a_post),
                      "control_pre": float(b_pre), "control_post": float(b_post)},
            "treated_change": float(a_post - a_pre),
            "control_change": float(b_post - b_pre),
            "estimate": float((a_post - a_pre) - (b_post - b_pre))}


def did_regression(df, treated_col, post_col, y_col, alpha=0.05):
    """DiD via regression: Y ~ treated + post + treated:post.

    The interaction coefficient [Treat x Post, beta_3] IS the DiD estimate,
    packaged with a standard error, confidence interval, and p-value.
    """
    f = f"{y_col} ~ {treated_col} + {post_col} + {treated_col}:{post_col}"
    fit = smf.ols(f, data=df).fit()
    inter = [c for c in fit.params.index if ":" in c][0]
    ci = fit.conf_int(alpha=alpha).loc[inter].tolist()
    return {"estimate": float(fit.params[inter]),
            "se": float(fit.bse[inter]),
            "ci": [float(ci[0]), float(ci[1])],
            "p": float(fit.pvalues[inter]),
            "interaction_term": inter}


def placebo_did(df, treated_col, fake_post_col, y_col):
    """Falsification test: run DiD where we KNOW nothing happened.

    Pretend treatment started earlier (or on units never treated) and check
    the method finds ~0. If it hallucinates an effect here, don't trust it
    where it matters.
    """
    r = did_regression(df, treated_col, fake_post_col, y_col)
    return {"placebo_estimate": r["estimate"], "p": r["p"]}


def plot_group_trends(df, group_col, time_col, y_col, path,
                      vline=None, vline_label=None, ylabel=None, title=None):
    """The eyeball test: mean outcome by group over time [parallel trends plot]."""
    piv = df.groupby([group_col, time_col])[y_col].mean().unstack(group_col)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for g, color in zip(piv.columns, ["#d1242f", "#1f6feb", "#2a7d2a", "#9467bd"]):
        ax.plot(piv.index, piv[g], marker="o", label=str(g), color=color)
    if vline is not None:
        ax.axvline(vline, color="gray", linestyle="--", label=vline_label)
    ax.set_xlabel(time_col); ax.set_ylabel(ylabel or y_col)
    if title:
        ax.set_title(title)
    ax.legend(); ax.grid(alpha=0.25)
    fig.savefig(path); plt.close(fig)
    return str(path)
