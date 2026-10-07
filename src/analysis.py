"""Experiment analysis: KPIs, statistical tests, segmentation and business impact.

Every function takes the session-level experiment DataFrame (one row per session)
and returns a tidy DataFrame or dict, so results can be shown in the notebook,
written to CSV, or tested.

Metric definitions
------------------
* Conversion rate      = orders / sessions
* AOV                  = net revenue / orders            (converted sessions only)
* Revenue per session  = net revenue / sessions          (non-converting sessions count as 0)
* Profit per session   = contribution profit / sessions  (primary decision metric)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize, proportions_ztest

from src import config as C


def load_experiment(path=C.EXPERIMENT_CSV) -> pd.DataFrame:
    return pd.read_csv(path)


# --------------------------------------------------------------------------- #
# KPIs
# --------------------------------------------------------------------------- #
def _kpis(g: pd.DataFrame) -> pd.Series:
    sessions = len(g)
    orders = int(g["converted"].sum())
    gross = g["gross_order_value_inr"].sum()
    return pd.Series({
        "sessions": sessions,
        "orders": orders,
        "conversion": orders / sessions,
        "aov_inr": g.loc[g["converted"] == 1, "net_revenue_inr"].mean(),
        "units_per_order": g.loc[g["converted"] == 1, "units"].mean(),
        "discount_rate": g["discount_inr"].sum() / gross if gross > 0 else np.nan,
        "revenue_per_session_inr": g["net_revenue_inr"].mean(),
        "profit_per_session_inr": g["contribution_profit_inr"].mean(),
    })


def variant_summary(df: pd.DataFrame) -> pd.DataFrame:
    """KPI table by variant, with relative lift versus control."""
    out = df.groupby("variant").apply(_kpis, include_groups=False)
    out[["sessions", "orders"]] = out[["sessions", "orders"]].astype(int)
    base = out.loc[C.CONTROL]
    for col, name in [("conversion", "conv_lift"), ("aov_inr", "aov_lift"),
                      ("revenue_per_session_inr", "rps_lift"), ("profit_per_session_inr", "profit_lift")]:
        out[name] = out[col] / base[col] - 1
    return out


def segment_summary(df: pd.DataFrame) -> pd.DataFrame:
    """KPI table by customer segment x variant (watch the session counts!)."""
    out = df.groupby(["customer_segment", "variant"]).apply(_kpis, include_groups=False)
    out[["sessions", "orders"]] = out[["sessions", "orders"]].astype(int)
    return out


# --------------------------------------------------------------------------- #
# Validity checks
# --------------------------------------------------------------------------- #
def sample_ratio_mismatch(df: pd.DataFrame) -> dict:
    """Chi-square goodness-of-fit: were sessions split evenly across variants?"""
    counts = df["variant"].value_counts().sort_index()
    chi2, p = stats.chisquare(counts)
    return {"counts": counts.to_dict(), "chi2": chi2, "p_value": p, "srm_detected": p < 0.001}


def minimum_detectable_effect(n_per_arm: int, baseline_rate: float,
                              alpha: float = C.ALPHA, power: float = 0.80) -> float:
    """Smallest absolute conversion change detectable with this sample size."""
    h = NormalIndPower().solve_power(nobs1=n_per_arm, alpha=alpha, power=power, ratio=1.0)
    # invert Cohen's h: h = 2*asin(sqrt(p2)) - 2*asin(sqrt(p1))
    p2 = np.sin(np.arcsin(np.sqrt(baseline_rate)) + h / 2) ** 2
    return p2 - baseline_rate


# --------------------------------------------------------------------------- #
# Statistical tests
# --------------------------------------------------------------------------- #
def chi_square_conversion(df: pd.DataFrame) -> dict:
    """H0: conversion is independent of variant."""
    table = pd.crosstab(df["variant"], df["converted"])
    chi2, p, dof, _ = stats.chi2_contingency(table)
    return {"test": "Chi-square: conversion vs variant", "statistic": chi2, "dof": dof, "p_value": p}


def anova_aov(df: pd.DataFrame) -> dict:
    """H0: mean order value (net revenue, converted sessions) is equal across variants."""
    orders = df[df["converted"] == 1]
    groups = [g["net_revenue_inr"].values for _, g in orders.groupby("variant")]
    f, p = stats.f_oneway(*groups)
    return {"test": "One-way ANOVA: order value", "statistic": f, "p_value": p}


def tukey_aov(df: pd.DataFrame) -> pd.DataFrame:
    """Tukey HSD pairwise comparison of order value (controls family-wise error)."""
    orders = df[df["converted"] == 1]
    res = pairwise_tukeyhsd(orders["net_revenue_inr"], orders["variant"], alpha=C.ALPHA)
    table = pd.DataFrame(res.summary().data[1:], columns=res.summary().data[0])
    return table.rename(columns={"meandiff": "mean_diff_inr", "p-adj": "p_adj", "reject": "significant"})


def conversion_vs_control(df: pd.DataFrame) -> pd.DataFrame:
    """Two-proportion z-test of each treatment against control, Holm-adjusted."""
    ctrl = df[df["variant"] == C.CONTROL]["converted"]
    rows = []
    for v, g in df[df["variant"] != C.CONTROL].groupby("variant"):
        z, p = proportions_ztest([g["converted"].sum(), ctrl.sum()], [len(g), len(ctrl)])
        rows.append({"variant": v, "diff_pp": (g["converted"].mean() - ctrl.mean()) * 100, "z": z, "p_value": p})
    out = pd.DataFrame(rows)
    out["p_holm"] = multipletests(out["p_value"], method="holm")[1]
    out["significant"] = out["p_holm"] < C.ALPHA
    return out


def profit_vs_control(df: pd.DataFrame) -> pd.DataFrame:
    """Welch t-test on profit per session (the primary metric), Holm-adjusted.

    Profit per session is zero-inflated and right-skewed, but with 2,000 sessions
    per arm the sampling distribution of the mean is close to normal (CLT); the
    bootstrap in `bootstrap_profit_lift` gives a distribution-free cross-check.
    """
    ctrl = df[df["variant"] == C.CONTROL]["contribution_profit_inr"]
    rows = []
    for v, g in df[df["variant"] != C.CONTROL].groupby("variant"):
        x = g["contribution_profit_inr"]
        t, p = stats.ttest_ind(x, ctrl, equal_var=False)
        diff = x.mean() - ctrl.mean()
        se = np.sqrt(x.var(ddof=1) / len(x) + ctrl.var(ddof=1) / len(ctrl))
        rows.append({"variant": v, "diff_inr": diff, "ci_low": diff - 1.96 * se,
                     "ci_high": diff + 1.96 * se, "t": t, "p_value": p})
    out = pd.DataFrame(rows)
    out["p_holm"] = multipletests(out["p_value"], method="holm")[1]
    out["significant"] = out["p_holm"] < C.ALPHA
    return out


def bootstrap_profit_lift(df: pd.DataFrame, treatment: str, n_boot: int = 5000,
                          seed: int = C.SEED) -> np.ndarray:
    """Bootstrap distribution of (treatment - control) profit per session."""
    rng = np.random.default_rng(seed)
    t = df.loc[df["variant"] == treatment, "contribution_profit_inr"].to_numpy()
    c = df.loc[df["variant"] == C.CONTROL, "contribution_profit_inr"].to_numpy()
    tb = rng.choice(t, (n_boot, len(t))).mean(axis=1)
    cb = rng.choice(c, (n_boot, len(c))).mean(axis=1)
    return tb - cb


# --------------------------------------------------------------------------- #
# Decision and business impact
# --------------------------------------------------------------------------- #
def pick_winner(summary: pd.DataFrame, max_conv_drop_pp: float = 2.0) -> str:
    """Highest profit per session among variants that pass the conversion guardrail."""
    base_conv = summary.loc[C.CONTROL, "conversion"]
    eligible = summary[(summary["conversion"] - base_conv) * 100 >= -max_conv_drop_pp]
    return eligible["profit_per_session_inr"].idxmax()


def monthly_impact(summary: pd.DataFrame, winner: str,
                   sessions: int = C.MONTHLY_SESSIONS, boot: np.ndarray | None = None) -> dict:
    """Scale the per-session profit difference to monthly and annual figures."""
    diff = summary.loc[winner, "profit_per_session_inr"] - summary.loc[C.CONTROL, "profit_per_session_inr"]
    out = {"winner": winner, "profit_diff_per_session": diff,
           "monthly_incremental_profit": diff * sessions, "annual_incremental_profit": diff * sessions * 12}
    if boot is not None:
        lo, hi = np.percentile(boot, [2.5, 97.5])
        out["monthly_ci_low"], out["monthly_ci_high"] = lo * sessions, hi * sessions
    return out
