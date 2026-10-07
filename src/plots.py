"""Charts for the README and notebook (matplotlib, static PNG)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config as C

# Colour follows the variant everywhere; control is a neutral baseline.
VARIANT_COLORS = {
    "A - Control": "#8a8984",
    "B - Charm Pricing": "#2a78d6",
    "C - Premium Framing": "#eb6834",
    "D - Anchor + Combo": "#1baf7a",
}
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "font.size": 10.5, "axes.titlesize": 12.5, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True,
})


def _short(v: str) -> str:
    return v.split(" - ")[1] if " - " in v else v


def _save(fig, name: str | None):
    if name:
        C.FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(C.FIG_DIR / name, dpi=160, bbox_inches="tight")
    return fig


def kpi_bars(df: pd.DataFrame, save: str | None = "kpi_by_variant.png"):
    """Conversion, AOV and profit/session by variant with 95% CIs."""
    variants = list(C.VARIANTS)
    specs = [
        ("Conversion rate", lambda g: g["converted"], 100, "{:.1f}%"),
        ("Avg order value (Rs)", lambda g: g.loc[g["converted"] == 1, "net_revenue_inr"], 1, "Rs {:.0f}"),
        ("Profit per session (Rs)", lambda g: g["contribution_profit_inr"], 1, "Rs {:.0f}"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, (title, getter, scale, fmt) in zip(axes, specs):
        means, errs = [], []
        for v in variants:
            x = getter(df[df["variant"] == v]) * scale
            means.append(x.mean())
            errs.append(1.96 * x.std(ddof=1) / np.sqrt(len(x)))
        x_pos = np.arange(len(variants))
        ax.bar(x_pos, means, width=0.62, color=[VARIANT_COLORS[v] for v in variants],
               yerr=errs, error_kw={"ecolor": INK_2, "elinewidth": 1.2, "capsize": 3})
        for i, (m, e) in enumerate(zip(means, errs)):
            ax.text(i, m + e, fmt.format(m), ha="center", va="bottom", fontsize=9.5, color=INK)
        ax.set_xticks(x_pos, [_short(v).replace(" ", "\n", 1) for v in variants], fontsize=9)
        ax.set_title(title)
        ax.set_ylim(0, max(np.add(means, errs)) * 1.15)
    fig.suptitle("Anchor + Combo leads on all three metrics (bars = mean, whiskers = 95% CI)",
                 x=0.01, ha="left", fontsize=11, color=INK_2)
    fig.tight_layout()
    return _save(fig, save)


def lift_vs_control(profit_tests: pd.DataFrame, save: str | None = "profit_lift_vs_control.png"):
    """Difference in profit per session vs control with 95% CI."""
    t = profit_tests.set_index("variant").loc[[v for v in C.VARIANTS if v != C.CONTROL]]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    y = np.arange(len(t))[::-1]
    ax.axvline(0, color=INK_2, lw=1)
    for yi, (v, r) in zip(y, t.iterrows()):
        ax.plot([r.ci_low, r.ci_high], [yi, yi], color=VARIANT_COLORS[v], lw=2.5, solid_capstyle="round")
        ax.plot(r.diff_inr, yi, "o", ms=9, color=VARIANT_COLORS[v], mec=SURFACE, mew=2)
        star = "significant" if r.significant else "not significant"
        ax.text(r.ci_high + 1.5, yi, f"{r.diff_inr:+.1f} Rs  ({star})", va="center", fontsize=9.5)
    ax.set_yticks(y, [_short(v) for v in t.index])
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_xlabel("Profit per session vs control (Rs), 95% CI")
    ax.set_title("Profit per session: difference from control")
    ax.set_xlim(min(t.ci_low.min(), 0) - 5, t.ci_high.max() + 30)
    fig.tight_layout()
    return _save(fig, save)


def bootstrap_hist(boot: np.ndarray, sessions: int = C.MONTHLY_SESSIONS,
                   save: str | None = "bootstrap_monthly_impact.png"):
    """Bootstrap distribution of the winner's monthly incremental profit."""
    monthly = boot * sessions / 1e5  # in Rs lakh
    lo, hi = np.percentile(monthly, [2.5, 97.5])
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.hist(monthly, bins=50, color=VARIANT_COLORS["D - Anchor + Combo"], edgecolor=SURFACE, linewidth=0.6)
    for x in (lo, hi):
        ax.axvline(x, color=INK_2, ls="--", lw=1)
    ax.axvline(monthly.mean(), color=INK, lw=1.5)
    ax.set_title("Anchor + Combo: monthly incremental profit, 5,000 bootstrap resamples")
    ax.set_xlabel(f"Rs lakh per month at {sessions:,} sessions  |  95% interval: {lo:.1f} to {hi:.1f}")
    ax.set_ylabel("Resamples")
    fig.tight_layout()
    return _save(fig, save)


def segment_heatmap(seg: pd.DataFrame, min_sessions: int = 100,
                    save: str | None = "segment_profit_heatmap.png"):
    """Profit per session by segment x variant; small cells are greyed out."""
    pps = seg["profit_per_session_inr"].unstack()[list(C.VARIANTS)]
    n = seg["sessions"].unstack()[list(C.VARIANTS)]
    order = n.sum(axis=1).sort_values(ascending=False).index
    pps, n = pps.loc[order], n.loc[order]
    masked = pps.where(n >= min_sessions)

    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", ["#e8f1fb", "#2a78d6", "#0d3c73"])
    cmap.set_bad(GRID)
    im = ax.imshow(masked.to_numpy(dtype=float), cmap=cmap, aspect="auto")
    for i in range(pps.shape[0]):
        for j in range(pps.shape[1]):
            small = n.iat[i, j] < min_sessions
            val = masked.iat[i, j]
            light = (not small) and val > np.nanpercentile(masked.to_numpy(dtype=float), 60)
            label = f"n={n.iat[i, j]}\ntoo small" if small else f"Rs {val:.0f}\nn={n.iat[i, j]}"
            ax.text(j, i, label, ha="center", va="center", fontsize=8.5,
                    color=SURFACE if light else (INK_2 if small else INK))
    ax.set_xticks(range(pps.shape[1]), [_short(v) for v in pps.columns])
    ax.set_yticks(range(pps.shape[0]), pps.index)
    ax.grid(False)
    ax.set_title(f"Profit per session by segment (cells under {min_sessions} sessions hidden)")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Rs / session")
    fig.tight_layout()
    return _save(fig, save)
