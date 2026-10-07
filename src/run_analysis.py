"""Run the full analysis end to end and write tables + figures to `outputs/`.

    python -m src.run_analysis                     # analyse the published dataset
    python -m src.run_analysis --data data/processed/regenerated_sessions.csv
"""
from __future__ import annotations

import argparse

import pandas as pd

from src import analysis as A
from src import config as C
from src import plots as P


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(C.EXPERIMENT_CSV))
    args = parser.parse_args()

    df = A.load_experiment(args.data)
    C.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 140)

    summary = A.variant_summary(df)
    seg = A.segment_summary(df)
    srm = A.sample_ratio_mismatch(df)
    chi = A.chi_square_conversion(df)
    anova = A.anova_aov(df)
    tukey = A.tukey_aov(df)
    conv_tests = A.conversion_vs_control(df)
    profit_tests = A.profit_vs_control(df)
    winner = A.pick_winner(summary)
    boot = A.bootstrap_profit_lift(df, winner)
    impact = A.monthly_impact(summary, winner, boot=boot)
    mde = A.minimum_detectable_effect(C.SESSIONS_PER_VARIANT, summary.loc[C.CONTROL, "conversion"])

    summary.to_csv(C.TABLE_DIR / "variant_summary.csv")
    seg.to_csv(C.TABLE_DIR / "segment_summary.csv")
    tukey.to_csv(C.TABLE_DIR / "tukey_hsd_order_value.csv", index=False)
    conv_tests.to_csv(C.TABLE_DIR / "conversion_vs_control.csv", index=False)
    profit_tests.to_csv(C.TABLE_DIR / "profit_vs_control.csv", index=False)
    pd.DataFrame([chi, anova]).to_csv(C.TABLE_DIR / "omnibus_tests.csv", index=False)
    pd.Series(impact).to_csv(C.TABLE_DIR / "business_impact.csv", header=["value"])

    P.kpi_bars(df)
    P.lift_vs_control(profit_tests)
    P.bootstrap_hist(boot)
    P.segment_heatmap(seg)

    print("\n=== Variant summary ===")
    print(summary.round(3).to_string())
    print(f"\nSRM check: chi2={srm['chi2']:.2f}, p={srm['p_value']:.3f}")
    print(f"MDE (conversion, 80% power): {mde * 100:.1f} pp")
    print(f"\n{chi['test']}: chi2={chi['statistic']:.2f}, p={chi['p_value']:.3g}")
    print(f"{anova['test']}: F={anova['statistic']:.2f}, p={anova['p_value']:.3g}")
    print("\n=== Profit per session vs control (Welch, Holm) ===")
    print(profit_tests.round(4).to_string(index=False))
    print(f"\nWinner: {winner}")
    print(f"Monthly incremental profit @ {C.MONTHLY_SESSIONS:,} sessions: Rs {impact['monthly_incremental_profit']:,.0f} "
          f"(95% bootstrap interval Rs {impact['monthly_ci_low']:,.0f} to Rs {impact['monthly_ci_high']:,.0f})")
    print(f"\nTables -> {C.TABLE_DIR}\nFigures -> {C.FIG_DIR}")


if __name__ == "__main__":
    main()
