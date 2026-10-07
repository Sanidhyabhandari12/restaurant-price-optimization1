"""Generate the simulated pricing-experiment sessions from genuine UCI consumer data.

What is real and what is simulated
----------------------------------
* Real (UCI): consumer IDs, their budget and occupation (activity), and every
  rating they gave. Each user's mean overall rating drives their baseline
  purchase propensity, and users who rated more restaurants appear in more sessions.
* Simulated: which menu variant a session saw, whether it converted, basket
  size, prices, discounts and costs. No public dataset contains real restaurant
  A/B-test outcomes, so these are generated from the explicit assumptions in
  `config.py`.

Usage
-----
    python -m src.simulate                      # writes data/processed/regenerated_sessions.csv
    python -m src.simulate --seed 7 --out my.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src import config as C
from src.economics import add_unit_economics

DATA_TYPE_LABEL = "Simulated experiment outcome calibrated with genuine UCI profiles/ratings"


# --------------------------------------------------------------------------- #
# Load and prepare genuine UCI data
# --------------------------------------------------------------------------- #
def load_uci() -> dict[str, pd.DataFrame]:
    """Read the three UCI tables used by the project."""
    read = lambda name: pd.read_csv(C.RAW_DIR / name, dtype=str)
    ratings = read("rating_final.csv")
    for col in ["rating", "food_rating", "service_rating"]:
        ratings[col] = ratings[col].astype(int)
    return {
        "ratings": ratings,
        "profiles": read("userprofile.csv"),
        "restaurants": read("geoplaces2.csv"),
    }


def build_user_table(ratings: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    """One row per UCI consumer with segment, budget and observed rating behaviour."""
    rating_stats = (
        ratings.groupby("userID")
        .agg(mean_rating=("rating", "mean"), n_ratings=("rating", "size"))
        .reset_index()
    )
    users = profiles[["userID", "activity", "budget"]].merge(rating_stats, on="userID", how="inner")
    users["customer_segment"] = users["activity"].map(C.SEGMENT_MAP).fillna(C.DEFAULT_SEGMENT)
    users["budget"] = users["budget"].map(C.BUDGET_MAP).fillna(C.DEFAULT_BUDGET)
    users["mean_rating"] = users["mean_rating"].round(3)
    return users.rename(columns={"userID": "user_id"})


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #
def conversion_probability(df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """Baseline propensity from UCI attributes + session context + treatment effect."""
    v = df["variant"]
    p = (
        C.CONV_INTERCEPT
        + C.CONV_RATING_COEF * df["observed_user_rating_0_2"]
        + df["budget"].map(C.CONV_BUDGET)
        + df["customer_segment"].map(C.CONV_SEGMENT)
        + df["channel"].map(C.CONV_CHANNEL)
        + C.CONV_WEEKEND * df["is_weekend"]
        + v.map({k: s["conv_effect"] for k, s in C.VARIANTS.items()})
        # heterogeneous treatment effects
        + C.HTE_CHARM_LOW_BUDGET * ((v == "B - Charm Pricing") & (df["budget"] == "Low"))
        + C.HTE_PREMIUM_HIGH_BUDGET * ((v == "C - Premium Framing") & (df["budget"] == "High"))
        + C.HTE_COMBO_PRICE_SENSITIVE
        * ((v == "D - Anchor + Combo") & df["customer_segment"].isin(["Students", "Working Class"]))
        + rng.normal(0, C.CONV_NOISE_SD, len(df))
    )
    return np.clip(p, 0.02, 0.98).round(4)


def simulate_sessions(seed: int = C.SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    uci = load_uci()
    users = build_user_table(uci["ratings"], uci["profiles"])

    n = C.SESSIONS_PER_VARIANT * len(C.VARIANTS)

    # Balanced allocation: exactly SESSIONS_PER_VARIANT per arm, randomly ordered
    variants = np.repeat(list(C.VARIANTS), C.SESSIONS_PER_VARIANT)
    rng.shuffle(variants)

    # Users visit in proportion to how active they were in the UCI data
    weights = users["n_ratings"] / users["n_ratings"].sum()
    picked = users.sample(n=n, replace=True, weights=weights, random_state=seed).reset_index(drop=True)

    dates = pd.Timestamp(C.START_DATE) + pd.to_timedelta(rng.integers(0, C.N_DAYS, n), unit="D")

    df = pd.DataFrame({
        "session_id": [f"S{i:05d}" for i in range(1, n + 1)],
        "session_date": dates.strftime("%Y-%m-%d"),
        "day": dates.day_name(),
        "user_id": picked["user_id"],
        "customer_segment": picked["customer_segment"],
        "budget": picked["budget"],
        "channel": rng.choice(list(C.CHANNEL_PROBS), n, p=list(C.CHANNEL_PROBS.values())),
        "variant": variants,
        "observed_user_rating_0_2": picked["mean_rating"],
    })
    df["is_weekend"] = df["day"].isin(["Saturday", "Sunday"]).astype(int)

    # 1) Conversion
    df["conversion_probability_model"] = conversion_probability(df, rng)
    df["converted"] = (rng.random(n) < df["conversion_probability_model"]).astype(int)

    # 2) Basket size: Poisson items per order, at least one item if converted
    lam = C.BASE_UNITS_LAMBDA * (1 + df["variant"].map({k: s["units_effect"] for k, s in C.VARIANTS.items()}))
    units = np.maximum(1, rng.poisson(lam))
    df["units"] = np.where(df["converted"] == 1, units, 0)

    # 3) Order value: base price x budget factor x variant price multiplier x noise
    price_per_unit = (
        C.BASE_ITEM_PRICE
        * df["budget"].map(C.BUDGET_PRICE_FACTOR)
        * df["variant"].map({k: s["price_mult"] for k, s in C.VARIANTS.items()})
        * np.clip(rng.normal(1, C.ORDER_PRICE_NOISE_SD, n), 0.5, 1.6)
    )
    df["gross_order_value_inr"] = (df["units"] * price_per_unit).round(2)
    df["discount_inr"] = (
        df["gross_order_value_inr"] * df["variant"].map({k: s["discount"] for k, s in C.VARIANTS.items()})
    ).round(2)

    # 4) Unit economics (revenue, variable cost, contribution profit)
    df = add_unit_economics(df)
    df["data_type"] = DATA_TYPE_LABEL

    # Shuffle row order so the file carries no allocation pattern
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    cols = [
        "session_id", "session_date", "day", "user_id", "customer_segment", "budget", "channel",
        "variant", "converted", "units", "gross_order_value_inr", "discount_inr", "net_revenue_inr",
        "variable_cost_inr", "contribution_profit_inr", "observed_user_rating_0_2",
        "conversion_probability_model", "data_type",
    ]
    return df[cols]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=C.SEED)
    parser.add_argument("--out", default=str(C.REGENERATED_CSV))
    args = parser.parse_args()

    df = simulate_sessions(args.seed)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df):,} sessions to {args.out}")
    print(df.groupby("variant")[["converted", "contribution_profit_inr"]].mean().round(3))


if __name__ == "__main__":
    main()
