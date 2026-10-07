"""Project configuration: paths, experiment design and unit-economics assumptions.

All commercial numbers here (prices, costs, treatment effects) are modelling
assumptions, not values observed in the UCI dataset. They are kept in one place
so every result in the project can be traced back to an explicit input.
"""
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "uci"
PROCESSED_DIR = ROOT / "data" / "processed"
EXPERIMENT_CSV = PROCESSED_DIR / "restaurant_pricing_experiment_dataset.csv"
REGENERATED_CSV = PROCESSED_DIR / "regenerated_sessions.csv"
FIG_DIR = ROOT / "outputs" / "figures"
TABLE_DIR = ROOT / "outputs" / "tables"

# --------------------------------------------------------------------------- #
# Source data (UCI Restaurant & Consumer Data, DOI 10.24432/C5DP41, CC BY 4.0)
# --------------------------------------------------------------------------- #
SELECTED_PLACE_ID = "135059"  # Restaurant Bar Hacienda los Martinez, price tier: medium

SEGMENT_MAP = {  # UCI `activity` -> business segment
    "student": "Students",
    "professional": "Professionals",
    "working-class": "Working Class",
}
DEFAULT_SEGMENT = "Other"  # unemployed / missing ("?")

BUDGET_MAP = {"low": "Low", "medium": "Medium", "high": "High"}
DEFAULT_BUDGET = "Medium"  # missing ("?") budgets treated as medium

# --------------------------------------------------------------------------- #
# Experiment design
# --------------------------------------------------------------------------- #
CONTROL = "A - Control"
VARIANTS = {
    # name: price multiplier, conversion effect (abs. prob.), units effect (rel.), discount rate
    "A - Control":         {"price_mult": 1.000, "conv_effect":  0.000, "units_effect":  0.00, "discount": 0.000},
    "B - Charm Pricing":   {"price_mult": 0.997, "conv_effect":  0.022, "units_effect":  0.02, "discount": 0.003},
    "C - Premium Framing": {"price_mult": 1.055, "conv_effect": -0.014, "units_effect": -0.01, "discount": 0.000},
    "D - Anchor + Combo":  {"price_mult": 1.025, "conv_effect":  0.055, "units_effect":  0.12, "discount": 0.018},
}
VARIANT_DESCRIPTIONS = {
    "A - Control": "Current menu; rounded prices and standard item descriptions.",
    "B - Charm Pricing": "Prices ending in 9 (e.g. Rs 299) to reduce perceived price.",
    "C - Premium Framing": "Premium descriptions and higher reference-quality cues.",
    "D - Anchor + Combo": "High-priced anchor item next to a prominently framed value combo.",
}
SESSIONS_PER_VARIANT = 2000
START_DATE = "2026-01-01"
N_DAYS = 84
CHANNEL_PROBS = {"Delivery": 0.42, "Dine-in": 0.40, "Takeaway": 0.18}
SEED = 42

# Conversion-propensity model (probability scale)
CONV_INTERCEPT = 0.34
CONV_RATING_COEF = 0.148          # per point of the user's mean UCI rating (0-2)
CONV_BUDGET = {"High": 0.000, "Medium": -0.041, "Low": -0.068}
CONV_SEGMENT = {"Other": 0.000, "Students": 0.005, "Professionals": 0.005, "Working Class": 0.005}
CONV_CHANNEL = {"Delivery": 0.000, "Dine-in": -0.015, "Takeaway": -0.015}
CONV_WEEKEND = 0.025
CONV_NOISE_SD = 0.010
# Heterogeneous treatment effects (the behavioural hypotheses)
HTE_CHARM_LOW_BUDGET = 0.018      # charm pricing works better for low-budget users
HTE_PREMIUM_HIGH_BUDGET = 0.025   # premium framing works for high-budget users
HTE_COMBO_PRICE_SENSITIVE = 0.015 # combos resonate with students / working class

# Basket model
BASE_UNITS_LAMBDA = 1.70          # Poisson mean for items per order (floored at 1)
BASE_ITEM_PRICE = 340.0           # INR, model input (not from UCI)
BUDGET_PRICE_FACTOR = {"Low": 0.87, "Medium": 1.00, "High": 1.20}
ORDER_PRICE_NOISE_SD = 0.16

# --------------------------------------------------------------------------- #
# Unit economics
# --------------------------------------------------------------------------- #
COGS_RATIO = 0.34
PACKAGING_COST_PER_ORDER = 18.0
PAYMENT_FEE_RATE = 0.018

# Business scaling
MONTHLY_SESSIONS = 50_000
ALPHA = 0.05
