"""Order-level unit economics shared by the simulation and the analysis."""
import pandas as pd

from src import config as C


def add_unit_economics(df: pd.DataFrame) -> pd.DataFrame:
    """Add net revenue, variable cost and contribution profit columns.

    contribution profit = net revenue
                          - food cost (COGS_RATIO x net revenue)
                          - packaging (flat per order)
                          - payment fee (PAYMENT_FEE_RATE x net revenue)

    Sessions that did not convert carry zero revenue and zero cost.
    """
    out = df.copy()
    ordered = out["converted"] == 1
    out["net_revenue_inr"] = (out["gross_order_value_inr"] - out["discount_inr"]).round(2)
    variable = (
        C.COGS_RATIO * out["net_revenue_inr"]
        + C.PACKAGING_COST_PER_ORDER
        + C.PAYMENT_FEE_RATE * out["net_revenue_inr"]
    )
    out["variable_cost_inr"] = variable.where(ordered, 0.0).round(2)
    out["contribution_profit_inr"] = (out["net_revenue_inr"] - out["variable_cost_inr"]).round(2)
    return out
