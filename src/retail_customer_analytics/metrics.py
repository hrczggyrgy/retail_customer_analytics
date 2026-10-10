"""Metrics computation module for retail customer analytics.

This module provides metric computation functions used by both the CLI
and Streamlit app, with support for both pandas and Polars backends.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from retail_customer_analytics.polars_utils import (
    compute_period_metrics_pl,
    to_polars,
)

LOGGER = logging.getLogger("retail_analysis.metrics")


def compute_period_metrics(
    df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    use_polars: bool = True,
) -> dict[str, Any]:
    """Compute all executive KPIs for a given period.

    Uses Polars for memory efficiency when available, falls back to pandas.

    Args:
        df: Transaction DataFrame with required columns
        start: Period start date (inclusive)
        end: Period end date (inclusive)
        use_polars: Whether to use Polars for computation (default: True)

    Returns:
        Dictionary with computed metrics
    """
    # Filter to period
    period_df = df[(df["transaction_day"] >= start) & (df["transaction_day"] <= end)].copy()

    if period_df.empty:
        return {
            "revenue": 0.0,
            "orders": 0,
            "customers": 0,
            "aov": 0.0,
            "units_per_order": 0.0,
            "avg_unit_price": 0.0,
            "revenue_per_customer": 0.0,
            "orders_per_customer": 0.0,
            "units_sold": 0,
        }

    # Use pre-calculated revenue column if available
    if "revenue" not in period_df.columns:
        period_df["revenue"] = period_df["quantity"] * period_df["price"]

    # Try Polars for memory efficiency
    if use_polars:
        try:
            pl_df = to_polars(period_df)
            if pl_df is not None:
                return compute_period_metrics_pl(pl_df)
        except Exception as e:
            LOGGER.debug(f"Polars computation failed, falling back to pandas: {e}")

    # Fallback to pandas
    revenue = period_df["revenue"].sum()
    orders = period_df["transaction_id"].nunique()
    customers = period_df["customer_id"].nunique()
    units_sold = period_df["quantity"].sum()

    aov = revenue / orders if orders > 0 else 0.0
    units_per_order = units_sold / orders if orders > 0 else 0.0
    avg_unit_price = revenue / units_sold if units_sold > 0 else 0.0
    revenue_per_customer = revenue / customers if customers > 0 else 0.0
    orders_per_customer = orders / customers if customers > 0 else 0.0

    return {
        "revenue": float(revenue),
        "orders": int(orders),
        "customers": int(customers),
        "aov": aov,
        "units_per_order": units_per_order,
        "avg_unit_price": avg_unit_price,
        "revenue_per_customer": revenue_per_customer,
        "orders_per_customer": orders_per_customer,
        "units_sold": int(units_sold),
    }


def compute_metric_delta(
    current: float,
    prior: float,
    metric_name: str,
) -> dict[str, Any]:
    """Compute absolute and percentage change between current and prior values.

    Args:
        current: Current period value
        prior: Prior period value
        metric_name: Name of the metric (for favorable direction lookup)

    Returns:
        Dictionary with change, pct_change, and trend
    """
    if prior == 0:
        change = current
        pct_change = 0.0
        trend = "neutral"
    else:
        change = current - prior
        pct_change = (change / prior) * 100
        trend = "up" if change > 0 else ("down" if change < 0 else "neutral")

    return {
        "change": change,
        "pct_change": pct_change,
        "trend": trend,
    }


def format_metric_delta(
    current: float,
    prior: float,
    metric_name: str,
    fmt_func: callable = lambda x: f"{x:,.2f}",
) -> dict[str, str]:
    """Format metric delta for display.

    Args:
        current: Current period value
        prior: Prior period value
        metric_name: Name of the metric
        fmt_func: Function to format numeric values

    Returns:
        Dictionary with formatted change and percentage
    """
    delta = compute_metric_delta(current, prior, metric_name)

    change_str = fmt_func(delta["change"])
    pct_str = f"{delta['pct_change']:+.1f}%"

    return {
        "change": change_str,
        "pct_change": pct_str,
        "trend": delta["trend"],
    }


# Favorable direction for each metric (up is good or down is good)
FAVORABLE_DIRECTION = {
    "Revenue": "up",
    "Orders": "up",
    "Active Customers": "up",
    "AOV": "up",
    "Units/Order": "up",
    "Revenue/Customer": "up",
    "Returns": "down",
    "Churn": "down",
    "Return Rate": "down",
}


def get_delta_color(trend: str, metric_label: str) -> str:
    """Get delta color based on trend and metric favorable direction.

    Args:
        trend: "up", "down", or "neutral"
        metric_label: Metric label to look up favorable direction

    Returns:
        Color code: "normal", "inverse", or "off"
    """
    favorable = FAVORABLE_DIRECTION.get(metric_label, "up")
    if trend == favorable:
        return "normal"
    elif trend != "neutral":
        return "inverse"
    else:
        return "off"
