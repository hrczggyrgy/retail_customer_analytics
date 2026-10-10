"""Canonical event construction for retail customer analytics.

This module provides the single authoritative implementation of purchase event
(and basket) construction, ensuring consistent frequency, recency, and basket
denominators across CLI and Streamlit entry points.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from retail_customer_analytics.ingestion import prepare_transaction_frame

LOGGER = logging.getLogger("retail_analysis.events")


@dataclass
class EventConfig:
    """Configuration for event construction."""

    basket_grain: str = "trip"  # "trip" (customer-day) or "transaction"
    transaction_key_mode: str = "customer-transaction"  # "customer-transaction" or "transaction-only"
    min_trip_value: float = 0.0  # minimum positive trip value to qualify


def build_purchase_events(
    transactions: pd.DataFrame,
    as_of: pd.Timestamp,
    config: EventConfig | None = None,
) -> pd.DataFrame:
    """Build canonical purchase events (trips) up to snapshot date.

    This is the single source of truth for event construction used by
    features, models, basket affinity, and all downstream analytics.

    Args:
        transactions: Normalized transaction DataFrame
        as_of: Snapshot cutoff date - only transactions <= as_of are used
        config: EventConfig with grain and key policy

    Returns:
        DataFrame with one row per purchase event:
        - customer_id
        - transaction_day (or transaction_id for transaction grain)
        - trip_value: sum of positive line revenue
        - n_lines: number of line items
        - n_products: number of distinct products
        - n_units: total units
    """
    if config is None:
        config = EventConfig()

    # Only positive purchases before cutoff with valid customer
    buys = transactions[
        (transactions["quantity"] > 0)
        & transactions["customer_id"].notna()
        & transactions["transaction_day"].notna()
        & (transactions["transaction_day"] <= as_of)
    ].copy()

    if buys.empty:
        return _empty_events(config.basket_grain)

    if config.basket_grain == "transaction":
        buys = buys[buys["transaction_id"].notna()]

    buys["rev_pos"] = buys["line_revenue"].clip(lower=0).fillna(0.0)

    if config.basket_grain == "trip":
        keys = ["customer_id", "transaction_day"]
        spec = {
            "trip_value": ("rev_pos", "sum"),
            "n_lines": ("quantity", "size"),
            "n_products": ("product_id", "nunique"),
            "n_units": ("quantity", "sum"),
        }
    else:
        keys = ["customer_id", "transaction_id"]
        spec = {
            "trip_value": ("rev_pos", "sum"),
            "n_lines": ("quantity", "size"),
            "n_products": ("product_id", "nunique"),
            "n_units": ("quantity", "sum"),
            "transaction_day": ("transaction_day", "min"),
        }

    events = buys.groupby(keys, sort=False).agg(**spec).reset_index()
    events = events[events["trip_value"] > config.min_trip_value].reset_index(drop=True)

    return events


def build_baskets(
    transactions: pd.DataFrame,
    as_of: pd.Timestamp,
    config: EventConfig | None = None,
    item_level: str = "product_id",
) -> pd.DataFrame:
    """Build baskets for affinity analysis.

    Args:
        transactions: Normalized transaction DataFrame
        as_of: Analysis cutoff date
        config: EventConfig with grain and key policy
        item_level: "product_id" or "category"

    Returns:
        DataFrame with columns: customer_id, basket_key, items (list of item IDs)
    """
    if config is None:
        config = EventConfig()

    # Filter positive purchases
    base = transactions[
        (transactions["quantity"] > 0)
        & transactions["customer_id"].notna()
        & transactions["transaction_day"].notna()
        & transactions[item_level].notna()
        & (transactions["transaction_day"] <= as_of)
    ].copy()

    # Apply basket grain
    if config.basket_grain == "trip":
        base = base[["customer_id", "transaction_day", item_level]].drop_duplicates()
        basket_key_cols = ["customer_id", "transaction_day"]
    else:  # transaction grain
        if config.transaction_key_mode == "customer-transaction":
            base = base[base["transaction_id"].notna()][
                ["customer_id", "transaction_id", "transaction_day", item_level]
            ].drop_duplicates()
            basket_key_cols = ["customer_id", "transaction_id"]
        else:  # transaction-only
            base = base[base["transaction_id"].notna()][
                ["transaction_id", "transaction_day", "customer_id", item_level]
            ].drop_duplicates()
            basket_key_cols = ["transaction_id"]

    if base.empty:
        return pd.DataFrame(columns=["customer_id", "basket_key", "items"])

    # Aggregate to baskets with deduplication
    baskets = (
        base.groupby(basket_key_cols)[item_level]
        .agg(lambda s: sorted(set(s)))
        .reset_index()
        .rename(columns={item_level: "items"})
    )

    # Create a single basket key column
    if len(basket_key_cols) == 1:
        baskets["basket_key"] = baskets[basket_key_cols[0]].astype(str)
    else:
        baskets["basket_key"] = baskets[basket_key_cols].astype(str).agg("_".join, axis=1)

    # Ensure customer_id is present
    if "customer_id" not in baskets.columns:
        baskets["customer_id"] = baskets[basket_key_cols[0]]

    return baskets[["customer_id", "basket_key", "items"]]


def match_returns(
    transactions: pd.DataFrame,
    as_of: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Link each return line to the latest earlier purchase of the same customer and product.

    Uses full transaction_date timestamps when available for precise ordering.
    Falls back to transaction_day (calendar day) when timestamps are not available.

    Args:
        transactions: Normalized transaction DataFrame
        as_of: Optional cutoff date for returns matching (defaults to max date in data)

    Returns:
        Tuple of (matched_returns DataFrame, diagnostics dict)
    """
    has_timestamps = "transaction_date" in transactions.columns and transactions["transaction_date"].notna().any()
    time_col = "transaction_date" if has_timestamps else "transaction_day"

    ret = transactions[
        (transactions["quantity"] < 0)
        & transactions["customer_id"].notna()
        & transactions["product_id"].notna()
        & transactions[time_col].notna()
    ].copy()

    info: dict[str, Any] = {
        "return_lines": int(len(ret)),
        "matched_return_lines": 0,
        "matched_return_value_share": None,
        "median_days_to_return": None,
        "matching_precision": "timestamp" if has_timestamps else "calendar_day",
    }

    if ret.empty:
        ret["matched"] = pd.Series(dtype=bool)
        return ret, info

    if as_of is not None:
        ret = ret[ret[time_col] <= as_of]

    buys = (
        transactions[
            (transactions["quantity"] > 0)
            & transactions["customer_id"].notna()
            & transactions["product_id"].notna()
            & transactions[time_col].notna()
        ][["customer_id", "product_id", time_col]]
        .drop_duplicates()
        .rename(columns={time_col: "purchase_time"})
        .sort_values("purchase_time")
    )

    merged = pd.merge_asof(
        ret.sort_values(time_col),
        buys,
        left_on=time_col,
        right_on="purchase_time",
        by=["customer_id", "product_id"],
        direction="backward",
    )

    merged["matched"] = merged["purchase_time"].notna()
    value = (-merged["line_revenue"]).clip(lower=0).fillna(0.0)
    info["matched_return_lines"] = int(merged["matched"].sum())

    total = float(value.sum())
    info["matched_return_value_share"] = (
        float(value[merged["matched"]].sum() / total) if total > 0 else None
    )

    if merged["matched"].any():
        delta = merged[time_col] - merged["purchase_time"]
        if has_timestamps:
            info["median_days_to_return"] = float(
                delta.dt.total_seconds()[merged["matched"]].median() / 86400
            )
        else:
            info["median_days_to_return"] = float(delta.dt.days[merged["matched"]].median())

    return merged, info


def add_price_index(
    transactions: pd.DataFrame,
    min_price_observations: int = 5,
    promo_threshold: float = 0.10,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Add price index and promo proxy columns to transactions.

    price_index = unit price / product median price over positive-quantity lines.
    promo_like = price_index <= 1.0 - promo_threshold
    """
    pos = (
        (transactions["quantity"] > 0)
        & (transactions["price"] > 0)
        & transactions["product_id"].notna()
    )
    grp = transactions[pos].groupby("product_id")["price"]
    ref, cnt, nun = grp.median(), grp.size(), grp.nunique()
    ref = ref[cnt >= min_price_observations]

    transactions = transactions.copy()
    transactions["reference_price"] = transactions["product_id"].map(ref)
    transactions["price_index"] = np.where(
        pos, transactions["price"] / transactions["reference_price"], np.nan
    )
    transactions["promo_like"] = (
        transactions["price_index"] <= (1.0 - promo_threshold)
    ) & transactions["price_index"].notna()

    eligible = int(len(ref))
    varying = int((nun.reindex(ref.index) > 1).sum()) if eligible else 0

    info = {
        "products_with_reference_price": eligible,
        "products_with_price_variation": varying,
        "promo_like_line_share": float(
            transactions.loc[transactions["price_index"].notna(), "promo_like"].mean()
        )
        if eligible
        else None,
        "usable": bool(eligible and varying / max(eligible, 1) >= 0.05),
    }

    return transactions, info


def _empty_events(grain: str) -> pd.DataFrame:
    """Return empty events DataFrame with correct columns."""
    if grain == "transaction":
        return pd.DataFrame(
            columns=[
                "customer_id",
                "transaction_id",
                "transaction_day",
                "trip_value",
                "n_lines",
                "n_products",
                "n_units",
            ]
        )
    return pd.DataFrame(
        columns=[
            "customer_id",
            "transaction_day",
            "trip_value",
            "n_lines",
            "n_products",
            "n_units",
        ]
    )


def validate_event_grain(
    events: pd.DataFrame,
    grain: str,
    transaction_key_mode: str = "customer-transaction",
) -> dict[str, Any]:
    """Validate event construction and return diagnostics."""
    info: dict[str, Any] = {
        "grain": grain,
        "transaction_key_mode": transaction_key_mode,
        "total_events": int(len(events)),
        "unique_customers": int(events["customer_id"].nunique()) if len(events) else 0,
    }

    if grain == "trip":
        # Check for multiple events per customer-day (should not happen)
        dup = events.groupby(["customer_id", "transaction_day"]).size()
        info["duplicate_customer_days"] = int((dup > 1).sum())
    else:
        # Check for multiple events per transaction_id (should not happen with customer-transaction)
        if transaction_key_mode == "customer-transaction":
            dup = events.groupby(["customer_id", "transaction_id"]).size()
        else:
            dup = events.groupby("transaction_id").size()
        info["duplicate_transaction_ids"] = int((dup > 1).sum())

    if "transaction_day" in events.columns:
        info["date_range"] = {
            "min": events["transaction_day"].min(),
            "max": events["transaction_day"].max(),
        }

    return info