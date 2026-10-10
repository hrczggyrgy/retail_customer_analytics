"""Canonical feature engineering module for retail customer analytics.

This module provides a single, authoritative implementation of feature construction
that is used by both the CLI pipeline and the Streamlit app.

All features are built with explicit point-in-time semantics: every feature
uses only data available at or before the snapshot cutoff date.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .ingestion import REQUIRED_COLUMNS

LOGGER = logging.getLogger("retail_analysis.features")


def build_customer_snapshot(
    transactions: pd.DataFrame,
    as_of: pd.Timestamp,
    event_grain: str = "trip",
    windows_days: Sequence[int] = (30, 90, 365),
    recent_window_days: int = 90,
) -> pd.DataFrame:
    """
    Build customer features for a specific snapshot date.
    
    This is the canonical feature builder that replaces duplicated logic
    in retail_customer_analysis.py and app.py.
    
    Args:
        transactions: Normalized transaction DataFrame (from ingestion.prepare_transaction_frame)
        as_of: Snapshot cutoff date - only transactions <= as_of are used
        event_grain: "trip" (customer-day) or "transaction" (per transaction_id)
        windows_days: Rolling windows for recent activity features
        recent_window_days: Window for recent-vs-prior comparison
        
    Returns:
        DataFrame with one row per customer, indexed by customer_id
    """
    # Build purchase events
    trips = _build_trips_for_snapshot(transactions, as_of, event_grain)
    
    if trips.empty:
        return pd.DataFrame()
    
    # Core RFM and cadence features
    cust = _core_customer_features(trips, as_of)
    
    # Rolling window features
    win = _window_features(transactions, trips, cust.index, as_of, windows_days, recent_window_days)
    cust = cust.join(win)
    
    # Breadth features
    breadth = _breadth_features(transactions, cust.index, as_of)
    cust = cust.join(breadth)
    
    # Return features (if return data available)
    ret = _return_features(transactions, cust.index, as_of)
    cust = cust.join(ret)
    
    # Net spend = gross - returns
    if "gross_spend" in cust.columns and "return_value" in cust.columns:
        cust["net_spend"] = (cust["gross_spend"] - cust["return_value"]).clip(lower=0)
    elif "gross_spend" in cust.columns:
        cust["net_spend"] = cust["gross_spend"]
    else:
        cust["net_spend"] = 0.0
    
    # Price/promo proxy features
    promo = _promo_features(transactions, cust.index, as_of)
    cust = cust.join(promo)
    
    # Category affinity features
    cat_aff = _category_affinity_features(transactions, cust.index, as_of)
    cust = cust.join(cat_aff)
    
    # Behavioral trend features
    trend = _trend_features(transactions, cust.index, as_of)
    cust = cust.join(trend)
    
    # Basket composition features
    basket = _basket_composition_features(transactions, trips, cust.index, as_of)
    cust = cust.join(basket)
    
    return cust


def _build_trips_for_snapshot(
    transactions: pd.DataFrame,
    as_of: pd.Timestamp,
    grain: str = "trip",
) -> pd.DataFrame:
    """Build purchase events (trips) up to snapshot date."""
    # Only positive purchases before cutoff
    buys = transactions[
        (transactions["quantity"] > 0) 
        & transactions["customer_id"].notna() 
        & transactions["transaction_day"].notna()
        & (transactions["transaction_day"] <= as_of)
    ].copy()
    
    if grain == "transaction":
        buys = buys[buys["transaction_id"].notna()]
    
    if buys.empty:
        return pd.DataFrame(columns=["customer_id", "transaction_day", "trip_value", "n_lines", "n_products", "n_units"])
    
    buys["rev_pos"] = buys["line_revenue"].clip(lower=0).fillna(0.0)
    
    if grain == "trip":
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
    
    trips = buys.groupby(keys, sort=False).agg(**spec).reset_index()
    trips = trips[trips["trip_value"] > 0].reset_index(drop=True)
    
    return trips


def _core_customer_features(
    trips: pd.DataFrame,
    as_of: pd.Timestamp,
) -> pd.DataFrame:
    """Core RFM, cadence, and recency features."""
    t = trips.sort_values(["customer_id", "transaction_day"]).copy()
    t["gap"] = t.groupby("customer_id")["transaction_day"].diff().dt.days
    
    c = t.groupby("customer_id").agg(
        n_trips=("transaction_day", "size"),
        first_day=("transaction_day", "min"),
        last_day=("transaction_day", "max"),
        gross_spend=("trip_value", "sum"),
        mean_trip_value=("trip_value", "mean"),
        mean_basket_products=("n_products", "mean"),
        median_basket_products=("n_products", "median"),
        median_gap_days=("gap", "median"),
        mean_gap_days=("gap", "mean"),
        std_gap_days=("gap", "std"),
        n_gaps=("gap", "count"),
    )
    
    c["x"] = c["n_trips"] - 1
    c["t_x"] = (c["last_day"] - c["first_day"]).dt.days.astype(float)
    c["T"] = (as_of - c["first_day"]).dt.days.astype(float)
    c["recency_days"] = (as_of - c["last_day"]).dt.days.astype(float)
    
    # Cadence stats
    c["interpurchase_cv"] = c["std_gap_days"] / c["mean_gap_days"].where(c["mean_gap_days"] > 0)
    med = c["median_gap_days"]
    # Default inactivity thresholds (configurable via args in full pipeline)
    inactivity_min_days = 30.0
    inactivity_cadence_multiplier = 2.0
    inactivity_fallback_days = 90.0
    thr = np.where(med > 0, np.maximum(inactivity_min_days, med * inactivity_cadence_multiplier), inactivity_fallback_days)
    c["heuristic_threshold_days"] = thr
    c["heuristic_inactive_flag"] = c["recency_days"] > thr
    
    # Regularity (gamma shape parameter for interpurchase times)
    var = c["std_gap_days"] ** 2
    k_hat = (c["mean_gap_days"] ** 2 / var).where((c["n_gaps"] >= 3) & (var > 0))
    c["regularity_shape_raw"] = k_hat
    c["regularity_shape_shrunk"] = ((c["n_gaps"] * k_hat.fillna(1.0)) + 3.0) / (c["n_gaps"] + 3.0)
    
    return c


def _window_features(
    transactions: pd.DataFrame,
    trips: pd.DataFrame,
    index: pd.Index,
    as_of: pd.Timestamp,
    windows: Sequence[int],
    recent_days: int,
) -> pd.DataFrame:
    """Rolling-window spend/trip features and recent-vs-prior change."""
    f = pd.DataFrame(index=index)
    cl = transactions[
        transactions["customer_id"].notna() 
        & transactions["transaction_day"].notna() 
        & (transactions["transaction_day"] <= as_of)
    ].copy()
    
    def agg(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        sub = cl[(cl["transaction_day"] >= start) & (cl["transaction_day"] <= end)]
        gross = sub["line_revenue"].where(sub["quantity"] > 0).groupby(sub["customer_id"]).sum()
        ret = (-sub["line_revenue"]).where(sub["quantity"] < 0).groupby(sub["customer_id"]).sum()
        tt = trips[(trips["transaction_day"] >= start) & (trips["transaction_day"] <= end)].groupby("customer_id").size()
        d = pd.DataFrame({"gross_purchase_revenue": gross, "return_value": ret, "trips": tt}).reindex(index).fillna(0.0)
        d["net_revenue"] = d["gross_purchase_revenue"] - d["return_value"]
        return d
    
    for w in windows:
        d = agg(as_of - pd.Timedelta(days=w - 1), as_of)
        f[f"gross_purchase_revenue_{w}d"], f[f"return_value_{w}d"] = d["gross_purchase_revenue"], d["return_value"]
        f[f"net_revenue_{w}d"], f[f"trips_{w}d"] = d["net_revenue"], d["trips"].astype(int)
    
    rec = agg(as_of - pd.Timedelta(days=recent_days - 1), as_of)
    pri = agg(as_of - pd.Timedelta(days=2 * recent_days - 1), as_of - pd.Timedelta(days=recent_days))
    
    f["recent_gross_purchase_revenue"], f["prior_gross_purchase_revenue"] = rec["gross_purchase_revenue"], pri["gross_purchase_revenue"]
    f["recent_trips"], f["prior_trips"] = rec["trips"].astype(int), pri["trips"].astype(int)
    f["recent_revenue_change_pct_if_prior_positive"] = (
        (rec["gross_purchase_revenue"] - pri["gross_purchase_revenue"]) / pri["gross_purchase_revenue"]
    ).where(pri["gross_purchase_revenue"] > 0)
    f["recent_trip_change_pct_if_prior_positive"] = (
        (rec["trips"] - pri["trips"]) / pri["trips"]
    ).where(pri["trips"] > 0)
    
    return f


def _breadth_features(
    transactions: pd.DataFrame,
    index: pd.Index,
    as_of: pd.Timestamp,
) -> pd.DataFrame:
    """Distinct products/categories/departments, top category/department, HHI."""
    pos = transactions[
        (transactions["quantity"] > 0) 
        & transactions["customer_id"].notna() 
        & (transactions["transaction_day"] <= as_of)
    ].copy()
    pos["rev"] = pos["line_revenue"].clip(lower=0).fillna(0.0)
    
    f = pd.DataFrame(index=index)
    f["unique_products_purchased"] = pos.groupby("customer_id")["product_id"].nunique().reindex(index)
    
    for col, name in [("category", "categories"), ("department", "departments")]:
        sub = pos[pos[col].notna()]
        f[f"unique_{name}_purchased"] = sub.groupby("customer_id")[col].nunique().reindex(index)
        if sub.empty:
            f[f"top_{col}_by_spend"] = None
            f[f"{col}_spend_hhi"] = np.nan
            continue
        sp = sub.groupby(["customer_id", col])["rev"].sum().reset_index()
        top = sp.sort_values(["customer_id", "rev", col], ascending=[True, False, True]).drop_duplicates("customer_id").set_index("customer_id")[col]
        tot = sp.groupby("customer_id")["rev"].transform("sum")
        share = (sp["rev"] / tot).where(tot > 0)
        f[f"top_{col}_by_spend"] = top.reindex(index)
        f[f"{col}_spend_hhi"] = (share ** 2).groupby(sp["customer_id"]).sum(min_count=1).reindex(index)
    
    return f


def _return_features(
    transactions: pd.DataFrame,
    index: pd.Index,
    as_of: pd.Timestamp,
) -> pd.DataFrame:
    """Return value ratio, return-unit ratio, unmatched return share."""
    f = pd.DataFrame(index=index)
    
    # Match returns to purchases (same logic as in retail_customer_analysis)
    ret = transactions[
        (transactions["quantity"] < 0) 
        & transactions["customer_id"].notna() 
        & transactions["product_id"].notna()
        & transactions["transaction_day"].notna()
        & (transactions["transaction_day"] <= as_of)
    ].copy()
    
    if ret.empty:
        f["return_value"] = 0.0
        f["unmatched_return_value"] = 0.0
        f["return_value_ratio"] = 0.0
        f["return_unit_ratio"] = 0.0
        f["unmatched_return_share"] = 0.0
        return f
    
    buys = transactions[
        (transactions["quantity"] > 0) 
        & transactions["customer_id"].notna() 
        & transactions["product_id"].notna()
        & transactions["transaction_day"].notna()
        & (transactions["transaction_day"] <= as_of)
    ][["customer_id", "product_id", "transaction_day"]].drop_duplicates()
    buys = buys.rename(columns={"transaction_day": "purchase_day"}).sort_values("purchase_day")
    
    merged = pd.merge_asof(
        ret.sort_values("transaction_day"), 
        buys, 
        left_on="transaction_day", 
        right_on="purchase_day",
        by=["customer_id", "product_id"], 
        direction="backward"
    )
    merged["matched"] = merged["purchase_day"].notna()
    
    rv = (-merged["line_revenue"]).clip(lower=0).fillna(0.0)
    
    f["return_value"] = rv.groupby(merged["customer_id"]).sum().reindex(index).fillna(0.0)
    um = ~merged["matched"]
    f["unmatched_return_value"] = rv[um].groupby(merged.loc[um, "customer_id"]).sum().reindex(index).fillna(0.0)
    f["return_value_ratio"] = (f["return_value"] / f["return_value"].where(f["return_value"] > 0, np.inf)).clip(0, 1)
    
    # Return unit ratio
    ret_units = (-merged["quantity"]).clip(lower=0).fillna(0).astype(int)
    f["return_units"] = ret_units.groupby(merged["customer_id"]).sum().reindex(index).fillna(0).astype(int)
    buy_units = merged["quantity"].clip(lower=0).fillna(0).astype(int)  # This won't work directly
    
    # Simplified: use total positive quantity as denominator
    pos_qty = transactions[
        (transactions["quantity"] > 0) 
        & transactions["customer_id"].notna() 
        & (transactions["transaction_day"] <= as_of)
    ].groupby("customer_id")["quantity"].sum()
    f["return_unit_ratio"] = (f["return_units"] / pos_qty.reindex(index).fillna(0)).clip(0, 1)
    
    # Unmatched share
    f["unmatched_return_share"] = (
        f["unmatched_return_value"] / f["return_value"]
    ).where(f["return_value"] > 0).fillna(0.0)
    
    return f


def _promo_features(
    transactions: pd.DataFrame,
    index: pd.Index,
    as_of: pd.Timestamp,
) -> pd.DataFrame:
    """Price-index based promo proxy features."""
    f = pd.DataFrame(index=index)
    
    pos = transactions[
        (transactions["quantity"] > 0) 
        & transactions["customer_id"].notna() 
        & transactions["product_id"].notna()
        & transactions["price"].notna()
        & (transactions["transaction_day"] <= as_of)
    ].copy()
    
    if pos.empty:
        f["promo_spend_share"] = np.nan
        f["mean_price_index"] = np.nan
        f["price_index_median"] = np.nan
        return f
    
    # Compute reference prices per product
    ref = pos.groupby("product_id")["price"].median()
    pos["reference_price"] = pos["product_id"].map(ref)
    pos["price_index"] = pos["price"] / pos["reference_price"]
    
    # Promo threshold (configurable)
    promo_threshold = 0.10
    pos["promo_like"] = pos["price_index"] <= (1.0 - promo_threshold)
    
    rev = pos["line_revenue"].clip(lower=0).fillna(0.0)
    tot = rev.groupby(pos["customer_id"]).sum().reindex(index)
    promo_rev = rev[pos["promo_like"]].groupby(pos.loc[pos["promo_like"], "customer_id"]).sum().reindex(index).fillna(0.0)
    wi = (pos["price_index"] * rev).groupby(pos["customer_id"]).sum().reindex(index)
    
    f["promo_spend_share"] = (promo_rev / tot).where(tot > 0)
    f["mean_price_index"] = (wi / tot).where(tot > 0)
    f["price_index_median"] = pos.groupby("customer_id")["price_index"].median().reindex(index)
    
    return f


def _category_affinity_features(
    transactions: pd.DataFrame,
    index: pd.Index,
    as_of: pd.Timestamp,
    max_categories: int = 30,
) -> pd.DataFrame:
    """Per-customer category spend share and recency."""
    f = pd.DataFrame(index=index)
    
    pos = transactions[
        (transactions["quantity"] > 0) 
        & transactions["customer_id"].notna() 
        & transactions["category"].notna()
        & (transactions["transaction_day"] <= as_of)
    ].copy()
    pos["rev"] = pos["line_revenue"].clip(lower=0).fillna(0.0)
    
    if pos.empty:
        return f
    
    # Top categories by spend
    top_cats = pos.groupby("category")["rev"].sum().sort_values(ascending=False).head(max_categories).index
    pos["category_grouped"] = pos["category"].where(pos["category"].isin(top_cats), "__other__")
    
    # Category spend shares
    cat_spend = pos.groupby(["customer_id", "category_grouped"])["rev"].sum().reset_index()
    cat_pivot = cat_spend.pivot(index="customer_id", columns="category_grouped", values="rev").fillna(0)
    cat_pivot = cat_pivot.reindex(index).fillna(0)
    
    # Total spend per customer
    total_spend = cat_pivot.sum(axis=1)
    
    # Spend share per category
    for cat in cat_pivot.columns:
        f[f"category_{cat}_spend_share"] = (cat_pivot[cat] / total_spend).where(total_spend > 0)
    
    # Top category by spend
    top_cat = cat_pivot.idxmax(axis=1)
    f["top_category_by_spend_affinity"] = top_cat.reindex(index)
    
    # Category entropy (diversity)
    shares = cat_pivot.div(total_spend.replace(0, np.nan), axis=0)
    entropy = -(shares * np.log(shares + 1e-10)).sum(axis=1)
    f["category_entropy"] = entropy.reindex(index)
    
    return f


def _trend_features(
    transactions: pd.DataFrame,
    index: pd.Index,
    as_of: pd.Timestamp,
) -> pd.DataFrame:
    """Behavioral trend features: monthly revenue slope, volatility, active months fraction."""
    f = pd.DataFrame(index=index)
    
    pos = transactions[
        (transactions["quantity"] > 0) 
        & transactions["customer_id"].notna() 
        & (transactions["transaction_day"] <= as_of)
    ].copy()
    pos["rev"] = pos["line_revenue"].clip(lower=0).fillna(0.0)
    pos["month"] = pos["transaction_day"].dt.to_period("M")
    
    if pos.empty:
        f["monthly_revenue_slope"] = np.nan
        f["monthly_revenue_cv"] = np.nan
        f["active_months_fraction"] = 0.0
        return f
    
    monthly = pos.groupby(["customer_id", "month"])["rev"].sum().reset_index()
    
    # Need at least 3 months for trend
    month_counts = monthly.groupby("customer_id")["month"].nunique()
    valid_customers = month_counts[month_counts >= 3].index
    
    if len(valid_customers) == 0:
        f["monthly_revenue_slope"] = np.nan
        f["monthly_revenue_cv"] = np.nan
        f["active_months_fraction"] = (month_counts > 0).reindex(index).astype(int)
        return f
    
    # Compute slope on log(1+rev) for each customer
    slopes = {}
    cvs = {}
    for cust_id in valid_customers:
        cust_months = monthly[monthly["customer_id"] == cust_id].sort_values("month")
        if len(cust_months) >= 3:
            # Convert period to numeric (months since first)
            x = np.arange(len(cust_months), dtype=float)
            y = np.log1p(cust_months["rev"].values)
            if np.ptp(x) > 0 and np.ptp(y) > 0:
                slope = np.polyfit(x, y, 1)[0]
                slopes[cust_id] = slope
            cvs[cust_id] = np.std(cust_months["rev"]) / np.mean(cust_months["rev"]) if np.mean(cust_months["rev"]) > 0 else np.nan
    
    f["monthly_revenue_slope"] = pd.Series(slopes).reindex(index)
    f["monthly_revenue_cv"] = pd.Series(cvs).reindex(index)
    
    # Active months fraction (observed months / total possible months in window)
    # Window: last 12 months or available history
    window_start = as_of - pd.DateOffset(months=12)
    total_months = ((as_of.to_period("M") - window_start.to_period("M")).n)
    f["active_months_fraction"] = (month_counts / total_months).reindex(index).fillna(0).clip(0, 1)
    
    return f


def _basket_composition_features(
    transactions: pd.DataFrame,
    trips: pd.DataFrame,
    index: pd.Index,
    as_of: pd.Timestamp,
) -> pd.DataFrame:
    """Basket size, breadth, and concentration features."""
    f = pd.DataFrame(index=index)
    
    # Positive purchases only
    pos = transactions[
        (transactions["quantity"] > 0) 
        & transactions["customer_id"].notna() 
        & (transactions["transaction_day"] <= as_of)
    ].copy()
    
    if pos.empty:
        f["mean_distinct_products_per_trip"] = 0.0
        f["mean_distinct_categories_per_trip"] = 0.0
        f["mean_units_per_trip"] = 0.0
        f["trip_revenue_cv"] = np.nan
        return f
    
    # Per-trip metrics
    event_col = "transaction_day"  # Default to trip grain
    trip_products = pos.groupby([pos["customer_id"], pos[event_col]])["product_id"].nunique()
    trip_categories = pos[pos["category"].notna()].groupby([pos["customer_id"], pos[event_col]])["category"].nunique()
    trip_units = pos.groupby([pos["customer_id"], pos[event_col]])["quantity"].sum()
    trip_revenue = pos["line_revenue"].clip(lower=0).fillna(0).groupby([pos["customer_id"], pos[event_col]]).sum()
    
    f["mean_distinct_products_per_trip"] = trip_products.groupby(level=0).mean().reindex(index)
    f["mean_distinct_categories_per_trip"] = trip_categories.groupby(level=0).mean().reindex(index)
    f["mean_units_per_trip"] = trip_units.groupby(level=0).mean().reindex(index)
    f["trip_revenue_cv"] = (trip_revenue.groupby(level=0).std() / trip_revenue.groupby(level=0).mean()).reindex(index)
    
    return f


# Convenience function for backward compatibility
def build_customer_features(
    transactions: pd.DataFrame,
    as_of: pd.Timestamp,
    event_grain: str = "trip",
    windows_days: Sequence[int] = (30, 90, 365),
    recent_window_days: int = 90,
) -> pd.DataFrame:
    """Alias for build_customer_snapshot with explicit parameter names."""
    return build_customer_snapshot(
        transactions=transactions,
        as_of=as_of,
        event_grain=event_grain,
        windows_days=windows_days,
        recent_window_days=recent_window_days,
    )