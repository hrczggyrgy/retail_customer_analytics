"""Retail Customer Analytics — Streamlit application.

Purpose:
    Turn transaction-level retail data into decision-support analytics.

Data contract:
    invoice_id, product_id, description, department, category,
    quantity, transaction_datetime, unit_price, customer_id

Architecture:
    Monolithic Streamlit app organized by numbered sections:
    00. Module docstring and imports
    01. Constants and configuration
    02. Data schema and validation
    03. Generic formatting helpers
    04. Period and filter context
    05. Core transaction metrics
    06. Customer analytics
    07. Retention and lifecycle analytics
    08. Product and category analytics
    09. Basket and cross-sell analytics
    10. Anomaly analytics
    11. Action generation
    12. Matplotlib plotting helpers
    13. Streamlit UI components
    14. Tab renderers
    15. Main application entry point

    Core analytics: retail_customer_analysis.py
    Chart/report generation: insight_engine.py
    Synthetic data: data_generator.py

Important distinctions:
    - Observed historical revenue is not predicted customer value.
    - Inactivity is not confirmed churn.
    - Product association is not proven causation.
    - Price variation is not confirmed promotion exposure.

Run:
    pip install streamlit pandas numpy scipy matplotlib
    streamlit run app.py

Validation:
    python app.py --validate
"""

from __future__ import annotations

import io
import json
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

# Import shared modules (requires src on path)
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, str(APP_DIR / "src"))

# ruff: noqa: E402
from retail_customer_analytics.ingestion import (
    fmt_currency,
    fmt_number,
    fmt_pct,
)
from retail_customer_analytics.ingestion import (
    prepare_transaction_frame as canonical_prepare_transaction_frame,
)
from retail_customer_analytics.affinity import analyze_affinity_from_app

# ruff: noqa: E402
from retail_customer_analytics.metrics import (
    compute_period_metrics,
)

# ruff: noqa: E402
from retail_customer_analytics.period_manager import (
    compute_preset_date_range,
    ensure_not_empty,
    get_period_context,
    get_period_transactions,
)

# ============================================================
# 01. Constants and configuration
# ============================================================
# Per-metric favorable direction for delta coloring
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

APP_CONFIG = {
    "default_period_days": 28,
    "default_inactivity_window_days": 90,
    "anomaly_window_days": 28,
    "anomaly_sensitivity": 2.0,
    "anomaly_minimum_history_days": 14,
    "basket_min_support": 0.01,
    "basket_min_pair_count": 5,
    "basket_high_lift_threshold": 2.0,
    "high_value_customer_percentile": 0.20,
    "max_chart_points": 500,
    "max_table_rows": 100,
}

REQUIRED = [
    "customer_id",
    "transaction_date",
    "transaction_id",
    "product_id",
    "product_description",
    "department",
    "category",
    "price",
    "quantity",
]

TABS = [
    "1 Overview",
    "2 Trust",
    "3 Customers",
    "4 Retention & Behaviour",
    "5 Products & Baskets",
    "6 Explorer & Data",
]

METRIC_DEFINITIONS = {
    "revenue": {
        "formula": "sum(quantity * unit_price)",
        "interpretation": "Total observed sales value.",
        "scope": "Positive purchases unless otherwise stated.",
    },
    "orders": {
        "formula": "count(distinct transaction_id)",
        "interpretation": "Number of distinct transactions.",
        "scope": "Positive purchases only.",
    },
    "active_customers": {
        "formula": "count(distinct customer_id)",
        "interpretation": "Customers with at least one purchase in period.",
        "scope": "Current period only.",
    },
    "aov": {
        "formula": "revenue / orders",
        "interpretation": "Average order value.",
        "scope": "Positive purchases in period.",
    },
    "units_per_order": {
        "formula": "sum(quantity) / orders",
        "interpretation": "Average units per transaction.",
        "scope": "Positive purchases in period.",
    },
    "revenue_per_customer": {
        "formula": "revenue / active_customers",
        "interpretation": "Average revenue per active customer.",
        "scope": "Current period only.",
    },
}
CHARTS_BY_TAB = {
    "trust": ["01_model_validation", "02_calibration", "05_cluster_selection"],
    "customers": [
        "03_customer_map",
        "04_segment_value",
        "06_rfm_segments",
        "09_revenue_concentration",
        "18_lorenz_segments",
        "19_action_quadrant",
    ],
    "behaviour": [
        "07_repeat_survival",
        "08_cohort_retention",
        "14_momentum",
        "16_cadence",
        "15_promo_proxy",
        "22_cohort_ltv",
    ],
    "products": [
        "10_revenue_trend",
        "11_categories",
        "12_basket_rules",
        "13_next_trip",
        "20_affinity_network",
        "21_promo_return_lollipop",
    ],
    "data": ["17_data_quality"],
}
HEADLINE_CHARTS = [
    "01_model_validation",
    "03_customer_map",
    "04_segment_value",
    "07_repeat_survival",
    "09_revenue_concentration",
    "12_basket_rules",
    "18_lorenz_segments",
    "19_action_quadrant",
]
SEGMENT_HINTS = {
    "Active / higher": "Hypothesis: protect and grow. Test retention perks against a holdout group.",
    "Active / lower": "Hypothesis: raise basket value or breadth. Test cross-category offers.",
    "Lapsing": "Hypothesis: win-back candidates. Test timed reminders against a holdout group.",
    "Lapsed": "Hypothesis: low-cost reactivation only. Check cost against expected revenue first.",
}

# ============================================================
# 02. Data schema and validation
# ============================================================

# ============================================================
# 03. Generic formatting helpers
# ============================================================


# ============================================================
# Validation Baseline - Phase 0
# ============================================================
def run_cleanup_validation() -> dict:
    """
    Run validation baseline on fixed synthetic dataset.

    Returns dict with 'passed' (bool), 'results' (dict), and 'expected' (dict).
    Call via: python app.py --validate
    """
    import numpy as np
    import pandas as pd

    # Fixed seed for reproducible synthetic data
    seed = 42
    n_customers = 500
    days = 365

    # Generate fixed synthetic data
    rng = np.random.default_rng(seed)
    k = 8  # categories
    cats = [f"Category {i:02d}" for i in range(k)]
    depts = [f"Department {i % 3 + 1}" for i in range(k)]
    prod_per_cat = 8
    base_price = {c: float(rng.lognormal(np.log(4 + 1.6 * c), 0.25)) for c in range(k)}
    products = {
        c: [
            (f"P{c:02d}{j:02d}", round(base_price[c] * float(rng.uniform(0.7, 1.4)), 2))
            for j in range(prod_per_cat)
        ]
        for c in range(k)
    }
    lam = rng.gamma(0.8, 1 / 12.0, n_customers)
    p_drop = rng.beta(0.8, 2.5, n_customers)
    first = rng.integers(0, max(int(days * 0.7), 1), n_customers)
    pref = rng.dirichlet(np.ones(k) * 0.4, n_customers)
    size_factor = rng.gamma(2.0, 0.8, n_customers)
    t0 = pd.Timestamp("2024-01-01")
    rows: list[tuple] = []
    for i in range(n_customers):
        horizon = days - first[i]
        n_buy = int(rng.geometric(p_drop[i]))
        gaps = rng.exponential(1 / lam[i], size=n_buy)
        times = np.concatenate([[0.0], np.cumsum(gaps)]) + first[i]
        times = times[times < days]
        for t in times:
            day = t0 + pd.Timedelta(days=int(t))
            n_items = 1 + rng.poisson(size_factor[i])
            chosen = set(rng.choice(k, size=min(n_items, k), replace=True, p=pref[i]))
            if 0 in chosen and k > 1 and rng.random() < 0.7:
                chosen.add(1)
            receipt = rng.integers(0, 2) if rng.random() < 0.15 else 0
            for c in chosen:
                pid, price = products[c][rng.integers(prod_per_cat)]
                if rng.random() < 0.15:
                    price = round(price * 0.8, 2)
                qty = int(1 + rng.poisson(0.6))
                cid = None if rng.random() < 0.03 else f"C{i:05d}"
                tid = f"T{i:05d}_{int(t)}_{receipt}"
                rows.append(
                    (
                        cid,
                        day.strftime("%Y-%m-%d"),
                        tid,
                        pid,
                        f"Product {pid}",
                        depts[c],
                        cats[c],
                        price,
                        qty,
                    )
                )
                if rng.random() < 0.03:
                    rday = day + pd.Timedelta(days=int(rng.integers(1, 15)))
                    if (rday - t0).days < days:
                        rows.append(
                            (
                                cid,
                                rday.strftime("%Y-%m-%d"),
                                tid + "R",
                                pid,
                                f"Product {pid}",
                                depts[c],
                                cats[c],
                                price,
                                -1,
                            )
                        )

    df = pd.DataFrame(rows, columns=REQUIRED)
    df["customer_id"] = df["customer_id"].astype("string")
    df["transaction_id"] = df["transaction_id"].astype("string")
    df["product_id"] = df["product_id"].astype("string")
    df["product_description"] = df["product_description"].astype("string")
    df["department"] = df["department"].astype("string")
    df["category"] = df["category"].astype("string")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    df["transaction_day"] = df["transaction_date"].dt.normalize()

    # Filter to positive quantity for core metrics (returns handled separately)
    pos_df = df[df["quantity"] > 0].copy()
    pos_df["revenue"] = pos_df["quantity"] * pos_df["price"]

    # --- Expected values (computed once, saved as reference) ---
    # These are the "golden master" values for the fixed synthetic dataset (seed=42, n=500, days=365)
    # Generated on 2026-10-10 from actual computation
    expected = {
        # Core transaction metrics (last 28 days)
        "core_metrics": {
            "revenue": 5822.43,
            "orders": 193,
            "customers": 83,
            "aov": 30.17,
            "units_per_order": 3.33,
            "avg_unit_price": 9.06,
            "revenue_per_customer": 70.15,
            "orders_per_customer": 2.33,
            "units_sold": 643,
        },
        # Prior period comparison (28 days before)
        "prior_metrics": {
            "revenue": 6606.32,
            "orders": 224,
            "customers": 101,
            "aov": 29.49,
            "units_per_order": 3.48,
            "avg_unit_price": 8.48,
            "revenue_per_customer": 65.41,
            "orders_per_customer": 2.22,
            "units_sold": 779,
        },
        # Year-over-year (same period last year - not applicable for 365-day data, use prior)
        "yoy_metrics": {
            "revenue": 6606.32,
            "orders": 224,
            "customers": 101,
            "aov": 29.49,
            "units_per_order": 3.48,
            "avg_unit_price": 8.48,
            "revenue_per_customer": 65.41,
            "orders_per_customer": 2.22,
            "units_sold": 779,
        },
        # RFM (as of last date)
        "rfm": {
            "total_customers": 498,
            "segment_counts": {
                "Champions": 111,
                "Loyal Customers": 24,
                "Potential Loyalists": 86,
                "New Customers": 26,
                "At Risk High Value": 62,
                "At Risk": 16,
                "Hibernating": 91,
                "Needs Attention": 82,
            },
            "total_monetary": 81735.61,
        },
        # Retention (90-day inactivity window)
        "retention": {
            "retention_rate": 0.49,
            "reactivation_rate": 0.06,
            "repeat_purchase_rate": 0.87,
            "repeat_revenue_share": 0.95,
            "new_customer_share": 0.00,
            "inactive_high_value_count": 56,
            "inactive_high_value_revenue": 21149.80,
        },
        # Product metrics (current period)
        "product_metrics": {
            "total_products": 64,
            "top_product_revenue_share": 0.043171,
            "top_5_revenue_share": 0.185622,
        },
        # Basket analysis
        "basket": {
            "total_pairs": 30,
            "pairs_above_min_support": 30,
            "avg_lift": 7.46,
            "max_lift": 19.30,
        },
        # Anomaly detection (28-day window, 2 sigma)
        "anomalies": {
            "anomaly_count": 6,
            "anomaly_metrics": ["revenue", "orders", "customers", "aov"],
            "max_deviation_pct": 125.18,
        },
        # Action matrix
        "actions": {
            "action_count": 5,
            "top_priority": "Retention",
        },
    }

    # --- Run actual computations ---
    results = {}
    tolerance = 0.05  # 5% tolerance for floating point

    def _compare(actual: float, expected_val: float, name: str) -> bool:
        if expected_val == 0:
            return actual == 0
        rel_diff = abs(actual - expected_val) / abs(expected_val)
        passed = rel_diff <= tolerance
        if not passed:
            print(
                f"  FAIL {name}: actual={actual:.2f}, expected={expected_val:.2f}, diff={rel_diff * 100:.1f}%"
            )
        else:
            print(f"  OK   {name}: {actual:.2f} ~ {expected_val:.2f}")
        return passed

    # Use last 28 days as current period
    max_date = pos_df["transaction_day"].max()
    current_end = max_date
    current_start = current_end - pd.Timedelta(days=27)
    prior_end = current_start - pd.Timedelta(days=1)
    prior_start = prior_end - pd.Timedelta(days=27)

    # Current period metrics
    curr = pos_df[
        (pos_df["transaction_day"] >= current_start) & (pos_df["transaction_day"] <= current_end)
    ]
    prior = pos_df[
        (pos_df["transaction_day"] >= prior_start) & (pos_df["transaction_day"] <= prior_end)
    ]

    # Use metrics module for consistency
    curr_metrics = compute_period_metrics(curr, current_start, current_end, use_polars=False)
    prior_metrics = compute_period_metrics(prior, prior_start, prior_end, use_polars=False)

    all_passed = True

    print("=== CORE METRICS (CURRENT) ===")
    for k, v in expected["core_metrics"].items():
        if not _compare(curr_metrics[k], v, f"core.{k}"):
            all_passed = False

    print("\n=== PRIOR PERIOD METRICS ===")
    for k, v in expected["prior_metrics"].items():
        if not _compare(prior_metrics[k], v, f"prior.{k}"):
            all_passed = False

    # Waterfall decomposition
    print("\n=== REVENUE WATERFALL ===")
    c_cust = curr_metrics["customers"]
    c_freq = curr_metrics["orders_per_customer"]
    c_aov = curr_metrics["aov"]
    c_upo = curr_metrics["units_per_order"]
    c_aup = curr_metrics["avg_unit_price"]
    p_cust = prior_metrics["customers"]
    p_freq = prior_metrics["orders_per_customer"]
    p_aov = prior_metrics["aov"]
    p_upo = prior_metrics["units_per_order"]
    p_aup = prior_metrics["avg_unit_price"]

    rev_current = curr_metrics["revenue"]
    rev_prior = prior_metrics["revenue"]

    cust_effect = (c_cust - p_cust) * p_freq * p_aov
    freq_effect = c_cust * (c_freq - p_freq) * p_aov
    aov_effect = c_cust * c_freq * (c_aov - p_aov)
    upo_effect = c_cust * c_freq * (c_upo - p_upo) * p_aup
    aup_effect = c_cust * c_freq * c_upo * (c_aup - p_aup)

    waterfall_data = [
        {"label": "Prior Period Revenue", "value": rev_prior, "type": "total"},
        {"label": "Customer Count", "value": cust_effect, "type": "driver"},
        {"label": "Purchase Frequency", "value": freq_effect, "type": "driver"},
        {"label": "Units per Order", "value": upo_effect, "type": "subdriver"},
        {"label": "Avg Unit Price", "value": aup_effect, "type": "subdriver"},
        {"label": "Current Period Revenue", "value": rev_current, "type": "total"},
    ]
    results["waterfall"] = waterfall_data
    print(f"  Revenue change: {rev_current - rev_prior:.2f}")
    print(f"  Decomposed sum: {cust_effect + freq_effect + aov_effect:.2f}")
    print(f"  Residual: {(rev_current - rev_prior) - (cust_effect + freq_effect + aov_effect):.2f}")

    # RFM
    print("\n=== RFM ===")
    analysis_date = current_end
    rfm_df = pos_df[pos_df["transaction_day"] <= analysis_date].copy()
    rfm = (
        rfm_df.groupby("customer_id")
        .agg(
            last_purchase=("transaction_day", "max"),
            first_purchase=("transaction_day", "min"),
            frequency=("transaction_id", "nunique"),
            monetary=("revenue", "sum"),
        )
        .reset_index()
    )
    rfm["recency_days"] = (analysis_date - rfm["last_purchase"]).dt.days
    rfm["tenure_days"] = (rfm["last_purchase"] - rfm["first_purchase"]).dt.days
    rfm["aov"] = rfm["monetary"] / rfm["frequency"]

    # Simple RFM scoring (quintiles)
    def safe_qcut(series, q=5, reverse=False):
        unique_vals = series.nunique()
        if unique_vals < 2:
            return pd.Series((q + 1) // 2, index=series.index, dtype=int)
        actual_q = min(q, unique_vals)
        try:
            ranked = series.rank(method="average")
            bins = pd.qcut(ranked, q=actual_q, labels=False, duplicates="drop")
            n_bins = bins.max() + 1
            bins = bins + 1
            if reverse:
                bins = n_bins + 1 - bins
            return bins.astype(int)
        except ValueError:
            return pd.Series((q + 1) // 2, index=series.index, dtype=int)

    rfm["R_score"] = safe_qcut(rfm["recency_days"], q=5, reverse=True)
    rfm["F_score"] = safe_qcut(rfm["frequency"], q=5)
    rfm["M_score"] = safe_qcut(rfm["monetary"], q=5)
    rfm["RFM_score"] = (
        rfm["R_score"].astype(str) + rfm["F_score"].astype(str) + rfm["M_score"].astype(str)
    )

    def assign_segment(row):
        r, f, m = row["R_score"], row["F_score"], row["M_score"]
        if r >= 4 and f >= 4 and m >= 4:
            return "Champions"
        elif r >= 3 and f >= 4 and m >= 3:
            return "Loyal Customers"
        elif r >= 4 and f <= 2:
            return "New Customers"
        elif r >= 3 and f >= 3:
            return "Potential Loyalists"
        elif r <= 2 and f >= 3 and m >= 3:
            return "At Risk High Value"
        elif r <= 2 and f >= 2:
            return "At Risk"
        elif r <= 2 and f <= 2 and m <= 2:
            return "Hibernating"
        else:
            return "Needs Attention"

    rfm["segment"] = rfm.apply(assign_segment, axis=1)

    seg_counts = rfm["segment"].value_counts().to_dict()
    total_monetary = rfm["monetary"].sum()

    if not _compare(len(rfm), expected["rfm"]["total_customers"], "rfm.total_customers"):
        all_passed = False
    if not _compare(total_monetary, expected["rfm"]["total_monetary"], "rfm.total_monetary"):
        all_passed = False
    for seg, exp_count in expected["rfm"]["segment_counts"].items():
        if not _compare(seg_counts.get(seg, 0), exp_count, f"rfm.segment.{seg}"):
            all_passed = False

    # Retention metrics
    print("\n=== RETENTION ===")
    inactivity_window_days = 90
    curr_customers = set(curr["customer_id"].dropna().unique())
    prior_customers = set(prior["customer_id"].dropna().unique())
    all_time_df = pos_df[pos_df["transaction_day"] < current_start].copy()
    all_time_customers = set(all_time_df["customer_id"].dropna().unique())

    retained = curr_customers & prior_customers
    retention_rate = len(retained) / len(prior_customers) if prior_customers else 0

    dormancy_cutoff = current_start - pd.Timedelta(days=inactivity_window_days)
    recently_active = set(
        pos_df[
            (pos_df["transaction_day"] < current_start)
            & (pos_df["transaction_day"] >= dormancy_cutoff)
        ]["customer_id"]
        .dropna()
        .unique()
    )
    dormant = all_time_customers - recently_active
    reactivated = dormant & curr_customers
    reactivation_rate = len(reactivated) / len(dormant) if dormant else 0

    new_customers = curr_customers - all_time_customers
    new_customer_share = len(new_customers) / len(curr_customers) if curr_customers else 0

    all_purchases = pos_df[pos_df["transaction_day"] <= current_end].copy()
    cust_orders = all_purchases.groupby("customer_id")["transaction_id"].nunique()
    repeat_cust = (cust_orders > 1).sum()
    total_cust = len(cust_orders)
    repeat_purchase_rate = repeat_cust / total_cust if total_cust > 0 else 0

    all_purchases["revenue"] = all_purchases["quantity"] * all_purchases["price"]
    repeat_revenue = all_purchases[
        all_purchases["customer_id"].isin(cust_orders[cust_orders > 1].index)
    ]["revenue"].sum()
    total_revenue = all_purchases["revenue"].sum()
    repeat_revenue_share = repeat_revenue / total_revenue if total_revenue > 0 else 0

    hist_rev = all_purchases.groupby("customer_id")["revenue"].sum().sort_values(ascending=False)
    top_20_pct = int(len(hist_rev) * 0.2)
    valuable_cust = set(hist_rev.head(top_20_pct).index) if top_20_pct > 0 else set()
    inactive_valuable = valuable_cust - curr_customers
    inactive_valuable_revenue = (
        hist_rev.loc[list(inactive_valuable)].sum() if inactive_valuable else 0
    )

    if not _compare(
        retention_rate, expected["retention"]["retention_rate"], "retention.retention_rate"
    ):
        all_passed = False
    if not _compare(
        reactivation_rate, expected["retention"]["reactivation_rate"], "retention.reactivation_rate"
    ):
        all_passed = False
    if not _compare(
        repeat_purchase_rate,
        expected["retention"]["repeat_purchase_rate"],
        "retention.repeat_purchase_rate",
    ):
        all_passed = False
    if not _compare(
        repeat_revenue_share,
        expected["retention"]["repeat_revenue_share"],
        "retention.repeat_revenue_share",
    ):
        all_passed = False
    if not _compare(
        new_customer_share,
        expected["retention"]["new_customer_share"],
        "retention.new_customer_share",
    ):
        all_passed = False
    if not _compare(
        len(inactive_valuable),
        expected["retention"]["inactive_high_value_count"],
        "retention.inactive_high_value_count",
    ):
        all_passed = False
    if not _compare(
        inactive_valuable_revenue,
        expected["retention"]["inactive_high_value_revenue"],
        "retention.inactive_high_value_revenue",
    ):
        all_passed = False

    # Product metrics
    print("\n=== PRODUCT METRICS ===")
    prod_rev = curr.groupby("product_id")["revenue"].sum().sort_values(ascending=False)
    total_prod_rev = prod_rev.sum()
    top_1_share = prod_rev.iloc[0] / total_prod_rev if len(prod_rev) > 0 else 0
    top_5_share = prod_rev.head(5).sum() / total_prod_rev if len(prod_rev) >= 5 else 1.0

    if not _compare(
        len(prod_rev), expected["product_metrics"]["total_products"], "product.total_products"
    ):
        all_passed = False
    if not _compare(
        top_1_share, expected["product_metrics"]["top_product_revenue_share"], "product.top_1_share"
    ):
        all_passed = False
    if not _compare(
        top_5_share, expected["product_metrics"]["top_5_revenue_share"], "product.top_5_share"
    ):
        all_passed = False

    # Basket analysis
    print("\n=== BASKET ANALYSIS ===")
    total_orders = curr["transaction_id"].nunique()
    order_products = curr.groupby("transaction_id")["product_id"].apply(list).reset_index()
    order_products.columns = ["transaction_id", "products"]

    prod_metrics = (
        curr.groupby("product_id")
        .agg(
            orders=("transaction_id", "nunique"),
            revenue=("revenue", "sum"),
            units=("quantity", "sum"),
        )
        .reset_index()
    )
    prod_metrics["support"] = prod_metrics["orders"] / total_orders

    min_support = 0.01
    freq_products = prod_metrics[prod_metrics["support"] >= min_support]["product_id"].tolist()

    from itertools import combinations

    pairs = []
    for _, row in order_products.iterrows():
        products_in_order = sorted({p for p in row["products"] if p in freq_products})
        if len(products_in_order) >= 2:
            for a, b in combinations(products_in_order, 2):
                pairs.append((a, b))

    pair_df = pd.DataFrame(pairs, columns=["product_a", "product_b"])
    if not pair_df.empty:
        pair_counts = (
            pair_df.groupby(["product_a", "product_b"]).size().reset_index(name="co_occurrence")
        )
        pair_counts["support"] = pair_counts["co_occurrence"] / total_orders
        pair_counts = pair_counts[pair_counts["support"] >= min_support].copy()

        pair_counts = pair_counts.merge(
            prod_metrics[["product_id", "orders"]].rename(
                columns={"product_id": "product_a", "orders": "orders_a"}
            ),
            on="product_a",
            how="left",
        )
        pair_counts = pair_counts.merge(
            prod_metrics[["product_id", "orders"]].rename(
                columns={"product_id": "product_b", "orders": "orders_b"}
            ),
            on="product_b",
            how="left",
        )
        pair_counts["confidence_a_to_b"] = pair_counts["co_occurrence"] / pair_counts["orders_a"]
        pair_counts["lift"] = pair_counts["confidence_a_to_b"] / (
            pair_counts["orders_b"] / total_orders
        )

        total_pairs = len(pair_counts)
        avg_lift = pair_counts["lift"].mean() if total_pairs > 0 else 0
        max_lift = pair_counts["lift"].max() if total_pairs > 0 else 0

        if not _compare(total_pairs, expected["basket"]["total_pairs"], "basket.total_pairs"):
            all_passed = False
        if not _compare(
            total_pairs,
            expected["basket"]["pairs_above_min_support"],
            "basket.pairs_above_min_support",
        ):
            all_passed = False
        if not _compare(avg_lift, expected["basket"]["avg_lift"], "basket.avg_lift"):
            all_passed = False
        if not _compare(max_lift, expected["basket"]["max_lift"], "basket.max_lift"):
            all_passed = False
    else:
        print("  SKIP basket: no pairs found")

    # Anomaly detection
    print("\n=== ANOMALY DETECTION ===")
    full_start = current_start - pd.Timedelta(days=56)
    full_end = current_end
    daily_df = pos_df[
        (pos_df["transaction_day"] >= full_start) & (pos_df["transaction_day"] <= full_end)
    ].copy()

    daily = (
        daily_df.groupby("transaction_day")
        .agg(
            revenue=("revenue", "sum"),
            orders=("transaction_id", "nunique"),
            customers=("customer_id", "nunique"),
            units=("quantity", "sum"),
        )
        .reset_index()
    )
    daily["aov"] = daily["revenue"] / daily["orders"].replace(0, np.nan)

    window_days = 28
    k_sigma = 2.0
    metrics = ["revenue", "orders", "customers", "aov"]
    total_anomalies = 0
    anomaly_metrics = []
    max_dev = 0

    for metric in metrics:
        if metric not in daily.columns or daily[metric].isna().all():
            continue
        history = daily[metric].shift(1)
        rolling_mean = history.rolling(window=window_days, min_periods=7).mean()
        rolling_std = history.rolling(window=window_days, min_periods=7).std()
        upper = rolling_mean + k_sigma * rolling_std
        lower = rolling_mean - k_sigma * rolling_std

        curr_mask = (daily["transaction_day"] >= current_start) & (
            daily["transaction_day"] <= current_end
        )
        curr_data = daily[curr_mask].copy()

        for _, row in curr_data.iterrows():
            idx = row.name
            val = row[metric]
            exp = rolling_mean.loc[idx]
            up = upper.loc[idx]
            lo = lower.loc[idx]

            if pd.notna(val) and pd.notna(exp) and (val > up or val < lo):
                total_anomalies += 1
                pct_dev = abs((val - exp) / exp * 100) if exp != 0 else 0
                max_dev = max(max_dev, pct_dev)
                if metric not in anomaly_metrics:
                    anomaly_metrics.append(metric)

    if not _compare(total_anomalies, expected["anomalies"]["anomaly_count"], "anomalies.count"):
        all_passed = False
    if not _compare(max_dev, expected["anomalies"]["max_deviation_pct"], "anomalies.max_deviation"):
        all_passed = False

    # Action matrix (simplified check)
    print("\n=== ACTION MATRIX ===")
    action_count = 0
    if len(inactive_valuable) > 0:
        action_count += 1
    at_risk = rfm[rfm["segment"].isin(["At Risk High Value", "At Risk"])]
    if len(at_risk) > 0:
        action_count += 1
    if new_customer_share > 0.2:
        action_count += 1
    if (
        total_pairs > 0
        and "pair_counts" in locals()
        and len(pair_counts[pair_counts["lift"] > 2.0]) > 0
    ):
        action_count += 1
    if total_anomalies > 0:
        action_count += 1
    champions = rfm[rfm["segment"] == "Champions"]
    if len(champions) > 0:
        action_count += 1

    if not _compare(action_count, expected["actions"]["action_count"], "actions.count"):
        all_passed = False

    results = {
        "core_metrics": curr_metrics,
        "prior_metrics": prior_metrics,
        "rfm": {
            "total_customers": len(rfm),
            "segment_counts": seg_counts,
            "total_monetary": total_monetary,
        },
        "retention": {
            "retention_rate": retention_rate,
            "reactivation_rate": reactivation_rate,
            "repeat_purchase_rate": repeat_purchase_rate,
            "repeat_revenue_share": repeat_revenue_share,
            "new_customer_share": new_customer_share,
            "inactive_high_value_count": len(inactive_valuable),
            "inactive_high_value_revenue": inactive_valuable_revenue,
        },
        "product_metrics": {
            "total_products": len(prod_rev),
            "top_1_share": top_1_share,
            "top_5_share": top_5_share,
        },
        "basket": {
            "total_pairs": total_pairs if "total_pairs" in locals() else 0,
            "avg_lift": avg_lift if "avg_lift" in locals() else 0,
            "max_lift": max_lift if "max_lift" in locals() else 0,
        },
        "anomalies": {
            "anomaly_count": total_anomalies,
            "anomaly_metrics": anomaly_metrics,
            "max_deviation_pct": max_dev,
        },
        "actions": {"action_count": action_count},
    }

    print(f"\n=== VALIDATION {'PASSED' if all_passed else 'FAILED'} ===")
    return {"passed": all_passed, "results": results, "expected": expected}


def _run_validation_cli():
    """CLI entry point for validation."""
    print("Running cleanup validation baseline...")
    result = run_cleanup_validation()
    if result["passed"]:
        print("\n✅ All validation checks PASSED")
        sys.exit(0)
    else:
        print("\n❌ Some validation checks FAILED")
        sys.exit(1)


# ============================================================
# 04. Period and filter context
# ============================================================
# ============================================================
# 05. Core transaction metrics
# ============================================================


def section(title: str, question: str = "") -> None:
    """Render a section header with optional business question."""
    st.subheader(title)
    if question:
        st.caption(question)


def chart_card(fig, takeaway: str, *, key: str = "", scope: str = "") -> None:
    """Render a matplotlib figure with a takeaway caption and close the figure.

    Args:
        fig: matplotlib figure
        takeaway: markdown caption explaining the chart
        key: optional key for tracking
        scope: optional scope label (e.g., "Selected period · Positive purchases")
    """
    if scope:
        st.caption(f"📊 Scope: {scope}")
    st.pyplot(fig, use_container_width=True, clear_figure=True)
    if takeaway:
        st.markdown(f"*{takeaway}*")
    if key:
        st.markdown(f"`{key}`")
    import matplotlib.pyplot as plt

    plt.close(fig)


def kpi_row(metrics: list[tuple[str, str, str | None]]) -> None:
    """Render a row of KPI metrics with optional deltas.
    metrics: list of (label, value, delta) tuples
    """
    if not metrics:
        return
    cols = st.columns(len(metrics))
    for col, (label, value, delta) in zip(cols, metrics, strict=False):
        col.metric(label, value, delta=delta)


# Segment colour map for consistent visual identity
SEGMENT_COLORS = {
    "Champions": "#2b6cb0",
    "Active / higher": "#2b6cb0",
    "Loyal / high value": "#2f855a",
    "Active / lower": "#38a169",
    "Potential loyalists": "#4299e1",
    "Recent / low frequency": "#4299e1",
    "Frequent / loyal": "#805ad5",
    "Previously high-value / lapsed": "#dd6b20",
    "At Risk": "#e53e3e",
    "At risk by recency": "#e53e3e",
    "Lapsing": "#e53e3e",
    "Hibernating / low frequency": "#a0aec0",
    "Lapsed": "#a0aec0",
    "Mixed / needs attention": "#718096",
    "Unclustered": "#718096",
}


def segment_color(label: str) -> str:
    """Get consistent colour for a segment label."""
    return SEGMENT_COLORS.get(str(label), "#4a5568")


# ============================================================
# 06. Customer analytics
# ============================================================
def kpi_tile_row(metrics: list[dict]) -> None:
    """Render a row of executive KPI tiles with current, prior, change, pct_change.

    metrics: list of dicts with keys:
        - label: metric name
        - current: current period value (formatted string)
        - prior: comparison period value (formatted string)
        - change: absolute change (formatted string)
        - pct_change: percentage change (formatted string, e.g. "+5.2%")
        - trend: "up" | "down" | "neutral" for color coding
        - help_text: optional tooltip
    """
    if not metrics:
        return

    n = len(metrics)
    cols = st.columns(n)

    for col, m in zip(cols, metrics, strict=False):
        # Use per-metric favorable direction instead of assuming "up" is always good
        favorable = FAVORABLE_DIRECTION.get(m.get("label", ""), "up")
        trend = m.get("trend", "neutral")
        delta_color = (
            "normal" if trend == favorable else ("inverse" if trend != "neutral" else "off")
        )

        col.metric(
            label=m["label"],
            value=m["current"],
            delta=f"{m['change']} ({m['pct_change']})" if m.get("change") else None,
            delta_color=delta_color,
            help=m.get("help_text", ""),
        )


def period_selector_sidebar(df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp, str]:
    """Global period selector in sidebar. Returns (current_start, current_end, compare_mode).

    compare_mode: "prior" | "yoy" | "custom"
    """
    if df.empty or "transaction_day" not in df.columns:
        return None, None, "prior"

    min_date = df["transaction_day"].min()
    max_date = df["transaction_day"].max()

    with st.sidebar.expander("📅 Period Selection", expanded=True):
        # Preset options
        preset = st.selectbox(
            "Current period",
            [
                "Last 7 days",
                "Last 28 days",
                "Last 90 days",
                "Month to date",
                "Quarter to date",
                "Custom range",
            ],
            index=1,
        )

        if preset == "Custom range":
            date_range = st.date_input(
                "Custom date range",
                value=(max_date - pd.Timedelta(days=27), max_date),
                min_value=min_date,
                max_value=max_date,
            )
            if len(date_range) == 2:
                current_start = pd.Timestamp(date_range[0])
                current_end = pd.Timestamp(date_range[1])
            else:
                current_start = max_date - pd.Timedelta(days=27)
                current_end = max_date
        else:
            current_start, current_end = compute_preset_date_range(preset, max_date)

        st.caption(
            f"Current: {current_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')}"
        )

        # Comparison period
        compare_mode = st.selectbox(
            "Compare to",
            ["Prior period (equal length)", "Same period last year", "Custom comparison"],
            index=0,
        )

        if compare_mode == "Custom comparison":
            comp_range = st.date_input(
                "Comparison date range",
                value=(
                    current_start - pd.Timedelta(days=(current_end - current_start).days + 1),
                    current_start - pd.Timedelta(days=1),
                ),
                min_value=min_date,
                max_value=max_date,
            )
            if len(comp_range) == 2:
                st.session_state["custom_compare_start"] = pd.Timestamp(comp_range[0])
                st.session_state["custom_compare_end"] = pd.Timestamp(comp_range[1])
            compare_mode = "custom"
        else:
            compare_mode = "prior" if compare_mode == "Prior period (equal length)" else "yoy"

    return current_start, current_end, compare_mode


# ============================================================
# 07. Retention and lifecycle analytics
# ============================================================
def prepare_transaction_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize transaction DataFrame once per analysis run.

    Uses the canonical ingestion module for consistent preprocessing.

    Returns DataFrame with explicit columns and dtypes.
    """
    # Map app column names to canonical names
    col_map = {
        "invoice_id": "transaction_id",
        "description": "product_description",
        "unit_price": "price",
        "transaction_datetime": "transaction_date",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Use canonical preparation
    df = canonical_prepare_transaction_frame(df)

    # Ensure revenue column exists (alias for line_revenue)
    if "line_revenue" in df.columns:
        df["revenue"] = df["line_revenue"]

    return df


# ============================================================
# Customer Analytics - Canonical Feature Builder (Phase 3)
# ============================================================
# This is now a thin wrapper around the canonical builder in features.py
# to maintain API compatibility with existing app code.
def build_customer_features(
    transactions: pd.DataFrame,
    analysis_end_date: pd.Timestamp,
    inactivity_window_days: int = 90,
) -> pd.DataFrame:
    """
    Build one canonical row per customer from transaction history.

    This wraps the canonical builder from retail_customer_analytics.features
    and adds app-specific transformations (RFM segments, lifecycle, txn_ prefix).

    Business rules:
        - Frequency is the number of distinct invoices (transaction_id) - uses transaction grain.
        - Monetary value is observed historical revenue (positive quantity only).
        - Recency is days since last purchase as of analysis_end_date.
        - Tenure is days between first and last purchase.
        - Dormancy requires no purchase during the inactivity window before analysis_end_date.

    Inputs:
        transactions: Normalized transaction-level DataFrame (from prepare_transaction_frame).
        analysis_end_date: Date used to calculate recency and lifecycle.
        inactivity_window_days: Days without purchase before dormancy.

    Returns:
        DataFrame with one row per customer, columns prefixed:
        - txn_: Transaction-derived (observed) fields
        - lifecycle_: Lifecycle status fields
        - segment_rfm: RFM segment label

    Does not include:
        Model-derived fields (p_alive, expected_revenue, etc.) - those come from pipeline.
    """
    # Use canonical builder with transaction grain to match app's invoice-level frequency
    from retail_customer_analytics.features import build_customer_snapshot

    # Canonical builder expects 'transaction_day' column and returns customer_id as index
    cust = build_customer_snapshot(
        transactions=transactions,
        as_of=analysis_end_date,
        event_grain="transaction",  # App uses invoice-level frequency
        windows_days=(30, 90, 365),
        recent_window_days=inactivity_window_days,
    )

    if cust.empty:
        return pd.DataFrame()

    # Transform canonical columns to app's expected txn_ prefixed columns
    cust = cust.reset_index()

    # Map columns
    column_map = {
        "customer_id": "customer_id",
        "n_trips": "txn_frequency",
        "gross_spend": "txn_monetary",
        "recency_days": "txn_recency_days",
        "first_day": "txn_first_purchase",
        "last_day": "txn_last_purchase",
        "mean_trip_value": "txn_aov",
        "n_products": "txn_distinct_products",  # Note: this is per-trip avg, not lifetime
        "unique_products_purchased": "txn_distinct_products",
        "unique_categories_purchased": "txn_distinct_categories",
        "unique_departments_purchased": "txn_distinct_departments",
        "top_category_by_spend": "top_category_by_spend",
        "top_department_by_spend": "top_department_by_spend",
    }

    # Rename existing columns
    for old, new in column_map.items():
        if old in cust.columns:
            cust[new] = cust[old]

    # Compute tenure
    if "txn_first_purchase" in cust.columns and "txn_last_purchase" in cust.columns:
        cust["txn_tenure_days"] = (cust["txn_last_purchase"] - cust["txn_first_purchase"]).dt.days

    # RFM Scoring (quintiles 1-5, 5 is best) - TIE-SAFE
    def safe_qcut_score(series, q=5, reverse=False):
        unique_vals = series.nunique()
        if unique_vals < 2:
            return pd.Series((q + 1) // 2, index=series.index, dtype=int)
        actual_q = min(q, unique_vals)
        try:
            ranked = series.rank(method="average")
            bins = pd.qcut(ranked, q=actual_q, labels=False, duplicates="drop")
            n_bins = bins.max() + 1
            bins = bins + 1
            if reverse:
                bins = n_bins + 1 - bins
            return bins.astype(int)
        except ValueError:
            return pd.Series((q + 1) // 2, index=series.index, dtype=int)

    # Recency: lower is better -> reverse score
    cust["txn_r_score"] = safe_qcut_score(cust["txn_recency_days"], q=5, reverse=True)
    # Frequency: higher is better
    cust["txn_f_score"] = safe_qcut_score(cust["txn_frequency"], q=5)
    # Monetary: higher is better
    cust["txn_m_score"] = safe_qcut_score(cust["txn_monetary"], q=5)

    cust["txn_rfm_score"] = (
        cust["txn_r_score"].astype(str)
        + cust["txn_f_score"].astype(str)
        + cust["txn_m_score"].astype(str)
    )

    # RFM Segment mapping (matching app's compute_rfm segments)
    def assign_rfm_segment(row):
        r, f, m = row["txn_r_score"], row["txn_f_score"], row["txn_m_score"]
        if r >= 4 and f >= 4 and m >= 4:
            return "Champions"
        elif r >= 3 and f >= 4 and m >= 3:
            return "Loyal Customers"
        elif r >= 4 and f <= 2:
            return "New Customers"
        elif r >= 3 and f >= 3:
            return "Potential Loyalists"
        elif r <= 2 and f >= 3 and m >= 3:
            return "At Risk High Value"
        elif r <= 2 and f >= 2:
            return "At Risk"
        elif r <= 2 and f <= 2 and m <= 2:
            return "Hibernating"
        else:
            return "Needs Attention"

    cust["segment_rfm"] = cust.apply(assign_rfm_segment, axis=1)

    # Value tier (percentile-based) - handle case where all values are the same
    monetary_ranked = cust["txn_monetary"].rank(method="first")
    if monetary_ranked.nunique() > 1:
        cust["segment_value_tier"] = pd.qcut(
            monetary_ranked,
            q=[0, 0.01, 0.05, 0.10, 0.20, 0.70, 1.0],
            labels=["Top 1%", "Next 4%", "Next 5%", "Next 10%", "Middle 50%", "Bottom 30%"],
            duplicates="drop",
        )
    else:
        # All customers have same monetary value - assign middle tier
        cust["segment_value_tier"] = "Middle 50%"

    # Lifecycle classification
    cust = classify_customer_lifecycle(cust, analysis_end_date, inactivity_window_days)

    return cust


def classify_customer_lifecycle(
    customer_features: pd.DataFrame,
    analysis_end_date: pd.Timestamp,
    inactivity_window_days: int = 90,
) -> pd.DataFrame:
    """
    Classify customer lifecycle status based on transaction history.

    Definitions (consistent, not dependent on comparison period):
        - New: First purchase within the analysis period
        - Active: Purchased in current period AND not dormant
        - Retained: Active in both current and prior period
        - Dormant: No purchase in [analysis_end_date - inactivity_window_days, analysis_end_date)
        - Reactivated: Was dormant, purchased in current period
        - Inactive High-Value: Top 20% by historical monetary, not active in current period

    Args:
        customer_features: DataFrame from build_customer_features()
        analysis_end_date: End of current analysis period
        inactivity_window_days: Days of inactivity before dormancy

    Returns:
        DataFrame with lifecycle_* columns added.
    """
    df = customer_features.copy()

    # New customers: first purchase is at or after analysis_end_date - inactivity_window_days
    # (simplified: first purchase in the current period)
    # We'll use a more precise definition: first purchase within the last period_len_days
    # For now, use a simple heuristic based on recency
    df["lifecycle_is_new"] = df["txn_recency_days"] <= inactivity_window_days
    # Actually, "new" means first purchase ever is recent
    # We need the first purchase date - let's use a simpler approach
    # New = first purchase within inactivity_window_days of analysis_end_date
    df["lifecycle_is_new"] = df["txn_first_purchase"] >= (
        analysis_end_date - pd.Timedelta(days=inactivity_window_days)
    )

    # Dormant: no purchase in the inactivity window before analysis_end_date
    dormancy_cutoff = analysis_end_date - pd.Timedelta(days=inactivity_window_days)
    df["lifecycle_is_dormant"] = df["txn_last_purchase"] < dormancy_cutoff

    # Active: not dormant (has purchase in the inactivity window)
    df["lifecycle_is_active"] = ~df["lifecycle_is_dormant"]

    # Lifecycle status (single label)
    def assign_lifecycle(row):
        if row["lifecycle_is_new"]:
            return "New"
        elif row["lifecycle_is_dormant"]:
            return "Dormant"
        else:
            return "Active"

    df["lifecycle_status"] = df.apply(assign_lifecycle, axis=1)

    # Retained/Reactivated will be computed in _compute_retention_metrics with period context
    # Those are period-relative, not absolute lifecycle states

    return df


def merge_model_features(
    txn_features: pd.DataFrame,
    model_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge transaction-derived features with model-derived features from pipeline.

    Uses explicit prefixes to avoid confusion:
    - txn_: Transaction-derived (observed)
    - model_: Model-derived (predicted)
    - segment_: Segment labels
    - lifecycle_: Lifecycle status

    Args:
        txn_features: Output from build_customer_features()
        model_features: Pipeline's customer_features table

    Returns:
        Combined DataFrame with all features.
    """
    if model_features.empty:
        return txn_features

    # Select model columns to merge (avoid duplicates)
    model_cols = ["customer_id"]
    # Model-derived fields
    for col in [
        "p_alive",
        "expected_trips_horizon",
        "expected_trip_value",
        "expected_revenue_horizon",
        "bg_nbd_r",
        "bg_nbd_alpha",
        "gg_p",
        "gg_q",
        "gg_gamma",
    ]:
        if col in model_features.columns:
            model_cols.append(col)

    # Pipeline RFM segment (different scoring method)
    if "rfm_segment" in model_features.columns:
        model_cols.append("rfm_segment")
    if "rfm_recency_score" in model_features.columns:
        model_cols.append("rfm_recency_score")
    if "rfm_frequency_score" in model_features.columns:
        model_cols.append("rfm_frequency_score")
    if "rfm_monetary_score" in model_features.columns:
        model_cols.append("rfm_monetary_score")
    if "rfm_score" in model_features.columns:
        model_cols.append("rfm_score")

    # Cluster/segment
    if "cluster_label" in model_features.columns:
        model_cols.append("cluster_label")
    if "top_category_by_spend" in model_features.columns:
        model_cols.append("top_category_by_spend")
    if "top_department_by_spend" in model_features.columns:
        model_cols.append("top_department_by_spend")

    # Promo/return features
    for col in ["promo_spend_share", "return_value_ratio", "return_lines"]:
        if col in model_features.columns:
            model_cols.append(col)

    # Cohort
    if "observed_first_purchase_cohort_month" in model_features.columns:
        model_cols.append("observed_first_purchase_cohort_month")

    model_df = model_features[model_cols].copy()

    # Rename model columns with model_ prefix (except customer_id and segment labels)
    rename_map = {}
    for col in model_df.columns:
        if col not in [
            "customer_id",
            "rfm_segment",
            "cluster_label",
            "observed_first_purchase_cohort_month",
        ]:
            rename_map[col] = f"model_{col}"
    model_df = model_df.rename(columns=rename_map)

    # Merge
    merged = txn_features.merge(model_df, on="customer_id", how="left")

    return merged


# ============================================================
# Product & Category Analytics - Canonical Feature Builders (Phase 4)
# ============================================================
def build_product_features(
    transactions: pd.DataFrame,
    context: dict,
    period: str = "current",
) -> pd.DataFrame:
    """
    Build one canonical row per product for a given period.

    Business rules:
        - Revenue is sum of quantity * price (positive quantity only).
        - Orders is distinct transaction_ids containing the product.
        - Units is sum of quantity.
        - Penetration is orders / total orders in period.
        - Revenue per order is revenue / orders.
        - Average selling price is revenue / units.
        - Current vs prior comparison uses period context.

    Args:
        transactions: Normalized transaction DataFrame (from prepare_transaction_frame).
        context: Period context from get_period_context().
        period: "current" or "prior"

    Returns:
        DataFrame with one row per product, including:
        - Product identifiers and descriptions
        - Revenue, units, orders, customers
        - Penetration, revenue per order, avg selling price
        - Prior period comparison (if period="current" and prior exists)
        - Growth, share, rank, rank change
    """
    df = get_period_transactions(transactions, context, period)

    if not ensure_not_empty(df, f"product features {period}"):
        return pd.DataFrame()

    total_orders = df["transaction_id"].nunique()
    total_revenue = df["revenue"].sum()

    # Product-level aggregation
    prod = (
        df.groupby("product_id")
        .agg(
            revenue=("revenue", "sum"),
            orders=("transaction_id", "nunique"),
            units=("quantity", "sum"),
            customers=("customer_id", "nunique"),
        )
        .reset_index()
    )

    # Merge descriptions
    if "product_description" in transactions.columns:
        desc_map = transactions[
            ["product_id", "product_description", "category", "department"]
        ].drop_duplicates()
        prod = prod.merge(desc_map, on="product_id", how="left")
    else:
        prod["product_description"] = prod["product_id"]
        prod["category"] = "Unknown"
        prod["department"] = "Unknown"

    # Derived metrics
    prod["basket_penetration"] = prod["orders"] / total_orders * 100 if total_orders > 0 else 0
    prod["rev_per_order"] = prod["revenue"] / prod["orders"].replace(0, np.nan)
    prod["avg_selling_price"] = prod["revenue"] / prod["units"].replace(0, np.nan)
    prod["revenue_share"] = prod["revenue"] / total_revenue * 100 if total_revenue > 0 else 0

    # Current period: add prior comparison
    if period == "current" and "prior_start" in context:
        prior_df = get_period_transactions(transactions, context, "prior")
        if not prior_df.empty:
            prior_prod = (
                prior_df.groupby("product_id")
                .agg(
                    prior_revenue=("revenue", "sum"),
                    prior_orders=("transaction_id", "nunique"),
                    prior_units=("quantity", "sum"),
                )
                .reset_index()
            )
            prod = prod.merge(prior_prod, on="product_id", how="left")
            prod["prior_revenue"] = prod["prior_revenue"].fillna(0)
            prod["prior_orders"] = prod["prior_orders"].fillna(0)
            prod["prior_units"] = prod["prior_units"].fillna(0)

            prod["revenue_change"] = prod["revenue"] - prod["prior_revenue"]
            prod["pct_change"] = (
                prod["revenue_change"] / prod["prior_revenue"].replace(0, np.nan) * 100
            )

            # Ranks
            prod["revenue_rank"] = prod["revenue"].rank(ascending=False, method="min").astype(int)
            prod["prior_revenue_rank"] = (
                prod["prior_revenue"].rank(ascending=False, method="min").astype(int)
            )
            prod["rank_change"] = (
                prod["prior_revenue_rank"] - prod["revenue_rank"]
            )  # positive = improved

    return prod


def build_category_features(
    transactions: pd.DataFrame,
    context: dict,
    period: str = "current",
) -> pd.DataFrame:
    """
    Build one canonical row per category for a given period.

    Args:
        transactions: Normalized transaction DataFrame.
        context: Period context from get_period_context().
        period: "current" or "prior"

    Returns:
        DataFrame with one row per category, including revenue, orders, units,
        customers, growth, share, and contribution to total change.
    """
    df = get_period_transactions(transactions, context, period)

    if not ensure_not_empty(df, f"category features {period}"):
        return pd.DataFrame()

    total_revenue = df["revenue"].sum()

    cat = (
        df.groupby("category")
        .agg(
            revenue=("revenue", "sum"),
            orders=("transaction_id", "nunique"),
            units=("quantity", "sum"),
            customers=("customer_id", "nunique"),
        )
        .reset_index()
    )

    cat["revenue_share"] = cat["revenue"] / total_revenue * 100 if total_revenue > 0 else 0
    cat["avg_order_value"] = cat["revenue"] / cat["orders"].replace(0, np.nan)

    # Current period: add prior comparison
    if period == "current" and "prior_start" in context:
        prior_df = get_period_transactions(transactions, context, "prior")
        if not prior_df.empty:
            prior_cat = (
                prior_df.groupby("category")
                .agg(
                    prior_revenue=("revenue", "sum"),
                )
                .reset_index()
            )
            cat = cat.merge(prior_cat, on="category", how="left")
            cat["prior_revenue"] = cat["prior_revenue"].fillna(0)

            cat["revenue_change"] = cat["revenue"] - cat["prior_revenue"]
            cat["growth"] = cat["revenue_change"] / cat["prior_revenue"].replace(0, np.nan) * 100

            # Contribution to total revenue change
            total_change = cat["revenue_change"].sum()
            cat["contribution_pct"] = (
                (cat["revenue_change"] / total_change * 100) if total_change != 0 else 0
            )

    return cat


def build_basket_associations(
    transactions: pd.DataFrame,
    context: dict,
    min_support: float = 0.01,
    min_pair_count: int = 5,
) -> dict:
    """
    Build product pair associations for a period using the shared affinity engine.

    Args:
        transactions: Normalized transaction DataFrame.
        context: Period context from get_period_context().
        min_support: Minimum support threshold (fraction of orders).
        min_pair_count: Minimum co-occurrence count (used for compatibility).

    Returns:
        Dict with:
        - "product_metrics": Single product metrics (support, orders, revenue)
        - "pairs": Pair associations (support, confidence, lift, joint_revenue)
        - "parameters": Parameters used for reproducibility
        - "diagnostics": Run diagnostics from shared engine
    """
    # Use shared affinity engine
    result = analyze_affinity_from_app(
        raw_df=transactions,
        period_params=context,
        min_support=min_support,
        basket_level="product",
        basket_grain="trip",
        max_candidates=100,
        max_baskets=5000,
        max_items_per_basket=30,
        max_pair_operations=100_000,
    )

    # Rename columns to match expected format
    pairs = result.get("pairs", pd.DataFrame())
    if not pairs.empty:
        pairs = pairs.rename(columns={
            "antecedent": "product_a",
            "consequent": "product_b",
            "pair_baskets": "co_occurrence",
            "basket_support": "support",
            "confidence": "confidence_a_to_b",
            "basket_lift": "lift",
        })

    return {
        "product_metrics": result.get("single", pd.DataFrame()),
        "pairs": pairs,
        "parameters": {"min_support": min_support, "min_pair_count": min_pair_count},
        "diagnostics": result.get("diagnostics", {}),
    }


# Currency configuration - can be overridden via config or environment
DEFAULT_CURRENCY = "EUR"
CURRENCY_SYMBOLS = {"EUR": "€", "USD": "$", "GBP": "£", "JPY": "¥"}


def revenue_waterfall_data(current: dict, prior: dict) -> list[dict]:
    """Compute revenue waterfall decomposition.

    Revenue = Customers × Orders/Customer × AOV
    AOV = Units/Order × Avg Unit Price
    """
    # Base components
    c_cust = current["customers"]
    c_freq = current["orders_per_customer"]
    c_aov = current["aov"]
    c_upo = current["units_per_order"]
    c_aup = current["avg_unit_price"]

    p_cust = prior["customers"]
    p_freq = prior["orders_per_customer"]
    p_aov = prior["aov"]
    p_upo = prior["units_per_order"]
    p_aup = prior["avg_unit_price"]

    # Revenue totals
    rev_current = current["revenue"]
    rev_prior = prior["revenue"]

    # Decomposition: change each driver while holding others at prior levels
    # Customer count effect
    cust_effect = (c_cust - p_cust) * p_freq * p_aov
    # Frequency effect (orders per customer)
    freq_effect = c_cust * (c_freq - p_freq) * p_aov
    # AOV effect
    aov_effect = c_cust * c_freq * (c_aov - p_aov)

    # AOV sub-decomposition
    upo_effect = c_cust * c_freq * (c_upo - p_upo) * p_aup
    aup_effect = c_cust * c_freq * c_upo * (c_aup - p_aup)

    # Residual (interaction effects)
    total_decomposed = cust_effect + freq_effect + aov_effect
    (rev_current - rev_prior) - total_decomposed

    return [
        {"label": "Prior Period Revenue", "value": rev_prior, "type": "total"},
        {"label": "Customer Count", "value": cust_effect, "type": "driver"},
        {"label": "Purchase Frequency", "value": freq_effect, "type": "driver"},
        {"label": "Units per Order", "value": upo_effect, "type": "subdriver"},
        {"label": "Avg Unit Price", "value": aup_effect, "type": "subdriver"},
        {"label": "Current Period Revenue", "value": rev_current, "type": "total"},
    ]


# ============================================================
# 10. Anomaly analytics
# ============================================================

# ============================================================
# 11. Action generation
# ============================================================

# ============================================================
# 12. Matplotlib plotting helpers
# ============================================================


def plot_revenue_waterfall(
    waterfall_data: list[dict], currency_fmt: callable = fmt_currency
) -> tuple:
    """Create matplotlib waterfall chart for revenue decomposition."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 5))

    labels = [d["label"] for d in waterfall_data]
    values = [d["value"] for d in waterfall_data]
    types = [d["type"] for d in waterfall_data]

    # Calculate running totals for bar positions
    running = 0
    bar_bottoms = []
    bar_heights = []

    for i, (v, t) in enumerate(zip(values, types, strict=False)):
        if t == "total":
            if i == 0:  # First total (prior)
                bar_bottoms.append(0)
                bar_heights.append(v)
                running = v
            else:  # Last total (current)
                bar_bottoms.append(0)
                bar_heights.append(v)
        elif t == "driver":
            bar_bottoms.append(running)
            bar_heights.append(v)
            running += v
        elif t == "subdriver":
            bar_bottoms.append(running)
            bar_heights.append(v)
            running += v

    # Colors
    colors = []
    for t in types:
        if t == "total":
            colors.append("#1a365d")  # Dark blue
        elif t == "driver":
            colors.append("#2b6cb0" if bar_heights[colors.__len__()] >= 0 else "#e53e3e")
        else:  # subdriver
            colors.append("#4299e1" if bar_heights[colors.__len__()] >= 0 else "#fc8181")

    x_pos = range(len(labels))
    bars = ax.bar(
        x_pos,
        bar_heights,
        bottom=bar_bottoms,
        color=colors,
        edgecolor="white",
        linewidth=0.5,
        width=0.6,
    )

    # Add value labels on bars
    for _, (bar, v, t) in enumerate(zip(bars, values, types, strict=False)):
        height = bar.get_height()
        bottom = bar.get_y()
        center = bottom + height / 2

        if t == "total":
            label = currency_fmt(v)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bottom + height + max(abs(v) for v in values) * 0.02,
                label,
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=10,
            )
        else:
            label = f"{'+' if v >= 0 else ''}{currency_fmt(v)}"
            color = "black" if abs(height) > max(abs(v) for v in values) * 0.1 else "white"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                center,
                label,
                ha="center",
                va="center",
                fontsize=9,
                fontweight="medium",
                color=color,
            )

    # Connecting lines between bars
    for i in range(len(bars) - 1):
        if types[i] != "total" and types[i + 1] != "total":
            x1 = bars[i].get_x() + bars[i].get_width()
            y1 = bars[i].get_y() + bars[i].get_height()
            x2 = bars[i + 1].get_x()
            y2 = bars[i + 1].get_y()
            ax.plot([x1, x2], [y1, y2], "k--", alpha=0.3, linewidth=0.5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("Revenue", fontsize=11)
    ax.set_title(
        "Revenue Change Decomposition: What Drove the Change?",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: currency_fmt(x)))
    ax.axhline(y=0, color="gray", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    return fig, ax


def plot_kpi_sparklines(
    periods: list[str], values: list[list], labels: list[str], colors: list[str] = None
) -> tuple:
    """Create small multiples sparkline chart for KPI trends."""
    import matplotlib.pyplot as plt

    n = len(values)
    if colors is None:
        colors = ["#2b6cb0", "#2f855a", "#dd6b20", "#6b46c1", "#975a16", "#d53f8c"]

    fig, axes = plt.subplots(1, n, figsize=(3 * n, 2.5), sharex=True)
    if n == 1:
        axes = [axes]

    for i, (ax, vals, label) in enumerate(zip(axes, values, labels, strict=False)):
        color = colors[i % len(colors)]
        ax.plot(range(len(vals)), vals, color=color, linewidth=2, marker="o", markersize=4)
        ax.fill_between(range(len(vals)), vals, alpha=0.1, color=color)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_xticks(range(len(periods)))
        ax.set_xticklabels(periods, rotation=30, ha="right", fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3)

        # Highlight last point
        ax.plot(
            len(vals) - 1,
            vals[-1],
            "o",
            color=color,
            markersize=8,
            markerfacecolor="white",
            markeredgewidth=2,
        )

        # Add value label for last point
        ax.annotate(
            f"{vals[-1]:,.0f}",
            xy=(len(vals) - 1, vals[-1]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            color=color,
        )

    plt.tight_layout()
    return fig, axes


def compute_rfm(df: pd.DataFrame, analysis_date: pd.Timestamp) -> pd.DataFrame:
    """Compute RFM metrics for each customer as of analysis_date.

    Assumes df is normalized via prepare_transaction_frame() with pre-calculated revenue column.
    """
    # Filter to analysis date
    df = df[df["transaction_day"] <= analysis_date].copy()

    # Use pre-calculated revenue if available, otherwise compute as fallback
    if "revenue" not in df.columns:
        df["revenue"] = df["quantity"] * df["price"]

    # Customer-level aggregation
    rfm = (
        df.groupby("customer_id")
        .agg(
            last_purchase=("transaction_day", "max"),
            first_purchase=("transaction_day", "min"),
            frequency=("transaction_id", "nunique"),
            monetary=("revenue", "sum"),
            total_quantity=("quantity", "sum"),
        )
        .reset_index()
    )

    # Recency in days
    rfm["recency_days"] = (analysis_date - rfm["last_purchase"]).dt.days
    rfm["tenure_days"] = (rfm["last_purchase"] - rfm["first_purchase"]).dt.days
    rfm["aov"] = rfm["monetary"] / rfm["frequency"]

    # RFM Scoring (quintiles 1-5, 5 is best) - TIE-SAFE
    # Use rank(method='average') then qcut to spread tied values across their percentile range
    # This ensures identical values get identical scores while preserving discrimination

    def safe_qcut_score(series, q=5, reverse=False):
        """Tie-safe quantile scoring using rank(method='average') + qcut.

        Tied values get the average rank, which places them in the middle of their percentile range.
        Then qcut creates bins. This gives better discrimination than qcut on raw values
        while maintaining tie-safety (identical values -> identical scores).
        """
        unique_vals = series.nunique()
        if unique_vals < 2:
            return pd.Series((q + 1) // 2, index=series.index, dtype=int)

        actual_q = min(q, unique_vals)
        try:
            # Rank with average method for ties (tied values get same average rank)
            ranked = series.rank(method="average")
            # qcut on ranks - this spreads tied values properly
            bins = pd.qcut(ranked, q=actual_q, labels=False, duplicates="drop")
            n_bins = bins.max() + 1
            bins = bins + 1

            if reverse:
                bins = n_bins + 1 - bins

            return bins.astype(int)
        except ValueError:
            return pd.Series((q + 1) // 2, index=series.index, dtype=int)

    # Recency: lower is better -> reverse score
    rfm["R_score"] = safe_qcut_score(rfm["recency_days"], q=5, reverse=True)
    # Frequency: higher is better
    rfm["F_score"] = safe_qcut_score(rfm["frequency"], q=5)
    # Monetary: higher is better
    rfm["M_score"] = safe_qcut_score(rfm["monetary"], q=5)

    rfm["RFM_score"] = (
        rfm["R_score"].astype(str) + rfm["F_score"].astype(str) + rfm["M_score"].astype(str)
    )

    # Segment mapping
    def assign_segment(row):
        r, f, m = row["R_score"], row["F_score"], row["M_score"]
        if r >= 4 and f >= 4 and m >= 4:
            return "Champions"
        elif r >= 3 and f >= 4 and m >= 3:
            return "Loyal Customers"
        elif r >= 4 and f <= 2:
            return "New Customers"
        elif r >= 3 and f >= 3:
            return "Potential Loyalists"
        elif r <= 2 and f >= 3 and m >= 3:
            return "At Risk High Value"
        elif r <= 2 and f >= 2:
            return "At Risk"
        elif r <= 2 and f <= 2 and m <= 2:
            return "Hibernating"
        else:
            return "Needs Attention"

    rfm["segment"] = rfm.apply(assign_segment, axis=1)

    return rfm


def plot_rfm_segments(rfm: pd.DataFrame) -> tuple:
    """Create RFM visualizations: segment bars + quadrant scatter."""
    import matplotlib.pyplot as plt

    # Check required columns exist
    required_cols = ["segment", "customer_id", "monetary", "recency_days", "frequency"]
    missing = [c for c in required_cols if c not in rfm.columns]
    if missing:
        # Return empty figure with error message
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        for ax in axes:
            ax.text(
                0.5,
                0.5,
                f"Missing columns for RFM plot: {missing}\nRequired: segment, customer_id, monetary, recency_days, frequency",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=10,
                color="red",
            )
            ax.set_axis_off()
        return fig, axes

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # 1. Segment summary bars
    seg_summary = (
        rfm.groupby("segment")
        .agg(
            customers=("customer_id", "count"),
            revenue=("monetary", "sum"),
            avg_recency=("recency_days", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_monetary=("monetary", "mean"),
        )
        .sort_values("revenue", ascending=False)
    )

    colors = [segment_color(s) for s in seg_summary.index]
    bars = axes[0].barh(
        range(len(seg_summary)), seg_summary["revenue"], color=colors, edgecolor="white", height=0.6
    )
    axes[0].set_yticks(range(len(seg_summary)))
    axes[0].set_yticklabels(
        [f"{s} ({c:,})" for s, c in zip(seg_summary.index, seg_summary["customers"], strict=False)],
        fontsize=9,
    )
    axes[0].set_xlabel("Revenue", fontsize=10)
    axes[0].set_title("Revenue by RFM Segment", fontsize=12, fontweight="bold")
    axes[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: fmt_currency(x)))
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    # Add revenue labels
    for bar, rev in zip(bars, seg_summary["revenue"], strict=False):
        axes[0].text(
            bar.get_width() + max(seg_summary["revenue"]) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            fmt_currency(rev),
            va="center",
            fontsize=9,
        )

    # 2. Quadrant scatter: Recency vs Monetary, sized by Frequency
    ax = axes[1]
    for seg in rfm["segment"].unique():
        s = rfm[rfm["segment"] == seg]
        # Drop rows with NaN in required columns to ensure same-size arrays
        s = s.dropna(subset=["recency_days", "monetary", "frequency"])
        if len(s) == 0:
            continue
        ax.scatter(
            s["recency_days"],
            s["monetary"],
            s=np.clip(s["frequency"] * 10, 20, 300),
            alpha=0.5,
            label=seg,
            color=segment_color(seg),
            edgecolors="white",
            linewidth=0.3,
        )

    # Quadrant lines
    med_r = rfm["recency_days"].median()
    med_m = rfm["monetary"].median()
    ax.axvline(med_r, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.axhline(med_m, color="gray", linestyle="--", alpha=0.5, linewidth=1)

    # Quadrant labels
    ax.text(
        0.05,
        0.95,
        "Champions\n(Recent, High Value)",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        bbox={"boxstyle": "round", "facecolor": "#c6f6d5", "alpha": 0.5},
    )
    ax.text(
        0.95,
        0.95,
        "At Risk\nHigh Value",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        ha="right",
        bbox={"boxstyle": "round", "facecolor": "#fed7d7", "alpha": 0.5},
    )
    ax.text(
        0.05,
        0.05,
        "New/Low Value",
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
        bbox={"boxstyle": "round", "facecolor": "#bee3f8", "alpha": 0.5},
    )
    ax.text(
        0.95,
        0.05,
        "Hibernating\nLow Value",
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
        ha="right",
        bbox={"boxstyle": "round", "facecolor": "#feebc8", "alpha": 0.5},
    )

    ax.set_xlabel("Recency (days since last purchase)", fontsize=10)
    ax.set_ylabel("Monetary Value (Total Revenue)", fontsize=10)
    ax.set_title(
        "Customer Map: Recency vs Value (bubble = Frequency)", fontsize=12, fontweight="bold"
    )
    ax.set_yscale("log")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)

    plt.tight_layout()
    return fig, axes


def plot_pareto(
    df: pd.DataFrame,
    value_col: str,
    label_col: str,
    top_n: int = 20,
    title: str = "Pareto Analysis",
) -> tuple:
    """Create Pareto chart with cumulative percentage of TOTAL business."""
    import matplotlib.pyplot as plt

    # Calculate total from FULL population before truncating
    total_value = df[value_col].sum()

    pareto = (
        df.nlargest(top_n, value_col).sort_values(value_col, ascending=False).reset_index(drop=True)
    )
    # Cumulative share of TOTAL business, not just displayed subset
    pareto["cum_pct"] = pareto[value_col].cumsum() / total_value * 100
    pareto["rank"] = range(1, len(pareto) + 1)

    fig, ax1 = plt.subplots(figsize=(12, 5))

    # Bar chart
    ax1.bar(range(len(pareto)), pareto[value_col], color="#2b6cb0", edgecolor="white", width=0.7)
    ax1.set_ylabel(value_col.replace("_", " ").title(), fontsize=10, color="#2b6cb0")
    ax1.tick_params(axis="y", labelcolor="#2b6cb0")

    # Labels
    ax1.set_xticks(range(len(pareto)))
    ax1.set_xticklabels(
        [str(label_val)[:20] for label_val in pareto[label_col]],
        rotation=45,
        ha="right",
        fontsize=8,
    )

    # Cumulative line - shows share of TOTAL business
    ax2 = ax1.twinx()
    ax2.plot(
        range(len(pareto)),
        pareto["cum_pct"],
        color="#e53e3e",
        marker="o",
        linewidth=2,
        markersize=4,
    )
    ax2.set_ylabel("Cumulative % of Total Business", fontsize=10, color="#e53e3e")
    ax2.tick_params(axis="y", labelcolor="#e53e3e")
    ax2.axhline(y=80, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax2.text(len(pareto) - 1, 82, "80% threshold", fontsize=8, color="gray")

    # Add note about total share captured
    total_share = pareto["cum_pct"].iloc[-1] if len(pareto) > 0 else 0
    ax1.text(
        0.02,
        0.98,
        f"Top {len(pareto)} = {total_share:.1f}% of total",
        transform=ax1.transAxes,
        fontsize=9,
        va="top",
        ha="left",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8, "edgecolor": "gray"},
    )

    ax1.set_title(title, fontsize=13, fontweight="bold", pad=15)
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    plt.tight_layout()
    return fig, (ax1, ax2)


def plot_cohort_retention_heatmap(cohort_data: pd.DataFrame) -> tuple:
    """Plot cohort retention heatmap from analysis output."""
    import matplotlib.pyplot as plt

    # cohort_data expected from analysis: cohort_month, months_since_cohort, retention_rate
    if cohort_data.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(
            0.5, 0.5, "No cohort data available", ha="center", va="center", transform=ax.transAxes
        )
        return fig, ax

    # Pivot
    pivot = cohort_data.pivot(
        index="cohort_month", columns="months_since_cohort", values="retention_rate"
    )
    pivot = pivot.sort_index()

    fig, ax = plt.subplots(figsize=(12, max(4, 0.3 * len(pivot) + 2)))

    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"M{c}" for c in pivot.columns], fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(d)[:10] for d in pivot.index], fontsize=9)

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            if not np.isnan(val):
                color = "white" if val < 0.5 else "black"
                ax.text(
                    j, i, f"{val * 100:.0f}%", ha="center", va="center", fontsize=8, color=color
                )

    ax.set_xlabel("Months Since First Purchase", fontsize=10)
    ax.set_ylabel("Cohort (First Purchase Month)", fontsize=10)
    ax.set_title("Cohort Retention Heatmap", fontsize=13, fontweight="bold", pad=15)

    from matplotlib.ticker import PercentFormatter

    cbar = plt.colorbar(im, ax=ax, format=PercentFormatter(xmax=1.0))
    cbar.set_label("Retention Rate", fontsize=9)

    plt.tight_layout()
    return fig, ax


# Need numpy for some functions


# --------------------------------------------------------------------------- #
# Synthetic demo data
# --------------------------------------------------------------------------- #
def make_synthetic(
    n_customers: int = 3000,
    days: int = 730,
    seed: int = 7,
    missing_id_rate: float = 0.03,
    return_rate: float = 0.03,
    promo_share: float = 0.15,
    n_categories: int = 12,
    start: str = "2024-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    """Line-item data from a BG/NBD-like process with category preferences, promotions, returns and split receipts."""
    rng = np.random.default_rng(seed)
    k = int(n_categories)
    cats = [f"Category {i:02d}" for i in range(k)]
    depts = [f"Department {i % 3 + 1}" for i in range(k)]
    prod_per_cat = 10
    base_price = {c: float(rng.lognormal(np.log(4 + 1.6 * c), 0.25)) for c in range(k)}
    products = {
        c: [
            (f"P{c:02d}{j:02d}", round(base_price[c] * float(rng.uniform(0.7, 1.4)), 2))
            for j in range(prod_per_cat)
        ]
        for c in range(k)
    }
    lam = rng.gamma(0.8, 1 / 12.0, n_customers)
    p_drop = rng.beta(0.8, 2.5, n_customers)
    first = rng.integers(0, max(int(days * 0.7), 1), n_customers)
    pref = rng.dirichlet(np.ones(k) * 0.4, n_customers)
    size_factor = rng.gamma(2.0, 0.8, n_customers)
    t0 = pd.Timestamp(start)
    # Default end to today to avoid future-dated demo data
    if end is None:
        end_ts = pd.Timestamp.now().normalize()
    else:
        end_ts = pd.Timestamp(end)
    max_days = (end_ts - t0).days
    if max_days <= 0:
        max_days = days
    rows: list[tuple] = []
    for i in range(n_customers):
        horizon = max_days - first[i]
        if horizon <= 0:
            continue
        n_buy = int(rng.geometric(p_drop[i]))
        gaps = rng.exponential(1 / lam[i], size=n_buy)
        times = np.concatenate([[0.0], np.cumsum(gaps)]) + first[i]
        times = times[times < max_days]
        for t in times:
            day = t0 + pd.Timedelta(days=int(t))
            n_items = 1 + rng.poisson(size_factor[i])
            chosen = set(rng.choice(k, size=min(n_items, k), replace=True, p=pref[i]))
            if 0 in chosen and k > 1 and rng.random() < 0.7:
                chosen.add(1)
            receipt = rng.integers(0, 2) if rng.random() < 0.15 else 0
            for c in chosen:
                pid, price = products[c][rng.integers(prod_per_cat)]
                if rng.random() < promo_share:
                    price = round(price * 0.8, 2)
                qty = int(1 + rng.poisson(0.6))
                cid = None if rng.random() < missing_id_rate else f"C{i:05d}"
                tid = f"T{i:05d}_{int(t)}_{receipt}"
                rows.append(
                    (
                        cid,
                        day.strftime("%Y-%m-%d"),
                        tid,
                        pid,
                        f"Product {pid}",
                        depts[c],
                        cats[c],
                        price,
                        qty,
                    )
                )
                if rng.random() < return_rate:
                    rday = day + pd.Timedelta(days=int(rng.integers(1, 15)))
                    if (rday - t0).days < max_days:
                        rows.append(
                            (
                                cid,
                                rday.strftime("%Y-%m-%d"),
                                tid + "R",
                                pid,
                                f"Product {pid}",
                                depts[c],
                                cats[c],
                                price,
                                -1,
                            )
                        )
    return pd.DataFrame(rows, columns=REQUIRED)


# --------------------------------------------------------------------------- #
# Upload handling
# --------------------------------------------------------------------------- #
def read_upload(file: Any) -> pd.DataFrame:
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(
            file,
            dtype=str,
            keep_default_na=False,
            na_values=["", "NULL", "null"],
            encoding_errors="replace",
        )
    if name.endswith((".parquet", ".pq")):
        return pd.read_parquet(file)
    raise ValueError("Please upload a .csv, .parquet or .pq file.")


def auto_map(columns: list[str]) -> dict[str, str | None]:
    norm = {c.strip().lower().replace(" ", "_"): c for c in columns}
    aliases = {
        "customer_id": [
            "customer_id",
            "customerid",
            "customer",
            "cust_id",
            "client_id",
            "loyalty_id",
        ],
        "transaction_date": [
            "transaction_date",
            "date",
            "order_date",
            "purchase_date",
            "datetime",
            "timestamp",
        ],
        "transaction_id": [
            "transaction_id",
            "basket_id",
            "order_id",
            "receipt_id",
            "invoice_id",
            "invoiceno",
        ],
        "product_id": ["product_id", "sku", "item_id", "stockcode", "article_id"],
        "product_description": ["product_description", "description", "product_name", "item_name"],
        "department": ["department", "dept", "division"],
        "category": ["category", "product_category", "commodity", "sub_category"],
        "price": ["price", "unit_price", "unitprice"],
        "quantity": ["quantity", "qty", "units"],
    }
    return {req: next((norm[a] for a in al if a in norm), None) for req, al in aliases.items()}


# --------------------------------------------------------------------------- #
# Pipeline glue
# --------------------------------------------------------------------------- #
def parse_windows(text: str) -> str:
    vals = sorted({int(v) for v in text.replace(" ", "").split(",") if v})
    if not vals or min(vals) <= 0:
        raise ValueError("Rolling windows must be positive integers, e.g. 30,90,365")
    return ",".join(map(str, vals))


def build_args(p: dict[str, Any]) -> list[str]:
    a = [
        "--horizon-days",
        str(p["horizon"]),
        "--burn-in-days",
        str(p["burn_in"]),
        "--windows-days",
        parse_windows(p["windows"]),
        "--recent-window-days",
        str(p["recent"]),
        "--frequency-grain",
        p["grain"],
        "--transaction-key-mode",
        p["key_mode"],
        "--basket-level",
        p["basket_level"],
        "--min-clusters",
        str(p["k_min"]),
        "--max-clusters",
        str(p["k_max"]),
        "--random-seed",
        str(p["seed"]),
    ]
    if p["as_of"]:
        a += ["--as-of-date", p["as_of"]]
    for flag, key in (
        ("--drop-exact-duplicates", "drop_dupes"),
        ("--exclude-zero-quantity", "excl_zero"),
        ("--exclude-negative-prices", "excl_neg"),
    ):
        if p[key]:
            a.append(flag)
    return a


def run_pipeline(df: pd.DataFrame, params: dict[str, Any], progress=None) -> dict[str, Any]:
    """Write the input, run retail_customer_analysis, then insight_engine. Returns a result handle."""
    import insight_engine as ie
    import retail_customer_analysis as rca

    work = Path(tempfile.mkdtemp(prefix="insight_lab_"))
    inp, res = work / "input.csv", work / "results"
    df[REQUIRED].to_csv(inp, index=False)
    t0 = time.time()
    if progress:
        progress("Cleaning data, fitting models and validating on a time holdout...")
    code = rca.main(["--input", str(inp), "--output-dir", str(res)] + build_args(params))
    if code != 0:
        raise RuntimeError(
            "The analysis script failed. Check the data (dates, price, quantity) and the settings."
        )
    if progress:
        progress("Drawing charts...")
    out = res / "insights"
    ie_code = ie.main(
        ["--results-dir", str(res), "--output-dir", str(out), "--dpi", "110", "--top-n", "12"]
    )
    summary = (
        json.loads((out / "insight_summary.json").read_text(encoding="utf-8"))
        if (out / "insight_summary.json").is_file()
        else {"charts": [], "kpis": {}, "skipped": []}
    )
    meta = (
        json.loads((res / "run_metadata.json").read_text(encoding="utf-8"))
        if (res / "run_metadata.json").is_file()
        else {}
    )
    return {
        "work": str(work),
        "results": str(res),
        "insights": str(out),
        "summary": summary,
        "meta": meta,
        "seconds": round(time.time() - t0, 1),
        "rows": int(len(df)),
        "chart_exit": ie_code,
    }


@st.cache_data(show_spinner=False, ttl=3600)
def read_results_table(res_dir: str, stem: str, mtime: float = 0.0) -> pd.DataFrame:
    for ext in ("csv", "parquet"):
        p = Path(res_dir) / f"{stem}.{ext}"
        if p.is_file():
            try:
                return pd.read_csv(p, low_memory=False) if ext == "csv" else pd.read_parquet(p)
            except Exception:
                return pd.DataFrame()
    return pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=3600)
def load_chart_data(res_dir: str, chart_keys: list[str]) -> dict[str, pd.DataFrame]:
    """Load all tables needed for a set of charts at once."""
    # Map chart keys to required tables
    chart_to_tables = {
        "01_model_validation": ["holdout_metrics", "holdout_calibration"],
        "02_calibration": ["holdout_calibration", "holdout_metrics"],
        "03_customer_map": ["customer_features"],
        "04_segment_value": ["cluster_profiles"],
        "05_cluster_selection": ["cluster_diagnostics"],
        "06_rfm_segments": ["rfm_segment_profiles"],
        "07_repeat_survival": ["repeat_survival"],
        "08_cohort_retention": ["cohort_retention_new_customers"],
        "09_revenue_concentration": ["customer_features"],
        "10_revenue_trend": ["monthly_summary", "weekday_summary"],
        "11_categories": ["category_summary"],
        "12_basket_rules": ["basket_affinity"],
        "13_next_trip": ["next_trip_affinity"],
        "14_momentum": ["customer_features"],
        "15_promo_proxy": ["customer_features"],
        "16_cadence": ["customer_features", "interpurchase_intervals"],
        "17_data_quality": ["data_quality_issues"],
        "18_lorenz_segments": ["customer_features"],
        "19_action_quadrant": ["customer_features"],
        "20_affinity_network": ["basket_affinity"],
        "21_promo_return_lollipop": ["category_summary"],
        "22_cohort_ltv": ["cohort_retention_new_customers"],
    }

    needed = set()
    for k in chart_keys:
        needed.update(chart_to_tables.get(k, []))

    return {name: read_results_table(res_dir, name) for name in needed}


@st.cache_data(show_spinner=False, ttl=3600)
def compute_kpis_from_tables(tables: dict[str, pd.DataFrame]) -> list[tuple[str, str, str | None]]:
    """Compute KPI values from cached tables for display."""
    kpis = []
    c = tables.get("customer_features", pd.DataFrame())
    if not c.empty:
        kpis.append(("Customers scored", f"{len(c):,}", None))
        if "p_alive" in c:
            kpis.append(
                (
                    "Likely active (P(alive) ≥ 0.5)",
                    f"{(c['p_alive'] >= 0.5).mean() * 100:.0f}%",
                    None,
                )
            )
        if "expected_revenue_horizon" in c:
            e = c["expected_revenue_horizon"].clip(lower=0)
            kpis.append(("Expected revenue, next horizon", f"{e.sum():,.0f}", None))
            if len(e) >= 10 and e.sum() > 0:
                top10 = e.nlargest(max(1, int(np.ceil(len(e) * 0.1)))).sum() / e.sum()
                kpis.append(("Top 10% share", f"{top10 * 100:.0f}%", None))

    s = tables.get("repeat_survival", pd.DataFrame())
    if not s.empty:
        r = s[(s["group"] == "all_customers") & (s["day"] == 90)]
        if len(r):
            kpis.append(
                ("2nd trip within 90d", f"{r.iloc[0]['repeat_probability'] * 100:.0f}%", None)
            )

    m = tables.get("holdout_metrics", pd.DataFrame())
    if not m.empty:
        bg = m[m["model"] == "BG/NBD P(alive)"]["auc_any_purchase"]
        lg = m[m["model"] == "legacy_cadence_heuristic"]["auc_any_purchase"]
        if len(bg) and len(lg):
            kpis.append(
                (
                    "AUC: BG/NBD vs legacy",
                    f"{bg.iloc[0]:.2f} vs {lg.iloc[0]:.2f}",
                    f"{bg.iloc[0] - lg.iloc[0]:+.2f}",
                )
            )

    a = tables.get("basket_affinity", pd.DataFrame())
    if not a.empty and "passes_all_filters" in a:
        kpis.append(
            ("Supported basket rules", f"{int(a['passes_all_filters'].astype(bool).sum()):,}", None)
        )

    return kpis


def zip_dir(path: str) -> bytes:
    buf = io.BytesIO()
    root = Path(path)
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(root.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(root))
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Small UI helpers (tolerant of Streamlit version differences)
# --------------------------------------------------------------------------- #
def show_df(df: pd.DataFrame, **kw: Any) -> None:
    """Stretch to container width across Streamlit versions."""
    for extra in ({"width": "stretch"}, {"use_container_width": True}, {}):
        try:
            st.dataframe(df, hide_index=True, **extra, **kw)
            return
        except Exception:
            continue


def show_image(path: Path) -> None:
    for extra in (
        {"width": "stretch"},
        {"use_container_width": True},
        {"use_column_width": True},
        {},
    ):
        try:
            st.image(str(path), **extra)
            return
        except Exception:
            continue


def table(R: dict[str, Any], stem: str) -> pd.DataFrame:
    p = next(
        (
            Path(R["results"]) / f"{stem}.{e}"
            for e in ("csv", "parquet")
            if (Path(R["results"]) / f"{stem}.{e}").is_file()
        ),
        None,
    )
    return read_results_table(R["results"], stem, p.stat().st_mtime if p else 0.0)


def show_charts(R: dict[str, Any], keys: list[str]) -> None:
    by_key = {c["key"]: c for c in R["summary"].get("charts", [])}
    shown = 0
    for k in keys:
        c = by_key.get(k)
        if not c:
            continue
        shown += 1
        st.subheader(c["title"])
        st.markdown(c["takeaway"])
        show_image(Path(R["insights"]) / c["file"])
    skipped = [s for s in R["summary"].get("skipped", []) if any(s["chart"] in k for k in keys)]
    if skipped:
        with st.expander(f"{len(skipped)} chart(s) not produced"):
            for s in skipped:
                st.write(f"- **{s['chart']}**: {s['reason']}")
    if not shown:
        st.info("No charts available for this section with the current data.")


def pretty(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=lambda c: c.replace("_", " "))


# --------------------------------------------------------------------------- #
# Sidebar and data step
# --------------------------------------------------------------------------- #
def sidebar(raw_df: pd.DataFrame | None = None) -> dict[str, Any]:
    sb = st.sidebar
    sb.title("Customer Insight Lab")
    sb.caption("Understand customers from line-item purchase data.")
    source = sb.radio("1. Data source", ["Synthetic demo data", "Upload my data"], index=0)
    p: dict[str, Any] = {"source": source}
    if source == "Synthetic demo data":
        with sb.expander("Synthetic data settings", expanded=True):
            p["n_customers"] = int(st.number_input("Customers", 500, 20000, 3000, step=500))
            p["days"] = int(st.slider("History (days)", 365, 1095, 730, step=5))
            p["syn_seed"] = int(st.number_input("Random seed", 0, 10_000, 7))
            p["missing_id"] = float(
                st.slider("Rows without customer id", 0.0, 0.20, 0.03, step=0.01)
            )
            p["return_rate"] = float(st.slider("Return rate (lines)", 0.0, 0.15, 0.03, step=0.01))
            p["promo_share"] = float(st.slider("Promo-priced lines", 0.0, 0.50, 0.15, step=0.05))
    else:
        p["file"] = sb.file_uploader("Line-item CSV or Parquet", type=["csv", "parquet", "pq"])
        sb.caption(
            "Needs one row per line item with: "
            + ", ".join(REQUIRED)
            + ". Negative quantity = return."
        )
    with sb.expander("2. Analysis settings"):
        p["horizon"] = int(
            st.number_input("Forecast and holdout horizon (days)", 14, 365, 90, step=15)
        )
        p["burn_in"] = int(
            st.number_input("Burn-in for 'new customer' (days)", 14, 365, 90, step=15)
        )
        p["grain"] = st.selectbox(
            "Purchase event",
            ["trip", "transaction"],
            index=0,
            help="trip = one customer-day (merges split receipts)",
        )
        p["basket_level"] = st.selectbox("Basket analysis level", ["category", "product"], index=0)
        p["k_min"], p["k_max"] = (
            int(v) for v in st.slider("Segment count range (k)", 2, 10, (3, 7))
        )
        p["windows"] = st.text_input("Rolling windows (days)", "30,90,365")
        p["recent"] = int(st.number_input("Recent vs prior window (days)", 14, 365, 90, step=15))
        use_as_of = st.checkbox("Set analysis cut-off date manually", value=False)
        p["as_of"] = st.date_input("Cut-off date").isoformat() if use_as_of else None
        p["key_mode"] = st.selectbox(
            "Transaction id scope", ["customer-transaction", "transaction-only"], index=0
        )
        p["drop_dupes"] = st.checkbox("Drop exact duplicate rows", value=False)
        p["excl_zero"] = st.checkbox("Exclude zero-quantity rows", value=False)
        p["excl_neg"] = st.checkbox("Exclude negative-price rows", value=False)
        p["seed"] = int(st.number_input("Analysis random seed", 0, 10_000, 42))

    # Period selection (only shown after analysis completes)
    if raw_df is not None and not raw_df.empty:
        with sb.expander("📅 Period Selection", expanded=True):
            min_date = raw_df["transaction_day"].min()
            max_date = raw_df["transaction_day"].max()

            preset = st.selectbox(
                "Current period",
                [
                    "Last 7 days",
                    "Last 28 days",
                    "Last 90 days",
                    "Month to date",
                    "Quarter to date",
                    "Custom range",
                ],
                index=1,
                key="period_preset",
            )

            if preset == "Last 7 days":
                current_end = max_date
                current_start = current_end - pd.Timedelta(days=6)
            elif preset == "Last 28 days":
                current_end = max_date
                current_start = current_end - pd.Timedelta(days=27)
            elif preset == "Last 90 days":
                current_end = max_date
                current_start = current_end - pd.Timedelta(days=89)
            elif preset == "Month to date":
                current_end = max_date
                current_start = current_end.replace(day=1)
            elif preset == "Quarter to date":
                current_end = max_date
                quarter_start_month = ((current_end.month - 1) // 3) * 3 + 1
                current_start = current_end.replace(month=quarter_start_month, day=1)
            else:
                date_range = st.date_input(
                    "Custom date range",
                    value=(max_date - pd.Timedelta(days=27), max_date),
                    min_value=min_date,
                    max_value=max_date,
                    key="custom_date_range",
                )
                if len(date_range) == 2:
                    current_start = pd.Timestamp(date_range[0])
                    current_end = pd.Timestamp(date_range[1])
                else:
                    current_start = max_date - pd.Timedelta(days=27)
                    current_end = max_date

            st.caption(
                f"Current: {current_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')}"
            )

            compare_mode = st.selectbox(
                "Compare to",
                ["Prior period (equal length)", "Same period last year"],
                index=0,
                key="compare_mode",
            )
            compare_mode = "prior" if compare_mode == "Prior period (equal length)" else "yoy"

            p["current_start"] = current_start
            p["current_end"] = current_end
            p["compare_mode"] = compare_mode

    return p


@st.cache_data(show_spinner=False)
def cached_synthetic(
    n: int, days: int, seed: int, missing: float, ret: float, promo: float
) -> pd.DataFrame:
    return make_synthetic(n, days, seed, missing, ret, promo)


def data_step(p: dict[str, Any]) -> tuple[pd.DataFrame | None, str | None]:
    """Return (dataframe ready to analyse or None, message). Handles preview and column mapping."""
    if p["source"] == "Synthetic demo data":
        st.info(
            f"Synthetic data will be generated when you press **Run analysis**: {p['n_customers']:,} customers over {p['days']} days, "
            "with category preferences, promotions, returns, split receipts and some missing customer ids."
        )
        return None, "synthetic"
    if p.get("file") is None:
        return None, "Upload a file in the sidebar to begin."
    try:
        raw = read_upload(p["file"])
    except Exception as exc:
        return None, f"Could not read the file: {exc}"
    st.subheader("Your data")
    st.caption(f"{len(raw):,} rows and {len(raw.columns)} columns. First rows:")
    show_df(raw.head(10))
    guess = auto_map(list(raw.columns))
    missing = [r for r in REQUIRED if guess[r] is None]
    mapping = dict(guess)
    if missing:
        st.warning("Some required columns were not found. Map them below: " + ", ".join(missing))
    with st.expander("Column mapping", expanded=bool(missing)):
        opts = ["(none)"] + list(raw.columns)
        for r in REQUIRED:
            idx = opts.index(guess[r]) if guess[r] in opts else 0
            choice = st.selectbox(r, opts, index=idx, key=f"map_{r}")
            mapping[r] = None if choice == "(none)" else choice
    if any(mapping[r] is None for r in REQUIRED):
        return (
            None,
            "Map all required columns to continue (a column can be mapped to 'none' only if you add it to the file).",
        )
    used = [mapping[r] for r in REQUIRED]
    if len(set(used)) < len(used):
        return None, "Each required column must map to a different source column."
    df = raw[used].copy()
    df.columns = REQUIRED

    # Convert types to match expected schema
    df["customer_id"] = df["customer_id"].astype("string")
    df["transaction_id"] = df["transaction_id"].astype("string")
    df["product_id"] = df["product_id"].astype("string")
    df["product_description"] = df["product_description"].astype("string")
    df["department"] = df["department"].astype("string")
    df["category"] = df["category"].astype("string")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")

    return df, None


# --------------------------------------------------------------------------- #
# Result tabs
# --------------------------------------------------------------------------- #
def tab_overview(
    R: dict[str, Any], raw_df: pd.DataFrame | None = None, period_params: dict | None = None
) -> None:
    st.header("1 Overview")
    st.caption("Question answered here: what is the headline picture of this customer base?")

    # Load and cache chart data
    chart_data = load_chart_data(R["results"], HEADLINE_CHARTS)
    kpis = compute_kpis_from_tables(chart_data)

    # Executive KPI Tiles with Period Comparison
    if raw_df is not None and period_params:
        ctx = get_period_context(period_params)
        current_metrics = compute_period_metrics(raw_df, ctx["current_start"], ctx["current_end"])
        prior_metrics = compute_period_metrics(raw_df, ctx["prior_start"], ctx["prior_end"])

        # Build KPI tiles with comparison
        kpi_metrics = [
            {
                "label": "Revenue",
                "current": fmt_currency(current_metrics["revenue"]),
                "prior": fmt_currency(prior_metrics["revenue"]),
                "change": fmt_currency(current_metrics["revenue"] - prior_metrics["revenue"]),
                "pct_change": fmt_pct(
                    (current_metrics["revenue"] / prior_metrics["revenue"] - 1) * 100
                )
                if prior_metrics["revenue"] > 0
                else "N/A",
                "trend": "up" if current_metrics["revenue"] >= prior_metrics["revenue"] else "down",
                "help_text": "Total revenue = quantity × unit_price",
            },
            {
                "label": "Orders",
                "current": fmt_number(current_metrics["orders"]),
                "prior": fmt_number(prior_metrics["orders"]),
                "change": fmt_number(current_metrics["orders"] - prior_metrics["orders"]),
                "pct_change": fmt_pct(
                    (current_metrics["orders"] / prior_metrics["orders"] - 1) * 100
                )
                if prior_metrics["orders"] > 0
                else "N/A",
                "trend": "up" if current_metrics["orders"] >= prior_metrics["orders"] else "down",
                "help_text": "Unique transaction_ids",
            },
            {
                "label": "Active Customers",
                "current": fmt_number(current_metrics["customers"]),
                "prior": fmt_number(prior_metrics["customers"]),
                "change": fmt_number(current_metrics["customers"] - prior_metrics["customers"]),
                "pct_change": fmt_pct(
                    (current_metrics["customers"] / prior_metrics["customers"] - 1) * 100
                )
                if prior_metrics["customers"] > 0
                else "N/A",
                "trend": "up"
                if current_metrics["customers"] >= prior_metrics["customers"]
                else "down",
                "help_text": "Unique customer_ids with purchases",
            },
            {
                "label": "AOV",
                "current": fmt_currency(current_metrics["aov"]),
                "prior": fmt_currency(prior_metrics["aov"]),
                "change": fmt_currency(current_metrics["aov"] - prior_metrics["aov"]),
                "pct_change": fmt_pct((current_metrics["aov"] / prior_metrics["aov"] - 1) * 100)
                if prior_metrics["aov"] > 0
                else "N/A",
                "trend": "up" if current_metrics["aov"] >= prior_metrics["aov"] else "down",
                "help_text": "Average Order Value = Revenue / Orders",
            },
            {
                "label": "Units/Order",
                "current": f"{current_metrics['units_per_order']:.1f}",
                "prior": f"{prior_metrics['units_per_order']:.1f}",
                "change": f"{current_metrics['units_per_order'] - prior_metrics['units_per_order']:+.1f}",
                "pct_change": fmt_pct(
                    (current_metrics["units_per_order"] / prior_metrics["units_per_order"] - 1)
                    * 100
                )
                if prior_metrics["units_per_order"] > 0
                else "N/A",
                "trend": "up"
                if current_metrics["units_per_order"] >= prior_metrics["units_per_order"]
                else "down",
                "help_text": "Average basket size",
            },
            {
                "label": "Revenue/Customer",
                "current": fmt_currency(current_metrics["revenue_per_customer"]),
                "prior": fmt_currency(prior_metrics["revenue_per_customer"]),
                "change": fmt_currency(
                    current_metrics["revenue_per_customer"] - prior_metrics["revenue_per_customer"]
                ),
                "pct_change": fmt_pct(
                    (
                        current_metrics["revenue_per_customer"]
                        / prior_metrics["revenue_per_customer"]
                        - 1
                    )
                    * 100
                )
                if prior_metrics["revenue_per_customer"] > 0
                else "N/A",
                "trend": "up"
                if current_metrics["revenue_per_customer"] >= prior_metrics["revenue_per_customer"]
                else "down",
                "help_text": "Average revenue per active customer",
            },
        ]
        kpi_tile_row(kpi_metrics)

        # KPI Sparkline Trends - Small multiples across recent periods
        st.markdown("---")
        section(
            "KPI Trends",
            "Recent trajectory across key metrics. Each sparkline shows the last 6 periods of the same length as your current selection.",
        )
        sparkline_result = _plot_kpi_sparklines(raw_df, period_params)
        if sparkline_result:
            chart_card(
                sparkline_result[0],
                "Track whether improvements are sustained or one-off. Consistent upward trends across Revenue, Orders, and Customers signal healthy growth.",
                scope="Selected period length, last 6 periods · Positive purchases",
            )

        # Revenue Waterfall Decomposition
        st.markdown("---")
        section(
            "Revenue Change Decomposition",
            "What drove the revenue change? Customer count, purchase frequency, units per order, or average unit price?",
        )
        waterfall_data = revenue_waterfall_data(current_metrics, prior_metrics)
        fig, _ = plot_revenue_waterfall(waterfall_data)
        pct_change = (
            ((current_metrics["revenue"] / prior_metrics["revenue"] - 1) * 100)
            if prior_metrics["revenue"] > 0
            else 0
        )
        chart_card(
            fig,
            f"Revenue changed by {fmt_currency(current_metrics['revenue'] - prior_metrics['revenue'])} ({fmt_pct(pct_change)}) vs comparison period.",
            scope=f"Current: {ctx['current_start'].strftime('%Y-%m-%d')} to {ctx['current_end'].strftime('%Y-%m-%d')} · Prior: {ctx['prior_start'].strftime('%Y-%m-%d')} to {ctx['prior_end'].strftime('%Y-%m-%d')} · Positive purchases",
        )

        # Category/Department Contribution to Revenue Change
        st.markdown("---")
        section(
            "Category Contribution to Revenue Change",
            "Which categories drove growth or decline? Contribution = (category revenue change) / (total revenue change)",
        )
        contrib_result = _plot_category_contribution_waterfall(raw_df, period_params)
        if contrib_result:
            chart_card(
                contrib_result[0],
                "Categories with the largest bars (positive or negative) are where to focus assortment, pricing, or promotion reviews.",
                scope="Current vs prior period · Category-level · Positive purchases",
            )

        # Year-over-Year Trend Comparison
        st.markdown("---")
        section(
            "Year-over-Year Comparison",
            "Current period vs same period last year, aligned by calendar. Removes seasonality to show true performance.",
        )
        yoy_result = _plot_yoy_comparison(raw_df, period_params)
        if yoy_result:
            chart_card(
                yoy_result[0],
                "Divergence between current and prior year lines indicates structural change, not just seasonality.",
            )

        # Executive Narrative
        st.markdown("---")
        section("Executive Summary", "Automated narrative synthesizing the key drivers.")
        _render_executive_narrative(current_metrics, prior_metrics, period_params)

        # Anomaly Detection
        st.markdown("---")
        section(
            "Anomaly Detection",
            "Statistical exceptions in daily KPIs vs. lagged rolling baseline (±2σ).",
        )
        anomaly_data = _compute_anomalies(raw_df, period_params)
        if anomaly_data and anomaly_data.get("anomalies"):
            summary_fig = _plot_anomaly_summary(anomaly_data)
            if summary_fig:
                chart_card(
                    summary_fig[0],
                    "Anomaly counts and max deviations by metric in the current period.",
                    scope=f"Current period: {anomaly_data.get('current_start', 'N/A')} to {anomaly_data.get('current_end', 'N/A')} · Rolling baseline: 28-day lagged ±2σ",
                )

            # Show time series for metrics with anomalies
            for metric in anomaly_data["anomalies"].keys():
                ts_fig = _plot_anomaly_timeseries(anomaly_data, metric)
                if ts_fig:
                    chart_card(
                        ts_fig[0],
                        f"{metric.replace('_', ' ').title()} with expected range and anomalies marked.",
                        scope=f"Current period vs lagged rolling baseline (28-day, ±2σ) · Metric: {metric.replace('_', ' ').title()}",
                    )
        else:
            st.info("No anomalies detected in the current period.")

    else:
        # Fallback to simple KPIs
        kpi_row(kpis)

    # Key findings from insight engine
    st.subheader("Key Findings from Models")
    by_key = {c["key"]: c for c in R["summary"].get("charts", [])}
    found = [by_key[k] for k in HEADLINE_CHARTS if k in by_key]
    if found:
        for c in found:
            st.markdown(f"- **{c['title']}** {c['takeaway']}")
    else:
        st.info("No headline findings available.")

    # Revenue trend + segment mix
    col1, col2 = st.columns([2, 1])
    with col1:
        show_charts(R, ["10_revenue_trend"])
    with col2:
        st.markdown("**Segment Revenue Share**")
        show_charts(R, ["04_segment_value"])

    meta = R["meta"]
    warnings = meta.get("warnings") or []
    if warnings:
        st.subheader("Warnings from the pipeline")
        for w in warnings:
            st.warning(w)

    with st.expander("Run details"):
        st.write(f"{R['rows']:,} input rows, analysis and charts took {R['seconds']} seconds.")
        cfg = meta.get("configuration", {})
        keep = [
            "horizon_days",
            "burn_in_days",
            "frequency_grain",
            "basket_level",
            "min_clusters",
            "max_clusters",
            "windows_days",
            "as_of_date",
        ]
        st.json({k: cfg.get(k) for k in keep if k in cfg})

    with st.expander("How to read these results"):
        st.markdown(
            "- **P(alive)** is the model's probability that a customer is still active; it is not a promise to buy.\n"
            "- **Expected revenue** is trips × value per trip over the horizon. It is revenue, not profit, and is not discounted.\n"
            "- **Segments** describe behaviour. They are useful only if they are stable and differ on future behaviour.\n"
            "- **Basket rules** and **transitions** are associations, not causal effects.\n"
            "- **Promo share** is a price-based proxy; no promotion field exists in the data."
        )


def _plot_kpi_sparklines(raw_df: pd.DataFrame, period_params: dict) -> tuple | None:
    """Generate small multiples sparklines for key KPIs across recent periods."""
    try:
        ctx = get_period_context(period_params)
        ctx["current_start"]
        current_end = ctx["current_end"]
        period_len = ctx["period_len_days"]

        # Generate 6 periods going back
        n_periods = 6
        periods = []
        period_labels = []

        for i in range(n_periods):
            p_end = current_end - pd.Timedelta(days=i * period_len)
            p_start = p_end - pd.Timedelta(days=period_len - 1)
            periods.append((p_start, p_end))
            period_labels.append(p_start.strftime("%m/%d"))

        periods = list(reversed(periods))
        period_labels = list(reversed(period_labels))

        # Compute metrics for each period
        metric_names = ["Revenue", "Orders", "Customers", "AOV", "Units/Order", "Revenue/Cust"]
        metric_values = {name: [] for name in metric_names}

        for p_start, p_end in periods:
            m = compute_period_metrics(raw_df, p_start, p_end)
            metric_values["Revenue"].append(m["revenue"])
            metric_values["Orders"].append(m["orders"])
            metric_values["Customers"].append(m["customers"])
            metric_values["AOV"].append(m["aov"])
            metric_values["Units/Order"].append(m["units_per_order"])
            metric_values["Revenue/Cust"].append(m["revenue_per_customer"])

        # Plot
        fig, axes = plot_kpi_sparklines(
            period_labels, [metric_values[name] for name in metric_names], metric_names
        )
        return fig, axes
    except Exception:
        return None


def _plot_category_contribution_waterfall(
    raw_df: pd.DataFrame, period_params: dict
) -> tuple | None:
    """Waterfall chart showing category contribution to total revenue change."""
    try:
        ctx = get_period_context(period_params)
        ctx["current_start"]
        ctx["current_end"]
        ctx["prior_start"]
        ctx["prior_end"]

        # Filter data using consolidated helper
        curr_df = get_period_transactions(raw_df, ctx, "current")
        prior_df = get_period_transactions(raw_df, ctx, "prior")

        if not ensure_not_empty(curr_df, "current period categories") or not ensure_not_empty(
            prior_df, "prior period categories"
        ):
            return None

        # Category revenue (revenue already calculated in prepare_transaction_frame)
        curr_cat = curr_df.groupby("category")["revenue"].sum().reset_index()
        prior_cat = prior_df.groupby("category")["revenue"].sum().reset_index()

        # Merge
        merged = curr_cat.merge(
            prior_cat, on="category", how="outer", suffixes=("_curr", "_prior")
        ).fillna(0)
        merged["change"] = merged["revenue_curr"] - merged["revenue_prior"]
        merged["pct_change"] = (
            merged["change"] / merged["revenue_prior"].replace(0, np.nan) * 100
        ).round(1)

        # Sort by absolute contribution
        total_change = merged["change"].sum()
        if total_change == 0:
            return None

        merged["contribution_pct"] = (merged["change"] / total_change * 100).round(1)
        merged = merged.sort_values("change", ascending=True)

        # Take top 10 by absolute change
        merged = (
            merged.nlargest(10, "change", keep="all")
            if total_change > 0
            else merged.nsmallest(10, "change", keep="all")
        )

        # Plot
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 5))

        labels = [
            f"{row['category'][:25]} ({row['contribution_pct']:+.1f}%)"
            for _, row in merged.iterrows()
        ]
        values = merged["change"].values

        colors = ["#2f855a" if v >= 0 else "#e53e3e" for v in values]
        bars = ax.barh(range(len(labels)), values, color=colors, edgecolor="white", height=0.6)

        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Revenue Change", fontsize=10)
        ax.set_title(
            "Category Contribution to Revenue Change", fontsize=13, fontweight="bold", pad=15
        )
        ax.axvline(x=0, color="gray", linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", alpha=0.3)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: fmt_currency(x)))

        # Add value labels
        for bar, val in zip(bars, values, strict=False):
            ax.text(
                val + (max(values) - min(values)) * 0.01
                if val >= 0
                else val - (max(values) - min(values)) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                fmt_currency(val),
                va="center",
                ha="left" if val >= 0 else "right",
                fontsize=9,
                fontweight="bold",
            )

        plt.tight_layout()
        return fig, ax
    except Exception:
        return None


def _plot_yoy_comparison(raw_df: pd.DataFrame, period_params: dict) -> tuple | None:
    """Year-over-year trend comparison aligned by week/month."""
    try:
        ctx = get_period_context(period_params)
        current_start = ctx["current_start"]
        current_end = ctx["current_end"]

        # For YoY, we always compare to same period last year regardless of compare_mode
        prior_start = current_start - pd.DateOffset(years=1)
        prior_end = current_end - pd.DateOffset(years=1)

        # Determine granularity based on period length
        period_days = (current_end - current_start).days + 1
        if period_days <= 14:
            freq = "D"
            date_fmt = "%m/%d"
        elif period_days <= 60:
            freq = "W-MON"
            date_fmt = "%m/%d"
        else:
            freq = "M"
            date_fmt = "%b %Y"

        curr_df = get_period_transactions(raw_df, ctx, "current")
        prior_df = raw_df[
            (raw_df["transaction_day"] >= prior_start) & (raw_df["transaction_day"] <= prior_end)
        ].copy()

        if not ensure_not_empty(curr_df, "current period YoY") or not ensure_not_empty(
            prior_df, "prior year YoY"
        ):
            return None

        # Resample (revenue already calculated in prepare_transaction_frame)
        curr_ts = curr_df.set_index("transaction_day")["revenue"].resample(freq).sum()
        prior_ts = prior_df.set_index("transaction_day")["revenue"].resample(freq).sum()

        # Align indices for plotting
        curr_labels = [d.strftime(date_fmt) for d in curr_ts.index]
        [d.strftime(date_fmt) for d in prior_ts.index]

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 5))

        ax.plot(
            range(len(curr_ts)),
            curr_ts.values,
            "o-",
            color="#2b6cb0",
            linewidth=2,
            label=f"Current ({current_start.strftime('%Y')})",
            markersize=5,
        )
        ax.plot(
            range(len(prior_ts)),
            prior_ts.values,
            "s--",
            color="#a0aec0",
            linewidth=1.5,
            label=f"Prior Year ({prior_start.strftime('%Y')})",
            markersize=4,
        )

        ax.fill_between(range(len(curr_ts)), curr_ts.values, alpha=0.1, color="#2b6cb0")
        ax.fill_between(range(len(prior_ts)), prior_ts.values, alpha=0.1, color="#a0aec0")

        ax.set_xticks(range(len(curr_labels)))
        ax.set_xticklabels(curr_labels, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Revenue", fontsize=10)
        ax.set_title(
            f"Year-over-Year Revenue Comparison ({freq} granularity)",
            fontsize=13,
            fontweight="bold",
            pad=15,
        )
        ax.legend(fontsize=10, loc="upper left")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: fmt_currency(x)))

        # Add growth rate annotations at the end
        if len(curr_ts) > 0 and len(prior_ts) > 0:
            curr_total = curr_ts.sum()
            prior_total = prior_ts.sum()
            if prior_total > 0:
                growth = (curr_total / prior_total - 1) * 100
                ax.annotate(
                    f"Total: {fmt_pct(growth)} YoY",
                    xy=(0.98, 0.95),
                    xycoords="axes fraction",
                    ha="right",
                    va="top",
                    fontsize=11,
                    fontweight="bold",
                    color="#2f855a" if growth >= 0 else "#e53e3e",
                    bbox={
                        "boxstyle": "round,pad=0.3",
                        "facecolor": "white",
                        "edgecolor": "gray",
                        "alpha": 0.9,
                    },
                )

        plt.tight_layout()
        return fig, ax
    except Exception:
        return None


def _render_executive_narrative(
    current_metrics: dict, prior_metrics: dict, period_params: dict
) -> None:
    """Generate and render an executive narrative summarizing key drivers."""
    rev_change = current_metrics["revenue"] - prior_metrics["revenue"]
    rev_pct = (
        (current_metrics["revenue"] / prior_metrics["revenue"] - 1) * 100
        if prior_metrics["revenue"] > 0
        else 0
    )

    # Decompose drivers
    c_cust = current_metrics["customers"]
    c_freq = current_metrics["orders_per_customer"]
    c_aov = current_metrics["aov"]
    c_upo = current_metrics["units_per_order"]
    c_aup = current_metrics["avg_unit_price"]

    p_cust = prior_metrics["customers"]
    p_freq = prior_metrics["orders_per_customer"]
    p_aov = prior_metrics["aov"]
    p_upo = prior_metrics["units_per_order"]
    p_aup = prior_metrics["avg_unit_price"]

    # Contributions
    cust_effect = (c_cust - p_cust) * p_freq * p_aov
    freq_effect = c_cust * (c_freq - p_freq) * p_aov
    upo_effect = c_cust * c_freq * (c_upo - p_upo) * p_aup
    aup_effect = c_cust * c_freq * c_upo * (c_aup - p_aup)

    effects = {
        "Customer Count": cust_effect,
        "Purchase Frequency": freq_effect,
        "Units per Order": upo_effect,
        "Average Unit Price": aup_effect,
    }

    # Find primary driver
    primary_driver = max(effects.items(), key=lambda x: abs(x[1]))

    direction = "increased" if rev_change >= 0 else "decreased"
    primary_name, primary_val = primary_driver
    primary_pct = (abs(primary_val) / abs(rev_change) * 100) if abs(rev_change) > 1e-10 else 0

    # Build narrative
    narrative = f"""
    **Revenue {direction} {fmt_pct(rev_pct)}** ({fmt_currency(abs(rev_change))}) versus the comparison period.

    The primary driver was **{primary_name}**, explaining **{primary_pct:.0f}%** of the total change ({fmt_currency(primary_val)}).
    """

    # Add secondary drivers
    sorted_effects = sorted(effects.items(), key=lambda x: abs(x[1]), reverse=True)
    for name, val in sorted_effects[1:3]:
        if abs(rev_change) > 1e-10 and abs(val) > 0.01 * abs(rev_change):
            pct = (abs(val) / abs(rev_change) * 100) if abs(rev_change) > 1e-10 else 0
            narrative += f"\n\n- **{name}** contributed {fmt_currency(val)} ({pct:.0f}% of change)."

    # Add context on metric health
    narrative += "\n\n**Metric Health:**"
    # Customer base
    if p_cust > 0:
        cust_pct = (c_cust / p_cust - 1) * 100
        if c_cust >= p_cust:
            narrative += f" ✅ Customer base **grew** ({fmt_pct(cust_pct)})"
        else:
            narrative += f" ⚠️ Customer base **shrank** ({fmt_pct(cust_pct)})"
    else:
        narrative += " ⚠️ Customer base **N/A** (no prior customers)"

    # AOV
    if p_aov > 0:
        aov_pct = (c_aov / p_aov - 1) * 100
        if c_aov >= p_aov:
            narrative += f" | ✅ AOV **improved** ({fmt_pct(aov_pct)})"
        else:
            narrative += f" | ⚠️ AOV **declined** ({fmt_pct(aov_pct)})"
    else:
        narrative += " | ⚠️ AOV **N/A**"

    # Basket size
    if p_upo > 0:
        upo_pct = (c_upo / p_upo - 1) * 100
        if c_upo >= p_upo:
            narrative += f" | ✅ Basket size **grew** ({fmt_pct(upo_pct)})"
        else:
            narrative += f" | ⚠️ Basket size **shrank** ({fmt_pct(upo_pct)})"
    else:
        narrative += " | ⚠️ Basket size **N/A**"

    st.markdown(narrative)


def tab_trust(
    R: dict[str, Any], raw_df: pd.DataFrame | None = None, period_params: dict | None = None
) -> None:
    st.header("Can we trust it?")
    st.caption(
        "The models were fitted on older data and scored on the following period, against simple baselines."
    )

    chart_data = load_chart_data(R["results"], CHARTS_BY_TAB["trust"])
    kpis = compute_kpis_from_tables(chart_data)
    kpi_row(kpis)

    st.markdown("---")

    # Model Validation
    section(
        "Model Validation: Does the model beat simple rules on unseen data?",
        "AUC for who buys again, MAE for trip/revenue forecasts. Higher AUC / Lower MAE = better",
    )
    show_charts(R, CHARTS_BY_TAB["trust"])

    # Calibration
    st.markdown("---")
    section(
        "Forecast Calibration: Are predicted probabilities reliable?",
        "Decile calibration plot - points on the diagonal = well calibrated",
    )
    show_charts(R, ["02_calibration"])

    # Cluster Stability
    st.markdown("---")
    section(
        "Segment Stability: Are segments real or an artefact?",
        "Silhouette (separation) and Bootstrap ARI (stability) across k values",
    )
    show_charts(R, ["05_cluster_selection"])

    # Reconciliation Table
    st.markdown("---")
    section(
        "Accounting Reconciliation: Do the numbers add up?",
        "Gross - Returns = Net at every aggregation level",
    )
    m = table(R, "model_parameters")
    if len(m):
        st.subheader("Model Parameters")
        show_df(pretty(m))

    # Synthetic parameter recovery
    if R.get("source_label") == "Synthetic demo data":
        st.markdown("---")
        section(
            "Synthetic Recovery: Did we recover the true parameters?",
            "Only available for synthetic data with known ground truth",
        )
        st.info(
            "Synthetic parameter recovery chart available when running on generated data with ground truth"
        )

    for stem, title in (
        ("holdout_metrics", "Holdout metrics"),
        ("holdout_calibration", "Calibration by decile"),
        ("cluster_diagnostics", "Cluster selection diagnostics"),
        ("model_parameters", "Fitted model parameters"),
    ):
        t = table(R, stem)
        if len(t):
            with st.expander(title):
                show_df(pretty(t))


def segment_profile(cf: pd.DataFrame) -> pd.DataFrame:
    if cf.empty or "cluster_label" not in cf.columns:
        return pd.DataFrame()
    g = cf.groupby(cf["cluster_label"].fillna("Unclustered"))
    out = g.agg(
        customers=("p_alive", "size"),
        mean_p_alive=("p_alive", "mean"),
        expected_trips=("expected_trips_horizon", "mean"),
        expected_value_per_trip=("expected_trip_value", "mean"),
        total_expected_revenue=("expected_revenue_horizon", "sum"),
    ).reset_index()
    if "top_category_by_spend" in cf.columns:
        mode = (
            g["top_category_by_spend"]
            .agg(lambda s: s.mode().iloc[0] if s.notna().any() else None)
            .rename("most_common_top_category")
        )
        out = out.merge(mode.reset_index(), on="cluster_label", how="left")
    for col in ("promo_spend_share", "return_value_ratio", "recency_days"):
        if col in cf.columns:
            out = out.merge(
                g[col].mean().rename(f"mean_{col}").reset_index(), on="cluster_label", how="left"
            )
    out["share_of_customers"] = out["customers"] / out["customers"].sum()
    tot = out["total_expected_revenue"].sum()
    out["share_of_expected_revenue"] = out["total_expected_revenue"] / tot if tot > 0 else np.nan
    out["hypothesis_to_test"] = [
        next(
            (v for k, v in SEGMENT_HINTS.items() if str(lab).startswith(k)),
            "Describe first; test actions against a holdout group.",
        )
        for lab in out["cluster_label"]
    ]
    return out.sort_values("total_expected_revenue", ascending=False)


def tab_customers(
    R: dict[str, Any], raw_df: pd.DataFrame | None = None, period_params: dict | None = None
) -> None:
    st.header("Customers")
    st.caption(
        "Question answered here: who are the customers, how active are they, and where does future revenue sit?"
    )

    chart_data = load_chart_data(R["results"], CHARTS_BY_TAB["customers"])
    kpis = compute_kpis_from_tables(chart_data)
    kpi_row(kpis)

    st.markdown("---")

    # Segment z-score heatmap
    section(
        "Segment Identity: What defines each segment?",
        "Z-scored metrics across segments - blue = below average, red = above average",
    )

    cf = table(R, "customer_features")
    prof = segment_profile(cf)
    if len(prof):
        # Z-score heatmap
        metrics_for_heatmap = [
            c
            for c in prof.columns
            if c.startswith(("mean_", "share_", "expected_")) and c not in ("mean_recency_days",)
        ]
        if metrics_for_heatmap:
            heat_df = prof.set_index("cluster_label")[metrics_for_heatmap]
            # Z-score
            z_df = (heat_df - heat_df.mean()) / heat_df.std().replace(0, 1)

            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(11, max(3, 0.5 * len(prof))))
            im = ax.imshow(z_df.T, cmap="RdBu_r", aspect="auto", vmin=-2, vmax=2)
            ax.set_xticks(range(len(z_df)))
            ax.set_xticklabels([segment_color(s) for s in z_df.index], rotation=20, ha="right")
            # Color the tick labels
            for _i, (tick, label) in enumerate(zip(ax.get_xticklabels(), z_df.index, strict=False)):
                tick.set_color(segment_color(label))
            ax.set_yticks(range(len(z_df.columns)))
            ax.set_yticklabels(
                [
                    c.replace("mean_", "").replace("share_of_", "").replace("_", " ").title()
                    for c in z_df.columns
                ],
                fontsize=8,
            )
            ax.set_title("Segment Profile Heatmap (Z-scores)")
            fig.colorbar(im, ax=ax, label="Z-score")
            chart_card(
                fig,
                f"Segments are most differentiated on {z_df.std(axis=1).idxmax() if len(z_df) else 'N/A'}.",
            )

        st.subheader("Segment profiles")
        st.caption("Hypotheses are starting points to test with a holdout group, not conclusions.")
        show_df(pretty(prof))

    # P(alive) vs Expected Revenue Quadrant
    st.markdown("---")
    section(
        "Action Quadrant: Who to protect, grow, win back, or ignore?",
        "X = P(alive), Y = Expected revenue. Top-right = protect, top-left = grow, bottom-right = win back, bottom-left = ignore",
    )

    show_charts(
        R, ["03_customer_map", "04_segment_value", "06_rfm_segments", "09_revenue_concentration"]
    )

    for stem, title in (
        ("rfm_segment_profiles", "RFM segment table"),
        ("customer_revenue_concentration", "Revenue concentration figures"),
    ):
        t = table(R, stem)
        if len(t):
            with st.expander(title):
                show_df(pretty(t))


def tab_behaviour(
    R: dict[str, Any], raw_df: pd.DataFrame | None = None, period_params: dict | None = None
) -> None:
    st.header("Behaviour")
    st.caption(
        "Question answered here: how do customers come back, how regular are they, and who is changing?"
    )

    chart_data = load_chart_data(R["results"], CHARTS_BY_TAB["behaviour"])
    kpis = compute_kpis_from_tables(chart_data)
    kpi_row(kpis)

    st.markdown("---")

    # Cohort Retention Heatmap
    section(
        "Cohort Retention: Do newer cohorts stay active?",
        "Rows = first-purchase month, Cols = months since first purchase. Darker = higher retention",
    )
    show_charts(R, ["08_cohort_retention"])

    # Cohort Revenue per Customer (LTV)
    st.markdown("---")
    section("Cohort Revenue & Retention", "Cohort retention heatmap + revenue quality")
    show_charts(R, ["22_cohort_ltv"])

    # Time-to-second-purchase survival curves
    st.markdown("---")
    section(
        "Activation Speed: Time to second purchase by segment",
        "Kaplan-Meier survival curves - faster rise = faster activation",
    )
    show_charts(R, ["07_repeat_survival"])

    # Inter-purchase interval violins
    st.markdown("---")
    section(
        "Purchase Rhythm: How regular are customers?",
        "Inter-purchase interval distribution by segment",
    )
    show_charts(R, ["16_cadence"])

    # Momentum chart
    st.markdown("---")
    section(
        "Momentum: Which segments are growing or shrinking?",
        "Recent vs prior period spend change by segment",
    )
    show_charts(R, ["14_momentum"])

    for stem, title in (
        ("repeat_survival", "Second-trip probabilities (all landmarks)"),
        ("cohort_retention_new_customers", "Cohort retention table"),
    ):
        t = table(R, stem)
        if len(t):
            with st.expander(title):
                show_df(pretty(t))


def tab_rfm(
    R: dict[str, Any], raw_df: pd.DataFrame | None = None, period_params: dict | None = None
) -> None:
    """RFM Analysis tab with customer segmentation, value concentration, and action recommendations."""
    st.header("RFM Analysis")
    st.caption(
        "Question answered here: which customers are champions, at risk, or need attention? How concentrated is revenue?"
    )

    cf = table(R, "customer_features")
    if cf.empty:
        st.info("No customer features available.")
        return

    # Build canonical customer features from raw transactions
    if raw_df is not None and period_params:
        analysis_date = period_params["current_end"]
        with st.spinner("Building canonical customer features..."):
            txn_features = build_customer_features(raw_df, analysis_date)
            # Merge with model features from pipeline
            rfm = merge_model_features(txn_features, cf)
    else:
        # Fallback: use pipeline features directly (they already have RFM)
        st.info("Using pre-computed RFM from analysis pipeline.")
        rfm = cf.copy()
        if "rfm_segment" in cf.columns:
            rfm["segment_rfm"] = cf["rfm_segment"]
        if "net_spend" in cf.columns:
            rfm["txn_monetary"] = cf["net_spend"]
        if "recency_days" in cf.columns:
            rfm["txn_recency_days"] = cf["recency_days"]
        if "n_trips" in cf.columns:
            rfm["txn_frequency"] = cf["n_trips"]
        if "customer_id" in cf.columns:
            rfm["customer_id"] = cf["customer_id"]

    # Ensure we have required columns for plotting
    required_cols = [
        "segment_rfm",
        "customer_id",
        "txn_monetary",
        "txn_recency_days",
        "txn_frequency",
    ]
    missing = [c for c in required_cols if c not in rfm.columns]
    if missing:
        st.warning(f"Missing columns for RFM visualization: {missing}")
        return

    # === RFM Segments Visualization ===
    st.subheader("RFM Segments")
    # Temporarily rename for plot_rfm_segments compatibility
    plot_df = rfm.rename(
        columns={
            "segment_rfm": "segment",
            "txn_monetary": "monetary",
            "txn_recency_days": "recency_days",
            "txn_frequency": "frequency",
        }
    )
    fig, axes = plot_rfm_segments(plot_df)
    champions_pct = (
        rfm[rfm["segment_rfm"] == "Champions"]["txn_monetary"].sum()
        / rfm["txn_monetary"].sum()
        * 100
        if "txn_monetary" in rfm.columns
        else 0
    )
    chart_card(
        fig,
        f"Identified {rfm['segment_rfm'].nunique()} RFM segments. Champions represent {champions_pct:.0f}% of revenue.",
        scope="As of analysis date · Recency/Frequency/Monetary quintiles · Tie-safe scoring",
    )

    # Segment summary table
    st.markdown("---")
    st.subheader("Segment Summary")
    seg_summary = (
        rfm.groupby("segment_rfm")
        .agg(
            customers=("customer_id", "count"),
            revenue=("txn_monetary", "sum")
            if "txn_monetary" in rfm.columns
            else ("customer_id", "count"),
            avg_recency=("txn_recency_days", "mean")
            if "txn_recency_days" in rfm.columns
            else ("customer_id", "count"),
            avg_frequency=("txn_frequency", "mean")
            if "txn_frequency" in rfm.columns
            else ("customer_id", "count"),
            avg_monetary=("txn_monetary", "mean")
            if "txn_monetary" in rfm.columns
            else ("customer_id", "count"),
        )
        .sort_values("revenue", ascending=False)
        .reset_index()
    )

    display_cols = ["segment_rfm", "customers"]
    if "revenue" in seg_summary.columns:
        display_cols.append("revenue")
    if "avg_recency" in seg_summary.columns:
        display_cols.extend(["avg_recency", "avg_frequency", "avg_monetary"])

    show_df(pretty(seg_summary[display_cols]))

    # === Customer Value Concentration (Pareto & Lorenz) ===
    if "txn_monetary" in rfm.columns:
        st.markdown("---")
        st.subheader("Customer Value Concentration")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Pareto: Top Customers by Lifetime Revenue**")
            # Temporarily rename for _plot_customer_pareto compatibility
            pareto_df = rfm.rename(columns={"txn_monetary": "monetary"})
            pareto_fig = _plot_customer_pareto(pareto_df)
            if pareto_fig:
                chart_card(
                    pareto_fig[0],
                    "Shows how much revenue is concentrated in top customers. 80% threshold indicates Pareto principle.",
                    scope="Full observed history · Lifetime revenue (txn_monetary) · Exclusive tier labels",
                )

        with col2:
            st.markdown("**Lorenz Curve: Revenue Inequality**")
            lorenz_df = rfm.rename(columns={"txn_monetary": "monetary"})
            lorenz_fig = _plot_customer_lorenz(lorenz_df)
            if lorenz_fig:
                chart_card(
                    lorenz_fig[0],
                    "Area between curve and diagonal = revenue concentration. Gini coefficient quantifies inequality.",
                    scope="Full observed history · Lifetime revenue (txn_monetary) · Excludes non-positive monetary",
                )

        # Value tiers
        st.markdown("---")
        st.markdown("**Value Tier Breakdown**")
        tiers_df = rfm.rename(columns={"txn_monetary": "monetary"})
        tiers_fig = _plot_customer_value_tiers(tiers_df)
        if tiers_fig:
            chart_card(
                tiers_fig[0],
                "Compares customer count share vs revenue share by value tier (exclusive bands).",
                scope="Full observed history · Lifetime revenue (txn_monetary) · Exclusive bands: Top 1%, Next 4%, Next 5%, Next 10%, Middle 50%, Bottom 30%",
            )

        # Concentration table
        st.markdown("**Concentration Summary**")
        n = len(rfm)
        rfm_sorted = rfm.sort_values("txn_monetary", ascending=False).reset_index(drop=True)
        rfm_sorted["cum_revenue_pct"] = (
            rfm_sorted["txn_monetary"].cumsum() / rfm_sorted["txn_monetary"].sum() * 100
        )
        rfm_sorted["cum_customer_pct"] = np.arange(1, n + 1) / n * 100

        # Exclusive bands with correct labels
        tiers = [
            ("Top 1%", 0, 1),
            ("Next 4%", 1, 5),
            ("Next 5%", 5, 10),
            ("Next 10%", 10, 20),
            ("Middle 50%", 20, 70),
            ("Bottom 30%", 70, 100),
        ]

        tier_data = []
        for label, pct_start, pct_end in tiers:
            start_idx = int(n * pct_start / 100)
            end_idx = int(n * pct_end / 100)
            tier_customers = rfm_sorted.iloc[start_idx:end_idx]
            if len(tier_customers) == 0:
                continue
            tier_data.append(
                {
                    "Tier": label,
                    "Customers": len(tier_customers),
                    "Customer %": f"{len(tier_customers) / n * 100:.1f}%",
                    "Revenue": fmt_currency(tier_customers["txn_monetary"].sum()),
                    "Revenue %": f"{tier_customers['txn_monetary'].sum() / rfm['txn_monetary'].sum() * 100:.1f}%",
                    "Avg Revenue/Cust": fmt_currency(tier_customers["txn_monetary"].mean()),
                }
            )

        tier_df = pd.DataFrame(tier_data)
        show_df(pretty(tier_df))

    # Action recommendations
    st.markdown("---")
    st.subheader("Recommended Actions by Segment")
    action_map = {
        "Champions": "🟢 **Protect & Grow** - VIP treatment, early access, loyalty rewards",
        "Loyal Customers": "🔵 **Cross-sell** - Bundle complementary categories, increase basket size",
        "Potential Loyalists": "🟡 **Nurture** - Targeted offers to increase frequency",
        "New Customers": "🟠 **Onboard** - Welcome series, second purchase incentives",
        "At Risk High Value": "🔴 **Win-back** - Personal outreach, special offers",
        "At Risk": "🔴 **Reactivate** - Win-back campaigns, feedback surveys",
        "Hibernating": "⚪ **Low-cost reactivation** - Email reminders, small discounts",
        "Needs Attention": "⚪ **Diagnose** - Analyze behavior patterns, test interventions",
    }

    for seg, action in action_map.items():
        count = len(rfm[rfm["segment_rfm"] == seg]) if seg in rfm["segment_rfm"].values else 0
        if count > 0:
            st.markdown(f"**{seg}** ({count:,} customers): {action}")


def tab_products(
    R: dict[str, Any], raw_df: pd.DataFrame | None = None, period_params: dict | None = None
) -> None:
    st.header("Products & Baskets (Pipeline)")
    st.caption("Pipeline outputs: affinity rules, category analysis, promo/return analysis.")

    chart_data = load_chart_data(R["results"], CHARTS_BY_TAB["products"])
    kpis = compute_kpis_from_tables(chart_data)
    kpi_row(kpis)

    st.markdown("---")

    # Affinity Network
    section(
        "Affinity Network: Cross-sell structure",
        "Network graph of statistically significant basket rules (FDR-controlled)",
    )
    show_charts(R, ["20_affinity_network"])

    aff = table(R, "basket_affinity")
    if len(aff) and "passes_all_filters" in aff.columns:
        ok = aff[aff["passes_all_filters"].astype(bool)]
        st.subheader("Supported basket rules")
        st.caption(
            f"{len(ok):,} of {len(aff):,} directed rules pass FDR control, an odds-ratio interval above 1 and split-half persistence."
        )
        if len(ok):
            show_df(pretty(ok))

    # Category Pareto
    st.markdown("---")
    section(
        "Category Pareto: Assortment concentration",
        "Top categories by revenue with cumulative share line",
    )
    show_charts(R, ["11_categories"])

    # Margin Risk: Return rate by category
    st.markdown("---")
    section(
        "Margin Risk: Return rate by category",
        "Lollipop chart showing return rate for top categories",
    )
    show_charts(R, ["21_promo_return_lollipop"])

    # Next-trip transitions
    st.markdown("---")
    section(
        "Next-Trip Transitions: What do customers buy next?",
        "Significant transitions that beat chance (FDR-controlled)",
    )
    nxt = table(R, "next_trip_affinity")
    if len(nxt):
        with st.expander("Next-trip transitions"):
            show_df(pretty(nxt))

    # Basket rules lift matrix
    show_charts(R, ["12_basket_rules", "13_next_trip"])

    # Period-based Basket Analysis (selected period) - using canonical builder
    if raw_df is not None and period_params:
        st.markdown("---")
        st.subheader("Product Cross-sell (Selected Period)")
        basket_data = build_basket_associations(raw_df, period_params)
        pairs = basket_data.get("pairs", pd.DataFrame())

        if pairs.empty:
            st.info("No product pairs meet the current support threshold in the selected period.")
        else:
            params = basket_data.get("parameters", {})
            st.caption(
                f"{len(pairs):,} product pairs found with min support {params.get('min_support', 0.01):.1%}."
            )

            # Top pairs by lift
            top_pairs_fig = _plot_basket_top_pairs(pairs, top_n=15)
            if top_pairs_fig:
                chart_card(
                    top_pairs_fig[0],
                    "Top product pairs by lift. Lift > 1 = positive association.",
                    scope=f"Selected period · Product-level · Min support: {params.get('min_support', 0.01):.1%} · Deduplicated per invoice",
                )

            # Heatmap
            heatmap_fig = _plot_basket_heatmap(pairs, top_n=20)
            if heatmap_fig:
                chart_card(
                    heatmap_fig[0],
                    "Lift matrix for top products. Red = positive association, Blue = negative.",
                    scope="Selected period · Top products by lift sum · NaN = not computed (not independence)",
                )

            # Network graph
            network_fig = _plot_basket_network(pairs, top_n=30)
            if network_fig:
                chart_card(
                    network_fig[0],
                    "Product association network. Edge width = lift, Node size = connections.",
                    scope="Selected period · Top products by lift · Deduplicated per invoice",
                )

    for stem, title in (
        ("category_summary", "Category summary"),
        ("department_summary", "Department summary"),
        ("product_summary", "Product summary"),
    ):
        t = table(R, stem)
        if len(t):
            with st.expander(title):
                show_df(pretty(t.head(500)))


def tab_explorer(
    R: dict[str, Any], raw_df: pd.DataFrame | None = None, period_params: dict | None = None
) -> None:
    import matplotlib.pyplot as plt

    st.header("Customer explorer")
    st.caption(
        "Filter customers, export a target list, or look up a single customer. Build your own view."
    )

    cf = table(R, "customer_features")
    if cf.empty:
        st.info("No customer table available.")
        return

    # Choose-your-metric x choose-your-dimension builder
    st.subheader("Chart Builder")
    c1, c2, c3 = st.columns(3)
    numeric_cols = cf.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = cf.select_dtypes(include=["object", "category"]).columns.tolist()

    chart_type = c1.selectbox("Chart type", ["Bar", "Line", "Heatmap", "Scatter", "Box"])
    x_dim = c2.selectbox(
        "X dimension (categorical)",
        cat_cols,
        index=cat_cols.index("cluster_label") if "cluster_label" in cat_cols else 0,
    )
    y_metric = c3.selectbox(
        "Y metric (numeric)",
        numeric_cols,
        index=numeric_cols.index("expected_revenue_horizon")
        if "expected_revenue_horizon" in numeric_cols
        else 0,
    )

    if chart_type == "Scatter":
        x_metric = c1.selectbox(
            "X metric (numeric)",
            numeric_cols,
            index=numeric_cols.index("p_alive") if "p_alive" in numeric_cols else 0,
        )
        color_dim = c2.selectbox(
            "Color by",
            cat_cols,
            index=cat_cols.index("cluster_label") if "cluster_label" in cat_cols else 0,
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        for lab in sorted(cf[color_dim].dropna().unique()):
            s = cf[cf[color_dim] == lab]
            ax.scatter(
                s[x_metric],
                s[y_metric],
                alpha=0.5,
                label=str(lab)[:30],
                color=segment_color(lab),
                s=15,
            )
        ax.set_xlabel(x_metric.replace("_", " ").title())
        ax.set_ylabel(y_metric.replace("_", " ").title())
        ax.set_title(f"{y_metric} vs {x_metric} by {color_dim}")
        ax.legend(fontsize=8, markerscale=2)
        chart_card(fig, f"Scatter of {len(cf)} customers colored by {color_dim}")

    elif chart_type == "Bar":
        agg = cf.groupby(x_dim)[y_metric].mean().sort_values(ascending=False).head(20)
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = [segment_color(idx) for idx in agg.index]
        ax.bar(range(len(agg)), agg.values, color=colors)
        ax.set_xticks(range(len(agg)))
        ax.set_xticklabels([str(x)[:25] for x in agg.index], rotation=45, ha="right")
        ax.set_ylabel(y_metric.replace("_", " ").title())
        ax.set_title(f"Mean {y_metric} by {x_dim}")
        chart_card(fig, f"Top {x_dim} by {y_metric}: {agg.index[0]} leads with {agg.iloc[0]:.1f}")

    elif chart_type == "Heatmap":
        if len(numeric_cols) >= 2:
            corr = cf[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
            ax.set_xticks(range(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(len(corr.columns)))
            ax.set_yticklabels(corr.columns, fontsize=8)
            ax.set_title("Metric Correlation Heatmap")
            fig.colorbar(im, ax=ax, label="Correlation")
            chart_card(fig, "Correlation between numeric customer features")

    elif chart_type == "Box":
        if "cluster_label" in cf.columns:
            groups = sorted(cf["cluster_label"].dropna().unique())
            data = [cf[cf["cluster_label"] == g][y_metric].dropna() for g in groups]
            fig, ax = plt.subplots(figsize=(10, 5))
            bp = ax.boxplot(
                data, tick_labels=[str(g)[:20] for g in groups], patch_artist=True, showfliers=False
            )
            for patch, g in zip(bp["boxes"], groups, strict=False):
                patch.set_facecolor(segment_color(g))
                patch.set_alpha(0.6)
            ax.set_ylabel(y_metric.replace("_", " ").title())
            ax.set_title(f"{y_metric} distribution by segment")
            chart_card(
                fig,
                f"Segment {max(groups, key=lambda g: cf[cf['cluster_label'] == g][y_metric].median())} has highest median {y_metric}",
            )

    # Drill-down: Period comparison by dimension
    if raw_df is not None and period_params:
        st.markdown("---")
        st.subheader("Drill-down: Period Comparison by Dimension")
        drill_dim = st.selectbox(
            "Dimension", cat_cols, index=cat_cols.index("category") if "category" in cat_cols else 0
        )
        drill_metric = st.selectbox(
            "Metric",
            ["revenue", "orders", "customers", "aov", "units_per_order"],
            index=numeric_cols.index("revenue") if "revenue" in numeric_cols else 0,
        )

        drill_fig = _plot_drilldown_path(raw_df, period_params, drill_dim, drill_metric, top_n=15)
        if drill_fig:
            chart_card(
                drill_fig[0],
                f"{drill_metric.replace('_', ' ').title()} by {drill_dim} for current vs prior period.",
            )

    st.markdown("---")

    # Segment comparison
    st.subheader("Segment Comparison")
    labels = sorted(cf["cluster_label"].dropna().unique()) if "cluster_label" in cf else []
    rfm = sorted(cf["rfm_segment"].dropna().unique()) if "rfm_segment" in cf else []
    col1, col2 = st.columns(2)
    seg_a = col1.selectbox("Segment A", labels, index=0 if labels else 0)
    seg_b = col2.selectbox("Segment B", labels, index=1 if len(labels) > 1 else 0)

    if seg_a != seg_b:
        metrics_to_compare = [c for c in numeric_cols if c in cf.columns][:10]
        comp_a = cf[cf["cluster_label"] == seg_a][metrics_to_compare].mean()
        comp_b = cf[cf["cluster_label"] == seg_b][metrics_to_compare].mean()
        diff = ((comp_b - comp_a) / comp_a.replace(0, np.nan) * 100).round(1)
        comp_df = pd.DataFrame(
            {seg_a: comp_a.round(2), seg_b: comp_b.round(2), "Diff %": diff}
        ).reset_index()
        comp_df.columns = ["Metric", seg_a, seg_b, "Diff %"]
        show_df(pretty(comp_df))

    st.markdown("---")

    # Existing filter and export
    st.subheader("Filter & Export")
    c1, c2, c3 = st.columns(3)
    sel_cl = c1.multiselect("Segment", labels, default=labels)
    sel_rfm = c2.multiselect("RFM group", rfm, default=rfm)
    lo, hi = c3.slider("P(alive) range", 0.0, 1.0, (0.0, 1.0), step=0.05)
    min_rev = st.number_input(
        "Minimum expected revenue",
        0.0,
        float(max(cf["expected_revenue_horizon"].max(), 1.0))
        if "expected_revenue_horizon" in cf
        else 1.0,
        0.0,
    )
    d = cf.copy()
    if labels:
        d = d[
            d["cluster_label"].isin(sel_cl) | d["cluster_label"].isna() & ("Unclustered" in sel_cl)
        ]
    if rfm:
        d = d[d["rfm_segment"].isin(sel_rfm)]
    if "p_alive" in d:
        d = d[(d["p_alive"].fillna(0) >= lo) & (d["p_alive"].fillna(0) <= hi)]
    if "expected_revenue_horizon" in d:
        d = d[d["expected_revenue_horizon"].fillna(0) >= min_rev].sort_values(
            "expected_revenue_horizon", ascending=False
        )
    default_cols = [
        c
        for c in [
            "customer_id",
            "cluster_label",
            "rfm_segment",
            "p_alive",
            "expected_trips_horizon",
            "expected_trip_value",
            "expected_revenue_horizon",
            "n_trips",
            "recency_days",
            "gross_spend",
            "return_value_ratio",
            "promo_spend_share",
            "top_category_by_spend",
        ]
        if c in d.columns
    ]
    cols = st.multiselect("Columns", list(d.columns), default=default_cols)
    st.write(f"**{len(d):,}** customers match. Showing the first 1,000.")
    show_df(d[cols].head(1000) if cols else d.head(1000))
    st.download_button(
        "Download filtered customers (CSV)",
        d[cols].to_csv(index=False).encode("utf-8")
        if cols
        else d.to_csv(index=False).encode("utf-8"),
        file_name="filtered_customers.csv",
        mime="text/csv",
    )

    st.subheader("Customer lookup")
    q = st.text_input("Customer id")
    if q:
        hit = cf[cf["customer_id"].astype(str) == q.strip()]
        if hit.empty:
            st.warning("No customer with that id in the scored table.")
        else:
            show_df(
                hit.T.reset_index()
                .rename(columns={"index": "field", hit.index[0]: "value"})
                .astype(str)
            )

    # Prioritized Action Matrix
    st.markdown("---")
    st.header("🎯 Prioritized Action Matrix")
    st.caption(
        "Actions derived from RFM, basket analysis, anomaly detection, and retention metrics."
    )

    if raw_df is not None and period_params:
        with st.spinner("Computing action priorities..."):
            # Compute basket and anomaly data
            basket_data = _compute_basket_analysis(raw_df, period_params)
            anomaly_data = _compute_anomalies(raw_df, period_params)

            # Prepare canonical customer features for actions
            if raw_df is not None and period_params:
                analysis_date = period_params["current_end"]
                txn_features = build_customer_features(raw_df, analysis_date)
                rfm_for_actions = merge_model_features(txn_features, cf)
            else:
                rfm_for_actions = cf.copy()
                if "rfm_segment" in cf.columns:
                    rfm_for_actions["segment_rfm"] = cf["rfm_segment"]
                if "net_spend" in cf.columns:
                    rfm_for_actions["txn_monetary"] = cf["net_spend"]

            # Get basket data using canonical builder
            basket_data = (
                build_basket_associations(raw_df, period_params)
                if raw_df is not None and period_params
                else {"pairs": pd.DataFrame()}
            )

            # Get anomaly data
            anomaly_data = (
                _compute_anomalies(raw_df, period_params)
                if raw_df is not None and period_params
                else {}
            )

            # Get retention metrics
            retention_metrics = (
                _compute_retention_metrics(raw_df, period_params)
                if raw_df is not None and period_params
                else {}
            )

            # Get category data for category opportunities
            category_data = (
                build_category_features(raw_df, period_params, "current")
                if raw_df is not None and period_params
                else pd.DataFrame()
            )

            # Generate and render actions
            actions = _generate_action_matrix(
                rfm_for_actions, basket_data, anomaly_data, category_data, retention_metrics
            )
            _render_action_matrix(actions)
    else:
        st.info("Action matrix requires raw transaction data and period selection.")


def tab_data(
    R: dict[str, Any], raw_df: pd.DataFrame | None = None, period_params: dict | None = None
) -> None:
    st.header("Data and downloads")
    st.caption("Question answered here: what is wrong with the data, and where are the files?")

    chart_data = load_chart_data(R["results"], CHARTS_BY_TAB["data"])
    kpis = compute_kpis_from_tables(chart_data)
    kpi_row(kpis)

    st.markdown("---")

    show_charts(R, CHARTS_BY_TAB["data"])

    q = table(R, "data_quality_issues")
    if len(q):
        st.subheader("All data-quality checks")
        show_df(pretty(q))

    mv = table(R, "product_metadata_variants")
    if len(mv):
        with st.expander("Products with conflicting category or department"):
            show_df(pretty(mv))

    st.subheader("Downloads")
    c1, c2, c3, c4 = st.columns(4)
    c1.download_button(
        "All result tables (ZIP)",
        zip_dir(R["results"]),
        file_name="customer_insights_tables.zip",
        mime="application/zip",
    )
    c2.download_button(
        "All charts (ZIP)",
        zip_dir(R["insights"]),
        file_name="customer_insights_charts.zip",
        mime="application/zip",
    )
    html_path = Path(R["insights"]) / "insights.html"
    if html_path.is_file():
        c3.download_button(
            "Dashboard (HTML)", html_path.read_bytes(), file_name="insights.html", mime="text/html"
        )
    rp = Path(R["results"]) / "analysis_report.md"
    if rp.is_file():
        c4.download_button(
            "Written report (Markdown)",
            rp.read_bytes(),
            file_name="analysis_report.md",
            mime="text/markdown",
        )


def _plot_customer_pareto(rfm: pd.DataFrame) -> tuple | None:
    """Pareto chart of customers by lifetime revenue."""
    try:
        if "monetary" not in rfm.columns:
            return None

        # Total revenue from FULL customer population
        total_revenue = rfm["monetary"].sum()
        n_total = len(rfm)

        pareto = (
            rfm.nlargest(min(200, len(rfm)), "monetary")
            .sort_values("monetary", ascending=False)
            .reset_index(drop=True)
        )
        # Cumulative share of TOTAL business, not just displayed subset
        pareto["cum_pct"] = pareto["monetary"].cumsum() / total_revenue * 100
        pareto["rank"] = range(1, len(pareto) + 1)

        import matplotlib.pyplot as plt

        fig, ax1 = plt.subplots(figsize=(12, 5))

        ax1.bar(
            range(len(pareto)), pareto["monetary"], color="#2b6cb0", edgecolor="white", width=0.7
        )
        ax1.set_ylabel("Lifetime Revenue", fontsize=10, color="#2b6cb0")
        ax1.tick_params(axis="y", labelcolor="#2b6cb0")
        ax1.set_xticks(range(0, len(pareto), max(1, len(pareto) // 20)))
        ax1.set_xticklabels(
            [str(r) for r in pareto["rank"].iloc[:: max(1, len(pareto) // 20)]], fontsize=8
        )
        ax1.set_xlabel("Customer Rank (by Lifetime Revenue)", fontsize=10)

        ax2 = ax1.twinx()
        ax2.plot(
            range(len(pareto)),
            pareto["cum_pct"],
            color="#e53e3e",
            marker="o",
            linewidth=2,
            markersize=3,
        )
        ax2.set_ylabel("Cumulative % of Total Revenue", fontsize=10, color="#e53e3e")
        ax2.tick_params(axis="y", labelcolor="#e53e3e")
        ax2.axhline(y=80, color="gray", linestyle="--", alpha=0.5, linewidth=1)
        ax2.text(len(pareto) - 1, 82, "80% threshold", fontsize=8, color="gray")

        # Add note about total share captured
        total_share = pareto["cum_pct"].iloc[-1] if len(pareto) > 0 else 0
        ax1.text(
            0.02,
            0.98,
            f"Top {len(pareto)} of {n_total} = {total_share:.1f}% of total",
            transform=ax1.transAxes,
            fontsize=9,
            va="top",
            ha="left",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8, "edgecolor": "gray"},
        )

        # Add reference lines for top tiers using FULL population count
        for pct, label in [(1, "Top 1%"), (5, "Top 5%"), (10, "Top 10%"), (20, "Top 20%")]:
            idx = int(n_total * pct / 100)
            if idx < len(pareto):
                ax1.axvline(x=idx, color="gray", linestyle=":", alpha=0.5, linewidth=1)
                ax1.text(
                    idx,
                    ax1.get_ylim()[1] * 0.95,
                    label,
                    fontsize=7,
                    rotation=90,
                    va="top",
                    ha="right",
                    color="gray",
                )

        ax1.set_title(
            "Customer Lifetime Value Pareto (Observed Historical)",
            fontsize=13,
            fontweight="bold",
            pad=15,
        )
        ax1.spines["top"].set_visible(False)
        ax2.spines["top"].set_visible(False)

        plt.tight_layout()
        return fig, (ax1, ax2)
    except Exception:
        return None


def _plot_customer_lorenz(rfm: pd.DataFrame) -> tuple | None:
    """Lorenz curve for customer revenue concentration."""
    try:
        if "monetary" not in rfm.columns:
            return None

        vals = rfm["monetary"].dropna().values
        vals = vals[vals > 0]
        if len(vals) < 2:
            return None

        vals = np.sort(vals)
        x = np.concatenate([[0], np.arange(1, len(vals) + 1) / len(vals)])
        y = np.concatenate([[0], np.cumsum(vals) / vals.sum()])
        gini = 1 - 2 * np.trapz(y, x)

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(x, y, color="#2b6cb0", lw=2, label=f"Observed (Gini {gini:.2f})")
        ax.plot([0, 1], [0, 1], "--", color="#1a202c", lw=1, label="Equality")

        # Shade area
        ax.fill_between(x, y, x, alpha=0.1, color="#2b6cb0")

        # Reference points - vals sorted ascending, so y[idx] = bottom share
        # Top share = 1 - bottom share
        for pct in [0.5, 0.8, 0.9, 0.95, 0.99]:
            idx = int(pct * len(x))
            if idx < len(x):
                ax.plot(x[idx], y[idx], "o", color="#e53e3e", markersize=4)
                top_share = (1 - y[idx]) * 100
                ax.annotate(
                    f"Top {int((1 - pct) * 100)}%: {top_share:.1f}% revenue",
                    xy=(x[idx], y[idx]),
                    xytext=(5, -15),
                    textcoords="offset points",
                    fontsize=8,
                    color="#e53e3e",
                )

        # Disclosure about excluded non-positive monetary
        n_excluded = len(rfm) - len(vals)
        if n_excluded > 0:
            ax.text(
                0.02,
                0.02,
                f"Note: {n_excluded} customers with non-positive monetary excluded",
                transform=ax.transAxes,
                fontsize=7,
                color="gray",
                va="bottom",
            )

        ax.set_xlabel("Cumulative Share of Customers", fontsize=10)
        ax.set_ylabel("Cumulative Share of Revenue", fontsize=10)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
        ax.set_title("Revenue Concentration: Lorenz Curve", fontsize=13, fontweight="bold", pad=15)
        ax.legend(loc="upper left")
        ax.grid(alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()
        return fig, ax
    except Exception:
        return None


def _plot_customer_value_tiers(rfm: pd.DataFrame) -> tuple | None:
    """Stacked bar: customer count share vs revenue share by value tier."""
    try:
        if "monetary" not in rfm.columns:
            return None

        n = len(rfm)
        rfm_sorted = rfm.sort_values("monetary", ascending=False).reset_index(drop=True)
        rfm_sorted["cum_revenue_pct"] = (
            rfm_sorted["monetary"].cumsum() / rfm_sorted["monetary"].sum() * 100
        )
        rfm_sorted["cum_customer_pct"] = np.arange(1, n + 1) / n * 100

        # Define tiers - using exclusive bands with correct labels
        tiers = [
            ("Top 1%", 0, 1),
            ("Next 4%", 1, 5),
            ("Next 5%", 5, 10),
            ("Next 10%", 10, 20),
            ("Middle 50%", 20, 70),
            ("Bottom 30%", 70, 100),
        ]

        tier_data = []
        for label, pct_start, pct_end in tiers:
            start_idx = int(n * pct_start / 100)
            end_idx = int(n * pct_end / 100)
            tier_customers = rfm_sorted.iloc[start_idx:end_idx]
            if len(tier_customers) == 0:
                continue
            tier_data.append(
                {
                    "tier": label,
                    "customers": len(tier_customers),
                    "customer_pct": len(tier_customers) / n * 100,
                    "revenue": tier_customers["monetary"].sum(),
                    "revenue_pct": tier_customers["monetary"].sum() / rfm["monetary"].sum() * 100,
                    "avg_revenue": tier_customers["monetary"].mean(),
                }
            )

        tier_df = pd.DataFrame(tier_data)

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Customer share
        colors = ["#2b6cb0", "#2f855a", "#38a169", "#4299e1", "#a0aec0", "#718096"]
        bars1 = axes[0].barh(
            range(len(tier_df)),
            tier_df["customer_pct"],
            color=colors[: len(tier_df)],
            edgecolor="white",
        )
        axes[0].set_xlabel("Share of Customers (%)", fontsize=10)
        axes[0].set_title(
            "Customer Count by Value Tier (Exclusive Bands)", fontsize=12, fontweight="bold"
        )
        axes[0].set_yticks(range(len(tier_df)))
        axes[0].set_yticklabels(tier_df["tier"], fontsize=9)
        axes[0].invert_yaxis()
        axes[0].spines["top"].set_visible(False)
        axes[0].spines["right"].set_visible(False)
        axes[0].grid(axis="x", alpha=0.3)

        for bar, val in zip(bars1, tier_df["customer_pct"], strict=False):
            axes[0].text(
                val + 0.5,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%",
                va="center",
                fontsize=9,
            )

        # Revenue share
        bars2 = axes[1].barh(
            range(len(tier_df)),
            tier_df["revenue_pct"],
            color=colors[: len(tier_df)],
            edgecolor="white",
        )
        axes[1].set_xlabel("Share of Revenue (%)", fontsize=10)
        axes[1].set_title("Revenue by Value Tier (Exclusive Bands)", fontsize=12, fontweight="bold")
        axes[1].set_yticks(range(len(tier_df)))
        axes[1].set_yticklabels(tier_df["tier"], fontsize=9)
        axes[1].invert_yaxis()
        axes[1].spines["top"].set_visible(False)
        axes[1].spines["right"].set_visible(False)
        axes[1].grid(axis="x", alpha=0.3)

        for bar, val in zip(bars2, tier_df["revenue_pct"], strict=False):
            axes[1].text(
                val + 0.5,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%",
                va="center",
                fontsize=9,
            )

        plt.tight_layout()
        return fig, axes
    except Exception:
        return None


def _compute_retention_metrics(
    raw_df: pd.DataFrame, period_params: dict, inactivity_window_days: int = 90
) -> dict:
    """Compute retention, repeat purchase, reactivation metrics.

    Args:
        raw_df: Normalized transaction data (from prepare_transaction_frame)
        period_params: Period parameters from sidebar
        inactivity_window_days: Days of inactivity before a customer is considered dormant
    """
    try:
        ctx = get_period_context(period_params)
        current_start = ctx["current_start"]
        current_end = ctx["current_end"]
        ctx["prior_start"]
        ctx["prior_end"]

        curr_df = get_period_transactions(raw_df, ctx, "current")
        prior_df = get_period_transactions(raw_df, ctx, "prior")

        curr_customers = set(curr_df["customer_id"].dropna().unique())
        prior_customers = set(prior_df["customer_id"].dropna().unique())

        # All-time customers up to current period start
        all_time_df = raw_df[raw_df["transaction_day"] < current_start]
        all_time_customers = set(all_time_df["customer_id"].dropna().unique())

        # Retention: customers active in both periods
        retained = curr_customers & prior_customers
        retention_rate = len(retained) / len(prior_customers) if prior_customers else 0

        # Reactivation: customers dormant at period start who purchased in current period
        # Dormant = no purchase in [current_start - inactivity_window_days, current_start)
        dormancy_cutoff = current_start - pd.Timedelta(days=inactivity_window_days)
        recently_active = set(
            raw_df[
                (raw_df["transaction_day"] < current_start)
                & (raw_df["transaction_day"] >= dormancy_cutoff)
            ]["customer_id"]
            .dropna()
            .unique()
        )
        dormant = all_time_customers - recently_active
        reactivated = dormant & curr_customers
        reactivation_rate = len(reactivated) / len(dormant) if dormant else 0

        # New customers: first purchase in current period
        new_customers = curr_customers - all_time_customers
        new_customer_share = len(new_customers) / len(curr_customers) if curr_customers else 0

        # Repeat purchase rate (all time)
        all_purchases = raw_df[raw_df["transaction_day"] <= current_end]
        cust_orders = all_purchases.groupby("customer_id")["transaction_id"].nunique()
        repeat_cust = (cust_orders > 1).sum()
        total_cust = len(cust_orders)
        repeat_purchase_rate = repeat_cust / total_cust if total_cust > 0 else 0

        # Repeat revenue share (revenue already calculated)
        repeat_revenue = all_purchases[
            all_purchases["customer_id"].isin(cust_orders[cust_orders > 1].index)
        ]["revenue"].sum()
        total_revenue = all_purchases["revenue"].sum()
        repeat_revenue_share = repeat_revenue / total_revenue if total_revenue > 0 else 0

        # Inactive valuable: top 20% by historical revenue, not in current
        hist_rev = (
            all_purchases.groupby("customer_id")["revenue"].sum().sort_values(ascending=False)
        )
        top_20_pct = int(len(hist_rev) * 0.2)
        valuable_cust = set(hist_rev.head(top_20_pct).index) if top_20_pct > 0 else set()
        inactive_valuable = valuable_cust - curr_customers

        return {
            "retention_rate": retention_rate,
            "reactivation_rate": reactivation_rate,
            "new_customer_share": new_customer_share,
            "repeat_purchase_rate": repeat_purchase_rate,
            "repeat_revenue_share": repeat_revenue_share,
            "curr_customers": len(curr_customers),
            "prior_customers": len(prior_customers),
            "retained": len(retained),
            "reactivated": len(reactivated),
            "new_customers": len(new_customers),
            "inactive_valuable_count": len(inactive_valuable),
            "inactive_valuable_revenue": hist_rev.loc[list(inactive_valuable)].sum()
            if inactive_valuable
            else 0,
            "inactivity_window_days": inactivity_window_days,
            "dormancy_cutoff": dormancy_cutoff,
        }
    except Exception:
        return {}


def _plot_retention_metrics(metrics: dict) -> tuple | None:
    """Bullet chart for retention metrics."""
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 4))

        metrics_list = [
            ("Retention Rate", metrics.get("retention_rate", 0)),
            ("Reactivation Rate", metrics.get("reactivation_rate", 0)),
            ("Repeat Purchase Rate", metrics.get("repeat_purchase_rate", 0)),
            ("Repeat Revenue Share", metrics.get("repeat_revenue_share", 0)),
            ("New Customer Share", metrics.get("new_customer_share", 0)),
        ]

        y_pos = np.arange(len(metrics_list))
        values = [m[1] for m in metrics_list]
        labels = [m[0] for m in metrics_list]

        bars = ax.barh(
            y_pos, [v * 100 for v in values], color="#2b6cb0", edgecolor="white", height=0.6
        )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel("Rate (%)", fontsize=10)
        ax.set_title("Retention & Loyalty Metrics", fontsize=13, fontweight="bold", pad=15)
        ax.set_xlim(0, 100)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", alpha=0.3)

        for bar, val in zip(bars, values, strict=False):
            ax.text(
                bar.get_width() + 1,
                bar.get_y() + bar.get_height() / 2,
                f"{val * 100:.1f}%",
                va="center",
                fontsize=10,
                fontweight="bold",
            )

        # Add reference lines
        for ref in [20, 40, 60, 80]:
            ax.axvline(ref, color="gray", linestyle=":", alpha=0.3, linewidth=0.5)

        plt.tight_layout()
        return fig, ax
    except Exception:
        return None


def _plot_category_mix_trend(raw_df: pd.DataFrame, period_params: dict) -> tuple | None:
    """Stacked area chart of category revenue mix over time."""
    try:
        ctx = get_period_context(period_params)
        current_start = ctx["current_start"]
        current_end = ctx["current_end"]
        prior_start = ctx["prior_start"]
        prior_end = ctx["prior_end"]

        # Use wider window for trend
        full_start = min(prior_start, current_start) - pd.Timedelta(days=90)
        full_end = max(prior_end, current_end)

        df = raw_df[
            (raw_df["transaction_day"] >= full_start) & (raw_df["transaction_day"] <= full_end)
        ]

        if not ensure_not_empty(df, "category mix trend"):
            return None

        # Determine granularity
        period_days = (full_end - full_start).days
        if period_days <= 60:
            freq = "W-MON"
        else:
            freq = "M"

        # Get top 6 categories by revenue
        top_cats = df.groupby("category")["revenue"].sum().nlargest(6).index
        df["category"] = df["category"].where(df["category"].isin(top_cats), "Other")

        # Pivot
        pivot = (
            df.groupby([pd.Grouper(key="transaction_day", freq=freq), "category"])["revenue"]
            .sum()
            .unstack(fill_value=0)
        )

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 5))

        colors = ["#2b6cb0", "#2f855a", "#dd6b20", "#6b46c1", "#975a16", "#d53f8c", "#a0aec0"]
        pivot.plot.area(ax=ax, color=colors[: len(pivot.columns)], alpha=0.7)

        # Highlight current period
        ax.axvspan(current_start, current_end, alpha=0.15, color="yellow", label="Current Period")
        if period_params["compare_mode"] == "prior":
            ax.axvspan(prior_start, prior_end, alpha=0.15, color="blue", label="Prior Period")
        else:
            ax.axvspan(prior_start, prior_end, alpha=0.15, color="orange", label="Prior Year")

        ax.set_ylabel("Revenue", fontsize=10)
        ax.set_title(
            f"Category Revenue Mix Over Time ({freq} granularity)",
            fontsize=13,
            fontweight="bold",
            pad=15,
        )
        ax.legend(loc="upper left", fontsize=8, ncol=2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: fmt_currency(x)))

        fig.autofmt_xdate(rotation=30)
        plt.tight_layout()
        return fig, ax
    except Exception:
        return None


# ============================================================
# 08. Product and category analytics
# ============================================================


# ============================================================
# 09. Basket and cross-sell analytics
# ============================================================
def _plot_product_quadrant_from_features(prod_features: pd.DataFrame) -> tuple | None:
    """Quadrant scatter from pre-computed product features."""
    try:
        if prod_features.empty or "basket_penetration" not in prod_features.columns:
            return None

        top_prods = prod_features.nlargest(50, "revenue")

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 7))

        # Color by department
        dept_colors = dict(
            zip(
                top_prods["department"].unique(),
                ["#2b6cb0", "#2f855a", "#dd6b20", "#6b46c1", "#975a16", "#d53f8c"],
                strict=False,
            )
        )

        for dept in top_prods["department"].unique():
            dept_data = top_prods[top_prods["department"] == dept]
            ax.scatter(
                dept_data["basket_penetration"],
                dept_data["rev_per_order"],
                s=dept_data["revenue"] / top_prods["revenue"].max() * 1000 + 50,
                alpha=0.6,
                label=dept,
                color=dept_colors.get(dept, "#4a5568"),
                edgecolors="white",
                linewidth=0.5,
            )

        # Quadrant lines
        med_pen = top_prods["basket_penetration"].median()
        med_rev = top_prods["rev_per_order"].median()
        ax.axvline(med_pen, color="gray", linestyle="--", alpha=0.5)
        ax.axhline(med_rev, color="gray", linestyle="--", alpha=0.5)

        # Quadrant labels
        ax.text(
            0.05,
            0.95,
            "Niche High-Value\n(Low Penetration, High Rev/Order)",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "#bee3f8", "alpha": 0.7},
        )
        ax.text(
            0.95,
            0.95,
            "Broad-Reach High-Value\n(High Penetration, High Rev/Order)",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "#c6f6d5", "alpha": 0.7},
        )
        ax.text(
            0.05,
            0.05,
            "Low-Impact\n(Low Penetration, Low Rev/Order)",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "#feebc8", "alpha": 0.7},
        )
        ax.text(
            0.95,
            0.05,
            "High-Volume, Low-Value\n(High Penetration, Low Rev/Order)",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "#fed7d7", "alpha": 0.7},
        )

        ax.set_xlabel("Basket Penetration (% of orders)", fontsize=10)
        ax.set_ylabel("Revenue per Order", fontsize=10)
        ax.set_title(
            "Product Portfolio: Penetration vs Revenue per Order",
            fontsize=13,
            fontweight="bold",
            pad=15,
        )
        ax.legend(fontsize=8, loc="upper right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        return fig, ax
    except Exception:
        return None


def _plot_product_rank_change_from_features(prod_features: pd.DataFrame) -> tuple | None:
    """Rank change chart from pre-computed product features."""
    try:
        if prod_features.empty or "prior_revenue_rank" not in prod_features.columns:
            return None

        # Top 15 by current revenue
        top_15 = prod_features.nsmallest(15, "revenue_rank")

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))

        y_pos = np.arange(len(top_15))
        ax.hlines(
            y_pos,
            top_15["prior_revenue_rank"],
            top_15["revenue_rank"],
            color="#a0aec0",
            linewidth=2,
        )
        ax.plot(
            top_15["prior_revenue_rank"],
            y_pos,
            "o",
            color="#a0aec0",
            markersize=8,
            label="Prior Rank",
        )
        ax.plot(
            top_15["revenue_rank"], y_pos, "o", color="#2b6cb0", markersize=8, label="Current Rank"
        )

        # Add rank change labels
        for i, (_, row) in enumerate(top_15.iterrows()):
            change = row["rank_change"]
            if change > 0:
                color = "#2f855a"
                label = f"▲ {change}"
            elif change < 0:
                color = "#e53e3e"
                label = f"▼ {abs(change)}"
            else:
                color = "#718096"
                label = "—"
            ax.text(
                max(row["prior_revenue_rank"], row["revenue_rank"]) + 0.5,
                i,
                label,
                va="center",
                fontsize=10,
                fontweight="bold",
                color=color,
            )
            # Product name
            name = row.get("product_description", row.get("product_id", "Unknown"))
            ax.text(
                min(row["prior_revenue_rank"], row["revenue_rank"]) - 0.5,
                i,
                str(name)[:30],
                ha="right",
                va="center",
                fontsize=9,
            )

        ax.set_yticks(y_pos)
        ax.set_yticklabels([])
        ax.set_xlabel("Revenue Rank", fontsize=10)
        ax.set_title(
            "Product Rank Change: Current vs Prior Period", fontsize=13, fontweight="bold", pad=15
        )
        ax.legend(fontsize=9)
        ax.invert_xaxis()  # Rank 1 on left
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", alpha=0.3)

        plt.tight_layout()
        return fig, ax
    except Exception:
        return None


def _plot_top_product_trends_from_features(
    raw_df: pd.DataFrame, period_params: dict, prod_features: pd.DataFrame, top_n: int = 8
) -> tuple | None:
    """Small multiples trend for top N products using pre-computed features."""
    try:
        ctx = get_period_context(period_params)
        current_start = ctx["current_start"]
        current_end = ctx["current_end"]

        # Wider window for trend
        full_start = current_start - pd.Timedelta(days=180)
        full_end = current_end

        df = raw_df[
            (raw_df["transaction_day"] >= full_start) & (raw_df["transaction_day"] <= full_end)
        ]

        if not ensure_not_empty(df, "top product trends"):
            return None

        # Top N products by revenue in current period (from pre-computed features)
        top_prods = prod_features.nlargest(top_n, "revenue")["product_id"].tolist()

        # Determine granularity
        period_days = (full_end - full_start).days
        if period_days <= 60:
            freq = "W-MON"
        else:
            freq = "M"

        df_top = df[df["product_id"].isin(top_prods)]
        if "product_description" in raw_df.columns:
            desc_map = raw_df[["product_id", "product_description"]].drop_duplicates()
            df_top = df_top.merge(desc_map, on="product_id", how="left")
        else:
            df_top["product_description"] = df_top["product_id"]

        pivot = (
            df_top.groupby([pd.Grouper(key="transaction_day", freq=freq), "product_description"])[
                "revenue"
            ]
            .sum()
            .unstack(fill_value=0)
        )

        import matplotlib.pyplot as plt

        n_cols = 4
        n_rows = int(np.ceil(top_n / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2.5 * n_rows), sharey=True)
        axes = axes.flatten() if top_n > 1 else [axes]

        colors = [
            "#2b6cb0",
            "#2f855a",
            "#dd6b20",
            "#6b46c1",
            "#975a16",
            "#d53f8c",
            "#4a5568",
            "#0987a0",
        ]

        for i, (ax, prod) in enumerate(zip(axes, pivot.columns, strict=False)):
            if i >= top_n:
                ax.set_visible(False)
                continue

            color = colors[i % len(colors)]
            ax.plot(
                pivot.index, pivot[prod].values, color=color, linewidth=2, marker="o", markersize=4
            )
            ax.fill_between(pivot.index, pivot[prod].values, alpha=0.1, color=color)

            # Highlight current period
            ax.axvspan(current_start, current_end, alpha=0.15, color="yellow")

            ax.set_title(prod[:25], fontsize=9, fontweight="bold")
            ax.set_ylabel("Revenue", fontsize=8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(axis="y", alpha=0.3)
            ax.tick_params(axis="x", rotation=30, labelsize=7)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: fmt_currency(x)))

        fig.suptitle(f"Top {top_n} Products: Revenue Trend", fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        return fig, axes
    except Exception:
        return None


def _plot_category_contribution_from_features(cat_features: pd.DataFrame) -> tuple | None:
    """Category growth vs share scatter from pre-computed features."""
    try:
        if cat_features.empty or "growth" not in cat_features.columns:
            return None

        merged = cat_features.copy()
        total_rev = merged["revenue"].sum()

        # Filter to categories with meaningful revenue
        merged = merged[merged["revenue"] > total_rev * 0.005].copy()

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 7))

        # Quadrant lines
        total_growth = merged["growth"].mean() if len(merged) > 0 else 0
        ax.axvline(
            total_growth,
            color="gray",
            linestyle="--",
            alpha=0.5,
            linewidth=1,
            label=f"Total Business: {total_growth:.1f}%",
        )
        ax.axhline(
            merged["revenue_share"].median(), color="gray", linestyle="--", alpha=0.5, linewidth=1
        )

        # Scatter
        scatter = ax.scatter(
            merged["growth"],
            merged["revenue_share"],
            s=merged["revenue"] / merged["revenue"].max() * 2000 + 50,
            c=merged["revenue_change"],
            cmap="RdYlGn",
            alpha=0.7,
            edgecolors="white",
            linewidth=0.5,
        )

        # Labels
        for _, row in merged.iterrows():
            ax.annotate(
                str(row["category"])[:20],
                (row["growth"], row["revenue_share"]),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=8,
                alpha=0.8,
            )

        ax.set_xlabel("Revenue Growth (%)", fontsize=10)
        ax.set_ylabel("Revenue Share (%)", fontsize=10)
        ax.set_title(
            "Category Performance: Growth vs Share", fontsize=13, fontweight="bold", pad=15
        )

        # Quadrant labels
        ax.text(
            0.95,
            0.95,
            "Scale Drivers\n(High Share, High Growth)",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "#c6f6d5", "alpha": 0.7},
        )
        ax.text(
            0.05,
            0.95,
            "Emerging Stars\n(Low Share, High Growth)",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "#bee3f8", "alpha": 0.7},
        )
        ax.text(
            0.95,
            0.05,
            "Declining Priorities\n(High Share, Low Growth)",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "#fed7d7", "alpha": 0.7},
        )
        ax.text(
            0.05,
            0.05,
            "Small Declines\n(Low Share, Low Growth)",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "#feebc8", "alpha": 0.7},
        )

        plt.colorbar(scatter, ax=ax, label="Revenue Change")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        return fig, ax
    except Exception:
        return None


def _compute_basket_analysis(
    raw_df: pd.DataFrame, period_params: dict, min_support: float = 0.01
) -> dict:
    """Compute product pair associations: support, confidence, lift.

    Uses the shared affinity engine to avoid unbounded pair materialization.
    """
    result = analyze_affinity_from_app(
        raw_df=raw_df,
        period_params=period_params,
        min_support=min_support,
        basket_level="product",
        basket_grain="trip",
        max_candidates=100,
        max_baskets=5000,
        max_items_per_basket=30,
        max_pair_operations=100_000,
    )

    pairs = result.get("pairs", pd.DataFrame())
    if not pairs.empty:
        pairs = pairs.rename(columns={
            "antecedent": "product_a",
            "consequent": "product_b",
            "pair_baskets": "co_occurrence",
            "basket_support": "support",
            "confidence": "confidence_a_to_b",
            "basket_lift": "lift",
        })

    return {
        "single": result.get("single", pd.DataFrame()),
        "pairs": pairs,
    }


def _plot_basket_network(pairs_df: pd.DataFrame, top_n: int = 30) -> tuple | None:
    """Network graph of top product associations."""
    try:
        import networkx as nx

        if pairs_df.empty or len(pairs_df) < 2:
            return None

        # Take top N by lift
        top_pairs = pairs_df.head(top_n).copy()

        G = nx.Graph()
        for _, row in top_pairs.iterrows():
            G.add_edge(
                row["product_a"],
                row["product_b"],
                weight=row["lift"],
                support=row["support"],
                confidence=row["confidence_a_to_b"],
            )

        if len(G.nodes()) == 0:
            return None

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 10))

        # Spring layout
        pos = nx.spring_layout(G, k=3, iterations=50, seed=42)

        # Node sizes by degree
        node_sizes = [G.degree(n) * 300 + 200 for n in G.nodes()]

        # Edge widths by lift
        edges = G.edges()
        weights = [G[u][v]["weight"] for u, v in edges]
        max_w = max(weights) if weights else 1
        min_w = min(weights) if weights else 1

        # Draw edges
        for (u, v), w in zip(edges, weights, strict=False):
            width = 1 + 4 * (w - min_w) / max(1e-6, max_w - min_w)
            nx.draw_networkx_edges(
                G, pos, edgelist=[(u, v)], width=width, alpha=0.5, edge_color="#2b6cb0", ax=ax
            )

        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="#2b6cb0", alpha=0.7, ax=ax)

        # Labels (shortened)
        labels = {n: str(n)[:15] for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight="bold", ax=ax)

        ax.set_title(
            f"Product Association Network (Top {len(G.nodes())} Products, {len(G.edges())} Pairs)",
            fontsize=13,
            fontweight="bold",
            pad=15,
        )
        ax.set_axis_off()

        plt.tight_layout()
        return fig, ax
    except Exception:
        return None


def _plot_basket_heatmap(pairs_df: pd.DataFrame, top_n: int = 20) -> tuple | None:
    """Heatmap of product pair lift."""
    try:
        if pairs_df.empty:
            return None

        # Get all unique products from BOTH columns
        all_prods = pd.concat([pairs_df["product_a"], pairs_df["product_b"]]).unique()
        if len(all_prods) < 2:
            return None

        # Limit to top N
        if len(all_prods) > top_n:
            # Score products by sum of lift in their pairs
            prod_scores = (
                pairs_df.groupby("product_a")["lift"]
                .sum()
                .add(pairs_df.groupby("product_b")["lift"].sum(), fill_value=0)
                .sort_values(ascending=False)
            )
            top_prods = prod_scores.head(top_n).index.tolist()
            pairs_df = pairs_df[
                pairs_df["product_a"].isin(top_prods) & pairs_df["product_b"].isin(top_prods)
            ].copy()

        # Create matrix using UNION of both columns
        products = sorted(pd.concat([pairs_df["product_a"], pairs_df["product_b"]]).unique())
        if len(products) < 2:
            return None

        # Initialize with NaN (missing = no association computed, not independence)
        matrix = pd.DataFrame(np.nan, index=products, columns=products, dtype=float)

        for _, row in pairs_df.iterrows():
            matrix.loc[row["product_a"], row["product_b"]] = row["lift"]
            matrix.loc[row["product_b"], row["product_a"]] = row["lift"]

        # Mask diagonal (self-association)
        np.fill_diagonal(matrix.values, np.nan)

        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm

        fig, ax = plt.subplots(figsize=(10, 8))

        vals = matrix.to_numpy(float)
        hi = max(float(np.nanmax(vals)), 1.05)
        lo = min(float(np.nanmin(vals)), 0.95)

        cmap = plt.get_cmap("RdBu_r").copy()
        cmap.set_bad("#f7fafc")

        norm = TwoSlopeNorm(vcenter=1.0, vmin=min(lo, 0.5), vmax=hi)
        im = ax.imshow(np.ma.masked_invalid(vals), cmap=cmap, norm=norm, aspect="auto")

        ax.set_xticks(range(len(products)))
        ax.set_xticklabels([str(p)[:15] for p in products], rotation=60, ha="right", fontsize=8)
        ax.set_yticks(range(len(products)))
        ax.set_yticklabels([str(p)[:15] for p in products], fontsize=8)

        ax.set_title(
            "Product Pair Lift Matrix (Lift > 1 = Positive Association)",
            fontsize=13,
            fontweight="bold",
            pad=15,
        )

        fig.colorbar(im, ax=ax, label="Lift")

        plt.tight_layout()
        return fig, ax
    except Exception:
        return None


def _plot_basket_top_pairs(pairs_df: pd.DataFrame, top_n: int = 15) -> tuple | None:
    """Horizontal bar chart of top pairs by lift."""
    try:
        if pairs_df.empty:
            return None

        top = pairs_df.head(top_n).iloc[::-1].copy()

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(top) + 1)))

        labels = [f"{row['product_a'][:12]} + {row['product_b'][:12]}" for _, row in top.iterrows()]
        y_pos = np.arange(len(top))

        bars = ax.barh(y_pos, top["lift"], color="#2b6cb0", edgecolor="white", height=0.6)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Lift", fontsize=10)
        ax.set_title(
            f"Top {len(top)} Product Pairs by Lift", fontsize=13, fontweight="bold", pad=15
        )
        ax.axvline(x=1, color="gray", linestyle="--", alpha=0.5, linewidth=1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", alpha=0.3)

        # Add annotations
        for bar, (_, row) in zip(bars, top.iterrows(), strict=False):
            ax.text(
                bar.get_width() + 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"Lift: {row['lift']:.2f} | Supp: {row['support'] * 100:.1f}% | Conf: {row['confidence_a_to_b'] * 100:.1f}%",
                va="center",
                fontsize=8,
            )

        plt.tight_layout()
        return fig, ax
    except Exception:
        return None


def _compute_anomalies(
    raw_df: pd.DataFrame, period_params: dict, window_days: int = 28, k: float = 2.0
) -> dict:
    """Detect anomalies in daily KPIs using rolling mean ± k*std."""
    try:
        ctx = get_period_context(period_params)
        current_start = ctx["current_start"]
        current_end = ctx["current_end"]

        # Use wider window for rolling stats
        full_start = current_start - pd.Timedelta(days=window_days * 2)
        full_end = current_end

        df = raw_df[
            (raw_df["transaction_day"] >= full_start) & (raw_df["transaction_day"] <= full_end)
        ]

        if not ensure_not_empty(df, "anomaly detection"):
            return {}

        # Daily aggregation (revenue already calculated)
        daily = (
            df.groupby("transaction_day")
            .agg(
                revenue=("revenue", "sum"),
                orders=("transaction_id", "nunique"),
                customers=("customer_id", "nunique"),
                units=("quantity", "sum"),
            )
            .reset_index()
        )

        daily["aov"] = daily["revenue"] / daily["orders"].replace(0, np.nan)
        daily["units_per_order"] = daily["units"] / daily["orders"].replace(0, np.nan)
        daily["revenue_per_customer"] = daily["revenue"] / daily["customers"].replace(0, np.nan)

        # Rolling stats - use LAGGED history so current day doesn't affect its own expected value
        metrics = [
            "revenue",
            "orders",
            "customers",
            "aov",
            "units_per_order",
            "revenue_per_customer",
        ]
        anomalies = {}

        for metric in metrics:
            if metric not in daily.columns or daily[metric].isna().all():
                continue

            # Shift by 1 so today's value doesn't influence today's expected range
            history = daily[metric].shift(1)
            rolling_mean = history.rolling(window=window_days, min_periods=7).mean()
            rolling_std = history.rolling(window=window_days, min_periods=7).std()

            upper = rolling_mean + k * rolling_std
            lower = rolling_mean - k * rolling_std

            # Store metric-specific bands for plotting
            daily[f"{metric}_expected"] = rolling_mean
            daily[f"{metric}_upper"] = upper
            daily[f"{metric}_lower"] = lower

            # Flag anomalies in current period
            curr_mask = (daily["transaction_day"] >= current_start) & (
                daily["transaction_day"] <= current_end
            )
            curr_data = daily[curr_mask].copy()

            metric_anomalies = []
            for _, row in curr_data.iterrows():
                idx = row.name
                val = row[metric]
                exp = rolling_mean.loc[idx]
                up = upper.loc[idx]
                lo = lower.loc[idx]

                if pd.notna(val) and pd.notna(exp) and (val > up or val < lo):
                    pct_dev = (val - exp) / exp * 100 if exp != 0 else 0
                    metric_anomalies.append(
                        {
                            "date": row["transaction_day"],
                            "value": val,
                            "expected": exp,
                            "upper": up,
                            "lower": lo,
                            "pct_deviation": pct_dev,
                            "direction": "high" if val > up else "low",
                        }
                    )

            if metric_anomalies:
                anomalies[metric] = metric_anomalies

        return {
            "daily": daily,
            "anomalies": anomalies,
            "current_start": current_start,
            "current_end": current_end,
        }
    except Exception:
        return {}


def _plot_anomaly_timeseries(anomaly_data: dict, metric: str) -> tuple | None:
    """Time series with anomaly markers for a specific metric."""
    try:
        daily = anomaly_data.get("daily", pd.DataFrame())
        anomalies = anomaly_data.get("anomalies", {}).get(metric, [])
        current_start = anomaly_data.get("current_start")
        current_end = anomaly_data.get("current_end")

        if daily.empty or metric not in daily.columns:
            return None

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 4))

        # Plot all data
        ax.plot(
            daily["transaction_day"],
            daily[metric],
            color="#1a202c",
            linewidth=1.5,
            label="Actual",
            alpha=0.7,
        )

        # Plot expected band if available (metric-specific columns)
        exp_col = f"{metric}_expected"
        upper_col = f"{metric}_upper"
        lower_col = f"{metric}_lower"
        if exp_col in daily.columns and upper_col in daily.columns and lower_col in daily.columns:
            ax.fill_between(
                daily["transaction_day"],
                daily[lower_col],
                daily[upper_col],
                alpha=0.2,
                color="#2b6cb0",
                label="Expected Range (±2σ)",
            )
            ax.plot(
                daily["transaction_day"],
                daily[exp_col],
                color="#2b6cb0",
                linewidth=1,
                linestyle="--",
                label="Expected",
                alpha=0.7,
            )

        # Highlight current period
        if current_start and current_end:
            (daily["transaction_day"] >= current_start) & (daily["transaction_day"] <= current_end)
            ax.axvspan(
                current_start, current_end, alpha=0.1, color="yellow", label="Current Period"
            )

        # Mark anomalies
        if anomalies:
            anom_dates = [a["date"] for a in anomalies]
            anom_vals = [a["value"] for a in anomalies]
            ax.scatter(
                anom_dates,
                anom_vals,
                color="#e53e3e",
                s=80,
                marker="X",
                zorder=10,
                label="Anomaly",
                edgecolors="white",
                linewidth=1,
            )

            for a in anomalies:
                ax.annotate(
                    f"{a['pct_deviation']:+.0f}%",
                    xy=(a["date"], a["value"]),
                    xytext=(0, 15),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                    color="#e53e3e",
                    fontweight="bold",
                )

        ax.set_title(
            f"{metric.replace('_', ' ').title()} - Anomaly Detection (Lagged Baseline)",
            fontsize=13,
            fontweight="bold",
            pad=15,
        )
        ax.set_ylabel(metric.replace("_", " ").title(), fontsize=10)
        ax.legend(loc="upper left", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3)

        fig.autofmt_xdate(rotation=30)
        plt.tight_layout()
        return fig, ax
    except Exception:
        return None


def _plot_anomaly_summary(anomaly_data: dict) -> tuple | None:
    """Summary bar chart of all anomalies by metric."""
    try:
        anomalies = anomaly_data.get("anomalies", {})
        if not anomalies:
            return None

        # Count anomalies per metric
        metric_counts = {m: len(v) for m, v in anomalies.items() if v}
        if not metric_counts:
            return None

        # Get max deviation per metric
        metric_max_dev = {}
        for m, v in anomalies.items():
            if v:
                metric_max_dev[m] = max(abs(a["pct_deviation"]) for a in v)

        metrics = list(metric_counts.keys())
        counts = [metric_counts[m] for m in metrics]
        max_devs = [metric_max_dev.get(m, 0) for m in metrics]

        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Count
        bars1 = ax1.barh(metrics, counts, color="#e53e3e", edgecolor="white")
        ax1.set_xlabel("Anomaly Count", fontsize=10)
        ax1.set_title("Anomalies by Metric (Current Period)", fontsize=12, fontweight="bold")
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax1.grid(axis="x", alpha=0.3)

        for bar, c in zip(bars1, counts, strict=False):
            ax1.text(
                bar.get_width() + 0.1,
                bar.get_y() + bar.get_height() / 2,
                str(c),
                va="center",
                fontsize=9,
            )

        # Max deviation
        colors = ["#e53e3e" if d > 0 else "#2f855a" for d in max_devs]
        bars2 = ax2.barh(metrics, max_devs, color=colors, edgecolor="white")
        ax2.set_xlabel("Max % Deviation from Expected", fontsize=10)
        ax2.set_title("Largest Deviation by Metric", fontsize=12, fontweight="bold")
        ax2.axvline(x=0, color="gray", linewidth=0.5)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.grid(axis="x", alpha=0.3)

        for bar, d in zip(bars2, max_devs, strict=False):
            ax2.text(
                bar.get_width() + 0.5,
                bar.get_y() + bar.get_height() / 2,
                f"{d:.0f}%",
                va="center",
                fontsize=9,
            )

        plt.tight_layout()
        return fig, (ax1, ax2)
    except Exception:
        return None


def _generate_action_matrix(
    rfm: pd.DataFrame,
    basket_data: dict,
    anomaly_data: dict,
    category_data: dict,
    retention_metrics: dict,
) -> list:
    """Generate prioritized action items from all analyses.

    Priority rules (explicit, not static):
    1. High-value inactive: count > 0
    2. At-risk high-value: monetary at risk > 0
    3. New customer onboarding: new_customer_share > 20%
    4. Cross-sell: high-lift pairs exist (lift > 2.0)
    5. Anomalies: any anomalies detected
    6. Champions: count > 0
    """
    actions = []

    try:
        # Column name handling for both old and new canonical formats
        seg_col = "segment_rfm" if "segment_rfm" in rfm.columns else "segment"
        mon_col = "txn_monetary" if "txn_monetary" in rfm.columns else "monetary"
        id_col = "customer_id"

        # 1. High-value inactive customers
        if retention_metrics and retention_metrics.get("inactive_valuable_count", 0) > 0:
            inactive_ids = []  # Would need customer IDs from retention computation
            actions.append(
                {
                    "priority": 1,
                    "category": "Retention",
                    "title": f"Win-back {retention_metrics['inactive_valuable_count']} high-value inactive customers",
                    "impact": fmt_currency(retention_metrics["inactive_valuable_revenue"]),
                    "impact_type": "historical_revenue_at_risk",
                    "evidence": f"Top 20% by historical revenue not active in current period (inactivity window: {retention_metrics.get('inactivity_window_days', 90)} days)",
                    "action": "Personal outreach with tailored offers; test against holdout group",
                    "metric_to_track": "Reactivation rate, revenue recovered",
                    "affected_entities": inactive_ids,
                    "affected_count": retention_metrics["inactive_valuable_count"],
                    "priority_rule": "inactive_valuable_count > 0",
                }
            )

        # 2. At-risk high-value RFM segment
        if seg_col in rfm.columns and mon_col in rfm.columns:
            at_risk = rfm[rfm[seg_col].isin(["At Risk High Value", "At Risk"])]
            if len(at_risk) > 0:
                at_risk_ids = at_risk[id_col].tolist() if id_col in at_risk.columns else []
                actions.append(
                    {
                        "priority": 2,
                        "category": "Retention",
                        "title": f"Reactivate {len(at_risk)} at-risk customers",
                        "impact": fmt_currency(at_risk[mon_col].sum()),
                        "impact_type": "historical_revenue_at_risk",
                        "evidence": f"RFM segments 'At Risk High Value' and 'At Risk' with total historical revenue {fmt_currency(at_risk[mon_col].sum())}",
                        "action": "Targeted win-back campaigns; personalized product recommendations",
                        "metric_to_track": "Reactivation rate, revenue per reactivated customer",
                        "affected_entities": at_risk_ids,
                        "affected_count": len(at_risk),
                        "priority_rule": "at_risk_monetary_sum > 0",
                    }
                )

        # 3. New customer onboarding
        if retention_metrics and retention_metrics.get("new_customer_share", 0) > 0:
            new_cust_rev = retention_metrics.get("new_customer_share", 0) * 100
            if new_cust_rev > 20:  # High new customer share
                actions.append(
                    {
                        "priority": 3,
                        "category": "Acquisition",
                        "title": f"Optimize onboarding for {new_cust_rev:.0f}% new customers",
                        "impact": "High (lifetime value)",
                        "impact_type": "future_lifetime_value",
                        "evidence": f"New customers represent {new_cust_rev:.0f}% of active base",
                        "action": "Welcome series, second-purchase incentive, category discovery",
                        "metric_to_track": "Second purchase rate within 30/60/90 days",
                        "affected_entities": [],
                        "affected_count": 0,
                        "priority_rule": "new_customer_share > 20%",
                    }
                )

        # 4. Cross-sell opportunities from basket analysis
        if basket_data and "pairs" in basket_data and not basket_data["pairs"].empty:
            pairs = basket_data["pairs"]
            high_lift = pairs[pairs["lift"] > 2.0].head(5)
            if len(high_lift) > 0:
                actions.append(
                    {
                        "priority": 4,
                        "category": "Cross-sell",
                        "title": f"Test {len(high_lift)} high-lift product bundles",
                        "impact": "Medium-High (AOV uplift)",
                        "impact_type": "aov_uplift_potential",
                        "evidence": f"Pairs with lift > 2.0 indicate strong association (e.g., {high_lift.iloc[0]['product_a']} + {high_lift.iloc[0]['product_b']}: lift {high_lift.iloc[0]['lift']:.1f})",
                        "action": "A/B test bundle offers or placement; measure attachment rate lift",
                        "metric_to_track": "Bundle attachment rate, AOV, units per order",
                        "affected_entities": high_lift[
                            ["product_a", "product_b", "lift", "joint_revenue"]
                        ].to_dict("records"),
                        "affected_count": len(high_lift),
                        "priority_rule": "high_lift_pairs_count > 0",
                    }
                )

        # 5. Category opportunities (from category growth vs share)
        if (
            category_data is not None
            and not category_data.empty
            and "growth" in category_data.columns
        ):
            # Emerging stars: high growth, low share
            emerging = category_data[
                (category_data["growth"] > 0)
                & (category_data["revenue_share"] < category_data["revenue_share"].median())
            ]
            # Scale drivers: high growth, high share
            scale_drivers = category_data[
                (category_data["growth"] > 0)
                & (category_data["revenue_share"] >= category_data["revenue_share"].median())
            ]
            # Declining priorities: negative growth, high share
            declining = category_data[
                (category_data["growth"] < 0)
                & (category_data["revenue_share"] >= category_data["revenue_share"].median())
            ]

            cat_actions = []
            if not emerging.empty:
                cat_actions.append(
                    f"Emerging stars: {', '.join(emerging['category'].head(3).tolist())}"
                )
            if not scale_drivers.empty:
                cat_actions.append(
                    f"Scale drivers: {', '.join(scale_drivers['category'].head(3).tolist())}"
                )
            if not declining.empty:
                cat_actions.append(
                    f"Declining priorities: {', '.join(declining['category'].head(3).tolist())}"
                )

            if cat_actions:
                actions.append(
                    {
                        "priority": 5,
                        "category": "Merchandising",
                        "title": f"Review {len(cat_actions)} category opportunities",
                        "impact": "Medium (mix optimization)",
                        "impact_type": "revenue_mix_improvement",
                        "evidence": "; ".join(cat_actions),
                        "action": "Invest in emerging stars; protect scale drivers; diagnose declining priorities",
                        "metric_to_track": "Category revenue growth, share of wallet",
                        "affected_entities": category_data[
                            ["category", "growth", "revenue_share", "revenue_change"]
                        ].to_dict("records"),
                        "affected_count": len(category_data),
                        "priority_rule": "category_opportunities_count > 0",
                    }
                )

        # 6. Anomaly-driven investigations
        if anomaly_data and anomaly_data.get("anomalies"):
            total_anoms = sum(len(v) for v in anomaly_data["anomalies"].values())
            if total_anoms > 0:
                # Collect affected dates per metric
                anomaly_details = []
                for metric, anoms in anomaly_data["anomalies"].items():
                    for a in anoms:
                        anomaly_details.append(
                            {
                                "metric": metric,
                                "date": str(a["date"]),
                                "pct_deviation": a["pct_deviation"],
                                "direction": a["direction"],
                            }
                        )
                actions.append(
                    {
                        "priority": 6,
                        "category": "Diagnostics",
                        "title": f"Investigate {total_anoms} anomalous KPI signals",
                        "impact": "Variable",
                        "impact_type": "risk_mitigation",
                        "evidence": f"Anomalies detected in: {', '.join(anomaly_data['anomalies'].keys())}",
                        "action": "Drill into affected segments/categories/dates; identify root cause",
                        "metric_to_track": "Return to expected range",
                        "affected_entities": anomaly_details,
                        "affected_count": total_anoms,
                        "priority_rule": "total_anomalies > 0",
                    }
                )

        # 7. Champions - protect and grow
        if seg_col in rfm.columns and mon_col in rfm.columns:
            champions = rfm[rfm[seg_col] == "Champions"]
            if len(champions) > 0:
                champion_ids = champions[id_col].tolist() if id_col in champions.columns else []
                actions.append(
                    {
                        "priority": 7,
                        "category": "Growth",
                        "title": f"Deepen {len(champions)} Champions relationships",
                        "impact": fmt_currency(champions[mon_col].sum()),
                        "impact_type": "future_revenue_potential",
                        "evidence": f"Champions = high recency, frequency, monetary (total historical revenue: {fmt_currency(champions[mon_col].sum())})",
                        "action": "VIP program, early access, referral incentives, cross-category expansion",
                        "metric_to_track": "Revenue per champion, category breadth, referral rate",
                        "affected_entities": champion_ids,
                        "affected_count": len(champions),
                        "priority_rule": "champions_count > 0",
                    }
                )

        # Sort by priority (explicit rule-based)
        actions.sort(key=lambda x: x["priority"])
        return actions
    except Exception:
        return []


def _render_action_matrix(actions: list) -> None:
    """Render action matrix as prioritized cards with affected entities and export."""
    if not actions:
        st.info("No actions generated. Run analyses to populate.")
        return

    st.subheader("Prioritized Action Matrix")
    st.caption(
        "Actions ranked by explicit priority rules. Each card shows affected entities (downloadable)."
    )

    for action in actions:
        priority_colors = {1: "🔴", 2: "🟠", 3: "🟡", 4: "🔵", 5: "🟣", 6: "🟢"}
        icon = priority_colors.get(action["priority"], "⚪")

        affected = action.get("affected_entities", [])
        affected_count = action.get(
            "affected_count", len(affected) if isinstance(affected, list) else 0
        )

        with st.expander(
            f"{icon} **Priority {action['priority']}** | {action['category']} | {action['title']} ({affected_count} affected)",
            expanded=action["priority"] <= 2,
        ):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"**Evidence:** {action['evidence']}")
                st.markdown(f"**Recommended Action:** {action['action']}")
                st.markdown(f"**Metric to Track:** {action['metric_to_track']}")
                st.markdown(f"**Priority Rule:** `{action.get('priority_rule', 'N/A')}`")
                st.markdown(f"**Impact Type:** {action.get('impact_type', 'N/A')}")

            with col2:
                st.metric("Est. Impact", action["impact"])
                st.metric("Affected", f"{affected_count:,}")
                st.markdown(f"*Priority: {action['priority']}*")

            # Affected entities table + export
            if affected and isinstance(affected, list) and len(affected) > 0:
                st.markdown("---")
                st.markdown(f"**Affected Entities ({len(affected):,})**")
                if isinstance(affected[0], dict):
                    # Structured data (pairs, anomalies)
                    df_affected = pd.DataFrame(affected)
                else:
                    # Simple list (customer IDs)
                    df_affected = pd.DataFrame({"entity_id": affected})

                st.dataframe(df_affected.head(100), use_container_width=True)
                if len(df_affected) > 100:
                    st.caption(f"Showing first 100 of {len(df_affected)}")

                csv = df_affected.to_csv(index=False).encode("utf-8")
                st.download_button(
                    f"Download all {len(df_affected)} entities (CSV)",
                    csv,
                    file_name=f"action_{action['priority']}_{action['category'].lower()}_entities.csv",
                    mime="text/csv",
                    key=f"dl_action_{action['priority']}_{action['category']}",
                )


def _plot_drilldown_path(
    raw_df: pd.DataFrame, period_params: dict, dimension: str, metric: str, top_n: int = 10
) -> tuple | None:
    """Drill-down bar chart: show metric by dimension for current vs prior."""
    try:
        ctx = get_period_context(period_params)

        curr_df = get_period_transactions(raw_df, ctx, "current")
        prior_df = get_period_transactions(raw_df, ctx, "prior")

        if not ensure_not_empty(curr_df, f"drilldown {metric} current") or not ensure_not_empty(
            prior_df, f"drilldown {metric} prior"
        ):
            return None

        if metric == "revenue":
            curr_agg = curr_df.groupby(dimension)["revenue"].sum().reset_index()
            prior_agg = prior_df.groupby(dimension)["revenue"].sum().reset_index()
        elif metric == "orders":
            curr_agg = curr_df.groupby(dimension)["transaction_id"].nunique().reset_index()
            curr_agg.columns = [dimension, "revenue"]
            prior_agg = prior_df.groupby(dimension)["transaction_id"].nunique().reset_index()
            prior_agg.columns = [dimension, "revenue"]
        elif metric == "customers":
            curr_agg = curr_df.groupby(dimension)["customer_id"].nunique().reset_index()
            curr_agg.columns = [dimension, "revenue"]
            prior_agg = prior_df.groupby(dimension)["customer_id"].nunique().reset_index()
            prior_agg.columns = [dimension, "revenue"]
        else:
            return None

        merged = curr_agg.merge(
            prior_agg, on=dimension, how="outer", suffixes=("_curr", "_prior")
        ).fillna(0)
        merged["change"] = merged["revenue_curr"] - merged["revenue_prior"]
        merged["pct_change"] = (
            merged["change"] / merged["revenue_prior"].replace(0, np.nan) * 100
        ).round(1)

        # Top N by absolute change
        merged = (
            merged.nlargest(top_n, "change", keep="all")
            if merged["change"].sum() > 0
            else merged.nsmallest(top_n, "change", keep="all")
        )

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(merged) + 1)))

        labels = [
            f"{str(row[dimension])[:25]} ({row['pct_change']:+.1f}%)"
            for _, row in merged.iterrows()
        ]
        values = merged["change"].values

        colors = ["#2f855a" if v >= 0 else "#e53e3e" for v in values]
        bars = ax.barh(range(len(labels)), values, color=colors, edgecolor="white", height=0.6)

        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel(f"{metric.title()} Change", fontsize=10)
        ax.set_title(
            f"{dimension.title()} Drill-Down: {metric.title()} Change (Current vs Prior)",
            fontsize=13,
            fontweight="bold",
            pad=15,
        )
        ax.axvline(x=0, color="gray", linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", alpha=0.3)
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(
                lambda x, _: fmt_currency(x) if metric == "revenue" else fmt_number(x)
            )
        )

        for bar, val in zip(bars, values, strict=False):
            ax.text(
                val + (max(values) - min(values)) * 0.01
                if val >= 0
                else val - (max(values) - min(values)) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                fmt_currency(val) if metric == "revenue" else fmt_number(val),
                va="center",
                ha="left" if val >= 0 else "right",
                fontsize=9,
                fontweight="bold",
            )

        plt.tight_layout()
        return fig, ax
    except Exception:
        return None


def tab_cohort(
    R: dict[str, Any], raw_df: pd.DataFrame | None = None, period_params: dict | None = None
) -> None:
    """Cohort Retention tab with heatmap, retention metrics, and reactivation analysis."""
    st.header("7 Cohort Retention")
    st.caption(
        "Question answered here: do newer cohorts retain better? Which cohorts are most valuable? What's the repeat purchase rate?"
    )

    # Use pre-computed cohort data from analysis
    cohort_data = table(R, "cohort_retention_new_customers")

    if cohort_data.empty:
        st.info("No cohort retention data available. Run analysis with sufficient history.")
        return

    # === Cohort Retention Heatmap ===
    st.subheader("Cohort Retention Heatmap")
    fig, ax = plot_cohort_retention_heatmap(cohort_data)
    chart_card(
        fig,
        f"Tracking {len(cohort_data['cohort_month'].unique())} cohorts. Darker green = higher retention.",
    )

    # === Retention & Loyalty Metrics ===
    if raw_df is not None and period_params:
        st.markdown("---")
        st.subheader("Retention & Loyalty Metrics")
        # Use 90-day inactivity window (configurable)
        inactivity_window = 90
        retention_metrics = _compute_retention_metrics(
            raw_df, period_params, inactivity_window_days=inactivity_window
        )
        if retention_metrics:
            fig, ax = _plot_retention_metrics(retention_metrics)
            if fig:
                chart_card(
                    fig,
                    f"Retention: {retention_metrics['retention_rate'] * 100:.1f}% | Reactivation: {retention_metrics['reactivation_rate'] * 100:.1f}% | Repeat Purchase Rate: {retention_metrics['repeat_purchase_rate'] * 100:.1f}%",
                )

            # Show inactivity window definition
            st.caption(
                f"Reactivation uses {inactivity_window}-day inactivity window before period start (dormant = no purchase since {retention_metrics.get('dormancy_cutoff', 'N/A')})."
            )

            # Key metrics in columns
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("Retention Rate", f"{retention_metrics['retention_rate'] * 100:.1f}%")
            col2.metric("Reactivation Rate", f"{retention_metrics['reactivation_rate'] * 100:.1f}%")
            col3.metric(
                "Repeat Purchase Rate", f"{retention_metrics['repeat_purchase_rate'] * 100:.1f}%"
            )
            col4.metric(
                "Repeat Revenue Share", f"{retention_metrics['repeat_revenue_share'] * 100:.1f}%"
            )
            col5.metric(
                "New Customer Share", f"{retention_metrics['new_customer_share'] * 100:.1f}%"
            )
            col6.metric("Inactivity Window", f"{inactivity_window} days")

            # Inactive valuable customers
            if retention_metrics.get("inactive_valuable_count", 0) > 0:
                st.warning(
                    f"⚠️ **{retention_metrics['inactive_valuable_count']} high-value customers (top 20%) inactive** — Historical revenue associated: {fmt_currency(retention_metrics['inactive_valuable_revenue'])}"
                )

    # Cohort size and revenue
    st.markdown("---")
    st.subheader("Cohort Size & Revenue")

    cf = table(R, "customer_features")
    if not cf.empty and "observed_first_purchase_cohort_month" in cf.columns:
        cohort_sizes = (
            cf.groupby("observed_first_purchase_cohort_month")
            .agg(
                customers=("customer_id", "count"),
                revenue=("net_spend", "sum")
                if "net_spend" in cf.columns
                else ("customer_id", "count"),
            )
            .reset_index()
        )
        cohort_sizes.columns = ["cohort_month", "customers", "revenue"]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Cohort Sizes**")
            show_df(pretty(cohort_sizes))
        with col2:
            st.markdown("**Cohort Revenue**")
            show_df(pretty(cohort_sizes.sort_values("revenue", ascending=False)))

    # Retention table
    st.markdown("---")
    st.subheader("Retention Table (Complete Periods Only)")
    show_df(
        pretty(
            cohort_data[cohort_data["period_complete"]].sort_values(
                ["cohort_month", "months_since_cohort"]
            )
        )
    )

    # Key insights
    st.markdown("---")
    st.subheader("Key Insights")

    # Average month-1 retention
    m1_data = cohort_data[
        (cohort_data["months_since_cohort"] == 1) & (cohort_data["period_complete"])
    ]
    if len(m1_data):
        avg_m1 = m1_data["retention_rate"].mean()
        st.markdown(f"- **Average Month-1 Retention:** {avg_m1 * 100:.1f}%")

        # Trend
        if len(m1_data) > 2:
            recent = m1_data.tail(3)["retention_rate"].mean()
            earlier = m1_data.head(3)["retention_rate"].mean()
            trend = "improving" if recent > earlier else "declining"
            st.markdown(
                f"- **Month-1 Retention Trend:** {trend} (recent 3 cohorts: {recent * 100:.1f}% vs first 3: {earlier * 100:.1f}%)"
            )

    # Best/Worst cohorts
    if len(cohort_data):
        complete = cohort_data[cohort_data["period_complete"]]
        if len(complete):
            best = complete.loc[complete["retention_rate"].idxmax()]
            worst = complete.loc[complete["retention_rate"].idxmin()]
            st.markdown(
                f"- **Best Cohort Month:** {best['cohort_month']} at M{best['months_since_cohort']} = {best['retention_rate'] * 100:.1f}%"
            )
            st.markdown(
                f"- **Worst Cohort Month:** {worst['cohort_month']} at M{worst['months_since_cohort']} = {worst['retention_rate'] * 100:.1f}%"
            )


def tab_product_pareto(
    R: dict[str, Any], raw_df: pd.DataFrame | None = None, period_params: dict | None = None
) -> None:
    """Product Pareto Analysis tab with quadrant scatter, rank changes, and trends."""
    st.header("Products & Baskets")
    st.caption(
        "Question answered here: which products drive revenue? Which are declining? What's the portfolio structure?"
    )

    # Product summary from analysis (pipeline output)
    prod_summary = table(R, "product_summary")
    cat_summary = table(R, "category_summary")

    # Build canonical product/category features from raw transactions
    if raw_df is not None and period_params:
        with st.spinner("Building product & category features..."):
            prod_features = build_product_features(raw_df, period_params, "current")
            cat_features = build_category_features(raw_df, period_params, "current")
            basket_data = build_basket_associations(raw_df, period_params)
    else:
        prod_features = pd.DataFrame()
        cat_features = pd.DataFrame()
        basket_data = {"product_metrics": pd.DataFrame(), "pairs": pd.DataFrame()}

    # Use pipeline summary for full-history Pareto (not period-limited)
    if prod_summary.empty and prod_features.empty:
        st.info("No product data available.")
        return

    # Product Pareto (full history from pipeline)
    if not prod_summary.empty:
        label_col = (
            "product_description" if "product_description" in prod_summary.columns else "product_id"
        )
        if label_col not in prod_summary.columns:
            prod_summary[label_col] = prod_summary["product_id"]

        st.subheader("Product Revenue Pareto (Full History)")
        fig, _ = plot_pareto(
            prod_summary,
            "gross_purchase_revenue",
            label_col,
            top_n=20,
            title="Top 20 Products by Revenue",
        )
        chart_card(
            fig,
            f"Top 20 products account for {prod_summary.nlargest(20, 'gross_purchase_revenue')['gross_purchase_revenue'].sum() / prod_summary['gross_purchase_revenue'].sum() * 100:.1f}% of revenue.",
            scope="Full observed history · Gross purchase revenue (positive qty only)",
        )

    # Category Pareto (full history from pipeline)
    if not cat_summary.empty:
        st.markdown("---")
        st.subheader("Category Revenue Pareto (Full History)")
        fig, _ = plot_pareto(
            cat_summary,
            "gross_purchase_revenue",
            "category",
            top_n=15,
            title="Top 15 Categories by Revenue",
        )
        chart_card(fig, "Top categories drive the majority of revenue.")

    # Period-based product features (current period)
    if not prod_features.empty:
        # Product Portfolio Quadrant: Penetration vs Revenue per Order
        st.markdown("---")
        st.subheader("Product Portfolio: Penetration vs Revenue per Order (Current Period)")
        quad_fig = _plot_product_quadrant_from_features(prod_features)
        if quad_fig:
            chart_card(
                quad_fig[0],
                "X=Basket Penetration, Y=Revenue per Order. Top-left=Niche High-Value, Top-right=Broad-Reach High-Value, Bottom-left=Low-Impact, Bottom-right=High-Volume Low-Value.",
            )

        # Product Movers - period over period (using pre-computed features)
        if "revenue_change" in prod_features.columns:
            st.markdown("---")
            st.subheader("Product Movers (Current vs Prior Period)")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Top Gainers**")
                gainers = prod_features.nlargest(10, "revenue_change")[
                    [
                        "product_description",
                        "revenue",
                        "prior_revenue",
                        "revenue_change",
                        "pct_change",
                    ]
                ]
                show_df(pretty(gainers))

            with col2:
                st.markdown("**Top Decliners**")
                decliners = prod_features.nsmallest(10, "revenue_change")[
                    [
                        "product_description",
                        "revenue",
                        "prior_revenue",
                        "revenue_change",
                        "pct_change",
                    ]
                ]
                show_df(pretty(decliners))

            # Rank change chart
            st.markdown("---")
            st.subheader("Product Rank Change")
            rank_fig = _plot_product_rank_change_from_features(prod_features)
            if rank_fig:
                chart_card(
                    rank_fig[0],
                    "Lines connect prior rank to current rank. Green = improved rank, Red = declined rank.",
                    scope="Current vs prior period · Product-level revenue rank",
                )

            # Top product trends
            st.markdown("---")
            st.subheader("Top Product Revenue Trends")
            trend_fig = _plot_top_product_trends_from_features(
                raw_df, period_params, prod_features, top_n=8
            )
            if trend_fig:
                chart_card(
                    trend_fig[0],
                    "Revenue trends for top products over last ~6 months. Yellow highlight = current period.",
                    scope="Last ~6 months · Top products by current revenue · Positive purchases",
                )

    # Category Performance: Growth vs Share (using pre-computed features)
    if not cat_features.empty and "growth" in cat_features.columns:
        st.markdown("---")
        st.subheader("Category Performance: Growth vs Share (Current Period)")
        cat_fig = _plot_category_contribution_from_features(cat_features)
        if cat_fig:
            chart_card(
                cat_fig[0],
                "X=Revenue Growth %, Y=Revenue Share %. Top-right=Scale Drivers, Top-left=Emerging Stars, Bottom-right=Declining Priorities, Bottom-left=Small Declines. Vertical line=Total Business Growth.",
                scope="Current vs prior period · Category-level · Positive purchases",
            )

        st.markdown("---")
        st.subheader("Category Revenue Mix Over Time")
        mix_fig = _plot_category_mix_trend(raw_df, period_params)
        if mix_fig:
            chart_card(
                mix_fig[0],
                "Stacked area shows how category mix evolves. Yellow/blue highlights = current/prior period.",
                scope="Last ~6 months · Top 6 categories + Other · Current/prior period highlights",
            )

    # Basket Analysis (current period)
    if raw_df is not None and period_params:
        st.markdown("---")
        st.subheader("Product Cross-sell (Selected Period)")
        pairs = basket_data.get("pairs", pd.DataFrame())

        if pairs.empty:
            st.info("No product pairs meet the current support threshold in the selected period.")
        else:
            params = basket_data.get("parameters", {})
            st.caption(
                f"{len(pairs):,} product pairs found with min support {params.get('min_support', 0.01):.1%}."
            )

            # Top pairs by lift
            top_pairs_fig = _plot_basket_top_pairs(pairs, top_n=15)
            if top_pairs_fig:
                chart_card(
                    top_pairs_fig[0],
                    "Top product pairs by lift. Lift > 1 = positive association.",
                    scope=f"Selected period · Product-level · Min support: {params.get('min_support', 0.01):.1%} · Deduplicated per invoice",
                )

            # Heatmap
            heatmap_fig = _plot_basket_heatmap(pairs, top_n=20)
            if heatmap_fig:
                chart_card(
                    heatmap_fig[0],
                    "Lift matrix for top products. Red = positive association, Blue = negative.",
                    scope="Selected period · Top products by lift sum · NaN = not computed (not independence)",
                )

            # Network graph
            network_fig = _plot_basket_network(pairs, top_n=30)
            if network_fig:
                chart_card(
                    network_fig[0],
                    "Product association network. Edge width = lift, Node size = connections.",
                    scope="Selected period · Top products by lift · Deduplicated per invoice",
                )

    # Detailed tables
    st.markdown("---")
    for stem, title in (
        ("product_summary", "Product Summary"),
        ("category_summary", "Category Summary"),
        ("department_summary", "Department Summary"),
    ):
        t = table(R, stem)
        if len(t):
            with st.expander(title):
                show_df(pretty(t.head(500)))


def landing() -> None:
    st.title("Customer Insight Lab")
    st.markdown(
        "Turn line-item purchase data into a validated picture of your customers, using only the data you already have."
    )
    c1, c2, c3 = st.columns(3)
    c1.markdown("**1. Choose data**\n\nUpload a CSV or Parquet file, or try the synthetic demo.")
    c2.markdown(
        "**2. Run analysis**\n\nData checks, probabilistic customer models, time-based validation, stable segments, basket tests."
    )
    c3.markdown(
        "**3. Read in order**\n\nOverview, trust, customers, behaviour, products, explorer, downloads."
    )
    with st.expander("Required columns"):
        show_df(
            pd.DataFrame(
                {
                    "column": REQUIRED,
                    "meaning": [
                        "customer key (blank = anonymous)",
                        "purchase date",
                        "receipt or order id",
                        "product key",
                        "product name",
                        "department",
                        "category",
                        "unit price",
                        "units; negative = return",
                    ],
                }
            )
        )


def main() -> None:
    st.set_page_config(page_title="Customer Insight Lab", page_icon=None, layout="wide")
    try:
        import insight_engine  # noqa: F401
        import retail_customer_analysis  # noqa: F401
    except ImportError as exc:
        st.error(
            f"Missing module: {exc}. Keep app.py, retail_customer_analysis.py and insight_engine.py in the same folder."
        )
        st.stop()
        return

    # Initialize session state for raw data
    if "raw_df" not in st.session_state:
        st.session_state["raw_df"] = None

    # Sidebar with raw data for period selector
    p = sidebar(st.session_state["raw_df"])
    if "result" not in st.session_state:
        landing()
    else:
        st.title("Customer Insight Lab")
    df, msg = data_step(p)
    ready = df is not None or msg == "synthetic"
    if msg and msg != "synthetic":
        st.info(msg)
    run = st.sidebar.button("Run analysis", type="primary", disabled=not ready)
    if run:
        try:
            if p["source"] == "Synthetic demo data":
                with st.spinner("Generating synthetic data..."):
                    df = cached_synthetic(
                        p["n_customers"],
                        p["days"],
                        p["syn_seed"],
                        p["missing_id"],
                        p["return_rate"],
                        p["promo_share"],
                    )
            with st.status("Running analysis...", expanded=True) as status:
                R = run_pipeline(df, p, progress=lambda m: st.write(m))
                status.update(label=f"Done in {R['seconds']} seconds", state="complete")
            old = st.session_state.get("result")
            if old and old.get("work") != R["work"]:
                shutil.rmtree(old["work"], ignore_errors=True)
            st.session_state["result"] = R
            st.session_state["source_label"] = p["source"]
            # Store normalized raw data for period selection and analytics
            st.session_state["raw_df"] = prepare_transaction_frame(df)
        except Exception as exc:
            st.error(f"{type(exc).__name__}: {exc}")
    R = st.session_state.get("result")
    if not R:
        return

    # Get period parameters from sidebar
    period_params = {}
    if "current_start" in p:
        period_params = {
            "current_start": p["current_start"],
            "current_end": p["current_end"],
            "compare_mode": p["compare_mode"],
        }

    st.caption(
        f"Showing results for: {st.session_state.get('source_label', 'data')}  |  {R['rows']:,} rows"
    )

    # Wrapper functions for consolidated tabs
    def tab_customers_and_rfm(R, raw_df, period_params):
        """Combined Customers & RFM tab."""
        tab_customers(R, raw_df, period_params)
        st.markdown("---")
        tab_rfm(R, raw_df, period_params)

    def tab_retention_and_behaviour(R, raw_df, period_params):
        """Combined Retention & Behaviour tab."""
        tab_behaviour(R, raw_df, period_params)
        st.markdown("---")
        tab_cohort(R, raw_df, period_params)

    def tab_products_and_baskets(R, raw_df, period_params):
        """Combined Products & Baskets tab."""
        tab_product_pareto(R, raw_df, period_params)
        st.markdown("---")
        tab_products(R, raw_df, period_params)

    def tab_explorer_and_data(R, raw_df, period_params):
        """Combined Explorer & Data tab."""
        tab_explorer(R, raw_df, period_params)
        st.markdown("---")
        tab_data(R, raw_df, period_params)

    # Prepare raw data for tabs
    raw_df = st.session_state.get("raw_df")

    # Updated tabs with new sections (6 tabs)
    tabs = st.tabs(TABS)
    tab_functions = [
        tab_overview,
        tab_trust,
        tab_customers_and_rfm,
        tab_retention_and_behaviour,
        tab_products_and_baskets,
        tab_explorer_and_data,
    ]
    for tab, fn in zip(tabs, tab_functions, strict=False):
        with tab:
            fn(R, raw_df, period_params)


if __name__ == "__main__":
    # Handle CLI validation flag
    if "--validate" in sys.argv:
        _run_validation_cli()
    else:
        main()
