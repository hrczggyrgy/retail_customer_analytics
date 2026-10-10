#!/usr/bin/env python3
"""Retail customer analytics: lightweight, validated, probabilistic pipeline.

Purpose
-------
Single-file command-line pipeline for line-item retail data. It keeps the
auditable accounting, data-quality and descriptive layers of the earlier Polars
version and adds probabilistic customer models with out-of-time validation.
Dependencies: pandas, NumPy, SciPy only (no scikit-learn, PyMC or lifetimes).

Analytical layers
-----------------
1. Data quality and accounting: typed conversion, issue counts, optional cleaning
   flags, identifier-consistency checks, accounting invariants (gross - returns
   = net, and every roll-up reproduces the totals).
2. Returns are linked to the latest earlier purchase of the same customer and
   product; the unmatched share is reported.
3. Purchase events: by default one customer-day "trip" (split receipts merged);
   --frequency-grain transaction restores one event per transaction key.
4. Descriptive customer features: RFM, rolling windows (--windows-days), recent
   vs prior change, breadth (distinct products/categories/departments), top
   category/department and concentration, cadence statistics, price-index promo
   proxy, return ratios.
5. BG/NBD (purchase counts) and Gamma-Gamma (trip value) by maximum likelihood:
   P(alive), expected trips and expected revenue over --horizon-days. The legacy
   cadence inactivity flag is kept as a baseline column only.
6. Time-based holdout: fit up to (as_of - horizon), score the next horizon,
   against naive baselines and the legacy flag (MAE, RMSE, Spearman, AUC,
   decile calibration).
7. Segmentation on window-free behavioural features with silhouette and
   bootstrap-stability (adjusted Rand) selection of k.
8. Left-censoring aware survival: Kaplan-Meier time to second trip for customers
   new after a burn-in window versus customers already active at file start;
   cohort retention for new customers only.
9. Basket affinity with inference: customer-level sampling and tests,
   Benjamini-Hochberg FDR, Woolf odds-ratio limits, split-half persistence,
   plus next-trip transitions.

Input
-----
Required columns: customer_id, transaction_date, transaction_id, product_id,
product_description, department, category, price (unit price), quantity
(negative = return). CSV or Parquet (Parquet needs pyarrow).

Example
-------
python retail_customer_analysis.py --input transactions.csv \
    --output-dir retail_output --horizon-days 90 --windows-days 30,90,365

Key limitations
---------------
- Revenue is not profit: no cost, margin, discount or tax fields exist, and
  expected revenue is undiscounted.
- Dates describe the observed file only. BG/NBD conditions on each customer's
  first observed trip (valid under left-censoring), but first-observed cohorts
  are not true acquisition cohorts.
- Price index and promo share are proxies; the data contain no promotion field,
  so uplift or causal claims are not supported.
- Basket rules and transitions are associations, not causal effects.
- Customers without customer_id are excluded from customer-level results.
- Segments are descriptive; judge them by stability and holdout behaviour.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import platform
import sys
from collections import Counter
from collections.abc import Sequence
from datetime import date, datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

# Use shared ingestion module
sys.path.insert(0, str(Path(__file__).parent / "src"))
from retail_customer_analytics.ingestion import (
    clean_lines,
    load_lines,
    structural_checks,
)

LOGGER = logging.getLogger("retail_analysis")
RFM_LABELS = [
    ("Champions", lambda r, f, m: (r >= 4) & (f >= 4) & (m >= 4)),
    ("Frequent / loyal", lambda r, f, m: (r >= 3) & (f >= 4)),
    ("Previously high-value / lapsed", lambda r, f, m: (r <= 2) & (f >= 4) & (m >= 4)),
    ("At risk by recency", lambda r, f, m: (r <= 2) & (f >= 3)),
    ("Recent / low frequency", lambda r, f, m: (r >= 4) & (f <= 2)),
    ("Hibernating / low frequency", lambda r, f, m: (r <= 2) & (f <= 2)),
    ("Potential loyalists", lambda r, f, m: (r >= 3) & (f >= 3)),
]


# --------------------------------------------------------------------------- #
# Arguments and logging
# --------------------------------------------------------------------------- #
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "level": record.levelname,
                "message": record.getMessage(),
            },
            ensure_ascii=False,
        )


def configure_logging(level: str) -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper()))


def parse_positive_int_list(value: str) -> list[int]:
    try:
        vals = [int(v.strip()) for v in value.split(",") if v.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not vals or any(v <= 0 for v in vals):
        raise argparse.ArgumentTypeError("all window values must be positive")
    return sorted(set(vals))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Lightweight validated retail customer analytics.")
    p.add_argument("--input", required=True, help="Input .csv, .parquet or .pq file")
    p.add_argument("--output-dir", default="retail_analysis_output")
    p.add_argument(
        "--format",
        choices=["csv", "parquet"],
        default="csv",
        help="Table format (parquet needs pyarrow; falls back to csv)",
    )
    p.add_argument("--date-format", default=None, help="Optional strptime format tried first")
    p.add_argument(
        "--as-of-date", default=None, help="Analysis cutoff YYYY-MM-DD (default: max date)"
    )
    p.add_argument("--horizon-days", type=int, default=90, help="Forecast and holdout horizon")
    p.add_argument(
        "--burn-in-days",
        type=int,
        default=90,
        help="Customers whose first trip falls in this initial window are 'existing at start'",
    )
    p.add_argument(
        "--windows-days",
        type=parse_positive_int_list,
        default=[30, 90, 365],
        help="Rolling windows",
    )
    p.add_argument(
        "--recent-window-days", type=int, default=90, help="Recent vs prior equal-length window"
    )
    p.add_argument(
        "--transaction-key-mode",
        choices=["customer-transaction", "transaction-only"],
        default="customer-transaction",
        help="Is transaction_id unique within customer_id or globally?",
    )
    p.add_argument(
        "--frequency-grain",
        choices=["trip", "transaction"],
        default="trip",
        help="trip = one customer-day (merges split receipts); transaction = one transaction key",
    )
    p.add_argument("--drop-exact-duplicates", action="store_true")
    p.add_argument("--exclude-zero-quantity", action="store_true")
    p.add_argument("--exclude-negative-prices", action="store_true")
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument(
        "--penalizer",
        type=float,
        default=1e-3,
        help="Ridge penalty on log-parameters of the models",
    )
    p.add_argument(
        "--inactivity-min-days",
        type=float,
        default=30.0,
        help="Legacy heuristic (validation baseline)",
    )
    p.add_argument("--inactivity-cadence-multiplier", type=float, default=2.0)
    p.add_argument("--inactivity-fallback-days", type=float, default=90.0)
    p.add_argument(
        "--promo-threshold",
        type=float,
        default=0.10,
        help="Price-index discount flagging a promo-like line",
    )
    p.add_argument(
        "--min-price-observations",
        type=int,
        default=5,
        help="Lines per product for a reference price",
    )
    p.add_argument("--max-clr-categories", type=int, default=30)
    p.add_argument("--min-clusters", type=int, default=3)
    p.add_argument("--max-clusters", type=int, default=7)
    p.add_argument("--cluster-max-customers", type=int, default=20000)
    p.add_argument("--stability-bootstraps", type=int, default=15)
    p.add_argument("--stability-threshold", type=float, default=0.70)
    p.add_argument("--basket-level", choices=["category", "product"], default="category")
    p.add_argument("--min-support", type=float, default=0.01)
    p.add_argument("--min-confidence", type=float, default=0.20)
    p.add_argument("--min-lift", type=float, default=1.0)
    p.add_argument("--fdr-alpha", type=float, default=0.05)
    p.add_argument("--max-affinity-items", type=int, default=100)
    p.add_argument("--max-affinity-baskets", type=int, default=50000)
    p.add_argument("--max-items-per-basket", type=int, default=30)
    p.add_argument("--max-affinity-pair-operations", type=int, default=1_000_000)
    p.add_argument(
        "--rolling-origins",
        type=int,
        default=1,
        help="Number of rolling-origin validation cutoffs (1 = single holdout)",
    )
    p.add_argument(
        "--origin-spacing-days",
        type=int,
        default=None,
        help="Days between rolling origins (default: horizon-days)",
    )
    p.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    return p


def validate_args(a: argparse.Namespace, p: argparse.ArgumentParser) -> None:
    for name in (
        "horizon_days",
        "burn_in_days",
        "recent_window_days",
        "max_clusters",
        "min_clusters",
        "cluster_max_customers",
        "stability_bootstraps",
        "max_affinity_items",
        "max_affinity_baskets",
        "max_items_per_basket",
        "max_affinity_pair_operations",
        "max_clr_categories",
    ):
        if getattr(a, name) <= 0:
            p.error(f"--{name.replace('_', '-')} must be positive")
    if a.min_clusters < 2 or a.min_clusters > a.max_clusters:
        p.error("need 2 <= --min-clusters <= --max-clusters")
    for name in ("min_support", "min_confidence", "fdr_alpha", "stability_threshold"):
        if not 0.0 <= getattr(a, name) <= 1.0:
            p.error(f"--{name.replace('_', '-')} must be in [0, 1]")
    if not 0.0 < a.promo_threshold < 1.0:
        p.error("--promo-threshold must be in (0, 1)")
    if a.as_of_date:
        try:
            datetime.strptime(a.as_of_date, "%Y-%m-%d")
        except ValueError:
            p.error("--as-of-date must use YYYY-MM-DD")


# --------------------------------------------------------------------------- #
# Purchase events, returns, price index
# --------------------------------------------------------------------------- #
def build_trips(lines: pd.DataFrame, grain: str = "trip") -> pd.DataFrame:
    """Purchase events per customer. grain='trip': one row per customer-day (split receipts merged);
    grain='transaction': one row per (customer_id, transaction_id), dated at its earliest line."""
    cols = ["customer_id", "transaction_day", "trip_value", "n_lines", "n_products", "n_units"]
    if grain == "transaction":
        cols = [
            "customer_id",
            "transaction_id",
            "transaction_day",
            "trip_value",
            "n_lines",
            "n_products",
            "n_units",
        ]
    buys = lines[
        (lines["quantity"] > 0) & lines["customer_id"].notna() & lines["transaction_day"].notna()
    ]
    if grain == "transaction":
        buys = buys[buys["transaction_id"].notna()]
    if buys.empty:
        return pd.DataFrame(columns=cols)
    buys = buys.assign(rev_pos=buys["line_revenue"].clip(lower=0).fillna(0.0))
    keys = (
        ["customer_id", "transaction_day"] if grain == "trip" else ["customer_id", "transaction_id"]
    )
    spec = {
        "trip_value": ("rev_pos", "sum"),
        "n_lines": ("quantity", "size"),
        "n_products": ("product_id", "nunique"),
        "n_units": ("quantity", "sum"),
    }
    if grain == "transaction":
        spec["transaction_day"] = ("transaction_day", "min")
    trips = buys.groupby(keys, sort=False).agg(**spec).reset_index()
    return trips[trips["trip_value"] > 0][cols].reset_index(drop=True)


def match_returns(lines: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Link each return line to the latest earlier purchase of the same customer and product.

    Uses full transaction_date timestamps when available for precise ordering.
    Falls back to transaction_day (calendar day) when timestamps are not available.
    """
    # Determine which timestamp column to use for matching
    has_timestamps = "transaction_date" in lines.columns and lines["transaction_date"].notna().any()
    time_col = "transaction_date" if has_timestamps else "transaction_day"

    ret = lines[
        (lines["quantity"] < 0)
        & lines["customer_id"].notna()
        & lines["product_id"].notna()
        & lines[time_col].notna()
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
    buys = (
        lines[
            (lines["quantity"] > 0)
            & lines["customer_id"].notna()
            & lines["product_id"].notna()
            & lines[time_col].notna()
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
            # Return difference in days with fractional precision
            info["median_days_to_return"] = float(
                delta.dt.total_seconds()[merged["matched"]].median() / 86400
            )
        else:
            info["median_days_to_return"] = float(delta.dt.days[merged["matched"]].median())
    return merged, info


def add_price_index(
    lines: pd.DataFrame, args: argparse.Namespace
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """price_index = unit price / product median price over positive-quantity lines."""
    pos = (lines["quantity"] > 0) & (lines["price"] > 0) & lines["product_id"].notna()
    grp = lines[pos].groupby("product_id")["price"]
    ref, cnt, nun = grp.median(), grp.size(), grp.nunique()
    ref = ref[cnt >= args.min_price_observations]
    lines = lines.copy()
    lines["reference_price"] = lines["product_id"].map(ref)
    lines["price_index"] = np.where(pos, lines["price"] / lines["reference_price"], np.nan)
    lines["promo_like"] = (lines["price_index"] <= 1.0 - args.promo_threshold) & lines[
        "price_index"
    ].notna()
    eligible = int(len(ref))
    varying = int((nun.reindex(ref.index) > 1).sum()) if eligible else 0
    info = {
        "products_with_reference_price": eligible,
        "products_with_price_variation": varying,
        "promo_like_line_share": float(lines.loc[lines["price_index"].notna(), "promo_like"].mean())
        if eligible
        else None,
        "usable": bool(eligible and varying / max(eligible, 1) >= 0.05),
    }
    return lines, info


# --------------------------------------------------------------------------- #
# Customer table at a cutoff
# --------------------------------------------------------------------------- #
def customer_table(
    trips: pd.DataFrame, as_of: pd.Timestamp, args: argparse.Namespace
) -> pd.DataFrame:
    """Per-customer BG/NBD inputs plus cadence statistics, using trips up to `as_of` only."""
    t = (
        trips[trips["transaction_day"] <= as_of]
        .sort_values(["customer_id", "transaction_day"])
        .copy()
    )
    if t.empty:
        return pd.DataFrame()
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
    c["interpurchase_cv"] = c["std_gap_days"] / c["mean_gap_days"].where(c["mean_gap_days"] > 0)
    med = c["median_gap_days"]
    thr = np.where(
        med > 0,
        np.maximum(args.inactivity_min_days, med * args.inactivity_cadence_multiplier),
        args.inactivity_fallback_days,
    )
    c["heuristic_threshold_days"] = thr
    c["heuristic_inactive_flag"] = c["recency_days"] > thr
    var = c["std_gap_days"] ** 2
    k_hat = (c["mean_gap_days"] ** 2 / var).where((c["n_gaps"] >= 3) & (var > 0))
    c["regularity_shape_raw"] = k_hat
    c["regularity_shape_shrunk"] = ((c["n_gaps"] * k_hat.fillna(1.0)) + 3.0) / (c["n_gaps"] + 3.0)
    return c


def add_returns_and_promo(
    c: pd.DataFrame,
    lines: pd.DataFrame,
    matched_returns: pd.DataFrame,
    as_of: pd.Timestamp,
    price_usable: bool,
) -> pd.DataFrame:
    c = c.copy()
    ret = matched_returns[matched_returns["transaction_day"] <= as_of]
    c["return_value"], c["unmatched_return_value"] = 0.0, 0.0
    if len(ret):
        rv = (-ret["line_revenue"]).clip(lower=0).fillna(0.0)
        c["return_value"] = rv.groupby(ret["customer_id"]).sum().reindex(c.index).fillna(0.0)
        um = ~ret["matched"]
        c["unmatched_return_value"] = (
            rv[um].groupby(ret.loc[um, "customer_id"]).sum().reindex(c.index).fillna(0.0)
        )
    c["net_spend"] = (c["gross_spend"] - c["return_value"]).clip(lower=0)
    c["return_value_ratio"] = (c["return_value"] / c["gross_spend"]).clip(0, 1)
    c["promo_spend_share"], c["mean_price_index"] = np.nan, np.nan
    if price_usable:
        pl_ = lines[
            (lines["quantity"] > 0)
            & lines["customer_id"].notna()
            & lines["price_index"].notna()
            & (lines["transaction_day"] <= as_of)
        ]
        rev = pl_["line_revenue"].clip(lower=0).fillna(0.0)
        tot = rev.groupby(pl_["customer_id"]).sum().reindex(c.index)
        promo = (
            rev[pl_["promo_like"]]
            .groupby(pl_.loc[pl_["promo_like"], "customer_id"])
            .sum()
            .reindex(c.index)
            .fillna(0.0)
        )
        wi = (pl_["price_index"] * rev).groupby(pl_["customer_id"]).sum().reindex(c.index)
        c["promo_spend_share"] = (promo / tot).where(tot > 0)
        c["mean_price_index"] = (wi / tot).where(tot > 0)
    return c


def category_mix(
    lines: pd.DataFrame, customers: pd.Index, as_of: pd.Timestamp, max_cat: int
) -> tuple[np.ndarray, list[str]]:
    """Centred log-ratio (CLR) transform of category spend shares per customer."""
    pos = lines[
        (lines["quantity"] > 0)
        & lines["customer_id"].notna()
        & lines["category"].notna()
        & (lines["transaction_day"] <= as_of)
    ]
    pos = pos.assign(rev=pos["line_revenue"].clip(lower=0).fillna(0.0))
    pos = pos[pos["customer_id"].isin(customers)]
    if pos.empty:
        return np.zeros((len(customers), 0)), []
    top = pos.groupby("category")["rev"].sum().sort_values(ascending=False).head(max_cat).index
    cat = pos["category"].where(pos["category"].isin(top), "__other__")
    spend = pos.groupby([pos["customer_id"], cat])["rev"].sum().reset_index()
    cats = sorted(spend["category"].unique())
    ci, ji = {k: i for i, k in enumerate(customers)}, {k: i for i, k in enumerate(cats)}
    m = np.zeros((len(customers), len(cats)))
    np.add.at(
        m,
        (spend["customer_id"].map(ci).to_numpy(), spend["category"].map(ji).to_numpy()),
        spend["rev"].to_numpy(),
    )
    tot = m.sum(axis=1, keepdims=True)
    share = np.divide(m, tot, out=np.zeros_like(m), where=tot > 0)
    logs = np.log(share + 1.0 / (10 * max(len(cats), 1)))
    return logs - logs.mean(axis=1, keepdims=True), cats


def top_components(clr: np.ndarray, k: int = 3) -> np.ndarray:
    if clr.shape[1] < 2 or clr.shape[0] < 5:
        return np.zeros((clr.shape[0], 0))
    u, s, _ = np.linalg.svd(clr - clr.mean(axis=0), full_matrices=False)
    k = min(k, len(s))
    return u[:, :k] * s[:k]


# --------------------------------------------------------------------------- #
# BG/NBD (Fader, Hardie & Lee 2005)
# --------------------------------------------------------------------------- #
def _bgnbd_loglik(params: np.ndarray, x: np.ndarray, tx: np.ndarray, T: np.ndarray) -> np.ndarray:
    r, alpha, a, b = np.exp(params)
    ln_a1 = special.gammaln(r + x) - special.gammaln(r) + r * np.log(alpha)
    ln_a2 = (
        special.gammaln(a + b)
        + special.gammaln(b + x)
        - special.gammaln(b)
        - special.gammaln(a + b + x)
    )
    ln_a3 = -(r + x) * np.log(alpha + T)
    with np.errstate(divide="ignore", invalid="ignore"):
        ln_a4 = np.where(
            x > 0, np.log(a) - np.log(b + x - 1.0) - (r + x) * np.log(alpha + tx), -np.inf
        )
    return ln_a1 + ln_a2 + np.logaddexp(ln_a3, ln_a4)


def fit_bgnbd(x: np.ndarray, tx: np.ndarray, T: np.ndarray, penalizer: float) -> dict[str, Any]:
    """Maximum likelihood on unique (x, t_x, T) rows with weights; mild ridge on log-parameters."""
    rows, w = np.unique(np.column_stack([x, tx, T]), axis=0, return_counts=True)
    xs, txs, Ts = rows[:, 0], rows[:, 1], rows[:, 2]
    scale = max(float(np.median(T[T > 0])) if np.any(T > 0) else 30.0, 1.0)

    def nll(p: np.ndarray) -> float:
        val = -(w * _bgnbd_loglik(p, xs, txs, Ts)).sum() + penalizer * w.sum() * float(np.sum(p**2))
        return float(val) if np.isfinite(val) else 1e12

    best = None
    for start in (np.log([0.5, scale / 4, 1.0, 1.0]), np.log([1.0, scale, 0.8, 2.5])):
        res = optimize.minimize(nll, start, method="L-BFGS-B", bounds=[(-7, 7)] * 4)
        if best is None or res.fun < best.fun:
            best = res
    r, alpha, a, b = np.exp(best.x)
    return {
        "status": "fitted",
        "r": float(r),
        "alpha": float(alpha),
        "a": float(a),
        "b": float(b),
        "converged": bool(best.success),
        "neg_loglik": float(best.fun),
        "n_customers": int(len(x)),
    }


def bgnbd_predict(
    par: dict[str, Any], x: np.ndarray, tx: np.ndarray, T: np.ndarray, horizon: float
):
    """P(alive) and E[purchases in (T, T+horizon]] given history."""
    r, alpha, a, b = par["r"], par["alpha"], par["a"], par["b"]
    if abs(a - 1.0) < 1e-4:
        a = 1.0 + 1e-4
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        log_ratio = (r + x) * (np.log(alpha + T) - np.log(alpha + tx))
        term = np.where(x > 0, a / (b + x - 1.0) * np.exp(log_ratio), 0.0)
        p_alive = 1.0 / (1.0 + term)
        c = a + b + x - 1.0
        z = horizon / (alpha + T + horizon)
        hyp = special.hyp2f1(r + x, b + x, c, z)
        core = (
            (a + b + x - 1.0)
            / (a - 1.0)
            * (1.0 - np.exp((r + x) * (np.log(alpha + T) - np.log(alpha + T + horizon))) * hyp)
        )
        expected = core / (1.0 + term)
    expected = np.where(~np.isfinite(expected) | (c <= 0) | (expected < 0), np.nan, expected)
    return p_alive, expected


# --------------------------------------------------------------------------- #
# Gamma-Gamma (Fader & Hardie 2013)
# --------------------------------------------------------------------------- #
def fit_gamma_gamma(n: np.ndarray, mbar: np.ndarray, penalizer: float) -> dict[str, Any]:
    ok = (n >= 2) & (mbar > 0) & np.isfinite(mbar)
    n, m = n[ok].astype(float), mbar[ok].astype(float)
    if len(n) < 30:
        return {"status": "skipped", "reason": f"only {len(n)} customers with >=2 trips (need 30)"}

    def nll(p: np.ndarray) -> float:
        pp, q, g = np.exp(p)
        ll = (
            special.gammaln(pp * n + q)
            - special.gammaln(pp * n)
            - special.gammaln(q)
            + q * np.log(g)
            + (pp * n - 1) * np.log(m)
            + pp * n * np.log(n)
            - (pp * n + q) * np.log(g + n * m)
        )
        val = -ll.sum() + penalizer * len(n) * float(np.sum(p**2))
        return float(val) if np.isfinite(val) else 1e12

    scale = float(np.median(m))
    best = None
    for start in (np.log([2.0, 3.0, 2.0 * scale]), np.log([1.0, 2.0, scale])):
        res = optimize.minimize(
            nll,
            start,
            method="L-BFGS-B",
            bounds=[(-5, 5), (np.log(1.05), 6), (-5, np.log(scale) + 8)],
        )
        if best is None or res.fun < best.fun:
            best = res
    p, q, g = np.exp(best.x)
    rho = float(stats.spearmanr(n, m)[0]) if len(n) > 2 and np.ptp(n) > 0 else float("nan")
    return {
        "status": "fitted",
        "p": float(p),
        "q": float(q),
        "gamma": float(g),
        "converged": bool(best.success),
        "n_customers": int(len(n)),
        "spearman_trips_vs_mean_value": rho,
        "independence_warning": bool(np.isfinite(rho) and abs(rho) > 0.3),
    }


def gamma_gamma_predict(par: dict[str, Any], n: np.ndarray, mbar: np.ndarray) -> np.ndarray:
    """E[M | n, mbar] = p (gamma + n*mbar) / (p*n + q - 1): shrinks the observed mean toward the population mean."""
    if par.get("status") != "fitted":
        return mbar.astype(float)
    return par["p"] * (par["gamma"] + n * mbar) / (par["p"] * n + par["q"] - 1.0)


def score_customers(
    c: pd.DataFrame, horizon: float, args: argparse.Namespace
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Fit both models and attach P(alive), expected trips, expected trip value and horizon revenue.

    Returns customer DataFrame with prediction_source column indicating:
    - 'bgnbd_fitted': BG/NBD fitted and prediction used
    - 'bgnbd_fallback': BG/NBD failed, used empirical rate fallback
    - 'gg_fitted': Gamma-Gamma fitted
    - 'gg_fallback': Gamma-Gamma failed/skipped, used observed mean
    """
    c = c.copy()
    x, tx, T = c["x"].to_numpy(float), c["t_x"].to_numpy(float), c["T"].to_numpy(float)

    # BG/NBD fitting
    bgnbd_status = "fitted"
    bgnbd_converged = False
    try:
        if len(c) < 50:
            raise ValueError(f"only {len(c)} customers; BG/NBD needs at least 50")
        bg = fit_bgnbd(x, tx, T, args.penalizer)
        if not bg.get("converged", False):
            bgnbd_converged = False
            LOGGER.warning("BG/NBD optimizer did not converge; treating as fallback")
            raise ValueError("BG/NBD did not converge")
        bgnbd_converged = True
        p_alive, exp_trips = bgnbd_predict(bg, x, tx, T, float(horizon))
    except Exception as exc:
        LOGGER.warning("BG/NBD not fitted (%s); using empirical rates", exc)
        bg = {"status": "failed", "reason": str(exc), "converged": False}
        bgnbd_status = "failed"
        p_alive, exp_trips = np.full(len(c), np.nan), np.full(len(c), np.nan)

    # Determine prediction source for each customer
    c["p_alive"] = p_alive
    bgnbd_finite = np.isfinite(exp_trips)
    c["expected_trips_horizon"] = np.where(
        bgnbd_finite, exp_trips, x / np.maximum(T, 1.0) * horizon
    )
    c["prediction_source_bgnbd"] = np.where(bgnbd_finite, "bgnbd_fitted", "bgnbd_fallback")

    # Gamma-Gamma fitting
    n, m = c["n_trips"].to_numpy(float), c["mean_trip_value"].to_numpy(float)
    gg = fit_gamma_gamma(n, m, args.penalizer)

    if gg.get("status") == "fitted" and gg.get("converged", False):
        gg_status = "fitted"
        gg_converged = True
    else:
        gg_status = "fallback"
        gg_converged = False
        if gg.get("status") == "skipped":
            gg_status = "skipped"

    c["expected_trip_value"] = gamma_gamma_predict(gg, n, m)
    c["expected_revenue_horizon"] = c["expected_trips_horizon"] * c["expected_trip_value"]
    c["prediction_source_gg"] = np.where(
        np.isfinite(c["expected_trip_value"]) & gg_converged, "gg_fitted", "gg_fallback"
    )

    # Add model status summary
    bg["status"] = bgnbd_status
    bg["converged"] = bgnbd_converged
    gg["status"] = gg_status
    gg["converged"] = gg_converged

    return c, bg, gg


# --------------------------------------------------------------------------- #
# Time-based validation
# --------------------------------------------------------------------------- #
def auc_score(score: np.ndarray, label: np.ndarray) -> float | None:
    label = label.astype(bool)
    n1, n0 = int(label.sum()), int((~label).sum())
    if n1 == 0 or n0 == 0:
        return None
    ranks = stats.rankdata(score)
    return float((ranks[label].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def _spearman(pred: np.ndarray, y: np.ndarray) -> float | None:
    if len(pred) < 3 or np.ptp(pred) <= 0 or np.ptp(y) <= 0:
        return None
    rho = stats.spearmanr(pred, y)[0]
    return float(rho) if np.isfinite(rho) else None


def _population_mean_baseline(trips: pd.DataFrame, cut: pd.Timestamp, H: int) -> float:
    """
    Estimate population-mean trip count from calibration data ONLY.
    Uses historical windows of length H before the cutoff to compute
    the average repeat trips per customer, avoiding any holdout leakage.
    """
    # Look at complete historical windows of length H before the cutoff
    # We use the same logic as customer_table to compute x/T for each customer
    # and estimate the population mean rate from calibration period data
    cal_trips = trips[trips["transaction_day"] <= cut]
    if cal_trips.empty:
        return 0.0

    # For each customer, count trips in the H-day window before cutoff
    window_start = cut - pd.Timedelta(days=H)
    window_trips = cal_trips[
        (cal_trips["transaction_day"] > window_start) & (cal_trips["transaction_day"] <= cut)
    ]

    if window_trips.empty:
        return 0.0

    # Mean trips per customer in this historical window
    customer_trip_counts = window_trips.groupby("customer_id").size()
    # Include customers with zero trips in the window (they're in cal_trips but not window_trips)
    all_cal_customers = cal_trips["customer_id"].unique()
    full_counts = customer_trip_counts.reindex(all_cal_customers, fill_value=0)

    return float(full_counts.mean())


def _compute_metrics(
    name: str,
    target: str,
    pred: np.ndarray,
    y: np.ndarray,
    bought: np.ndarray | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Compute comprehensive metrics for a model prediction."""
    ok = np.isfinite(pred)
    if not ok.any():
        return {
            "model": name,
            "target": target,
            "n_customers": 0,
            "mae": None,
            "rmse": None,
            "wape": None,
            "aggregate_bias": None,
            "spearman": None,
            "auc_any_purchase": None,
            "predicted_total": None,
            "actual_total": None,
            "mae_ci95_lower": None,
            "mae_ci95_upper": None,
            "bootstrap_seed": seed,
        }

    p = pred[ok]
    a = y[ok]
    n = len(p)

    # Basic metrics
    mae = float(np.mean(np.abs(p - a)))
    rmse = float(np.sqrt(np.mean((p - a) ** 2)))

    # WAPE (Weighted Absolute Percentage Error)
    denom = np.sum(np.abs(a))
    wape = float(np.sum(np.abs(p - a)) / denom) if denom > 0 else None

    # Aggregate bias
    sum_a = float(np.sum(a))
    sum_p = float(np.sum(p))
    aggregate_bias = float((sum_p - sum_a) / sum_a) if sum_a > 0 else None

    # Spearman correlation
    spearman = _spearman(p, a)

    # AUC for binary target
    auc = None
    if bought is not None:
        auc = auc_score(p, bought[ok])

    # Bootstrap confidence intervals for MAE (simple percentile)
    mae_ci_lower, mae_ci_upper = None, None
    if n >= 10:
        try:
            n_boot = min(200, n * 2)
            rng = np.random.default_rng(seed)
            boot_maes = []
            for _ in range(n_boot):
                idx = rng.choice(n, n, replace=True)
                boot_maes.append(np.mean(np.abs(p[idx] - a[idx])))
            mae_ci_lower = float(np.percentile(boot_maes, 2.5))
            mae_ci_upper = float(np.percentile(boot_maes, 97.5))
        except Exception:
            pass

    return {
        "model": name,
        "target": target,
        "n_customers": int(n),
        "mae": mae,
        "rmse": rmse,
        "wape": wape,
        "aggregate_bias": aggregate_bias,
        "spearman": spearman,
        "auc_any_purchase": auc,
        "predicted_total": float(p.sum()),
        "actual_total": float(a.sum()),
        "mae_ci95_lower": mae_ci_lower,
        "mae_ci95_upper": mae_ci_upper,
        "bootstrap_seed": seed,
    }


def holdout_validation(
    trips: pd.DataFrame, as_of: pd.Timestamp, args: argparse.Namespace
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Fit on data up to as_of - horizon; score against the following horizon."""
    H = int(args.horizon_days)
    cut = as_of - pd.Timedelta(days=H)
    info: dict[str, Any] = {
        "status": "skipped",
        "reason": None,
        "cutoff_date": cut.date().isoformat(),
        "horizon_days": H,
    }
    empty = pd.DataFrame()
    if trips.empty or trips["transaction_day"].min() > cut - pd.Timedelta(days=H):
        info["reason"] = "history shorter than two horizons; calibration period too short"
        return empty, empty, info
    cal = customer_table(trips, cut, args)
    if len(cal) < 100:
        info["reason"] = f"only {len(cal)} calibration customers (need 100)"
        return empty, empty, info
    scored, bg, gg = score_customers(cal, H, args)
    hold = trips[(trips["transaction_day"] > cut) & (trips["transaction_day"] <= as_of)]
    act = hold.groupby("customer_id").agg(
        actual_trips=("transaction_day", "size"), actual_spend=("trip_value", "sum")
    )
    d = scored.join(act, how="left").fillna({"actual_trips": 0, "actual_spend": 0.0})
    d["pred_empirical_rate"] = d["x"] / np.maximum(d["T"], 1.0) * H
    # FIX: Use calibration-period population mean, not holdout mean (prevents leakage)
    d["pred_constant"] = _population_mean_baseline(trips, cut, H)
    d["pred_revenue_naive"] = d["pred_empirical_rate"] * d["mean_trip_value"]
    y, bought = d["actual_trips"].to_numpy(float), d["actual_trips"].to_numpy(float) > 0

    rows = []
    # Use random seed for reproducible bootstrap CIs
    metrics_seed = args.random_seed
    for name, col in (
        ("BG/NBD", "expected_trips_horizon"),
        ("empirical_rate_baseline", "pred_empirical_rate"),
        ("population_mean_baseline", "pred_constant"),
    ):
        pred = d[col].to_numpy(float)
        rows.append(_compute_metrics(name, "trips_in_horizon", pred, y, bought, seed=metrics_seed))

    # Binary target: any purchase in horizon (0/1)
    # Use bought (boolean) as actual, not y (count)
    rows.append(
        _compute_metrics(
            "legacy_cadence_heuristic",
            "any_purchase_in_horizon",
            (~d["heuristic_inactive_flag"]).to_numpy(float),
            bought.astype(float),
            bought,
            seed=metrics_seed,
        )
    )

    pa = d["p_alive"].to_numpy(float)
    ok = np.isfinite(pa)
    rows.append(
        _compute_metrics(
            "BG/NBD P(alive)",
            "any_purchase_in_horizon",
            pa[ok],
            bought[ok].astype(float),
            bought[ok],
            seed=metrics_seed,
        )
    )

    s = d["actual_spend"].to_numpy(float)
    for name, col in (
        ("BG/NBD x Gamma-Gamma", "expected_revenue_horizon"),
        ("naive_rate_x_mean_value", "pred_revenue_naive"),
    ):
        pred = d[col].to_numpy(float)
        rows.append(_compute_metrics(name, "revenue_in_horizon", pred, s, seed=metrics_seed))

    dd = d[d["expected_trips_horizon"].notna()].copy()
    dd["decile"] = (
        pd.qcut(
            dd["expected_trips_horizon"].rank(method="first"), 10, labels=False, duplicates="drop"
        )
        + 1
    )
    calib = (
        dd.groupby("decile")
        .agg(
            customers=("actual_trips", "size"),
            mean_predicted_trips=("expected_trips_horizon", "mean"),
            mean_actual_trips=("actual_trips", "mean"),
            share_bought=("actual_trips", lambda v: float((v > 0).mean())),
            mean_p_alive=("p_alive", "mean"),
        )
        .reset_index()
    )
    info.update(
        {
            "status": "completed",
            "calibration_customers": int(len(d)),
            "bgnbd": bg,
            "gamma_gamma": gg,
        }
    )
    return pd.DataFrame(rows), calib, info


def rolling_origin_validation(
    trips: pd.DataFrame,
    as_of: pd.Timestamp,
    args: argparse.Namespace,
    n_origins: int = 4,
    origin_spacing_days: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Run holdout validation at multiple historical cutoff dates (rolling origins).

    This provides more robust evidence of model generalization by evaluating
    at multiple points in time rather than a single holdout.

    Args:
        trips: Purchase event DataFrame
        as_of: Final analysis date (latest cutoff)
        args: CLI arguments
        n_origins: Number of historical origins to evaluate (including final)
        origin_spacing_days: Days between origins (default: horizon_days)

    Returns:
        Tuple of (aggregated_metrics, per_origin_details, summary_info)
    """
    H = int(args.horizon_days)
    spacing = origin_spacing_days if origin_spacing_days is not None else H

    # Generate cutoff dates: as_of, as_of - spacing, as_of - 2*spacing, ...
    cutoffs = []
    for i in range(n_origins):
        cut = as_of - pd.Timedelta(days=i * spacing)
        # Need enough history before cut: cut - H must be >= min date
        if trips.empty:
            break
        min_date = trips["transaction_day"].min()
        if pd.isna(min_date) or cut - pd.Timedelta(days=H) <= min_date:
            continue
        # Need enough data after cut for evaluation
        if cut >= as_of:
            continue
        cutoffs.append(cut)

    if len(cutoffs) < 2:
        # Not enough data for rolling origins, fall back to single holdout
        return holdout_validation(trips, as_of, args)

    cutoffs = sorted(cutoffs)  # Oldest first
    all_metrics = []
    all_calibs = []
    origin_infos = []

    for cut in cutoffs:
        # Create modified args with this cutoff as the "as_of" for validation
        eval_horizon = min(H, int((as_of - cut).days))
        if eval_horizon < H * 0.5:  # Skip if evaluation window too small
            continue

        class ModifiedArgs(argparse.Namespace):
            def __init__(self, base_args, horizon_days):
                super().__init__(**vars(base_args))
                self.horizon_days = horizon_days

        mod_args = ModifiedArgs(args, eval_horizon)

        metrics, calib, info = holdout_validation(trips, cut, mod_args)
        if info.get("status") == "completed":
            # Add origin identifier
            metrics["origin_cutoff"] = cut.date().isoformat()
            metrics["origin_eval_horizon"] = eval_horizon
            calib["origin_cutoff"] = cut.date().isoformat()
            info["origin_cutoff"] = cut.date().isoformat()
            info["origin_eval_horizon"] = eval_horizon
            all_metrics.append(metrics)
            all_calibs.append(calib)
            origin_infos.append(info)

    if not all_metrics:
        # Fallback
        return holdout_validation(trips, as_of, args)

    # Aggregate metrics across origins
    combined_metrics = pd.concat(all_metrics, ignore_index=True)
    combined_calib = pd.concat(all_calibs, ignore_index=True)

    # For report compatibility, return the detailed per-origin metrics
    # in the format expected by write_report (with n_customers, mae, etc.)
    detailed_metrics = combined_metrics.copy()

    # Compute aggregate statistics per model/target across origins
    agg_rows = []
    for (model, target), group in combined_metrics.groupby(["model", "target"]):
        numeric_cols = [
            "mae",
            "rmse",
            "spearman",
            "auc_any_purchase",
            "predicted_total",
            "actual_total",
            "n_customers",
        ]
        agg = {}
        for col in numeric_cols:
            if col in group.columns:
                vals = group[col].dropna()
                if len(vals) > 0:
                    agg[f"{col}_mean"] = float(vals.mean())
                    agg[f"{col}_std"] = float(vals.std()) if len(vals) > 1 else 0.0
                    agg[f"{col}_min"] = float(vals.min())
                    agg[f"{col}_max"] = float(vals.max())
                    agg["n_origins"] = len(vals)

        agg_rows.append({"model": model, "target": target, **agg})

    pd.DataFrame(agg_rows)

    # Summary info - include fields expected by report generation
    final_cutoff = cutoffs[-1] if cutoffs else (as_of - pd.Timedelta(days=H))
    summary_info = {
        "status": "completed",
        "n_origins": len(origin_infos),
        "origins": [
            {
                "cutoff": o["origin_cutoff"],
                "eval_horizon": o["origin_eval_horizon"],
                "calibration_customers": o.get("calibration_customers", 0),
            }
            for o in origin_infos
        ],
        "horizon_days": H,
        "as_of_date": as_of.date().isoformat(),
        "cutoff_date": final_cutoff.date().isoformat(),  # For report compatibility
        "final_cutoff": final_cutoff.date().isoformat(),
        "calibration_customers": int(origin_infos[-1].get("calibration_customers", 0))
        if origin_infos
        else 0,
    }

    # Return detailed per-origin metrics for report, aggregated for analysis
    return detailed_metrics, combined_calib, summary_info


# --------------------------------------------------------------------------- #
# RFM (communication layer)
# --------------------------------------------------------------------------- #
def edge_scores(values: np.ndarray, higher_is_better: bool) -> np.ndarray:
    """Quintile scores from unique quantile edges; heavy ties collapse levels instead of splitting ties arbitrarily."""
    v = np.asarray(values, float)
    edges = np.unique(np.quantile(v, [0.2, 0.4, 0.6, 0.8]))
    s = np.searchsorted(edges, v, side="left") + 1
    levels = len(edges) + 1
    s5 = np.full(len(v), 3.0) if levels == 1 else np.rint(1 + (s - 1) * 4.0 / (levels - 1))
    return (s5 if higher_is_better else 6 - s5).astype(int)


def assign_rfm(c: pd.DataFrame) -> pd.DataFrame:
    c = c.copy()
    r = edge_scores(c["recency_days"].to_numpy(), False)
    f = edge_scores(c["n_trips"].to_numpy(), True)
    m = edge_scores(c["net_spend"].to_numpy(), True)
    seg = np.full(len(c), "Mixed / needs attention", dtype=object)
    done = np.zeros(len(c), bool)
    for label, rule in RFM_LABELS:
        hit = rule(r, f, m) & ~done
        seg[hit] = label
        done |= hit
    c["rfm_recency_score"], c["rfm_frequency_score"], c["rfm_monetary_score"] = r, f, m
    c["rfm_score"] = [f"{a}{b}{d}" for a, b, d in zip(r, f, m, strict=False)]
    c["rfm_segment"] = seg
    return c


# --------------------------------------------------------------------------- #
# Clustering with stability (NumPy only)
# --------------------------------------------------------------------------- #
def _assign(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
    d = (x**2).sum(1, keepdims=True) - 2 * x @ centers.T + (centers**2).sum(1)[None, :]
    return d.argmin(1)


def kmeans(x: np.ndarray, k: int, rng: np.random.Generator, n_init: int = 3, iters: int = 100):
    best = (np.inf, None, None)
    for _ in range(n_init):
        centers = x[[rng.integers(len(x))]]
        for _ in range(1, k):  # k-means++ seeding
            d2 = np.clip(
                np.min(
                    (x**2).sum(1, keepdims=True) - 2 * x @ centers.T + (centers**2).sum(1)[None, :],
                    axis=1,
                ),
                0,
                None,
            )
            tot = d2.sum()
            centers = np.vstack(
                [centers, x[rng.choice(len(x), p=d2 / tot) if tot > 0 else rng.integers(len(x))]]
            )
        labels = _assign(x, centers)
        for _ in range(iters):
            new = np.vstack(
                [x[labels == j].mean(0) if np.any(labels == j) else centers[j] for j in range(k)]
            )
            new_labels = _assign(x, new)
            centers = new
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
        inertia = float(((x - centers[labels]) ** 2).sum())
        if inertia < best[0]:
            best = (inertia, labels, centers)
    return best[1], best[2], best[0]


def adjusted_rand(a: np.ndarray, b: np.ndarray) -> float:
    _, ia = np.unique(a, return_inverse=True)
    _, ib = np.unique(b, return_inverse=True)
    cont = np.zeros((ia.max() + 1, ib.max() + 1))
    np.add.at(cont, (ia, ib), 1)
    comb = lambda v: v * (v - 1) / 2.0  # noqa: E731
    sum_c, sa, sb, n = (
        comb(cont).sum(),
        comb(cont.sum(1)).sum(),
        comb(cont.sum(0)).sum(),
        comb(len(a)),
    )
    expected = sa * sb / n if n > 0 else 0
    maxi = 0.5 * (sa + sb)
    return float((sum_c - expected) / (maxi - expected)) if maxi != expected else 1.0


def silhouette(x: np.ndarray, labels: np.ndarray) -> float | None:
    ks = np.unique(labels)
    if len(ks) < 2:
        return None
    d = np.sqrt(np.clip((x**2).sum(1)[:, None] - 2 * x @ x.T + (x**2).sum(1)[None, :], 0, None))
    s = np.zeros(len(x))
    for i in range(len(x)):
        same = labels == labels[i]
        if same.sum() <= 1:
            continue
        a_i = d[i, same].sum() / (same.sum() - 1)
        b_i = min(d[i, labels == k].mean() for k in ks if k != labels[i])
        s[i] = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0.0
    return float(s.mean())


def robust_scale(x: np.ndarray) -> np.ndarray:
    med = np.nanmedian(x, axis=0)
    q1, q3 = np.nanpercentile(x, [25, 75], axis=0)
    scale, sd = (q3 - q1) / 1.349, np.nanstd(x, axis=0)
    scale = np.where(scale > 1e-9, scale, np.where(sd > 1e-9, sd, 1.0))
    return np.clip((x - med) / scale, -6, 6)


def cluster_customers(
    c: pd.DataFrame, pcs: np.ndarray, args: argparse.Namespace
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    c = c.copy()
    c["cluster_id"] = pd.NA
    c["cluster_label"] = "Unclustered"
    cols = {
        "log_expected_trips": np.log1p(c["expected_trips_horizon"].clip(lower=0)),
        "p_alive": c["p_alive"].fillna(
            c["p_alive"].median() if c["p_alive"].notna().any() else 0.5
        ),
        "log_expected_trip_value": np.log1p(c["expected_trip_value"].clip(lower=0)),
        "return_value_ratio": c["return_value_ratio"].fillna(0.0),
        "log_regularity_shape": np.log(c["regularity_shape_shrunk"].clip(lower=0.2, upper=20)),
    }
    if c["promo_spend_share"].notna().any():
        cols["promo_spend_share"] = c["promo_spend_share"].fillna(c["promo_spend_share"].median())
    x = np.column_stack(list(cols.values()) + ([pcs] if pcs.shape[1] else []))
    names = list(cols) + [f"category_mix_pc{i + 1}" for i in range(pcs.shape[1])]
    base = {"model": "kmeans_numpy_robust_scaled", "features_used": ",".join(names)}
    if len(c) < 50:
        return (
            c,
            pd.DataFrame(
                [
                    {
                        **base,
                        "status": "skipped",
                        "reason": "fewer than 50 customers",
                        "selected": False,
                    }
                ]
            ),
            pd.DataFrame(),
        )
    x = robust_scale(np.where(np.isfinite(x), x, np.nan))
    x = np.where(np.isfinite(x), x, 0.0)
    rng = np.random.default_rng(args.random_seed)
    xt = x[np.sort(rng.choice(len(x), min(len(x), args.cluster_max_customers), replace=False))]
    sil_idx = rng.choice(len(xt), min(len(xt), 1500), replace=False)
    stab_idx = rng.choice(len(xt), min(len(xt), 4000), replace=False)
    rows, fitted = [], {}
    for k in range(args.min_clusters, min(args.max_clusters, len(xt) - 1) + 1):
        labels, centers, inertia = kmeans(xt, k, rng)
        sizes = np.bincount(labels, minlength=k)
        ref = _assign(xt[stab_idx], centers)
        aris = []
        for _ in range(args.stability_bootstraps):
            b = rng.choice(stab_idx, len(stab_idx), replace=True)
            _, cb, _ = kmeans(xt[b], k, rng, n_init=1, iters=50)
            aris.append(adjusted_rand(ref, _assign(xt[stab_idx], cb)))
        rows.append(
            {
                **base,
                "status": "candidate",
                "candidate_k": k,
                "silhouette": silhouette(xt[sil_idx], labels[sil_idx]),
                "inertia": inertia,
                "stability_ari_mean": float(np.mean(aris)),
                "stability_ari_p05": float(np.percentile(aris, 5)),
                "smallest_cluster_share": float(sizes.min() / sizes.sum()),
                "training_customers": int(len(xt)),
            }
        )
        fitted[k] = centers
    diag = pd.DataFrame(rows)
    ok = diag[
        (diag["stability_ari_mean"] >= args.stability_threshold)
        & (diag["smallest_cluster_share"] >= 0.02)
    ]
    if len(ok):
        best_k = int(
            ok.sort_values(["silhouette", "candidate_k"], ascending=[False, True]).iloc[0][
                "candidate_k"
            ]
        )
        reason = "highest silhouette among candidates meeting stability and size thresholds"
    else:
        best_k = int(
            diag.sort_values(["stability_ari_mean", "candidate_k"], ascending=[False, True]).iloc[
                0
            ]["candidate_k"]
        )
        reason = "no candidate met the stability threshold; most stable k chosen - treat segments as exploratory"
    diag["selected"] = diag["candidate_k"] == best_k
    diag["selection_reason"] = np.where(diag["selected"], reason, "")
    c["cluster_id"] = _assign(x, fitted[best_k])
    prof = (
        c.groupby("cluster_id")
        .agg(
            customers=("n_trips", "size"),
            median_recency_days=("recency_days", "median"),
            median_p_alive=("p_alive", "median"),
            median_expected_trips=("expected_trips_horizon", "median"),
            median_expected_trip_value=("expected_trip_value", "median"),
            total_expected_revenue=("expected_revenue_horizon", "sum"),
            median_trips_observed=("n_trips", "median"),
            mean_promo_spend_share=("promo_spend_share", "mean"),
            mean_return_value_ratio=("return_value_ratio", "mean"),
        )
        .reset_index()
    )
    med_rev = float(c["expected_revenue_horizon"].median())
    labels = {}
    for _, row in prof.iterrows():
        sub = c[c["cluster_id"] == row["cluster_id"]]
        active = (
            row["median_p_alive"] >= 0.5
            if np.isfinite(row["median_p_alive"])
            else row["median_recency_days"] <= 90
        )
        high = float(sub["expected_revenue_horizon"].median()) >= med_rev
        labels[int(row["cluster_id"])] = (
            "Active / higher expected value"
            if active and high
            else "Active / lower expected value"
            if active
            else "Lapsing / higher past value"
            if high
            else "Lapsed / low expected value"
        )
    seen = Counter()
    for cid in sorted(labels):
        seen[labels[cid]] += 1
        if seen[labels[cid]] > 1:
            labels[cid] = f"{labels[cid]} (cluster {cid})"
    c["cluster_label"] = c["cluster_id"].map(labels)
    prof["cluster_label"] = prof["cluster_id"].map(labels)
    prof["share_of_customers"] = prof["customers"] / len(c)
    return c, diag, prof


def compare_rfm_vs_cluster(c: pd.DataFrame) -> pd.DataFrame:
    """
    Compare RFM segments with cluster assignments.

    Returns a crosstab showing how RFM segments map to clusters,
    plus summary statistics for each combination.
    """
    if "rfm_segment" not in c.columns or "cluster_label" not in c.columns:
        return pd.DataFrame()

    # Cross-tabulation
    crosstab = pd.crosstab(c["rfm_segment"], c["cluster_label"], margins=True)
    crosstab.columns = [str(x) for x in crosstab.columns]
    crosstab.index = [str(x) for x in crosstab.index]

    # Row/column percentages
    len(c)
    crosstab.div(crosstab["All"], axis=0) * 100
    crosstab.div(crosstab.loc["All"], axis=1) * 100

    # Build detailed comparison
    rows = []
    for rfm_seg in c["rfm_segment"].unique():
        if rfm_seg == "All" or pd.isna(rfm_seg):
            continue
        for clust in c["cluster_label"].unique():
            if clust == "All" or pd.isna(clust):
                continue
            mask = (c["rfm_segment"] == rfm_seg) & (c["cluster_label"] == clust)
            n = int(mask.sum())
            if n == 0:
                continue
            sub = c[mask]
            rows.append(
                {
                    "rfm_segment": rfm_seg,
                    "cluster_label": clust,
                    "n_customers": n,
                    "pct_of_rfm": n / (c["rfm_segment"] == rfm_seg).sum() * 100,
                    "pct_of_cluster": n / (c["cluster_label"] == clust).sum() * 100,
                    "median_recency": float(sub["recency_days"].median())
                    if "recency_days" in sub.columns
                    else np.nan,
                    "median_p_alive": float(sub["p_alive"].median())
                    if "p_alive" in sub.columns
                    else np.nan,
                    "median_expected_revenue": float(sub["expected_revenue_horizon"].median())
                    if "expected_revenue_horizon" in sub.columns
                    else np.nan,
                    "repeat_rate": float((sub["n_trips"] >= 2).mean())
                    if "n_trips" in sub.columns
                    else np.nan,
                }
            )

    return pd.DataFrame(rows)


def cluster_stability_across_snapshots(
    lines: pd.DataFrame,
    snapshots: list[pd.Timestamp],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run clustering at multiple snapshot dates and measure stability.

    Returns:
        - stability_metrics: ARI and label agreement between consecutive snapshots
        - cluster_labels_per_snapshot: cluster assignments for each snapshot
    """
    from retail_customer_analytics.features import build_customer_snapshot

    all_labels = {}
    all_profiles = {}

    for snap in snapshots:
        # Build features at this snapshot
        cust = build_customer_snapshot(
            transactions=lines,
            as_of=snap,
            event_grain=args.frequency_grain,
            windows_days=args.windows_days,
            recent_window_days=args.recent_window_days,
        )
        if len(cust) < 50:
            continue

        # Score customers
        from retail_customer_analysis import assign_rfm, score_customers

        cust, bg, gg = score_customers(cust, args.horizon_days, args)
        cust = assign_rfm(cust)

        # Cluster
        try:
            pcs = top_components(category_mix(lines, cust.index, snap, args.max_clr_categories)[0])
            cust, diag, prof = cluster_customers(cust, pcs, args)

            # Get selected k
            selected = diag[diag["selected"]]
            if len(selected) == 0:
                continue
            int(selected.iloc[0]["candidate_k"])

            # Save labels
            all_labels[snap] = cust[["cluster_id", "cluster_label"]].copy().reset_index()
            all_labels[snap]["snapshot"] = snap
            all_profiles[snap] = prof

        except Exception as e:
            LOGGER.warning(f"Clustering failed at snapshot {snap}: {e}")
            continue

    if len(all_labels) < 2:
        return pd.DataFrame(), pd.DataFrame()

    # Compare consecutive snapshots
    snapshots_sorted = sorted(all_labels.keys())
    stability_rows = []

    for i in range(1, len(snapshots_sorted)):
        snap_prev = snapshots_sorted[i - 1]
        snap_curr = snapshots_sorted[i]

        labels_prev = all_labels[snap_prev].set_index("customer_id")
        labels_curr = all_labels[snap_curr].set_index("customer_id")

        # Common customers
        common = labels_prev.index.intersection(labels_curr.index)
        if len(common) < 20:
            continue

        # Adjusted Rand Index
        ari = adjusted_rand(
            labels_prev.loc[common, "cluster_id"].to_numpy(),
            labels_curr.loc[common, "cluster_id"].to_numpy(),
        )

        # Label agreement (exact match)
        agreement = (
            labels_prev.loc[common, "cluster_id"] == labels_curr.loc[common, "cluster_id"]
        ).mean()

        # Cluster-level stability
        stability_rows.append(
            {
                "snapshot_prev": snap_prev.date().isoformat(),
                "snapshot_curr": snap_curr.date().isoformat(),
                "n_common_customers": len(common),
                "ari": float(ari),
                "label_agreement": float(agreement),
                "n_clusters_prev": labels_prev["cluster_id"].nunique(),
                "n_clusters_curr": labels_curr["cluster_id"].nunique(),
            }
        )

        # Per-cluster tracking
        for cl in sorted(labels_curr.loc[common, "cluster_id"].unique()):
            curr_mask = labels_curr.loc[common, "cluster_id"] == cl
            if not curr_mask.any():
                continue
            common_in_cl = common[curr_mask]
            if len(common_in_cl) == 0:
                continue
            prev_labels = labels_prev.loc[common_in_cl, "cluster_id"]
            # Most common previous cluster
            prev_mode = prev_labels.mode()
            stability_rows.append(
                {
                    "snapshot_prev": snap_prev.date().isoformat(),
                    "snapshot_curr": snap_curr.date().isoformat(),
                    "cluster_curr": int(cl),
                    "n_customers": int(curr_mask.sum()),
                    "prev_cluster_mode": int(prev_mode.iloc[0]) if len(prev_mode) > 0 else None,
                    "stability_pct": float((prev_labels == prev_mode.iloc[0]).mean())
                    if len(prev_mode) > 0
                    else 0.0,
                }
            )

    stability_df = pd.DataFrame(stability_rows)
    labels_df = pd.concat(all_labels.values(), ignore_index=True) if all_labels else pd.DataFrame()

    return stability_df, labels_df


# --------------------------------------------------------------------------- #
# Survival (time to second trip) and cohort retention
# --------------------------------------------------------------------------- #
def kaplan_meier(
    duration: np.ndarray, event: np.ndarray, landmarks: Sequence[int]
) -> list[dict[str, Any]]:
    d, e = np.asarray(duration, float), np.asarray(event, bool)
    if len(d) == 0:
        return []
    ev_times = np.unique(d[e])
    n_risk = np.array([(d >= t).sum() for t in ev_times])
    n_ev = np.array([((d == t) & e).sum() for t in ev_times])
    surv = np.cumprod(1.0 - n_ev / np.maximum(n_risk, 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        green = np.cumsum(np.where(n_risk > n_ev, n_ev / (n_risk * (n_risk - n_ev)), 0.0))
    out = []
    for lm in landmarks:
        if lm > d.max():
            continue
        i = np.searchsorted(ev_times, lm, side="right") - 1
        s_ = 1.0 if i < 0 else float(surv[i])
        se = 0.0 if i < 0 else float(s_ * math.sqrt(green[i]))
        out.append(
            {
                "day": int(lm),
                "survival_no_repeat": s_,
                "repeat_probability": 1 - s_,
                "repeat_ci95_lower": max(0.0, 1 - s_ - 1.96 * se),
                "repeat_ci95_upper": min(1.0, 1 - s_ + 1.96 * se),
                "at_risk": int((d >= lm).sum()),
            }
        )
    return out


def repeat_survival(
    trips: pd.DataFrame, as_of: pd.Timestamp, burn_in_days: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    t = trips[trips["transaction_day"] <= as_of].sort_values(["customer_id", "transaction_day"])
    if t.empty:
        return pd.DataFrame(), pd.DataFrame()
    start = t["transaction_day"].min()
    g = t.groupby("customer_id")["transaction_day"]
    first = g.min()
    second = g.apply(lambda s: s.iloc[1] if len(s) > 1 else pd.NaT)
    event = second.notna()
    dur = np.where(event, (second - first).dt.days, (as_of - first).dt.days).astype(float)
    is_new = first >= start + pd.Timedelta(days=burn_in_days)
    cohort_q = first.dt.to_period("Q").astype(str)
    groups = {
        "all_customers": np.ones(len(first), bool),
        "new_after_burn_in": is_new.to_numpy(),
        "existing_at_start_(left_censored)": ~is_new.to_numpy(),
    }
    for q in sorted(cohort_q[is_new].unique()):
        groups[f"new_cohort_{q}"] = ((cohort_q == q) & is_new).to_numpy()
    rows = []
    for name, mask in groups.items():
        if mask.sum() < 20:
            continue
        for row in kaplan_meier(dur[mask], event.to_numpy()[mask], [7, 14, 30, 60, 90, 180, 365]):
            rows.append({"group": name, "customers": int(mask.sum()), **row})
    info = pd.DataFrame(
        {
            "customer_id": first.index,
            "first_trip_day": first.values,
            "is_new_after_burn_in": is_new.values,
            "days_to_second_trip_or_censor": dur,
            "repeated": event.values,
        }
    )
    return pd.DataFrame(rows), info


def wilson(
    successes: int, total: int, z: float = 1.959963984540054
) -> tuple[float | None, float | None]:
    if total <= 0:
        return None, None
    p = successes / total
    den = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / den
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def cohort_retention(
    trips: pd.DataFrame, cohort_info: pd.DataFrame, as_of: pd.Timestamp
) -> pd.DataFrame:
    """Monthly activity retention for NEW customers only (existing-at-start customers are left-censored)."""
    if cohort_info.empty or not cohort_info["is_new_after_burn_in"].any():
        return pd.DataFrame()
    new = cohort_info[cohort_info["is_new_after_burn_in"]].copy()
    new["cohort_month"] = new["first_trip_day"].dt.to_period("M")
    t = trips[
        (trips["transaction_day"] <= as_of) & trips["customer_id"].isin(new["customer_id"])
    ].copy()
    t["activity_month"] = t["transaction_day"].dt.to_period("M")
    act = (
        t[["customer_id", "activity_month"]]
        .drop_duplicates()
        .merge(new[["customer_id", "cohort_month"]], on="customer_id")
    )
    act["age"] = (act["activity_month"].dt.year - act["cohort_month"].dt.year) * 12 + (
        act["activity_month"].dt.month - act["cohort_month"].dt.month
    )
    counts = act.groupby(["cohort_month", "age"])["customer_id"].nunique()
    sizes = new.groupby("cohort_month")["customer_id"].nunique()
    cur = as_of.to_period("M")
    rows = []
    for cm, size in sizes.items():
        for age in range(int((cur - cm).n) + 1):
            n_act = int(counts.get((cm, age), 0))
            lo, hi = wilson(n_act, int(size))
            rows.append(
                {
                    "cohort_month": str(cm),
                    "cohort_size": int(size),
                    "months_since_cohort": age,
                    "active_customers": n_act,
                    "retention_rate": n_act / size,
                    "ci95_lower": lo,
                    "ci95_upper": hi,
                    "period_complete": bool(
                        (cm + age).to_timestamp(how="end").normalize() <= as_of
                    ),
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Summaries, windowed/breadth features and structural tables
# --------------------------------------------------------------------------- #
def dimension_summary(lines: pd.DataFrame, by: str | Sequence[str]) -> pd.DataFrame:
    d = lines.assign(
        buy_rev=lines["line_revenue"].where(lines["quantity"] > 0),
        ret_rev=(-lines["line_revenue"]).where(lines["quantity"] < 0),
        units_bought=lines["quantity"].where(lines["quantity"] > 0),
        units_ret=(-lines["quantity"]).where(lines["quantity"] < 0),
    )
    out = (
        d.groupby(by, dropna=False)
        .agg(
            line_items=("quantity", "size"),
            gross_purchase_revenue=("buy_rev", "sum"),
            return_value=("ret_rev", "sum"),
            net_revenue=("line_revenue", "sum"),
            units_purchased=("units_bought", "sum"),
            units_returned=("units_ret", "sum"),
            distinct_customers=("customer_id", "nunique"),
            distinct_products=("product_id", "nunique"),
        )
        .reset_index()
    )
    return out.sort_values("gross_purchase_revenue", ascending=False)


def revenue_concentration(c: pd.DataFrame) -> pd.DataFrame:
    s = np.sort(c["gross_spend"].clip(lower=0).to_numpy(float))
    n, tot = len(s), float(s.sum())
    if n == 0 or tot <= 0:
        return pd.DataFrame()
    top = lambda f: float(s[-max(1, math.ceil(n * f)) :].sum() / tot)  # noqa: E731
    return pd.DataFrame(
        [
            {
                "customers": n,
                "total_gross_spend": tot,
                "top_1pct_share": top(0.01),
                "top_5pct_share": top(0.05),
                "top_10pct_share": top(0.10),
                "gini": float(2.0 * np.dot(np.arange(1, n + 1), s) / (n * tot) - (n + 1.0) / n),
                "hhi": float(((s / tot) ** 2).sum()),
            }
        ]
    )


def window_features(
    lines: pd.DataFrame,
    trips: pd.DataFrame,
    index: pd.Index,
    as_of: pd.Timestamp,
    windows: Sequence[int],
    recent_days: int,
) -> pd.DataFrame:
    """Rolling-window spend/trip features and recent-vs-prior change (inclusive calendar-day windows ending at as_of)."""
    f = pd.DataFrame(index=index)
    cl = lines[
        lines["customer_id"].notna()
        & lines["transaction_day"].notna()
        & (lines["transaction_day"] <= as_of)
    ]

    def agg(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        sub = cl[(cl["transaction_day"] >= start) & (cl["transaction_day"] <= end)]
        gross = sub["line_revenue"].where(sub["quantity"] > 0).groupby(sub["customer_id"]).sum()
        ret = (-sub["line_revenue"]).where(sub["quantity"] < 0).groupby(sub["customer_id"]).sum()
        tt = (
            trips[(trips["transaction_day"] >= start) & (trips["transaction_day"] <= end)]
            .groupby("customer_id")
            .size()
        )
        d = (
            pd.DataFrame({"gross_purchase_revenue": gross, "return_value": ret, "trips": tt})
            .reindex(index)
            .fillna(0.0)
        )
        d["net_revenue"] = d["gross_purchase_revenue"] - d["return_value"]
        return d

    for w in windows:
        d = agg(as_of - pd.Timedelta(days=w - 1), as_of)
        f[f"gross_purchase_revenue_{w}d"], f[f"return_value_{w}d"] = (
            d["gross_purchase_revenue"],
            d["return_value"],
        )
        f[f"net_revenue_{w}d"], f[f"trips_{w}d"] = d["net_revenue"], d["trips"].astype(int)
    rec = agg(as_of - pd.Timedelta(days=recent_days - 1), as_of)
    pri = agg(
        as_of - pd.Timedelta(days=2 * recent_days - 1), as_of - pd.Timedelta(days=recent_days)
    )
    f["recent_gross_purchase_revenue"], f["prior_gross_purchase_revenue"] = (
        rec["gross_purchase_revenue"],
        pri["gross_purchase_revenue"],
    )
    f["recent_trips"], f["prior_trips"] = rec["trips"].astype(int), pri["trips"].astype(int)
    f["recent_revenue_change_pct_if_prior_positive"] = (
        (rec["gross_purchase_revenue"] - pri["gross_purchase_revenue"])
        / pri["gross_purchase_revenue"]
    ).where(pri["gross_purchase_revenue"] > 0)
    f["recent_trip_change_pct_if_prior_positive"] = (
        (rec["trips"] - pri["trips"]) / pri["trips"]
    ).where(pri["trips"] > 0)
    return f


def breadth_features(lines: pd.DataFrame, index: pd.Index, as_of: pd.Timestamp) -> pd.DataFrame:
    """Distinct products/categories/departments, top category/department by spend, and their concentration (HHI)."""
    pos = lines[
        (lines["quantity"] > 0) & lines["customer_id"].notna() & (lines["transaction_day"] <= as_of)
    ]
    pos = pos.assign(rev=pos["line_revenue"].clip(lower=0).fillna(0.0))
    f = pd.DataFrame(index=index)
    f["unique_products_purchased"] = (
        pos.groupby("customer_id")["product_id"].nunique().reindex(index)
    )
    for col in ("category", "department"):
        sub = pos[pos[col].notna()]
        f[
            "unique_categories_purchased" if col == "category" else "unique_departments_purchased"
        ] = sub.groupby("customer_id")[col].nunique().reindex(index)
        if sub.empty:
            f[f"top_{col}_by_spend"], f[f"{col}_spend_hhi"] = None, np.nan
            continue
        sp = sub.groupby(["customer_id", col])["rev"].sum().reset_index()
        top = (
            sp.sort_values(["customer_id", "rev", col], ascending=[True, False, True])
            .drop_duplicates("customer_id")
            .set_index("customer_id")[col]
        )
        tot = sp.groupby("customer_id")["rev"].transform("sum")
        share = (sp["rev"] / tot).where(tot > 0)
        f[f"top_{col}_by_spend"] = top.reindex(index)
        f[f"{col}_spend_hhi"] = (
            (share**2).groupby(sp["customer_id"]).sum(min_count=1).reindex(index)
        )
    return f


def transaction_summary(lines: pd.DataFrame, mode: str) -> pd.DataFrame:
    keys = (
        ["customer_id", "transaction_id"] if mode == "customer-transaction" else ["transaction_id"]
    )
    d = lines[lines[keys].notna().all(axis=1)]
    if d.empty:
        return pd.DataFrame()
    d = d.assign(
        buy_rev=d["line_revenue"].where(d["quantity"] > 0),
        ret_rev=(-d["line_revenue"]).where(d["quantity"] < 0),
        units_bought=d["quantity"].where(d["quantity"] > 0),
    )
    out = (
        d.groupby(keys, sort=True)
        .agg(
            first_day=("transaction_day", "min"),
            distinct_dates=("transaction_day", "nunique"),
            line_items=("quantity", "size"),
            gross_purchase_revenue=("buy_rev", "sum"),
            return_value=("ret_rev", "sum"),
            net_revenue=("line_revenue", "sum"),
            units_purchased=("units_bought", "sum"),
            distinct_products=("product_id", "nunique"),
            distinct_customers=("customer_id", "nunique"),
        )
        .reset_index()
    )
    out["is_purchase_transaction"] = out["units_purchased"] > 0
    return out


def interpurchase_table(trips: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    t = (
        trips[trips["transaction_day"] <= as_of]
        .sort_values(["customer_id", "transaction_day"])[["customer_id", "transaction_day"]]
        .copy()
    )
    t["interpurchase_interval_days"] = t.groupby("customer_id")["transaction_day"].diff().dt.days
    return t


def product_tables(
    lines: pd.DataFrame, grain: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Product summary with metadata-conflict flags, conflict variants, and observed repeat-buyer behaviour."""
    d = lines[lines["product_id"].notna()]
    prod = dimension_summary(d, "product_id")
    meta = d.groupby("product_id").agg(
        description_variants=("product_description", "nunique"),
        department_variants=("department", "nunique"),
        category_variants=("category", "nunique"),
        observed_description=("product_description", "first"),
        observed_department=("department", "first"),
        observed_category=("category", "first"),
    )
    meta["has_metadata_conflict"] = (
        meta[["description_variants", "department_variants", "category_variants"]] > 1
    ).any(axis=1)
    prod = prod.merge(meta.reset_index(), on="product_id", how="left")
    variants = (
        d.groupby(["product_id", "product_description", "department", "category"], dropna=False)
        .size()
        .rename("variant_line_items")
        .reset_index()
    )
    variants = variants[variants["product_id"].isin(meta.index[meta["has_metadata_conflict"]])]
    b = lines[(lines["quantity"] > 0) & lines["customer_id"].notna() & lines["product_id"].notna()]
    unit = "transaction_day" if grain == "trip" else "transaction_id"
    cp = b[b[unit].notna()].groupby(["product_id", "customer_id"])[unit].nunique()
    if cp.empty:
        rep = pd.DataFrame(
            columns=["product_id", "repeat_buyers", "observed_buyers", "repeat_buyer_rate"]
        )
    else:
        rep = (
            (cp >= 2)
            .groupby(level="product_id")
            .agg(repeat_buyers="sum", observed_buyers="size")
            .reset_index()
        )
        rep["repeat_buyer_rate"] = rep["repeat_buyers"] / rep["observed_buyers"]
        ci = [
            wilson(int(a), int(n))
            for a, n in zip(rep["repeat_buyers"], rep["observed_buyers"], strict=False)
        ]
        rep["repeat_rate_ci95_lower"], rep["repeat_rate_ci95_upper"] = (
            [c_[0] for c_ in ci],
            [c_[1] for c_ in ci],
        )
        rep["repeat_unit"] = "customer-day trips" if grain == "trip" else "transaction ids"
        rep = rep.sort_values("observed_buyers", ascending=False)
    prod = prod.merge(
        rep[["product_id", "observed_buyers", "repeat_buyers", "repeat_buyer_rate"]],
        on="product_id",
        how="left",
    )
    return prod.sort_values("gross_purchase_revenue", ascending=False), variants, rep


def accounting_check(lines: pd.DataFrame, summaries: dict[str, pd.DataFrame]) -> dict[str, float]:
    """Invariants: gross - returns = net, and every roll-up reproduces the same totals."""
    rev, qty = lines["line_revenue"], lines["quantity"]
    gross, ret, net = (
        float(rev.where(qty > 0).sum()),
        float((-rev).where(qty < 0).sum()),
        float(rev.sum()),
    )
    zero_nonzero = bool(((qty == 0) & rev.notna() & (rev != 0)).any())
    if not zero_nonzero and not np.isclose(gross - ret, net, rtol=1e-9, atol=1e-7):
        raise AssertionError(
            "accounting invariant failed: gross purchase revenue - return value != net revenue"
        )
    for name, tbl in summaries.items():
        for col, target in (
            ("gross_purchase_revenue", gross),
            ("return_value", ret),
            ("net_revenue", net),
        ):
            if col in tbl.columns and not np.isclose(
                float(tbl[col].sum()), target, rtol=1e-8, atol=1e-6
            ):
                raise AssertionError(f"{name} roll-up invariant failed for {col}")
    return {"gross_purchase_revenue": gross, "return_value": ret, "net_revenue": net}


# --------------------------------------------------------------------------- #
# Basket affinity with inference
# --------------------------------------------------------------------------- #
def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    q = np.minimum.accumulate((p[order] * n / np.arange(1, n + 1))[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


def _hash_series(values: pd.Series, seed: int) -> np.ndarray:
    return pd.util.hash_pandas_object(values.astype(str), index=False).to_numpy(
        dtype="uint64"
    ) ^ np.uint64(seed * 2654435761 % (2**63))


def _pair_stats(
    n_ab: np.ndarray, n_a: np.ndarray, n_b: np.ndarray, n: int
) -> dict[str, np.ndarray]:
    """Lift, hypergeometric p-value and a Woolf odds-ratio CI (Haldane-corrected) from 2x2 counts."""
    a, b, c, d = n_ab, n_a - n_ab, n_b - n_ab, n - n_a - n_b + n_ab
    with np.errstate(divide="ignore", invalid="ignore"):
        lift = n_ab * n / (n_a * n_b)
        a2, b2, c2, d2 = (v + 0.5 for v in (a, b, c, d))
        log_or = np.log(a2 * d2 / (b2 * c2))
        se = np.sqrt(1 / a2 + 1 / b2 + 1 / c2 + 1 / d2)
    return {
        "lift": lift,
        "or_ci95_lower": np.exp(log_or - 1.96 * se),
        "odds_ratio": np.exp(log_or),
        "p_value": stats.hypergeom.sf(n_ab - 1, n, n_a, n_b),
    }


def basket_affinity(
    lines: pd.DataFrame, as_of: pd.Timestamp, args: argparse.Namespace
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Customer-level association tests on baskets (trip or transaction grain per --frequency-grain) plus next-trip transitions.

    When frequency_grain='trip' (default), baskets are customer-day shopping occasions.
    When frequency_grain='transaction', baskets are individual transaction_ids.
    """
    item_col = "category" if args.basket_level == "category" else "product_id"
    det: dict[str, Any] = {
        "status": "skipped",
        "reason": None,
        "basket_level": args.basket_level,
        "basket_grain": args.frequency_grain,
    }
    base = lines[
        (lines["quantity"] > 0)
        & lines["customer_id"].notna()
        & lines["transaction_day"].notna()
        & lines[item_col].notna()
        & (lines["transaction_day"] <= as_of)
    ]
    # Apply frequency grain: trip = customer-day, transaction = (customer_id, transaction_id)
    if args.frequency_grain == "trip":
        base = base[["customer_id", "transaction_day", item_col]].drop_duplicates()
        basket_key = ["customer_id", "transaction_day"]
    else:  # transaction grain
        base = base[base["transaction_id"].notna()][
            ["customer_id", "transaction_id", "transaction_day", item_col]
        ].drop_duplicates()
        basket_key = ["customer_id", "transaction_id"]

    if base.empty:
        det["reason"] = "no identifiable positive-purchase baskets"
        return pd.DataFrame(), pd.DataFrame(), det
    n_b = base.groupby("customer_id")[basket_key[1]].nunique()
    det["eligible_customers"], det["eligible_baskets"] = int(len(n_b)), int(n_b.sum())
    h = pd.Series(_hash_series(pd.Series(n_b.index), args.random_seed), index=n_b.index)
    keep = n_b.loc[h.sort_values().index].cumsum() <= args.max_affinity_baskets
    keep_ids = keep[keep].index
    if len(keep_ids) < 30:
        det["reason"] = "fewer than 30 customers within the basket budget"
        return pd.DataFrame(), pd.DataFrame(), det
    base = base[base["customer_id"].isin(keep_ids)]
    pop = (
        base.groupby(item_col)["customer_id"]
        .nunique()
        .sort_values(ascending=False)
        .head(args.max_affinity_items)
    )
    rank, names = {k: i for i, k in enumerate(pop.index)}, list(pop.index)
    base = base[base[item_col].isin(rank)].assign(ix=lambda d: d[item_col].map(rank))
    baskets = (
        base.groupby(["customer_id", basket_key[1]])["ix"]
        .agg(lambda s: sorted(set(s))[: args.max_items_per_basket])
        .reset_index()
        .sort_values(["customer_id", basket_key[1]])
    )
    half = {cid: int(v & np.uint64(1)) for cid, v in zip(h.index, h.to_numpy(), strict=False)}
    item_b, pair_b = Counter(), Counter()
    item_c, pair_c, n_c = [Counter(), Counter()], [Counter(), Counter()], [0, 0]
    nxt_pair, nxt_from, nxt_to, n_trans = Counter(), Counter(), Counter(), 0
    ops, n_baskets, n_customers = 0, 0, 0
    for cid, grp in baskets.groupby("customer_id", sort=False):
        cost = sum(len(s) * (len(s) - 1) // 2 for s in grp["ix"])
        if ops + cost > args.max_affinity_pair_operations:
            break
        ops += cost
        hh = half[cid]
        c_items, c_pairs, prev = set(), set(), None
        for items in grp["ix"]:
            n_baskets += 1
            item_b.update(items)
            pairs = list(combinations(items, 2))
            pair_b.update(pairs)
            c_items.update(items)
            c_pairs.update(pairs)
            if prev is not None:
                n_trans += 1
                nxt_from.update(prev)
                nxt_to.update(items)
                nxt_pair.update((a_, b_) for a_ in prev for b_ in items)
            prev = items
        n_customers += 1
        n_c[hh] += 1
        item_c[hh].update(c_items)
        pair_c[hh].update(c_pairs)
    det.update(
        {
            "customers_analyzed": n_customers,
            "baskets_analyzed": n_baskets,
            "pair_operations": ops,
            "candidate_items": len(names),
            "sampling": "customers ordered by hash; all baskets of a customer kept together",
        }
    )
    if n_baskets == 0 or not pair_b:
        det["reason"] = "no co-purchase pairs within the budget"
        return pd.DataFrame(), pd.DataFrame(), det
    N = n_c[0] + n_c[1]
    rows = []
    for (i, j), cnt in pair_b.items():
        if cnt / n_baskets < args.min_support:
            continue
        ab, a_, b_ = (
            [cc[0].get(k, 0), cc[1].get(k, 0)]
            for cc, k in ((pair_c, (i, j)), (item_c, i), (item_c, j))
        )
        rows.append((i, j, cnt, sum(ab), sum(a_), sum(b_), ab, a_, b_))
    if not rows:
        det["reason"] = "no pair reached --min-support"
        return pd.DataFrame(), pd.DataFrame(), det
    arr = lambda k: np.array([r_[k] for r_ in rows], float)  # noqa: E731
    n_ab, n_a, n_bb = arr(3), arr(4), arr(5)
    st = _pair_stats(n_ab, n_a, n_bb, N)
    q = benjamini_hochberg(st["p_value"])
    half_lift = np.array(
        [
            [
                (r_[6][hh] * n_c[hh] / (r_[7][hh] * r_[8][hh]))
                if min(r_[6][hh], r_[7][hh], r_[8][hh]) > 0
                else np.nan
                for hh in (0, 1)
            ]
            for r_ in rows
        ]
    )
    out = []
    for idx, (i, j, cnt, *_rest) in enumerate(rows):
        for ante, cons in ((i, j), (j, i)):
            conf = cnt / item_b[ante]
            out.append(
                {
                    "basket_level": args.basket_level,
                    "antecedent": names[ante],
                    "consequent": names[cons],
                    "pair_baskets": int(cnt),
                    "basket_support": cnt / n_baskets,
                    "confidence": conf,
                    "basket_lift": conf / (item_b[cons] / n_baskets),
                    "customers_with_pair": int(n_ab[idx]),
                    "customer_lift": float(st["lift"][idx]),
                    "customer_odds_ratio": float(st["odds_ratio"][idx]),
                    "customer_odds_ratio_ci95_lower": float(st["or_ci95_lower"][idx]),
                    "p_value": float(st["p_value"][idx]),
                    "q_value_bh": float(q[idx]),
                    "lift_half_a": float(half_lift[idx, 0]),
                    "lift_half_b": float(half_lift[idx, 1]),
                    "persistent_both_halves": bool(np.all(half_lift[idx] > 1.0)),
                }
            )
    aff = pd.DataFrame(out)
    det["pairs_tested"] = len(rows)
    aff["passes_all_filters"] = (
        (aff["confidence"] >= args.min_confidence)
        & (aff["basket_lift"] >= args.min_lift)
        & (aff["q_value_bh"] <= args.fdr_alpha)
        & (aff["customer_odds_ratio_ci95_lower"] > 1.0)
        & aff["persistent_both_halves"]
    )
    aff = aff.sort_values(
        ["passes_all_filters", "customer_lift"], ascending=[False, False]
    ).reset_index(drop=True)
    tr = pd.DataFrame()
    if n_trans >= 30:
        trows = [
            (a_, b_, c_) for (a_, b_), c_ in nxt_pair.items() if c_ / n_trans >= args.min_support
        ]
        if trows:
            n_ab_t = np.array([t_[2] for t_ in trows], float)
            n_a_t = np.array([nxt_from[t_[0]] for t_ in trows], float)
            n_b_t = np.array([nxt_to[t_[1]] for t_ in trows], float)
            st_t = _pair_stats(n_ab_t, n_a_t, n_b_t, n_trans)
            tr = pd.DataFrame(
                {
                    "basket_level": args.basket_level,
                    "from_item": [names[t_[0]] for t_ in trows],
                    "next_trip_item": [names[t_[1]] for t_ in trows],
                    "transitions": n_ab_t.astype(int),
                    "p_next_given_from": n_ab_t / n_a_t,
                    "p_next_overall": n_b_t / n_trans,
                    "next_trip_lift": st_t["lift"],
                    "odds_ratio_ci95_lower": st_t["or_ci95_lower"],
                    "p_value": st_t["p_value"],
                    "q_value_bh": benjamini_hochberg(st_t["p_value"]),
                }
            )
            tr = tr[
                (tr["q_value_bh"] <= args.fdr_alpha) & (tr["odds_ratio_ci95_lower"] > 1.0)
            ].sort_values("next_trip_lift", ascending=False)
            det["transitions_tested"] = len(trows)
    det["status"] = "completed"
    return aff, tr, det


# --------------------------------------------------------------------------- #
# Output helpers
# --------------------------------------------------------------------------- #
def json_safe(v: Any) -> Any:
    if isinstance(v, (pd.Timestamp, datetime, date)):
        return v.isoformat()
    if isinstance(v, np.generic):
        return json_safe(v.item())
    if isinstance(v, dict):
        return {str(k): json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [json_safe(x) for x in v]
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def have_parquet() -> bool:
    try:
        import pyarrow  # noqa: F401

        return True
    except ImportError:
        return False


def write_table(df: pd.DataFrame, out: Path, stem: str, fmt_: str, written: list[str]) -> None:
    if df is None:
        return
    if fmt_ == "parquet" and have_parquet():
        df.to_parquet(out / f"{stem}.parquet", index=False, compression="zstd")
        written.append(f"{stem}.parquet")
    else:
        if fmt_ == "parquet":
            LOGGER.warning("pyarrow not installed; writing %s as CSV", stem)
        df.to_csv(out / f"{stem}.csv", index=False)
        written.append(f"{stem}.csv")


def fmt(v: Any, d: int = 2) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if not math.isfinite(f) else f"{f:,.{d}f}"


def write_report(out: Path, ctx: dict[str, Any]) -> None:
    q, val, bg, gg, acc = (
        ctx["quality"],
        ctx["validation_metrics"],
        ctx["bgnbd"],
        ctx["gamma_gamma"],
        ctx["accounting"],
    )
    L = [
        "# Retail customer analytics report",
        "",
        f"**As-of date:** {ctx['as_of']}  |  **Horizon:** {ctx['horizon']} days  |  **Frequency grain:** {ctx['grain']}  |  **Customers scored:** {ctx['n_customers']:,}",
        "",
        "## Data quality and accounting",
        "",
        f"Input rows {q['input_rows']:,}; retained {q['cleaned_rows']:,}. Rows without customer_id: {q['issue_counts']['missing_customer_id_rows']:,} "
        "(excluded from customer-level results; check whether anonymous baskets differ from identified ones).",
        f"Gross purchase revenue {fmt(acc['gross_purchase_revenue'])}; return value {fmt(acc['return_value'])}; net revenue {fmt(acc['net_revenue'])} "
        "(revenue, not profit). Accounting invariants passed.",
        f"Returns: {ctx['returns'].get('return_lines', 0):,} lines; {fmt(100 * (ctx['returns'].get('matched_return_value_share') or 0), 1)}% "
        "of return value is matched to an earlier purchase of the same customer and product.",
        "",
        "## Model validation (holdout)",
        "",
    ]
    if ctx["validation_info"]["status"] == "completed" and len(val):
        L += [
            f"Models were fitted on data up to {ctx['validation_info']['cutoff_date']} and scored on the next {ctx['horizon']} days.",
            "",
            "| model | target | n | MAE | Spearman | AUC | predicted total | actual total |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in val.to_dict("records"):
            L.append(
                f"| {r['model']} | {r['target']} | {r['n_customers']:,} | {fmt(r['mae'], 3)} | {fmt(r['spearman'], 3)} | "
                f"{fmt(r['auc_any_purchase'], 3)} | {fmt(r['predicted_total'], 1)} | {fmt(r['actual_total'], 1)} |"
            )
        L += [
            "",
            "Use a model only where it beats its baseline here; decile calibration is in `holdout_calibration`.",
        ]
    else:
        L.append(f"Validation skipped: {ctx['validation_info'].get('reason')}.")
    L += ["", "## Fitted models (final cutoff)", ""]
    L.append(
        f"BG/NBD: r={fmt(bg['r'], 3)}, alpha={fmt(bg['alpha'], 2)}, a={fmt(bg['a'], 3)}, b={fmt(bg['b'], 3)}; converged={bg['converged']}."
        if bg.get("status") == "fitted"
        else f"BG/NBD not fitted ({bg.get('reason')}); empirical rates used."
    )
    if gg.get("status") == "fitted":
        L.append(
            f"Gamma-Gamma: p={fmt(gg['p'], 3)}, q={fmt(gg['q'], 3)}, gamma={fmt(gg['gamma'], 2)}; Spearman(trips, mean value)="
            f"{fmt(gg['spearman_trips_vs_mean_value'], 3)}"
            + (
                " - independence assumption looks violated; treat value forecasts with caution."
                if gg.get("independence_warning")
                else "."
            )
        )
    else:
        L.append(f"Gamma-Gamma not fitted ({gg.get('reason')}); observed mean trip value used.")
    L.append("Expected revenue is revenue, not profit, and is undiscounted.")
    L += ["", "## Segments", ""]
    if len(ctx["cluster_profiles"]):
        sel = ctx["cluster_diag"][ctx["cluster_diag"]["selected"]].iloc[0]
        L.append(
            f"Selected k={int(sel['candidate_k'])} (silhouette {fmt(sel['silhouette'], 3)}, bootstrap ARI {fmt(sel['stability_ari_mean'], 3)}): {sel['selection_reason']}."
        )
        for r in ctx["cluster_profiles"].to_dict("records"):
            L.append(
                f"- **{r['cluster_label']}**: {int(r['customers']):,} customers ({100 * r['share_of_customers']:.1f}%), median P(alive) "
                f"{fmt(r['median_p_alive'], 2)}, median expected trips {fmt(r['median_expected_trips'], 2)}, total expected revenue {fmt(r['total_expected_revenue'], 0)}."
            )
    else:
        L.append("Clustering skipped; see `cluster_diagnostics`.")
    L += ["", "## Repeat behaviour and censoring", ""]
    s = ctx["survival"]
    for g in ("all_customers", "new_after_burn_in", "existing_at_start_(left_censored)"):
        sub = s[(s["group"] == g) & (s["day"] == 90)] if len(s) else s
        if len(sub):
            r = sub.iloc[0]
            L.append(
                f"- {g}: second trip within 90 days {100 * r['repeat_probability']:.1f}% (95% CI {100 * r['repeat_ci95_lower']:.1f}-{100 * r['repeat_ci95_upper']:.1f}%, n={int(r['customers']):,})."
            )
    L += [
        "",
        "## Basket affinity",
        "",
        f"Status: {ctx['affinity_details']['status']}; customers analyzed {ctx['affinity_details'].get('customers_analyzed', 0):,}; "
        f"pairs tested {ctx['affinity_details'].get('pairs_tested', 0):,}. Rules marked `passes_all_filters` have BH q <= {ctx['fdr_alpha']}, a customer-level "
        "odds-ratio CI above 1 and lift above 1 in both customer halves. They are associations, not causal effects.",
    ]
    if len(ctx["affinity"]):
        for r in ctx["affinity"][ctx["affinity"]["passes_all_filters"]].head(10).to_dict("records"):
            L.append(
                f"- `{r['antecedent']}` -> `{r['consequent']}`: customer lift {fmt(r['customer_lift'], 2)} (odds-ratio CI lower "
                f"{fmt(r['customer_odds_ratio_ci95_lower'], 2)}), confidence {fmt(100 * r['confidence'], 1)}%."
            )
    pi = ctx["price_info"]
    L += [
        "",
        "## Price and promo proxy",
        "",
        f"Reference price available for {pi['products_with_reference_price']:,} products; {pi['products_with_price_variation']:,} show price variation. "
        + (
            "Promo-like shares were computed (proxy only)."
            if pi["usable"]
            else "Too little price variation: promo features were NOT computed."
        ),
    ]
    if ctx["warnings"]:
        L += ["", "## Warnings", ""] + [f"- {w}" for w in ctx["warnings"]]
    (out / "analysis_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_analysis(args: argparse.Namespace) -> int:
    started = datetime.now(timezone.utc)
    path, out = (
        Path(args.input).expanduser().resolve(),
        Path(args.output_dir).expanduser().resolve(),
    )
    if not path.is_file():
        raise FileNotFoundError(f"input file does not exist: {path}")
    out.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Loading input")
    raw, issues = load_lines(path, args.date_format)
    lines, removed = clean_lines(raw, args)
    issues.update(structural_checks(lines))
    max_day, min_day = lines["transaction_day"].max(), lines["transaction_day"].min()
    if pd.isna(max_day) and not args.as_of_date:
        raise ValueError("no valid transaction dates and no --as-of-date supplied")
    as_of = pd.Timestamp(args.as_of_date) if args.as_of_date else max_day
    future = int((lines["transaction_day"] > as_of).sum())
    lines = lines[~(lines["transaction_day"] > as_of)].reset_index(drop=True)
    warnings: list[str] = []
    skipped: list[dict[str, str]] = []
    if issues["missing_customer_id_rows"]:
        warnings.append(
            f"{issues['missing_customer_id_rows']} rows lack customer_id and are excluded from customer-level analyses."
        )
    if issues["malformed_transaction_date_rows"]:
        warnings.append(
            f"{issues['malformed_transaction_date_rows']} unparseable dates were dropped from date-based analyses."
        )
    if issues["negative_price_rows"]:
        warnings.append(
            f"{issues['negative_price_rows']} negative-price rows retained in accounting; trip values clip line revenue at 0."
        )
    if issues["exact_duplicate_excess_rows"] and not args.drop_exact_duplicates:
        warnings.append(
            f"{issues['exact_duplicate_excess_rows']} exact duplicate rows retained (may be genuine); use --drop-exact-duplicates to remove."
        )
    if (
        issues["transaction_ids_used_by_multiple_customers"]
        and args.transaction_key_mode == "transaction-only"
    ):
        warnings.append(
            f"{issues['transaction_ids_used_by_multiple_customers']} transaction_ids occur under several customers although transaction-only mode was selected."
        )
    if issues["product_ids_with_conflicting_category_or_department"]:
        warnings.append(
            f"{issues['product_ids_with_conflicting_category_or_department']} product_ids have conflicting category/department; see product_metadata_variants."
        )

    LOGGER.info("Building purchase events, returns and price index")
    trips = build_trips(lines, args.frequency_grain)
    matched, ret_info = match_returns(lines)
    lines, price_info = add_price_index(lines, args)
    if not price_info["usable"]:
        warnings.append(
            "Price shows little variation within products; promo-proxy features were not computed."
        )

    cat_s, dep_s = dimension_summary(lines, "category"), dimension_summary(lines, "department")
    prod_s, prod_variants, prod_repeat = product_tables(lines, args.frequency_grain)
    accounting = accounting_check(
        lines,
        {"category": cat_s, "department": dep_s, "product": dimension_summary(lines, "product_id")},
    )
    month_s = dimension_summary(
        lines.assign(month=lines["transaction_day"].dt.to_period("M").astype(str)), "month"
    ).sort_values("month")
    weekday_s = dimension_summary(
        lines[lines["transaction_day"].notna()].assign(
            iso_weekday_monday_1=lines["transaction_day"].dt.dayofweek + 1
        ),
        "iso_weekday_monday_1",
    ).sort_values("iso_weekday_monday_1")

    ctx: dict[str, Any] = {
        "as_of": as_of.date().isoformat(),
        "horizon": args.horizon_days,
        "returns": ret_info,
        "price_info": price_info,
        "fdr_alpha": args.fdr_alpha,
        "warnings": warnings,
        "grain": args.frequency_grain,
        "accounting": accounting,
    }
    empty = pd.DataFrame()
    cust, bg, gg = (
        empty,
        {"status": "skipped", "reason": "no trips"},
        {"status": "skipped", "reason": "no trips"},
    )
    val_metrics = val_calib = cluster_diag = cluster_prof = surv = retention = aff = nxt = ipt = (
        empty
    )
    val_info: dict[str, Any] = {"status": "skipped", "reason": "no trips"}
    aff_det: dict[str, Any] = {"status": "skipped"}

    if trips.empty:
        warnings.append(
            "No identifiable positive-purchase events; customer-level analyses are empty."
        )
    else:
        LOGGER.info("Validating models on a time-based holdout")
        try:
            if getattr(args, "rolling_origins", 1) > 1:
                val_metrics, val_calib, val_info = rolling_origin_validation(
                    trips,
                    as_of,
                    args,
                    n_origins=args.rolling_origins,
                    origin_spacing_days=args.origin_spacing_days,
                )
            else:
                val_metrics, val_calib, val_info = holdout_validation(trips, as_of, args)
        except Exception as exc:
            val_info = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
            skipped.append({"analysis": "holdout validation", "reason": val_info["reason"]})
        LOGGER.info("Fitting final customer models")
        # Use canonical feature builder (includes all feature families: rolling, breadth, returns, promo, category affinity, trends, basket)
        from retail_customer_analytics.features import build_customer_snapshot

        cust = build_customer_snapshot(
            transactions=lines,
            as_of=as_of,
            event_grain=args.frequency_grain,
            windows_days=args.windows_days,
            recent_window_days=args.recent_window_days,
        )
        # Score with probabilistic models
        cust, bg, gg = score_customers(cust, args.horizon_days, args)
        cust = assign_rfm(cust)
        pcs = top_components(category_mix(lines, cust.index, as_of, args.max_clr_categories)[0])
        LOGGER.info("Clustering with stability checks")
        try:
            cust, cluster_diag, cluster_prof = cluster_customers(cust, pcs, args)
        except Exception as exc:
            skipped.append({"analysis": "clustering", "reason": f"{type(exc).__name__}: {exc}"})
            cluster_diag = pd.DataFrame(
                [{"status": "failed", "reason": str(exc), "selected": False}]
            )

        # RFM vs Cluster comparison
        rfm_cluster_comparison = compare_rfm_vs_cluster(cust) if len(cust) else pd.DataFrame()

        # Cross-snapshot clustering stability (if enough history)
        cluster_stability = pd.DataFrame()
        cluster_labels_snapshots = pd.DataFrame()
        if not trips.empty and trips["transaction_day"].nunique() >= 3:
            # Generate quarterly snapshots
            max_date = trips["transaction_day"].max()
            min_date = trips["transaction_day"].min()
            if (max_date - min_date).days >= 180:  # Need at least 6 months
                snapshot_dates = pd.date_range(
                    min_date + pd.Timedelta(days=90),
                    max_date - pd.Timedelta(days=args.horizon_days),
                    freq="QE",
                )
                if len(snapshot_dates) >= 2:
                    try:
                        cluster_stability, cluster_labels_snapshots = (
                            cluster_stability_across_snapshots(lines, snapshot_dates.tolist(), args)
                        )
                    except Exception as exc:
                        LOGGER.warning(f"Cross-snapshot clustering stability failed: {exc}")
                        cluster_stability = pd.DataFrame()

        surv, cohort_info = repeat_survival(trips, as_of, args.burn_in_days)
        retention = cohort_retention(trips, cohort_info, as_of)
        ipt = interpurchase_table(trips, as_of)
    LOGGER.info("Computing basket affinity")
    try:
        aff, nxt, aff_det = basket_affinity(lines, as_of, args)
    except Exception as exc:
        aff_det = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
    if aff_det.get("status") != "completed":
        skipped.append({"analysis": "basket affinity", "reason": str(aff_det.get("reason"))})

    quality = {
        "input_path": str(path),
        "input_rows": int(len(raw)),
        "cleaned_rows": int(len(lines)),
        "issue_counts": issues,
        "exclusions": removed,
        "rows_after_as_of_excluded": future,
        "minimum_observed_date": min_day,
        "analysis_as_of_date": as_of,
        "returns": ret_info,
        "price_index": price_info,
        "accounting": accounting,
        "warnings": warnings,
    }
    written: list[str] = []
    (out / "data_quality.json").write_text(
        json.dumps(json_safe(quality), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    written.append("data_quality.json")
    issue_table = pd.DataFrame(
        [
            {"issue": k, "count_or_value": json.dumps(json_safe(v), ensure_ascii=False)}
            for k, v in issues.items()
        ]
    )
    model_rows = [
        {"component": "bgnbd_final", **bg},
        {"component": "gamma_gamma_final", **gg},
        {
            "component": "holdout",
            **{
                k: (json.dumps(json_safe(v)) if isinstance(v, dict) else v)
                for k, v in val_info.items()
            },
        },
    ]
    rfm_prof = (
        cust.groupby("rfm_segment")
        .agg(
            customers=("n_trips", "size"),
            median_recency_days=("recency_days", "median"),
            mean_trips=("n_trips", "mean"),
            median_net_spend=("net_spend", "median"),
            mean_p_alive=("p_alive", "mean"),
            repeat_customer_rate=("n_trips", lambda v: float((v >= 2).mean())),
            heuristic_inactive_rate=("heuristic_inactive_flag", "mean"),
        )
        .reset_index()
        .sort_values("customers", ascending=False)
        if len(cust)
        else empty
    )
    tables = [
        ("data_quality_issues", issue_table),
        ("customer_features", cust.reset_index() if len(cust) else cust),
        ("holdout_metrics", val_metrics),
        ("holdout_calibration", val_calib),
        ("model_parameters", pd.DataFrame(model_rows)),
        ("cluster_diagnostics", cluster_diag),
        ("cluster_profiles", cluster_prof),
        ("rfm_segment_profiles", rfm_prof),
        ("rfm_cluster_comparison", rfm_cluster_comparison),
        ("cluster_stability", cluster_stability),
        ("cluster_labels_snapshots", cluster_labels_snapshots),
        ("repeat_survival", surv),
        ("cohort_retention_new_customers", retention),
        ("interpurchase_intervals", ipt),
        ("basket_affinity", aff),
        ("next_trip_affinity", nxt),
        ("customer_revenue_concentration", revenue_concentration(cust) if len(cust) else empty),
        ("transaction_summary", transaction_summary(lines, args.transaction_key_mode)),
        ("product_summary", prod_s),
        ("product_metadata_variants", prod_variants),
        ("product_repeat_behavior", prod_repeat),
        ("category_summary", cat_s),
        ("department_summary", dep_s),
        ("monthly_summary", month_s),
        ("weekday_summary", weekday_s),
    ]
    for stem, tbl in tables:
        write_table(tbl, out, stem, args.format, written)
    ctx.update(
        {
            "quality": quality,
            "validation_metrics": val_metrics,
            "validation_info": val_info,
            "bgnbd": bg,
            "gamma_gamma": gg,
            "n_customers": int(len(cust)),
            "cluster_profiles": cluster_prof,
            "cluster_diag": cluster_diag,
            "survival": surv,
            "affinity": aff,
            "affinity_details": aff_det,
        }
    )
    write_report(out, ctx)
    written.append("analysis_report.md")
    done = datetime.now(timezone.utc)
    meta = {
        "run_started_utc": started,
        "run_completed_utc": done,
        "elapsed_seconds": (done - started).total_seconds(),
        "python_version": platform.python_version(),
        "package_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": __import__("scipy").__version__,
        },
        "configuration": {**vars(args), "input": str(path), "output_dir": str(out)},
        "skipped_analyses": skipped,
        "affinity_details": aff_det,
        "warnings": warnings,
        "output_files": sorted(set(written + ["run_metadata.json"])),
    }
    (out / "run_metadata.json").write_text(
        json.dumps(json_safe(meta), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    LOGGER.info("Analysis complete: %s (%d files)", out, len(meta["output_files"]))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(args, parser)
    configure_logging(args.log_level)
    try:
        return run_analysis(args)
    except Exception as exc:
        LOGGER.error("Analysis failed: %s: %s", type(exc).__name__, exc)
        LOGGER.debug("Stack trace", exc_info=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
