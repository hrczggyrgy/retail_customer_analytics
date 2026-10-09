#!/usr/bin/env python3
"""Analyze retail line-item transaction data with Polars.

Purpose
-------
A single-file, command-line retail analytics pipeline for data validation,
accounting-reconciled sales/returns summaries, customer features and RFM
segments, bounded customer clustering, observed purchase cohorts/retention,
and bounded basket-affinity rules.

Dependencies
------------
Python 3.10+; Polars, NumPy, SciPy, and scikit-learn. Clustering is skipped
with an explicit diagnostic if scikit-learn is unavailable; all other outputs
remain available.

Example installation command
----------------------------
    python -m pip install polars numpy scipy scikit-learn

Example execution command
-------------------------
    python retail_customer_analysis.py --input transactions.csv \
        --output-dir retail_output --write-csv --windows-days 30,90,365

Input assumptions
-----------------
Required columns are customer_id, transaction_date, transaction_id,
product_id, product_description, department, category, price, and quantity.
Each row is a line item. Price is assumed to be a unit price; line revenue is
price * quantity; negative quantity denotes a return. Missing customer or
transaction identifiers are retained for line-level summaries but cannot
participate in analyses that require identifiable transactions. By default,
transaction keys are (customer_id, transaction_id). Dates are parsed using
an optional user format followed by a documented list of common formats; for
ambiguous slash-separated dates, month/day/year is tried before day/month/year.
Dates, cohorts, and windows describe only the records observed in this file,
not the customer's complete lifetime or true acquisition date.

Output artifacts
----------------
The output directory receives data_quality.json and data_quality_issues,
customer_features, customer_revenue_concentration, interpurchase_intervals,
transaction_summary, product_summary,
product_metadata_variants, product_repeat_behavior, category_summary,
department_summary, monthly_summary, weekday_summary, cohort_summary,
retention_table, basket_affinity, model_diagnostics, rfm_segment_profiles,
cluster_profiles, analysis_report.md, and run_metadata.json. Tabular outputs
are Parquet and are also written as CSV when --write-csv is used. Known output
filenames are overwritten on reruns; unrelated files in the output directory
are left untouched.

Key methodological limitations
------------------------------
Revenue is accounting revenue, not profit: no cost, margin, discount, or tax
fields are assumed. Return lines are not matched to original sales. Cohorts
are first-observed-purchase cohorts, not verified acquisition cohorts, and
retention means a later identified positive-purchase transaction in the month.
Inactivity flags are cadence-based heuristics, not calibrated churn probabilities.
Clusters are descriptive and sample-/feature-dependent. Basket rules are
observational, not causal; item-pair work is explicitly capped and may use a
deterministic hash-ordered subset of baskets. No supervised target is invented.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import platform
import sys
from collections import Counter
from datetime import date, datetime, time, timedelta, timezone
from importlib import metadata as importlib_metadata
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import polars as pl
from scipy.stats import rankdata

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import RobustScaler

    SKLEARN_AVAILABLE = True
except ImportError:  # Descriptive outputs should still run without clustering.
    KMeans = None  # type: ignore[assignment,misc]
    silhouette_score = None  # type: ignore[assignment,misc]
    RobustScaler = None  # type: ignore[assignment,misc]
    SKLEARN_AVAILABLE = False


COLUMNS = [
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
ID_COLUMNS = ["customer_id", "transaction_id", "product_id"]
TEXT_COLUMNS = ["product_description", "department", "category"]
NUMERIC_COLUMNS = ["price", "quantity"]
LOGGER = logging.getLogger("retail_analysis")


class JsonFormatter(logging.Formatter):
    """Format log records as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "level": record.levelname,
                "logger": record.name,
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
    """Parse a comma-separated list of unique positive integers."""
    try:
        values = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("all window values must be positive")
    return sorted(set(values))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze retail line-item data with auditable metrics and bounded models."
    )
    parser.add_argument("--input", required=True, help="Input .csv, .parquet, or .pq file")
    parser.add_argument("--output-dir", default="retail_analysis_output")
    parser.add_argument("--write-csv", action="store_true", help="Also write CSV copies of tables")
    parser.add_argument(
        "--date-format",
        default=None,
        help="Optional Python/Chrono strptime format tried before built-in formats",
    )
    parser.add_argument("--as-of-date", default=None, help="Analysis cutoff date YYYY-MM-DD")
    parser.add_argument("--windows-days", type=parse_positive_int_list, default=[30, 90, 365])
    parser.add_argument("--recent-window-days", type=int, default=90)
    parser.add_argument(
        "--transaction-key-mode",
        choices=["customer-transaction", "transaction-only"],
        default="customer-transaction",
        help="Whether transaction_id is only unique within customer_id or globally unique",
    )
    parser.add_argument("--drop-exact-duplicates", action="store_true")
    parser.add_argument("--exclude-zero-quantity", action="store_true")
    parser.add_argument("--exclude-negative-prices", action="store_true")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--min-support", type=float, default=0.01)
    parser.add_argument("--min-confidence", type=float, default=0.20)
    parser.add_argument("--min-lift", type=float, default=1.0)
    parser.add_argument("--basket-level", choices=["category", "product"], default="category")
    parser.add_argument("--max-affinity-items", type=int, default=200)
    parser.add_argument("--max-affinity-baskets", type=int, default=50000)
    parser.add_argument("--max-items-per-basket", type=int, default=40)
    parser.add_argument("--max-affinity-pair-operations", type=int, default=1000000)
    parser.add_argument("--cluster-max-customers", type=int, default=50000)
    parser.add_argument("--cluster-max-evaluation-sample", type=int, default=2000)
    parser.add_argument("--max-clusters", type=int, default=6)
    parser.add_argument("--inactivity-min-days", type=float, default=30.0)
    parser.add_argument("--inactivity-cadence-multiplier", type=float, default=2.0)
    parser.add_argument("--inactivity-fallback-days", type=float, default=90.0)
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    return parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.recent_window_days <= 0:
        parser.error("--recent-window-days must be positive")
    if not (0.0 <= args.min_support <= 1.0):
        parser.error("--min-support must be between 0 and 1")
    if not (0.0 <= args.min_confidence <= 1.0):
        parser.error("--min-confidence must be between 0 and 1")
    if args.min_lift < 0:
        parser.error("--min-lift must be non-negative")
    for name in (
        "max_affinity_items",
        "max_affinity_baskets",
        "max_items_per_basket",
        "max_affinity_pair_operations",
        "cluster_max_customers",
        "cluster_max_evaluation_sample",
        "max_clusters",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.inactivity_min_days < 0 or args.inactivity_fallback_days < 0:
        parser.error("inactivity day thresholds must be non-negative")
    if args.inactivity_cadence_multiplier <= 0:
        parser.error("--inactivity-cadence-multiplier must be positive")
    if args.as_of_date:
        try:
            datetime.strptime(args.as_of_date, "%Y-%m-%d").date()
        except ValueError:
            parser.error("--as-of-date must use YYYY-MM-DD")


def read_lazy_input(path: Path) -> pl.LazyFrame:
    """Open supported input formats without eagerly loading the source."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pl.scan_csv(
            str(path),
            infer_schema=False,
            null_values=["", "NULL", "null"],
            encoding="utf8-lossy",
        )
    if suffix in {".parquet", ".pq"}:
        return pl.scan_parquet(str(path))
    raise ValueError("input extension must be .csv, .parquet, or .pq")


def _date_parse_expression(date_format: str | None) -> pl.Expr:
    """Create a deterministic coalesced date parser expression."""
    raw = pl.col("__raw_transaction_date").str.strip_chars()
    formats: list[str] = []
    if date_format:
        formats.append(date_format)
    formats.extend(
        [
            "%+",
            "%Y-%m-%dT%H:%M:%S%.f",
            "%Y-%m-%d %H:%M:%S%.f",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
            "%m/%d/%Y %H:%M:%S",
            "%m/%d/%Y",
            "%d/%m/%Y %H:%M:%S",
            "%d/%m/%Y",
            "%m-%d-%Y",
            "%d-%m-%Y",
        ]
    )
    unique_formats = list(dict.fromkeys(formats))
    return pl.coalesce(
        [raw.str.strptime(pl.Datetime, format=fmt, strict=False, exact=True) for fmt in unique_formats]
    )


def convert_input(lazy: pl.LazyFrame, date_format: str | None) -> pl.DataFrame:
    """Apply conservative string, numeric, and date conversions."""
    raw_exprs = [
        pl.col(column).cast(pl.String, strict=False).alias(f"__raw_{column}")
        for column in COLUMNS
    ]
    frame = lazy.select(COLUMNS).with_columns(raw_exprs)

    converted: list[pl.Expr] = []
    for column in ID_COLUMNS + TEXT_COLUMNS:
        raw = pl.col(f"__raw_{column}").str.strip_chars()
        converted.append(
            pl.when(raw.is_null() | (raw == ""))
            .then(None)
            .otherwise(raw)
            .alias(column)
        )
    converted.append(_date_parse_expression(date_format).alias("transaction_date"))
    for column in NUMERIC_COLUMNS:
        raw_num = pl.col(f"__raw_{column}").cast(pl.Float64, strict=False)
        converted.append(
            pl.when(raw_num.is_finite()).then(raw_num).otherwise(None).alias(column)
        )
    frame = frame.with_columns(converted)

    presence_exprs = [
        (
            pl.col(f"__raw_{column}").is_not_null()
            & (pl.col(f"__raw_{column}").str.strip_chars() != "")
        )
        .fill_null(False)
        .alias(f"__raw_present_{column}")
        for column in ("transaction_date", "price", "quantity")
    ]
    invalid_exprs = [
        (
            pl.col(f"__raw_present_{column}") & pl.col(column).is_null()
        ).fill_null(False).alias(f"__invalid_{column}_conversion")
        for column in ("transaction_date", "price", "quantity")
    ]
    frame = frame.with_columns(presence_exprs).with_columns(invalid_exprs)
    raw_revenue = pl.col("price") * pl.col("quantity")
    frame = frame.with_columns(
        pl.col("transaction_date").dt.date().alias("transaction_day"),
        pl.when(raw_revenue.is_finite()).then(raw_revenue).otherwise(None).alias("__line_revenue"),
    )
    return frame.collect()


def _sum_int(frame: pl.DataFrame, expression: pl.Expr) -> int:
    if frame.height == 0:
        return 0
    value = frame.select(expression.sum()).item()
    return int(value or 0)


def _count_where(frame: pl.DataFrame, condition: pl.Expr) -> int:
    if frame.height == 0:
        return 0
    value = frame.select(condition.fill_null(False).cast(pl.UInt64).sum()).item()
    return int(value or 0)


def duplicate_statistics(frame: pl.DataFrame) -> tuple[int, int]:
    """Return excess duplicate rows and number of duplicated normalized patterns."""
    if frame.height == 0:
        return 0, 0
    repeated = (
        frame.group_by(COLUMNS)
        .agg(pl.len().alias("__copies"))
        .filter(pl.col("__copies") > 1)
    )
    duplicated_patterns = repeated.height
    excess = repeated.select((pl.col("__copies") - 1).sum()).item() if repeated.height else 0
    return int(excess or 0), int(duplicated_patterns)


def line_financial_aggregations() -> list[pl.Expr]:
    """Expressions shared by line, transaction, product, and time summaries."""
    quantity = pl.col("quantity")
    revenue = pl.col("__line_revenue")
    return [
        pl.len().alias("line_item_count"),
        revenue.sum().fill_null(0.0).alias("net_revenue"),
        pl.when((quantity > 0) & (revenue > 0))
        .then(revenue)
        .otherwise(0.0)
        .sum()
        .fill_null(0.0)
        .alias("positive_purchase_spend"),
        pl.when(quantity > 0).then(revenue).otherwise(None).sum().fill_null(0.0).alias(
            "gross_purchase_revenue"
        ),
        pl.when(quantity < 0).then(-revenue).otherwise(None).sum().fill_null(0.0).alias(
            "return_value"
        ),
        pl.when(quantity > 0).then(quantity).otherwise(None).sum().fill_null(0.0).alias(
            "units_purchased"
        ),
        pl.when(quantity < 0).then(-quantity).otherwise(None).sum().fill_null(0.0).alias(
            "units_returned"
        ),
        quantity.sum().fill_null(0.0).alias("net_units"),
        pl.when(quantity > 0).then(1).otherwise(None).sum().fill_null(0).cast(pl.Int64).alias(
            "purchase_line_item_count"
        ),
        pl.when(quantity < 0).then(1).otherwise(None).sum().fill_null(0).cast(pl.Int64).alias(
            "return_line_item_count"
        ),
        pl.when(quantity == 0).then(1).otherwise(None).sum().fill_null(0).cast(pl.Int64).alias(
            "zero_quantity_line_item_count"
        ),
        revenue.is_not_null().cast(pl.Int64).sum().fill_null(0).alias(
            "line_items_with_calculable_revenue"
        ),
    ]


def summarize_transaction_lines(lines: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    """Aggregate line items to the requested transaction-key grain."""
    if not keys:
        raise ValueError("at least one transaction key column is required")
    identified = lines.filter(
        pl.all_horizontal([pl.col(key).is_not_null() for key in keys])
    )
    aggs = line_financial_aggregations() + [
        pl.col("transaction_day").drop_nulls().min().alias("transaction_day"),
        pl.col("transaction_day").drop_nulls().max().alias("last_line_day"),
        pl.col("transaction_day").drop_nulls().n_unique().alias("distinct_transaction_dates"),
        pl.col("product_id").drop_nulls().n_unique().alias("distinct_product_ids_all_lines"),
        pl.col("product_id")
        .filter(pl.col("quantity") > 0)
        .drop_nulls()
        .n_unique()
        .alias("basket_size_distinct_products"),
        pl.col("customer_id").drop_nulls().n_unique().alias("customer_count"),
    ]
    if "customer_id" not in keys:
        aggs.extend(
            [
                pl.col("customer_id").drop_nulls().first().alias("customer_id"),
                pl.col("customer_id").drop_nulls().unique().sort().alias("customer_ids"),
            ]
        )
    result = identified.group_by(keys).agg(aggs)
    result = result.with_columns(
        (pl.col("units_purchased") > 0).alias("is_purchase_transaction"),
        pl.when(pl.col("units_purchased") > 0)
        .then(pl.col("gross_purchase_revenue"))
        .otherwise(None)
        .alias("purchase_order_value"),
        (pl.col("gross_purchase_revenue") - pl.col("return_value")).alias(
            "net_revenue_from_gross_less_returns"
        ),
    )
    return result.sort(keys)


def distinct_transaction_count_expr(transaction_key_mode: str) -> pl.Expr:
    """Return a distinct transaction count expression under the configured key policy."""
    if transaction_key_mode == "transaction-only":
        return pl.col("transaction_id").drop_nulls().n_unique()
    return pl.struct(["customer_id", "transaction_id"]).filter(
        pl.col("customer_id").is_not_null() & pl.col("transaction_id").is_not_null()
    ).n_unique()


def dimension_summary(
    frame: pl.DataFrame, dimension: str, transaction_key_mode: str
) -> pl.DataFrame:
    """Summarize line-item financial and activity metrics by a single dimension."""
    return frame.group_by(dimension).agg(
        line_financial_aggregations()
        + [
            pl.col("customer_id").drop_nulls().n_unique().alias("distinct_customers"),
            distinct_transaction_count_expr(transaction_key_mode).alias("distinct_identified_transactions"),
            pl.col("product_id").drop_nulls().n_unique().alias("distinct_product_ids"),
            pl.col("price").mean().alias("mean_unit_price_per_line_item"),
            pl.col("price").median().alias("median_unit_price_per_line_item"),
        ]
    ).sort("gross_purchase_revenue", descending=True, nulls_last=True)


def build_product_outputs(
    frame: pl.DataFrame, transaction_key_mode: str
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build product-level totals and conflict-preserving metadata variants."""
    metadata_counts = (
        frame.filter(pl.col("product_id").is_not_null())
        .group_by("product_id")
        .agg(
            pl.col("product_description").drop_nulls().n_unique().alias("description_variants"),
            pl.col("department").drop_nulls().n_unique().alias("department_variants"),
            pl.col("category").drop_nulls().n_unique().alias("category_variants"),
            pl.len().alias("source_line_item_count"),
        )
        .with_columns(
            (
                (pl.col("description_variants") > 1)
                | (pl.col("department_variants") > 1)
                | (pl.col("category_variants") > 1)
            ).alias("has_metadata_conflict")
        )
    )
    variants = (
        frame.filter(pl.col("product_id").is_not_null())
        .group_by(["product_id", "product_description", "department", "category"])
        .agg(pl.len().alias("variant_line_item_count"))
        .join(metadata_counts, on="product_id", how="left")
        .filter(pl.col("has_metadata_conflict"))
        .sort(["product_id", "product_description", "department", "category"])
    )
    products = frame.group_by("product_id").agg(
        line_financial_aggregations()
        + [
            pl.col("product_description").drop_nulls().first().alias("product_description"),
            pl.col("product_description").drop_nulls().unique().sort().alias("observed_descriptions"),
            pl.col("department").drop_nulls().first().alias("department"),
            pl.col("department").drop_nulls().unique().sort().alias("observed_departments"),
            pl.col("category").drop_nulls().first().alias("category"),
            pl.col("category").drop_nulls().unique().sort().alias("observed_categories"),
            pl.col("customer_id").drop_nulls().n_unique().alias("distinct_customers"),
            distinct_transaction_count_expr(transaction_key_mode).alias("distinct_identified_transactions"),
        ]
    )
    products = products.join(
        metadata_counts.select(
            "product_id",
            "description_variants",
            "department_variants",
            "category_variants",
            "has_metadata_conflict",
        ),
        on="product_id",
        how="left",
    ).with_columns(
        pl.col("has_metadata_conflict").fill_null(False),
        pl.col("description_variants").fill_null(0),
        pl.col("department_variants").fill_null(0),
        pl.col("category_variants").fill_null(0),
    ).sort("gross_purchase_revenue", descending=True, nulls_last=True)
    return products, variants


def build_product_repeat_behavior(
    time_lines: pl.DataFrame, transaction_key_mode: str
) -> pl.DataFrame:
    """Measure observed repeat purchasing by customer-product, excluding returns."""
    lines = time_lines.filter(
        (pl.col("quantity") > 0)
        & pl.col("customer_id").is_not_null()
        & pl.col("transaction_id").is_not_null()
        & pl.col("product_id").is_not_null()
    )
    if lines.height == 0:
        return pl.DataFrame(
            schema={
                "product_id": pl.String,
                "observed_buyers": pl.Int64,
                "repeat_buyers": pl.Int64,
                "repeat_buyer_rate": pl.Float64,
            }
        )
    customer_product = lines.group_by(["customer_id", "product_id"]).agg(
        pl.col("transaction_id").drop_nulls().n_unique().alias("purchase_transaction_count")
    )
    return customer_product.group_by("product_id").agg(
        pl.len().alias("observed_buyers"),
        (pl.col("purchase_transaction_count") >= 2).cast(pl.Int64).sum().alias("repeat_buyers"),
    ).with_columns(
        pl.when(pl.col("observed_buyers") > 0)
        .then(pl.col("repeat_buyers") / pl.col("observed_buyers"))
        .otherwise(None)
        .alias("repeat_buyer_rate"),
        pl.lit(transaction_key_mode).alias("customer_purchase_key_definition"),
    ).sort("observed_buyers", descending=True)


def _safe_divide(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    return pl.when(denominator != 0).then(numerator / denominator).otherwise(None)


def month_add(month_start: date, months: int) -> date:
    """Add whole calendar months to a first-of-month date."""
    month_index = month_start.year * 12 + month_start.month - 1 + months
    return date(month_index // 12, month_index % 12 + 1, 1)


def month_end(month_start: date) -> date:
    """Return the last calendar day of a month."""
    return month_add(month_start, 1) - timedelta(days=1)


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float | None, float | None]:
    """Return a Wilson score confidence interval for a binomial proportion."""
    if total <= 0:
        return None, None
    proportion = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = (proportion + z2 / (2.0 * total)) / denominator
    half_width = z * math.sqrt(
        (proportion * (1.0 - proportion) / total) + (z2 / (4.0 * total * total))
    ) / denominator
    return max(0.0, center - half_width), min(1.0, center + half_width)


def build_cohort_outputs(
    purchase_transactions: pl.DataFrame, as_of_date: date | None
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Create first-observed-purchase cohorts and fixed-denominator monthly retention."""
    cohort_schema = {
        "cohort_month": pl.Date,
        "cohort_size": pl.Int64,
        "first_purchase_observation_date": pl.Date,
        "last_purchase_observation_date": pl.Date,
    }
    retention_schema = {
        "cohort_month": pl.Date,
        "cohort_size": pl.Int64,
        "months_since_cohort": pl.Int64,
        "activity_month": pl.Date,
        "active_customers": pl.Int64,
        "retention_rate": pl.Float64,
        "retention_ci_95_lower": pl.Float64,
        "retention_ci_95_upper": pl.Float64,
        "period_complete": pl.Boolean,
        "period_end_date": pl.Date,
    }
    if purchase_transactions.height == 0 or as_of_date is None:
        return pl.DataFrame(schema=cohort_schema), pl.DataFrame(schema=retention_schema)

    first_dates = purchase_transactions.group_by("customer_id").agg(
        pl.col("transaction_day").min().alias("first_purchase_observation_date"),
        pl.col("transaction_day").max().alias("last_purchase_observation_date"),
    ).with_columns(
        pl.col("first_purchase_observation_date").dt.truncate("1mo").alias("cohort_month")
    )
    cohorts = first_dates.group_by("cohort_month").agg(
        pl.len().alias("cohort_size"),
        pl.col("first_purchase_observation_date").min().alias("first_purchase_observation_date"),
        pl.col("last_purchase_observation_date").max().alias("last_purchase_observation_date"),
    ).sort("cohort_month")

    activity = (
        purchase_transactions.select("customer_id", "transaction_day")
        .with_columns(pl.col("transaction_day").dt.truncate("1mo").alias("activity_month"))
        .select("customer_id", "activity_month")
        .unique()
        .join(first_dates.select("customer_id", "cohort_month"), on="customer_id", how="inner")
        .with_columns(
            (
                (pl.col("activity_month").dt.year() - pl.col("cohort_month").dt.year()) * 12
                + pl.col("activity_month").dt.month()
                - pl.col("cohort_month").dt.month()
            ).alias("months_since_cohort")
        )
        .filter(pl.col("months_since_cohort") >= 0)
    )
    counts = activity.group_by(["cohort_month", "months_since_cohort", "activity_month"]).agg(
        pl.col("customer_id").n_unique().alias("active_customers")
    )
    cohort_dict = {row["cohort_month"]: int(row["cohort_size"]) for row in cohorts.iter_rows(named=True)}
    active_dict = {
        (row["cohort_month"], int(row["months_since_cohort"])): int(row["active_customers"])
        for row in counts.iter_rows(named=True)
    }
    current_month = date(as_of_date.year, as_of_date.month, 1)
    rows: list[dict[str, Any]] = []
    for cohort_month, size in sorted(cohort_dict.items()):
        max_age = (current_month.year - cohort_month.year) * 12 + current_month.month - cohort_month.month
        for age in range(max_age + 1):
            activity_month = month_add(cohort_month, age)
            active = active_dict.get((cohort_month, age), 0)
            end_date = month_end(activity_month)
            rows.append(
                {
                    "cohort_month": cohort_month,
                    "cohort_size": size,
                    "months_since_cohort": age,
                    "activity_month": activity_month,
                    "active_customers": active,
                    "retention_rate": (active / size) if size > 0 else None,
                    "retention_ci_95_lower": wilson_interval(active, size)[0],
                    "retention_ci_95_upper": wilson_interval(active, size)[1],
                    "period_complete": end_date <= as_of_date,
                    "period_end_date": end_date,
                }
            )
    retention = pl.DataFrame(rows, schema=retention_schema) if rows else pl.DataFrame(schema=retention_schema)
    return cohorts, retention.sort(["cohort_month", "months_since_cohort"])


def _rank_quintiles(values: np.ndarray, higher_is_better: bool) -> np.ndarray:
    """Map values to tied-rank quintile scores from 1 to 5."""
    if values.size == 0:
        return np.array([], dtype=np.int8)
    ranks = rankdata(values, method="average")
    # Mid-rank percentiles keep small samples away from artificial extremes.
    percentiles = (ranks - 0.5) / float(values.size)
    scores = np.clip(np.ceil(percentiles * 5.0), 1, 5).astype(np.int8)
    return scores if higher_is_better else (6 - scores).astype(np.int8)


def _rfm_label(r: np.ndarray, f: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Assign readable RFM labels using transparent score rules."""
    conditions = [
        (r >= 4) & (f >= 4) & (m >= 4),
        (r >= 3) & (f >= 4),
        (r <= 2) & (f >= 4) & (m >= 4),
        (r <= 2) & (f >= 3),
        (r >= 4) & (f <= 2),
        (r <= 2) & (f <= 2),
        (r >= 3) & (f >= 3),
    ]
    choices = [
        "Champions",
        "Frequent / loyal",
        "Previously high-value / lapsed",
        "At risk by recency",
        "Recent / low frequency",
        "Hibernating / low frequency",
        "Potential loyalists",
    ]
    return np.select(conditions, choices, default="Mixed / needs attention")


def build_customer_features(
    time_lines: pl.DataFrame,
    customer_transactions: pl.DataFrame,
    as_of_date: date | None,
    windows_days: Sequence[int],
    recent_window_days: int,
    inactivity_min_days: float,
    inactivity_cadence_multiplier: float,
    inactivity_fallback_days: float,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build customer features and a sorted interpurchase-interval table."""
    customer_lines = time_lines.filter(pl.col("customer_id").is_not_null())
    customer_base = customer_lines.group_by("customer_id").agg(
        line_financial_aggregations()
        + [
            pl.col("transaction_day").min().alias("first_observed_line_date"),
            pl.col("transaction_day").max().alias("last_observed_line_date"),
            pl.col("product_id").filter(pl.col("quantity") > 0).drop_nulls().n_unique().alias(
                "unique_products_purchased"
            ),
            pl.col("category").filter(pl.col("quantity") > 0).drop_nulls().n_unique().alias(
                "unique_categories_purchased"
            ),
            pl.col("department").filter(pl.col("quantity") > 0).drop_nulls().n_unique().alias(
                "unique_departments_purchased"
            ),
        ]
    )
    valid_customer_tx = customer_transactions.filter(
        pl.col("customer_id").is_not_null() & pl.col("transaction_day").is_not_null()
    )
    purchase_tx = valid_customer_tx.filter(pl.col("units_purchased") > 0).sort(
        ["customer_id", "transaction_day", "transaction_id"]
    )
    tx_customer_agg = valid_customer_tx.group_by("customer_id").agg(
        pl.len().alias("identified_transaction_count"),
        pl.col("transaction_day").min().alias("first_identified_transaction_date"),
        pl.col("transaction_day").max().alias("last_identified_transaction_date"),
    )
    purchase_customer_agg = purchase_tx.group_by("customer_id").agg(
        pl.len().alias("purchase_transaction_count"),
        pl.col("transaction_day").min().alias("first_purchase_date"),
        pl.col("transaction_day").max().alias("last_purchase_date"),
        pl.col("basket_size_distinct_products").mean().alias("average_basket_size"),
        pl.col("basket_size_distinct_products").median().alias("median_basket_size"),
        pl.col("gross_purchase_revenue").mean().alias("mean_gross_purchase_value_per_transaction"),
        pl.col("gross_purchase_revenue").mean().alias("gross_purchase_average_order_value"),
    )

    if purchase_tx.height:
        with_intervals = purchase_tx.with_columns(
            pl.col("transaction_day").diff().over("customer_id").dt.total_days().alias(
                "interpurchase_interval_days"
            )
        )
        cadence = with_intervals.group_by("customer_id").agg(
            pl.col("interpurchase_interval_days").drop_nulls().count().alias("interpurchase_interval_count"),
            pl.col("interpurchase_interval_days").drop_nulls().mean().alias("mean_interpurchase_days"),
            pl.col("interpurchase_interval_days").drop_nulls().median().alias("median_interpurchase_days"),
            pl.col("interpurchase_interval_days").drop_nulls().std(ddof=1).alias(
                "interpurchase_interval_std_days"
            ),
        )
        intervals = with_intervals.select(
            "customer_id", "transaction_id", "transaction_day", "interpurchase_interval_days"
        )
    else:
        cadence = pl.DataFrame(
            schema={
                "customer_id": pl.String,
                "interpurchase_interval_count": pl.UInt32,
                "mean_interpurchase_days": pl.Float64,
                "median_interpurchase_days": pl.Float64,
                "interpurchase_interval_std_days": pl.Float64,
            }
        )
        intervals = pl.DataFrame(
            schema={
                "customer_id": pl.String,
                "transaction_id": pl.String,
                "transaction_day": pl.Date,
                "interpurchase_interval_days": pl.Int64,
            }
        )

    features = (
        customer_base.join(tx_customer_agg, on="customer_id", how="left")
        .join(purchase_customer_agg, on="customer_id", how="left")
        .join(cadence, on="customer_id", how="left")
        .with_columns(
            pl.col("identified_transaction_count").fill_null(0),
            pl.col("purchase_transaction_count").fill_null(0),
            pl.col("interpurchase_interval_count").fill_null(0),
        )
    )

    # Customer-category preferences and spending concentration are computed only
    # on positive-quantity lines. HHI uses positive line revenue to avoid invalid
    # or negative unit-price records producing nonsensical spend shares.
    category_lines = customer_lines.filter(
        (pl.col("quantity") > 0) & pl.col("category").is_not_null()
    )
    if category_lines.height:
        category_spend = category_lines.group_by(["customer_id", "category"]).agg(
            pl.when(pl.col("__line_revenue") > 0)
            .then(pl.col("__line_revenue"))
            .otherwise(0.0)
            .sum()
            .alias("positive_purchase_spend"),
            pl.col("__line_revenue").sum().fill_null(0.0).alias("gross_purchase_revenue"),
        )
        top_category = (
            category_spend.sort(
                ["customer_id", "gross_purchase_revenue", "category"],
                descending=[False, True, False],
                nulls_last=True,
            )
            .group_by("customer_id", maintain_order=True)
            .agg(
                pl.col("category").first().alias("top_category_by_gross_purchase_revenue"),
                pl.col("positive_purchase_spend").sum().alias("positive_category_spend_total"),
            )
        )
        category_shares = category_spend.join(
            category_spend.group_by("customer_id").agg(
                pl.col("positive_purchase_spend").sum().alias("__customer_positive_spend")
            ),
            on="customer_id",
            how="left",
        ).with_columns(
            _safe_divide(
                pl.col("positive_purchase_spend"), pl.col("__customer_positive_spend")
            ).alias("__category_spend_share")
        )
        concentration = category_shares.group_by("customer_id").agg(
            pl.when(pl.col("__customer_positive_spend").first() > 0)
            .then(pl.col("__category_spend_share").pow(2).sum())
            .otherwise(None)
            .alias("category_spend_hhi")
        )
        features = features.join(top_category, on="customer_id", how="left").join(
            concentration, on="customer_id", how="left"
        )
    else:
        features = features.with_columns(
            pl.lit(None, dtype=pl.String).alias("top_category_by_gross_purchase_revenue"),
            pl.lit(None, dtype=pl.Float64).alias("positive_category_spend_total"),
            pl.lit(None, dtype=pl.Float64).alias("category_spend_hhi"),
        )

    department_lines = customer_lines.filter(
        (pl.col("quantity") > 0) & pl.col("department").is_not_null()
    )
    if department_lines.height:
        department_spend = department_lines.group_by(["customer_id", "department"]).agg(
            pl.when(pl.col("__line_revenue") > 0)
            .then(pl.col("__line_revenue"))
            .otherwise(0.0)
            .sum()
            .alias("positive_department_spend"),
            pl.col("__line_revenue").sum().fill_null(0.0).alias("gross_purchase_revenue"),
        )
        top_department = (
            department_spend.sort(
                ["customer_id", "gross_purchase_revenue", "department"],
                descending=[False, True, False],
                nulls_last=True,
            )
            .group_by("customer_id", maintain_order=True)
            .agg(
                pl.col("department").first().alias("top_department_by_gross_purchase_revenue"),
                pl.col("positive_department_spend").sum().alias("positive_department_spend_total"),
            )
        )
        department_shares = department_spend.join(
            department_spend.group_by("customer_id").agg(
                pl.col("positive_department_spend").sum().alias("__customer_positive_department_spend")
            ),
            on="customer_id",
            how="left",
        ).with_columns(
            _safe_divide(
                pl.col("positive_department_spend"), pl.col("__customer_positive_department_spend")
            ).alias("__department_spend_share")
        )
        department_concentration = department_shares.group_by("customer_id").agg(
            pl.when(pl.col("__customer_positive_department_spend").first() > 0)
            .then(pl.col("__department_spend_share").pow(2).sum())
            .otherwise(None)
            .alias("department_spend_hhi")
        )
        features = features.join(top_department, on="customer_id", how="left").join(
            department_concentration, on="customer_id", how="left"
        )
    else:
        features = features.with_columns(
            pl.lit(None, dtype=pl.String).alias("top_department_by_gross_purchase_revenue"),
            pl.lit(None, dtype=pl.Float64).alias("positive_department_spend_total"),
            pl.lit(None, dtype=pl.Float64).alias("department_spend_hhi"),
        )

    if as_of_date is not None:
        features = features.with_columns(
            (pl.lit(as_of_date) - pl.col("last_purchase_date")).dt.total_days().alias("recency_days"),
            (pl.lit(as_of_date) - pl.col("first_purchase_date")).dt.total_days().alias(
                "observed_customer_tenure_days"
            ),
        )
    else:
        features = features.with_columns(
            pl.lit(None, dtype=pl.Int64).alias("recency_days"),
            pl.lit(None, dtype=pl.Int64).alias("observed_customer_tenure_days"),
        )

    features = features.with_columns(
        _safe_divide(
            pl.col("return_line_item_count"),
            pl.col("return_line_item_count") + pl.col("purchase_line_item_count"),
        ).alias("return_line_item_rate"),
        _safe_divide(pl.col("units_returned"), pl.col("units_purchased")).alias(
            "returned_to_purchased_units_ratio"
        ),
        pl.when((pl.col("gross_purchase_revenue") > 0) & (pl.col("return_value") >= 0))
        .then(pl.col("return_value") / pl.col("gross_purchase_revenue"))
        .otherwise(None)
        .alias("return_value_to_gross_purchase_revenue_ratio"),
        _safe_divide(
            pl.col("interpurchase_interval_std_days"), pl.col("mean_interpurchase_days")
        ).alias("interpurchase_cadence_coefficient_of_variation"),
        (pl.col("purchase_transaction_count") >= 2).alias("repeat_purchase_customer_flag"),
    )

    # Rolling windows use inclusive calendar-day windows ending on the as-of date.
    for window in windows_days:
        if as_of_date is None:
            features = features.with_columns(
                pl.lit(0.0).alias(f"net_revenue_{window}d"),
                pl.lit(0.0).alias(f"gross_purchase_revenue_{window}d"),
                pl.lit(0.0).alias(f"return_value_{window}d"),
                pl.lit(0, dtype=pl.Int64).alias(f"purchase_transactions_{window}d"),
            )
            continue
        start = as_of_date - timedelta(days=window - 1)
        win_lines = customer_lines.filter(
            (pl.col("transaction_day") >= pl.lit(start))
            & (pl.col("transaction_day") <= pl.lit(as_of_date))
        )
        if win_lines.height:
            line_win = win_lines.group_by("customer_id").agg(
                pl.col("__line_revenue").sum().fill_null(0.0).alias(f"net_revenue_{window}d"),
                pl.when(pl.col("quantity") > 0)
                .then(pl.col("__line_revenue"))
                .otherwise(None)
                .sum()
                .fill_null(0.0)
                .alias(f"gross_purchase_revenue_{window}d"),
                pl.when(pl.col("quantity") < 0)
                .then(-pl.col("__line_revenue"))
                .otherwise(None)
                .sum()
                .fill_null(0.0)
                .alias(f"return_value_{window}d"),
            )
        else:
            line_win = pl.DataFrame(
                schema={
                    "customer_id": pl.String,
                    f"net_revenue_{window}d": pl.Float64,
                    f"gross_purchase_revenue_{window}d": pl.Float64,
                    f"return_value_{window}d": pl.Float64,
                }
            )
        win_tx = purchase_tx.filter(
            (pl.col("transaction_day") >= pl.lit(start))
            & (pl.col("transaction_day") <= pl.lit(as_of_date))
        ).group_by("customer_id").agg(pl.len().alias(f"purchase_transactions_{window}d"))
        features = features.join(line_win, on="customer_id", how="left").join(
            win_tx, on="customer_id", how="left"
        ).with_columns(
            pl.col(f"net_revenue_{window}d").fill_null(0.0),
            pl.col(f"gross_purchase_revenue_{window}d").fill_null(0.0),
            pl.col(f"return_value_{window}d").fill_null(0.0),
            pl.col(f"purchase_transactions_{window}d").fill_null(0),
        )

    if as_of_date is not None:
        recent_start = as_of_date - timedelta(days=recent_window_days - 1)
        prior_start = as_of_date - timedelta(days=2 * recent_window_days - 1)
        prior_end = as_of_date - timedelta(days=recent_window_days)
        recent = customer_lines.filter(
            (pl.col("transaction_day") >= pl.lit(recent_start))
            & (pl.col("transaction_day") <= pl.lit(as_of_date))
        ).group_by("customer_id").agg(
            pl.when(pl.col("quantity") > 0)
            .then(pl.col("__line_revenue"))
            .otherwise(None)
            .sum()
            .fill_null(0.0)
            .alias("recent_gross_purchase_revenue"),
            pl.col("__line_revenue").sum().fill_null(0.0).alias("recent_net_revenue"),
        )
        prior = customer_lines.filter(
            (pl.col("transaction_day") >= pl.lit(prior_start))
            & (pl.col("transaction_day") <= pl.lit(prior_end))
        ).group_by("customer_id").agg(
            pl.when(pl.col("quantity") > 0)
            .then(pl.col("__line_revenue"))
            .otherwise(None)
            .sum()
            .fill_null(0.0)
            .alias("prior_gross_purchase_revenue"),
            pl.col("__line_revenue").sum().fill_null(0.0).alias("prior_net_revenue"),
        )
        recent_tx = purchase_tx.filter(
            (pl.col("transaction_day") >= pl.lit(recent_start))
            & (pl.col("transaction_day") <= pl.lit(as_of_date))
        ).group_by("customer_id").agg(pl.len().alias("recent_purchase_transactions"))
        prior_tx = purchase_tx.filter(
            (pl.col("transaction_day") >= pl.lit(prior_start))
            & (pl.col("transaction_day") <= pl.lit(prior_end))
        ).group_by("customer_id").agg(pl.len().alias("prior_purchase_transactions"))
        features = (
            features.join(recent, on="customer_id", how="left")
            .join(prior, on="customer_id", how="left")
            .join(recent_tx, on="customer_id", how="left")
            .join(prior_tx, on="customer_id", how="left")
            .with_columns(
                pl.col("recent_gross_purchase_revenue").fill_null(0.0),
                pl.col("recent_net_revenue").fill_null(0.0),
                pl.col("prior_gross_purchase_revenue").fill_null(0.0),
                pl.col("prior_net_revenue").fill_null(0.0),
                pl.col("recent_purchase_transactions").fill_null(0),
                pl.col("prior_purchase_transactions").fill_null(0),
            )
            .with_columns(
                pl.when(pl.col("prior_gross_purchase_revenue") > 0)
                .then(
                    (pl.col("recent_gross_purchase_revenue") - pl.col("prior_gross_purchase_revenue"))
                    / pl.col("prior_gross_purchase_revenue")
                )
                .otherwise(None)
                .alias("recent_gross_purchase_revenue_change_pct_if_prior_positive"),
                pl.when(pl.col("prior_purchase_transactions") > 0)
                .then(
                    (pl.col("recent_purchase_transactions") - pl.col("prior_purchase_transactions"))
                    / pl.col("prior_purchase_transactions")
                )
                .otherwise(None)
                .alias("recent_purchase_transaction_change_pct_if_prior_positive"),
            )
        )
    else:
        features = features.with_columns(
            pl.lit(0.0).alias("recent_gross_purchase_revenue"),
            pl.lit(0.0).alias("recent_net_revenue"),
            pl.lit(0.0).alias("prior_gross_purchase_revenue"),
            pl.lit(0.0).alias("prior_net_revenue"),
            pl.lit(0, dtype=pl.Int64).alias("recent_purchase_transactions"),
            pl.lit(0, dtype=pl.Int64).alias("prior_purchase_transactions"),
            pl.lit(None, dtype=pl.Float64).alias("recent_gross_purchase_revenue_change_pct_if_prior_positive"),
            pl.lit(None, dtype=pl.Float64).alias("recent_purchase_transaction_change_pct_if_prior_positive"),
        )

    if as_of_date is not None:
        recency = features.get_column("recency_days").cast(pl.Float64).to_numpy()
        cadence_median = features.get_column("median_interpurchase_days").cast(pl.Float64).to_numpy()
        buyers = features.get_column("purchase_transaction_count").to_numpy() > 0
        threshold = np.full(features.height, np.nan, dtype=float)
        cadence_available = np.isfinite(cadence_median) & (cadence_median > 0)
        threshold[buyers & cadence_available] = np.maximum(
            inactivity_min_days,
            cadence_median[buyers & cadence_available] * inactivity_cadence_multiplier,
        )
        threshold[buyers & ~cadence_available] = inactivity_fallback_days
        valid_flag = buyers & np.isfinite(recency) & np.isfinite(threshold)
        flag_values = np.full(features.height, None, dtype=object)
        flag_values[valid_flag] = (recency[valid_flag] > threshold[valid_flag])
        features = features.with_columns(
            pl.Series("inactivity_heuristic_threshold_days", threshold).fill_nan(None),
            pl.Series("inactivity_risk_flag", flag_values.tolist(), dtype=pl.Boolean),
            pl.lit("Cadence heuristic only; not a calibrated churn probability").alias(
                "inactivity_indicator_interpretation"
            ),
        )
    else:
        features = features.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("inactivity_heuristic_threshold_days"),
            pl.lit(None, dtype=pl.Boolean).alias("inactivity_risk_flag"),
            pl.lit("No valid analysis as-of date; inactivity not evaluated").alias(
                "inactivity_indicator_interpretation"
            ),
        )

    if as_of_date is not None and purchase_tx.height:
        cohort_customer = purchase_tx.group_by("customer_id").agg(
            pl.col("transaction_day").min().dt.truncate("1mo").alias("observed_first_purchase_cohort_month")
        )
        features = features.join(cohort_customer, on="customer_id", how="left")
    else:
        features = features.with_columns(
            pl.lit(None, dtype=pl.Date).alias("observed_first_purchase_cohort_month")
        )

    return features.sort("customer_id"), intervals


def assign_rfm_segments(features: pl.DataFrame) -> pl.DataFrame:
    """Add tied-rank RFM quintiles and an interpretable rule-based segment."""
    if features.height == 0:
        return features.with_columns(
            pl.lit(None, dtype=pl.Int8).alias("rfm_recency_score"),
            pl.lit(None, dtype=pl.Int8).alias("rfm_frequency_score"),
            pl.lit(None, dtype=pl.Int8).alias("rfm_monetary_score"),
            pl.lit(None, dtype=pl.String).alias("rfm_score"),
            pl.lit(None, dtype=pl.String).alias("rfm_segment"),
        )
    buyer_mask = features.get_column("purchase_transaction_count").to_numpy() > 0
    recency = features.get_column("recency_days").cast(pl.Float64).to_numpy()
    frequency = features.get_column("purchase_transaction_count").cast(pl.Float64).to_numpy()
    monetary = features.get_column("net_revenue").cast(pl.Float64).to_numpy()
    n = features.height
    r_full = np.full(n, np.nan, dtype=float)
    f_full = np.full(n, np.nan, dtype=float)
    m_full = np.full(n, np.nan, dtype=float)
    label_values = np.full(n, None, dtype=object)
    score_values = np.full(n, None, dtype=object)
    indices = np.flatnonzero(buyer_mask & np.isfinite(recency) & np.isfinite(frequency) & np.isfinite(monetary))
    if indices.size:
        r = _rank_quintiles(recency[indices], higher_is_better=False)
        f = _rank_quintiles(frequency[indices], higher_is_better=True)
        m = _rank_quintiles(monetary[indices], higher_is_better=True)
        labels = _rfm_label(r, f, m)
        r_full[indices], f_full[indices], m_full[indices] = r, f, m
        label_values[indices] = labels.astype(object)
        score_values[indices] = np.char.add(np.char.add(r.astype(str), f.astype(str)), m.astype(str)).astype(object)
    return features.with_columns(
        pl.Series("rfm_recency_score", r_full).fill_nan(None).cast(pl.Int8),
        pl.Series("rfm_frequency_score", f_full).fill_nan(None).cast(pl.Int8),
        pl.Series("rfm_monetary_score", m_full).fill_nan(None).cast(pl.Int8),
        pl.Series("rfm_score", score_values.tolist(), dtype=pl.String),
        pl.Series("rfm_segment", label_values.tolist(), dtype=pl.String),
    )



def add_customer_revenue_concentration(
    features: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Add customer spend shares and summarize concentration among observed buyers."""
    summary_schema = {
        "scope": pl.String,
        "customers_with_identifiable_positive_purchase_transaction": pl.Int64,
        "positive_purchase_spend_total": pl.Float64,
        "top_1pct_customer_spend_share": pl.Float64,
        "top_5pct_customer_spend_share": pl.Float64,
        "top_10pct_customer_spend_share": pl.Float64,
        "customer_spend_hhi": pl.Float64,
        "customer_spend_gini": pl.Float64,
        "concentration_basis": pl.String,
    }
    if features.height == 0:
        return features.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("positive_gross_purchase_spend_share_of_observed_buyers")
        ), pl.DataFrame(schema=summary_schema)

    frequency = features.get_column("purchase_transaction_count").cast(pl.Float64).fill_null(0).to_numpy()
    spend_all = features.get_column("positive_purchase_spend").cast(pl.Float64).fill_null(0).to_numpy()
    buyer_mask = frequency > 0
    shares = np.full(features.height, np.nan, dtype=float)
    buyer_spend = np.clip(spend_all[buyer_mask], 0.0, None)
    total = float(buyer_spend.sum())
    if total > 0:
        shares[buyer_mask] = buyer_spend / total
        sorted_spend = np.sort(buyer_spend)
        n = int(sorted_spend.size)
        def top_share(fraction: float) -> float:
            k = max(1, int(math.ceil(n * fraction)))
            return float(sorted_spend[-k:].sum() / total)
        positions = np.arange(1, n + 1, dtype=float)
        gini = float((2.0 * np.dot(positions, sorted_spend) / (n * total)) - ((n + 1.0) / n)) if n else None
        hhi = float(np.square(buyer_spend / total).sum())
        top_1 = top_share(0.01)
        top_5 = top_share(0.05)
        top_10 = top_share(0.10)
    else:
        gini = hhi = top_1 = top_5 = top_10 = None

    features = features.with_columns(
        pl.Series("positive_gross_purchase_spend_share_of_observed_buyers", shares).fill_nan(None)
    )
    summary = pl.DataFrame(
        [
            {
                "scope": "customers_with_identifiable_positive_purchase_transaction",
                "customers_with_identifiable_positive_purchase_transaction": int(buyer_mask.sum()),
                "positive_purchase_spend_total": total,
                "top_1pct_customer_spend_share": top_1,
                "top_5pct_customer_spend_share": top_5,
                "top_10pct_customer_spend_share": top_10,
                "customer_spend_hhi": hhi,
                "customer_spend_gini": gini,
                "concentration_basis": "Positive line-item revenue for positive-quantity lines; zero/negative line-item revenue excluded from spend shares; customers are those with at least one identifiable positive-purchase transaction.",
            }
        ],
        schema=summary_schema,
    )
    return features, summary

def _cluster_matrix(features: pl.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Create interpretable transformed numeric inputs for customer clustering."""
    recency = features.get_column("recency_days").cast(pl.Float64).fill_null(0).to_numpy()
    frequency = features.get_column("purchase_transaction_count").cast(pl.Float64).fill_null(0).to_numpy()
    gross = features.get_column("gross_purchase_revenue").cast(pl.Float64).fill_null(0).to_numpy()
    product_count = features.get_column("unique_products_purchased").cast(pl.Float64).fill_null(0).to_numpy()
    category_count = features.get_column("unique_categories_purchased").cast(pl.Float64).fill_null(0).to_numpy()
    tenure = features.get_column("observed_customer_tenure_days").cast(pl.Float64).fill_null(0).to_numpy()
    return_rate = features.get_column("return_line_item_rate").cast(pl.Float64).fill_null(0).to_numpy()
    matrix = np.column_stack(
        [
            np.log1p(np.clip(recency, 0, None)),
            np.log1p(np.clip(frequency, 0, None)),
            np.log1p(np.clip(gross, 0, None)),
            np.log1p(np.clip(product_count, 0, None)),
            np.log1p(np.clip(category_count, 0, None)),
            np.log1p(np.clip(tenure, 0, None)),
            np.clip(return_rate, 0, 1),
        ]
    )
    return matrix, [
        "log1p_recency_days",
        "log1p_purchase_transaction_count",
        "log1p_nonnegative_gross_purchase_revenue",
        "log1p_unique_products_purchased",
        "log1p_unique_categories_purchased",
        "log1p_observed_tenure_days",
        "return_line_item_rate_clipped_0_1",
    ]


def cluster_customers(
    features: pl.DataFrame,
    random_seed: int,
    max_train_customers: int,
    max_evaluation_sample: int,
    max_clusters: int,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Fit a scaled K-means segmentation when sample size and features permit."""
    n = features.height
    features = features.with_columns(
        pl.lit(None, dtype=pl.Int32).alias("cluster_id"),
        pl.lit(None, dtype=pl.String).alias("cluster_label"),
    )
    nonbuyers = pl.col("purchase_transaction_count") <= 0
    features = features.with_columns(
        pl.when(nonbuyers)
        .then(pl.lit("No observed identifiable purchase transaction"))
        .otherwise(pl.lit("Unclustered: model skipped or insufficient data"))
        .alias("cluster_label")
    )
    buyers = features.filter(pl.col("purchase_transaction_count") > 0)
    base_diagnostics = {
        "model": "KMeans on robust-scaled log-transformed customer features",
        "status": "skipped",
        "reason": "",
        "candidate_k": None,
        "silhouette_score": None,
        "inertia": None,
        "training_customers": 0,
        "customers_assigned": 0,
        "features_used": "",
    }
    if not SKLEARN_AVAILABLE:
        base_diagnostics["reason"] = "scikit-learn is not installed; descriptive outputs still generated"
        diag = pl.DataFrame([base_diagnostics])
        return features, diag, empty_cluster_profile()
    if buyers.height < 10:
        base_diagnostics["reason"] = f"only {buyers.height} customers have an identifiable positive-purchase transaction; at least 10 required"
        diag = pl.DataFrame([base_diagnostics])
        return features, diag, empty_cluster_profile()

    matrix, feature_names = _cluster_matrix(buyers)
    rng = np.random.default_rng(random_seed)
    if buyers.height > max_train_customers:
        sampled_positions = np.sort(rng.choice(buyers.height, size=max_train_customers, replace=False))
        train_matrix_raw = matrix[sampled_positions]
    else:
        train_matrix_raw = matrix
    medians = np.nanmedian(train_matrix_raw, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    train_matrix_raw = np.where(np.isfinite(train_matrix_raw), train_matrix_raw, medians)
    matrix = np.where(np.isfinite(matrix), matrix, medians)
    variable = np.nanstd(train_matrix_raw, axis=0) > 1e-10
    if not np.any(variable):
        base_diagnostics["reason"] = "all clustering features are constant in the training sample"
        base_diagnostics["training_customers"] = int(train_matrix_raw.shape[0])
        base_diagnostics["features_used"] = ",".join(feature_names)
        return features, pl.DataFrame([base_diagnostics]), empty_cluster_profile()
    train_matrix_raw = train_matrix_raw[:, variable]
    matrix_used = matrix[:, variable]
    used_names = [name for name, keep in zip(feature_names, variable) if keep]
    scaler = RobustScaler()
    train_scaled = scaler.fit_transform(train_matrix_raw)
    all_scaled = scaler.transform(matrix_used)
    if train_scaled.shape[0] < 3:
        base_diagnostics["reason"] = "fewer than three customers remain in clustering training sample"
        return features, pl.DataFrame([base_diagnostics]), empty_cluster_profile()

    evaluation_n = min(max_evaluation_sample, train_scaled.shape[0])
    eval_idx = np.sort(rng.choice(train_scaled.shape[0], size=evaluation_n, replace=False))
    eval_matrix = train_scaled[eval_idx]
    max_k = min(max_clusters, train_scaled.shape[0] - 1, evaluation_n - 1)
    diagnostic_rows: list[dict[str, Any]] = []
    fitted: dict[int, Any] = {}
    for k in range(2, max_k + 1):
        try:
            model = KMeans(n_clusters=k, random_state=random_seed, n_init=10, max_iter=300)
            model.fit(train_scaled)
            eval_labels = model.predict(eval_matrix)
            unique_labels = np.unique(eval_labels)
            if 1 < unique_labels.size < evaluation_n:
                score = float(silhouette_score(eval_matrix, eval_labels))
            else:
                score = None
            diagnostic_rows.append(
                {
                    **base_diagnostics,
                    "status": "candidate",
                    "reason": None if score is not None else "silhouette undefined for sampled labels",
                    "candidate_k": int(k),
                    "silhouette_score": score,
                    "inertia": float(model.inertia_),
                    "training_customers": int(train_scaled.shape[0]),
                    "customers_assigned": int(buyers.height),
                    "features_used": ",".join(used_names),
                }
            )
            fitted[k] = model
        except (ValueError, FloatingPointError) as exc:
            diagnostic_rows.append(
                {
                    **base_diagnostics,
                    "status": "candidate_failed",
                    "reason": str(exc),
                    "candidate_k": int(k),
                    "training_customers": int(train_scaled.shape[0]),
                    "customers_assigned": int(buyers.height),
                    "features_used": ",".join(used_names),
                }
            )
    valid_scores = [row for row in diagnostic_rows if row.get("silhouette_score") is not None]
    if not valid_scores:
        base_diagnostics["reason"] = "no candidate k produced a valid silhouette score"
        base_diagnostics["training_customers"] = int(train_scaled.shape[0])
        base_diagnostics["features_used"] = ",".join(used_names)
        return features, pl.DataFrame(diagnostic_rows + [base_diagnostics]), empty_cluster_profile()
    best_row = max(valid_scores, key=lambda row: (float(row["silhouette_score"]), -int(row["candidate_k"])))
    best_k = int(best_row["candidate_k"])
    best_model = fitted[best_k]
    buyer_labels = best_model.predict(all_scaled).astype(np.int32)
    buyer_ids = buyers.get_column("customer_id").to_list()
    assignment = pl.DataFrame(
        {
            "customer_id": buyer_ids,
            "cluster_id": buyer_labels,
        }
    )
    # Cluster labels are descriptive summaries, not externally validated personas.
    buyer_assigned = buyers.drop("cluster_id", "cluster_label").join(
        assignment, on="customer_id", how="left"
    )
    cluster_stats = buyer_assigned.group_by("cluster_id").agg(
        pl.len().alias("customer_count"),
        pl.col("recency_days").median().alias("median_recency_days"),
        pl.col("purchase_transaction_count").median().alias("median_purchase_transaction_count"),
        pl.col("gross_purchase_revenue").median().alias("median_gross_purchase_revenue"),
        pl.col("gross_purchase_revenue").mean().alias("mean_gross_purchase_revenue"),
    ).sort("cluster_id")
    med_recency = float(buyers.get_column("recency_days").median() or 0.0)
    med_freq = float(buyers.get_column("purchase_transaction_count").median() or 0.0)
    med_spend = float(buyers.get_column("gross_purchase_revenue").median() or 0.0)
    label_map: dict[int, str] = {}
    for row in cluster_stats.iter_rows(named=True):
        recent = float(row["median_recency_days"] or 0.0) <= med_recency
        frequent = float(row["median_purchase_transaction_count"] or 0.0) >= med_freq
        high_spend = float(row["median_gross_purchase_revenue"] or 0.0) >= med_spend
        if recent and frequent and high_spend:
            label = "Recent / frequent / higher spend"
        elif frequent and high_spend:
            label = "Frequent / higher spend"
        elif not recent and high_spend:
            label = "Lapsed / higher historical spend"
        elif frequent:
            label = "Frequent / moderate spend"
        elif recent and high_spend:
            label = "Recent / higher spend"
        elif not recent:
            label = "Less recent / occasional"
        else:
            label = "Recent / developing"
        label_map[int(row["cluster_id"])] = label
    used_labels: Counter[str] = Counter()
    for cluster_id in sorted(label_map):
        used_labels[label_map[cluster_id]] += 1
        if used_labels[label_map[cluster_id]] > 1:
            label_map[cluster_id] = f"{label_map[cluster_id]} (cluster {cluster_id})"

    features = features.drop("cluster_id", "cluster_label").join(assignment, on="customer_id", how="left")
    label_expr = pl.col("cluster_id").replace_strict(
        label_map,
        default="Unclustered: model skipped or insufficient data",
        return_dtype=pl.String,
    )
    features = features.with_columns(
        pl.when(pl.col("cluster_id").is_not_null())
        .then(label_expr)
        .otherwise(
            pl.when(pl.col("purchase_transaction_count") <= 0)
            .then(pl.lit("No observed identifiable purchase transaction"))
            .otherwise(pl.lit("Unclustered: model skipped or insufficient data"))
        )
        .alias("cluster_label")
    )
    diagnostics: list[dict[str, Any]] = []
    for row in diagnostic_rows:
        row = dict(row)
        row["selected"] = row.get("candidate_k") == best_k
        if row.get("candidate_k") == best_k:
            row["status"] = "selected"
        diagnostics.append(row)
    profiles = cluster_stats.with_columns(
        pl.col("cluster_id").replace_strict(label_map, default="Unlabelled", return_dtype=pl.String).alias(
            "cluster_label"
        )
    )
    return features.sort("customer_id"), pl.DataFrame(diagnostics), profiles.sort("cluster_id")


def empty_cluster_profile() -> pl.DataFrame:
    """Return a schema-stable empty cluster profile table."""
    return pl.DataFrame(
        schema={
            "cluster_id": pl.Int32,
            "customer_count": pl.Int64,
            "median_recency_days": pl.Float64,
            "median_purchase_transaction_count": pl.Float64,
            "median_gross_purchase_revenue": pl.Float64,
            "mean_gross_purchase_revenue": pl.Float64,
            "cluster_label": pl.String,
        }
    )


def build_rfm_profiles(features: pl.DataFrame) -> pl.DataFrame:
    """Summarize assigned RFM segments without implying causal effects."""
    if features.height == 0 or "rfm_segment" not in features.columns:
        return pl.DataFrame(
            schema={
                "rfm_segment": pl.String,
                "customer_count": pl.UInt32,
                "median_recency_days": pl.Float64,
                "mean_purchase_transaction_count": pl.Float64,
                "median_net_revenue": pl.Float64,
                "mean_gross_purchase_revenue": pl.Float64,
                "repeat_purchase_customer_rate": pl.Float64,
                "inactivity_heuristic_rate": pl.Float64,
            }
        )
    base = features.filter(pl.col("rfm_segment").is_not_null())
    if base.height == 0:
        return build_rfm_profiles(pl.DataFrame())
    return base.group_by("rfm_segment").agg(
        pl.len().alias("customer_count"),
        pl.col("recency_days").median().alias("median_recency_days"),
        pl.col("purchase_transaction_count").mean().alias("mean_purchase_transaction_count"),
        pl.col("net_revenue").median().alias("median_net_revenue"),
        pl.col("gross_purchase_revenue").mean().alias("mean_gross_purchase_revenue"),
        (pl.col("purchase_transaction_count") >= 2).cast(pl.Float64).mean().alias("repeat_purchase_customer_rate"),
        pl.col("inactivity_risk_flag").cast(pl.Float64).mean().alias("inactivity_heuristic_rate"),
    ).sort("customer_count", descending=True)


def _empty_affinity() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "basket_level": pl.String,
            "antecedent": pl.String,
            "consequent": pl.String,
            "antecedent_count": pl.Int64,
            "consequent_count": pl.Int64,
            "pair_count": pl.Int64,
            "n_baskets_analyzed": pl.Int64,
            "support": pl.Float64,
            "confidence": pl.Float64,
            "lift": pl.Float64,
            "association_interpretation": pl.String,
        }
    )


def basket_affinity(
    time_lines: pl.DataFrame,
    basket_level: str,
    transaction_key_mode: str,
    min_support: float,
    min_confidence: float,
    min_lift: float,
    max_items: int,
    max_baskets: int,
    max_items_per_basket: int,
    max_pair_operations: int,
    random_seed: int,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Calculate bounded support, confidence, and lift for unordered item pairs."""
    key_columns = ["customer_id", "transaction_id"]
    if transaction_key_mode == "transaction-only":
        key_columns = ["transaction_id"]
    item_column = "category" if basket_level == "category" else "product_id"
    details: dict[str, Any] = {
        "status": "skipped",
        "reason": None,
        "basket_level": basket_level,
        "eligible_purchase_baskets": 0,
        "baskets_sampled": 0,
        "baskets_analyzed": 0,
        "candidate_items": 0,
        "pair_operations": 0,
        "max_pair_operations": max_pair_operations,
    }
    lines = time_lines.filter(
        (pl.col("quantity") > 0)
        & pl.col(item_column).is_not_null()
        & pl.all_horizontal([pl.col(key).is_not_null() for key in key_columns])
    ).select(key_columns + [item_column]).unique()
    all_purchase_keys = time_lines.filter(
        (pl.col("quantity") > 0)
        & pl.all_horizontal([pl.col(key).is_not_null() for key in key_columns])
    ).select(key_columns).unique()
    details["eligible_purchase_baskets"] = all_purchase_keys.height
    if all_purchase_keys.height == 0 or lines.height == 0:
        details["reason"] = "no identifiable positive-purchase baskets with the selected item field"
        return _empty_affinity(), details

    key_sample = (
        all_purchase_keys.with_columns(pl.struct(key_columns).hash(seed=int(random_seed)).alias("__sample_hash"))
        .sort("__sample_hash")
        .head(max_baskets)
    )
    details["baskets_sampled"] = key_sample.height
    sampled_items = lines.join(key_sample, on=key_columns, how="inner")
    item_support_initial = sampled_items.group_by(item_column).agg(
        pl.len().alias("__initial_item_basket_count")
    )
    top_items = (
        item_support_initial.sort(
            ["__initial_item_basket_count", item_column], descending=[True, False]
        )
        .head(max_items)
        .rename({item_column: "__item"})
    )
    details["candidate_items"] = top_items.height
    if top_items.height < 2:
        details["reason"] = "fewer than two distinct candidate items in sampled baskets"
        return _empty_affinity(), details

    ranked = (
        sampled_items.rename({item_column: "__item"})
        .join(top_items.select("__item", "__initial_item_basket_count"), on="__item", how="inner")
        .sort(["__sample_hash", "__initial_item_basket_count", "__item"], descending=[False, True, False])
    )
    item_lists = ranked.group_by(key_columns, maintain_order=True).agg(
        pl.col("__item").head(max_items_per_basket).alias("__items")
    )
    bounded_baskets = (
        key_sample.join(item_lists, on=key_columns, how="left")
        .with_columns(pl.col("__items").list.len().fill_null(0).alias("__item_count"))
        .with_columns(
            (pl.col("__item_count") * (pl.col("__item_count") - 1) // 2).alias("__pair_cost")
        )
        .sort("__sample_hash")
        .with_columns(pl.col("__pair_cost").cum_sum().alias("__cumulative_pair_cost"))
        .filter(pl.col("__cumulative_pair_cost") <= max_pair_operations)
    )
    details["baskets_analyzed"] = bounded_baskets.height
    details["pair_operations"] = int(bounded_baskets.get_column("__pair_cost").sum() or 0) if bounded_baskets.height else 0
    if bounded_baskets.height == 0:
        details["reason"] = "pair-operation budget is too small for the first sampled basket"
        return _empty_affinity(), details

    item_counts: Counter[str] = Counter()
    pair_counts: Counter[tuple[str, str]] = Counter()
    analyzed_n = bounded_baskets.height
    for item_row in bounded_baskets.select("__items").iter_rows():
        items = item_row[0] or []
        normalized_items = [str(item) for item in items]
        for item in normalized_items:
            item_counts[item] += 1
        for left, right in combinations(normalized_items, 2):
            pair = (left, right) if left < right else (right, left)
            pair_counts[pair] += 1
    detail_warning = (
        "A deterministic hash-ordered basket subset and per-basket item cap were used. "
        "Supports describe the analyzed subset, not necessarily the complete input."
    )
    details.update(
        {
            "status": "completed" if pair_counts else "skipped",
            "reason": None if pair_counts else "no co-purchase pairs remained after caps",
            "association_interpretation": detail_warning,
        }
    )
    if not pair_counts or analyzed_n == 0:
        return _empty_affinity(), details

    out: list[dict[str, Any]] = []
    for (left, right), count in pair_counts.items():
        support = count / analyzed_n
        if support < min_support:
            continue
        left_count = item_counts[left]
        right_count = item_counts[right]
        if left_count == 0 or right_count == 0:
            continue
        left_support = left_count / analyzed_n
        right_support = right_count / analyzed_n
        rules = [
            (left, right, count / left_count, (count / left_count) / right_support),
            (right, left, count / right_count, (count / right_count) / left_support),
        ]
        for antecedent, consequent, confidence, lift in rules:
            if confidence >= min_confidence and lift >= min_lift:
                out.append(
                    {
                        "basket_level": basket_level,
                        "antecedent": antecedent,
                        "consequent": consequent,
                        "antecedent_count": item_counts[antecedent],
                        "consequent_count": item_counts[consequent],
                        "pair_count": int(count),
                        "n_baskets_analyzed": int(analyzed_n),
                        "support": float(support),
                        "confidence": float(confidence),
                        "lift": float(lift),
                        "association_interpretation": "Observational association; not causal and based on the analyzed basket subset",
                    }
                )
    if not out:
        details["reason"] = "no rules met the configured support, confidence, and lift thresholds"
        return _empty_affinity(), details
    result = pl.DataFrame(out).sort(
        ["lift", "confidence", "support", "pair_count"], descending=[True, True, True, True]
    )
    return result, details


def monthly_and_weekday_summaries(
    time_lines: pl.DataFrame, customer_transactions: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Summarize dated activity at month and ISO weekday grain."""
    monthly_lines = time_lines.with_columns(
        pl.col("transaction_day").dt.truncate("1mo").alias("month")
    ).group_by("month").agg(line_financial_aggregations())
    dated_tx = customer_transactions.filter(pl.col("transaction_day").is_not_null())
    if dated_tx.height:
        monthly_tx = dated_tx.with_columns(
            pl.col("transaction_day").dt.truncate("1mo").alias("month")
        ).group_by("month").agg(
            pl.len().alias("identified_transactions"),
            (pl.col("units_purchased") > 0).cast(pl.Int64).sum().alias("purchase_transactions"),
            (pl.col("units_purchased") <= 0).cast(pl.Int64).sum().alias("non_purchase_transactions"),
        )
    else:
        monthly_tx = pl.DataFrame(
            schema={
                "month": pl.Date,
                "identified_transactions": pl.Int64,
                "purchase_transactions": pl.Int64,
                "non_purchase_transactions": pl.Int64,
            }
        )
    monthly = monthly_lines.join(monthly_tx, on="month", how="left").with_columns(
        pl.col("identified_transactions").fill_null(0),
        pl.col("purchase_transactions").fill_null(0),
        pl.col("non_purchase_transactions").fill_null(0),
    ).sort("month")
    weekdays = time_lines.with_columns(
        pl.col("transaction_day").dt.weekday().alias("iso_weekday_monday_1_sunday_7")
    ).group_by("iso_weekday_monday_1_sunday_7").agg(
        line_financial_aggregations()
    ).sort("iso_weekday_monday_1_sunday_7")
    return monthly, weekdays


def table_to_csv_compatible(frame: pl.DataFrame) -> pl.DataFrame:
    """Cast nested columns to strings for optional CSV output."""
    expressions: list[pl.Expr] = []
    for name, dtype in frame.schema.items():
        if str(dtype).startswith(("List(", "Struct(", "Array(")):
            expressions.append(pl.col(name).cast(pl.String).alias(name))
    return frame.with_columns(expressions) if expressions else frame


def write_table(
    frame: pl.DataFrame, output_dir: Path, stem: str, write_csv: bool, written: list[str]
) -> None:
    """Write a tabular artifact, deterministically overwriting known filenames."""
    parquet_path = output_dir / f"{stem}.parquet"
    frame.write_parquet(parquet_path, compression="zstd")
    written.append(parquet_path.name)
    if write_csv:
        csv_path = output_dir / f"{stem}.csv"
        table_to_csv_compatible(frame).write_csv(csv_path)
        written.append(csv_path.name)


def json_safe(value: Any) -> Any:
    """Convert common NumPy/date/path values to JSON-serializable values."""
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _number(frame: pl.DataFrame, column: str, default: float = 0.0) -> float:
    if column not in frame.columns or frame.height == 0:
        return default
    result = frame.select(pl.col(column).sum()).item()
    return float(result or 0.0)


def _fmt_number(value: Any, digits: int = 2) -> str:
    if value is None:
        return "not available"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "not available"
    return f"{number:,.{digits}f}"


def generate_report(
    output_dir: Path,
    quality: dict[str, Any],
    cleaned: pl.DataFrame,
    customer_features: pl.DataFrame,
    customer_concentration_summary: pl.DataFrame,
    category_summary: pl.DataFrame,
    rfm_profiles: pl.DataFrame,
    cluster_profiles: pl.DataFrame,
    affinity: pl.DataFrame,
    affinity_details: dict[str, Any],
    cohort_summary: pl.DataFrame,
    retention: pl.DataFrame,
    as_of_date: date | None,
    config: dict[str, Any],
    skipped_analyses: list[dict[str, str]],
    warnings: list[str],
) -> None:
    """Render a Markdown report from generated tables and observed values."""
    if cleaned.height:
        line_totals = cleaned.select(
            pl.col("__line_revenue").sum().fill_null(0.0).alias("net"),
            pl.when(pl.col("quantity") > 0)
            .then(pl.col("__line_revenue"))
            .otherwise(None)
            .sum()
            .fill_null(0.0)
            .alias("gross"),
            pl.when(pl.col("quantity") < 0)
            .then(-pl.col("__line_revenue"))
            .otherwise(None)
            .sum()
            .fill_null(0.0)
            .alias("returns"),
            pl.col("__line_revenue").is_not_null().cast(pl.Int64).sum().alias("valid_revenue_lines"),
        ).row(0, named=True)
        gross = float(line_totals["gross"])
        returns = float(line_totals["returns"])
        net = float(line_totals["net"])
        valid_revenue_lines = int(line_totals["valid_revenue_lines"] or 0)
    else:
        gross = returns = net = 0.0
        valid_revenue_lines = 0
    buyers = (
        customer_features.filter(pl.col("purchase_transaction_count") > 0).height
        if customer_features.height
        else 0
    )
    repeat_customers = (
        customer_features.filter(pl.col("purchase_transaction_count") >= 2).height
        if customer_features.height
        else 0
    )
    inactive_buyers = (
        customer_features.filter(pl.col("inactivity_risk_flag") == True).height
        if customer_features.height and "inactivity_risk_flag" in customer_features.columns
        else 0
    )
    lines: list[str] = [
        "# Retail transaction analytics report",
        "",
        f"**Analysis as-of date:** {as_of_date.isoformat() if as_of_date else 'not available (no valid dates found)'}  ",
        f"**Cleaned line items:** {cleaned.height:,}  ",
        f"**Customer rows in feature table:** {customer_features.height:,}  ",
        f"**Customers with an identifiable positive-purchase transaction:** {buyers:,}",
        "",
        "## Executive description",
        "",
        f"Gross purchase revenue calculated from positive-quantity line items is **{_fmt_number(gross)}**. Return value is **{_fmt_number(returns)}**, and net revenue (signed sum of calculable line-item revenue) is **{_fmt_number(net)}**. These are revenue measures, not profit or margin.",
        f"There are **{repeat_customers:,}** observed customers with at least two identifiable positive-purchase transactions and **{inactive_buyers:,}** purchasers above the configured cadence-based inactivity threshold. The latter is a heuristic flag, not a churn probability.",
        f"The cleaned dataset contains {valid_revenue_lines:,} line items for which `price * quantity` is calculable. The accounting identities and the exclusions that limit coverage are detailed below.",
        "",
        "Customer revenue concentration is calculated among customers with at least one identifiable positive-purchase transaction, using only positive line-item revenue on positive-quantity lines. This avoids allowing anomalous negative prices to produce negative concentration shares.",
    ]
    if customer_concentration_summary.height:
        concentration = customer_concentration_summary.row(0, named=True)
        lines.extend([
            f"Top 1% / 5% / 10% customer spend shares: {_fmt_number(100 * concentration['top_1pct_customer_spend_share'], 2) if concentration['top_1pct_customer_spend_share'] is not None else 'not available'}% / "
            f"{_fmt_number(100 * concentration['top_5pct_customer_spend_share'], 2) if concentration['top_5pct_customer_spend_share'] is not None else 'not available'}% / "
            f"{_fmt_number(100 * concentration['top_10pct_customer_spend_share'], 2) if concentration['top_10pct_customer_spend_share'] is not None else 'not available'}%; "
            f"customer spend HHI {_fmt_number(concentration['customer_spend_hhi'], 4)}; Gini {_fmt_number(concentration['customer_spend_gini'], 4)}."
        ])
    lines.extend([
        "",
        "## Category overview",
        "",
    ])
    if category_summary.height:
        lines.append("Top categories by gross purchase revenue (line-item grain; returns excluded from this ranking):")
        for row in category_summary.head(5).iter_rows(named=True):
            label = row.get("category") if row.get("category") is not None else "[missing category]"
            lines.append(
                f"- **{label}** — gross purchase revenue {_fmt_number(row.get('gross_purchase_revenue'))}; "
                f"net revenue {_fmt_number(row.get('net_revenue'))}; {int(row.get('line_item_count') or 0):,} line items."
            )
    else:
        lines.append("No category summary rows were available.")
    lines.extend(["", "## Customer segmentation", ""])
    if rfm_profiles.height:
        lines.append("RFM profile counts and behavior summaries:")
        for row in rfm_profiles.iter_rows(named=True):
            lines.append(
                f"- **{row['rfm_segment']}** — {int(row['customer_count']):,} customers; "
                f"median recency {_fmt_number(row.get('median_recency_days'), 1)} days; "
                f"mean purchase transactions {_fmt_number(row.get('mean_purchase_transaction_count'))}; "
                f"median observed net revenue {_fmt_number(row.get('median_net_revenue'))}."
            )
        lines.append("")
        lines.append(
            "RFM quintiles are tied-rank scores from 1 to 5: lower recency is better; higher purchase-transaction frequency and net revenue receive higher scores. Net revenue can be negative when observed returns exceed purchases. Customers without an identifiable positive-purchase transaction are not assigned an RFM label."
        )
    else:
        lines.append("RFM segments were not assigned because no eligible customer purchase histories were available.")
    lines.extend(["", "### K-means cluster profiles", ""])
    if cluster_profiles.height:
        for row in cluster_profiles.iter_rows(named=True):
            lines.append(
                f"- **Cluster {row['cluster_id']} — {row.get('cluster_label', 'Unlabelled')}**: "
                f"{int(row['customer_count']):,} customers; median recency {_fmt_number(row.get('median_recency_days'), 1)} days; "
                f"median purchase transactions {_fmt_number(row.get('median_purchase_transaction_count'), 1)}; "
                f"median gross purchase revenue {_fmt_number(row.get('median_gross_purchase_revenue'))}."
            )
        lines.append("")
        lines.append(
            "Clustering uses log-transformed nonnegative values for recency, purchase frequency, gross purchase revenue, product/category diversity, and observed tenure, plus a clipped return-line rate. Robust scaling and a fixed random seed are used. The selected k is based on the largest valid silhouette score among the tested candidates; cluster labels are descriptive and should be checked for stability before operational use."
        )
    else:
        lines.append("Clustering was skipped or produced no usable profiles; see `model_diagnostics.parquet` for the exact reason.")
    lines.extend(["", "## Basket affinity", ""])
    lines.append(
        f"Basket level: **{affinity_details.get('basket_level')}**. Status: **{affinity_details.get('status')}**. "
        f"Identifiable positive-purchase baskets in the dated cutoff subset: {int(affinity_details.get('eligible_purchase_baskets') or 0):,}; "
        f"baskets analyzed: {int(affinity_details.get('baskets_analyzed') or 0):,}; "
        f"pair operations: {int(affinity_details.get('pair_operations') or 0):,} of a configured maximum {int(affinity_details.get('max_pair_operations') or 0):,}."
    )
    if affinity.height:
        lines.append("Rules meeting the configured support, confidence, and lift thresholds (top 10 by lift):")
        for row in affinity.head(10).iter_rows(named=True):
            lines.append(
                f"- `{row['antecedent']}` → `{row['consequent']}` — support {_fmt_number(row['support'] * 100, 2)}% "
                f"({row['pair_count']:,}/{row['n_baskets_analyzed']:,} baskets), "
                f"confidence {_fmt_number(row['confidence'] * 100, 2)}%, lift {_fmt_number(row['lift'], 3)}."
            )
    else:
        lines.append(f"No affinity rules were emitted. Reason: {affinity_details.get('reason') or 'no rules passed thresholds'}.")
    lines.append(
        "Support is pair-containing baskets divided by baskets analyzed; confidence is pair-containing baskets divided by antecedent-containing baskets; lift is confidence divided by consequent support. Rules are observational associations, not causal effects. If caps were active, measures refer only to the deterministic analyzed subset."
    )
    lines.extend(["", "## Cohorts and retention", ""])
    if cohort_summary.height and retention.height:
        lines.append(
            f"There are {cohort_summary.height:,} first-observed-purchase monthly cohorts. Cohort size is the number of customers whose earliest identifiable positive-purchase transaction in the observed cutoff data falls in that month. Retention denominator is the full cohort size; a customer is active in a month if they have at least one identifiable positive-purchase transaction that month."
        )
        complete_later = retention.filter(
            (pl.col("period_complete") == True) & (pl.col("months_since_cohort") > 0)
        )
        if complete_later.height:
            latest_cohort = max(complete_later.get_column("cohort_month").to_list())
            latest_row = complete_later.filter(pl.col("cohort_month") == latest_cohort).sort(
                "months_since_cohort", descending=True
            ).row(0, named=True)
            lines.append(
                f"The latest cohort with a complete post-cohort month is **{latest_cohort.isoformat()}**; "
                f"its latest complete observed retention period is month {latest_row['months_since_cohort']} "
                f"at {_fmt_number(100 * latest_row['retention_rate'], 2)}% ({latest_row['active_customers']:,}/{latest_row['cohort_size']:,})."
            )
        else:
            lines.append("No fully observed post-cohort month exists at the selected as-of date; partial periods are marked in `retention_table.parquet`.")
    else:
        lines.append("Cohort and retention tables are empty because there were no identifiable positive-purchase transactions with valid dates before the analysis cutoff.")
    lines.extend(
        [
            "",
            "## Metric definitions and accounting policy",
            "",
            "- **Line-item revenue:** `price * quantity`. Rows with missing/invalid price or quantity have null line revenue and are excluded from revenue sums; they remain in non-financial counts unless removed by a configured cleaning flag.",
            "- **Gross purchase revenue:** sum of line revenue where quantity > 0. **Return value:** negative of the signed line revenue where quantity < 0. **Net revenue:** sum of signed line revenue over all calculable lines. Therefore gross purchase revenue minus return value equals net revenue, including anomalous negative-price lines. Negative prices are flagged; they are retained unless `--exclude-negative-prices` is set.",
            "- **Transaction grain:** one row per transaction key; by default the key is `(customer_id, transaction_id)`. Missing key values are excluded from transaction counts but not from line-level financial summaries. `basket_size_distinct_products` counts distinct product IDs on positive-quantity lines, not line count or units.",
            "- **Average order value (AOV):** mean gross purchase revenue per identifiable transaction with positive-quantity units; transaction-level `purchase_order_value` is the gross purchase revenue within that transaction. Customer AOV is the mean across that customer’s qualifying purchase transactions. **Customer frequency:** number of identifiable, valid-dated transactions containing positive quantity. **Customer monetary value:** observed net revenue in dated line items up to the cutoff; it is not predicted customer lifetime value.",
            "- **Recency:** calendar days between the as-of date and the last observed identifiable positive-purchase transaction. **Observed tenure:** days from the first such transaction in this file to the as-of date; neither measures true customer age or acquisition date.",
            "- **Interpurchase cadence:** intervals in whole calendar days between dated positive-purchase transactions; ties on the same date yield zero-day intervals. Cadence standard deviation is sample standard deviation and is null when fewer than two intervals exist.",
            "- **Category/department concentration:** HHI is the sum of squared category or department shares of positive purchase line revenue; it is calculated only where positive spend is available and ignores negative/zero revenue anomalies. HHI closer to 1 means higher concentration among the observed categories or departments.",
            "- **Customer revenue concentration:** top 1%/5%/10% spend shares, HHI, and Gini use positive line-item revenue from positive-quantity lines among customers with at least one identifiable positive-purchase transaction. These are observed-period concentration measures, not customer lifetime value.",
            "- **Retention uncertainty:** retention_table includes 95% Wilson score intervals for each binomial active-customer proportion. They quantify finite-cohort sampling uncertainty, not bias from incomplete records or nonrepresentative observation windows.",
            "- **Date parsing:** an explicitly supplied --date-format is tried first, followed by common ISO, US slash, European slash, and hyphenated formats. For ambiguous slash dates, month/day/year is tried before day/month/year; specify --date-format when the source convention is known.",
            "- **Recent change:** compares the last configured number of calendar days with the immediately preceding equal-length period. Percentage change is emitted only when the prior baseline is positive.",
            "- **Return behavior:** return lines are identified by quantity < 0. They are not matched to original purchases, and observed return ratios can be distorted by missing records or anomalous prices.",
            "",
            "## Data quality, exclusions, and cautions",
            "",
            f"Input rows: {quality.get('input_rows', 0):,}; rows retained for analysis: {quality.get('cleaned_rows', 0):,}. See `data_quality.json` for full issue counts, exclusions, metadata conflicts, and warnings.",
        ]
    )
    if quality.get("issue_counts"):
        for key, value in quality["issue_counts"].items():
            lines.append(f"- `{key}`: {value:,}" if isinstance(value, (int, float)) else f"- `{key}`: {value}")
    if quality.get("exclusions"):
        lines.append("")
        lines.append("Configured exclusions:")
        for key, value in quality["exclusions"].items():
            lines.append(f"- `{key}`: {value:,}")
    if warnings:
        lines.extend(["", "Warnings:"])
        lines.extend([f"- {warning}" for warning in warnings])
    else:
        lines.extend(["", "No additional runtime warnings were recorded."])
    if skipped_analyses:
        lines.extend(["", "Skipped or constrained analyses:"])
        for entry in skipped_analyses:
            lines.append(f"- **{entry['analysis']}**: {entry['reason']}")
    lines.extend(
        [
            "",
            "## Reproducibility and file policy",
            "",
            f"Random seed: `{config.get('random_seed')}`. Existing known output filenames are overwritten on each successful run; unrelated files in the output directory are preserved. All temporal features use the stated analysis cutoff, and rolling windows are inclusive calendar-day windows ending on that date.",
            "",
        ]
    )
    (output_dir / "analysis_report.md").write_text("\n".join(lines), encoding="utf-8")


def package_version(package_name: str) -> str | None:
    """Return a package version if installed."""
    try:
        return importlib_metadata.version(package_name)
    except importlib_metadata.PackageNotFoundError:
        return None


def run_analysis(args: argparse.Namespace) -> int:
    """Run the complete pipeline and write all declared artifacts."""
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"input file does not exist or is not a file: {input_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    started_utc = datetime.now(timezone.utc)
    LOGGER.info("Opening input file")
    lazy = read_lazy_input(input_path)
    available_columns = lazy.collect_schema().names()
    missing_columns = [name for name in COLUMNS if name not in available_columns]
    if missing_columns:
        raise ValueError(
            "input is missing required columns: " + ", ".join(missing_columns)
        )
    extra_columns = sorted(set(available_columns) - set(COLUMNS))
    if extra_columns:
        LOGGER.warning("Ignoring extra input columns: %s", ", ".join(extra_columns))
    converted = convert_input(lazy, args.date_format)
    input_rows = converted.height
    duplicate_excess, duplicate_patterns = duplicate_statistics(converted)

    issue_counts: dict[str, Any] = {
        "missing_customer_id_rows": _count_where(converted, pl.col("customer_id").is_null()),
        "missing_transaction_id_rows": _count_where(converted, pl.col("transaction_id").is_null()),
        "missing_product_id_rows": _count_where(converted, pl.col("product_id").is_null()),
        "missing_transaction_date_rows": _count_where(converted, ~pl.col("__raw_present_transaction_date")),
        "malformed_transaction_date_rows": _count_where(converted, pl.col("__invalid_transaction_date_conversion")),
        "missing_price_rows": _count_where(converted, pl.col("price").is_null()),
        "invalid_price_conversion_rows": _count_where(converted, pl.col("__invalid_price_conversion")),
        "negative_price_rows": _count_where(converted, pl.col("price") < 0),
        "zero_price_rows": _count_where(converted, pl.col("price") == 0),
        "missing_quantity_rows": _count_where(converted, pl.col("quantity").is_null()),
        "invalid_quantity_conversion_rows": _count_where(converted, pl.col("__invalid_quantity_conversion")),
        "negative_quantity_return_rows": _count_where(converted, pl.col("quantity") < 0),
        "zero_quantity_rows": _count_where(converted, pl.col("quantity") == 0),
        "uncomputable_line_revenue_rows": _count_where(converted, pl.col("__line_revenue").is_null()),
        "exact_duplicate_excess_rows": duplicate_excess,
        "exact_duplicate_patterns": duplicate_patterns,
        "ignored_extra_columns": extra_columns,
    }

    working = converted
    exclusions: dict[str, int] = {
        "exact_duplicate_rows_removed": 0,
        "zero_quantity_rows_removed": 0,
        "negative_price_rows_removed": 0,
    }
    if args.drop_exact_duplicates:
        working = working.unique(subset=COLUMNS, keep="first")
        exclusions["exact_duplicate_rows_removed"] = input_rows - working.height
    if args.exclude_zero_quantity:
        before = working.height
        working = working.filter(pl.col("quantity").is_null() | (pl.col("quantity") != 0))
        exclusions["zero_quantity_rows_removed"] = before - working.height
    if args.exclude_negative_prices:
        before = working.height
        working = working.filter(pl.col("price").is_null() | (pl.col("price") >= 0))
        exclusions["negative_price_rows_removed"] = before - working.height

    internal_columns = [
        f"__raw_{column}" for column in COLUMNS
    ] + [
        "__raw_present_transaction_date",
        "__raw_present_price",
        "__raw_present_quantity",
        "__invalid_transaction_date_conversion",
        "__invalid_price_conversion",
        "__invalid_quantity_conversion",
    ]
    cleaned = working.drop([col for col in internal_columns if col in working.columns])
    cleaned_rows = cleaned.height
    min_day = cleaned.select(pl.col("transaction_day").min()).item() if cleaned.height else None
    max_day = cleaned.select(pl.col("transaction_day").max()).item() if cleaned.height else None
    if args.as_of_date:
        as_of_date = datetime.strptime(args.as_of_date, "%Y-%m-%d").date()
        as_of_was_explicit = True
    else:
        as_of_date = max_day
        as_of_was_explicit = False

    if as_of_date is not None:
        time_lines = cleaned.filter(
            pl.col("transaction_day").is_not_null()
            & (pl.col("transaction_day") <= pl.lit(as_of_date))
        )
        future_rows = _count_where(cleaned, pl.col("transaction_day") > pl.lit(as_of_date))
    else:
        time_lines = cleaned.filter(pl.lit(False))
        future_rows = 0
    customer_time_lines = time_lines.filter(pl.col("customer_id").is_not_null())
    LOGGER.info("Building transaction and dimensional summaries")

    composite_transaction_lines = summarize_transaction_lines(
        cleaned, ["customer_id", "transaction_id"]
    )
    if args.transaction_key_mode == "customer-transaction":
        transaction_summary = composite_transaction_lines
    else:
        transaction_summary = summarize_transaction_lines(cleaned, ["transaction_id"])
    customer_transactions = summarize_transaction_lines(
        customer_time_lines.filter(pl.col("transaction_id").is_not_null()),
        ["customer_id", "transaction_id"],
    )
    time_transaction_keys = (
        ["transaction_id"]
        if args.transaction_key_mode == "transaction-only"
        else ["customer_id", "transaction_id"]
    )
    temporal_transactions = summarize_transaction_lines(
        time_lines.filter(pl.col("transaction_id").is_not_null()), time_transaction_keys
    )

    product_summary, product_metadata_variants = build_product_outputs(cleaned, args.transaction_key_mode)
    category_summary = dimension_summary(cleaned, "category", args.transaction_key_mode)
    department_summary = dimension_summary(cleaned, "department", args.transaction_key_mode)
    product_repeat = build_product_repeat_behavior(time_lines, args.transaction_key_mode)
    if product_repeat.height:
        product_summary = product_summary.join(product_repeat, on="product_id", how="left").with_columns(
            pl.col("observed_buyers").fill_null(0),
            pl.col("repeat_buyers").fill_null(0),
        )

    monthly_summary, weekday_summary = monthly_and_weekday_summaries(time_lines, temporal_transactions)
    LOGGER.info("Engineering customer features and cohorts")
    customer_features, interpurchase_intervals = build_customer_features(
        customer_time_lines,
        customer_transactions,
        as_of_date,
        args.windows_days,
        args.recent_window_days,
        args.inactivity_min_days,
        args.inactivity_cadence_multiplier,
        args.inactivity_fallback_days,
    )
    customer_features = assign_rfm_segments(customer_features)
    customer_features, customer_concentration_summary = add_customer_revenue_concentration(customer_features)
    customer_features, model_diagnostics, cluster_profiles = cluster_customers(
        customer_features,
        args.random_seed,
        args.cluster_max_customers,
        args.cluster_max_evaluation_sample,
        args.max_clusters,
    )
    rfm_profiles = build_rfm_profiles(customer_features)
    purchase_transactions = customer_transactions.filter(
        pl.col("transaction_day").is_not_null() & (pl.col("units_purchased") > 0)
    )
    cohort_summary, retention_table = build_cohort_outputs(purchase_transactions, as_of_date)

    LOGGER.info("Computing bounded basket affinity")
    affinity, affinity_details = basket_affinity(
        time_lines,
        args.basket_level,
        args.transaction_key_mode,
        args.min_support,
        args.min_confidence,
        args.min_lift,
        args.max_affinity_items,
        args.max_affinity_baskets,
        args.max_items_per_basket,
        args.max_affinity_pair_operations,
        args.random_seed,
    )

    # Quality counts that require grouping are computed after configured cleaning.
    product_meta_conflict_count = (
        product_metadata_variants.select(pl.col("product_id").n_unique()).item()
        if product_metadata_variants.height
        else 0
    )
    transaction_date_conflicts = (
        composite_transaction_lines.filter(pl.col("distinct_transaction_dates") > 1).height
        if composite_transaction_lines.height
        else 0
    )
    cross_customer_transaction_ids = (
        cleaned.filter(pl.col("transaction_id").is_not_null() & pl.col("customer_id").is_not_null())
        .group_by("transaction_id")
        .agg(pl.col("customer_id").n_unique().alias("__customers"))
        .filter(pl.col("__customers") > 1)
        .height
        if cleaned.height
        else 0
    )
    issue_counts.update(
        {
            "product_ids_with_inconsistent_nonmissing_metadata": product_meta_conflict_count,
            "customer_transaction_keys_spanning_multiple_calendar_dates": transaction_date_conflicts,
            "transaction_ids_used_by_multiple_customers": cross_customer_transaction_ids,
            "dated_rows_after_analysis_as_of_date": future_rows,
        }
    )

    warnings: list[str] = []
    if extra_columns:
        warnings.append("Extra input columns were ignored: " + ", ".join(extra_columns))
    if duplicate_excess and not args.drop_exact_duplicates:
        warnings.append(
            f"{duplicate_excess} excess exact normalized line rows were detected and retained; repeated lines can be genuine, so no deduplication was done by default."
        )
    if issue_counts["malformed_transaction_date_rows"]:
        warnings.append(
            f"{issue_counts['malformed_transaction_date_rows']} nonblank transaction_date values could not be parsed and were omitted from date-dependent analyses."
        )
    if issue_counts["missing_customer_id_rows"]:
        warnings.append(
            f"{issue_counts['missing_customer_id_rows']} rows lack customer_id and are excluded from customer-level features; line-level financial summaries retain them."
        )
    if issue_counts["missing_transaction_id_rows"]:
        warnings.append(
            f"{issue_counts['missing_transaction_id_rows']} rows lack transaction_id and are excluded from identifiable transaction frequency and basket affinity; line-level revenue is retained."
        )
    if issue_counts["uncomputable_line_revenue_rows"]:
        warnings.append(
            f"{issue_counts['uncomputable_line_revenue_rows']} rows have missing/invalid price or quantity and therefore contribute no line-item revenue."
        )
    if issue_counts["negative_price_rows"]:
        warnings.append(
            f"{issue_counts['negative_price_rows']} negative-price rows were retained unless --exclude-negative-prices was supplied; their signed revenue remains in accounting totals."
        )
    if product_meta_conflict_count:
        warnings.append(
            f"{product_meta_conflict_count} product IDs have conflicting nonmissing metadata; conflicting variants are preserved in product_metadata_variants.parquet."
        )
    if transaction_date_conflicts:
        warnings.append(
            f"{transaction_date_conflicts} customer-transaction keys span multiple calendar dates; transaction_day uses the earliest observed line date."
        )
    if cross_customer_transaction_ids and args.transaction_key_mode == "transaction-only":
        warnings.append(
            f"{cross_customer_transaction_ids} transaction IDs occur under multiple customers even though transaction-only mode was selected; transaction summaries and baskets may combine such records."
        )
    if not customer_features.height or customer_features.filter(pl.col("purchase_transaction_count") > 0).height == 0:
        warnings.append("No customers with identifiable positive-purchase transactions were available for RFM, cadence, or purchase-cohort analyses.")
    if affinity.height == 0:
        warnings.append(f"Basket affinity emitted no rules: {affinity_details.get('reason') or 'no rules met the configured thresholds'}.")
    if cleaned.height == 0:
        warnings.append("No rows remain after configured cleaning; only schema-stable empty outputs could be generated.")
    if as_of_date is None:
        warnings.append("No valid transaction dates were found and no --as-of-date was configured; temporal customer analyses are empty.")

    # Accounting and roll-up invariants: checked on the same cleaned line population.
    if cleaned.height:
        accounting = cleaned.select(
            pl.col("__line_revenue").sum().fill_null(0.0).alias("net"),
            pl.when(pl.col("quantity") > 0)
            .then(pl.col("__line_revenue")
            )
            .otherwise(None)
            .sum()
            .fill_null(0.0)
            .alias("gross"),
            pl.when(pl.col("quantity") < 0)
            .then(-pl.col("__line_revenue"))
            .otherwise(None)
            .sum()
            .fill_null(0.0)
            .alias("returns"),
        ).row(0, named=True)
        if not np.isclose(
            float(accounting["gross"]) - float(accounting["returns"]),
            float(accounting["net"]),
            rtol=1e-9,
            atol=1e-7,
        ):
            raise AssertionError("accounting invariant failed: gross purchase revenue - return value != net revenue")
        for label, summary in (
            ("product", product_summary),
            ("category", category_summary),
            ("department", department_summary),
        ):
            for column in ("net_revenue", "gross_purchase_revenue", "return_value"):
                summary_sum = _number(summary, column)
                if not np.isclose(summary_sum, float(accounting["net"] if column == "net_revenue" else accounting["gross"] if column == "gross_purchase_revenue" else accounting["returns"]), rtol=1e-8, atol=1e-6):
                    raise AssertionError(f"{label} aggregation invariant failed for {column}")

    quality = {
        "input_path": str(input_path),
        "input_rows": input_rows,
        "cleaned_rows": cleaned_rows,
        "source_columns": available_columns,
        "ignored_extra_columns": extra_columns,
        "issue_counts": issue_counts,
        "exclusions": exclusions,
        "temporal_and_analytic_exclusions": {
            "invalid_or_missing_date_rows_not_in_date_dependent_analyses": issue_counts["missing_transaction_date_rows"] + issue_counts["malformed_transaction_date_rows"],
            "rows_after_analysis_as_of_date_not_in_temporal_analyses": future_rows,
            "rows_missing_customer_id_not_in_customer_analyses": issue_counts["missing_customer_id_rows"],
            "rows_missing_transaction_id_not_in_transaction_frequency_and_basket_analyses": issue_counts["missing_transaction_id_rows"],
            "rows_with_uncomputable_revenue_excluded_from_revenue_sums": issue_counts["uncomputable_line_revenue_rows"],
        },
        "cleaning_policy": {
            "drop_exact_duplicates": bool(args.drop_exact_duplicates),
            "exclude_zero_quantity": bool(args.exclude_zero_quantity),
            "exclude_negative_prices": bool(args.exclude_negative_prices),
            "duplicate_definition": "Repeated normalized values across the nine required columns; retained unless explicitly configured for removal.",
        },
        "metadata_conflict_product_id_count": product_meta_conflict_count,
        "minimum_observed_date": min_day,
        "maximum_observed_date": max_day,
        "analysis_as_of_date": as_of_date,
        "as_of_date_explicitly_configured": as_of_was_explicit,
        "warnings": warnings,
    }
    skipped_analyses: list[dict[str, str]] = []
    if model_diagnostics.height and "status" in model_diagnostics.columns:
        skipped_rows = model_diagnostics.filter(pl.col("status") == "skipped")
        for row in skipped_rows.iter_rows(named=True):
            skipped_analyses.append({"analysis": str(row.get("model") or "customer clustering"), "reason": str(row.get("reason") or "not feasible")})
    if affinity.height == 0:
        skipped_analyses.append({"analysis": "basket affinity", "reason": str(affinity_details.get("reason") or "no rules returned")})
    if cohort_summary.height == 0:
        skipped_analyses.append({"analysis": "cohorts and retention", "reason": "no identifiable positive-purchase transactions with valid dates"})

    config_payload = {
        **vars(args),
        "input": str(input_path),
        "output_dir": str(output_dir),
        "windows_days": list(args.windows_days),
        "random_seed": args.random_seed,
    }
    config_payload = json_safe(config_payload)
    written: list[str] = []
    write_json(output_dir / "data_quality.json", quality)
    written.append("data_quality.json")
    issue_rows = [
        {"issue": str(key), "count_or_value": json.dumps(json_safe(value), ensure_ascii=False)}
        for key, value in issue_counts.items()
    ]
    issue_table = pl.DataFrame(issue_rows, schema={"issue": pl.String, "count_or_value": pl.String})
    write_table(issue_table, output_dir, "data_quality_issues", args.write_csv, written)

    tables: list[tuple[str, pl.DataFrame]] = [
        ("customer_features", customer_features),
        ("customer_revenue_concentration", customer_concentration_summary),
        ("interpurchase_intervals", interpurchase_intervals),
        ("transaction_summary", transaction_summary),
        ("product_summary", product_summary),
        ("product_metadata_variants", product_metadata_variants),
        ("product_repeat_behavior", product_repeat),
        ("category_summary", category_summary),
        ("department_summary", department_summary),
        ("monthly_summary", monthly_summary),
        ("weekday_summary", weekday_summary),
        ("cohort_summary", cohort_summary),
        ("retention_table", retention_table),
        ("basket_affinity", affinity),
        ("model_diagnostics", model_diagnostics),
        ("rfm_segment_profiles", rfm_profiles),
        ("cluster_profiles", cluster_profiles),
    ]
    for stem, table in tables:
        write_table(table, output_dir, stem, args.write_csv, written)

    generate_report(
        output_dir,
        quality,
        cleaned,
        customer_features,
        customer_concentration_summary,
        category_summary,
        rfm_profiles,
        cluster_profiles,
        affinity,
        affinity_details,
        cohort_summary,
        retention_table,
        as_of_date,
        config_payload,
        skipped_analyses,
        warnings,
    )
    written.append("analysis_report.md")
    completed_utc = datetime.now(timezone.utc)
    run_metadata = {
        "run_started_utc": started_utc,
        "run_completed_utc": completed_utc,
        "elapsed_seconds": (completed_utc - started_utc).total_seconds(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "package_versions": {
            "polars": getattr(pl, "__version__", None),
            "numpy": getattr(np, "__version__", None),
            "scipy": package_version("scipy"),
            "scikit-learn": package_version("scikit-learn"),
        },
        "configuration": config_payload,
        "input_rows": input_rows,
        "cleaned_rows": cleaned_rows,
        "minimum_observed_date": min_day,
        "maximum_observed_date": max_day,
        "analysis_as_of_date": as_of_date,
        "as_of_date_explicitly_configured": as_of_was_explicit,
        "affinity_run_details": affinity_details,
        "skipped_analyses": skipped_analyses,
        "warnings": warnings,
        "output_files_overwritten_or_written": sorted(set(written + ["run_metadata.json"])),
        "rerun_policy": "Known output filenames are overwritten; unrelated files in the output directory are preserved.",
    }
    write_json(output_dir / "run_metadata.json", run_metadata)
    LOGGER.info("Analysis complete: %s", str(output_dir))
    LOGGER.info("Wrote %d output artifacts", len(set(written + ["run_metadata.json"])))
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
