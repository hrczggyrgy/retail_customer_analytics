"""Polars utilities for memory-efficient data handling.

This module provides helper functions to convert between pandas and Polars,
and common data operations optimized for large datasets using Polars.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import polars as pl

LOGGER = logging.getLogger("retail_analysis.polars_utils")


def to_polars(df: pd.DataFrame) -> pl.DataFrame:
    """Convert pandas DataFrame to Polars DataFrame for memory efficiency.

    Args:
        df: pandas DataFrame

    Returns:
        Polars DataFrame
    """
    try:
        return pl.from_pandas(df)
    except Exception as e:
        LOGGER.warning(f"Failed to convert to Polars: {e}. Using pandas instead.")
        return None


def to_pandas(df: pl.DataFrame) -> pd.DataFrame:
    """Convert Polars DataFrame to pandas DataFrame.

    Args:
        df: Polars DataFrame

    Returns:
        pandas DataFrame
    """
    if df is None:
        return pd.DataFrame()
    return df.to_pandas()


def filter_by_date_range(
    df: pl.DataFrame,
    date_col: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pl.DataFrame:
    """Filter Polars DataFrame by date range.

    Args:
        df: Polars DataFrame
        date_col: Name of date column
        start_date: Start date (inclusive)
        end_date: End date (inclusive)

    Returns:
        Filtered Polars DataFrame
    """
    if df is None:
        return pl.DataFrame()

    return df.filter((pl.col(date_col) >= start_date) & (pl.col(date_col) <= end_date))


def compute_period_metrics_pl(
    df: pl.DataFrame,
    date_col: str = "transaction_day",
    revenue_col: str = "revenue",
    transaction_col: str = "transaction_id",
    customer_col: str = "customer_id",
    quantity_col: str = "quantity",
) -> dict[str, Any]:
    """Compute period metrics using Polars for memory efficiency.

    Args:
        df: Polars DataFrame with transaction data
        date_col: Date column name
        revenue_col: Revenue column name
        transaction_col: Transaction ID column name
        customer_col: Customer ID column name
        quantity_col: Quantity column name

    Returns:
        Dictionary with computed metrics
    """
    if df is None or df.height == 0:
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

    # Compute metrics using Polars aggregations
    metrics = df.select(
        [
            pl.col(revenue_col).sum().alias("revenue"),
            pl.col(transaction_col).n_unique().alias("orders"),
            pl.col(customer_col).n_unique().alias("customers"),
            pl.col(quantity_col).sum().alias("units_sold"),
        ]
    ).row(0)

    revenue, orders, customers, units_sold = metrics

    # Compute derived metrics
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


def group_by_metrics_pl(
    df: pl.DataFrame,
    group_cols: list[str],
    metrics: dict[str, str],
) -> pl.DataFrame:
    """Group by columns and compute metrics using Polars.

    Args:
        df: Polars DataFrame
        group_cols: List of columns to group by
        metrics: Dictionary mapping output column names to aggregation expressions
                  e.g., {"revenue": "sum(revenue)", "orders": "n_unique(transaction_id)"}

    Returns:
        Grouped Polars DataFrame with computed metrics
    """
    if df is None or df.height == 0:
        return pl.DataFrame()

    # Build aggregation expressions
    agg_exprs = []
    for col_name, expr_str in metrics.items():
        try:
            agg_exprs.append(eval(f"pl.{expr_str}").alias(col_name))
        except Exception as e:
            LOGGER.warning(f"Failed to parse aggregation '{expr_str}': {e}")

    if not agg_exprs:
        return df.group_by(group_cols).agg(pl.len().alias("count"))

    return df.group_by(group_cols).agg(agg_exprs)


def compute_rolling_metrics_pl(
    df: pl.DataFrame,
    date_col: str,
    value_col: str,
    window_days: int,
) -> pl.DataFrame:
    """Compute rolling metrics over time using Polars.

    Args:
        df: Polars DataFrame with date and value columns
        date_col: Date column name
        value_col: Value column to aggregate
        window_days: Rolling window size in days

    Returns:
        DataFrame with rolling metrics
    """
    if df is None or df.height == 0:
        return pl.DataFrame()

    # Sort by date first
    df_sorted = df.sort(date_col)

    # Compute rolling sum over the window
    result = df_sorted.with_columns(
        pl.col(value_col)
        .rolling_sum(
            window_size=window_days,
            min_periods=1,
        )
        .alias(f"{value_col}_rolling_{window_days}d")
    )

    return result


def memory_efficient_join(
    left: pl.DataFrame,
    right: pl.DataFrame,
    on: str | list[str],
    how: str = "inner",
) -> pl.DataFrame:
    """Perform memory-efficient join using Polars.

    Args:
        left: Left DataFrame
        right: Right DataFrame
        on: Column(s) to join on
        how: Join type ('inner', 'left', 'outer', etc.)

    Returns:
        Joined DataFrame
    """
    if left is None:
        return right
    if right is None:
        return left

    return left.join(right, on=on, how=how)


def optimize_dtypes(df: pl.DataFrame) -> pl.DataFrame:
    """Optimize column dtypes for memory efficiency.

    Args:
        df: Polars DataFrame

    Returns:
        DataFrame with optimized dtypes
    """
    if df is None:
        return pl.DataFrame()

    # Convert to appropriate categorical types for string columns with low cardinality
    for col in df.columns:
        if df[col].dtype == pl.String:
            unique_count = df[col].n_unique()
            if unique_count < len(df) * 0.5:  # If cardinality < 50% of rows
                df = df.with_columns(pl.col(col).cast(pl.Categorical))

    return df
