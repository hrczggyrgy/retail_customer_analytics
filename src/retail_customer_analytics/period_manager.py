"""Period and context management for retail customer analytics.

This module handles period selection, date range calculations, and
period context management used by the Streamlit app.
"""

from __future__ import annotations

import logging

import pandas as pd

LOGGER = logging.getLogger("retail_analysis.period_manager")


def get_period_context(period_params: dict) -> dict:
    """Compute period boundaries from period_params.

    Args:
        period_params: Dictionary with:
            - current_start: Current period start date
            - current_end: Current period end date
            - compare_mode: "prior" or "yoy" (default: "prior")

    Returns:
        Dictionary with:
            - current_start, current_end: the selected current period
            - prior_start, prior_end: the comparison period (prior or yoy)
            - period_len_days: length of current period in days
            - compare_mode: "prior" or "yoy"
    """
    current_start = period_params["current_start"]
    current_end = period_params["current_end"]
    compare_mode = period_params.get("compare_mode", "prior")

    period_len = (current_end - current_start).days + 1

    if compare_mode == "prior":
        prior_end = current_start - pd.Timedelta(days=1)
        prior_start = prior_end - pd.Timedelta(days=period_len - 1)
    else:  # yoy
        prior_start = current_start - pd.DateOffset(years=1)
        prior_end = current_end - pd.DateOffset(years=1)

    return {
        "current_start": current_start,
        "current_end": current_end,
        "prior_start": prior_start,
        "prior_end": prior_end,
        "period_len_days": period_len,
        "compare_mode": compare_mode,
    }


def get_period_transactions(
    transactions: pd.DataFrame,
    context: dict,
    period: str = "current",
) -> pd.DataFrame:
    """Filter transactions to a specific period using period context.

    Args:
        transactions: Normalized transaction DataFrame (from prepare_transaction_frame)
        context: Period context from get_period_context()
        period: "current" or "prior"

    Returns:
        Filtered DataFrame for the specified period.
    """
    if period == "current":
        start = context["current_start"]
        end = context["current_end"]
    elif period == "prior":
        start = context["prior_start"]
        end = context["prior_end"]
    else:
        raise ValueError(f"Unknown period: {period}. Use 'current' or 'prior'.")

    return transactions[
        (transactions["transaction_day"] >= start) & (transactions["transaction_day"] <= end)
    ].copy()


def compute_preset_date_range(
    preset: str,
    max_date: pd.Timestamp,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Compute date range for a preset period.

    Args:
        preset: Preset name ("Last 7 days", "Last 28 days", "Last 90 days",
                "Month to date", "Quarter to date")
        max_date: Maximum available date in data

    Returns:
        Tuple of (start_date, end_date)
    """
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
        # Default to last 28 days
        current_end = max_date
        current_start = current_end - pd.Timedelta(days=27)

    return current_start, current_end


def validate_date_range(
    start: pd.Timestamp,
    end: pd.Timestamp,
    min_date: pd.Timestamp,
    max_date: pd.Timestamp,
) -> bool:
    """Validate that a date range is within data bounds.

    Args:
        start: Start date
        end: End date
        min_date: Minimum available date in data
        max_date: Maximum available date in data

    Returns:
        True if valid, False otherwise
    """
    if start > end:
        return False
    if start < min_date or end > max_date:
        return False
    return True


def ensure_not_empty(df: pd.DataFrame, context: str = "") -> bool:
    """Standard empty-data guard.

    Args:
        df: DataFrame to check
        context: Context string for warning message

    Returns:
        True if DataFrame has data, False otherwise
    """
    if df is None or df.empty:
        if context:
            LOGGER.warning(f"Empty DataFrame for {context}")
        return False
    return True
