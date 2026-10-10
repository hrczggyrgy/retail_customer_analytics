"""Shared ingestion module for retail customer analytics.

Provides canonical schema validation, data cleaning, and normalization
used by both CLI and Streamlit app.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("retail_analysis.ingestion")

# Canonical column definitions
REQUIRED_COLUMNS = [
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

# Supported date formats (ISO formats only - ambiguous formats require explicit --date-format)
DATE_FORMATS = [
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
]

# Default currency
DEFAULT_CURRENCY = "EUR"


def parse_dates(raw: pd.Series, date_format: str | None) -> pd.Series:
    """
    Parse dates with explicit format preference.

    If date_format is provided, it is tried first.
    Otherwise, ISO formats (YYYY-MM-DD) are tried.
    Ambiguous month/day vs day/month formats are NOT automatically resolved -
    they will result in NaT if no explicit format is given.
    """
    raw = raw.astype("string").str.strip()
    out = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns]")

    formats = []
    if date_format:
        formats.append(date_format)
    formats.extend(DATE_FORMATS)

    for fmt_ in dict.fromkeys(formats):
        todo = out.isna() & raw.notna() & (raw != "")
        if not todo.any():
            break
        out.loc[todo] = pd.to_datetime(raw[todo], format=fmt_, errors="coerce")

    # Warn if any dates remain unparsed
    unparsed = out.isna() & raw.notna() & (raw != "")
    if unparsed.any():
        LOGGER.warning("%d date values could not be parsed. Use --date-format to specify format.", unparsed.sum())

    return out


def load_lines(path: Path, date_format: str | None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load and validate input file, returning normalized DataFrame and issue counts."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        raw = pd.read_csv(
            path, dtype=str, keep_default_na=False, na_values=["", "NULL", "null"], encoding_errors="replace"
        )
    elif suffix in {".parquet", ".pq"}:
        raw = pd.read_parquet(path)
    else:
        raise ValueError("input extension must be .csv, .parquet, or .pq")

    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError("input is missing required columns: " + ", ".join(missing))

    extra = sorted(set(raw.columns) - set(REQUIRED_COLUMNS))
    raw = raw[REQUIRED_COLUMNS].copy()

    df = pd.DataFrame(index=raw.index)

    # Normalize text columns
    for col in ID_COLUMNS + TEXT_COLUMNS:
        s = raw[col].astype("string").str.strip()
        s = s.where(s.notna() & (s != ""), other=pd.NA)
        df[col] = s.astype(object).where(s.notna(), None)

    # Normalize numeric columns
    present = {
        c: raw[c].notna() & (raw[c].astype("string").str.strip() != "") for c in ("transaction_date", "price", "quantity")
    }
    for col in ("price", "quantity"):
        v = pd.to_numeric(raw[col], errors="coerce").astype(float)
        df[col] = v.where(np.isfinite(v))

    # Parse dates
    if pd.api.types.is_datetime64_any_dtype(raw["transaction_date"]):
        dt = pd.to_datetime(raw["transaction_date"], errors="coerce")
    else:
        dt = parse_dates(raw["transaction_date"], date_format)
    df["transaction_date"] = dt
    df["transaction_day"] = dt.dt.normalize()

    # Calculate line revenue
    rev = df["price"] * df["quantity"]
    df["line_revenue"] = rev.where(np.isfinite(rev))

    # Issue counts
    issues = {
        "missing_customer_id_rows": int(df["customer_id"].isna().sum()),
        "missing_transaction_id_rows": int(df["transaction_id"].isna().sum()),
        "missing_product_id_rows": int(df["product_id"].isna().sum()),
        "missing_category_rows": int(df["category"].isna().sum()),
        "missing_transaction_date_rows": int((~present["transaction_date"]).sum()),
        "malformed_transaction_date_rows": int((present["transaction_date"] & dt.isna()).sum()),
        "missing_price_rows": int(df["price"].isna().sum()),
        "invalid_price_conversion_rows": int((present["price"] & df["price"].isna()).sum()),
        "negative_price_rows": int((df["price"] < 0).sum()),
        "zero_price_rows": int((df["price"] == 0).sum()),
        "missing_quantity_rows": int(df["quantity"].isna().sum()),
        "invalid_quantity_conversion_rows": int((present["quantity"] & df["quantity"].isna()).sum()),
        "negative_quantity_return_rows": int((df["quantity"] < 0).sum()),
        "zero_quantity_rows": int((df["quantity"] == 0).sum()),
        "uncomputable_line_revenue_rows": int(df["line_revenue"].isna().sum()),
        "exact_duplicate_excess_rows": int(df.duplicated(subset=REQUIRED_COLUMNS, keep="first").sum()),
        "ignored_extra_columns": extra,
    }
    return df, issues


def clean_lines(df: pd.DataFrame, args: Any) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply cleaning rules based on CLI/app arguments."""
    removed = {"exact_duplicate_rows_removed": 0, "zero_quantity_rows_removed": 0, "negative_price_rows_removed": 0}

    if getattr(args, "drop_exact_duplicates", False):
        n = len(df)
        df = df.drop_duplicates(subset=REQUIRED_COLUMNS, keep="first")
        removed["exact_duplicate_rows_removed"] = n - len(df)

    if getattr(args, "exclude_zero_quantity", False):
        n = len(df)
        df = df[df["quantity"].isna() | (df["quantity"] != 0)]
        removed["zero_quantity_rows_removed"] = n - len(df)

    if getattr(args, "exclude_negative_prices", False):
        n = len(df)
        df = df[df["price"].isna() | (df["price"] >= 0)]
        removed["negative_price_rows_removed"] = n - len(df)

    return df.reset_index(drop=True), removed


def structural_checks(df: pd.DataFrame) -> dict[str, int]:
    """Identifier-consistency checks that affect analytical validity."""
    ok = df["customer_id"].notna() & df["transaction_id"].notna()
    out = {
        "transaction_ids_used_by_multiple_customers": int(
            (df[ok].groupby("transaction_id")["customer_id"].nunique() > 1).sum()
        ),
        "customer_transaction_keys_spanning_multiple_days": int(
            (
                df[ok & df["transaction_day"].notna()].groupby(["customer_id", "transaction_id"])["transaction_day"].nunique()
                > 1
            ).sum()
        ),
    }
    pm = df[df["product_id"].notna()].groupby("product_id")[["department", "category"]].nunique()
    out["product_ids_with_conflicting_category_or_department"] = int((pm > 1).any(axis=1).sum())
    return out


def prepare_transaction_frame(df: pd.DataFrame, date_format: str | None = None) -> pd.DataFrame:
    """
    Normalize transaction DataFrame for Streamlit app.

    This is a thin wrapper around the canonical ingestion logic,
    adapted for the app's expected column names and formats.
    """
    # Map app column names to canonical names if needed
    col_map = {
        "invoice_id": "transaction_id",
        "description": "product_description",
        "unit_price": "price",
        "transaction_datetime": "transaction_date",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Ensure all required columns exist
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Use the same parsing logic
    df, _ = load_lines_from_dataframe(df, date_format)

    # Apply standard cleaning (no CLI flags in app)
    args = type("Args", (), {
        "drop_exact_duplicates": False,
        "exclude_zero_quantity": True,  # App drops zero quantity
        "exclude_negative_prices": False,
    })()
    df, _ = clean_lines(df, args)

    # Calculate revenue
    df["revenue"] = df["quantity"] * df["price"]

    return df


def load_lines_from_dataframe(raw: pd.DataFrame, date_format: str | None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load lines from an already-loaded DataFrame (for Streamlit app)."""
    # Ensure we only use canonical columns
    extra = sorted(set(raw.columns) - set(REQUIRED_COLUMNS))
    raw = raw[REQUIRED_COLUMNS].copy()

    df = pd.DataFrame(index=raw.index)

    for col in ID_COLUMNS + TEXT_COLUMNS:
        s = raw[col].astype("string").str.strip()
        s = s.where(s.notna() & (s != ""), other=pd.NA)
        df[col] = s.astype(object).where(s.notna(), None)

    present = {
        c: raw[c].notna() & (raw[c].astype("string").str.strip() != "") for c in ("transaction_date", "price", "quantity")
    }
    for col in ("price", "quantity"):
        v = pd.to_numeric(raw[col], errors="coerce").astype(float)
        df[col] = v.where(np.isfinite(v))

    if pd.api.types.is_datetime64_any_dtype(raw["transaction_date"]):
        dt = pd.to_datetime(raw["transaction_date"], errors="coerce")
    else:
        dt = parse_dates(raw["transaction_date"], date_format)
    df["transaction_date"] = dt
    df["transaction_day"] = dt.dt.normalize()

    rev = df["price"] * df["quantity"]
    df["line_revenue"] = rev.where(np.isfinite(rev))

    issues = {
        "missing_customer_id_rows": int(df["customer_id"].isna().sum()),
        "missing_transaction_id_rows": int(df["transaction_id"].isna().sum()),
        "missing_product_id_rows": int(df["product_id"].isna().sum()),
        "missing_category_rows": int(df["category"].isna().sum()),
        "missing_transaction_date_rows": int((~present["transaction_date"]).sum()),
        "malformed_transaction_date_rows": int((present["transaction_date"] & dt.isna()).sum()),
        "missing_price_rows": int(df["price"].isna().sum()),
        "invalid_price_conversion_rows": int((present["price"] & df["price"].isna()).sum()),
        "negative_price_rows": int((df["price"] < 0).sum()),
        "zero_price_rows": int((df["price"] == 0).sum()),
        "missing_quantity_rows": int(df["quantity"].isna().sum()),
        "invalid_quantity_conversion_rows": int((present["quantity"] & df["quantity"].isna()).sum()),
        "negative_quantity_return_rows": int((df["quantity"] < 0).sum()),
        "zero_quantity_rows": int((df["quantity"] == 0).sum()),
        "uncomputable_line_revenue_rows": int(df["line_revenue"].isna().sum()),
        "exact_duplicate_excess_rows": int(df.duplicated(subset=REQUIRED_COLUMNS, keep="first").sum()),
        "ignored_extra_columns": extra,
    }
    return df, issues


# Currency formatting
CURRENCY_SYMBOLS = {"EUR": "€", "USD": "$", "GBP": "£", "JPY": "¥"}


def fmt_currency(v: float, currency: str = DEFAULT_CURRENCY) -> str:
    """Format currency value with explicit currency symbol."""
    symbol = CURRENCY_SYMBOLS.get(currency.upper(), currency)
    if v >= 1e9:
        return f"{symbol}{v/1e9:.1f}B"
    elif v >= 1e6:
        return f"{symbol}{v/1e6:.1f}M"
    elif v >= 1e3:
        return f"{symbol}{v/1e3:.1f}K"
    else:
        return f"{symbol}{v:,.0f}"


def fmt_number(v: float) -> str:
    if v >= 1e6:
        return f"{v/1e6:.1f}M"
    elif v >= 1e3:
        return f"{v/1e3:.1f}K"
    else:
        return f"{v:,.0f}"


def fmt_pct(v: float) -> str:
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.1f}%"