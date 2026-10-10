"""Retail Customer Analytics - Lightweight, validated probabilistic pipeline."""

from .ingestion import (
    REQUIRED_COLUMNS,
    ID_COLUMNS,
    TEXT_COLUMNS,
    DATE_FORMATS,
    DEFAULT_CURRENCY,
    CURRENCY_SYMBOLS,
    parse_dates,
    load_lines,
    clean_lines,
    structural_checks,
    prepare_transaction_frame,
    load_lines_from_dataframe,
    fmt_currency,
    fmt_number,
    fmt_pct,
)

from .features import (
    build_customer_snapshot,
    build_customer_features,
    _build_trips_for_snapshot,
    _core_customer_features,
    _window_features,
    _breadth_features,
    _return_features,
    _promo_features,
    _category_affinity_features,
    _trend_features,
    _basket_composition_features,
)

__all__ = [
    "REQUIRED_COLUMNS",
    "ID_COLUMNS",
    "TEXT_COLUMNS",
    "DATE_FORMATS",
    "DEFAULT_CURRENCY",
    "CURRENCY_SYMBOLS",
    "parse_dates",
    "load_lines",
    "clean_lines",
    "structural_checks",
    "prepare_transaction_frame",
    "load_lines_from_dataframe",
    "fmt_currency",
    "fmt_number",
    "fmt_pct",
    "build_customer_snapshot",
    "build_customer_features",
    "_build_trips_for_snapshot",
    "_core_customer_features",
    "_window_features",
    "_breadth_features",
    "_return_features",
    "_promo_features",
    "_category_affinity_features",
    "_trend_features",
    "_basket_composition_features",
]

__version__ = "0.1.0"