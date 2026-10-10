#!/usr/bin/env python3
"""
Show all calculated results from the retail customer analytics pipeline.
This includes all KPIs, charts, and key tables that appear in the UI.

Usage:
    python show_all_results.py --input transactions.csv
    python show_all_results.py --synthetic  # Generate synthetic demo data
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

# Add package root to path
sys.path.insert(0, str(Path(__file__).parent))

import insight_engine as ie
import retail_customer_analysis as rca


def load_input_data(input_path: str) -> pd.DataFrame:
    """Load and map input data to pipeline expected format."""
    suffix = Path(input_path).suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(
            input_path, dtype=str, keep_default_na=False, na_values=["", "NULL", "null"]
        )
    elif suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(input_path)
    else:
        raise ValueError("Input must be .csv, .parquet, or .pq")

    # Try to auto-map common column names
    cols = df.columns.str.strip().str.lower()
    col_map = {}

    # Standard mappings
    for target, aliases in {
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
            "transaction_datetime",
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
    }.items():
        for alias in aliases:
            if alias in cols.values:
                col_map[target] = df.columns[cols == alias][0]
                break

    # Check required columns
    required = [
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
    missing = [c for c in required if c not in col_map]
    if missing:
        raise ValueError(
            f"Could not map required columns: {missing}. Available: {list(df.columns)}"
        )

    mapped = pd.DataFrame(
        {
            "customer_id": df[col_map["customer_id"]].astype("string"),
            "transaction_date": pd.to_datetime(df[col_map["transaction_date"]], errors="coerce"),
            "transaction_id": df[col_map["transaction_id"]].astype("string"),
            "product_id": df[col_map["product_id"]].astype("string"),
            "product_description": df[col_map["product_description"]].astype("string"),
            "department": df[col_map["department"]].astype("string"),
            "category": df[col_map["category"]].astype("string"),
            "price": pd.to_numeric(df[col_map["price"]], errors="coerce").astype(float),
            "quantity": pd.to_numeric(df[col_map["quantity"]], errors="coerce").astype("Int64"),
        }
    )

    # Drop rows with invalid core data
    mapped = mapped.dropna(subset=["customer_id", "transaction_date", "quantity", "price"])

    return mapped


def generate_synthetic_data(
    n_customers: int = 500, days: int = 180, seed: int = 42
) -> pd.DataFrame:
    """Generate synthetic data using the canonical generator."""
    import numpy as np

    rng = np.random.default_rng(seed)
    k = 5
    cats = [f"Category {i:02d}" for i in range(k)]
    depts = [f"Department {i % 2 + 1}" for i in range(k)]
    prod_per_cat = 4
    base_price = {c: float(rng.lognormal(np.log(5 + 2 * c), 0.2)) for c in range(k)}
    products = {
        c: [
            (f"P{c:02d}{j:02d}", round(base_price[c] * float(rng.uniform(0.8, 1.3)), 2))
            for j in range(prod_per_cat)
        ]
        for c in range(k)
    }
    lam = rng.gamma(0.5, 1 / 8.0, n_customers)
    p_drop = rng.beta(0.8, 2.0, n_customers)
    first = rng.integers(0, max(int(days * 0.6), 1), n_customers)
    pref = rng.dirichlet(np.ones(k) * 0.5, n_customers)
    size_factor = rng.gamma(1.5, 0.5, n_customers)
    t0 = pd.Timestamp("2024-01-01")
    rows = []
    for i in range(n_customers):
        days - first[i]
        n_buy = max(1, int(rng.geometric(p_drop[i])))
        gaps = rng.exponential(1 / lam[i], size=n_buy)
        times = np.concatenate([[0.0], np.cumsum(gaps)]) + first[i]
        times = times[times < days]
        for t in times:
            day = t0 + pd.Timedelta(days=int(t))
            n_items = 1 + rng.poisson(size_factor[i])
            chosen = set(rng.choice(k, size=min(n_items, k), replace=True, p=pref[i]))
            receipt = rng.integers(0, 2) if rng.random() < 0.15 else 0
            for c in chosen:
                pid, price = products[c][rng.integers(prod_per_cat)]
                if rng.random() < 0.1:
                    price = round(price * 0.8, 2)
                qty = int(1 + rng.poisson(0.4))
                cid = None if rng.random() < 0.02 else f"C{i:05d}"
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
                if rng.random() < 0.02:
                    rday = day + pd.Timedelta(days=int(rng.integers(1, 10)))
                    if rday <= t0 + pd.Timedelta(days=days):
                        rows.append(
                            (
                                cid,
                                rday.strftime("%Y-%m-%d"),
                                f"R{tid}",
                                pid,
                                f"Product {pid}",
                                depts[c],
                                cats[c],
                                price,
                                -qty,
                            )
                        )
    return pd.DataFrame(
        rows,
        columns=[
            "customer_id",
            "transaction_date",
            "transaction_id",
            "product_id",
            "product_description",
            "department",
            "category",
            "price",
            "quantity",
        ],
    )


def run_pipeline(
    df: pd.DataFrame, args: argparse.Namespace, tmpdir: str | None = None
) -> tuple[str, str, tempfile.TemporaryDirectory | None]:
    """Run the analysis pipeline and return results directories.

    If tmpdir is None, creates a new temporary directory that the caller must clean up.
    Returns (res_dir, out_dir, tmpdir_obj) where tmpdir_obj is the TemporaryDirectory
    context manager (or None if caller provided tmpdir).
    """
    if tmpdir is None:
        tmpdir_obj = tempfile.TemporaryDirectory()
        tmpdir = tmpdir_obj.name
    else:
        tmpdir_obj = None

    res = Path(tmpdir) / "results"
    out = Path(tmpdir) / "insights"

    # Write input
    inp = Path(tmpdir) / "input.csv"
    df.to_csv(inp, index=False)

    # Build CLI args
    cli_args = [
        "--input",
        str(inp),
        "--output-dir",
        str(res),
        "--horizon-days",
        str(args.horizon),
        "--windows-days",
        args.windows,
        "--burn-in-days",
        str(args.burn_in),
        "--frequency-grain",
        args.grain,
        "--min-clusters",
        str(args.k_min),
        "--max-clusters",
        str(args.k_max),
        "--random-seed",
        str(args.seed),
        "--rolling-origins",
        str(args.rolling_origins),
    ]
    if args.as_of:
        cli_args += ["--as-of-date", args.as_of]
    if args.origin_spacing_days:
        cli_args += ["--origin-spacing-days", str(args.origin_spacing_days)]

    # Run analysis
    code = rca.main(cli_args)
    if code != 0:
        if tmpdir_obj:
            tmpdir_obj.cleanup()
        raise RuntimeError("The analysis script failed. Check the data and settings.")

    # Run insight engine
    ie.main(["--results-dir", str(res), "--output-dir", str(out), "--dpi", "110", "--top-n", "12"])

    return str(res), str(out), tmpdir_obj


def print_results(res_dir: str, out_dir: str):
    """Print all results from the pipeline."""
    res = Path(res_dir)
    out = Path(out_dir)

    # Load insight summary
    summary_path = out / "insight_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
    else:
        summary = {"kpis": [], "charts": [], "skipped": []}

    # ========== KPIs ==========
    print("=" * 80)
    print("KPIs (from insight_summary.json)")
    print("=" * 80)
    kpis = summary.get("kpis", {})
    if isinstance(kpis, dict):
        for label, value in kpis.items():
            print(f"  {label}: {value}")
    elif isinstance(kpis, list):
        for item in kpis:
            if isinstance(item, dict):
                label = item.get("label", item.get("key", "N/A"))
                value = item.get("value", item.get("v", "N/A"))
                print(f"  {label}: {value}")
            else:
                print(f"  {item}")
    else:
        print(f"  {kpis}")

    # ========== CHARTS ==========
    print("\n" + "=" * 80)
    print("CHARTS GENERATED (from insight_summary.json)")
    print("=" * 80)
    charts = summary.get("charts", [])
    for c in charts:
        print(f"  [{c['key']}] {c['title']}")
        print(f"    → {c['takeaway']}")

    # ========== SKIPPED CHARTS ==========
    skipped = summary.get("skipped", [])
    if skipped:
        print("\n" + "=" * 80)
        print("SKIPPED CHARTS")
        print("=" * 80)
        for s in skipped:
            print(f"  {s['chart']}: {s['reason']}")

    # ========== KEY TABLES ==========
    print("\n" + "=" * 80)
    print("KEY TABLES SUMMARY")
    print("=" * 80)

    tables_to_show = [
        ("customer_features", "customer_features.csv"),
        ("holdout_metrics", "holdout_metrics.csv"),
        ("holdout_calibration", "holdout_calibration.csv"),
        ("cohort_retention_new_customers", "cohort_retention_new_customers.csv"),
        ("cluster_diagnostics", "cluster_diagnostics.csv"),
        ("repeat_survival", "repeat_survival.csv"),
        ("customer_revenue_concentration", "customer_revenue_concentration.csv"),
        ("category_summary", "category_summary.csv"),
        ("department_summary", "department_summary.csv"),
        ("product_summary", "product_summary.csv"),
        ("basket_affinity", "basket_affinity.csv"),
        ("next_trip_affinity", "next_trip_affinity.csv"),
        ("monthly_summary", "monthly_summary.csv"),
        ("cluster_profiles", "cluster_profiles.csv"),
        ("rfm_segment_profiles", "rfm_segment_profiles.csv"),
        ("transaction_summary", "transaction_summary.csv"),
        ("data_quality_issues", "data_quality_issues.csv"),
    ]

    for name, fname in tables_to_show:
        fpath = res / fname
        if not fpath.exists():
            continue
        try:
            df = pd.read_csv(fpath)
            print(f"\n{name}: {df.shape[0]:,} rows, {df.shape[1]} cols")

            # Print key stats for specific tables
            if name == "customer_features":
                if "p_alive" in df.columns:
                    alive = (df["p_alive"] >= 0.5).sum()
                    print(
                        f"  P(alive) >= 0.5: {alive:,} / {len(df):,} ({alive / len(df) * 100:.1f}%)"
                    )
                    print(
                        f"  P(alive) stats: mean={df['p_alive'].mean():.3f}, median={df['p_alive'].median():.3f}"
                    )
                if "expected_revenue_horizon" in df.columns:
                    print(
                        f"  Expected revenue (horizon): {df['expected_revenue_horizon'].sum():,.2f}"
                    )
                if "cluster_label" in df.columns:
                    print(f"  Clusters: {df['cluster_label'].nunique()}")
                    for cluster, count in df["cluster_label"].value_counts().items():
                        print(f"  {cluster}: {count:,} ({count / len(df) * 100:.1f}%)")

            elif name == "holdout_metrics":
                if len(df) > 0:
                    cols = [
                        c
                        for c in [
                            "model",
                            "target",
                            "auc_any_purchase",
                            "mae",
                            "spearman",
                            "rmse",
                            "predicted_total",
                            "actual_total",
                        ]
                        if c in df.columns
                    ]
                    print(df[cols].to_string(index=False))

            elif name == "holdout_calibration":
                print(df.to_string(index=False))

            elif name == "cohort_retention_new_customers":
                if len(df) > 0:
                    m1 = df[(df["months_since_cohort"] == 1) & (df["period_complete"])]
                    if len(m1):
                        print(
                            m1[["cohort_month", "cohort_size", "retention_rate"]].to_string(
                                index=False
                            )
                        )

            elif name == "cluster_diagnostics":
                cols = [
                    c
                    for c in [
                        "candidate_k",
                        "silhouette",
                        "stability_ari_mean",
                        "selected",
                        "selection_reason",
                    ]
                    if c in df.columns
                ]
                print(df[cols].to_string(index=False))

            elif name == "repeat_survival":
                if len(df) > 0:
                    s90 = df[df["day"] == 90]
                    if len(s90):
                        print(
                            s90[
                                [
                                    "group",
                                    "customers",
                                    "repeat_probability",
                                    "repeat_ci95_lower",
                                    "repeat_ci95_upper",
                                ]
                            ].to_string(index=False)
                        )

            elif name == "customer_revenue_concentration":
                print(df.to_string(index=False))

            elif name == "category_summary":
                if len(df) > 0:
                    cols = [
                        c
                        for c in [
                            "category",
                            "gross_purchase_revenue",
                            "return_value",
                            "net_revenue",
                        ]
                        if c in df.columns
                    ]
                    print(df.nlargest(10, "gross_purchase_revenue")[cols].to_string(index=False))

            elif name == "department_summary":
                cols = [
                    c
                    for c in ["department", "gross_purchase_revenue", "return_value", "net_revenue"]
                    if c in df.columns
                ]
                print(df[cols].to_string(index=False))

            elif name == "product_summary":
                if len(df) > 0:
                    cols = [
                        c
                        for c in [
                            "product_id",
                            "product_description",
                            "gross_purchase_revenue",
                            "repeat_buyer_rate",
                        ]
                        if c in df.columns
                    ]
                    print(df.nlargest(10, "gross_purchase_revenue")[cols].to_string(index=False))

            elif name == "basket_affinity":
                if len(df) > 0 and "passes_all_filters" in df.columns:
                    passed = df[df["passes_all_filters"].astype(bool)]
                    print(f"  {len(passed):,} / {len(df):,} rules pass all filters")
                    if len(passed) > 0:
                        cols = [
                            c
                            for c in [
                                "antecedent",
                                "consequent",
                                "customer_lift",
                                "confidence",
                                "q_value_bh",
                            ]
                            if c in passed.columns
                        ]
                        print(passed.nlargest(5, "customer_lift")[cols].to_string(index=False))

            elif name == "next_trip_affinity":
                if len(df) > 0:
                    cols = [
                        c
                        for c in [
                            "from_item",
                            "next_trip_item",
                            "next_trip_lift",
                            "p_next_given_from",
                            "q_value_bh",
                        ]
                        if c in df.columns
                    ]
                    print(df.nlargest(5, "next_trip_lift")[cols].to_string(index=False))

            elif name == "monthly_summary":
                print(df.to_string(index=False))

            elif name == "cluster_profiles":
                print(df.to_string(index=False))

            elif name == "rfm_segment_profiles":
                print(df.to_string(index=False))

            elif name == "transaction_summary":
                if len(df) > 0:
                    print(f"  Purchase transactions: {df['is_purchase_transaction'].sum():,}")
                    print(f"  Distinct customers: {df['distinct_customers'].sum():,}")

            elif name == "data_quality_issues":
                print(df.to_string(index=False))

        except Exception as e:
            print(f"\n{name}: Error reading - {e}")

    print("\n" + "=" * 80)
    print("DONE - All results displayed above")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Run retail customer analytics pipeline and display all results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", help="Input CSV or Parquet file")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate synthetic demo data instead of loading file",
    )
    parser.add_argument(
        "--n-customers", type=int, default=500, help="Number of synthetic customers"
    )
    parser.add_argument("--days", type=int, default=180, help="Days of synthetic history")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--horizon", type=int, default=90, help="Forecast horizon in days")
    parser.add_argument("--windows", default="30,90,365", help="Rolling windows (comma-separated)")
    parser.add_argument("--burn-in", type=int, default=30, help="Burn-in days for cohort analysis")
    parser.add_argument(
        "--grain", choices=["trip", "transaction"], default="trip", help="Frequency grain"
    )
    parser.add_argument("--k-min", type=int, default=3, help="Min clusters")
    parser.add_argument("--k-max", type=int, default=7, help="Max clusters")
    parser.add_argument("--as-of", help="Analysis cutoff date (YYYY-MM-DD)")
    parser.add_argument(
        "--rolling-origins",
        type=int,
        default=1,
        help="Number of rolling-origin validation cutoffs (1 = single holdout)",
    )
    parser.add_argument(
        "--origin-spacing-days",
        type=int,
        default=None,
        help="Days between rolling origins (default: horizon-days)",
    )

    args = parser.parse_args()

    if not args.input and not args.synthetic:
        parser.error("Either --input or --synthetic must be specified")

    if args.synthetic:
        print(
            f"Generating synthetic data: {args.n_customers} customers, {args.days} days, seed={args.seed}"
        )
        df = generate_synthetic_data(args.n_customers, args.days, args.seed)
    else:
        print(f"Loading data from: {args.input}")
        df = load_input_data(args.input)

    print(f"Input data: {len(df):,} rows, {df['customer_id'].nunique():,} customers")
    print(f"Date range: {df['transaction_date'].min()} to {df['transaction_date'].max()}")

    try:
        res_dir, out_dir, tmpdir_obj = run_pipeline(df, args)
        print_results(res_dir, out_dir)
    except Exception as e:
        print(f"Pipeline failed: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        # Cleanup is handled by TemporaryDirectory context manager in run_pipeline
        pass


if __name__ == "__main__":
    main()
