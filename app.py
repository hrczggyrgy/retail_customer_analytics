"""Customer Insight Lab: Streamlit UI for retail_customer_analysis.py and insight_engine.py.

Run
---
    pip install streamlit pandas numpy scipy matplotlib
    streamlit run app.py

Keep app.py, retail_customer_analysis.py and insight_engine.py in the same folder.

Flow
----
1. Choose data: upload a line-item CSV/Parquet (columns can be mapped) or generate synthetic demo data.
2. Press "Run analysis": the pipeline cleans data, fits the models, validates them on a time holdout,
   then the insight engine draws the charts.
3. Read the results in order: Overview -> Can we trust it? -> Customers -> Behaviour -> Products and baskets
   -> Customer explorer -> Data and downloads.

Notes
-----
Expected values are revenue forecasts, not profit. Basket rules are associations, not causal effects.
Promo share is a price-based proxy. Synthetic data is generated from a BG/NBD-like process, so model
fit on it only shows that the machinery works, not that it fits real customers.
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

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

REQUIRED = ["customer_id", "transaction_date", "transaction_id", "product_id", "product_description",
            "department", "category", "price", "quantity"]

TABS = ["1 Overview", "2 Can we trust it?", "3 Customers", "4 Behaviour", "5 Products and baskets", "6 Customer explorer", "7 Data and downloads"]
CHARTS_BY_TAB = {
    "trust": ["01_model_validation", "02_calibration", "05_cluster_selection"],
    "customers": ["03_customer_map", "04_segment_value", "06_rfm_segments", "09_revenue_concentration", "18_lorenz_segments", "19_action_quadrant"],
    "behaviour": ["07_repeat_survival", "08_cohort_retention", "14_momentum", "16_cadence", "15_promo_proxy", "22_cohort_ltv"],
    "products": ["10_revenue_trend", "11_categories", "12_basket_rules", "13_next_trip", "20_affinity_network", "21_promo_return_lollipop"],
    "data": ["17_data_quality"],
}
HEADLINE_CHARTS = ["01_model_validation", "03_customer_map", "04_segment_value", "07_repeat_survival", "09_revenue_concentration", "12_basket_rules", "18_lorenz_segments", "19_action_quadrant"]
SEGMENT_HINTS = {
    "Active / higher": "Hypothesis: protect and grow. Test retention perks against a holdout group.",
    "Active / lower": "Hypothesis: raise basket value or breadth. Test cross-category offers.",
    "Lapsing": "Hypothesis: win-back candidates. Test timed reminders against a holdout group.",
    "Lapsed": "Hypothesis: low-cost reactivation only. Check cost against expected revenue first.",
}

# ============================================================
# UI Helpers - layout, charts, KPIs
# ============================================================
def section(title: str, question: str = "") -> None:
    """Render a section header with optional business question."""
    st.subheader(title)
    if question:
        st.caption(question)

def chart_card(fig, takeaway: str, *, key: str = "") -> None:
    """Render a matplotlib figure with a takeaway caption and close the figure."""
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
    cols = st.columns(len(metrics))
    for col, (label, value, delta) in zip(cols, metrics):
        col.metric(label, value, delta=delta)

def metric_card(label: str, value: str, delta: str | None = None, help_text: str = "") -> None:
    """Single metric card using st.metric."""
    st.metric(label, value, delta=delta, help=help_text)

def two_col_chart_text(fig_left, takeaway_left: str, text_right: str, key_left: str = "") -> None:
    """Two-column layout: chart on left, commentary on right."""
    c1, c2 = st.columns([2, 1])
    with c1:
        chart_card(fig_left, takeaway_left, key=key_left)
    with c2:
        st.markdown(text_right)

def expander_table(df: pd.DataFrame, title: str, max_rows: int = 500) -> None:
    """Show a dataframe in an expander with pretty column names."""
    with st.expander(title):
        show_df(pretty(df.head(max_rows)))

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

def filter_sidebar(df: pd.DataFrame) -> pd.DataFrame:
    """Add global filters in sidebar and return filtered dataframe."""
    if df.empty:
        return df
    
    with st.sidebar.expander("📊 Global Filters", expanded=False):
        # Date range filter
        if "transaction_day" in df.columns and df["transaction_day"].notna().any():
            min_date = df["transaction_day"].min()
            max_date = df["transaction_day"].max()
            date_range = st.date_input(
                "Date range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
            )
            if len(date_range) == 2:
                df = df[(df["transaction_day"] >= pd.Timestamp(date_range[0])) & 
                        (df["transaction_day"] <= pd.Timestamp(date_range[1]))]
        
        # Department filter
        if "department" in df.columns:
            depts = sorted(df["department"].dropna().unique())
            sel_depts = st.multiselect("Department", depts, default=depts)
            if sel_depts:
                df = df[df["department"].isin(sel_depts)]
        
        # Category filter
        if "category" in df.columns:
            cats = sorted(df["category"].dropna().unique())
            sel_cats = st.multiselect("Category", cats, default=cats)
            if sel_cats:
                df = df[df["category"].isin(sel_cats)]
        
        # Segment filter
        if "cluster_label" in df.columns:
            segs = sorted(df["cluster_label"].dropna().unique())
            sel_segs = st.multiselect("Segment", segs, default=segs)
            if sel_segs:
                df = df[df["cluster_label"].isin(sel_segs)]
        
        # Cohort granularity
        cohort_grain = st.selectbox("Cohort granularity", ["Monthly", "Quarterly"], index=0)
        st.session_state["cohort_grain"] = cohort_grain
        
        # Compare-to period
        compare_mode = st.selectbox("Compare to", ["Prior period", "Same period last year", "None"], index=2)
        st.session_state["compare_mode"] = compare_mode
    
    return df


# --------------------------------------------------------------------------- #
# Synthetic demo data
# --------------------------------------------------------------------------- #
def make_synthetic(n_customers: int = 3000, days: int = 730, seed: int = 7, missing_id_rate: float = 0.03,
                   return_rate: float = 0.03, promo_share: float = 0.15, n_categories: int = 12,
                   start: str = "2024-01-01") -> pd.DataFrame:
    """Line-item data from a BG/NBD-like process with category preferences, promotions, returns and split receipts."""
    rng = np.random.default_rng(seed)
    k = int(n_categories)
    cats = [f"Category {i:02d}" for i in range(k)]
    depts = [f"Department {i % 3 + 1}" for i in range(k)]
    prod_per_cat = 10
    base_price = {c: float(rng.lognormal(np.log(4 + 1.6 * c), 0.25)) for c in range(k)}
    products = {c: [(f"P{c:02d}{j:02d}", round(base_price[c] * float(rng.uniform(0.7, 1.4)), 2)) for j in range(prod_per_cat)] for c in range(k)}
    lam = rng.gamma(0.8, 1 / 12.0, n_customers)
    p_drop = rng.beta(0.8, 2.5, n_customers)
    first = rng.integers(0, max(int(days * 0.7), 1), n_customers)
    pref = rng.dirichlet(np.ones(k) * 0.4, n_customers)
    size_factor = rng.gamma(2.0, 0.8, n_customers)
    t0 = pd.Timestamp(start)
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
                if rng.random() < promo_share:
                    price = round(price * 0.8, 2)
                qty = int(1 + rng.poisson(0.6))
                cid = None if rng.random() < missing_id_rate else f"C{i:05d}"
                tid = f"T{i:05d}_{int(t)}_{receipt}"
                rows.append((cid, day.strftime("%Y-%m-%d"), tid, pid, f"Product {pid}", depts[c], cats[c], price, qty))
                if rng.random() < return_rate:
                    rday = day + pd.Timedelta(days=int(rng.integers(1, 15)))
                    if (rday - t0).days < days:
                        rows.append((cid, rday.strftime("%Y-%m-%d"), tid + "R", pid, f"Product {pid}", depts[c], cats[c], price, -1))
    return pd.DataFrame(rows, columns=REQUIRED)


# --------------------------------------------------------------------------- #
# Upload handling
# --------------------------------------------------------------------------- #
def read_upload(file: Any) -> pd.DataFrame:
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file, dtype=str, keep_default_na=False, na_values=["", "NULL", "null"], encoding_errors="replace")
    if name.endswith((".parquet", ".pq")):
        return pd.read_parquet(file)
    raise ValueError("Please upload a .csv, .parquet or .pq file.")


def auto_map(columns: list[str]) -> dict[str, str | None]:
    norm = {c.strip().lower().replace(" ", "_"): c for c in columns}
    aliases = {"customer_id": ["customer_id", "customerid", "customer", "cust_id", "client_id", "loyalty_id"],
               "transaction_date": ["transaction_date", "date", "order_date", "purchase_date", "datetime", "timestamp"],
               "transaction_id": ["transaction_id", "basket_id", "order_id", "receipt_id", "invoice_id", "invoiceno"],
               "product_id": ["product_id", "sku", "item_id", "stockcode", "article_id"],
               "product_description": ["product_description", "description", "product_name", "item_name"],
               "department": ["department", "dept", "division"],
               "category": ["category", "product_category", "commodity", "sub_category"],
               "price": ["price", "unit_price", "unitprice"], "quantity": ["quantity", "qty", "units"]}
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
    a = ["--horizon-days", str(p["horizon"]), "--burn-in-days", str(p["burn_in"]), "--windows-days", parse_windows(p["windows"]),
         "--recent-window-days", str(p["recent"]), "--frequency-grain", p["grain"], "--transaction-key-mode", p["key_mode"],
         "--basket-level", p["basket_level"], "--min-clusters", str(p["k_min"]), "--max-clusters", str(p["k_max"]),
         "--random-seed", str(p["seed"])]
    if p["as_of"]:
        a += ["--as-of-date", p["as_of"]]
    for flag, key in (("--drop-exact-duplicates", "drop_dupes"), ("--exclude-zero-quantity", "excl_zero"), ("--exclude-negative-prices", "excl_neg")):
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
        raise RuntimeError("The analysis script failed. Check the data (dates, price, quantity) and the settings.")
    if progress:
        progress("Drawing charts...")
    out = res / "insights"
    ie_code = ie.main(["--results-dir", str(res), "--output-dir", str(out), "--dpi", "110", "--top-n", "12"])
    summary = json.loads((out / "insight_summary.json").read_text(encoding="utf-8")) if (out / "insight_summary.json").is_file() else {"charts": [], "kpis": {}, "skipped": []}
    meta = json.loads((res / "run_metadata.json").read_text(encoding="utf-8")) if (res / "run_metadata.json").is_file() else {}
    return {"work": str(work), "results": str(res), "insights": str(out), "summary": summary, "meta": meta,
            "seconds": round(time.time() - t0, 1), "rows": int(len(df)), "chart_exit": ie_code}


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
            kpis.append(("Likely active (P(alive) ≥ 0.5)", f"{(c['p_alive'] >= 0.5).mean()*100:.0f}%", None))
        if "expected_revenue_horizon" in c:
            e = c["expected_revenue_horizon"].clip(lower=0)
            kpis.append(("Expected revenue, next horizon", f"{e.sum():,.0f}", None))
            if len(e) >= 10 and e.sum() > 0:
                top10 = e.nlargest(max(1, int(np.ceil(len(e) * 0.1)))).sum() / e.sum()
                kpis.append(("Top 10% share", f"{top10*100:.0f}%", None))
    
    s = tables.get("repeat_survival", pd.DataFrame())
    if not s.empty:
        r = s[(s["group"] == "all_customers") & (s["day"] == 90)]
        if len(r):
            kpis.append(("2nd trip within 90d", f"{r.iloc[0]['repeat_probability']*100:.0f}%", None))
    
    m = tables.get("holdout_metrics", pd.DataFrame())
    if not m.empty:
        bg = m[m["model"] == "BG/NBD P(alive)"]["auc_any_purchase"]
        lg = m[m["model"] == "legacy_cadence_heuristic"]["auc_any_purchase"]
        if len(bg) and len(lg):
            kpis.append(("AUC: BG/NBD vs legacy", f"{bg.iloc[0]:.2f} vs {lg.iloc[0]:.2f}", f"{bg.iloc[0]-lg.iloc[0]:+.2f}"))
    
    a = tables.get("basket_affinity", pd.DataFrame())
    if not a.empty and "passes_all_filters" in a:
        kpis.append(("Supported basket rules", f"{int(a['passes_all_filters'].astype(bool).sum()):,}", None))
    
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
    for extra in ({"width": "stretch"}, {"use_container_width": True}, {"use_column_width": True}, {}):
        try:
            st.image(str(path), **extra)
            return
        except Exception:
            continue


def table(R: dict[str, Any], stem: str) -> pd.DataFrame:
    p = next((Path(R["results"]) / f"{stem}.{e}" for e in ("csv", "parquet") if (Path(R["results"]) / f"{stem}.{e}").is_file()), None)
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
def sidebar() -> dict[str, Any]:
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
            p["missing_id"] = float(st.slider("Rows without customer id", 0.0, 0.20, 0.03, step=0.01))
            p["return_rate"] = float(st.slider("Return rate (lines)", 0.0, 0.15, 0.03, step=0.01))
            p["promo_share"] = float(st.slider("Promo-priced lines", 0.0, 0.50, 0.15, step=0.05))
    else:
        p["file"] = sb.file_uploader("Line-item CSV or Parquet", type=["csv", "parquet", "pq"])
        sb.caption("Needs one row per line item with: " + ", ".join(REQUIRED) + ". Negative quantity = return.")
    with sb.expander("2. Analysis settings"):
        p["horizon"] = int(st.number_input("Forecast and holdout horizon (days)", 14, 365, 90, step=15))
        p["burn_in"] = int(st.number_input("Burn-in for 'new customer' (days)", 14, 365, 90, step=15))
        p["grain"] = st.selectbox("Purchase event", ["trip", "transaction"], index=0, help="trip = one customer-day (merges split receipts)")
        p["basket_level"] = st.selectbox("Basket analysis level", ["category", "product"], index=0)
        p["k_min"], p["k_max"] = (int(v) for v in st.slider("Segment count range (k)", 2, 10, (3, 7)))
        p["windows"] = st.text_input("Rolling windows (days)", "30,90,365")
        p["recent"] = int(st.number_input("Recent vs prior window (days)", 14, 365, 90, step=15))
        use_as_of = st.checkbox("Set analysis cut-off date manually", value=False)
        p["as_of"] = st.date_input("Cut-off date").isoformat() if use_as_of else None
        p["key_mode"] = st.selectbox("Transaction id scope", ["customer-transaction", "transaction-only"], index=0)
        p["drop_dupes"] = st.checkbox("Drop exact duplicate rows", value=False)
        p["excl_zero"] = st.checkbox("Exclude zero-quantity rows", value=False)
        p["excl_neg"] = st.checkbox("Exclude negative-price rows", value=False)
        p["seed"] = int(st.number_input("Analysis random seed", 0, 10_000, 42))
    return p


@st.cache_data(show_spinner=False)
def cached_synthetic(n: int, days: int, seed: int, missing: float, ret: float, promo: float) -> pd.DataFrame:
    return make_synthetic(n, days, seed, missing, ret, promo)


def data_step(p: dict[str, Any]) -> tuple[pd.DataFrame | None, str | None]:
    """Return (dataframe ready to analyse or None, message). Handles preview and column mapping."""
    if p["source"] == "Synthetic demo data":
        st.info(f"Synthetic data will be generated when you press **Run analysis**: {p['n_customers']:,} customers over {p['days']} days, "
                "with category preferences, promotions, returns, split receipts and some missing customer ids.")
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
        return None, "Map all required columns to continue (a column can be mapped to 'none' only if you add it to the file)."
    used = [mapping[r] for r in REQUIRED]
    if len(set(used)) < len(used):
        return None, "Each required column must map to a different source column."
    df = raw[used].copy()
    df.columns = REQUIRED
    return df, None


# --------------------------------------------------------------------------- #
# Result tabs
# --------------------------------------------------------------------------- #
def tab_overview(R: dict[str, Any]) -> None:
    st.header("Overview")
    st.caption("Question answered here: what is the headline picture of this customer base?")
    
    # Load and cache chart data
    chart_data = load_chart_data(R["results"], HEADLINE_CHARTS)
    kpis = compute_kpis_from_tables(chart_data)
    
    # KPI Row
    kpi_row(kpis)
    
    # Revenue Bridge Waterfall - key finding
    st.markdown("---")
    section("Revenue Bridge: What changed?", "Prior period → New, Retained growth, Retained shrink, Reactivated, Churned, Returns → Current period")
    
    # Show headline charts
    st.subheader("Key findings")
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
        keep = ["horizon_days", "burn_in_days", "frequency_grain", "basket_level", "min_clusters", "max_clusters", "windows_days", "as_of_date"]
        st.json({k: cfg.get(k) for k in keep if k in cfg})
    
    with st.expander("How to read these results"):
        st.markdown(
            "- **P(alive)** is the model's probability that a customer is still active; it is not a promise to buy.\n"
            "- **Expected revenue** is trips × value per trip over the horizon. It is revenue, not profit, and is not discounted.\n"
            "- **Segments** describe behaviour. They are useful only if they are stable and differ on future behaviour.\n"
            "- **Basket rules** and **transitions** are associations, not causal effects.\n"
            "- **Promo share** is a price-based proxy; no promotion field exists in the data.")


def tab_trust(R: dict[str, Any]) -> None:
    st.header("Can we trust it?")
    st.caption("The models were fitted on older data and scored on the following period, against simple baselines.")
    
    chart_data = load_chart_data(R["results"], CHARTS_BY_TAB["trust"])
    kpis = compute_kpis_from_tables(chart_data)
    kpi_row(kpis)
    
    st.markdown("---")
    
    # Model Validation
    section("Model Validation: Does the model beat simple rules on unseen data?", "AUC for who buys again, MAE for trip/revenue forecasts. Higher AUC / Lower MAE = better")
    show_charts(R, CHARTS_BY_TAB["trust"])
    
    # Calibration
    st.markdown("---")
    section("Forecast Calibration: Are predicted probabilities reliable?", "Decile calibration plot - points on the diagonal = well calibrated")
    show_charts(R, ["02_calibration"])
    
    # Cluster Stability
    st.markdown("---")
    section("Segment Stability: Are segments real or an artefact?", "Silhouette (separation) and Bootstrap ARI (stability) across k values")
    show_charts(R, ["05_cluster_selection"])
    
    # Reconciliation Table
    st.markdown("---")
    section("Accounting Reconciliation: Do the numbers add up?", "Gross - Returns = Net at every aggregation level")
    m = table(R, "model_parameters")
    if len(m):
        st.subheader("Model Parameters")
        show_df(pretty(m))
    
    # Synthetic parameter recovery
    if R.get("source_label") == "Synthetic demo data":
        st.markdown("---")
        section("Synthetic Recovery: Did we recover the true parameters?", "Only available for synthetic data with known ground truth")
        st.info("Synthetic parameter recovery chart available when running on generated data with ground truth")
    
    for stem, title in (("holdout_metrics", "Holdout metrics"), ("holdout_calibration", "Calibration by decile"), ("cluster_diagnostics", "Cluster selection diagnostics"), ("model_parameters", "Fitted model parameters")):
        t = table(R, stem)
        if len(t):
            with st.expander(title):
                show_df(pretty(t))


def segment_profile(cf: pd.DataFrame) -> pd.DataFrame:
    if cf.empty or "cluster_label" not in cf.columns:
        return pd.DataFrame()
    g = cf.groupby(cf["cluster_label"].fillna("Unclustered"))
    out = g.agg(customers=("p_alive", "size"), mean_p_alive=("p_alive", "mean"), expected_trips=("expected_trips_horizon", "mean"),
                expected_value_per_trip=("expected_trip_value", "mean"), total_expected_revenue=("expected_revenue_horizon", "sum")).reset_index()
    if "top_category_by_spend" in cf.columns:
        mode = g["top_category_by_spend"].agg(lambda s: s.mode().iloc[0] if s.notna().any() else None).rename("most_common_top_category")
        out = out.merge(mode.reset_index(), on="cluster_label", how="left")
    for col in ("promo_spend_share", "return_value_ratio", "recency_days"):
        if col in cf.columns:
            out = out.merge(g[col].mean().rename(f"mean_{col}").reset_index(), on="cluster_label", how="left")
    out["share_of_customers"] = out["customers"] / out["customers"].sum()
    tot = out["total_expected_revenue"].sum()
    out["share_of_expected_revenue"] = out["total_expected_revenue"] / tot if tot > 0 else np.nan
    out["hypothesis_to_test"] = [next((v for k, v in SEGMENT_HINTS.items() if str(lab).startswith(k)), "Describe first; test actions against a holdout group.") for lab in out["cluster_label"]]
    return out.sort_values("total_expected_revenue", ascending=False)


def tab_customers(R: dict[str, Any]) -> None:
    st.header("Customers")
    st.caption("Question answered here: who are the customers, how active are they, and where does future revenue sit?")
    
    chart_data = load_chart_data(R["results"], CHARTS_BY_TAB["customers"])
    kpis = compute_kpis_from_tables(chart_data)
    kpi_row(kpis)
    
    st.markdown("---")
    
    # Segment z-score heatmap
    section("Segment Identity: What defines each segment?", "Z-scored metrics across segments - blue = below average, red = above average")
    
    cf = table(R, "customer_features")
    prof = segment_profile(cf)
    if len(prof):
        # Z-score heatmap
        metrics_for_heatmap = [c for c in prof.columns if c.startswith(("mean_", "share_", "expected_")) and c not in ("mean_recency_days",)]
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
            for i, (tick, label) in enumerate(zip(ax.get_xticklabels(), z_df.index)):
                tick.set_color(segment_color(label))
            ax.set_yticks(range(len(z_df.columns)))
            ax.set_yticklabels([c.replace("mean_", "").replace("share_of_", "").replace("_", " ").title() for c in z_df.columns], fontsize=8)
            ax.set_title("Segment Profile Heatmap (Z-scores)")
            fig.colorbar(im, ax=ax, label="Z-score")
            chart_card(fig, f"Segments are most differentiated on {z_df.std(axis=1).idxmax() if len(z_df) else 'N/A'}.")
        
        st.subheader("Segment profiles")
        st.caption("Hypotheses are starting points to test with a holdout group, not conclusions.")
        show_df(pretty(prof))
    
    # P(alive) vs Expected Revenue Quadrant
    st.markdown("---")
    section("Action Quadrant: Who to protect, grow, win back, or ignore?", "X = P(alive), Y = Expected revenue. Top-right = protect, top-left = grow, bottom-right = win back, bottom-left = ignore")
    
    show_charts(R, ["03_customer_map", "04_segment_value", "06_rfm_segments", "09_revenue_concentration"])
    
    for stem, title in (("rfm_segment_profiles", "RFM segment table"), ("customer_revenue_concentration", "Revenue concentration figures")):
        t = table(R, stem)
        if len(t):
            with st.expander(title):
                show_df(pretty(t))


def tab_behaviour(R: dict[str, Any]) -> None:
    st.header("Behaviour")
    st.caption("Question answered here: how do customers come back, how regular are they, and who is changing?")
    
    chart_data = load_chart_data(R["results"], CHARTS_BY_TAB["behaviour"])
    kpis = compute_kpis_from_tables(chart_data)
    kpi_row(kpis)
    
    st.markdown("---")
    
    # Cohort Retention Heatmap
    section("Cohort Retention: Do newer cohorts stay active?", "Rows = first-purchase month, Cols = months since first purchase. Darker = higher retention")
    show_charts(R, ["08_cohort_retention"])
    
    # Cohort Revenue per Customer (LTV)
    st.markdown("---")
    section("Cohort Revenue & Retention", "Cohort retention heatmap + revenue quality")
    show_charts(R, ["22_cohort_ltv"])
    
    # Time-to-second-purchase survival curves
    st.markdown("---")
    section("Activation Speed: Time to second purchase by segment", "Kaplan-Meier survival curves - faster rise = faster activation")
    show_charts(R, ["07_repeat_survival"])
    
    # Inter-purchase interval violins
    st.markdown("---")
    section("Purchase Rhythm: How regular are customers?", "Inter-purchase interval distribution by segment")
    show_charts(R, ["16_cadence"])
    
    # Momentum chart
    st.markdown("---")
    section("Momentum: Which segments are growing or shrinking?", "Recent vs prior period spend change by segment")
    show_charts(R, ["14_momentum"])
    
    for stem, title in (("repeat_survival", "Second-trip probabilities (all landmarks)"), ("cohort_retention_new_customers", "Cohort retention table")):
        t = table(R, stem)
        if len(t):
            with st.expander(title):
                show_df(pretty(t))


def tab_products(R: dict[str, Any]) -> None:
    st.header("Products and baskets")
    st.caption("Question answered here: what sells, what is returned, and what is bought together or next?")
    
    chart_data = load_chart_data(R["results"], CHARTS_BY_TAB["products"])
    kpis = compute_kpis_from_tables(chart_data)
    kpi_row(kpis)
    
    st.markdown("---")
    
    # Affinity Network
    section("Affinity Network: Cross-sell structure", "Network graph of statistically significant basket rules (FDR-controlled)")
    show_charts(R, ["20_affinity_network"])
    
    aff = table(R, "basket_affinity")
    if len(aff) and "passes_all_filters" in aff.columns:
        ok = aff[aff["passes_all_filters"].astype(bool)]
        st.subheader("Supported basket rules")
        st.caption(f"{len(ok):,} of {len(aff):,} directed rules pass FDR control, an odds-ratio interval above 1 and split-half persistence.")
        if len(ok):
            show_df(pretty(ok))
    
    # Category Pareto
    st.markdown("---")
    section("Category Pareto: Assortment concentration", "Top categories by revenue with cumulative share line")
    show_charts(R, ["11_categories"])
    
    # Margin Risk: Return rate by category
    st.markdown("---")
    section("Margin Risk: Return rate by category", "Lollipop chart showing return rate for top categories")
    show_charts(R, ["21_promo_return_lollipop"])
    
    # Next-trip transitions
    st.markdown("---")
    section("Next-Trip Transitions: What do customers buy next?", "Significant transitions that beat chance (FDR-controlled)")
    nxt = table(R, "next_trip_affinity")
    if len(nxt):
        with st.expander("Next-trip transitions"):
            show_df(pretty(nxt))
    
    # Basket rules lift matrix
    show_charts(R, ["12_basket_rules", "13_next_trip"])
    
    for stem, title in (("category_summary", "Category summary"), ("department_summary", "Department summary"), ("product_summary", "Product summary")):
        t = table(R, stem)
        if len(t):
            with st.expander(title):
                show_df(pretty(t.head(500)))


def tab_explorer(R: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt
    st.header("Customer explorer")
    st.caption("Filter customers, export a target list, or look up a single customer. Build your own view.")
    
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
    x_dim = c2.selectbox("X dimension (categorical)", cat_cols, index=cat_cols.index("cluster_label") if "cluster_label" in cat_cols else 0)
    y_metric = c3.selectbox("Y metric (numeric)", numeric_cols, index=numeric_cols.index("expected_revenue_horizon") if "expected_revenue_horizon" in numeric_cols else 0)
    
    if chart_type == "Scatter":
        x_metric = c1.selectbox("X metric (numeric)", numeric_cols, index=numeric_cols.index("p_alive") if "p_alive" in numeric_cols else 0)
        color_dim = c2.selectbox("Color by", cat_cols, index=cat_cols.index("cluster_label") if "cluster_label" in cat_cols else 0)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        for lab in sorted(cf[color_dim].dropna().unique()):
            s = cf[cf[color_dim] == lab]
            ax.scatter(s[x_metric], s[y_metric], alpha=0.5, label=str(lab)[:30], color=segment_color(lab), s=15)
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
            bp = ax.boxplot(data, tick_labels=[str(g)[:20] for g in groups], patch_artist=True, showfliers=False)
            for patch, g in zip(bp["boxes"], groups):
                patch.set_facecolor(segment_color(g))
                patch.set_alpha(0.6)
            ax.set_ylabel(y_metric.replace("_", " ").title())
            ax.set_title(f"{y_metric} distribution by segment")
            chart_card(fig, f"Segment {max(groups, key=lambda g: cf[cf['cluster_label']==g][y_metric].median())} has highest median {y_metric}")
    
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
        comp_df = pd.DataFrame({seg_a: comp_a.round(2), seg_b: comp_b.round(2), "Diff %": diff}).reset_index()
        comp_df.columns = ["Metric", seg_a, seg_b, "Diff %"]
        show_df(pretty(comp_df))
    
    st.markdown("---")
    
    # Existing filter and export
    st.subheader("Filter & Export")
    c1, c2, c3 = st.columns(3)
    sel_cl = c1.multiselect("Segment", labels, default=labels)
    sel_rfm = c2.multiselect("RFM group", rfm, default=rfm)
    lo, hi = c3.slider("P(alive) range", 0.0, 1.0, (0.0, 1.0), step=0.05)
    min_rev = st.number_input("Minimum expected revenue", 0.0, float(max(cf["expected_revenue_horizon"].max(), 1.0)) if "expected_revenue_horizon" in cf else 1.0, 0.0)
    d = cf.copy()
    if labels:
        d = d[d["cluster_label"].isin(sel_cl) | d["cluster_label"].isna() & ("Unclustered" in sel_cl)]
    if rfm:
        d = d[d["rfm_segment"].isin(sel_rfm)]
    if "p_alive" in d:
        d = d[(d["p_alive"].fillna(0) >= lo) & (d["p_alive"].fillna(0) <= hi)]
    if "expected_revenue_horizon" in d:
        d = d[d["expected_revenue_horizon"].fillna(0) >= min_rev].sort_values("expected_revenue_horizon", ascending=False)
    default_cols = [c for c in ["customer_id", "cluster_label", "rfm_segment", "p_alive", "expected_trips_horizon", "expected_trip_value", "expected_revenue_horizon",
                                "n_trips", "recency_days", "gross_spend", "return_value_ratio", "promo_spend_share", "top_category_by_spend"] if c in d.columns]
    cols = st.multiselect("Columns", list(d.columns), default=default_cols)
    st.write(f"**{len(d):,}** customers match. Showing the first 1,000.")
    show_df(d[cols].head(1000) if cols else d.head(1000))
    st.download_button("Download filtered customers (CSV)", d[cols].to_csv(index=False).encode("utf-8") if cols else d.to_csv(index=False).encode("utf-8"),
                       file_name="filtered_customers.csv", mime="text/csv")
    
    st.subheader("Customer lookup")
    q = st.text_input("Customer id")
    if q:
        hit = cf[cf["customer_id"].astype(str) == q.strip()]
        if hit.empty:
            st.warning("No customer with that id in the scored table.")
        else:
            show_df(hit.T.reset_index().rename(columns={"index": "field", hit.index[0]: "value"}).astype(str))


def tab_data(R: dict[str, Any]) -> None:
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
    c1.download_button("All result tables (ZIP)", zip_dir(R["results"]), file_name="customer_insights_tables.zip", mime="application/zip")
    c2.download_button("All charts (ZIP)", zip_dir(R["insights"]), file_name="customer_insights_charts.zip", mime="application/zip")
    html_path = Path(R["insights"]) / "insights.html"
    if html_path.is_file():
        c3.download_button("Dashboard (HTML)", html_path.read_bytes(), file_name="insights.html", mime="text/html")
    rp = Path(R["results"]) / "analysis_report.md"
    if rp.is_file():
        c4.download_button("Written report (Markdown)", rp.read_bytes(), file_name="analysis_report.md", mime="text/markdown")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def landing() -> None:
    st.title("Customer Insight Lab")
    st.markdown("Turn line-item purchase data into a validated picture of your customers, using only the data you already have.")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**1. Choose data**\n\nUpload a CSV or Parquet file, or try the synthetic demo.")
    c2.markdown("**2. Run analysis**\n\nData checks, probabilistic customer models, time-based validation, stable segments, basket tests.")
    c3.markdown("**3. Read in order**\n\nOverview, trust, customers, behaviour, products, explorer, downloads.")
    with st.expander("Required columns"):
        show_df(pd.DataFrame({"column": REQUIRED, "meaning": ["customer key (blank = anonymous)", "purchase date", "receipt or order id", "product key",
                                                                "product name", "department", "category", "unit price", "units; negative = return"]}))


def main() -> None:
    st.set_page_config(page_title="Customer Insight Lab", page_icon=None, layout="wide")
    try:
        import insight_engine  # noqa: F401
        import retail_customer_analysis  # noqa: F401
    except ImportError as exc:
        st.error(f"Missing module: {exc}. Keep app.py, retail_customer_analysis.py and insight_engine.py in the same folder.")
        st.stop()
        return
    p = sidebar()
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
                    df = cached_synthetic(p["n_customers"], p["days"], p["syn_seed"], p["missing_id"], p["return_rate"], p["promo_share"])
            with st.status("Running analysis...", expanded=True) as status:
                R = run_pipeline(df, p, progress=lambda m: st.write(m))
                status.update(label=f"Done in {R['seconds']} seconds", state="complete")
            old = st.session_state.get("result")
            if old and old.get("work") != R["work"]:
                shutil.rmtree(old["work"], ignore_errors=True)
            st.session_state["result"] = R
            st.session_state["source_label"] = p["source"]
        except Exception as exc:
            st.error(f"{type(exc).__name__}: {exc}")
    R = st.session_state.get("result")
    if not R:
        return
    st.caption(f"Showing results for: {st.session_state.get('source_label', 'data')}  |  {R['rows']:,} rows")
    tabs = st.tabs(TABS)
    for tab, fn in zip(tabs, (tab_overview, tab_trust, tab_customers, tab_behaviour, tab_products, tab_explorer, tab_data)):
        with tab:
            fn(R)


if __name__ == "__main__":
    main()