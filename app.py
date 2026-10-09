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
    "customers": ["03_customer_map", "04_segment_value", "06_rfm_segments", "09_revenue_concentration"],
    "behaviour": ["07_repeat_survival", "08_cohort_retention", "14_momentum", "16_cadence", "15_promo_proxy"],
    "products": ["10_revenue_trend", "11_categories", "12_basket_rules", "13_next_trip"],
    "data": ["17_data_quality"],
}
HEADLINE_CHARTS = ["01_model_validation", "03_customer_map", "04_segment_value", "07_repeat_survival", "09_revenue_concentration", "12_basket_rules"]
SEGMENT_HINTS = {
    "Active / higher": "Hypothesis: protect and grow. Test retention perks against a holdout group.",
    "Active / lower": "Hypothesis: raise basket value or breadth. Test cross-category offers.",
    "Lapsing": "Hypothesis: win-back candidates. Test timed reminders against a holdout group.",
    "Lapsed": "Hypothesis: low-cost reactivation only. Check cost against expected revenue first.",
}


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


@st.cache_data(show_spinner=False)
def read_results_table(res_dir: str, stem: str, mtime: float = 0.0) -> pd.DataFrame:
    for ext in ("csv", "parquet"):
        p = Path(res_dir) / f"{stem}.{ext}"
        if p.is_file():
            try:
                return pd.read_csv(p, low_memory=False) if ext == "csv" else pd.read_parquet(p)
            except Exception:
                return pd.DataFrame()
    return pd.DataFrame()


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
    kpis = R["summary"].get("kpis", {})
    items = list(kpis.items())
    for i in range(0, len(items), 4):
        cols = st.columns(4)
        for col, (label, value) in zip(cols, items[i:i + 4]):
            col.metric(label, value)
    st.subheader("Key findings")
    by_key = {c["key"]: c for c in R["summary"].get("charts", [])}
    found = [by_key[k] for k in HEADLINE_CHARTS if k in by_key]
    if found:
        for c in found:
            st.markdown(f"- **{c['title']}** {c['takeaway']}")
    else:
        st.info("No headline findings available.")
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
    show_charts(R, CHARTS_BY_TAB["trust"])
    m = table(R, "holdout_metrics")
    if len(m):
        st.subheader("Holdout metrics")
        show_df(pretty(m))
    for stem, title in (("holdout_calibration", "Calibration by decile"), ("cluster_diagnostics", "Cluster selection diagnostics"), ("model_parameters", "Fitted model parameters")):
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
    show_charts(R, CHARTS_BY_TAB["customers"])
    cf = table(R, "customer_features")
    prof = segment_profile(cf)
    if len(prof):
        st.subheader("Segment profiles")
        st.caption("Hypotheses are starting points to test with a holdout group, not conclusions.")
        show_df(pretty(prof))
    for stem, title in (("rfm_segment_profiles", "RFM segment table"), ("customer_revenue_concentration", "Revenue concentration figures")):
        t = table(R, stem)
        if len(t):
            with st.expander(title):
                show_df(pretty(t))


def tab_behaviour(R: dict[str, Any]) -> None:
    st.header("Behaviour")
    st.caption("Question answered here: how do customers come back, how regular are they, and who is changing?")
    show_charts(R, CHARTS_BY_TAB["behaviour"])
    for stem, title in (("repeat_survival", "Second-trip probabilities (all landmarks)"), ("cohort_retention_new_customers", "Cohort retention table")):
        t = table(R, stem)
        if len(t):
            with st.expander(title):
                show_df(pretty(t))


def tab_products(R: dict[str, Any]) -> None:
    st.header("Products and baskets")
    st.caption("Question answered here: what sells, what is returned, and what is bought together or next?")
    show_charts(R, CHARTS_BY_TAB["products"])
    aff = table(R, "basket_affinity")
    if len(aff) and "passes_all_filters" in aff.columns:
        ok = aff[aff["passes_all_filters"].astype(bool)]
        st.subheader("Supported basket rules")
        st.caption(f"{len(ok):,} of {len(aff):,} directed rules pass FDR control, an odds-ratio interval above 1 and split-half persistence.")
        if len(ok):
            show_df(pretty(ok))
    nxt = table(R, "next_trip_affinity")
    if len(nxt):
        with st.expander("Next-trip transitions"):
            show_df(pretty(nxt))
    for stem, title in (("category_summary", "Category summary"), ("department_summary", "Department summary"), ("product_summary", "Product summary")):
        t = table(R, stem)
        if len(t):
            with st.expander(title):
                show_df(pretty(t.head(500)))


def tab_explorer(R: dict[str, Any]) -> None:
    st.header("Customer explorer")
    st.caption("Filter customers, export a target list, or look up a single customer.")
    cf = table(R, "customer_features")
    if cf.empty:
        st.info("No customer table available.")
        return
    c1, c2, c3 = st.columns(3)
    labels = sorted(cf["cluster_label"].dropna().unique()) if "cluster_label" in cf else []
    rfm = sorted(cf["rfm_segment"].dropna().unique()) if "rfm_segment" in cf else []
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
    c1, c2, c3 = st.columns(3)
    c1.download_button("All result tables and charts (ZIP)", zip_dir(R["results"]), file_name="customer_insights_results.zip", mime="application/zip")
    html_path = Path(R["insights"]) / "insights.html"
    if html_path.is_file():
        c2.download_button("Dashboard (HTML)", html_path.read_bytes(), file_name="insights.html", mime="text/html")
    rp = Path(R["results"]) / "analysis_report.md"
    if rp.is_file():
        c3.download_button("Written report (Markdown)", rp.read_bytes(), file_name="analysis_report.md", mime="text/markdown")


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