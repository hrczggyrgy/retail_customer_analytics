#!/usr/bin/env python3
"""Insight engine: turns the output tables of retail_customer_analysis.py into charts and an HTML dashboard.

What it does
------------
Reads the tables written by retail_customer_analysis.py (CSV, or Parquet when
pyarrow is installed) and produces:
  * one image per insight (PNG or SVG),
  * insights.html - a self-contained dashboard (images embedded) with KPI cards
    and a one-line, number-backed takeaway under every chart,
  * insight_summary.json - KPIs, chart manifest, and the reason for any chart skipped.

Charts (each is skipped with a stated reason when its inputs are missing)
-------------------------------------------------------------------------
 1 Model validation   2 Forecast calibration   3 Alive-vs-value customer map
 4 Segment value share  5 Cluster selection   6 RFM segments
 7 Repeat-purchase survival (new vs existing)   8 Cohort retention heatmap
 9 Revenue concentration (observed and expected)   10 Revenue trend and weekday pattern
11 Category revenue and returns   12 Basket rules and lift matrix
13 Next-trip transitions   14 Momentum by segment   15 Promo-proxy exposure
16 Cadence and regularity   17 Data-quality issues

Usage
-----
python insight_engine.py --results-dir retail_output
python insight_engine.py --results-dir retail_output --format svg --top-n 20
python insight_engine.py --input transactions.csv --results-dir retail_output \
    --analysis-args "--horizon-days 90 --windows-days 30,90,365"

Notes
-----
- Charts describe model outputs. Expected values are revenue forecasts (not profit),
  and basket rules are associations, not causal effects. Promo share is a price proxy.
- Dependencies: pandas, NumPy, matplotlib.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import logging
import math
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

LOGGER = logging.getLogger("insight_engine")

# ============================================================
# Theme & colour system
# ============================================================
PALETTE = ["#2b6cb0", "#dd6b20", "#2f855a", "#c53030", "#6b46c1", "#975a16", "#d53f8c", "#4a5568", "#b7791f", "#0987a0"]
GREY, INK = "#a0aec0", "#1a202c"
WEEKDAYS = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"}

# Segment colour map - consistent across all charts
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

# Sequential palette for heatmaps
HEATMAP_CMAP = "YlGnBu"
# Diverging palette for z-scores and deltas
DIVERGING_CMAP = "RdBu_r"

# Colour-blind safe check - these palettes are generally safe
# PALETTE uses distinct hues with good separation
# HEATMAP_CMAP and DIVERGING_CMAP are matplotlib built-ins

def apply_theme() -> None:
    """Apply consistent matplotlib theme for all charts."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#cbd5e0",
        "axes.labelcolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#edf2f7",
        "grid.linewidth": 0.8,
        "xtick.color": INK,
        "ytick.color": INK,
        "font.size": 10,
        "legend.frameon": False,
        "axes.axisbelow": True,
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    })

def segment_color(label: str) -> str:
    """Get consistent colour for a segment label."""
    return SEGMENT_COLORS.get(str(label), PALETTE[hash(str(label)) % len(PALETTE)])

def segment_colors(labels: list[str]) -> dict[str, str]:
    """Map a list of labels to consistent segment colours."""
    return {lab: segment_color(lab) for lab in sorted(set(map(str, labels)))}


class Skip(Exception):
    """Raised by a chart builder when its inputs are missing or unusable."""


@dataclass
class Chart:
    key: str
    title: str
    takeaway: str
    file: str


@dataclass
class Context:
    tables: dict[str, pd.DataFrame]
    meta: dict[str, Any]
    out: Path
    fmt: str
    dpi: int
    top_n: int
    max_points: int
    seed: int
    charts: list[Chart] = field(default_factory=list)
    skipped: list[dict[str, str]] = field(default_factory=list)

    def t(self, name: str, required: bool = True) -> pd.DataFrame:
        df = self.tables.get(name, pd.DataFrame())
        if required and df.empty:
            raise Skip(f"table '{name}' is missing or empty")
        return df

    def need(self, df: pd.DataFrame, cols: list[str], name: str) -> None:
        miss = [c for c in cols if c not in df.columns]
        if miss:
            raise Skip(f"table '{name}' lacks columns: {', '.join(miss)}")


# --------------------------------------------------------------------------- #
# Loading and helpers
# --------------------------------------------------------------------------- #
TABLES = ["customer_features", "holdout_metrics", "holdout_calibration", "cluster_diagnostics", "cluster_profiles",
          "rfm_segment_profiles", "repeat_survival", "cohort_retention_new_customers", "interpurchase_intervals",
          "basket_affinity", "next_trip_affinity", "customer_revenue_concentration", "category_summary",
          "department_summary", "monthly_summary", "weekday_summary", "data_quality_issues", "product_summary"]


def read_table(folder: Path, stem: str) -> pd.DataFrame:
    for ext in ("csv", "parquet"):
        p = folder / f"{stem}.{ext}"
        if not p.is_file():
            continue
        try:
            return pd.read_csv(p, low_memory=False) if ext == "csv" else pd.read_parquet(p)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
        except ImportError:
            LOGGER.warning("cannot read %s (pyarrow not installed)", p.name)
    return pd.DataFrame()


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except json.JSONDecodeError:
        return {}


def style() -> None:
    apply_theme()


def save(ctx: Context, key: str, fig: plt.Figure) -> str:
    name = f"{key}.{ctx.fmt}"
    fig.tight_layout()
    fig.savefig(ctx.out / name, dpi=ctx.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return name


def pct(v: float, d: int = 1) -> str:
    return "n/a" if v is None or not np.isfinite(v) else f"{100 * v:.{d}f}%"


def num(v: float, d: int = 2) -> str:
    return "n/a" if v is None or not np.isfinite(v) else f"{v:,.{d}f}"


def label_colors(labels: list[str]) -> dict[str, str]:
    return {lab: PALETTE[i % len(PALETTE)] for i, lab in enumerate(sorted(set(map(str, labels))))}


def shorten(s: Any, n: int = 28) -> str:
    s = "(missing)" if pd.isna(s) else str(s)
    return s if len(s) <= n else s[: n - 1] + "…"


# --------------------------------------------------------------------------- #
# Chart builders
# --------------------------------------------------------------------------- #
def chart_validation(ctx: Context) -> Chart:
    m = ctx.t("holdout_metrics")
    ctx.need(m, ["model", "target", "auc_any_purchase", "mae"], "holdout_metrics")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    a = m.dropna(subset=["auc_any_purchase"]).sort_values("auc_any_purchase")
    cols = [PALETTE[0] if "BG/NBD" in r else (PALETTE[3] if "legacy" in r else GREY) for r in a["model"]]
    axes[0].barh(a["model"], a["auc_any_purchase"], color=cols)
    axes[0].axvline(0.5, color=INK, lw=0.8, ls="--")
    axes[0].set_xlim(0.4, 1.0)
    axes[0].set_title("Who buys again? AUC on holdout")
    axes[0].set_xlabel("AUC (0.5 = coin flip)")
    for y, v in enumerate(a["auc_any_purchase"]):
        axes[0].text(v + 0.005, y, f"{v:.3f}", va="center", fontsize=9)
    mae = m.dropna(subset=["mae"])
    mae = mae.assign(label=mae["model"] + "\n(" + mae["target"].str.replace("_in_horizon", "") + ")")
    cols = [PALETTE[0] if "BG/NBD" in r else GREY for r in mae["model"]]
    axes[1].bar(mae["label"], mae["mae"], color=cols)
    axes[1].set_yscale("log")
    axes[1].set_title("Forecast error (MAE, log scale)")
    axes[1].tick_params(axis="x", labelsize=7.5, rotation=30)
    for lab in axes[1].get_xticklabels():
        lab.set_ha("right")
    bg = m[m["model"] == "BG/NBD P(alive)"]["auc_any_purchase"]
    lg = m[m["model"] == "legacy_cadence_heuristic"]["auc_any_purchase"]
    parts = []
    if len(bg) and len(lg):
        parts.append(f"BG/NBD P(alive) reaches AUC {bg.iloc[0]:.3f} vs {lg.iloc[0]:.3f} for the legacy cadence flag ({bg.iloc[0] - lg.iloc[0]:+.3f}).")
    t_bg = m[(m["model"] == "BG/NBD") & (m["target"] == "trips_in_horizon")]["mae"]
    t_ba = m[(m["target"] == "trips_in_horizon") & (m["model"] != "BG/NBD")]["mae"]
    if len(t_bg) and len(t_ba):
        parts.append(f"Trip MAE {t_bg.iloc[0]:.3f} vs {t_ba.min():.3f} for the best naive baseline ({pct(1 - t_bg.iloc[0] / t_ba.min(), 0)} lower).")
    return Chart("01_model_validation", "Does the model beat simple rules on unseen data?", " ".join(parts) or "Holdout metrics plotted.", save(ctx, "01_model_validation", fig))


def chart_calibration(ctx: Context) -> Chart:
    c = ctx.t("holdout_calibration")
    ctx.need(c, ["decile", "mean_predicted_trips", "mean_actual_trips", "share_bought", "mean_p_alive"], "holdout_calibration")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    ax.plot(c["mean_predicted_trips"], c["mean_actual_trips"], "o-", color=PALETTE[0], label="decile")
    hi = max(c["mean_predicted_trips"].max(), c["mean_actual_trips"].max()) * 1.1
    ax.plot([0, hi], [0, hi], ls="--", color=INK, lw=0.9, label="perfect calibration")
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_yscale("symlog", linthresh=0.01)
    ax.set_xlabel("Predicted trips in horizon (decile mean)")
    ax.set_ylabel("Actual trips in horizon")
    ax.set_title("Calibration by predicted-trip decile")
    ax.legend()
    ax = axes[1]
    ax.bar(c["decile"], c["share_bought"], color=PALETTE[0], alpha=0.85, label="share who bought")
    ax.plot(c["decile"], c["mean_p_alive"], "o-", color=PALETTE[1], label="mean P(alive)")
    ax.set_xlabel("Predicted-trip decile (10 = highest)")
    ax.set_ylabel("Share")
    ax.set_title("Who actually bought, by decile")
    ax.set_xticks(c["decile"])
    ax.legend(loc="upper left")
    top, bot = c.iloc[-1], c.iloc[0]
    m = ctx.tables.get("holdout_metrics", pd.DataFrame())
    tot = ""
    if not m.empty and {"model", "target", "predicted_total", "actual_total"} <= set(m.columns):
        r = m[(m["model"] == "BG/NBD") & (m["target"] == "trips_in_horizon")]
        if len(r) and r.iloc[0]["actual_total"]:
            dv = r.iloc[0]["predicted_total"] / r.iloc[0]["actual_total"] - 1
            tot = f" Predicted total trips are {pct(abs(dv), 1)} {'above' if dv >= 0 else 'below'} the actual total."
    return Chart("02_calibration", "Are the forecasts calibrated?", f"The top decile bought in {pct(top['share_bought'], 0)} of cases vs {pct(bot['share_bought'], 1)} in the bottom decile.{tot}",
                 save(ctx, "02_calibration", fig))


def chart_customer_map(ctx: Context) -> Chart:
    c = ctx.t("customer_features")
    ctx.need(c, ["expected_trips_horizon", "expected_trip_value", "p_alive"], "customer_features")
    d = c.dropna(subset=["expected_trips_horizon", "expected_trip_value"])
    if len(d) > ctx.max_points:
        d = d.sample(ctx.max_points, random_state=ctx.seed)
    labels = d["cluster_label"].fillna("Unclustered").astype(str) if "cluster_label" in d else pd.Series("All customers", index=d.index)
    colors = label_colors(list(labels))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    for lab in sorted(colors):
        s = d[labels == lab]
        ax.scatter(s["expected_trips_horizon"].clip(lower=1e-3), s["expected_trip_value"].clip(lower=1e-3), s=9, alpha=0.5, color=colors[lab], label=f"{shorten(lab, 34)} ({len(s):,})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Expected trips in horizon")
    ax.set_ylabel("Expected value per trip")
    ax.set_title("Customer map: how often x how much")
    ax.legend(fontsize=8, markerscale=2)
    axes[1].hist(c["p_alive"].dropna(), bins=30, color=PALETTE[0], alpha=0.85)
    axes[1].set_xlabel("P(alive)")
    axes[1].set_ylabel("Customers")
    axes[1].set_title("Is the customer still active?")
    p = c["p_alive"].dropna()
    act = c["p_alive"] >= 0.5
    rev = c.get("expected_revenue_horizon")
    extra = ""
    if rev is not None and rev.sum() > 0:
        extra = f" They hold {pct(rev[act].sum() / rev.sum(), 0)} of expected revenue."
    return Chart("03_customer_map", "Which customers are active, and what are they worth?", f"{pct((p >= 0.5).mean(), 0)} of customers have P(alive) of at least 0.5.{extra}", save(ctx, "03_customer_map", fig))


def chart_segment_value(ctx: Context) -> Chart:
    p = ctx.t("cluster_profiles")
    ctx.need(p, ["cluster_label", "customers", "total_expected_revenue"], "cluster_profiles")
    p = p.sort_values("total_expected_revenue", ascending=False)
    share_c, share_r = p["customers"] / p["customers"].sum(), p["total_expected_revenue"] / max(p["total_expected_revenue"].sum(), 1e-12)
    fig, ax = plt.subplots(figsize=(11, 4.6))
    x, w = np.arange(len(p)), 0.38
    ax.bar(x - w / 2, share_c, w, label="share of customers", color=GREY)
    ax.bar(x + w / 2, share_r, w, label="share of expected revenue", color=PALETTE[0])
    ax.set_xticks(x)
    ax.set_xticklabels([shorten(s, 26) for s in p["cluster_label"]], rotation=20, ha="right")
    ax.yaxis.set_major_formatter(lambda v, _: f"{100 * v:.0f}%")
    ax.set_title("Segment size vs forward value")
    ax.legend()
    i = share_r.idxmax()
    return Chart("04_segment_value", "Where does future revenue come from?", f"'{p.loc[i, 'cluster_label']}' is {pct(share_c[i], 0)} of customers but {pct(share_r[i], 0)} of expected revenue.", save(ctx, "04_segment_value", fig))


def chart_cluster_selection(ctx: Context) -> Chart:
    d = ctx.t("cluster_diagnostics")
    ctx.need(d, ["candidate_k", "silhouette", "stability_ari_mean"], "cluster_diagnostics")
    d = d.dropna(subset=["candidate_k"]).sort_values("candidate_k")
    thr = ctx.meta.get("configuration", {}).get("stability_threshold", 0.70)
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.plot(d["candidate_k"], d["silhouette"], "o-", color=PALETTE[0], label="silhouette")
    ax.plot(d["candidate_k"], d["stability_ari_mean"], "s-", color=PALETTE[1], label="bootstrap stability (ARI)")
    ax.axhline(thr, color=PALETTE[3], ls="--", lw=1, label=f"stability threshold {thr:.2f}")
    sel = d[d.get("selected", False).astype(bool)] if "selected" in d else d.iloc[0:0]
    if len(sel):
        ax.axvline(sel.iloc[0]["candidate_k"], color=INK, ls=":", lw=1.2)
        ax.text(sel.iloc[0]["candidate_k"], 0.02, " selected", fontsize=9)
    ax.set_xlabel("Number of clusters k")
    ax.set_xticks(d["candidate_k"])
    ax.set_title("Choosing k: separation and stability")
    ax.legend()
    ok = d[d["stability_ari_mean"] >= thr]
    txt = (f"{len(ok)} of {len(d)} candidate k values pass the stability threshold; k={int(sel.iloc[0]['candidate_k'])} was selected (silhouette {sel.iloc[0]['silhouette']:.3f}, ARI {sel.iloc[0]['stability_ari_mean']:.2f})."
           if len(sel) else f"{len(ok)} of {len(d)} candidate k values pass the stability threshold.")
    return Chart("05_cluster_selection", "Are the segments real or an artefact?", txt, save(ctx, "05_cluster_selection", fig))


def chart_rfm(ctx: Context) -> Chart:
    r = ctx.t("rfm_segment_profiles")
    ctx.need(r, ["rfm_segment", "customers", "median_net_spend"], "rfm_segment_profiles")
    r = r.sort_values("customers")
    cols = 3 if "heuristic_inactive_rate" in r.columns else 2
    fig, axes = plt.subplots(1, cols, figsize=(5 * cols, 4.6), sharey=True)
    axes[0].barh(r["rfm_segment"], r["customers"], color=PALETTE[0])
    axes[0].set_title("Customers")
    axes[1].barh(r["rfm_segment"], r["median_net_spend"], color=PALETTE[2])
    axes[1].set_title("Median net spend")
    if cols == 3:
        axes[2].barh(r["rfm_segment"], r["heuristic_inactive_rate"], color=PALETTE[3])
        axes[2].xaxis.set_major_formatter(lambda v, _: f"{100 * v:.0f}%")
        axes[2].set_title("Flagged inactive by legacy rule")
    top = r.sort_values("customers").iloc[-1]
    return Chart("06_rfm_segments", "RFM segments at a glance", f"The largest RFM group is '{top['rfm_segment']}' with {int(top['customers']):,} customers ({pct(top['customers'] / r['customers'].sum(), 0)}).", save(ctx, "06_rfm_segments", fig))


def chart_survival(ctx: Context) -> Chart:
    s = ctx.t("repeat_survival")
    ctx.need(s, ["group", "day", "repeat_probability", "repeat_ci95_lower", "repeat_ci95_upper", "customers"], "repeat_survival")
    groups = [("all_customers", "All customers", INK), ("new_after_burn_in", "New after burn-in", PALETTE[0]), ("existing_at_start_(left_censored)", "Existing at file start (left-censored)", PALETTE[1])]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for key, lab, col in groups:
        g = s[s["group"] == key].sort_values("day")
        if g.empty:
            continue
        ax.plot(g["day"], g["repeat_probability"], "o-", color=col, label=f"{lab} (n={int(g['customers'].iloc[0]):,})")
        ax.fill_between(g["day"], g["repeat_ci95_lower"], g["repeat_ci95_upper"], color=col, alpha=0.12)
    ax.set_xscale("symlog", linthresh=30)
    ax.set_xticks([7, 14, 30, 60, 90, 180, 365])
    ax.get_xaxis().set_major_formatter(lambda v, _: f"{int(v)}")
    ax.yaxis.set_major_formatter(lambda v, _: f"{100 * v:.0f}%")
    ax.set_xlabel("Days since first observed trip")
    ax.set_ylabel("Probability of a second trip")
    ax.set_title("Time to second trip (Kaplan-Meier, censoring-aware)")
    ax.legend()
    d90 = s[(s["day"] == 90) & s["group"].isin(["new_after_burn_in", "existing_at_start_(left_censored)"])].set_index("group")["repeat_probability"]
    txt = "Second-trip probability within 90 days: " + "; ".join(f"{k.split('_(')[0].replace('_', ' ')} {pct(v)}" for k, v in d90.items()) + "." if len(d90) else "Repeat curves plotted."
    return Chart("07_repeat_survival", "How quickly do first-time customers return?", txt, save(ctx, "07_repeat_survival", fig))


def chart_cohorts(ctx: Context) -> Chart:
    r = ctx.t("cohort_retention_new_customers")
    ctx.need(r, ["cohort_month", "months_since_cohort", "retention_rate", "period_complete"], "cohort_retention_new_customers")
    r = r[(r["months_since_cohort"] > 0) & r["period_complete"].astype(bool)]
    if r.empty:
        raise Skip("no complete post-cohort months")
    pv = r.pivot(index="cohort_month", columns="months_since_cohort", values="retention_rate").sort_index()
    fig, ax = plt.subplots(figsize=(10, max(3.5, 0.28 * len(pv) + 1.8)))
    im = ax.imshow(pv.to_numpy(), aspect="auto", cmap="YlGnBu", vmin=0)
    ax.set_xticks(range(len(pv.columns)))
    ax.set_xticklabels(pv.columns, fontsize=8)
    ax.set_yticks(range(len(pv.index)))
    ax.set_yticklabels(pv.index, fontsize=8)
    ax.grid(False)
    ax.set_xlabel("Months since first purchase")
    ax.set_title("Monthly activity of new customers (complete months only)")
    fig.colorbar(im, ax=ax, format=lambda v, _: f"{100 * v:.0f}%")
    m1 = pv[1].mean() if 1 in pv.columns else float("nan")
    return Chart("08_cohort_retention", "Do newer cohorts stay active?", f"On average {pct(m1)} of new customers buy again in the month after their first purchase month.", save(ctx, "08_cohort_retention", fig))


def lorenz(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    v = np.sort(np.clip(values[np.isfinite(values)], 0, None))
    if v.sum() <= 0:
        raise Skip("no positive values for concentration curve")
    x = np.concatenate([[0], np.arange(1, len(v) + 1) / len(v)])
    y = np.concatenate([[0], np.cumsum(v) / v.sum()])
    return x, y, float(1 - 2 * (np.trapezoid if hasattr(np, 'trapezoid') else np.trapz)(y, x))


def chart_concentration(ctx: Context) -> Chart:
    c = ctx.t("customer_features")
    ctx.need(c, ["gross_spend"], "customer_features")
    fig, ax = plt.subplots(figsize=(7.5, 6))
    x, y, g = lorenz(c["gross_spend"].to_numpy(float))
    ax.plot(x, y, color=PALETTE[0], lw=2, label=f"observed spend (Gini {g:.2f})")
    top10 = 1 - np.interp(0.9, x, y)
    gx = None
    if "expected_revenue_horizon" in c.columns:
        x2, y2, g2 = lorenz(c["expected_revenue_horizon"].to_numpy(float))
        ax.plot(x2, y2, color=PALETTE[1], lw=2, label=f"expected revenue (Gini {g2:.2f})")
        gx = 1 - np.interp(0.9, x2, y2)
    ax.plot([0, 1], [0, 1], "--", color=INK, lw=0.9, label="equality")
    ax.set_xlabel("Cumulative share of customers (lowest to highest)")
    ax.set_ylabel("Cumulative share of revenue")
    ax.xaxis.set_major_formatter(lambda v, _: f"{100 * v:.0f}%")
    ax.yaxis.set_major_formatter(lambda v, _: f"{100 * v:.0f}%")
    ax.set_title("How concentrated is revenue?")
    ax.legend(loc="upper left")
    txt = f"The top 10% of customers account for {pct(top10, 0)} of observed spend" + (f" and {pct(gx, 0)} of expected revenue." if gx is not None else ".")
    return Chart("09_revenue_concentration", "Revenue concentration: past and future", txt, save(ctx, "09_revenue_concentration", fig))


def chart_trend(ctx: Context) -> Chart:
    m = ctx.t("monthly_summary")
    ctx.need(m, ["month", "gross_purchase_revenue", "return_value", "net_revenue"], "monthly_summary")
    m = m.sort_values("month")
    w = ctx.t("weekday_summary", required=False)
    fig, axes = plt.subplots(1, 2 if len(w) else 1, figsize=(12.5 if len(w) else 8.5, 4.6), gridspec_kw={"width_ratios": [2.2, 1]} if len(w) else None)
    axes = np.atleast_1d(axes)
    ax = axes[0]
    ax.plot(m["month"], m["gross_purchase_revenue"], "o-", color=PALETTE[0], label="gross purchases")
    ax.plot(m["month"], m["net_revenue"], "o-", color=PALETTE[2], label="net of returns")
    ax.bar(m["month"], m["return_value"], color=PALETTE[3], alpha=0.5, label="returns")
    step = max(1, len(m) // 10)
    ax.set_xticks(range(0, len(m), step))
    ax.set_xticklabels(m["month"].iloc[::step], rotation=45, ha="right", fontsize=8)
    ax.set_title("Monthly revenue")
    ax.legend()
    if len(w) and "iso_weekday_monday_1" in w.columns:
        w = w.sort_values("iso_weekday_monday_1")
        share = w["net_revenue"] / w["net_revenue"].sum()
        axes[1].bar([WEEKDAYS.get(int(d), str(d)) for d in w["iso_weekday_monday_1"]], share, color=PALETTE[0])
        axes[1].yaxis.set_major_formatter(lambda v, _: f"{100 * v:.0f}%")
        axes[1].set_title("Net revenue by weekday")
        best = WEEKDAYS.get(int(w.loc[share.idxmax(), "iso_weekday_monday_1"]), "?")
        wk = f" {best} is the strongest weekday ({pct(share.max(), 0)} of net revenue)."
    else:
        wk = ""
    ret = m["return_value"].sum() / max(m["gross_purchase_revenue"].sum(), 1e-12)
    return Chart("10_revenue_trend", "Revenue over time", f"Returns equal {pct(ret)} of gross purchase revenue over the period.{wk}", save(ctx, "10_revenue_trend", fig))


def chart_categories(ctx: Context) -> Chart:
    d = ctx.t("category_summary")
    ctx.need(d, ["category", "gross_purchase_revenue", "return_value"], "category_summary")
    d = d.sort_values("gross_purchase_revenue", ascending=False).head(ctx.top_n).iloc[::-1]
    rate = d["return_value"] / d["gross_purchase_revenue"].where(d["gross_purchase_revenue"] > 0)
    fig, axes = plt.subplots(1, 2, figsize=(12, max(4, 0.35 * len(d) + 1.5)), sharey=True)
    labs = [shorten(x) for x in d["category"]]
    axes[0].barh(labs, d["gross_purchase_revenue"], color=PALETTE[0])
    axes[0].set_title(f"Top {len(d)} categories by gross revenue")
    axes[1].barh(labs, rate, color=PALETTE[3])
    axes[1].xaxis.set_major_formatter(lambda v, _: f"{100 * v:.1f}%")
    axes[1].set_title("Return rate (value)")
    hi = rate.idxmax()
    return Chart("11_categories", "Category revenue and returns", f"The category with the highest return rate among the top {len(d)} is '{shorten(d.loc[hi, 'category'])}' at {pct(rate[hi])}.", save(ctx, "11_categories", fig))


def chart_rules(ctx: Context) -> Chart:
    a = ctx.t("basket_affinity")
    ctx.need(a, ["antecedent", "consequent", "customer_lift", "confidence", "passes_all_filters", "q_value_bh"], "basket_affinity")
    ok = a[a["passes_all_filters"].astype(bool)].sort_values("customer_lift", ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6), gridspec_kw={"width_ratios": [1, 1.15]})
    ax = axes[0]
    if len(ok):
        t = ok.head(ctx.top_n).iloc[::-1]
        ax.barh([f"{shorten(x, 18)} → {shorten(y, 18)}" for x, y in zip(t["antecedent"], t["consequent"])], t["customer_lift"], color=PALETTE[2])
        for i, (v, cf) in enumerate(zip(t["customer_lift"], t["confidence"])):
            ax.text(v, i, f" conf {100 * cf:.0f}%", va="center", fontsize=8)
        ax.axvline(1, color=INK, lw=0.8, ls="--")
        ax.set_xlabel("Customer-level lift")
    else:
        ax.text(0.5, 0.5, "No rule passed all filters\n(FDR, odds-ratio CI, split-half persistence)", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    ax.set_title("Statistically supported rules")
    ax = axes[1]
    items = pd.concat([a["antecedent"], a["consequent"]]).value_counts().head(min(ctx.top_n, 16)).index.tolist()
    mat = pd.DataFrame(np.nan, index=items, columns=items)
    sig = a[(a["q_value_bh"] <= ctx.meta.get("configuration", {}).get("fdr_alpha", 0.05))]
    for r in sig.itertuples():
        if r.antecedent in mat.index and r.consequent in mat.columns:
            mat.loc[r.antecedent, r.consequent] = r.customer_lift
    vals = mat.to_numpy(float)
    if np.isfinite(vals).any():
        hi = max(float(np.nanmax(vals)), 1.05)
        lo = min(float(np.nanmin(vals)), 0.95)
        cmap = plt.get_cmap("RdBu_r").copy()
        cmap.set_bad("#f7fafc")
        im = ax.imshow(np.ma.masked_invalid(vals), cmap=cmap, norm=TwoSlopeNorm(vcenter=1.0, vmin=min(lo, 0.5), vmax=hi))
        fig.colorbar(im, ax=ax, label="lift (significant pairs only)")
    ax.set_xticks(range(len(items)))
    ax.set_xticklabels([shorten(i, 14) for i in items], rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([shorten(i, 14) for i in items], fontsize=8)
    ax.grid(False)
    ax.set_title("Lift matrix: antecedent (row) → consequent (column)")
    n_pass = int(a["passes_all_filters"].astype(bool).sum())
    txt = (f"{n_pass} of {len(a):,} directed rules pass every filter; the strongest is '{ok.iloc[0]['antecedent']}' → '{ok.iloc[0]['consequent']}' (customer lift {ok.iloc[0]['customer_lift']:.2f})."
           if n_pass else f"None of {len(a):,} directed rules passed all filters, so no cross-category pattern is statistically supported.")
    return Chart("12_basket_rules", "Which categories go together?", txt, save(ctx, "12_basket_rules", fig))


def chart_next_trip(ctx: Context) -> Chart:
    n = ctx.t("next_trip_affinity")
    ctx.need(n, ["from_item", "next_trip_item", "next_trip_lift", "p_next_given_from", "p_next_overall"], "next_trip_affinity")
    t = n.sort_values("next_trip_lift", ascending=False).head(ctx.top_n).iloc[::-1]
    labs = [f"{shorten(a, 16)} → {shorten(b, 16)}" for a, b in zip(t["from_item"], t["next_trip_item"])]
    y = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(9.5, max(4, 0.4 * len(t) + 1.5)))
    ax.barh(y + 0.2, t["p_next_given_from"], 0.38, color=PALETTE[0], label="P(item on next trip | item on this trip)")
    ax.barh(y - 0.2, t["p_next_overall"], 0.38, color=GREY, label="P(item on next trip) overall")
    ax.set_yticks(y)
    ax.set_yticklabels(labs, fontsize=8)
    ax.xaxis.set_major_formatter(lambda v, _: f"{100 * v:.0f}%")
    ax.set_title("Next-trip transitions that beat chance (FDR-controlled)")
    ax.legend(loc="lower right", fontsize=8)
    same = int((n["from_item"] == n["next_trip_item"]).sum())
    return Chart("13_next_trip", "What do customers buy on the following trip?", f"{len(n):,} transitions are significant; {same:,} are repeat purchases of the same item, which points to habitual buying.", save(ctx, "13_next_trip", fig))


def chart_momentum(ctx: Context) -> Chart:
    c = ctx.t("customer_features")
    col = "recent_revenue_change_pct_if_prior_positive"
    ctx.need(c, [col], "customer_features")
    d = c.dropna(subset=[col]).copy()
    if d.empty:
        raise Skip("no customers with a positive prior-period spend")
    d["chg"] = d[col].clip(-1, 3)
    d["grp"] = d["cluster_label"].fillna("Unclustered").astype(str) if "cluster_label" in d else "All customers"
    groups = sorted(d["grp"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw={"width_ratios": [1.5, 1]})
    bp = axes[0].boxplot([d.loc[d["grp"] == g, "chg"] for g in groups], tick_labels=[shorten(g, 22) for g in groups], patch_artist=True, showfliers=False)
    for patch, colr in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(colr)
        patch.set_alpha(0.6)
    axes[0].axhline(0, color=INK, lw=0.8, ls="--")
    axes[0].yaxis.set_major_formatter(lambda v, _: f"{100 * v:.0f}%")
    axes[0].tick_params(axis="x", labelsize=8, rotation=20)
    axes[0].set_title("Recent vs prior-period spend change (clipped -100%..+300%)")
    wins = sorted({int(x.split("_")[-1][:-1]) for x in c.columns if x.startswith("net_revenue_") and x.endswith("d")})
    if wins:
        axes[1].bar([f"{w}d" for w in wins], [c[f"net_revenue_{w}d"].sum() for w in wins], color=PALETTE[0])
        axes[1].set_title("Net revenue by rolling window")
    else:
        axes[1].set_axis_off()
    med = d.groupby("grp")["chg"].median()
    return Chart("14_momentum", "Which segments are growing or shrinking?", f"Median recent change is highest for '{med.idxmax()}' ({pct(med.max(), 0)}) and lowest for '{med.idxmin()}' ({pct(med.min(), 0)}).", save(ctx, "14_momentum", fig))


def chart_promo(ctx: Context) -> Chart:
    c = ctx.t("customer_features")
    ctx.need(c, ["promo_spend_share"], "customer_features")
    s = c["promo_spend_share"].dropna()
    if s.empty:
        raise Skip("promo features were not computed (prices barely vary)")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    axes[0].hist(s, bins=30, color=PALETTE[4], alpha=0.85)
    axes[0].set_xlabel("Share of spend on promo-like lines")
    axes[0].set_ylabel("Customers")
    axes[0].set_title("Promo exposure per customer (price-index proxy)")
    if "cluster_label" in c.columns:
        g = c.groupby(c["cluster_label"].fillna("Unclustered"))["promo_spend_share"].mean().sort_values()
        axes[1].barh([shorten(i, 26) for i in g.index], g.values, color=PALETTE[4])
        axes[1].xaxis.set_major_formatter(lambda v, _: f"{100 * v:.0f}%")
        axes[1].set_title("Mean promo share by segment")
    else:
        axes[1].set_axis_off()
    return Chart("15_promo_proxy", "How promotion-driven are customers?", f"Customers put on average {pct(s.mean())} of their spend on promo-like lines (price at least 10% below the product's typical price). This is a proxy, not measured promotion response.", save(ctx, "15_promo_proxy", fig))


def chart_cadence(ctx: Context) -> Chart:
    c = ctx.t("customer_features")
    ctx.need(c, ["regularity_shape_shrunk"], "customer_features")
    i = ctx.t("interpurchase_intervals", required=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    gaps = i["interpurchase_interval_days"].dropna() if len(i) and "interpurchase_interval_days" in i.columns else pd.Series(dtype=float)
    gaps = gaps[gaps > 0]
    if len(gaps):
        axes[0].hist(gaps, bins=np.logspace(0, np.log10(max(gaps.max(), 2)), 40), color=PALETTE[0], alpha=0.85)
        axes[0].set_xscale("log")
        axes[0].axvline(gaps.median(), color=PALETTE[3], ls="--", label=f"median {gaps.median():.0f} days")
        axes[0].legend()
    axes[0].set_xlabel("Days between trips")
    axes[0].set_title("Interpurchase gaps")
    sh = c["regularity_shape_shrunk"].dropna().clip(0.2, 10)
    axes[1].hist(sh, bins=40, color=PALETTE[2], alpha=0.85)
    axes[1].axvline(1, color=INK, ls="--", lw=1, label="1 = random (Poisson-like)")
    axes[1].legend()
    axes[1].set_xlabel("Regularity shape (higher = more regular)")
    axes[1].set_title("How regular is each customer's rhythm?")
    reg = (c["regularity_shape_raw"] > 1.5).mean() if "regularity_shape_raw" in c else float("nan")
    return Chart("16_cadence", "Purchase rhythm", (f"Median gap between trips is {gaps.median():.0f} days. " if len(gaps) else "") + f"{pct(reg, 0)} of customers with enough history buy more regularly than a random process would (raw shape above 1.5).", save(ctx, "16_cadence", fig))


def chart_quality(ctx: Context) -> Chart:
    q = ctx.t("data_quality_issues")
    ctx.need(q, ["issue", "count_or_value"], "data_quality_issues")
    vals = pd.to_numeric(q["count_or_value"], errors="coerce")
    d = pd.DataFrame({"issue": q["issue"], "n": vals}).dropna()
    d = d[d["n"] > 0].sort_values("n")
    if d.empty:
        raise Skip("no data-quality issues were recorded")
    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.34 * len(d) + 1.4)))
    ax.barh(d["issue"].str.replace("_", " "), d["n"], color=PALETTE[3])
    ax.set_xscale("log")
    ax.set_title("Data-quality issues found (row or key counts)")
    return Chart("17_data_quality", "What should be fixed at the source?", f"{len(d)} issue types were found; the largest is '{d.iloc[-1]['issue'].replace('_', ' ')}' ({int(d.iloc[-1]['n']):,}).", save(ctx, "17_data_quality", fig))


# ============================================================
# NEW CHART BUILDERS - Phase 2-4 visualizations
# ============================================================

def chart_revenue_bridge(ctx: Context) -> Chart:
    """Revenue bridge waterfall: Prior -> New, Retained+, Retained-, Reactivated, Churned, Returns -> Current."""
    # This requires a revenue_bridge table from the analysis
    # For now, skip if table doesn't exist
    raise Skip("revenue_bridge table not yet produced by analysis pipeline")


def chart_lorenz_segments(ctx: Context) -> Chart:
    """Lorenz curve with segment coloring showing revenue concentration."""
    c = ctx.t("customer_features")
    ctx.need(c, ["gross_spend", "cluster_label"], "customer_features")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Overall Lorenz
    x, y, g = lorenz(c["gross_spend"].to_numpy(float))
    ax.plot(x, y, color=PALETTE[0], lw=2, label=f"Overall (Gini {g:.2f})")
    
    # By segment
    for lab in sorted(c["cluster_label"].dropna().unique()):
        seg_data = c[c["cluster_label"] == lab]["gross_spend"]
        if len(seg_data) > 10:
            try:
                x_s, y_s, g_s = lorenz(seg_data.to_numpy(float))
                ax.plot(x_s, y_s, color=segment_color(lab), lw=1.5, alpha=0.7, label=f"{shorten(lab, 20)} (Gini {g_s:.2f})")
            except Skip:
                pass
    
    ax.plot([0, 1], [0, 1], "--", color=INK, lw=0.9, label="Equality")
    ax.set_xlabel("Cumulative share of customers (lowest to highest)")
    ax.set_ylabel("Cumulative share of revenue")
    ax.xaxis.set_major_formatter(lambda v, _: f"{100 * v:.0f}%")
    ax.yaxis.set_major_formatter(lambda v, _: f"{100 * v:.0f}%")
    ax.set_title("Revenue Concentration by Segment (Lorenz Curves)")
    ax.legend(loc="upper left", fontsize=8)
    
    top10 = 1 - np.interp(0.9, x, y)
    return Chart("18_lorenz_segments", "How concentrated is revenue within each segment?", f"Top 10% of customers account for {pct(top10, 0)} of overall revenue.", save(ctx, "18_lorenz_segments", fig))


def chart_action_quadrant(ctx: Context) -> Chart:
    """P(alive) vs Expected Revenue quadrant with action labels."""
    c = ctx.t("customer_features")
    ctx.need(c, ["p_alive", "expected_revenue_horizon", "cluster_label"], "customer_features")
    
    d = c.dropna(subset=["p_alive", "expected_revenue_horizon"])
    if len(d) > ctx.max_points:
        d = d.sample(ctx.max_points, random_state=ctx.seed)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Quadrant boundaries
    p50 = d["p_alive"].median()
    rev50 = d["expected_revenue_horizon"].median()
    
    # Quadrant shading
    ax.axhspan(rev50, d["expected_revenue_horizon"].max(), xmin=0.5, facecolor="#c6f6d5", alpha=0.3)  # Protect
    ax.axhspan(rev50, d["expected_revenue_horizon"].max(), xmax=0.5, facecolor="#bee3f8", alpha=0.3)  # Grow
    ax.axhspan(d["expected_revenue_horizon"].min(), rev50, xmin=0.5, facecolor="#fed7d7", alpha=0.3)  # Win back
    ax.axhspan(d["expected_revenue_horizon"].min(), rev50, xmax=0.5, facecolor="#feebc8", alpha=0.3)  # Ignore
    
    # Quadrant labels
    ax.text(0.75, 0.9, "PROTECT\n(High P(alive), High Value)", transform=ax.transAxes, ha="center", va="top", fontsize=10, fontweight="bold", color="#276749")
    ax.text(0.25, 0.9, "GROW\n(Low P(alive), High Value)", transform=ax.transAxes, ha="center", va="top", fontsize=10, fontweight="bold", color="#2b6cb0")
    ax.text(0.75, 0.1, "WIN BACK\n(High P(alive), Low Value)", transform=ax.transAxes, ha="center", va="bottom", fontsize=10, fontweight="bold", color="#c53030")
    ax.text(0.25, 0.1, "IGNORE\n(Low P(alive), Low Value)", transform=ax.transAxes, ha="center", va="bottom", fontsize=10, fontweight="bold", color="#744210")
    
    # Scatter by segment
    for lab in sorted(d["cluster_label"].dropna().unique()):
        s = d[d["cluster_label"] == lab]
        ax.scatter(s["p_alive"], s["expected_revenue_horizon"], s=15, alpha=0.6, color=segment_color(lab), label=f"{shorten(lab, 25)} ({len(s):,})")
    
    ax.axvline(p50, color=INK, ls="--", lw=1)
    ax.axhline(rev50, color=INK, ls="--", lw=1)
    ax.set_xlabel("P(alive)")
    ax.set_ylabel("Expected revenue (horizon)")
    ax.set_yscale("log")
    ax.set_title("Action Quadrant: P(alive) vs Expected Revenue")
    ax.legend(fontsize=8, markerscale=2, loc="lower right")
    
    # Count in each quadrant
    q1 = ((d["p_alive"] >= p50) & (d["expected_revenue_horizon"] >= rev50)).sum()
    q2 = ((d["p_alive"] < p50) & (d["expected_revenue_horizon"] >= rev50)).sum()
    q3 = ((d["p_alive"] >= p50) & (d["expected_revenue_horizon"] < rev50)).sum()
    q4 = ((d["p_alive"] < p50) & (d["expected_revenue_horizon"] < rev50)).sum()
    
    return Chart("19_action_quadrant", "Who to protect, grow, win back, or ignore?", f"Protect: {q1:,} | Grow: {q2:,} | Win back: {q3:,} | Ignore: {q4:,} customers.", save(ctx, "19_action_quadrant", fig))


def chart_segment_migration(ctx: Context) -> Chart:
    """Sankey-style segment migration: period A -> period B."""
    raise Skip("segment_transitions table not yet produced by analysis pipeline")


def chart_dept_chord(ctx: Context) -> Chart:
    """Department chord diagram for cross-department flows."""
    # Try pyCirclize
    try:
        from pycirclize import Circos
        from pycirclize.utils import ColorCycler
        HAS_CIRCLIZE = True
    except ImportError:
        HAS_CIRCLIZE = False
    
    if not HAS_CIRCLIZE:
        raise Skip("pyCirclize not installed (pip install pycirclize)")
    
    # This needs a dept_flow_matrix table
    raise Skip("dept_flow_matrix table not yet produced by analysis pipeline")


def chart_affinity_network(ctx: Context) -> Chart:
    """Affinity network graph of significant basket rules."""
    try:
        import networkx as nx
        HAS_NETWORKX = True
    except ImportError:
        HAS_NETWORKX = False
    
    if not HAS_NETWORKX:
        raise Skip("networkx not installed (pip install networkx)")
    
    a = ctx.t("basket_affinity")
    ctx.need(a, ["antecedent", "consequent", "customer_lift", "passes_all_filters"], "basket_affinity")
    
    ok = a[a["passes_all_filters"].astype(bool)].sort_values("customer_lift", ascending=False).head(20)
    if ok.empty:
        raise Skip("no significant basket rules to plot")
    
    # Build graph
    G = nx.Graph()
    for _, r in ok.iterrows():
        G.add_edge(r["antecedent"], r["consequent"], weight=r["customer_lift"])
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=ctx.seed)
    
    # Draw edges with width proportional to lift
    edges = G.edges()
    weights = [G[u][v]["weight"] for u, v in edges]
    max_w = max(weights)
    min_w = min(weights)
    
    for (u, v), w in zip(edges, weights):
        width = 1 + 4 * (w - min_w) / max(1e-6, max_w - min_w)
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=width, alpha=0.5, edge_color=PALETTE[2], ax=ax)
    
    # Draw nodes
    node_sizes = [G.degree(n) * 300 for n in G.nodes()]
    node_colors = [segment_color(n) if n in SEGMENT_COLORS else PALETTE[i % len(PALETTE)] for i, n in enumerate(G.nodes())]
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8, ax=ax)
    
    # Labels
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold", ax=ax)
    
    ax.set_title("Affinity Network: Significant Cross-Category Rules (FDR-controlled)")
    ax.set_axis_off()
    
    return Chart("20_affinity_network", "Cross-sell structure as a network", f"{len(G.nodes())} categories, {len(G.edges())} significant rules. Thicker edges = higher lift.", save(ctx, "20_affinity_network", fig))


def chart_dow_hour_heatmap(ctx: Context) -> Chart:
    """Day of week x Hour of day transaction heatmap."""
    # This needs transaction timestamps with hour information
    raise Skip("transaction timestamps with hour not available in current output")


def chart_cohort_ltv(ctx: Context) -> Chart:
    """Cohort LTV fan chart - cumulative revenue per customer by cohort age."""
    raise Skip("cohort_ltv table not yet produced by analysis pipeline")


def chart_promo_return_lollipop(ctx: Context) -> Chart:
    """Discount share vs Return rate by category - lollipop chart."""
    cat = ctx.t("category_summary")
    ctx.need(cat, ["category", "gross_purchase_revenue", "return_value"], "category_summary")
    
    # Need promo share data - for now use return rate
    d = cat.sort_values("gross_purchase_revenue", ascending=False).head(15)
    d["return_rate"] = d["return_value"] / d["gross_purchase_revenue"].replace(0, np.nan)
    
    fig, ax = plt.subplots(figsize=(10, max(5, 0.4 * len(d) + 1)))
    
    y_pos = np.arange(len(d))
    ax.hlines(y_pos, 0, d["return_rate"], color=PALETTE[3], linewidth=2)
    ax.plot(d["return_rate"], y_pos, "o", color=PALETTE[3], markersize=10)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([shorten(x, 25) for x in d["category"]])
    ax.set_xlabel("Return rate (value)")
    ax.xaxis.set_major_formatter(lambda v, _: f"{100 * v:.1f}%")
    ax.set_title("Return Rate by Category (Top 15 by Revenue)")
    
    hi = d["return_rate"].idxmax()
    return Chart("21_promo_return_lollipop", "Margin risk: return rate by category", f"Highest return rate: '{shorten(d.loc[hi, 'category'])}' at {pct(d.loc[hi, 'return_rate'])}.", save(ctx, "21_promo_return_lollipop", fig))


def chart_cohort_retention_revenue(ctx: Context) -> Chart:
    """Cohort retention heatmap colored by revenue per customer."""
    r = ctx.t("cohort_retention_new_customers")
    ctx.need(r, ["cohort_month", "months_since_cohort", "retention_rate", "period_complete"], "cohort_retention_new_customers")
    
    r = r[(r["months_since_cohort"] > 0) & r["period_complete"].astype(bool)]
    if r.empty:
        raise Skip("no complete post-cohort months")
    
    # Also need cohort revenue per customer - check if available
    pv = r.pivot(index="cohort_month", columns="months_since_cohort", values="retention_rate").sort_index()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, max(3.5, 0.28 * len(pv) + 1.8)))
    
    # Retention heatmap
    im1 = axes[0].imshow(pv.to_numpy(), aspect="auto", cmap=HEATMAP_CMAP, vmin=0, vmax=1)
    axes[0].set_xticks(range(len(pv.columns)))
    axes[0].set_xticklabels(pv.columns, fontsize=8)
    axes[0].set_yticks(range(len(pv.index)))
    axes[0].set_yticklabels(pv.index, fontsize=8)
    axes[0].grid(False)
    axes[0].set_xlabel("Months since first purchase")
    axes[0].set_title("Cohort Retention Rate (Complete Months)")
    fig.colorbar(im1, ax=axes[0], format=lambda v, _: f"{100 * v:.0f}%")
    
    # Revenue per customer placeholder
    axes[1].text(0.5, 0.5, "Cohort Revenue per Customer\n(requires cohort_ltv table)", ha="center", va="center", transform=axes[1].transAxes, fontsize=12)
    axes[1].set_axis_off()
    axes[1].set_title("Cohort Revenue per Customer")
    
    m1 = pv[1].mean() if 1 in pv.columns else float("nan")
    return Chart("22_cohort_ltv", "Cohort retention and revenue quality", f"Average month-1 retention: {pct(m1)}. Revenue chart needs cohort_ltv table.", save(ctx, "22_cohort_ltv", fig))


# Add new builders to the list
BUILDERS: list[Callable[[Context], Chart]] = [chart_validation, chart_calibration, chart_customer_map, chart_segment_value, chart_cluster_selection,
                                              chart_rfm, chart_survival, chart_cohorts, chart_concentration, chart_trend, chart_categories,
                                              chart_rules, chart_next_trip, chart_momentum, chart_promo, chart_cadence, chart_quality,
                                              chart_lorenz_segments, chart_action_quadrant, chart_affinity_network, 
                                              chart_promo_return_lollipop, chart_cohort_retention_revenue]


# --------------------------------------------------------------------------- #
# KPIs, HTML, orchestration
# --------------------------------------------------------------------------- #
def compute_kpis(ctx: Context) -> list[tuple[str, str]]:
    k: list[tuple[str, str]] = []
    c = ctx.tables.get("customer_features", pd.DataFrame())
    if not c.empty:
        k.append(("Customers scored", f"{len(c):,}"))
        if "p_alive" in c:
            k.append(("Likely active (P(alive) ≥ 0.5)", pct((c["p_alive"] >= 0.5).mean(), 0)))
        if "expected_revenue_horizon" in c:
            e = c["expected_revenue_horizon"].clip(lower=0)
            k.append(("Expected revenue, next horizon", num(e.sum(), 0)))
            if len(e) >= 10 and e.sum() > 0:
                k.append(("Top 10% customers' share of it", pct(e.nlargest(max(1, int(math.ceil(len(e) * 0.1)))).sum() / e.sum(), 0)))
    s = ctx.tables.get("repeat_survival", pd.DataFrame())
    if not s.empty:
        r = s[(s["group"] == "all_customers") & (s["day"] == 90)]
        if len(r):
            k.append(("Second trip within 90 days", pct(r.iloc[0]["repeat_probability"], 0)))
    m = ctx.tables.get("holdout_metrics", pd.DataFrame())
    if not m.empty:
        bg, lg = m[m["model"] == "BG/NBD P(alive)"]["auc_any_purchase"], m[m["model"] == "legacy_cadence_heuristic"]["auc_any_purchase"]
        if len(bg) and len(lg):
            k.append(("AUC: BG/NBD vs legacy flag", f"{bg.iloc[0]:.2f} vs {lg.iloc[0]:.2f}"))
    a = ctx.tables.get("basket_affinity", pd.DataFrame())
    if not a.empty and "passes_all_filters" in a:
        k.append(("Supported basket rules", f"{int(a['passes_all_filters'].astype(bool).sum()):,}"))
    return k


def build_html(ctx: Context, kpis: list[tuple[str, str]]) -> str:
    def embed(ch: Chart) -> str:
        p = ctx.out / ch.file
        if ch.file.endswith(".svg"):
            return p.read_text(encoding="utf-8")
        return f'<img alt="{html.escape(ch.title)}" src="data:image/png;base64,{base64.b64encode(p.read_bytes()).decode()}">'

    cards = "".join(f'<div class="card"><div class="v">{html.escape(v)}</div><div class="l">{html.escape(l)}</div></div>' for l, v in kpis)
    secs = "".join(f'<section><h2>{html.escape(c.title)}</h2><p class="t">{html.escape(c.takeaway)}</p>{embed(c)}</section>' for c in ctx.charts)
    warn = ctx.meta.get("warnings") or []
    wsec = "<section><h2>Pipeline warnings</h2><ul>" + "".join(f"<li>{html.escape(str(w))}</li>" for w in warn) + "</ul></section>" if warn else ""
    sk = "".join(f"<li><b>{html.escape(s['chart'])}</b>: {html.escape(s['reason'])}</li>" for s in ctx.skipped)
    ssec = f"<section><h2>Charts not produced</h2><ul>{sk}</ul></section>" if sk else ""
    css = ("body{font-family:Inter,Segoe UI,Arial,sans-serif;margin:0;background:#f7fafc;color:#1a202c}header{background:#1a365d;color:#fff;padding:24px 40px}"
           "h1{margin:0;font-size:24px}main{max-width:1180px;margin:0 auto;padding:24px}.cards{display:flex;flex-wrap:wrap;gap:12px;margin-bottom:24px}"
           ".card{background:#fff;border-radius:8px;padding:14px 18px;min-width:170px;box-shadow:0 1px 3px rgba(0,0,0,.08)}.v{font-size:22px;font-weight:700;color:#2b6cb0}"
           ".l{font-size:12px;color:#4a5568}section{background:#fff;border-radius:8px;padding:18px 22px;margin-bottom:18px;box-shadow:0 1px 3px rgba(0,0,0,.08)}"
           "h2{margin:0 0 6px;font-size:17px}.t{margin:0 0 12px;color:#2d3748}img,svg{max-width:100%;height:auto}.note{font-size:12px;color:#718096}")
    when = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    note = ("Expected values are revenue forecasts, not profit. Basket rules and transitions are associations, not causal effects. "
            "Promo share is a price-based proxy. Segments are descriptive.")
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>Customer insight dashboard</title><style>{css}</style></head><body>"
            f"<header><h1>Customer insight dashboard</h1><div>Generated {when} from retail_customer_analysis.py outputs</div></header>"
            f"<main><div class='cards'>{cards}</div>{secs}{wsec}{ssec}<p class='note'>{html.escape(note)}</p></main></body></html>")


def run(args: argparse.Namespace) -> int:
    results = Path(args.results_dir).expanduser().resolve()
    if args.input:
        script = Path(__file__).resolve().with_name("retail_customer_analysis.py")
        if not script.is_file():
            raise FileNotFoundError(f"cannot find {script.name} next to insight_engine.py")
        cmd = [sys.executable, str(script), "--input", args.input, "--output-dir", str(results)] + shlex.split(args.analysis_args or "")
        LOGGER.info("Running analysis: %s", " ".join(cmd))
        if subprocess.run(cmd).returncode != 0:
            raise RuntimeError("analysis script failed")
    if not results.is_dir():
        raise FileNotFoundError(f"results directory not found: {results}")
    out = Path(args.output_dir).expanduser().resolve() if args.output_dir else results / "insights"
    out.mkdir(parents=True, exist_ok=True)
    style()
    meta = {**read_json(results / "run_metadata.json")}
    meta.setdefault("warnings", read_json(results / "data_quality.json").get("warnings", []))
    tables = {name: read_table(results, name) for name in TABLES}
    if all(df.empty for df in tables.values()):
        raise FileNotFoundError("no analysis tables found; run retail_customer_analysis.py first")
    ctx = Context(tables, meta, out, args.format, args.dpi, args.top_n, args.max_scatter_points, args.seed)
    for b in BUILDERS:
        key = b.__name__.replace("chart_", "")
        try:
            ctx.charts.append(b(ctx))
            LOGGER.info("built %s", key)
        except Skip as exc:
            ctx.skipped.append({"chart": key, "reason": str(exc)})
            LOGGER.warning("skipped %s: %s", key, exc)
        except Exception as exc:  # one broken chart must not stop the others
            plt.close("all")
            ctx.skipped.append({"chart": key, "reason": f"{type(exc).__name__}: {exc}"})
            LOGGER.error("failed %s: %s: %s", key, type(exc).__name__, exc)
    kpis = compute_kpis(ctx)
    if not args.no_html:
        (out / "insights.html").write_text(build_html(ctx, kpis), encoding="utf-8")
    summary = {"generated_utc": datetime.now(timezone.utc).isoformat(), "results_dir": str(results), "kpis": dict(kpis),
               "charts": [c.__dict__ for c in ctx.charts], "skipped": ctx.skipped}
    (out / "insight_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("Wrote %d charts to %s", len(ctx.charts), out)
    return 0 if ctx.charts else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Generate charts and an HTML dashboard from retail_customer_analysis.py outputs.")
    p.add_argument("--results-dir", required=True, help="Output directory of retail_customer_analysis.py")
    p.add_argument("--output-dir", default=None, help="Where to write charts (default: <results-dir>/insights)")
    p.add_argument("--format", choices=["png", "svg"], default="png")
    p.add_argument("--dpi", type=int, default=140)
    p.add_argument("--top-n", type=int, default=15)
    p.add_argument("--max-scatter-points", type=int, default=8000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-html", action="store_true")
    p.add_argument("--input", default=None, help="Optional transactions file: run retail_customer_analysis.py first")
    p.add_argument("--analysis-args", default="", help="Extra arguments for the analysis script, as one quoted string")
    p.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    args = p.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    try:
        return run(args)
    except Exception as exc:
        LOGGER.error("%s: %s", type(exc).__name__, exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
