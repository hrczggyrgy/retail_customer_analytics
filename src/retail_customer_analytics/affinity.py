"""Shared bounded basket affinity engine.

This module provides a single, authoritative implementation of basket affinity
analysis that eliminates unbounded pair-occurrence materialization and enforces
explicit computational budgets.
"""

from __future__ import annotations

import logging
from collections import Counter
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from retail_customer_analytics.ingestion import prepare_transaction_frame

LOGGER = logging.getLogger("retail_analysis.affinity")


class AffinityConfig:
    """Configuration for basket affinity analysis."""

    def __init__(
        self,
        basket_level: str = "category",
        basket_grain: str = "trip",
        transaction_key_mode: str = "customer-transaction",
        min_support: float = 0.01,
        min_confidence: float = 0.20,
        min_lift: float = 1.0,
        fdr_alpha: float = 0.05,
        max_candidates: int = 100,
        max_baskets: int = 50_000,
        max_items_per_basket: int = 30,
        max_pair_operations: int = 1_000_000,
        min_customers: int = 30,
        random_seed: int = 42,
    ):
        self.basket_level = basket_level
        self.basket_grain = basket_grain
        self.transaction_key_mode = transaction_key_mode
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_lift = min_lift
        self.fdr_alpha = fdr_alpha
        self.max_candidates = max_candidates
        self.max_baskets = max_baskets
        self.max_items_per_basket = max_items_per_basket
        self.max_pair_operations = max_pair_operations
        self.min_customers = min_customers
        self.random_seed = random_seed

    def to_dict(self) -> dict[str, Any]:
        return {
            "basket_level": self.basket_level,
            "basket_grain": self.basket_grain,
            "transaction_key_mode": self.transaction_key_mode,
            "min_support": self.min_support,
            "min_confidence": self.min_confidence,
            "min_lift": self.min_lift,
            "fdr_alpha": self.fdr_alpha,
            "max_candidates": self.max_candidates,
            "max_baskets": self.max_baskets,
            "max_items_per_basket": self.max_items_per_basket,
            "max_pair_operations": self.max_pair_operations,
            "min_customers": self.min_customers,
            "random_seed": self.random_seed,
        }


class AffinityResult:
    """Structured result from affinity analysis."""

    def __init__(
        self,
        associations: pd.DataFrame,
        transitions: pd.DataFrame,
        diagnostics: dict[str, Any],
    ):
        self.associations = associations
        self.transitions = transitions
        self.diagnostics = diagnostics

    def to_dict(self) -> dict[str, Any]:
        return {
            "associations": self.associations,
            "transitions": self.transitions,
            "diagnostics": self.diagnostics,
        }


def _hash_series(s: pd.Series, seed: int) -> np.ndarray:
    """Deterministic hash for reproducible sampling."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2**63 - 1, size=len(s), dtype=np.uint64)


def _pair_stats(
    n_ab: np.ndarray,
    n_a: np.ndarray,
    n_b: np.ndarray,
    N: int,
) -> dict[str, np.ndarray]:
    """Compute association statistics for item pairs."""
    # Support
    support = n_ab / N

    # Confidence A -> B
    conf_a_b = np.divide(n_ab, n_a, out=np.zeros_like(n_ab), where=n_a > 0)

    # Lift
    p_b = n_b / N
    lift = np.divide(conf_a_b, p_b, out=np.zeros_like(conf_a_b), where=p_b > 0)

    # Odds ratio (Woolf)
    n_00 = N - n_a - n_b + n_ab
    n_01 = n_b - n_ab
    n_10 = n_a - n_ab
    n_11 = n_ab

    # Add 0.5 continuity correction for zeros
    n_00_adj = n_00 + 0.5
    n_01_adj = n_01 + 0.5
    n_10_adj = n_10 + 0.5
    n_11_adj = n_11 + 0.5

    or_val = (n_11_adj * n_00_adj) / (n_10_adj * n_01_adj)
    log_or = np.log(or_val)
    se_log_or = np.sqrt(
        1 / n_11_adj + 1 / n_10_adj + 1 / n_01_adj + 1 / n_00_adj
    )

    # 95% CI for odds ratio
    z = 1.96
    or_ci_lower = np.exp(log_or - z * se_log_or)
    or_ci_upper = np.exp(log_or + z * se_log_or)

    # Chi-square p-value
    chi2 = (np.abs(n_ab * N - n_a * n_b) - 0.5 * N) ** 2 * N / (
        n_a * n_b * (N - n_a) * (N - n_b)
    )
    p_val = 1 - stats_chi2_cdf(chi2)

    return {
        "support": support,
        "confidence": conf_a_b,
        "lift": lift,
        "odds_ratio": or_val,
        "or_ci95_lower": or_ci_lower,
        "or_ci95_upper": or_ci_upper,
        "p_value": p_val,
    }


def stats_chi2_cdf(x: np.ndarray) -> np.ndarray:
    """Chi-square CDF with 1 degree of freedom."""
    from scipy import special

    return special.gammainc(0.5, x / 2)


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction."""
    m = len(p_values)
    if m == 0:
        return np.array([])
    order = np.argsort(p_values)
    ranked = p_values[order]
    q = ranked * m / (np.arange(m) + 1)
    # Make monotonic
    q = np.minimum.accumulate(q[::-1])[::-1]
    # Restore original order
    result = np.zeros(m)
    result[order] = q
    return np.clip(result, 0, 1)


def _build_baskets(
    transactions: pd.DataFrame,
    config: AffinityConfig,
    as_of: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build baskets according to the configured grain and key policy.

    Returns:
        DataFrame with columns: customer_id, basket_key, items (list of item indices)
        diagnostics dict with eligible counts
    """
    item_col = "category" if config.basket_level == "category" else "product_id"

    # Filter positive purchases with valid identifiers up to cutoff
    base = transactions[
        (transactions["quantity"] > 0)
        & transactions["customer_id"].notna()
        & transactions["transaction_day"].notna()
        & transactions[item_col].notna()
        & (transactions["transaction_day"] <= as_of)
    ].copy()

    # Apply basket grain
    if config.basket_grain == "trip":
        base = base[["customer_id", "transaction_day", item_col]].drop_duplicates()
        basket_key_cols = ["customer_id", "transaction_day"]
        basket_key_name = "basket_day"
    else:  # transaction grain
        if config.transaction_key_mode == "customer-transaction":
            base = base[base["transaction_id"].notna()][
                ["customer_id", "transaction_id", "transaction_day", item_col]
            ].drop_duplicates()
            basket_key_cols = ["customer_id", "transaction_id"]
        else:  # transaction-only
            base = base[base["transaction_id"].notna()][
                ["transaction_id", "transaction_day", "customer_id", item_col]
            ].drop_duplicates()
            basket_key_cols = ["transaction_id"]
        basket_key_name = "basket_transaction"

    if base.empty:
        return pd.DataFrame(), {
            "status": "skipped",
            "reason": "no identifiable positive-purchase baskets",
            "eligible_customers": 0,
            "eligible_baskets": 0,
        }

    # Count baskets per customer
    n_b = base.groupby("customer_id")[basket_key_cols[1]].nunique()
    eligible_customers = int(len(n_b))
    eligible_baskets = int(n_b.sum())

    # Deterministic customer sampling if over budget
    h = pd.Series(_hash_series(pd.Series(n_b.index), config.random_seed), index=n_b.index)
    keep = n_b.loc[h.sort_values().index].cumsum() <= config.max_baskets
    keep_ids = keep[keep].index

    if len(keep_ids) < config.min_customers:
        return pd.DataFrame(), {
            "status": "skipped",
            "reason": f"fewer than {config.min_customers} customers within the basket budget",
            "eligible_customers": eligible_customers,
            "eligible_baskets": eligible_baskets,
        }

    base = base[base["customer_id"].isin(keep_ids)]

    # Candidate item selection (top by customer reach)
    pop = (
        base.groupby(item_col)["customer_id"]
        .nunique()
        .sort_values(ascending=False)
        .head(config.max_candidates)
    )
    rank = {k: i for i, k in enumerate(pop.index)}
    names = list(pop.index)
    base = base[base[item_col].isin(rank)].copy()
    base["ix"] = base[item_col].map(rank)

    # Aggregate to baskets with deduplication and per-basket item limit
    baskets = (
        base.groupby(["customer_id", basket_key_cols[1]])["ix"]
        .agg(lambda s: sorted(set(s))[: config.max_items_per_basket])
        .reset_index()
        .sort_values(["customer_id", basket_key_cols[1]])
    )
    baskets = baskets.rename(columns={basket_key_cols[1]: basket_key_name})

    return baskets, {
        "eligible_customers": eligible_customers,
        "eligible_baskets": eligible_baskets,
        "candidate_items": len(names),
        "candidate_names": names,
        "basket_key": basket_key_name,
        "item_level": config.basket_level,
        "basket_grain": config.basket_grain,
        "transaction_key_mode": config.transaction_key_mode,
    }


def analyze_affinity(
    transactions: pd.DataFrame,
    as_of: pd.Timestamp,
    config: AffinityConfig | None = None,
) -> AffinityResult:
    """Analyze basket affinity with bounded computational budgets.

    This is the shared implementation used by both CLI and Streamlit.

    Args:
        transactions: Normalized transaction DataFrame
        as_of: Analysis cutoff date
        config: AffinityConfig with budget and filtering parameters

    Returns:
        AffinityResult with associations, transitions, and diagnostics
    """
    if config is None:
        config = AffinityConfig()

    det: dict[str, Any] = {
        "status": "skipped",
        "reason": None,
        **config.to_dict(),
    }

    # Build baskets
    baskets, basket_info = _build_baskets(transactions, config, as_of)
    det.update(basket_info)

    if baskets.empty:
        return AffinityResult(
            pd.DataFrame(), pd.DataFrame(), det
        )

    # Split-half assignment for persistence testing
    n_customers_kept = baskets["customer_id"].nunique()
    customer_hashes = _hash_series(
        pd.Series(baskets["customer_id"].unique()), config.random_seed
    )
    half_map = dict(zip(baskets["customer_id"].unique(), customer_hashes & 1, strict=False))

    # Incremental pair counting with budget enforcement
    item_counts = Counter()
    pair_counts = Counter()
    half_item_counts = [Counter(), Counter()]
    half_pair_counts = [Counter(), Counter()]
    half_customer_counts = [0, 0]

    nxt_pair = Counter()
    nxt_from = Counter()
    nxt_to = Counter()
    n_transitions = 0

    ops = 0
    n_baskets_processed = 0
    n_customers_processed = 0

    for cid, grp in baskets.groupby("customer_id", sort=False):
        # Calculate cost for this customer's baskets
        cost = sum(len(s) * (len(s) - 1) // 2 for s in grp["ix"])
        if ops + cost > config.max_pair_operations:
            break

        ops += cost
        hh = half_map[cid]
        half_customer_counts[hh] += 1

        c_items, c_pairs, prev = set(), set(), None

        for items in grp["ix"]:
            n_baskets_processed += 1
            item_counts.update(items)
            pairs = list(combinations(items, 2))
            pair_counts.update(pairs)
            c_items.update(items)
            c_pairs.update(pairs)

            # Next-trip transitions
            if prev is not None:
                n_transitions += 1
                nxt_from.update(prev)
                nxt_to.update(items)
                nxt_pair.update((a_, b_) for a_ in prev for b_ in items)
            prev = items

        n_customers_processed += 1
        half_item_counts[hh].update(c_items)
        half_pair_counts[hh].update(c_pairs)

    det.update({
        "customers_analyzed": n_customers_processed,
        "baskets_analyzed": n_baskets_processed,
        "pair_operations": ops,
        "sampling": "customers ordered by hash; all baskets of a customer kept together",
    })

    N = sum(half_customer_counts)

    if n_baskets_processed == 0 or not pair_counts:
        det["reason"] = "no co-purchase pairs within the budget"
        return AffinityResult(pd.DataFrame(), pd.DataFrame(), det)

    # Compute association metrics
    names = list(range(config.max_candidates))  # indices
    rows = []

    for (i, j), cnt in pair_counts.items():
        if cnt / n_baskets_processed < config.min_support:
            continue

        n_ab = sum(half_pair_counts[hh].get((i, j), 0) for hh in (0, 1))
        n_a = sum(half_item_counts[hh].get(i, 0) for hh in (0, 1))
        n_b = sum(half_item_counts[hh].get(j, 0) for hh in (0, 1))

        rows.append((i, j, cnt, n_ab, n_a, n_b))

    if not rows:
        det["reason"] = "no pair reached --min-support"
        return AffinityResult(pd.DataFrame(), pd.DataFrame(), det)

    n_ab_arr = np.array([r[3] for r in rows], float)
    n_a_arr = np.array([r[4] for r in rows], float)
    n_b_arr = np.array([r[5] for r in rows], float)

    st = _pair_stats(n_ab_arr, n_a_arr, n_b_arr, N)
    q_vals = benjamini_hochberg(st["p_value"])

    # Half-sample lift for persistence
    half_lift = []
    for idx, (i, j, *_rest) in enumerate(rows):
        hl = []
        for hh in (0, 1):
            n_ab_h = half_pair_counts[hh].get((i, j), 0)
            n_a_h = half_item_counts[hh].get(i, 0)
            n_b_h = half_item_counts[hh].get(j, 0)
            if min(n_ab_h, n_a_h, n_b_h) > 0:
                hl.append((n_ab_h * half_customer_counts[hh]) / (n_a_h * n_b_h))
            else:
                hl.append(np.nan)
        half_lift.append(hl)

    half_lift = np.array(half_lift)

    # Build output DataFrame
    item_names = det.get("candidate_names", [])
    out_rows = []
    for idx, (i, j, cnt, *_rest) in enumerate(rows):
        for ante, cons in ((i, j), (j, i)):
            conf = cnt / item_counts[ante] if item_counts[ante] > 0 else 0
            lift_val = conf / (item_counts[cons] / n_baskets_processed) if item_counts[cons] > 0 else 0

            out_rows.append({
                "basket_level": config.basket_level,
                "basket_grain": config.basket_grain,
                "antecedent": item_names[ante] if ante < len(item_names) else str(ante),
                "consequent": item_names[cons] if cons < len(item_names) else str(cons),
                "pair_baskets": int(cnt),
                "basket_support": cnt / n_baskets_processed,
                "confidence": conf,
                "basket_lift": lift_val,
                "customers_with_pair": int(n_ab_arr[idx]),
                "customer_lift": float(st["lift"][idx]),
                "customer_odds_ratio": float(st["odds_ratio"][idx]),
                "customer_odds_ratio_ci95_lower": float(st["or_ci95_lower"][idx]),
                "p_value": float(st["p_value"][idx]),
                "q_value_bh": float(q_vals[idx]),
                "lift_half_a": float(half_lift[idx, 0]) if idx < len(half_lift) else np.nan,
                "lift_half_b": float(half_lift[idx, 1]) if idx < len(half_lift) else np.nan,
                "persistent_both_halves": bool(np.all(half_lift[idx] > 1.0)) if idx < len(half_lift) else False,
            })

    aff_df = pd.DataFrame(out_rows)
    det["pairs_tested"] = len(rows)

    if not aff_df.empty:
        aff_df["passes_all_filters"] = (
            (aff_df["confidence"] >= config.min_confidence)
            & (aff_df["basket_lift"] >= config.min_lift)
            & (aff_df["q_value_bh"] <= config.fdr_alpha)
            & (aff_df["customer_odds_ratio_ci95_lower"] > 1.0)
            & aff_df["persistent_both_halves"]
        )
        aff_df = aff_df.sort_values(
            ["passes_all_filters", "customer_lift"], ascending=[False, False]
        ).reset_index(drop=True)

    # Next-trip transitions
    trans_df = pd.DataFrame()
    if n_transitions >= 30:
        trows = [
            (a_, b_, c_) for (a_, b_), c_ in nxt_pair.items()
            if c_ / n_transitions >= config.min_support
        ]
        if trows:
            n_ab_t = np.array([t[2] for t in trows], float)
            n_a_t = np.array([nxt_from[t[0]] for t in trows], float)
            n_b_t = np.array([nxt_to[t[1]] for t in trows], float)
            st_t = _pair_stats(n_ab_t, n_a_t, n_b_t, n_transitions)
            q_t = benjamini_hochberg(st_t["p_value"])

            trans_df = pd.DataFrame({
                "basket_level": config.basket_level,
                "basket_grain": config.basket_grain,
                "from_item": [item_names[t[0]] if t[0] < len(item_names) else str(t[0]) for t in trows],
                "next_trip_item": [item_names[t[1]] if t[1] < len(item_names) else str(t[1]) for t in trows],
                "transitions": n_ab_t.astype(int),
                "p_next_given_from": n_ab_t / n_a_t,
                "p_next_overall": n_b_t / n_transitions,
                "next_trip_lift": st_t["lift"],
                "odds_ratio_ci95_lower": st_t["or_ci95_lower"],
                "p_value": st_t["p_value"],
                "q_value_bh": q_t,
            })
            trans_df = trans_df[
                (trans_df["q_value_bh"] <= config.fdr_alpha)
                & (trans_df["odds_ratio_ci95_lower"] > 1.0)
            ].sort_values("next_trip_lift", ascending=False)

            det["transitions_tested"] = len(trows)

    # Determine final status
    if n_baskets_processed < det["eligible_baskets"]:
        if ops >= config.max_pair_operations:
            det["status"] = "truncated"
            det["reason"] = "pair-operation budget reached"
        elif n_customers_processed < det["eligible_customers"]:
            det["status"] = "sampled"
            det["reason"] = "basket budget reached; deterministic customer sample used"
        else:
            det["status"] = "completed"
    else:
        det["status"] = "completed"

    return AffinityResult(aff_df, trans_df, det)


def analyze_affinity_from_app(
    raw_df: pd.DataFrame,
    period_params: dict,
    min_support: float = 0.01,
    basket_level: str = "product",
    basket_grain: str = "trip",
    max_candidates: int = 100,
    max_baskets: int = 5000,
    max_items_per_basket: int = 30,
    max_pair_operations: int = 100_000,
) -> dict:
    """Wrapper for Streamlit app compatibility.

    Returns dict with 'single' (product metrics) and 'pairs' (associations).
    """
    # Prepare transactions
    df = prepare_transaction_frame(raw_df)

    # Get period transactions
    from app import get_period_context, get_period_transactions, ensure_not_empty

    ctx = get_period_context(period_params)
    curr_df = get_period_transactions(df, ctx, "current")

    if not ensure_not_empty(curr_df, "basket analysis current period"):
        return {"single": pd.DataFrame(), "pairs": pd.DataFrame()}

    total_orders = curr_df["transaction_id"].nunique()

    # Single product metrics
    prod_metrics = (
        curr_df.groupby("product_id")
        .agg(
            orders=("transaction_id", "nunique"),
            revenue=("revenue", "sum"),
            units=("quantity", "sum"),
        )
        .reset_index()
    )
    prod_metrics["support"] = prod_metrics["orders"] / total_orders

    # Use shared affinity engine
    config = AffinityConfig(
        basket_level=basket_level,
        basket_grain=basket_grain,
        min_support=min_support,
        max_candidates=max_candidates,
        max_baskets=max_baskets,
        max_items_per_basket=max_items_per_basket,
        max_pair_operations=max_pair_operations,
    )

    result = analyze_affinity(curr_df, ctx["current_end"], config)

    # Convert to app-compatible format
    pairs_df = result.associations.copy()

    if not pairs_df.empty:
        # Add joint revenue
        joint_revenue = (
            curr_df[curr_df["product_id"].isin(prod_metrics["product_id"])]
            .groupby("transaction_id")
            .apply(
                lambda x: (
                    x["revenue"].sum()
                    if len(set(x["product_id"]) & set(prod_metrics["product_id"])) >= 2
                    else 0
                )
            )
            .reset_index(name="joint_revenue")
        )

        pair_joint_rev = {}
        for _, row in pairs_df.iterrows():
            a, b = row["antecedent"], row["consequent"]
            invoices_with_both = set(curr_df[curr_df["product_id"] == a]["transaction_id"]) & set(
                curr_df[curr_df["product_id"] == b]["transaction_id"]
            )
            joint_rev = joint_revenue[joint_revenue["transaction_id"].isin(invoices_with_both)][
                "joint_revenue"
            ].sum()
            pair_joint_rev[(a, b)] = joint_rev

        pairs_df["joint_revenue"] = pairs_df.apply(
            lambda r: pair_joint_rev.get((r["antecedent"], r["consequent"]), 0), axis=1
        )

        # Merge descriptions
        if "product_description" in raw_df.columns:
            desc_map = raw_df[["product_id", "product_description"]].drop_duplicates()
            pairs_df = pairs_df.merge(
                desc_map.rename(columns={"product_id": "antecedent", "product_description": "desc_a"}),
                on="antecedent",
                how="left",
            )
            pairs_df = pairs_df.merge(
                desc_map.rename(columns={"product_id": "consequent", "product_description": "desc_b"}),
                on="consequent",
                how="left",
            )

    return {
        "single": prod_metrics,
        "pairs": pairs_df,
        "diagnostics": result.diagnostics,
    }