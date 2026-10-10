"""Shared test fixtures and configuration for retail_customer_analytics tests."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import sys

# Add the package root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import retail_customer_analysis as rca
import data_generator as dg


@pytest.fixture
def sample_transactions():
    """Generate a small fixed synthetic dataset for testing."""
    np.random.seed(42)
    n_customers = 100
    days = 180
    
    rng = np.random.default_rng(42)
    k = 5
    cats = [f"Category {i:02d}" for i in range(k)]
    depts = [f"Department {i % 2 + 1}" for i in range(k)]
    prod_per_cat = 4
    base_price = {c: float(rng.lognormal(np.log(5 + 2 * c), 0.2)) for c in range(k)}
    products = {c: [(f"P{c:02d}{j:02d}", round(base_price[c] * float(rng.uniform(0.8, 1.3)), 2)) for j in range(prod_per_cat)] for c in range(k)}
    lam = rng.gamma(0.5, 1 / 8.0, n_customers)
    p_drop = rng.beta(0.8, 2.0, n_customers)
    first = rng.integers(0, max(int(days * 0.6), 1), n_customers)
    pref = rng.dirichlet(np.ones(k) * 0.5, n_customers)
    size_factor = rng.gamma(1.5, 0.5, n_customers)
    t0 = pd.Timestamp("2024-01-01")
    rows = []
    for i in range(n_customers):
        horizon = days - first[i]
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
                rows.append((cid, day.strftime("%Y-%m-%d"), tid, pid, f"Product {pid}", depts[c], cats[c], price, qty))
                if rng.random() < 0.02:
                    rday = day + pd.Timedelta(days=int(rng.integers(1, 10)))
                    if rday <= t0 + pd.Timedelta(days=days):
                        rows.append((cid, rday.strftime("%Y-%m-%d"), f"R{tid}", pid, f"Product {pid}", depts[c], cats[c], price, -qty))
    df = pd.DataFrame(rows, columns=["customer_id", "transaction_date", "transaction_id", "product_id", "product_description", "department", "category", "price", "quantity"])
    return df


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def cli_args():
    """Default CLI arguments for testing."""
    import argparse
    parser = rca.build_parser()
    # Parse with a dummy input to satisfy required argument
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("customer_id,transaction_date,transaction_id,product_id,product_description,department,category,price,quantity\n")
        dummy_path = f.name
    
    try:
        args = parser.parse_args([
            "--input", dummy_path,
            "--horizon-days", "30",
            "--windows-days", "7,14,30",
            "--burn-in-days", "30",
            "--random-seed", "42",
            "--min-clusters", "2",
            "--max-clusters", "4",
            "--cluster-max-customers", "500",
            "--stability-bootstraps", "3",
            "--min-support", "0.01",
            "--min-confidence", "0.1",
            "--max-affinity-items", "20",
            "--max-affinity-baskets", "1000",
            "--max-items-per-basket", "10",
            "--max-affinity-pair-operations", "50000",
            "--fdr-alpha", "0.1",
        ])
    finally:
        import os
        os.unlink(dummy_path)
    return args