# AGENTS.md — Retail Customer Analytics

## Quick Commands

```bash
# Install
pip install -r requirements.txt
pip install -r requirements-dev.txt  # for testing/linting

# Run tests
python -m pytest tests/ -v                    # all tests (89 tests, ~37s)
python -m pytest tests/test_basket_affinity.py -v  # single test file
python -m pytest tests/ -k "test_trip_grain" -v    # filter by name

# Lint & Format
ruff check . --output-format=github
ruff format . --check
ruff format .  # auto-fix

# Type check
mypy .  # optional, not in CI

# CLI Pipeline
python retail_customer_analysis.py --input data.csv --output-dir out --horizon-days 90
python insight_engine.py --results-dir out --output-dir insights  # HTML dashboard

# Streamlit App
streamlit run app.py

# Synthetic Data / Smoke Tests
python data_generator.py --customers 1000 --start 2025-01-01 --seed 42 --output demo_data
python show_all_results.py --synthetic --n-customers 200 --days 90 --seed 42
```

## Architecture — Ground Truth

**Canonical Dataflow** (see `DATAFLOW.mmd` for diagram):
```
RAW CSV → ingestion.prepare_transaction_frame() → NORMALIZED DF
    ↓
    ├── events.py → build_purchase_events(), build_baskets(), match_returns()
    ├── affinity.py → analyze_affinity() (bounded, incremental pair counting)
    └── features.py → build_customer_snapshot() (71+ cols, return-aware)
```

**Single Source of Truth per Task:**
| Task | Module | Function |
|------|--------|----------|
| Trip/Event Construction | `events.py` | `build_purchase_events()` |
| Basket Construction | `events.py` | `build_baskets()` |
| Return Matching | `events.py` | `match_returns()` |
| Price Index | `events.py` | `add_price_index()` |
| Basket Affinity | `affinity.py` | `analyze_affinity()` |
| Customer Features | `features.py` | `build_customer_snapshot()` |
| Feature Validation | `features.py` | `validate_feature_contract()` |

**NO DUPLICATES** — These were removed/replaced:
- ❌ `app.py::build_basket_associations()` → `affinity.analyze_affinity_from_app()`
- ❌ `app.py::_compute_basket_analysis()` → `affinity.analyze_affinity_from_app()`
- ❌ `retail_customer_analysis.py::build_trips()` → wrapper to `events.build_purchase_events()`
- ❌ `retail_customer_analysis.py::basket_affinity()` → wrapper to `affinity.analyze_affinity()`

## Key Constraints

1. **Explicit Budgets** — `AffinityConfig` enforces: `max_candidates`, `max_baskets`, `max_items_per_basket`, `max_pair_operations`, `min_customers`. Always returns `status` + diagnostics.

2. **Feature Contract** — 21 REQUIRED features must exist (enforced by `validate_feature_contract()`):
   ```
   n_trips, recency_days, gross_spend, net_spend, return_value,
   return_value_ratio, x, T, t_x, mean_trip_value, ...
   ```

3. **Return-Aware Semantics** (current dataset has no returns, but features maintain distinction):
   - `net_spend = gross_spend - return_value`
   - `return_value_ratio = return_value / gross_spend` (0.0 when no returns)

4. **Point-in-Time Correctness** — All shared functions accept `as_of` cutoff. Features only use data ≤ cutoff.

5. **Deterministic** — `random_seed` propagated through all config classes (`AffinityConfig`, `EventConfig`, CLI args).

6. **Status Tracking** — Affinity returns `completed|sampled|truncated|skipped|failed` with coverage/budget metadata.

## Testing Quirks

- **Fixtures**: `tests/conftest.py` has shared transaction fixtures
- **Run single test**: `python -m pytest tests/test_basket_affinity.py::TestBasketAffinity::test_trip_grain_merges_same_day_transactions -v`
- **Warnings**: Affinity tests emit RuntimeWarnings (log/zero in odds-ratio calc) — expected, not failures
- **E2E smoke test**: `python show_all_results.py --synthetic --n-customers 200 --days 90 --seed 42` (runs in CI)

## Project Structure

```
├── app.py                      # Streamlit UI (uses shared modules)
├── retail_customer_analysis.py # CLI pipeline (wrappers to shared modules)
├── insight_engine.py           # Charts + HTML dashboard (reads CSV artifacts)
├── data_generator.py           # Synthetic data (returns (df, meta) tuple)
├── show_all_results.py         # End-to-end demo script
├── src/retail_customer_analytics/
│   ├── __init__.py             # Exports all public APIs
│   ├── ingestion.py            # Schema validation, normalization
│   ├── events.py               # Canonical event/basket/return construction
│   ├── affinity.py             # Bounded basket affinity engine
│   ├── features.py             # Canonical feature builder + validation
│   ├── metrics.py              # Business metrics helpers
│   ├── period_manager.py       # Date ranges, period filtering
│   └── polars_utils.py         # Polars conversions (legacy, not used in core)
├── tests/
│   ├── conftest.py             # Shared fixtures
│   ├── test_basket_affinity.py # Event grain, exact counts, transitions
│   ├── test_resource_limits.py # Budgets, sampling, correctness
│   ├── test_features.py        # Feature contract, point-in-time
│   ├── test_models.py          # BG/NBD, Gamma-Gamma, fallbacks
│   └── test_temporal_validation.py  # Holdout, leakage prevention
├── .github/workflows/ci.yml    # CI: test (3.10,3.11,3.12), lint, e2e smoke
├── pyproject.toml              # Tool config (ruff, pytest, mypy, coverage)
├── requirements.txt            # numpy, pandas, scipy, matplotlib, streamlit, pyarrow
├── requirements-dev.txt        # pytest, pytest-cov, ruff, mypy
└── DATAFLOW.mmd                # Mermaid architecture diagram (ground truth)
```

## Common Gotchas

| Issue | Fix |
|-------|-----|
| `data_generator.generate()` returns `(df, meta)` tuple | Unpack: `df, meta = generate(...)` |
| Streamlit app needs `src` on sys.path | App handles this internally via `sys.path.insert(0, str(APP_DIR / "src"))` |
| CLI `build_trips()` is a wrapper | Don't modify logic there — change `events.py` |
| Affinity uses integer indices internally | Output maps back to names via `candidate_names` in diagnostics |
| Feature builder needs `cust.index` for return features | Call `_return_features(transactions, cust.index, as_of)` AFTER core features |

## CI Order

```yaml
# .github/workflows/ci.yml runs:
1. lint (ruff check + format)
2. test (pytest on 3.10, 3.11, 3.12)
3. e2e smoke (show_all_results.py --synthetic)
```

Run locally in same order:
```bash
ruff check . && ruff format . --check && python -m pytest tests/ -v
```

## Key Files to Read First

1. `DATAFLOW.mmd` — Architecture diagram (source of truth)
2. `README.md` — Quick start, schema, concepts, limitations
3. `src/retail_customer_analytics/__init__.py` — Public API exports
4. `improvement_plan_v2.md` — Current priorities (P0: budgets, feature contract, tests)