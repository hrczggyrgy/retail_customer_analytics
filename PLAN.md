# Retail Customer Analytics — Implementation Plan & Todo List

Based on: `improvement_plan_v1.md` (audit date: October 10, 2026)

---

## Priority Definitions

- **P0**: Correctness or foundational reliability issue — must fix before trusting model comparisons or expanding use
- **P1**: High-value improvement for consistent analytics and robust evaluation
- **P2**: Valuable capability/usability enhancement after foundation is dependable
- **P3**: Conditional/advanced capability whose value is not yet established

---

## Phase 0 — Quick Wins & Correctness (P0/P1)

### R01: Fix Holdout Leakage in Population-Mean Baseline (P0)
- **Location**: `retail_customer_analysis.py` → `holdout_validation()` ~line 581
- **Change**: Estimate constant benchmark from calibration/training data only (not holdout outcomes)
- **Acceptance**: Baseline prediction unchanged when holdout labels modified; metric outputs reproduce on fixed synthetic data
- **Effort**: Small
- **Dependencies**: None

### R02: Unify Purchase-Event Grain, Identity & Financial Definitions (P0)
- **Locations**: 
  - `retail_customer_analysis.py`: `build_trips()` ~291, `transaction_summary()` ~949, `basket_affinity()` ~1042
  - `app.py`: `build_customer_features()` ~1060, `build_product_features()` ~1315, `build_basket_associations()` ~1455
- **Change**: Define shared policy for transaction IDs, qualifying purchases, trip aggregation, return handling, gross/net monetary measures
- **Acceptance**: Consistent event counts between UI/CLI; transaction ID reuse doesn't merge baskets; product summaries one row per key; revenue reconciliations hold
- **Effort**: Medium
- **Dependencies**: Business agreement on invoice vs customer-day semantics

### R03: Create Automated Regression Suite, CI & README (P0)
- **Locations**: New `tests/`, `README.md`, `.github/workflows/ci.yml`, `requirements-dev.txt`, `pyproject.toml`
- **Change**: Start with tests for leakage bug, event definitions, point-in-time features, return matching, model status, empty data, accounting invariants
- **Acceptance**: Clean checkout has setup instructions; deterministic synthetic data passes unit + e2e smoke tests; `show_all_results.py` works without local dataset; CI runs on changes; Python version + input schema documented
- **Effort**: Medium
- **Dependencies**: None

### R04: Correct Temporal/Data-Contract Issues & Currency Labels (P1)
- **Locations**:
  - `retail_customer_analysis.py`: `parse_dates()` ~196, `match_returns()` ~310
  - `app.py`: `fmt_currency()` ~1575, ingestion ~968
  - `data_generator.py`: `generate()` defaults ~228
- **Change**: Preserve timestamps for return matching; require/explicitly select ambiguous date formats; make currency explicit; prevent unmarked future-dated synthetic output
- **Acceptance**: Returns not attributed to later same-day purchases; ambiguous date parsing observable; consistent currency labels; default synthetic run doesn't extend beyond current date; assumptions in docs/metadata
- **Effort**: Medium
- **Dependencies**: None

### R08: Fix Example Workflow Dependency on Absent Sample File (P1)
- **Location**: `show_all_results.py`
- **Change**: Accept `--input`, use documented schema contract, optionally invoke shared synthetic generator
- **Acceptance**: Script works with supplied CSV/Parquet and synthetic-demo mode without `sample_data.parquet`
- **Effort**: Small
- **Dependencies**: None

---

## Phase 1 — Foundational Reliability (P1)

### R05: Consolidate App Preprocessing & CLI Validation (P1)
- **Locations**: New shared ingestion module; `app.py` `prepare_transaction_frame()` ~968; CLI `load_lines()`/`clean_lines()`
- **Change**: Single schema/cleaning implementation used by both entry points
- **Acceptance**: Same input → same normalized rows, quality counts, accounting totals through both entry points
- **Effort**: Medium
- **Dependencies**: R02 (event/data definitions)

### R06: Make Configuration & Metric Declarations Canonical (P1)
- **Locations**: `app.py` `APP_CONFIG` ~71, `METRIC_DEFINITIONS` ~85
- **Change**: Make authoritative or remove unused; share with CLI and report layer
- **Acceptance**: Changing one threshold/metric scope changes documented consumers; tests check formula/scope consistency
- **Effort**: Medium
- **Dependencies**: None

### R07: Expose Model Status & Fallback Behavior (P1)
- **Locations**: `retail_customer_analysis.py` `fit_bgnbd()`, `fit_gamma_gamma()`, `score_customers()`
- **Change**: Standardize status values (`fitted_and_accepted`, `fitted_but_not_accepted`, `fallback`, `skipped`, `failed`); report counts per status
- **Acceptance**: Non-converged/failed fits visible in artifacts; non-finite predictions rejected; fallback counts/reasons in metadata/report
- **Effort**: Medium
- **Dependencies**: None

### R09: Test Reproducibility Across Time Snapshots (P1)
- **Change**: Add snapshot feature builders + shared event semantics to test suite; construct cutoff/horizon cases
- **Acceptance**: Features at cutoff t invariant to rows after t; identical inputs/configs produce identical features
- **Effort**: Medium
- **Dependencies**: R02, R03, R05

---

## Phase 2 — Core Analytical Enhancements (P1/P2)

### R10: Rolling-Origin Validation (Multiple Historical Cutoffs) (P1)
- **Locations**: `retail_customer_analysis.py` `holdout_validation()` ~563, orchestration ~1314-1326
- **Change**: Run multiple historical cutoff dates; refit at each cutoff; evaluate on future horizon; include corrected baseline; save per-origin metrics
- **Acceptance**: Each origin records cutoff, window, sample count, baseline/model metrics, fit status; no feature/target data crosses cutoff incorrectly
- **Effort**: Medium
- **Dependencies**: R01, R09 (point-in-time features)

### R11: Broader Forecast & Calibration Metrics (P1)
- **Locations**: `retail_customer_analysis.py` `holdout_metrics`, `insight_engine.py`
- **Change**: Add WAPE, aggregate bias, metric CIs, richer diagnostics; Brier/log loss/calibration only for supervised probability target
- **Acceptance**: Every model-target pair has defined metrics, denominators, sample sizes, leakage-free baseline; HTML report shows performance across origins
- **Effort**: Medium
- **Dependencies**: R10

### R12: Extend Canonical Feature Module (P1)
- **Change**: Stable behavioral trend, basket composition/entropy, return-unit ratios, point-in-time category affinity; reconcile rolling windows & cadence features
- **Acceptance**: Features have formulas, valid missing-value behavior, cutoff invariants, stable dtypes, ablation results
- **Effort**: Medium
- **Dependencies**: R02, R05, R09

### R13: Segment Selection Validation (P1)
- **Locations**: `retail_customer_analysis.py`, `app.py` Trust/Customers views
- **Change**: Add RFM-vs-cluster comparison, cross-snapshot stability reporting
- **Acceptance**: Segment solutions report size, profile, bootstrap/time stability, reason for selection/rejection
- **Effort**: Medium
- **Dependencies**: R12

### R14: Fix Incomplete/Misleading Chart Panels (P2)
- **Location**: `insight_engine.py`
- **Change**: Implement cohort-revenue table + uncertainty; add reconciled revenue bridge when table produced; fix chart labels
- **Acceptance**: No blank measurement panels; chart title/axes/takeaway match calculation; revenue bridges reconcile
- **Effort**: Medium
- **Dependencies**: Stable upstream metrics

---

## Phase 3 — Optional/Advanced Capabilities (P2/P3)

### R15: Supervised Purchase Propensity Model (P2/P3)
- **Change**: Add logistic regression for next-horizon purchase probability behind separate task in new model module
- **Acceptance**: Out-of-time PR-AUC, calibration, Brier/log loss, top-k metrics reported against baselines; model kept only if adds value
- **Effort**: Medium (adds scikit-learn dependency)
- **Dependencies**: R10, R11, R12

### R16: Expected Contribution (Margin-Adjusted) Model (P2)
- **Change**: Separately named expected-contribution model + cohort value outputs when cost/margin/return economics available
- **Acceptance**: Forecasts use declared cost/return definitions; backtests validate contribution vs realized; revenue/contribution never mixed
- **Effort**: Large
- **Dependencies**: Financial data contract

### R17: Action Impact Testing Framework (P3)
- **Change**: Use action rules + exported entities for randomized pilot + holdout reporting; effect estimation after exposure/outcome data
- **Acceptance**: Predeclared experiment reports incremental outcomes with uncertainty, treatment cost, comparison group
- **Effort**: Large
- **Dependencies**: Operational/experimental infrastructure

### R18: Operational Lineage & Batch Contract (P2)
- **Change**: Standardize metadata, add input/config identity + schema versions, redact absolute paths, document batch scoring
- **Acceptance**: Every run declares input/schema identity, cutoff, horizon, event grain, versions, seed, model statuses, output files without raw local paths
- **Effort**: Medium
- **Dependencies**: R03, R07

---

## Suggested Execution Order

1. **Week 1-2**: R01, R03, R04, R08 (Quick wins)
2. **Week 3-5**: R02, R05, R06, R07, R09 (Foundational)
3. **Week 6-9**: R10, R11, R12, R13 (Core analytical)
4. **Week 10-11**: R14 (Reporting fixes)
5. **Future**: R15-R18 (Conditional on business needs & data availability)

---

## Key Architectural Target

```
retail_customer_analytics/
├── app.py                         # Thin Streamlit interface
├── data_generator.py              # Canonical synthetic-data generator
├── retail_customer_analysis.py    # CLI compatibility wrapper
├── insight_engine.py               # Reporting entry point
├── src/
│   └── retail_customer_analytics/
│       ├── schema.py
│       ├── ingestion.py
│       ├── events.py
│       ├── features.py
│       ├── models.py
│       ├── validation.py
│       ├── segmentation.py
│       ├── affinity.py
│       ├── reporting.py
│       └── metadata.py
├── tests/
│   ├── test_schema.py
│   ├── test_events.py
│   ├── test_features.py
│   ├── test_temporal_validation.py
│   ├── test_models.py
│   └── test_end_to_end.py
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
├── README.md
└── .github/
    └── workflows/
        └── ci.yml
```

---

## Critical Success Criteria (from Plan)

The improved tool must reliably answer:

1. **What happened?** Reconciled, clearly defined, time-scoped customer and retail metrics
2. **Who behaves differently?** Stable, interpretable segments and measurable behavioral differences
3. **What is likely to happen next?** Forecasts validated on future periods, compared with leakage-free baselines
4. **What decisions might be worth testing?** Identifiable customers/products, estimated opportunity, uncertainty, explicit assumptions
5. **Can the results be trusted and reproduced?** Point-in-time features, automated tests, recorded configuration, documented data limitations

---

## Open Questions to Resolve (Before Advanced Modeling)

| # | Question |
|---|----------|
| 1 | Are transaction IDs globally unique or only within customer/store/register/channel? |
| 2 | Should a purchase event mean an invoice, customer-day trip, or business-specific event? |
| 3 | Are transaction timestamps at time-of-day precision? Which timezone? |
| 4 | Are returns reliably tied to original transactions or inferred? |
| 5 | Is the dataset single-currency? Treat prices as EUR by default? |
| 6 | Are missing customer IDs common enough to need anonymous-customer policy? |
| 7 | Does data include true acquisition dates or only observed purchase history? |
| 8 | Main priority: reporting, repeat-purchase forecasting, retention, cross-sell, or LTV? |
| 9 | Decision horizon: 30, 60, 90, or other days? |
| 10 | Need rankings, calibrated probabilities, aggregate forecasts, or all three? |
| 11 | Will outputs inform analysts or automatically drive communications/offers? |
| 12 | Are margin, cost, campaign exposure, intervention outcomes available? |
| 13 | Normal/largest expected transaction-file sizes? |
| 14 | Runs locally only or hosted for multiple users? |
| 15 | Customer-level exports permitted? Retention/deletion rules for uploaded data? |
| 16 | Must historical runs repeat exactly or within documented tolerance? |

---

## Notes

- All effort estimates are provisional (assume 1 dev familiar with pandas/codebase)
- No independent execution of full app/test suite was performed during audit
- Plan calls for tests where test evidence is absent rather than asserting runtime failures
- Dependency upgrades should be controlled and tested, not blind updates
- Keep production stack small; add deps only when concrete analytical requirement justifies them