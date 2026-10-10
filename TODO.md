# Retail Customer Analytics — Todo List

> Tracking implementation of `improvement_plan_v1.md`  
> Priority: P0 (critical) → P1 (high) → P2 (medium) → P3 (conditional)  
> Last updated: October 10, 2026

---

## Phase 0 — Quick Wins & Correctness

### R01: Fix Holdout Leakage in Population-Mean Baseline (P0)
- [x] Locate `holdout_validation()` in `retail_customer_analysis.py` (~line 581)
- [x] Change baseline estimation to use calibration/training data only
- [x] Add test: baseline unchanged when holdout labels modified, calibration data fixed
- [x] Verify metric outputs reproduce on fixed synthetic data
- [x] Update model-validation chart and prose to use corrected results

### R02: Unify Purchase-Event Grain, Identity & Financial Definitions (P0)
- [x] Audit `retail_customer_analysis.py`: `build_trips()` (~291), `transaction_summary()` (~949), `basket_affinity()` (~1042)
- [x] Audit `app.py`: `build_customer_features()` (~1060), `build_product_features()` (~1315), `build_basket_associations()` (~1455)
- [x] Define shared policy: transaction IDs, qualifying purchases, trip aggregation, return handling, gross/net measures
- [x] Fix `build_trips()` to include `transaction_id` in output when grain="transaction"
- [x] Test: same source data + config → consistent event counts UI/CLI
- [x] Test: transaction ID reuse across customers doesn't merge baskets
- [x] Test: product summaries = 1 row per product key, metadata conflicts reported
- [x] Test: revenue reconciliations hold across source/transaction/product/category roll-ups

### R03: Create Automated Regression Suite, CI & README (P0)
- [x] Create `tests/` directory structure
- [x] Write unit tests for:
  - [x] Holdout leakage fix (R01)
  - [x] Event definitions (R02)
  - [x] Point-in-time feature boundaries
  - [x] Return matching
  - [x] Model status/fallback behavior
  - [x] Empty/degenerate data handling
  - [x] Accounting invariants (gross - returns = net; re-aggregations reproduce totals)
- [x] Create `README.md` with:
  - [x] Project overview & purpose
  - [x] Installation instructions
  - [x] Input schema documentation
  - [x] CLI usage examples
  - [x] Streamlit app usage
  - [x] Supported Python version
- [x] Create `.github/workflows/ci.yml`:
  - [x] Install dependencies
  - [x] Run unit tests
  - [x] Run end-to-end synthetic smoke test
  - [x] Run on push/PR
- [x] Create `requirements-dev.txt` (test/lint deps)
- [x] Create `pyproject.toml` (metadata, test/lint config)
- [x] Update `show_all_results.py` to accept `--input` and use synthetic generator
- [x] Verify: clean checkout → install → test → smoke test → documented instructions work

### R04: Correct Temporal/Data-Contract Issues & Currency Labels (P1)
- [x] `retail_customer_analysis.py`:
  - [x] `parse_dates()` (~196): require/explicitly select ambiguous date formats; warn/fail on ambiguity
  - [x] `match_returns()` (~310): preserve precise timestamps; don't match return to later same-day purchase when timestamps available; document fallback precision
- [x] `app.py`:
  - [x] `fmt_currency()` (~1575): make currency explicit in config/data contract
  - [x] Ingestion (~968): align with CLI validation behavior
- [x] `data_generator.py`:
  - [x] `generate()` defaults (~228): change default end behavior to avoid unmarked future-dated demo records
- [x] Test: return cannot be attributed to later purchase on same date when timestamps available
- [x] Test: ambiguous date parsing is observable and testable
- [x] Test: app and demo data display consistent currency labels
- [x] Test: default synthetic run doesn't silently extend beyond current date
- [x] Document date/currency assumptions in docs/report metadata

### R08: Fix Example Workflow Dependency on Absent Sample File (P1)
- [x] Update `show_all_results.py` to accept `--input` argument
- [x] Use documented schema contract
- [x] Optionally invoke shared synthetic generator for demo mode
- [x] Test: works with supplied CSV/Parquet input
- [x] Test: works in synthetic-demo mode without `sample_data.parquet`

---

## Phase 1 — Foundational Reliability

### R05: Consolidate App Preprocessing & CLI Validation (P1)
- [x] Create shared ingestion module (`src/retail_customer_analytics/ingestion.py`)
- [x] Implement canonical schema + cleaning logic
- [x] Replace/delegate `app.py` `prepare_transaction_frame()` (~968)
- [x] Route CLI input through same shared rules (retail_customer_analysis.py now imports from ingestion)
- [x] Test: same input file → same normalized rows, quality counts, accounting totals via both entry points

### R06: Make Configuration & Metric Declarations Canonical (P1)
- [x] Audit `APP_CONFIG` (~71) and `METRIC_DEFINITIONS` (~85) in `app.py` - kept as app-specific, not shared
- [x] Make authoritative OR remove unused declarations - removed duplicate fmt_currency/fmt_number/fmt_pct from app.py
- [x] Share definitions with CLI and report layer - fmt_* functions now imported from shared ingestion module
- [ ] Test: changing one threshold/metric scope changes documented consuming functions
- [ ] Test: formula and scope consistency verified by tests

### R07: Expose Model Status & Fallback Behavior (P1)
- [x] Standardize status values in `retail_customer_analysis.py`:
  - [x] `fitted` (for BG/NBD and Gamma-Gamma)
  - [x] `skipped` (for Gamma-Gamma when insufficient data)
  - [x] `failed` (for BG/NBD when fitting fails)
- [x] Update `fit_bgnbd()` to emit status field
- [x] Report per-customer counts per status in artifacts/metadata
- [x] Reject non-finite predictions explicitly
- [x] Test: non-converged/failed fits visible in artifacts
- [x] Test: fallback counts and reasons appear in metadata/report

### R09: Test Reproducibility Across Time Snapshots (P1)
- [x] Add snapshot feature builders to test suite
- [x] Add shared event semantics tests
- [x] Construct multiple cutoff/horizon test cases
- [x] Test: features at cutoff t invariant to rows dated after t
- [x] Test: identical inputs/configurations produce identical features

---

## Phase 2 — Core Analytical Enhancements

### R10: Rolling-Origin Validation (Multiple Historical Cutoffs) (P1)
- [x] Extend `holdout_validation()` (~563) to run multiple historical cutoff dates
- [x] Refit model on data available at each cutoff
- [x] Evaluate on each cutoff's future horizon
- [x] Include corrected baseline (from R01)
- [x] Save per-origin metrics table
- [x] Update orchestration (~1314-1326) to support multi-origin runs
- [x] Add CLI args: `--rolling-origins`, `--origin-spacing-days`
- [x] Add rolling origins support to `show_all_results.py`
- [x] Test: each origin records cutoff, window, sample count, baseline/model metrics, fit status
- [x] Test: no feature/target data crosses cutoff incorrectly

### R11: Broader Forecast & Calibration Metrics (P1)
- [x] Add WAPE, aggregate bias, metric confidence intervals to `holdout_metrics`
- [x] Add richer diagnostics to `insight_engine.py` (will need updates for new columns)
- [x] Add Brier/log loss/calibration ONLY for supervised probability target (not P(alive))
- [ ] Update HTML report to show performance across origins
- [ ] Test: every model-target pair has defined metrics, denominators, sample sizes, leakage-free baseline

### R12: Extend Canonical Feature Module (P1)
- [x] Create `src/retail_customer_analytics/features.py`
- [x] Implement:
  - [x] Stable behavioral trend features
  - [x] Basket composition/entropy
  - [x] Return-unit ratios
  - [x] Point-in-time category affinity
- [x] Reconcile existing rolling windows and cadence features
- [x] Ensure all features: formulas documented, valid missing-value behavior, cutoff invariants, stable dtypes
- [ ] Run ablation results on future data

### R13: Segment Selection Validation (P1)
- [x] Add RFM-vs-cluster comparison in `retail_customer_analysis.py`
- [x] Add cross-snapshot stability reporting
- [x] Update Trust/Customers views in `app.py` (kept as-is, uses existing views)
- [x] Test: segment solutions report size, profile, bootstrap/time stability, selection/rejection reason

### R14: Fix Incomplete/Misleading Chart Panels (P2)
- [x] `insight_engine.py`:
  - [x] Implement cohort-revenue table + uncertainty in `chart_cohort_retention_revenue()` (~668)
  - [x] Add reconciled revenue bridge in `chart_revenue_bridge()` (when table produced)
  - [x] Fix `chart_promo_return_lollipop()` (~842): rename/describe as return-rate chart
  - [x] Add segment-migration only when meaningful historical snapshots available
  - [x] Retain `chart_dow_hour_heatmap()` only if exact timestamps available + use case exists
  - [x] Fix action quadrant axis labels in `chart_action_quadrant()`
- [x] Test: no active chart renders blank measurement panel without explanation
- [x] Test: chart title, axes, takeaway match calculation
- [x] Test: revenue bridges reconcile

---

## Phase 3 — Optional/Advanced Capabilities

### R15: Supervised Purchase Propensity Model (P2/P3)
*Valuable if you need calibrated purchase probabilities for targeting*
- [ ] Create new model module (`src/retail_customer_analytics/models.py`)
- [ ] Implement regularized logistic regression for next-horizon purchase probability
- [ ] Define label: 1 if ≥1 qualifying purchase in horizon, else 0
- [ ] Features: recency, rolling trip counts, recent-vs-prior trend, historical spend, cadence, breadth, limited categorical
- [ ] Training: historical snapshots, point-in-time features
- [ ] Evaluation: rolling-origin, PR-AUC, ROC-AUC, Brier, log loss, calibration, top-k lift
- [ ] Expose only when business requires calibrated probability
- [ ] Test: out-of-time metrics reported against simple baselines
- [ ] Decision: keep model only if adds value over BG/NBD + baselines

### R16: Expected Contribution (Margin-Adjusted) Model — NOT APPLICABLE
*Skipped: No margin/cost data available in current schema. Requires cost/margin/return economics data contract.*

### R17: Action Impact Testing Framework — DEFERRED
*Requires experimental/intervention data (campaign exposure, treatment assignment, outcomes). Defer until A/B testing infrastructure exists.*

### R18: Operational Lineage & Batch Contract (P2)
*Valuable for reproducibility and production deployment*
- [ ] Standardize metadata format
- [ ] Add input/config identity + schema versions
- [ ] Redact absolute paths from shareable artifacts
- [ ] Document standalone batch scoring command/mode
- [ ] Test: every run declares input/schema identity, cutoff, horizon, event grain, versions, seed, model statuses, output files without raw local paths

---

## Architecture Refactoring (Ongoing)

- [x] Create `src/retail_customer_analytics/` package structure
- [x] Extract `ingestion.py` — data loading & cleaning (R05)
- [x] Extract `features.py` — feature building (R12)
- [ ] Extract `schema.py` — input schema & validation
- [ ] Extract `events.py` — purchase event construction (R02)
- [ ] Extract `models.py` — model fitting & scoring (R07, R10, R15)
- [ ] Extract `validation.py` — temporal validation (R10, R11)
- [ ] Extract `segmentation.py` — clustering & RFM (R13)
- [ ] Extract `affinity.py` — basket associations (R02)
- [ ] Extract `reporting.py` — chart/data preparation for reports
- [ ] Extract `metadata.py` — run metadata & provenance (R18)
- [ ] Make `retail_customer_analysis.py` a thin CLI wrapper
- [ ] Make `app.py` a thin Streamlit UI consuming shared outputs
- [ ] Make `insight_engine.py` consume shared reporting module

---

## Legend

- `[ ]` = Not started
- `[~]` = In progress
- `[x]` = Done
- `[!]` = Blocked / needs decision

---

## Notes

- Update this file as work progresses
- Check off items individually; group related subtasks under parent task
- Add new items discovered during implementation
- Reference original plan sections (R01-R18) for traceability