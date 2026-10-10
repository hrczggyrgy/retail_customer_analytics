# Retail Customer Analytics

A lightweight, validated retail customer analytics pipeline for transaction-level data. Implements probabilistic customer models (BG/NBD, Gamma-Gamma), RFM segmentation, stability-checked clustering, survival analysis, basket affinity testing, and time-based holdout validation — all with explicit statistical safeguards and reproducible outputs.

## Features

### Core Analytics
- **Data Quality & Accounting**: Typed ingestion, issue counts, identifier consistency checks, accounting invariants (gross - returns = net)
- **Return Matching**: Links returns to latest earlier purchase of same customer/product; reports unmatched share
- **Purchase Events**: Configurable grain — customer-day trips (merges split receipts) or per-transaction
- **Customer Features**: RFM, rolling windows, recent-vs-prior change, breadth (distinct products/categories), cadence statistics, price-index promo proxy, return ratios
- **Probabilistic Models**: 
  - BG/NBD for expected purchase counts and P(alive)
  - Gamma-Gamma for expected trip value
  - Expected revenue = expected trips × expected trip value
- **Time-Based Holdout Validation**: Fit on calibration period, score on future horizon; compares against leakage-free baselines
- **Segmentation**: Robust-scaled K-means with silhouette and bootstrap stability (adjusted Rand index) for k-selection
- **Survival Analysis**: Kaplan-Meier time-to-second-trip with left-censoring awareness; cohort retention for new customers
- **Basket Affinity**: Customer-level association tests with FDR correction, Woolf odds-ratio CIs, split-half persistence, next-trip transitions

### Reporting
- **Static HTML Dashboard**: Self-contained with embedded charts, KPI cards, and numbered takeaways
- **Markdown Report**: Human-readable summary with model diagnostics, segment profiles, validation metrics
- **Structured Artifacts**: CSV/Parquet tables, JSON metadata, run provenance

### Engineering
- **Reproducible**: Fixed seeds, pinned dependencies, saved model parameters, run metadata
- **Tested**: Unit tests for event definitions, temporal validation, model status, accounting invariants
- **CI/CD**: GitHub Actions workflow with multi-Python testing and end-to-end smoke tests

## Installation

```bash
# Clone and install
git clone https://github.com/hrczggyrgy/retail_customer_analytics.git
cd retail_customer_analytics

# Install production dependencies
pip install -r requirements.txt

# Install development dependencies (for testing/linting)
pip install -r requirements-dev.txt
```

## Quick Start

### Command Line

```bash
# Generate synthetic demo data
python data_generator.py --customers 1000 --start 2025-01-01 --seed 42 --output demo_data

# Run analysis pipeline
python retail_customer_analysis.py \
    --input demo_data/transactions.csv \
    --output-dir retail_output \
    --horizon-days 90 \
    --windows-days 30,90,365 \
    --frequency-grain trip

# Generate HTML dashboard
python insight_engine.py --results-dir retail_output --output-dir retail_insights

# View results
open retail_insights/insights.html
```

### Streamlit App

```bash
streamlit run app.py
```

Upload your transaction CSV/Parquet or use the built-in synthetic data generator.

### View All Results (Script)

```bash
# With your own data
python show_all_results.py --input transactions.csv

# With synthetic demo data
python show_all_results.py --synthetic --n-customers 500 --days 180 --seed 42
```

## Input Schema

Required columns (exact names or common aliases):

| Column | Type | Description |
|--------|------|-------------|
| `customer_id` | string | Customer identifier (nullable for anonymous) |
| `transaction_date` | datetime | Transaction timestamp (ISO format preferred) |
| `transaction_id` | string | Transaction/basket/invoice identifier |
| `product_id` | string | Product/SKU identifier |
| `product_description` | string | Product description |
| `department` | string | Department/category group |
| `category` | string | Product category |
| `price` | float | Unit price (paid price, not list price) |
| `quantity` | int | Quantity (negative = return) |

**Currency**: All prices assumed to be in EUR by default. Configure via `DEFAULT_CURRENCY` in `app.py`.

**Date Format**: ISO 8601 (`YYYY-MM-DD` or `YYYY-MM-DD HH:MM:SS`) recommended. Use `--date-format` for other formats.

## Key Concepts

### Event Grain
- **Trip** (default): One event per customer-day. Merges split receipts same customer same day.
- **Transaction**: One event per `(customer_id, transaction_id)` pair.

Choose based on your business definition of a "purchase occasion."

### Holdout Validation
The pipeline uses a **time-based holdout**: models are fitted on data up to `as_of_date - horizon`, then scored on the following `horizon` days. This mimics real deployment where future is unknown.

**Critical**: The population-mean baseline is estimated from calibration data only — never from holdout outcomes (prevents leakage).

### Revenue vs Profit
All monetary outputs are **gross purchase revenue**, not profit. No cost, margin, discount, or tax fields exist in the schema.

### Promo Proxy
Price-index discount flag (`price <= 0.9 × product median`) is a **proxy**, not confirmed promotion exposure. Causal claims not supported.

## Output Artifacts

```
output_dir/
├── data_quality.json           # Issue counts, exclusions, accounting totals
├── run_metadata.json           # Run config, versions, timing, skipped analyses
├── analysis_report.md          # Human-readable Markdown report
├── customer_features.csv       # Per-customer features, scores, predictions
├── holdout_metrics.csv         # Model vs baseline metrics (MAE, RMSE, AUC, etc.)
├── holdout_calibration.csv     # Decile calibration tables
├── model_parameters.json       # Fitted BG/NBD, Gamma-Gamma parameters
├── cluster_diagnostics.csv     # Silhouette, ARI, stability for each k
├── cluster_profiles.csv        # Segment profiles and labels
├── rfm_segment_profiles.csv    # RFM segment sizes and metrics
├── repeat_survival.csv         # Kaplan-Meier survival curves
├── cohort_retention_new_customers.csv  # Monthly retention by cohort
├── basket_affinity.csv         # Association rules with statistical tests
├── next_trip_affinity.csv      # Next-trip transition probabilities
├── category_summary.csv        # Category revenue, returns, repeat buyers
├── department_summary.csv      # Department-level summaries
├── monthly_summary.csv         # Monthly revenue trends
├── product_summary.csv         # Product performance with repeat-buyer rates
├── transaction_summary.csv     # Transaction-level accounting
└── insights/
    ├── insights.html           # Self-contained HTML dashboard
    ├── insight_summary.json    # KPIs, chart manifest, skipped reasons
    └── *.png                   # Individual chart images
```

## Configuration

Key CLI arguments:

```bash
--input PATH              # Input CSV/Parquet (required)
--output-dir DIR          # Output directory (default: retail_analysis_output)
--as-of-date YYYY-MM-DD   # Analysis cutoff (default: max date in data)
--horizon-days INT        # Forecast/holdout horizon (default: 90)
--windows-days INT,...    # Rolling windows (default: 30,90,365)
--burn-in-days INT        # Burn-in for cohort analysis (default: 90)
--frequency-grain trip|transaction  # Event definition (default: trip)
--transaction-key-mode customer-transaction|transaction-only
--min-clusters INT        # Min K for clustering (default: 3)
--max-clusters INT        # Max K for clustering (default: 7)
--random-seed INT         # Random seed (default: 42)
```

## Testing

```bash
# Run unit tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=html

# Run end-to-end smoke test
python show_all_results.py --synthetic --n-customers 200 --days 90
```

## Project Structure

```
retail_customer_analytics/
├── app.py                      # Streamlit web application
├── retail_customer_analysis.py # Core analytical pipeline (CLI)
├── insight_engine.py           # Chart generation & HTML dashboard
├── data_generator.py           # Synthetic data generator
├── show_all_results.py         # End-to-end demo script
├── requirements.txt            # Production dependencies
├── requirements-dev.txt        # Development dependencies
├── pyproject.toml              # Project metadata, tool config
├── README.md                   # This file
├── .github/workflows/ci.yml    # CI/CD pipeline
├── tests/
│   ├── conftest.py             # Shared fixtures
│   ├── test_events.py          # Purchase events, returns, baskets
│   ├── test_models.py          # Model fitting, status, fallbacks
│   └── test_temporal_validation.py  # Holdout, point-in-time features
└── src/                        # (Future: modular package structure)
```

## Limitations

- **Revenue ≠ Profit**: No cost/margin fields; expected revenue is undiscounted
- **First-Observed ≠ Acquisition**: Cohorts based on first seen in data, not true signup
- **Promo Proxy ≠ Promotion**: Price variation used as discount proxy; no causal claims
- **Associations ≠ Causation**: Basket rules are co-occurrence, not lift-from-intervention
- **P(alive) ≠ Purchase Probability**: BG/NBD's P(alive) is not calibrated for next-horizon purchase
- **Anonymous Customers**: Rows without `customer_id` excluded from customer-level results

## Roadmap

See `PLAN.md` for detailed improvement plan. Priority areas:
1. **P0**: Fix holdout leakage (done), unify event definitions, add tests/CI
2. **P1**: Rolling-origin validation, canonical feature module, model status reporting
3. **P2**: Supervised propensity, margin-adjusted value, action impact testing

## Contributing

1. Fork and create a feature branch
2. Add tests for new functionality
3. Run `ruff check . && ruff format .` 
4. Run `python -m pytest tests/`
5. Submit PR with clear description

## License

MIT License — see LICENSE file for details.

## References

- BG/NBD: Fader, Hardie & Lee (2005), "Counting Your Customers the Easy Way"
- Gamma-Gamma: Fader & Hardie (2013), "Modeling Customer Lifetime Value"
- Temporal Validation: scikit-learn TimeSeriesSplit
- Probability Calibration: scikit-learn calibration guide
- Responsible AI: NIST AI Risk Management Framework