# Retail Customer Analytics — Improvement Plan

Repository: [hrczggyrgy/retail_customer_analytics](https://github.com/hrczggyrgy/retail_customer_analytics) Audit date: October 10, 2026 Scope: Repository implementation, statistical validity, customer features, predictive evaluation, visualizations, and end-to-end engineering.

Evidence conventions

- Verified: Directly observed in the repository's `main` branch.
- Inference: A likely weakness inferred from the code structure or behavior; verify against actual datasets and business requirements.
- Conditional: A recommendation that depends on data availability, sample size, commercial objectives, or deployment requirements not established by the repository.

Effort and impact estimates in the roadmap are provisional. No independent execution of the application or complete test suite was performed as part of this source review.

# 1. Executive Summary

## 1.1 Overall assessment

This repository already has a substantial analytical foundation. It goes beyond basic customer summaries: it implements probabilistic repeat-purchase modeling, customer value estimates, RFM and behavioral clustering, retention analysis, basket association testing, data-quality reporting, and both Streamlit and static HTML reporting.

The next major improvement should be analytical correctness and consistency, not a larger collection of algorithms.

The highest-priority verified issue is a data leak in the model evaluation baseline. In `retail_customer_analysis.py`, `holdout_validation()` estimates the population-mean baseline using the mean of the future holdout outcomes it is supposed to predict. This makes that benchmark unavailable at prediction time and invalidates its comparison with other models.

The second major issue is duplicated analytical logic. `app.py` and `retail_customer_analysis.py` use different definitions of a customer purchase event and have separate product, basket, feature, and segmentation implementations. A purchase counted as one event in the UI may not be counted the same way in model training or validation.

Third, the repository lacks visible dedicated test files, continuous integration, and a `README.md`. Although the project has a built-in fixed-synthetic-data validation harness, it does not replace a regression suite covering point-in-time correctness, data edge cases, model behavior, and report generation.

## 1.2 Recommended direction

| Priority | Improvement                                                                               | Expected value                                                   |
| -------- | ----------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| P0       | Fix holdout leakage and formalize time-based evaluation                                   | Makes model comparisons trustworthy                              |
| P0       | Establish one canonical definition of transactions, trips, revenue, and customer identity | Prevents inconsistent customer metrics and basket results        |
| P0       | Add automated tests, CI, and a practical README                                           | Prevents regressions and makes the project reproducible          |
| P1       | Correct temporal return matching, ambiguous date handling, and currency presentation      | Improves accounting interpretation and operational trust         |
| P1       | Expand rolling-origin validation and probability/forecast diagnostics                     | Establishes whether the existing probabilistic models generalize |
| P1       | Consolidate reusable feature and metric definitions                                       | Makes training, scoring, and reporting consistent                |
| P2       | Add supervised purchase propensity and richer value modeling when justified by data       | Supports new decisions without unnecessary complexity            |

## 1.3 What to preserve

Retain the lightweight Python stack, the standalone analytical CLI, the existing BG/NBD and Gamma-Gamma implementation, the time-based holdout framework, RFM segmentation, stability-aware clustering, censoring-aware survival analysis, basket association testing, and artifact-based reporting.

These are meaningful strengths. In particular, the use of SciPy, NumPy, pandas, and explicit statistical calculations is aligned with a maintainable, resource-conscious analytics tool. More dependencies should be added only when a concrete analytical requirement justifies them.

## 1.4 Definition of success

The improved tool should be able to answer five questions reliably:

1. What happened? Reconciled, clearly defined, time-scoped customer and retail metrics.
2. Who behaves differently? Stable, interpretable segments and measurable behavioral differences.
3. What is likely to happen next? Forecasts validated on future periods, compared with leakage-free baselines.
4. What decisions might be worth testing? Identifiable customers or products, estimated opportunity, uncertainty, and explicit assumptions.
5. Can the results be trusted and reproduced? Point-in-time features, automated tests, recorded configuration, and documented data limitations.

# 2. Repository Audit

## 2.1 Verified repository structure

The inspected `main` branch has seven root files. No `README.md`, dedicated `tests/` directory, GitHub Actions workflow, or packaging configuration was present in the inspected repository tree.

| File                                                                                                                           | Verified responsibility                                                                                                                                                                           | Assessment                                                                                |
| ------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| [`app.py`](https://github.com/hrczggyrgy/retail_customer_analytics/blob/main/app.py)                                           | Approximately 5,365 lines; Streamlit interface, preprocessing, customer/product/basket features, metrics, anomaly analysis, chart helpers, action matrix, validation harness                      | Capable but monolithic, with duplicated analytics                                         |
| [`retail_customer_analysis.py`](https://github.com/hrczggyrgy/retail_customer_analytics/blob/main/retail_customer_analysis.py) | Approximately 1,401 lines; standalone CLI, data-quality checks, event construction, returns, probabilistic models, feature engineering, clustering, retention, basket statistics, exported tables | Strongest candidate for the canonical analytical layer                                    |
| [`insight_engine.py`](https://github.com/hrczggyrgy/retail_customer_analytics/blob/main/insight_engine.py)                     | Approximately 1,033 lines; reads analytical artifacts and generates charts, an HTML dashboard, a summary JSON, and skipped-chart diagnostics                                                      | Useful separation between analysis and reporting, but some charts are incomplete          |
| [`data_generator.py`](https://github.com/hrczggyrgy/retail_customer_analytics/blob/main/data_generator.py)                     | Approximately 1,177 lines; synthetic retail data, product catalog, simulated customer characteristics, returns, and generator validation                                                          | Valuable source of repeatable test data and simulated ground truth                        |
| [`show_all_results.py`](https://github.com/hrczggyrgy/retail_customer_analytics/blob/main/show_all_results.py)                 | Loads a presumed sample dataset, maps its schema, runs the pipeline, and prints generated results                                                                                                 | Useful exploratory script, but dependent on an untracked local data file                  |
| [`requirements.txt`](https://github.com/hrczggyrgy/retail_customer_analytics/blob/main/requirements.txt)                       | Exact pins for Streamlit, pandas, NumPy, SciPy, Matplotlib, NetworkX, and PyArrow                                                                                                                 | Good starting point for reproducibility; no separate development dependency configuration |
| [`.gitignore`](https://github.com/hrczggyrgy/retail_customer_analytics/blob/main/.gitignore)                                   | Excludes data, generated outputs, caches, environments, and build artifacts                                                                                                                       | Sensible, but explains why the default sample-file workflow is not self-contained         |

A repository-level README could not be found. Consequently, the audit can compare the implementation with its module docstrings and repository structure, but cannot verify claims that might have been made in a README.

## 2.2 How the components currently connect

Verified flow:

1. `app.py` accepts uploaded data or generates synthetic data.
2. `prepare_transaction_frame()` in `app.py` converts identifiers, dates, quantities, and prices, then computes revenue.
3. `run_pipeline()` writes the input to a temporary CSV and calls `retail_customer_analysis.main()` with arguments.
4. The CLI produces analytical tables, quality reports, model parameters, and run metadata.
5. `run_pipeline()` calls `insight_engine.main()` to generate the report artifacts.
6. The Streamlit interface reads result tables and images into its tabs.

The standalone CLI can also run independently of Streamlit. The reporting layer consumes saved outputs rather than being fully coupled to the model-fitting functions.

This separation is worth preserving. The principal architectural problem is that parts of the UI perform their own analytics rather than relying on a single authoritative implementation.

## 2.3 Existing strengths worth preserving

### A. A relatively comprehensive analytical pipeline

In `retail_customer_analysis.py`, the current implementation includes:

- Typed ingestion and data-quality issue counts.
- Identifier consistency and accounting checks.
- Purchase-event construction with configurable trip or transaction granularity.
- Return matching and return diagnostics.
- Rolling-window customer features and behavioral cadence statistics.
- BG/NBD expected-purchase modeling and a Gamma-Gamma expected-trip-value estimate.
- Time-based holdout validation and decile summaries.
- RFM segmentation and robust-scaled K-means with silhouette and bootstrap stability diagnostics.
- Kaplan–Meier time-to-second-purchase analysis and customer cohort retention.
- Basket affinity and next-trip transitions with multiple-testing correction and split-half persistence checks.
- CSV/Parquet tables, JSON metadata, and a Markdown analysis report.

Adding duplicate implementations of these capabilities would increase maintenance burden without necessarily improving analytical quality.

### B. Explicit statistical limitations

The CLI and reporting layer already distinguish several important concepts:

- Historical revenue versus predicted revenue.
- Observed inactivity versus confirmed churn.
- Price-based promotion proxies versus measured campaign exposure.
- Product associations versus causal effects.
- First-observed transaction history versus a customer's true lifetime history.

These distinctions should remain visible throughout the user interface and documentation.

### C. Computational safeguards

The CLI places limits on customer clustering, candidate affinity items, basket processing, and pair-operation budgets. This is a good pattern for a tool intended to run without specialized infrastructure.

### D. Reproducibility building blocks

The repository uses random seeds, pinned package versions, saved model parameters, a run metadata file, and a fixed synthetic-data validation harness. These should be expanded into a formal testing and artifact-management workflow.

## 2.4 Verified gaps and methodological risks

| Finding                                                        | Evidence                                                                                                                                    | Why it matters                                                                                                                                                                                                         | Priority |
| -------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- |
| Holdout baseline uses future outcomes                          | `retail_customer_analysis.py`, `holdout_validation()`, approximately lines 563–616                                                          | The population-mean baseline uses the average future trip count from the holdout itself. It is not a valid prediction-time baseline, and its comparison with fitted models is contaminated.                            | P0       |
| Different purchase-event definitions across components         | `app.py`, `build_customer_features()` around line 1060; `build_basket_associations()` around line 1455; CLI `build_trips()` around line 291 | The UI counts distinct invoice IDs for frequency; the CLI defaults to customer-day trips. Basket logic also differs. Segment definitions, customer frequency, and purchase likelihood can therefore vary by component. | P0       |
| Return matching is only date-granular                          | `retail_customer_analysis.py`, `match_returns()` around line 310                                                                            | Matching is performed on normalized `transaction_day`, even though raw timestamps are retained. A same-day return may be matched to a purchase later that day when timestamps are available.                           | P1       |
| No dedicated automated test suite is visible                   | Repository tree contains no `tests/` directory or CI workflow                                                                               | A fixed-data validation harness is useful but does not systematically exercise ingestion, edge cases, time leakage, numerical failures, or clean-checkout execution.                                                   | P0       |
| Two synthetic data implementations exist                       | `app.py`, `make_synthetic()` around line 2037; `data_generator.py`, `generate()` around line 228                                            | Different generators may produce different semantics or behaviors, weakening reproducibility and test coverage.                                                                                                        | P1       |
| App and CLI data-quality behavior differs                      | `app.py`, `prepare_transaction_frame()` around line 968 versus CLI `load_lines()` and `clean_lines()`                                       | The app drops rows with invalid date, quantity, or price without producing the same detailed issue counts as the CLI. Analysts may not see a consistent accounting of excluded rows.                                   | P1       |
| Currency is not part of the explicit data contract             | `data_generator.py` documents EUR; `app.py`, `fmt_currency()` around line 1575, formats values using `$`                                    | Demo values and displayed labels can disagree. More generally, monetary data cannot be interpreted reliably without an explicit currency assumption.                                                                   | P1       |
| The default synthetic horizon ends in the future               | `data_generator.py`, `generate()` default argument at line 231 uses `end="2026-12-31"`                                                      | On the audit date, October 10, 2026, the default generator produces dates after the current date. If treated as real chronological observations, these can make “current” results misleading.                          | P1       |
| Some advertised reporting capabilities are incomplete          | `insight_engine.py`, chart builders around lines 668–873                                                                                    | Revenue bridge, segment migration, department flows, and true cohort revenue/LTV require data tables not produced by the pipeline. The active cohort report includes a placeholder value panel.                        | P2       |
| Some chart names overstate what is calculated                  | `insight_engine.py`, `chart_promo_return_lollipop()` around line 842                                                                        | The active chart calculates return rate; it does not calculate discount share or margin risk. Labels should match the actual measurement.                                                                              | P1       |
| Configuration and metric definitions are not fully centralized | `app.py`, `APP_CONFIG` around line 71 and `METRIC_DEFINITIONS` around line 85                                                               | Both constants are defined but are not consumed elsewhere in the file, based on repository-wide reference searches. They do not yet guarantee consistent behavior or definitions.                                      | P1       |
| The example result script assumes a missing local file         | `show_all_results.py`, `main()` around line 13; sample files excluded in `.gitignore`                                                       | A fresh checkout cannot run this script as written unless the user supplies the expected local `sample_data.parquet`.                                                                                                  | P2       |
| Run metadata contains local filesystem paths                   | `retail_customer_analysis.py`, quality/metadata serialization around lines 1345–1381                                                        | Shared output artifacts can reveal local file paths or deployment layout. This should be configurable or redacted.                                                                                                     | P1       |

The latest commit, dated October 10, 2026, reports cleanup work, centralized constants, dependency pinning, and validation on fixed synthetic data. These are useful changes, but the repository still needs tests that independently verify those properties and protect them from future regressions.

## 2.5 Further issues to verify with representative data

The following are not claims that the tool is universally wrong. They are specific code paths whose consequences depend on input conventions.

1. Invoice identifier scope. In the app's product and basket builders, `transaction_id` is often treated as globally unique. If IDs are only unique within customer, register, channel, or store, order counts and associations may be wrong. The CLI has a transaction-key mode and structural checks; the app must apply compatible rules.
2. Product metadata conflicts. `app.py` builds a distinct product description/category/department mapping and merges it to product summaries. If one product ID has inconsistent metadata, the merge may return duplicate product rows. The CLI explicitly detects metadata conflicts, which should become a shared validation rule.
3. Date parsing. `retail_customer_analysis.py`, `parse_dates()` around line 196, attempts month/day/year formats before day/month/year formats. Ambiguous dates can be misinterpreted without a declared format. This should not be resolved by guessing.
4. Model convergence. `score_customers()` catches broad exceptions and falls back to empirical rates when BG/NBD fitting fails. The Gamma-Gamma fit records convergence status, but the downstream status/reporting should make non-convergence explicit instead of treating a numerical result as automatically acceptable.
5. Gamma-Gamma assumptions. The code already computes the relationship between transaction count and mean trip value and records an independence warning. If customer frequency and transaction value are strongly related, the value model's assumptions may not hold well enough for reliable individual ranking.
6. Anomaly baselines. The Streamlit app includes configurable anomaly calculations. Those flags should be tested under changes in seasonality, low-volume periods, and incomplete observation windows before being interpreted as true business anomalies.

## 2.6 Gap between current purpose and implementation

The current codebase is strongest at descriptive and probabilistic analysis of transaction-level retail data. It is not yet a general-purpose customer decisioning system with every possible retail capability.

The schema provides customer, transaction, product, category, price, quantity, and time information. It does not establish that the tool has:

- Customer acquisition or account-creation dates.
- Campaign assignment and exposure logs.
- Reliable promotion flags or a reference-price history.
- Product cost, gross margin, fulfillment costs, or contact costs.
- Inventory, stock availability, or out-of-stock status.
- A record of customer visits, website activity, advertising exposure, or service interactions.
- Experiment assignment or treatment outcomes.

Therefore, purchase propensity, true churn interventions, promotional response, margin-adjusted customer lifetime value, and channel-level journeys must be conditional capabilities rather than assumed features of the current product.

# 3. Customer Questions and Capability Map

The roadmap should follow decisions the business needs to make, not the number of algorithms available.

| Business question                                                      | Minimum viable data                                                                           | Current capability                                                                                                            | Recommended next step                                                                                                                                         | How usefulness should be measured                                                                                              |
| ---------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| Who are our highest-value customers today?                             | Customer ID, date, quantity, unit price, agreed event grain                                   | Historical RFM, customer value tiers, revenue concentration, expected future revenue                                          | Reconcile gross/net spend, stabilize event definitions, and show value uncertainty                                                                            | Metric reconciliation, segment stability, customer rank stability, and contribution by value tier                              |
| Which customer groups behave differently?                              | Repeated purchases over sufficient history                                                    | Rule-based RFM and stability-checked K-means                                                                                  | Compare the clusters against RFM rules and a simpler behavioral segmentation; retain clusters only if they add decision value                                 | Bootstrap and temporal stability, cluster size, meaningful profiles, and different subsequent behavior                         |
| Which known customers are likely to buy again in the next period?      | Repeat purchases, known customers, temporal cutoff, sufficient history                        | BG/NBD expectations, `P(alive)`, cadence heuristic, time holdout                                                              | Fix the leaked benchmark and use rolling temporal holdouts. Add a purchase-propensity classifier only if a calibrated probability of any purchase is required | MAE/RMSE for trips, ranking quality, calibration, top-k lift, and aggregate forecast bias                                      |
| When do new customers make their second purchase?                      | Reliable observed first purchase and follow-up observation window                             | Kaplan–Meier time-to-second-trip analysis                                                                                     | Improve acquisition-cohort definition where signup data exists; show uncertainty and censoring                                                                | Calibration of repeat rates at fixed horizons, confidence intervals, and cohort comparisons                                    |
| Which customers are becoming inactive?                                 | Customer purchase histories and a business-defined inactivity horizon                         | Cadence-based heuristic and `P(alive)`                                                                                        | Distinguish a risk score from a defined churn event. Validate inactivity thresholds by category and customer cadence                                          | Future repurchase within an explicit horizon, recall at a contact budget, and calibration                                      |
| How much revenue might customers generate over a forecast horizon?     | Purchase history, prices, and a meaningful horizon                                            | BG/NBD × Gamma-Gamma predicted revenue                                                                                        | Validate across several as-of dates; rename it predicted revenue, not profit or proven lifetime value                                                         | Revenue MAE/RMSE/WAPE, aggregate bias, rank stability, and forecast calibration                                                |
| Which products or categories are bought together?                      | Transaction ID, customer ID if customer-level inference is desired, product/category identity | Association rules with FDR correction and persistence checks in the CLI                                                       | Standardize basket definition and report counts, uncertainty, and time-split persistence                                                                      | Out-of-time repeatability, lift at a minimum support, confidence intervals, and future-basket retrieval quality                |
| What do customers buy on subsequent trips?                             | Ordered purchase events                                                                       | Next-trip transitions based on customer-day baskets                                                                           | Make the event grain explicit; if timestamps are available, assess whether within-day order is genuinely needed                                               | Transition support, out-of-time transition stability, and observed next-trip rates                                             |
| Which products have repeat buyers or unusually high returns?           | Customer/product/date, quantity, price, reliable product identity                             | Product summaries, repeat-buyer metrics, return matching                                                                      | Validate return attribution and report denominators and uncertainty for small products                                                                        | Repeat-rate calibration against later periods, return-value and return-unit rates, matching coverage                           |
| Are customers responding to discounts?                                 | Actual discount, regular price, campaign/exposure, purchase opportunity; ideally margin       | Price-index proxy based on observed product prices                                                                            | First treat the proxy as descriptive. Add a genuine discount analysis only when promotional exposure is available                                             | Predictive holdout performance; incremental margin or conversion effect only from an appropriate experimental or causal design |
| Which customers should receive a retention or cross-sell intervention? | Predicted behavior, feasible action, contact costs, and eventually intervention outcomes      | Rule-based action matrix and affected-entity exports                                                                          | Use existing scores to prioritize experiments, not claim that the recommended action will cause an uplift                                                     | Prospective randomized test: incremental conversion, margin, customer retention, and cost per incremental outcome              |
| Where does revenue change come from?                                   | Reconciled periods and metrics                                                                | Current-period comparisons, trends, and a Streamlit waterfall helper; standalone revenue bridge lacks a required source table | Make the revenue bridge consistent across the UI and static report, including volume/order/AOV decomposition only where definitions support it                | Bridge reconciles exactly to total revenue change and remains stable under filter changes                                      |

### Priority implications

The first wave should improve customer value, repeat purchasing, retention analysis, and product affinity because these capabilities already exist and can benefit immediately from methodological fixes.

Supervised propensity, campaign-response modeling, and intervention optimization should follow only when their labels and business decision requirements are clearly defined.

# 4. Feature Engineering Plan

## 4.1 Establish a canonical point-in-time feature contract

The highest-value feature-engineering change is to define one reusable function that builds features for a particular customer snapshot.

A feature record should be keyed by:

- `customer_id`
- `as_of_date` or `snapshot_date`
- `event_grain`
- `lookback_window` where relevant

Every feature must use data available at or before that snapshot date. Training, validation, and future scoring should call the same feature-building implementation.

A generic contract would look like:

```
features = build_customer_snapshot(    transactions=transactions,    as_of_date=cutoff,    event_grain="customer_day",    windows_days=(30, 90, 365),)
```

The exact API is a recommendation, not an existing function signature. Internally, the event builder should define whether a purchase event means an invoice, customer-day trip, or another business-approved unit. The current CLI's default customer-day grain may be reasonable for some retail settings, but it should not silently become the universal definition.

## 4.2 Feature definitions and priorities

The following table separates features already present from recommended extensions.

| Feature family                               | Definition and inputs                                                                                                                                                                                | Aggregation and window                                             | Business interpretation                                                        | Leakage and implementation safeguards                                                                                                                                                                                 | Priority                       |
| -------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------ |
| Recency — already present                    | \\(R_c = t\_{\text{as-of}} - t\_{\text{last purchase}}\\), in days                                                                                                                                   | Customer; typically all available history up to snapshot           | How recently a customer purchased                                              | Last purchase must be on or before the snapshot. Specify how customer records with only anonymous transactions are handled.                                                                                           | P0: standardize                |
| Frequency — already present                  | \\(F_c(W)=\\#\\{\text{distinct qualifying events in }[t-W,t]\\}\\)                                                                                                                                   | Customer; 30/90/365-day windows and lifetime-to-date               | Recent activity and repeat behavior                                            | Use the same event key in the UI, CLI, and evaluation. Do not substitute invoice count for customer-day trips without labeling the change.                                                                            | P0: standardize                |
| Monetary value — already present             | Gross purchase revenue \\(G_c(W)=\sum\_{i \in W, q_i>0} q_i p_i\\); return value \\(U_c(W)=\sum\_{i \in W,q_i<0} -q_i p_i\\); net sales \\(N_c(W)=G_c(W)-U_c(W)\\)                                   | Customer; selected lookback and cumulative history                 | Spending, purchase scale, and returned value                                   | Keep financial totals exact; do not winsorize revenue before accounting. Report whether metrics are gross or net. Prevent unmatched refunds from being silently interpreted as fully attributed to a purchase.        | P0: standardize                |
| Average basket/trip value — already present  | Mean or median trip value \\(=\sum \text{trip value}/\\#\text{trips}\\)                                                                                                                              | Customer; history and rolling windows                              | Typical spend per purchase event                                               | Use the chosen event grain and comparable windows. Historical average is not the expected future value until validated.                                                                                               | P1                             |
| Rolling activity — already present           | Revenue, returns, net revenue, and event counts for \\(W\in\\{30,90,365\\}\\) days                                                                                                                   | Customer and snapshot                                              | Recent purchasing intensity and changes in activity                            | Calculate from transactions at or before each snapshot. In historical validation, recompute from that snapshot's data, not the current full history.                                                                  | P0: standardize                |
| Recent-versus-prior change — already present | \\(\Delta_W = (x\_{\text{recent}}-x\_{\text{prior}})/x\_{\text{prior}}\\) when prior value \\(>0\\); also retain absolute change                                                                     | Customer; adjacent equal-length windows                            | Momentum, reactivation, and activity decline                                   | Percentage changes are undefined for zero prior values. Keep missingness or use a separate new-activity flag; do not assign an arbitrary zero growth rate.                                                            | P1                             |
| Trend and variability — extension            | For monthly values \\(y_m\\), estimate a trend slope on \\(\log(1+y_m)\\); also calculate inter-month standard deviation, coefficient of variation when mean \\(>0\\), and fraction of active months | Customer; trailing 6–12 complete months when enough history exists | Separates regular shoppers, seasonal shoppers, and volatile one-off purchasers | Use only complete prior periods and compute all parameters from the training window. Short histories should fall back to simpler features.                                                                            | P1                             |
| Purchase cadence — partly present            | Mean, median, and standard deviation of interpurchase gaps; \\(CV = \sigma\_{\text{gap}}/\mu\_{\text{gap}}\\) when mean gap \\(>0\\); optionally time since expected next trip for regular shoppers  | Customer; ordered event dates                                      | Identifies typical repeat interval and unusual delay                           | A cadence estimate from one or two intervals is noisy. Shrink or suppress it for low-frequency customers. The CLI already has gap statistics and regularity features.                                                 | P1: improve validation         |
| Basket composition — partial                 | Mean distinct products and units per event; category count; category spend share \\(s\_{c,k}=R\_{c,k}/\sum_j R\_{c,j}\\); category entropy \\(H_c=-\sum_k s\_{c,k}\log(s\_{c,k})\\)                  | Customer; rolling window and cumulative history                    | Basket size, breadth, and concentration in a small number of categories        | Deduplicate product IDs within a basket for presence-based calculations. Revenue shares are only defined when total spend is positive. Control feature dimensionality for sparse categories.                          | P1                             |
| Category and product affinity — partial      | Per-customer category spend share and frequency; category repeat rate; recency of last category purchase; pair indicators where sample size allows                                                   | Customer/category and customer/product; 90/365-day windows         | Supports descriptive cross-sell analysis and customer profiles                 | Use only known historical purchases at the scoring cutoff. Distinguish stable preferences from a small number of incidental purchases.                                                                                | P1                             |
| Discount/price behavior — proxy present      | Current proxy uses `price_index = unit_price / product_median_unit_price`; proxy spend share is spend assigned to lines below a configured relative-price threshold                                  | Customer/product; only history preceding the snapshot              | Describes spending at unusually low observed prices                            | This is not confirmed promotion exposure or a causal discount response. Recompute reference prices at each historical snapshot; do not calculate an old training feature using future product prices.                 | P1: correct scope and labeling |
| Returns — partly present                     | Return value ratio \\(=U_c/G_c\\) when \\(G_c>0\\); return-unit ratio; unmatched return share; median observed days to return                                                                        | Customer and product; trailing and cumulative windows              | Helps identify return behavior and product-quality patterns                    | Use explicit purchase/return rules, including partial returns. Link using full timestamps and transaction identity when available. Never let return lines increase purchase frequency.                                | P1                             |
| Customer lifecycle — partly present          | Observed tenure \\(=t\_{\text{last}}-t\_{\text{first observed}}\\); active-window flags; time since last purchase; first-observed cohort                                                             | Customer at snapshot                                               | Describes observed lifecycle stage                                             | First observed purchase is not necessarily acquisition. If acquisition data becomes available, use actual acquisition dates and preserve a separate first-observed date.                                              | P1                             |
| Customer future value — model-derived        | Existing estimate: expected trips over horizon × expected trip value                                                                                                                                 | Customer at snapshot; fixed forecast horizon                       | Expected gross purchase revenue over the selected horizon                      | Do not use model outputs as training features when fitting those same outputs without out-of-fold or prior-snapshot construction. Label forecasts as revenue, not profit or true lifetime value.                      | P0: validate current output    |
| Segment/profile features — partial           | Existing RFM scores, robust-scaled behavioral model features, category-mix components                                                                                                                | Customer at snapshot                                               | Communicates customer behavior compactly                                       | Estimate transformations and any learned cutoffs on training data. Reuse saved transformations at validation/scoring time rather than learning them from the future scoring population if stable scores are required. | P1                             |

## 4.3 Improve preprocessing without distorting the economics

### Missing values

Use different policies by feature meaning:

- Missing customer IDs should exclude rows from customer-level model features while preserving them in aggregate sales and data-quality reporting where possible.
- Missing category or product description should remain distinguishable from an actual category such as `"Unknown"` unless that mapping is explicit.
- Zero purchase counts in a valid time window are real behavioral observations, not missing data.
- Missing price, quantity, and date should create a recorded data-quality issue and a documented exclusion or correction path.
- A missing model output caused by fitting failure must not be filled in silently and presented as a valid forecast.

### Outliers

- Use `log1p` for highly skewed positive spend, counts, and predicted values when appropriate for a model.
- Use robust scaling for clustering, as the CLI already does.
- Use robust descriptive statistics and quantiles for business reporting.
- Apply winsorization only to model inputs when it is justified and learned within the training period. Do not clip financial aggregates that are supposed to reconcile to source data.
- Validate extreme quantity, price, and revenue values against source rules rather than automatically assuming they are errors.

### Encoding and dimensionality

The current CLI compresses category-mix features before clustering and caps the number of categories. Preserve this lightweight strategy initially. Add only features that can be computed consistently at a snapshot, have clear business interpretation, and improve either forecast performance or segment usefulness.

For the future propensity model, a small set of scaled numerical features plus controlled categorical encoding is a strong starting point. Avoid a high-dimensional customer-by-product matrix unless the data is large enough and a defined recommendation problem justifies it.

## 4.4 Make the feature pipeline reproducible

The central feature-building contract should own:

1. Event-grain rules.
2. Point-in-time filters.
3. Lookback boundaries.
4. Financial sign conventions.
5. Missing-data treatment.
6. Feature names and definitions.
7. Training-fitted transformations and segmentation cutoffs.
8. Feature schema/version metadata.

A key acceptance test is snapshot parity: building the same customer snapshot from identical input data and configuration in training, holdout scoring, and batch scoring should produce identical feature values and dtypes.

# 5. Algorithm and Baseline Recommendations

## 5.1 Review of existing algorithms

| Existing method                                 | Keep or change?                      | Reason                                                                                                                                                                   |
| ----------------------------------------------- | ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| RFM scoring and rule-based segments             | Keep; standardize                    | Strong interpretable baseline for communication and customer prioritization. Ensure one definition across the app and CLI.                                               |
| BG/NBD                                          | Keep; improve evidence of validity   | Fits a recurring-purchase setting and is computationally lighter than many alternatives. Its predictions require temporal holdout evidence and appropriate assumptions.  |
| Gamma-Gamma                                     | Keep conditionally                   | Provides a practical expected transaction-value estimate, but its assumptions and optimizer convergence should be checked before relying on individual revenue rankings. |
| Empirical purchase-rate baseline                | Keep                                 | Simple and informative. It is an important comparator for BG/NBD.                                                                                                        |
| Population-mean trip baseline                   | Keep after fixing leakage            | A constant baseline is useful only if its mean is estimated from training/calibration data, not the holdout's future outcomes.                                           |
| Robust-scaled K-means                           | Keep; strengthen acceptance criteria | Lightweight, with existing silhouette and bootstrap stability diagnostics. The cluster solution should still be compared with simpler RFM segmentation.                  |
| Kaplan–Meier repeat-purchase analysis           | Keep                                 | Appropriate for time to second purchase when censoring is represented and the event/censor definitions are valid.                                                        |
| Basket affinity with FDR and persistence checks | Keep; harmonize event grain          | Stronger statistical safeguards than raw lift sorting alone. Clarify whether the inference unit is a customer or a basket.                                               |
| Rule-based action matrix                        | Keep as prioritization logic         | Useful for communicating hypotheses and affected customers, but it does not estimate intervention impact.                                                                |

## 5.2 Essential additions

### A. Fix the baseline before adding a new predictive model

Problem: The current `pred_constant` in `holdout_validation()` depends on actual trip counts in the future evaluation window.

Recommended change: Calculate the population-level constant using only calibration data. For example, derive the mean number of repeat trips expected over a comparable horizon from the pre-cutoff customer population and historical windows. Preserve the existing customer-specific empirical-rate baseline as a second comparator.

Why: The baseline is the reference point for deciding whether additional modeling is useful. If the benchmark is contaminated, the model-selection process is not trustworthy.

Acceptance evidence: The baseline prediction must be identical when the future holdout labels are permuted or replaced while leaving calibration data unchanged.

### B. Add supervised purchase propensity only if the business needs a probability of future purchase

BG/NBD `P(alive)` and expected trip counts are not equivalent to a supervised estimate of the probability that a given customer purchases at least once during a particular future horizon.

If the business question is specifically:

> Among customers known by the cutoff date, what is the probability of at least one qualifying purchase in the next 30, 60, or 90 days?

then a supervised baseline is justified.

Recommended first model: Regularized logistic regression.

- Label: `1` if the customer makes at least one qualifying purchase during the subsequent horizon, otherwise `0`.
- Inputs: Recency, rolling trip counts, recent-versus-prior trend, historical spend, cadence statistics, breadth, and a limited number of categorical features.
- Training: Construct training examples at historical snapshot dates. Generate each feature from information available by that snapshot.
- Evaluation: Rolling-origin validation, PR-AUC, ROC-AUC, Brier score, log loss, calibration, and top-k lift.
- Interpretability: Examine coefficients and the effects of standardized predictors, keeping correlated features in mind.
- Cost: Lightweight fitting, but requires a careful supervised dataset and additional dependency/testing work if implemented with scikit-learn.

This should be a separate model with a precise prediction target, not an attempt to relabel `P(alive)` as purchase probability.

### C. Make the current probabilistic models numerically auditable

In `score_customers()` and the fitting functions:

- Record optimizer convergence and failure reasons distinctly.
- Flag non-finite parameters or predictions.
- Define an explicit accepted-fit status.
- Compare fitted parameters across temporal snapshots.
- Preserve a baseline fallback, but flag its predictions as fallback estimates.
- Report how many customers received each type of estimate.

Do not treat “the function returned numbers” as equivalent to “the model fit passed validation.”

## 5.3 High-value enhancements

### A. Improve segmentation validation

The current custom K-means selects candidate cluster counts using silhouette, bootstrap ARI, and cluster-size constraints. Keep this framework but add:

- A comparison against the rule-based RFM groups.
- Segment profiles on features not used in clustering, where appropriate.
- Segment stability across time snapshots.
- A “do not segment” result if all candidate solutions are weak.
- A check that labels such as “Active / higher expected value” match the underlying evidence and are not interpreted as causal categories.

Use clusters only if they are distinguishable, stable enough for the intended use, and useful for organizing business decisions.

### B. Improve future-trip and revenue baselines

For expected trip counts, compare BG/NBD with:

1. Customer historical empirical rate.
2. A training-only population-average rate.
3. A recent-window rate, if recent history contains enough observations.
4. A seasonal naive baseline, but only when sufficient historical coverage exists to support it.

For revenue, compare the existing BG/NBD × Gamma-Gamma estimate with a baseline derived from purchase rate and historical value. Use comparable event and revenue definitions for both predictions and actual outcomes.

### C. Extend retention analysis only when its target is clear

The current Kaplan–Meier calculation focuses on time to second trip. That is a different target from time to next purchase for an established customer and from true churn.

If the tool needs a repeat-purchase propensity model with covariate effects, a simple discrete-time purchase-hazard model is a reasonable next candidate. It can be more directly interpretable than a complex survival model while supporting time-varying behavioral features.

Start only after the event, censoring, and customer eligibility definitions have been specified.

### D. Add genuine customer lifetime value only with financial inputs

The current output is an estimate of future gross purchase revenue over a specified horizon. It is not margin-adjusted, discounted lifetime value.

If cost and margin fields become available, a future model could estimate expected net contribution:

\\[ \operatorname{ExpectedContribution}\_c(H) = \sum\_{t=1}^{H} \mathbb{E}[\text{contribution}\_{c,t}\mid\mathcal{H}\_c] \\]

This requires a defined financial treatment of returned goods, discounts, costs, and potentially contact or fulfillment expenses. Do not infer margin from selling price.

## 5.4 Optional research or advanced extensions

| Extension                                      | Prerequisite                                                                               | Decision rule for adoption                                                                                                        |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------- |
| Gradient-boosted trees for purchase propensity | Sufficient supervised examples, stable label definition, baseline model in place           | Adopt only when temporal holdout gains are material and calibration, maintenance, and explanation requirements remain acceptable  |
| More flexible spend model                      | Persistent Gamma-Gamma failure or poor revenue calibration                                 | Adopt only when it improves out-of-time revenue forecasts and the gain cannot be achieved through better segmentation or features |
| Product recommendation beyond pair affinity    | Reliable transaction/product identity and a defined recommendation objective               | Validate retrieval/ranking against future purchases, not training co-occurrence alone                                             |
| Uplift or treatment-effect modeling            | Randomized treatment/exposure and outcome information                                      | Require prospective experimental evidence and treatment-specific validation                                                       |
| Aggregate time-series forecasting              | Sufficient historical periods, known reporting frequency, and clear aggregate planning use | Compare with seasonal-naive and simple trend baselines before adding a complex model                                              |

Do not add transformer-based sequential models, large embedding pipelines, or heavy recommendation infrastructure at this stage. The repository's most important weaknesses are data semantics, evaluation validity, and reproducibility—not an evident shortage of model capacity.

# 6. Evaluation and Metrics

## 6.1 First fix the verified evaluation leak

In `retail_customer_analysis.py`, `holdout_validation()` includes the calculation:

```
d["pred_constant"] = float(d["actual_trips"].mean())
```

Here, `actual_trips` is populated from the holdout period. Therefore the purported population-mean prediction uses information from the target window.

This must be changed to a prediction estimated solely from calibration/training data, ideally over comparable historical windows.

This does not establish that BG/NBD itself is leaking future labels into its parameter fit. It establishes that the existing population-mean baseline comparison is invalid until corrected.

## 6.2 Establish a common evaluation protocol

For the existing probabilistic purchase model and any future supervised model:

1. Define an as-of date \\(t\\), lookback window, customer eligibility rule, and forecast horizon \\(H\\).
2. Build training features only from transactions dated at or before \\(t\\).
3. Fit all model parameters and feature transformations on that training history.
4. Evaluate using transactions in \\((t,t+H]\\).
5. Repeat at multiple historical cutoff dates where the data supports it.
6. Keep a final future period or final rolling origin untouched until model selection is complete.
7. Report sample sizes, eligibility exclusions, model failures, and the exact configuration used.

Do not use random transaction-row splits for these customer forecasts. They allow the same customer's past and future events to be mixed across train and test. A time-ordered evaluation is more representative of the deployment setting. See the [scikit-learn time-series cross-validation documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) for the general principle of preserving temporal order.

A standard time-series splitter may need adaptation for this repository's fixed-horizon customer snapshots. The important requirement is point-in-time feature construction and future-only evaluation.

## 6.3 Metrics by analytical task

### Purchase counts and repeat-purchase forecasting

| Metric                                    | Definition                                                                           | What it tells you                                            | Potential pitfall                                                      |
| ----------------------------------------- | ------------------------------------------------------------------------------------ | ------------------------------------------------------------ | ---------------------------------------------------------------------- |
| Mean absolute error (MAE)                 | \\(\frac{1}{n}\sum_i\|\hat y_i-y_i\|\\)                                              | Typical customer-level count error                           | May hide large errors for a small group of very frequent customers     |
| Root mean squared error (RMSE)            | \\(\sqrt{\frac{1}{n}\sum_i(\hat y_i-y_i)^2}\\)                                       | Penalizes large customer-level errors more strongly          | Dominated by extreme purchase counts                                   |
| Spearman correlation                      | Rank correlation between predicted and observed outcomes                             | Whether the customer ordering is informative                 | High rank correlation can coexist with inaccurate total forecasts      |
| Aggregate forecast bias                   | \\(\frac{\sum_i\hat y_i-\sum_i y_i}{\sum_i y_i}\\), when the denominator is positive | Whether predicted total trips are systematically high or low | Can hide offsetting errors across segments                             |
| Weighted absolute percentage error (WAPE) | \\(\frac{\sum_i\|\hat y_i-y_i\|}{\sum_i\|y_i\|}\\), when the denominator is positive | Scaled aggregate absolute error                              | Undefined or unstable when total actual activity is zero or very small |

Avoid using percentage-error metrics that divide by individual customer outcomes when many customers have zero actual purchases.

### Binary purchase propensity, if added

| Metric                          | Interpretation                                                              | Pitfall                                                                                       |
| ------------------------------- | --------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| ROC-AUC                         | How well the score ranks purchasers above non-purchasers                    | Does not establish calibrated probabilities and can look strong with rare positives           |
| PR-AUC / average precision      | Ranking performance focused on the positive class                           | Its baseline is related to prevalence; it must be interpreted against the test-set prevalence |
| Brier score                     | Mean squared error of predicted probabilities                               | Combines calibration and discrimination; the raw value depends on event prevalence            |
| Log loss                        | Penalizes inaccurate probabilities, especially confident errors             | Can be dominated by a few extremely wrong predictions                                         |
| Calibration curve               | Compares mean predicted probability with observed event rate across bins    | Binned curves can hide local miscalibration and are noisy with small groups                   |
| Precision/recall at top \\(k\\) | Measures how many positive customers are found within a fixed action budget | Useful only when the action budget is defined and test-period capacity is comparable          |
| Top-\\(k\\) lift                | Purchase rate in the top-ranked group divided by the overall purchase rate  | Unstable with small groups and does not show incremental impact from a campaign               |

The existing `P(alive)` should not be assessed as though it were a calibrated probability of purchasing in the next 90 days. It represents a model-derived probability associated with ongoing purchasing behavior; its relationship to the next-horizon purchase event must be established separately.

For general probability-calibration principles, see the [scikit-learn calibration guide](https://scikit-learn.org/stable/modules/calibration.html).

### Revenue forecasting

Use:

- MAE and RMSE for customer-level revenue error.
- WAPE and aggregate bias for total revenue forecasts.
- Spearman correlation for customer ranking.
- Forecast-versus-actual plots across deciles and rolling origins.
- Error breakdowns by prior purchase frequency, recency, segment, and cohort.

Revenue evaluation must use the same scope for prediction and actuals. For example, if the model predicts gross revenue from positive purchases, do not score it against net revenue after returns without explicitly changing the prediction target.

Do not use MAPE as the primary metric when actual customer revenue may be zero.

### Retention and time to repeat purchase

For the existing Kaplan–Meier analysis:

- Report survival and repeat-purchase probabilities at stated horizons.
- Keep confidence intervals and the number of customers still at risk.
- Exclude incomplete cohorts from fixed-age retention comparisons.
- Separate customers whose first purchase is actually known from customers whose first purchase is merely the first one observed in the file.

If a future model predicts time to next purchase, evaluate ranking and horizon-specific calibration with appropriate treatment of right-censoring. Do not compare a censored customer's unobserved repeat time as though it were a confirmed non-event forever.

### Customer segmentation

Continue tracking the repository's existing silhouette, cluster size, and bootstrap ARI, then add:

- Stability across data snapshots.
- Minimum useful segment size.
- Feature profiles and separation from other segments.
- Changes in assigned segment over time.
- Whether segments support distinct actions or reveal behavior not already captured by RFM.

Silhouette is not a business-value metric. A highly separated cluster solution may still be commercially useless.

### Basket associations and transitions

The current pipeline's FDR-adjusted tests and split-half persistence are useful. Extend validation with:

- Number of customers and baskets supporting every reported rule.
- Odds ratio or lift with uncertainty bounds.
- Rule persistence on later time windows.
- Duplicate-item and repeated-line invariants.
- Held-out next-basket or next-trip occurrence rates where the use case requires predictive value.

Lift alone can exaggerate low-support relationships. Statistical significance does not establish commercial value or a causal cross-sell effect.

## 6.4 Robustness, uncertainty, and reproducibility

For the most important metrics:

- Compute customer-level bootstrap intervals for differences between models.
- For repeated temporal evaluation, summarize performance across distinct future windows rather than treating the entire customer population as independent evidence of temporal stability.
- Record row counts, eligible customers, event grain, cutoff dates, horizon, random seed, dependency versions, feature version, and optimizer status.
- Evaluate performance by cohort, recency, frequency, and spend groups.
- Test sensitivity to the purchase-event definition and inactivity threshold.
- Ensure failures and fallbacks are visible, not silently mixed with fitted-model predictions.

## 6.5 Offline prediction versus real business impact

Offline prediction demonstrates how well a model forecasts observed behavior under the test design. It does not establish how much additional revenue or retention a campaign will cause.

For campaign decisions, compare treatments with a randomized holdout group where feasible. Measure incremental conversion, incremental contribution margin, contact cost, and retention outcomes. The existing action matrix can supply experiment candidates; it should not be described as proof that an action will improve a customer's behavior.

# 7. Visualization and Dashboard Plan

The project already has useful business and diagnostic visualizations. The aim should be to fill meaningful gaps, correct misleading labels, and connect charts to validated metrics.

## 7.1 Business-facing views

| View                          | Question                                                                      | Data and chart design                                                                                                                         | Useful decision                                                   | Interpretation risks                                                                                                                     |
| ----------------------------- | ----------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| Executive performance trend   | What is changing in retail performance?                                       | Monthly line chart of gross purchase revenue, return value, net revenue, orders, and active customers; filter by date/category/department     | Determine whether a change is broad-based or concentrated         | Net and gross revenue must follow consistent definitions; partial periods distort comparisons                                            |
| Revenue change bridge         | Which components explain the change between comparable periods?               | Waterfall separating order volume, basket value, and category/product contributions; clearly label accounting-based versus modeled components | Focus investigation on the largest contributors                   | The standalone `insight_engine.py` revenue-bridge builder lacks a required source table. Do not produce a bridge that fails to reconcile |
| Customer value distribution   | How concentrated is observed and forecast value?                              | Existing Lorenz/concentration curves plus percentile tables; show historical spend and future-horizon revenue separately                      | Identify customer concentration and value tiers                   | Historical revenue and predicted value are different measures; revenue is not profit                                                     |
| Customer behavior map         | Which customers have high predicted activity and high expected basket value?  | Existing scatter of expected trips versus expected trip value, with `P(alive)` distribution and segment filter                                | Prioritize analysis of different behavioral groups                | Avoid treating the `P(alive)` threshold as a calibrated purchase probability or as confirmed churn                                       |
| RFM and cluster profiles      | How do segments differ?                                                       | Segment-size bars and profile table with median recency, frequency, spend, return ratio, and forecast metrics                                 | Decide whether simple RFM or clustering is enough                 | Segment labels may shift between snapshots; report that labels are descriptive                                                           |
| Cohort retention              | Which acquisition/first-observed cohorts return?                              | Heatmap of retention by cohort and complete cohort age; include cohort size and uncertainty                                                   | Compare cohort behavior at equivalent ages                        | Current first-observed cohorts are imperfect proxies for acquisition cohorts                                                             |
| Purchase cadence              | How long do customers typically wait before buying again?                     | Distribution of interpurchase gaps and repeat-purchase survival curve with confidence interval and number at risk                             | Set hypotheses for reminders and observation horizons             | A single purchase or a short history cannot provide a stable individual cadence estimate                                                 |
| Product/category performance  | Which products are growing, declining, or being returned?                     | Ranked table with revenue, units, orders, customer counts, repeat-buyer rate, and return rates; trend chart for material products             | Identify products for supply, range, and quality investigation    | Revenue ranking can mislead where availability, seasonality, or margin differ                                                            |
| Basket and next-trip affinity | Which associations are frequent and sufficiently robust to examine further?   | Current association table/network, filtered by support, confidence, lift, FDR, and persistence; show counts beside effect size                | Generate candidate cross-sell hypotheses                          | An observed association may reflect popularity or shopping mission, not a causal effect                                                  |
| Return behavior               | Where are returns concentrated?                                               | Return-value and return-unit rates by category/product, with denominators and confidence intervals for small samples                          | Identify products or categories for further quality investigation | A return ratio is not automatically a quality metric; return windows and matching coverage matter                                        |
| Action matrix                 | Which customers/products are candidates for investigation or experimentation? | Current affected-entity table with rule, priority, impact type, entity count, and downloadable CSV                                            | Build a transparent workflow for testing actions                  | A deterministic prioritization rule does not estimate treatment effect                                                                   |

## 7.2 Diagnostic and trust views

| View                                   | Required output                                                                                                      | Purpose                                                                    |
| -------------------------------------- | -------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| Data-quality overview                  | Missing/malformed dates, missing IDs, bad quantities/prices, duplicate lines, identifier conflicts, exclusion counts | Shows how much data is usable and why records were excluded                |
| Accounting reconciliation              | Gross revenue, return value, net revenue, and roll-up differences                                                    | Verifies source totals and derived summaries remain aligned                |
| Forecast comparison                    | MAE/RMSE/WAPE, aggregate bias, sample count, and paired difference from each baseline                                | Shows whether modeling improves forecasts                                  |
| Probability calibration                | Observed vs predicted probability by bin, Brier/log loss for supervised propensity, and calibration sample counts    | Detects over- or under-confident scores                                    |
| Forecast residuals                     | Predicted vs actual trips/revenue and signed errors by forecast horizon and customer segment                         | Reveals systematic errors hidden by one headline metric                    |
| Rolling-origin performance             | Metric distributions or trend lines across historical cutoff dates                                                   | Shows whether performance is stable over time                              |
| Segmentation diagnostics               | Silhouette, ARI, cluster sizes, profiles, and temporal stability                                                     | Prevents overconfidence in an unstable clustering solution                 |
| Model status and optimizer diagnostics | Fit status, converged/non-converged parameters, fallback count, and applicable warnings                              | Prevents fallback forecasts from being confused with accepted model output |
| Basket inference diagnostics           | Candidate count, tests run, FDR-adjusted findings, support, confidence intervals, and temporal persistence           | Helps analysts distinguish robust findings from low-support patterns       |
| Run configuration and provenance       | Cutoff, horizon, event grain, thresholds, software versions, seed, feature schema version                            | Makes exported results interpretable and reproducible                      |

## 7.3 Specific fixes to the existing reporting layer

1. Cohort revenue panel: The active `chart_cohort_retention_revenue()` in `insight_engine.py` currently produces a retention heatmap and a placeholder panel for revenue per customer. Implement the revenue table and its uncertainty before presenting it as a combined cohort-LTV chart; otherwise display retention only.
2. Revenue bridge: The standalone `chart_revenue_bridge()` currently skips because its required `revenue_bridge` table is not produced. Either build a shared, reconciled table or clearly distinguish this from the separate waterfall helper in `app.py`.
3. Promotion/return chart: `chart_promo_return_lollipop()` currently uses category return rate only. Rename and describe it as a return-rate chart unless real discount-share data is added.
4. Segment migration: The current segment-migration builder requires `segment_transitions`, which is not produced by the pipeline. Add it only when meaningful, comparable historical snapshots and consistent segment definitions are available.
5. Time-of-day charts: `chart_dow_hour_heatmap()` is unavailable from the current normalized daily event representation. Retain the feature only if exact timestamps are available and within-day shopping-time analysis has a real use case.
6. Action quadrant: Label the axes as current model outputs and observed estimates; do not imply that high forecast value proves high profitability or that an inactive status establishes churn.

A chart should exist because the underlying measure is defined, tested, and decision-relevant—not because a chart-builder function has already been written.

# 8. Architecture and Engineering Improvements

## 8.1 Move toward one analytical implementation

The first refactor should be incremental. Do not rewrite the 250 KB Streamlit file or the 86 KB CLI module in one pass.

Recommended target structure:

```
retail_customer_analytics/
├── app.py                         # Thin Streamlit interface
├── data_generator.py              # Canonical synthetic-data generator
├── retail_customer_analysis.py    # Temporary CLI compatibility wrapper
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

This is a proposed destination, not a description of files that currently exist.

### Extraction order

1. Data schema and normalization.
2. Purchase events and financial sign conventions.
3. Shared customer/product/category feature builders.
4. Model fitting and evaluation.
5. Reporting and chart inputs.
6. Thin UI adapters.

The CLI should continue to work while its implementation moves behind reusable modules. The Streamlit interface should consume shared outputs rather than independently recomputing business metrics.

## 8.2 Formalize input schema and business rules

Create an explicit schema contract for the existing nine expected columns:

`customer_id`, `transaction_date`, `transaction_id`, `product_id`, `product_description`, `department`, `category`, `price`, `quantity`.

Document:

- Required versus optional columns.
- Identifier uniqueness and scope.
- Date formats and timezone assumptions.
- Currency and price semantics.
- Quantity sign conventions for returns.
- Purchase-event definition.
- What qualifies as a customer purchase.
- Treatment of duplicate and zero-quantity rows.
- Whether negative prices are valid credits or invalid source data.
- Treatment of missing customer IDs.

The existing CLI already performs substantially more detailed validation than the Streamlit preprocessing helper. Make the same schema and validation implementation available to both.

### Date handling

If input has a declared format, require or use it. If the source contains ambiguous dates, warn or fail rather than silently choosing month/day versus day/month.

Keep the original timestamp when it exists and derive the daily grouping key separately. Document that timezone-naive timestamps represent a defined business timezone.

### Revenue and currency

Currency must be explicit in configuration or metadata. Update the UI formatting and generator defaults to agree with the chosen currency.

Do not infer a currency from the magnitude of prices. If multi-currency input becomes necessary, conversion must include a known exchange-rate policy and rate date; otherwise, the pipeline should explicitly operate on a single currency.

## 8.3 Make all analyses point-in-time correct

Implement a common snapshot function for feature creation and scoring.

Every rolling window, category preference, return feature, reference price, and transformed feature must be constructed from data available as of the snapshot cutoff.

The existing CLI already accepts `--as-of-date`, truncates future lines, and records the configured horizon. Preserve that behavior, but also use the same point-in-time contract for app-based summaries and for historical evaluation snapshots.

In addition, save the cutoff and event-grain policy as part of the output schema, not only as optional metadata.

## 8.4 Make model output and fallback behavior explicit

Standardize model result status values, for example:

- `fitted_and_accepted`
- `fitted_but_not_accepted`
- `fallback`
- `skipped`
- `failed`

The exact names are flexible. The important requirement is to distinguish a failed optimizer from a validated fit, and a fallback forecast from a model-produced forecast.

For each run, report the number of customers scored with each accepted model and the number who received fallback estimates. For probability forecasts, save calibration output and the chosen threshold separately from model-fitting results.

## 8.5 Make tests and CI part of the normal workflow

There is already useful test material to preserve:

- `app.py`'s fixed-seed, fixed-expected-metrics validation harness.
- `data_generator.py`'s schema and transaction/refund consistency validation.
- `retail_customer_analysis.py`'s accounting invariants.
- Existing clustering stability calculations.
- Basket association checks and FDR diagnostics.

Add a dedicated test suite that exercises these behaviors rather than relying only on aggregate golden-master numbers.

Critical tests include:

- Gross minus returns equals net revenue.
- Reaggregations reproduce transaction totals.
- Event counts obey the configured event grain.
- Date parsing does not silently reinterpret ambiguous input.
- No features use transactions beyond a snapshot cutoff.
- Return matching never links to a future purchase when timestamp ordering is available.
- The population-mean baseline is fitted from historical data.
- Model failure and non-convergence are explicitly reported.
- Basket presence is deduplicated appropriately.
- Empty, tiny, malformed, and degenerate datasets fail gracefully or return documented skips.
- A full synthetic run produces all expected required artifacts.
- HTML reporting survives missing optional charts without silently dropping core diagnostics.

CI should run a deterministic synthetic-data smoke test and the unit tests on a documented supported Python version. A second Python version can be added when package compatibility has been established.

## 8.6 Configuration, dependencies, and reproducibility

The pinned `requirements.txt` is a useful starting point. Keep the production stack small.

Recommended changes:

- Add `pyproject.toml` for project metadata, test/lint configuration, and packaging if packaging is introduced.
- Add separate development dependencies for testing and linting.
- Document the supported Python version; `data_generator.py` states Python 3.10+.
- Validate all numerical parameters, including inactivity thresholds and model penalization.
- Make `APP_CONFIG` and `METRIC_DEFINITIONS` authoritative or remove them if redundant.
- Centralize metric definitions used by the UI and report generators.
- Keep dependency upgrades controlled and tested; do not blindly update every pinned package at once.

For reproducibility, preserve the seed, configuration, package versions, input/schema identity, cutoff, event grain, model status, and feature schema version. A hash of a canonicalized input and configuration can help identify runs without exposing raw data.

The current metadata files store absolute input paths. Redact these paths in shareable artifacts or allow path recording to be disabled.

## 8.7 Improve operational workflow without unnecessary infrastructure

The repository already has a batch CLI and CSV/Parquet artifact workflow. This is sufficient for a local analytics product; a distributed orchestration service is not needed by default.

Prioritize:

- An explicit input/output CLI contract.
- Versioned or clearly separated output folders for repeatable runs.
- Atomic writes for output tables and metadata.
- A documented convention for failed, skipped, and incomplete analyses.
- A scoring command or documented batch mode that reads a new dataset and writes customer scores without requiring the full interactive UI.
- Configurable cleanup of temporary Streamlit work directories.
- Runtime and peak-memory measurements on representative data sizes.
- Stable output schemas for downstream dashboards.

The current basket and clustering budgets are useful. Benchmark them on small, medium, and larger synthetic datasets before increasing limits or adding more complex models.

## 8.8 Privacy and responsible use

Customer-level IDs appear in exported feature tables and downloadable customer/action lists. Even if identifiers are pseudonymous, those files should be treated as customer data.

Recommended safeguards:

- Avoid logging transaction rows or sensitive identifiers.
- Minimize customer-level data retained in temporary storage.
- Use explicit permissions before exposing the app to multiple users.
- Sanitize local paths from shareable reports.
- Document data retention and deletion for uploaded data.
- Limit exported columns to those necessary for a user's stated task.
- Document what each customer score means and its limitations.
- Do not use observational labels as claims of customer intent, profitability, or responsiveness.

If the tool eventually supports consequential operational actions, add a lightweight governance checklist covering intended use, failure modes, monitoring, and human review. The [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework) is a useful reference for organizing such controls without requiring enterprise infrastructure.

# 9. Prioritized Roadmap and Backlog

The estimates below are provisional. They assume one developer familiar with pandas and the current codebase, a working local Python environment, and no unexpected dependency or data-contract changes.

Priority definition

- P0: Correctness or foundational reliability issue that should be addressed before trusting model comparisons or expanding use.
- P1: High-value improvement needed for consistent analytics and robust evaluation.
- P2: Valuable capability or usability enhancement after the foundation is dependable.
- P3: Conditional/advanced capability whose value is not yet established.

## Phase 0 — Quick wins and correctness

| ID  | Priority and gap                                                           | Proposed change and location                                                                                                                                                                                                                       | Value and dependencies                                                                                                                                         | Effort | Computational/maintenance implications                                            | Testable acceptance criteria                                                                                                                                                                                                                       |
| --- | -------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ | --------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| R01 | P0 — Holdout mean baseline uses future outcomes                            | Fix `retail_customer_analysis.py`, `holdout_validation()` around line 581. Estimate the constant benchmark from calibration history or historical windows only.                                                                                    | Makes the baseline comparison statistically valid; no additional library required.                                                                             | Small  | Negligible runtime change; adds a baseline invariant test.                        | The constant forecast is unchanged if holdout outcomes are modified while calibration data stays fixed; metric outputs reproduce on fixed synthetic data.                                                                                          |
| R02 | P0 — UI and CLI use inconsistent purchase-event definitions                | Define an event grain and transaction-key policy. Start in `retail_customer_analysis.py` `build_trips()` and `app.py` `build_customer_features()`, `build_product_features()`, `build_basket_associations()`.                                      | Eliminates discrepancies between customer metrics, models, and basket analysis. Requires explicit business agreement on invoice versus customer-day semantics. | Medium | Simplifies future maintenance; may change historical metric values intentionally. | A fixed dataset produces consistent event counts, order denominators, and basket definitions across CLI and UI for the same configured grain.                                                                                                      |
| R03 | P0 — No dedicated tests, CI, or README                                     | Add `tests/`, CI, and `README.md`. Retain the built-in validation harness but make it part of the automated suite.                                                                                                                                 | Makes current correctness claims repeatable; required before a broader refactor.                                                                               | Medium | Low recurring compute cost; reduces long-term debugging cost.                     | A clean checkout can install dependencies, run automated unit tests, run an end-to-end synthetic smoke test, and follow documented instructions successfully.                                                                                      |
| R04 | P1 — Temporal matching, ambiguous dates, currency, and future demo horizon | Fix `retail_customer_analysis.py` `match_returns()` and `parse_dates()`; make currency explicit in `app.py`'s `fmt_currency()` and the data contract; change `data_generator.py` default end behavior to avoid unmarked future-dated demo records. | Improves trust in return diagnostics, ingestion, and monetary displays.                                                                                        | Medium | Minimal runtime cost; adds clear source-data contracts.                           | Return matching cannot attribute to a later purchase when timestamps are available; ambiguous dates warn/fail unless format is known; demo values use consistent currency labels; default demo horizon does not extend into an unexplained future. |

## Phase 1 — Foundational reliability

| ID  | Priority and gap                                              | Proposed change and location                                                                                                                                                           | Value and dependencies                                                                                | Effort | Computational/maintenance implications                                                              | Testable acceptance criteria                                                                                                                       |
| --- | ------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- | ------ | --------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| R05 | P1 — App preprocessing differs from CLI validation            | Consolidate schema and cleaning in a shared ingestion module. Replace or delegate `app.py` `prepare_transaction_frame()` and route both UI uploads and CLI input through shared rules. | One consistent report of exclusions and issue counts. Depends on R02's event and data definitions.    | Medium | Lower maintenance; moderate refactoring, without material model cost.                               | The same input file yields the same normalized rows, quality counts, and accounting totals through both entry points.                              |
| R06 | P1 — Configuration and metric declarations are not canonical  | Make `APP_CONFIG` and `METRIC_DEFINITIONS` authoritative in `app.py`, or remove unused declarations. Share definitions with the CLI and report layer.                                  | Prevents contradictory KPI labels/formulas and configurable defaults drifting across implementations. | Medium | Minimal runtime cost; simplifies future additions.                                                  | Changing one declared threshold or metric scope changes the documented consuming functions; tests check formula and scope consistency.             |
| R07 | P1 — Numerical model status can be obscured by fallback       | Improve optimizer status checks and model reporting in `retail_customer_analysis.py`, especially `fit_bgnbd()`, `fit_gamma_gamma()`, and `score_customers()`.                          | Makes forecasts auditable and avoids mixing fallbacks with accepted model predictions.                | Medium | Low additional compute; modest logging and result-schema work.                                      | Non-converged and failed fits are visible in artifacts; non-finite predictions are rejected; fallback counts and reason appear in metadata/report. |
| R08 | P1 — Example workflow depends on absent sample file           | Update `show_all_results.py` to accept `--input`, use the documented schema contract, and optionally invoke the shared synthetic generator.                                            | Improves clean-checkout usability and reduces schema duplication.                                     | Small  | Negligible compute; removes a brittle local path assumption.                                        | The script works with a supplied supported CSV/Parquet input and a documented synthetic-demo mode without requiring `sample_data.parquet`.         |
| R09 | P1 — Reproducibility is not systematically tested across time | Add snapshot feature builders and shared event semantics to the test suite, then construct several cutoff/horizon cases.                                                               | Enables safe historical feature engineering and rolling-origin evaluation. Depends on R02, R03, R05.  | Medium | Additional test runtime grows with snapshots; production scoring cost remains similar per snapshot. | Features at cutoff \\(t\\) are invariant to rows dated after \\(t\\); identical inputs/configurations produce identical features.                  |

## Phase 2 — Core analytical enhancements

| ID  | Priority and gap                                                      | Proposed change and location                                                                                                                                                                                              | Value and dependencies                                                                                                       | Effort | Computational/maintenance implications                                    | Testable acceptance criteria                                                                                                                                                   |
| --- | --------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- | ------ | ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| R10 | P1 — Single holdout is insufficient evidence                          | Extend `retail_customer_analysis.py` `holdout_validation()` to run multiple historical cutoff dates and write a per-origin metrics table.                                                                                 | Shows whether BG/NBD and revenue estimates generalize across periods. Depends on R01 and point-in-time feature construction. | Medium | More model fits and report rows; manageable in an offline batch pipeline. | Each origin records training cutoff, evaluation window, eligible sample count, baseline/model metrics, and fit status. No feature or target data cross the cutoff incorrectly. |
| R11 | P1 — Current metrics omit broader forecast and calibration evidence   | Add WAPE, aggregate bias, metric confidence intervals, and richer diagnostics to `holdout_metrics` and `insight_engine.py`. Add Brier/log loss/calibration only for a specifically defined supervised probability target. | Separates ranking, calibration, customer-level error, and total forecast quality. Depends on R10.                            | Medium | Small/medium output and plotting increase; low model complexity.          | Every model-target pair has defined metrics, denominators, sample sizes, and a valid leakage-free baseline; the HTML report shows performance across origins.                  |
| R12 | P1 — Useful features are present but incomplete or inconsistent       | Extend the canonical feature module with stable behavioral trend, basket composition/entropy, return-unit ratios, and point-in-time category affinity. Reconcile existing rolling windows and cadence features.           | Improves customer interpretation and may improve baseline segmentation or predictive performance. Depends on R02, R05, R09.  | Medium | Mostly pandas aggregation; control memory and categorical feature count.  | Features have formulas, valid missing-value behavior, cutoff invariants, stable output dtypes, and ablation results on future data.                                            |
| R13 | P1 — Segment selection may prioritize statistical separation over use | Add RFM-versus-cluster comparison and cross-snapshot stability reporting in `retail_customer_analysis.py` and the Trust/Customers views in `app.py`.                                                                      | Keeps clustering only when it offers additional stable, interpretable customer groupings.                                    | Medium | Modest extra snapshot calculations and report output.                     | Segment solutions report size, profile, bootstrap/time stability, and a reason for selection or rejection.                                                                     |
| R14 | P2 — Incomplete and misleading chart panels                           | Fix `insight_engine.py` cohort-revenue placeholders and chart labels. Add a reconciled revenue bridge only when the required table is produced.                                                                           | Reduces ambiguity and connects actual data to visible business questions. Depends on stable upstream metrics.                | Medium | Minor compute cost, mainly data and report changes.                       | No active chart renders a blank measurement panel without explanation; chart title, axes, and takeaway match the calculation; revenue bridges reconcile.                       |

## Phase 3 — Optional capabilities

| ID  | Priority and gap                                                              | Proposed change and location                                                                                                                                               | Value and dependencies                                                                                                                                  | Effort | Computational/maintenance implications                                  | Testable acceptance criteria                                                                                                                                        |
| --- | ----------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ | ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| R15 | P2/P3 — BG/NBD does not supply a supervised next-horizon purchase probability | Add logistic regression behind a separate propensity task in a new model module and expose it only if required.                                                            | Produces a target-aligned probability that can be calibrated and evaluated. Requires historical customer snapshots and enough positive/negative labels. | Medium | Adds a dependency if scikit-learn is chosen; fitting cost remains low.  | Out-of-time PR-AUC, calibration, Brier/log loss, and top-k metrics are reported against simple baselines. The model is kept only if it adds value.                  |
| R16 | P2 — Existing horizon revenue is not contribution value                       | If cost/margin and return economics become available, add a separately named expected-contribution model and cohort value outputs.                                         | Aligns prioritization with profit rather than sales alone. Depends on a validated financial data contract.                                              | Large  | More feature/model/financial reconciliation complexity.                 | Forecasts use declared cost and return definitions; backtests validate contribution value against future realized values; revenue and contribution are never mixed. |
| R17 | P3 — No direct test of action impact                                          | Use current action rules and exported affected entities to support a randomized pilot and holdout reporting. Add effect estimation only after exposure/outcome data exist. | Measures whether interventions actually improve outcomes.                                                                                               | Large  | Operational and experimental complexity rather than heavy compute.      | A predeclared experiment reports incremental outcomes with uncertainty, treatment cost, and an appropriate comparison group.                                        |
| R18 | P2 — Limited operational lineage and batch contract                           | Standardize metadata, add input/config identity and schema versions, redact absolute paths, and document standalone batch scoring.                                         | Makes results comparable across runs and safer to share.                                                                                                | Medium | Small metadata/artifact overhead; improves reproducibility and privacy. | Every run declares input/schema identity, cutoff, horizon, event grain, versions, seed, model statuses, and output files without requiring raw local paths.         |

## Suggested sequencing

- Quick wins: R01, R03, R04, and R08.
- Foundational changes: R02, R05, R06, R07, and R09.
- Core analytical enhancements: R10–R14.
- Conditional/advanced extensions: R15–R18.

The sequence is intentionally dependency-aware. For example, adding a propensity model before fixing snapshot semantics and evaluation would create more code without establishing that the new model is trustworthy.

# 10. First Five Recommended Changes

These five actions offer the best initial combination of analytical value, implementation practicality, and risk reduction.

## 1. Fix holdout leakage in the population-mean baseline

Location: `retail_customer_analysis.py`, `holdout_validation()` around line 581.

Change:

- Estimate the population-mean prediction from calibration/training data only.
- Ensure the baseline uses a defined historical observation window comparable with the forecast horizon.
- Add a test that changes holdout outcomes while keeping training/calibration data fixed.

Acceptance criteria:

- The baseline prediction is independent of the holdout labels.
- Its MAE/RMSE are computed on the same eligible holdout customers as the comparator models.
- The report distinguishes the constant baseline from the customer-specific empirical-rate baseline.
- The model-validation chart and its prose use the corrected results.

Why first: Every model-selection claim depends on a fair comparison.

## 2. Unify purchase-event grain, identity, and financial definitions

Locations:

- `retail_customer_analysis.py`: `build_trips()` around line 291; `transaction_summary()` around line 949; `basket_affinity()` around line 1042.
- `app.py`: `build_customer_features()` around line 1060; `build_product_features()` around line 1315; `build_basket_associations()` around line 1455.

Change:

Define a shared policy for transaction IDs, qualifying purchases, trip aggregation, return handling, and gross/net monetary measures. Route both UI and CLI through this implementation.

Do not simply replace every transaction count with customer-day counts. Decide the appropriate grain for each business question and label it explicitly.

Acceptance criteria:

- The same source data and configuration produce consistent transaction/trip counts between UI and CLI.
- A transaction ID reused across customers cannot merge their baskets accidentally.
- Product summaries contain one row per product key, with metadata conflicts reported rather than causing unexplained duplicate summaries.
- Revenue reconciliations hold across source, transaction, product, and category roll-ups.

Why second: Event semantics are foundational to RFM, future-purchase modeling, retention, product performance, and basket affinity.

## 3. Create an automated regression suite and clean-checkout workflow

Locations: New `tests/`, `README.md`, `.github/workflows/ci.yml`, and optionally `requirements-dev.txt`/`pyproject.toml`.

Change:

Start with tests for the known leakage bug, event definitions, point-in-time feature boundaries, return matching, model status, empty data, and accounting invariants. Use the existing generator and validation harness rather than inventing a separate test dataset for every scenario.

Acceptance criteria:

- A clean checkout has documented setup and execution instructions.
- A deterministic synthetic dataset passes unit tests and an end-to-end pipeline/report smoke test.
- The `show_all_results.py` example no longer requires an undocumented local-only dataset.
- CI runs automatically on changes and reports failures.
- The repository documents its supported Python version and input schema.

Why third: Refactoring and model improvements become safer only when behavioral expectations are encoded in tests.

## 4. Correct temporal/data-contract issues and currency labels

Locations:

- `retail_customer_analysis.py`: `parse_dates()` around line 196 and `match_returns()` around line 310.
- `app.py`: `fmt_currency()` around line 1575 and ingestion around line 968.
- `data_generator.py`: `generate()` defaults around line 228.

Change:

Preserve precise timestamps for return matching when the source has them; require or explicitly select ambiguous date formats; make currency an explicit display/data assumption; prevent unmarked future-dated default synthetic output.

For returns, do not assume that matching on normalized calendar day establishes a valid transaction ordering. If exact timestamps are unavailable, use a clearly documented fallback and report reduced matching precision.

Acceptance criteria:

- A return cannot be attributed to a later purchase on the same date when precise ordering is available.
- Ambiguous date parsing is observable and testable.
- The app and generated demonstration data display consistent currency labels.
- A default synthetic run does not silently look like a complete historical record extending beyond the current date.
- Date and currency assumptions appear in documentation/report metadata.

Why fourth: These fixes reduce the risk of showing plausible but incorrectly interpreted business data.

## 5. Replace the single-cutoff evaluation with rolling-origin validation

Locations:

- `retail_customer_analysis.py`: `holdout_validation()` around line 563 and run orchestration around lines 1314–1326.
- `insight_engine.py`: `chart_validation()` around line 231 and `chart_calibration()` around line 265.

Change:

Use multiple historical cutoff dates and the same forecast horizon at each cutoff. Refit on data available at each cutoff, then evaluate on its future horizon. Include the corrected baseline and save per-origin metrics.

Add aggregate forecast bias, WAPE when the denominator is valid, and customer-level error metrics. Report binary probability calibration only for a probability model with a correctly defined future purchase label.

Acceptance criteria:

- The same cutoff and horizon generate reproducible results.
- Every feature is built from information available at or before the cutoff.
- Metrics include eligible customers, actual and predicted totals, baseline comparisons, model status, and cutoff date.
- The report displays model performance across evaluation periods instead of relying on one test window.
- A model is not declared superior based solely on an isolated metric gain from one period.

Why fifth: It supplies the evidence needed to decide whether new features or algorithms should actually be adopted.

# 11. Assumptions, Limitations, and Open Questions

## 11.1 Assumptions used in this plan

1. The repository is intended to remain a relatively lightweight retail customer analytics application rather than become a large-scale enterprise data platform.
2. CSV and Parquet line-item data are the primary inputs.
3. Customer ID, transaction date, transaction ID, product/category, price, and quantity remain the core analytical fields.
4. Existing descriptive, probabilistic, segmentation, survival, and affinity capabilities should be preserved unless evaluation shows they lack value.
5. The application should produce both interpretable business outputs and reproducible analytical artifacts.
6. More complex modeling is conditional on evidence that it materially improves decision usefulness over the current methods and simple baselines.

## 11.2 Limitations of this audit

- The review was based on the repository files accessible on `main` as of October 10, 2026. No README was present in the inspected tree, and the absence of a README means stated project promises outside the code could not be evaluated.
- No production dataset, actual customer schema contract, business metric definitions, or operational deployment target was supplied.
- No independent execution of the full application and complete test matrix was performed during this source review. Where test evidence is absent, the plan calls for tests rather than asserting that runtime failures have been demonstrated.
- Effort and impact assessments are provisional. Actual effort depends on data size, expected response time, desired output formats, and whether this is a portfolio/demo tool or a business application.

## 11.3 Open questions to resolve before advanced modeling

### Data and measurement

1. Are transaction IDs globally unique, or unique only within customer, store, register, or channel?
2. Should a purchase event mean an invoice, a customer-day trip, or a business-specific event?
3. Are transaction timestamps truly available at time-of-day precision, and in which timezone?
4. Are returns reliably tied to original transactions, or must they be inferred from customer/product/date?
5. Is the dataset single-currency, and should the UI treat prices as EUR by default?
6. Are missing customer IDs common enough that customer-level metrics need an explicit anonymous-customer reporting policy?
7. Does the data include true customer acquisition dates or only observed purchase history?

### Business decisions

8. Is the main priority customer reporting, repeat-purchase forecasting, retention targeting, product cross-sell, or customer lifetime value?
9. What decision horizon matters most: 30, 60, 90, or another number of days?
10. Does the organization need rankings, calibrated purchase probabilities, aggregate forecasts, or all three?
11. Will model outputs merely inform analysts, or will they automatically determine communications or offers?
12. Are margin, cost, campaign exposure, and intervention outcomes available? Without them, profit LTV, causal discount analysis, and uplift modeling should remain out of scope.

### Scale and operation

13. What are the normal and largest expected transaction-file sizes?
14. Does the application run only locally or will it be hosted for multiple users?
15. Are customer-level exports permitted, and what retention/deletion rules apply to uploaded data?
16. Must historical runs be repeatable exactly, or is reproducibility within a documented tolerance sufficient for numerical optimization?

## 11.4 External references

The following sources support the recommended methodology; they are not claims that the repository already implements every described practice.

1. BG/NBD repeat-purchase modeling: Peter S. Fader, Bruce G. S. Hardie, and Ka Lok Lee, ["Counting Your Customers" the Easy Way: An Alternative to the Pareto/NBD Model](https://doi.org/10.1287/mksc.1040.0098), Marketing Science, 2005. Relevant to the probabilistic purchase model already present in `retail_customer_analysis.py`.
2. RFM and customer value: Peter S. Fader, Bruce G. S. Hardie, and Ka Lok Lee, [RFM and CLV: Using Iso-Value Curves for Customer Base Analysis](https://doi.org/10.1509/jmkr.2005.42.4.415), Journal of Marketing Research, 2005. Supports treating RFM as a useful descriptive baseline while evaluating future value with explicit behavioral assumptions and holdout evidence.
3. Temporal validation: [scikit-learn — TimeSeriesSplit documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html). Relevant to the need to preserve chronological order in validation; customer snapshot construction must still be explicitly implemented for this repository.
4. Probability calibration: [scikit-learn — Probability calibration](https://scikit-learn.org/stable/modules/calibration.html). Relevant if a future supervised purchase-propensity model requires calibrated probabilities rather than ranking scores alone.
5. Responsible deployment and governance: [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework). A reference for proportionate risk management, documentation, monitoring, and responsible use if analytics outputs become operational decisions.

## Final recommendation

Treat the next development cycle as a validity and consistency sprint: fix the leaked evaluation baseline; unify purchase-event and metric definitions; add automated tests and documentation; correct the date, return, and currency contracts; then introduce rolling-origin evaluation. Only after those changes should new algorithms be evaluated.

That sequence makes the existing tool more trustworthy, improves the business value of its current capabilities, and creates a defensible basis for later extensions such as purchase propensity, net contribution forecasting, or experimentally evaluated retention strategies.