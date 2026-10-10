# Retail Customer Analytics — Repository Improvement Plan

Repository: [hrczggyrgy/retail_customer_analytics](https://github.com/hrczggyrgy/retail_customer_analytics) Audit date: October 10, 2026 Document status: Source-based audit and prioritized implementation proposal Primary objective: Improve analytical correctness, eliminate memory-intensive basket processing, strengthen feature engineering and evaluation, and make the application reliable on larger retail datasets without unnecessary infrastructure.

## 1. Executive Summary

### 1.1 Overall assessment

The repository is already a substantial customer analytics application, rather than a basic RFM notebook. It includes probabilistic purchase modeling, customer segmentation, retention and survival analysis, transaction-level summaries, product affinity, synthetic data generation, a Streamlit interface, and an artifact-based reporting pipeline.

The most valuable next step is not to add more algorithms indiscriminately. It is to address computational bottlenecks, correct gaps in the canonical customer features, consolidate duplicated analytical logic, and ensure that model and business comparisons are reproducible.

The highest-priority findings are:

| Priority | Finding                                                                                                                                                               | Business or technical consequence                                                                                                                     |
| -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| P0       | The Streamlit basket-analysis implementations materialize every candidate product pair in a Python list before aggregating it.                                        | Peak RAM can grow dramatically with basket count, basket size, and product assortment.                                                                |
| P0       | The canonical feature builder assigns `net_spend = gross_spend` and does not add the customer-level return features expected by downstream modeling and segmentation. | Returned value is not consistently reflected in customer economics, and clustering can fail when `return_value_ratio` is missing.                     |
| P0       | Basket analysis is implemented in multiple places in `app.py`, with less restrictive memory controls than the CLI affinity pipeline.                                  | Different interfaces can produce inconsistent results and have different memory failure modes.                                                        |
| P1       | The existing time-aware validation framework needs stronger consolidated reporting across historical origins.                                                         | One holdout or one aggregate score is insufficient to establish robustness across retail periods.                                                     |
| P1       | The customer-feature pipeline still contains bounded dense customer-by-category representations.                                                                      | Category-level arrays can also consume substantial memory at very high customer counts, even though the category dimension is capped.                 |
| P1       | The feature and model interface needs a regression test for complete feature contracts, including returns, model status, and segmentation.                            | Component-level refactors can silently disable downstream analyses.                                                                                   |
| P2       | Supervised purchase propensity is not yet a completed production capability.                                                                                          | A calibrated purchase-probability model could support targeting, but should be introduced only after validation and feature contracts are dependable. |

Effort and impact estimates below are provisional because the code was inspected but the complete application and test suite were not executed.

### 1.2 The recommended strategic direction

Keep the existing lightweight analytical core and improve it in this order:

1. Remove the unbounded basket-pair materialization from Streamlit. Use incremental pair counting, candidate limits, deterministic processing budgets, and one shared implementation. Do not construct a transaction-by-product or product-by-product matrix during calculation.
2. Repair canonical return-aware customer features. Ensure net spend, return ratios, and other required fields are constructed consistently before scoring, segmentation, and reporting.
3. Test and benchmark the complete analytical paths. Add basket memory regression tests, feature-contract tests, and reproducible large-data benchmarks to the existing test suite and CI.
4. Make event definitions and evaluation outputs authoritative. Use the same event grain, basket identity, monetary definitions, and validation metrics throughout the CLI and Streamlit application.
5. Expand analytical capabilities selectively. Prioritize better repeat-purchase forecasts, segment stability, and decision-focused outputs before adding a supervised model.

### 1.3 Definition of success

The improved application should:

- Run basket analysis without creating an intermediate object proportional to the total number of basket-pair occurrences.
- Enforce predictable memory and computational budgets, report any sampling or truncation explicitly, and never silently imply complete analysis when results are partial.
- Produce consistent customer features and financial definitions at any selected historical cutoff.
- Compare forecasts with leakage-free baselines across multiple historical periods.
- Report uncertainty, model status, and data limitations alongside numerical results.
- Reproduce analysis outputs from a documented input, configuration, and random seed.
- Remain usable as a lightweight Python application without mandatory cloud services, databases, or distributed infrastructure.

## 2. Repository Audit

### 2.1 Audit scope and evidence conventions

The current repository was inspected through its source files, project configuration, documentation, tests, and CI configuration.

- Verified: Directly supported by the current repository contents.
- Inference: A likely weakness or consequence that should be confirmed through execution or representative data.
- Conditional: Appropriate only if the data, business requirements, and expected deployment justify it.

This is a static source audit. No claim is made that the application or complete automated test suite was executed successfully.

### 2.2 Current project structure

The repository already has a useful separation between the main analysis pipeline, Streamlit user interface, reporting, and feature modules.

| File or module                                               | Verified responsibility                                                                                                                            | Assessment                                                                                                                                |
| ------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `README.md`                                                  | Installation, input schema, quick-start instructions, analytical overview, output artifacts, limitations                                           | A useful documentation foundation.                                                                                                        |
| `app.py`                                                     | Streamlit application, uploaded data handling, customer/product analytics, basket analysis, visualizations, and pipeline orchestration             | Approximately 6,800 lines; the largest concentration of application logic and duplicated analytics.                                       |
| `retail_customer_analysis.py`                                | CLI ingestion, event construction, return matching, purchase modeling, segmentation, survival, basket affinity, validation, and exported artifacts | Strong existing analytical core, but a large single module that should gradually become a thin CLI wrapper.                               |
| `insight_engine.py`                                          | Loads result tables and generates charts, HTML reporting, KPIs, and skipped-chart diagnostics                                                      | Good separation of reporting from model execution.                                                                                        |
| `data_generator.py`                                          | Generates synthetic transactions, catalog information, and return records                                                                          | Valuable for reproducible tests and stress testing.                                                                                       |
| `show_all_results.py`                                        | End-to-end demonstration and display of analytical outputs, with input and synthetic modes                                                         | Useful integration-test entry point.                                                                                                      |
| `src/retail_customer_analytics/ingestion.py`                 | Shared normalization, date parsing, schema checks, currency formatting, and data-quality counts                                                    | Good foundation for consistent ingestion.                                                                                                 |
| `src/retail_customer_analytics/features.py`                  | Canonical point-in-time customer feature construction                                                                                              | Correct architectural direction; its current financial and downstream feature contracts need attention.                                   |
| `src/retail_customer_analytics/metrics.py`                   | Period-level business metrics and aggregation helpers                                                                                              | Appropriate shared utility layer.                                                                                                         |
| `src/retail_customer_analytics/period_manager.py`            | Date ranges, reporting periods, and period filtering                                                                                               | Worth preserving.                                                                                                                         |
| `src/retail_customer_analytics/polars_utils.py`              | Polars conversions, aggregations, dtype optimizations, and join utilities                                                                          | Offers a starting point for selective performance optimization, but does not make all pipeline operations memory-efficient automatically. |
| `tests/`                                                     | Fixtures and tests covering events, models, temporal validation, features, basket affinity, and related correctness cases                          | Existing tests provide a strong base; expand them to cover the newly identified gaps and resource limits.                                 |
| `.github/workflows/ci.yml`                                   | Automated testing, linting, and synthetic smoke tests                                                                                              | Existing CI should be extended rather than replaced.                                                                                      |
| `requirements.txt`, `requirements-dev.txt`, `pyproject.toml` | Runtime dependencies, development tools, test settings, and project metadata                                                                       | Functional foundation, but version reproducibility and dependency declarations need tightening.                                           |
| `PLAN.md`, `TODO.md`, `improvement_plan_v1.md`               | Work planning, completion tracking, and prior audit recommendations                                                                                | These documents need reconciliation to avoid treating already-completed tasks as open work.                                               |

### 2.3 How the components currently connect

The current architecture follows this broad path:

```
CSV / Parquet upload or CLI input
              |
              v
      Shared ingestion
              |
              v
      Customer transactions
              |
              v
     Event and return logic
              |
              v
   Customer features and models
              |
       +------+------+
       |             |
       v             v
 Segmentation    Basket affinity
       |             |
       +------+------+
              |
              v
      Structured artifacts
              |
       +------+------+
       |             |
       v             v
  Streamlit UI   HTML report
```

In `app.py`, `run_pipeline()` calls the CLI analysis entry point, writes results into a temporary work directory, and invokes `insight_engine.py` to generate report artifacts. The application also contains direct implementations of customer, product, and basket analytics.

The architecture should retain the reusable CLI and reporting components. The main improvement is to eliminate competing implementations of the same analytical tasks.

### 2.4 Verified strengths worth preserving

Probabilistic repeat-purchase and value estimation. The CLI implements BG/NBD and Gamma-Gamma models, including parameter fitting, prediction fallbacks, and time-based holdout validation. These are appropriate lightweight starting points for non-contractual retail purchasing data.

Interpretable customer analytics. RFM segmentation, behavioral features, category affinity, purchase cadence, and customer profiles already provide useful analytical foundations.

Retention and timing analysis. Kaplan–Meier repeat-purchase survival, first-observed cohort retention, and interpurchase intervals are more informative than a single arbitrary inactivity threshold.

Statistical basket analysis. `retail_customer_analysis.py::basket_affinity()` includes candidate limits, basket sampling, pair-operation budgets, association statistics, Benjamini–Hochberg multiple-testing correction, confidence intervals, and split-half persistence checks. This is an important existing implementation to preserve and refactor for reuse.

Data-quality and accounting controls. `src/retail_customer_analytics/ingestion.py` tracks missing identifiers, malformed dates, invalid numeric values, duplicate rows, and other issues. The CLI also performs financial accounting checks and reports conflicting product metadata.

Reproducibility foundations. Fixed seeds, JSON metadata, exported model parameters, synthetic fixtures, and CI are already present.

### 2.5 Critical finding: the basket-analysis RAM problem

The most important verified performance issue is in `app.py`.

The following implementations generate a Python list of all basket-pair occurrences, convert that list into a DataFrame, and then aggregate it:

- `build_basket_associations()` — approximately lines 1585–1740.
- `_compute_basket_analysis()` — approximately lines 5260–5410.

The pattern is effectively:

```
pairs = []for products in baskets:    for a, b in combinations(set(products), 2):        pairs.append((a, b))pair_df = pd.DataFrame(pairs, columns=["product_a", "product_b"])pair_counts = pair_df.groupby(["product_a", "product_b"]).size()
```

This creates an intermediate representation of every pair occurrence before reducing the data to unique pairs.

For a basket containing \\(k\\) distinct eligible products, the number of pair occurrences is:

\\[ \text{pairs}(k)=\frac{k(k-1)}{2} \\]

Across all baskets, the total is:

\\[ P=\sum\_{b=1}^{B}\frac{k_b(k_b-1)}{2} \\]

Here, \\(B\\) is the number of baskets and \\(k_b\\) is the number of eligible distinct products in basket \\(b\\).

The problem is not merely the final number of unique pairs. The application materializes repeated pair occurrences across all baskets, retains Python tuples, and subsequently creates a DataFrame and group-by intermediates. Peak memory can therefore be much larger than the final association table.

This calculation should be redesigned, not just moved to a different dense matrix format.

#### Required replacement

Create a shared implementation in `src/retail_customer_analytics/affinity.py` that:

1. Constructs a stable basket key according to the configured event definition.
2. Filters to eligible, sufficiently frequent products before pair counting.
3. Processes one basket at a time.
4. Deduplicates product IDs within each basket.
5. Updates pair counts incrementally.
6. Applies explicit limits to candidate items, pair operations, retained pair keys, and total baskets.
7. Produces association metrics from the aggregated counts.
8. Creates a small matrix only when rendering a bounded top-item visualization.

Use the CLI implementation in `retail_customer_analysis.py::basket_affinity()` as the starting point for computational controls, rather than maintaining a third independent algorithm.

For exact aggregation, a `collections.Counter` or dictionary keyed by a canonical product-pair tuple is sufficient. With at most \\(M\\) eligible products, the theoretical unique unordered-pair count is bounded by \\(M(M-1)/2\\). At the existing CLI default of 100 candidate items, that is at most 4,950 unique unordered pairs.

A hard candidate limit is important: streaming does not guarantee low memory if the number of distinct retained pairs can itself grow without bound.

#### Memory controls must be observable

When a computational budget is reached, the tool must distinguish between:

- `completed`: all eligible baskets were processed.
- `sampled`: a documented, reproducible sample of eligible data was processed.
- `truncated`: processing stopped because a budget was reached, so results are incomplete.
- `skipped`: there was insufficient usable data.
- `failed`: an unexpected error prevented completion.

Record the eligible basket count, processed basket count, candidate count, pair-operation count, retained pair count, configured budgets, and reason for any incomplete analysis.

Do not stop processing after an arbitrary early customer and label the output as complete. If sampling is required, select the sample deterministically and report its coverage.

### 2.6 Secondary memory risk: dense customer-by-category arrays

Two other paths deserve attention.

In `retail_customer_analysis.py::category_mix()`, the code creates:

```
m = np.zeros((len(customers), len(cats)))
```

The number of category columns is capped by `max_clr_categories`, with an additional aggregated category possible. However, the customer dimension is not similarly bounded in this allocation.

In `src/retail_customer_analytics/features.py::_category_affinity_features()`, customer spending is pivoted into a customer-by-category representation.

If \\(N\\) customers and \\(C\\) category columns are represented using 64-bit floating-point values, the dense value array alone requires approximately:

\\[ \text{memory bytes}=8NC \\]

With one million customers and 31 category columns, this is approximately 248 MB for one array, before pandas indexes, other feature arrays, SVD intermediates, model inputs, and copies.

This is a separate, conditional scalability risk from basket-pair materialization. It does not mean the existing arrays will necessarily fail on ordinary retail datasets.

Recommended changes:

- Keep global category selection bounded.
- Avoid materializing category-by-customer arrays when only a few aggregate measures are needed.
- Build customer/category spend in long form first.
- Compute category entropy, category concentration, and top-category features directly from grouped data.
- Use chunked processing or compact matrix representations only where the clustering method genuinely requires the complete category mix.
- Record memory and runtime before making a larger change to NumPy or pandas processing.

Sparse representations are an available tool, not a goal in themselves. See the [pandas sparse-data documentation](https://pandas.pydata.org/docs/user_guide/sparse.html).

### 2.7 Verified analytical consistency gap: net spend definition

In `src/retail_customer_analytics/features.py::build_customer_snapshot()`, the current implementation sets:

```
cust["net_spend"] = cust["gross_spend"]
```

This means the `net_spend` feature is not distinctly defined from `gross_spend`. While the current dataset has no returns (all quantity > 0), the feature builder should still maintain a clear semantic distinction between gross and net spend for future extensibility.

The same feature builder does not currently add the `return_value_ratio` feature. However, `retail_customer_analysis.py::cluster_customers()` expects `return_value_ratio` as part of its clustering inputs. In `run_analysis()`, the canonical customer snapshot is passed through scoring and segmentation, and the segmentation failure is caught and recorded in diagnostics.

This creates a concrete feature-contract risk: a feature omitted in a refactor can break a downstream analytical component without stopping the entire application.

Required remediation:

- Maintain semantic distinction: `net_spend = gross_spend - return_value` (where `return_value = 0` when no returns exist).
- Ensure `return_value_ratio` feature is present (with value 0.0 when no returns).
- Add all required output features to the canonical builder.
- Validate feature availability before modeling and clustering.
- Add regression tests that ensure clustering either runs with the expected fields or reports an explicit, actionable skip reason.

### 2.8 Other verified gaps and methodological risks

| Finding                                                                             | Evidence                                                                                                            | Recommendation                                                                                       |
| ----------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| Duplicate basket algorithms                                                         | `app.py::build_basket_associations()` and `_compute_basket_analysis()`                                              | Consolidate into one implementation and have the UI consume its result.                              |
| UI pair calculation has no equivalent explicit candidate and pair-operation budgets | The two UI calculations materialize the global list of pair occurrences.                                            | Apply configurable budgets consistently across every entry point.                                    |
| Basket identity may depend on globally unique transaction IDs                       | The UI basket builders group by `transaction_id` alone, while the CLI supports a `customer-transaction` key policy. | Use one explicit key policy and test reused invoice IDs across customers.                            |
| Feature-contract failures may be recorded without failing the entire pipeline       | `run_analysis()` catches clustering exceptions and records failed diagnostics.                                      | Expose component failure states prominently in the UI and summary report.                            |
| The project has overlapping implementation and planning documents                   | `README.md`, `PLAN.md`, `TODO.md`, `improvement_plan_v1.md`, and `REFACTORING_SUMMARY.md`                           | Use `TODO.md` for status and make `IMPROVEMENT_PLAN.md` the authoritative forward plan.              |
| Dependencies are not strictly pinned                                                | `requirements.txt` and `pyproject.toml` use minimum versions such as `pandas>=2.1.0`.                               | Document tested versions and add a reproducible lock or constraints file for supported environments. |

### 2.9 Important correction to earlier project planning

The current repository already contains `README.md`, `tests/`, `requirements-dev.txt`, `pyproject.toml`, and `.github/workflows/ci.yml`. `TODO.md` also marks multiple earlier improvements as complete, including holdout baseline correction, shared ingestion, feature extraction, and rolling-origin validation.

Some statements in `improvement_plan_v1.md` and `PLAN.md` describe those capabilities as absent or unfinished. They should not be carried forward as current findings without rechecking the source.

In particular, the current `retail_customer_analysis.py::holdout_validation()` computes its population-mean baseline from calibration history through `_population_mean_baseline()`, and the temporal-validation tests cover the baseline's invariance to changes in holdout outcomes. The previously identified population-mean baseline leakage is already addressed in the current code and should be protected by existing regression tests, not scheduled as a new implementation.

## 3. Customer Questions and Capability Map

The following map prioritizes business questions that the current schema can support and separates existing capability from proposed work.

| Business question                                                      | Required minimum data                                                                                 | Current capability                                                                    | Recommended method and outputs                                                                                                   | How usefulness should be measured                                                                                  |
| ---------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| Who are our different types of customers?                              | `customer_id`, `transaction_date`, `price`, `quantity`; preferably `product_id` and `category`        | RFM segmentation and stability-checked clustering                                     | Repair return-aware features; retain interpretable RFM profiles; keep clustering only where stable and meaningfully distinct     | Cluster size, bootstrap and temporal stability, profile separation, actionable segment differences                 |
| Which customers are likely to buy again soon?                          | Identified customer purchase history with enough repeated observations and a defined forecast horizon | BG/NBD expected purchase counts and `P(alive)`                                        | Retain BG/NBD as the baseline; optionally introduce supervised purchase propensity where calibrated probability is required      | Out-of-time count error; probability metrics and calibration for supervised propensity; improvement over baselines |
| Which customers have declining activity?                               | Customer-level purchase timestamps and sufficient history                                             | Recency, inactivity heuristic, rolling activity, cadence, survival analysis           | Add robust recent-versus-prior activity measures and observable lapse risk; treat churn as a horizon-specific prediction problem | Future lapse or repeat-purchase performance, retention at fixed horizons, stability across time origins            |
| Which customers have the highest future revenue potential?             | Purchase history, customer identity, transaction value; margin/cost fields for contribution value     | BG/NBD plus Gamma-Gamma expected revenue                                              | Improve return-aware spend features, model-status reporting, and forecast validation                                             | Future revenue MAE, WAPE, aggregate bias, calibration by value group                                               |
| Which categories or products are associated with customer preferences? | Customer, product/category, timestamp, positive purchase value or quantity                            | Category affinity, breadth, concentration and customer segments                       | Add recency-weighted category affinity and improved share/concentration features                                                 | Temporal stability, repeat purchase in category, improved forecast performance in ablation                         |
| Which products are often bought together?                              | A reliable basket key, product ID and purchase quantity                                               | Basket association testing in CLI and separate UI pair builders                       | Shared bounded pair-counting engine with support, confidence, lift, uncertainty and FDR controls                                 | Agreement with exact results on small data, pair support, stability, runtime and peak RAM                          |
| What tends to be purchased on a later trip?                            | Identified customers with chronologically ordered purchase events                                     | `next_trip_affinity.csv`                                                              | Preserve next-trip transition analysis; validate transition denominators and sequence definition                                 | Out-of-time transition lift and transition probability reliability                                                 |
| When are customers likely to buy again?                                | Multiple purchase dates per identified customer                                                       | Interpurchase intervals, cadence statistics, time-to-second-purchase survival         | Use robust per-customer cadence features and survival curves; add horizon-specific lapse risk if justified                       | Repeat-purchase timing error, survival calibration, performance across cohorts                                     |
| Which customer cohorts retain better?                                  | Customer IDs and first-observed transaction timestamps                                                | First-observed cohort retention                                                       | Continue cohort analysis with cohort sizes, complete cohort ages and uncertainty                                                 | Retention comparisons at equivalent cohort ages; uncertainty-aware interpretation                                  |
| Where are returns reducing customer value?                             | Customer/product identifiers, purchase and return timestamps, signed quantities and prices            | Return matching and return summaries (N/A - no returns in current dataset)            | Return handling infrastructure exists for future use; current features use gross spend as net spend                              | Reconciled gross = net values; return infrastructure ready for future data                                       |
| Which products or categories need further investigation?               | Product/category, quantities, prices, timestamps and returns                                          | Product/category summaries and reporting views                                        | Add repeat-buyer rate, return-rate denominators, data-quality indicators, and trends                                             | Reconciled totals, confidence intervals for small samples, prospective operational outcomes                        |
| Which campaigns or actions actually improve customer outcomes?         | Treatment assignment or exposure, treatment timing, outcomes and preferably margin                    | Action-oriented descriptive summaries, but no verified randomized action-effect model | Defer causal estimation until intervention data exists; prepare controlled pilot measurement                                     | Incremental outcome relative to control, uncertainty, treatment cost and contribution margin                       |

### 3.1 Essential additions

The immediate business value comes from consistent financial and customer features, reliable repeat-purchase estimates, efficient basket affinity, and useful diagnostics.

These capabilities are directly connected to the existing dataset and application.

### 3.2 Conditional additions

A supervised purchase propensity model is justified when the business needs a calibrated probability of purchase within a defined future horizon. It should not be added simply because a classification algorithm is available.

A customer journey model requires more than transaction data if the objective concerns email exposure, website visits, campaign contacts, channel switching, or touchpoints preceding purchase. Transaction sequences alone support purchase-sequence analysis, but not a complete customer journey.

Margin-adjusted customer value requires cost, margin, discount, tax, and return-economics data at an appropriate grain. The current monetary forecasts represent revenue, not profit.

## 4. Feature Engineering Plan

### 4.1 Feature-engineering principles

All customer features should use a common interface:

```
features = build_customer_snapshot(    transactions=transactions,    as_of=as_of,    event_grain=event_grain,    windows_days=(30, 90, 365),    recent_window_days=90,)
```

The existing `src/retail_customer_analytics/features.py::build_customer_snapshot()` is the natural foundation.

Each feature should have a defined input contract, aggregation grain, observation window, missing-data behavior, and point-in-time rule. The same transformation definitions should be used for training, historical evaluation, and future scoring.

A feature is not automatically suitable for a model just because it is available. The first question is whether it is known at the prediction timestamp and whether it is measurable consistently in the actual deployment environment.

### 4.2 Core transactional features

| Feature                  | Definition                                                                      | Required columns and aggregation                                               | Interpretation and risks                                                                                                                               | Priority |
| ------------------------ | ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------ | -------- |
| Recency                  | \\(R=t\_{\text{as-of}}-t\_{\text{last purchase}}\\), in days                    | `customer_id`, purchase timestamp; one value per customer at cutoff            | Lower means more recent activity. Use the same qualifying-purchase definition for all interfaces.                                                      | P0       |
| Frequency                | Number of qualifying purchases in the observation period                        | Customer-level count of canonical purchase events                              | The result depends on trip versus transaction grain. A purchase line is not a purchase event.                                                          | P0       |
| Monetary value           | Sum of positive purchase revenue over a declared window                         | `price`, `quantity`, timestamp and customer; aggregate positive purchase value | Describes observed gross purchase revenue, not profit.                                                                                                 | P0       |
| Net spend                | Gross purchase revenue minus return value under the declared accounting policy  | Purchases plus returns joined by customer and snapshot cutoff                  | Current dataset has no returns; net_spend = gross_spend. Feature builder should maintain the semantic distinction for future extensibility.             | P0       |
| Average order/trip value | Total qualifying purchase revenue divided by qualifying events                  | Canonical event table; lifetime or rolling window                              | Useful for separating high-frequency and high-basket-value customers. Exclude or explicitly handle zero-value events.                                  | P1       |
| Customer tenure          | \\(t\_{\text{as-of}}-t\_{\text{first observed purchase}}\\)                     | Customer purchase events                                                       | First-observed tenure is not necessarily true customer tenure because prior history may be missing.                                                    | P1       |
| Repeat count             | \\(\max(N\_{\text{events}}-1,0)\\)                                              | Canonical customer events                                                      | Align the count with the BG/NBD calibration definition.                                                                                                | P0       |
| Purchase cadence         | Mean, median and variability of interpurchase intervals                         | Timestamp differences sorted by customer                                       | Sparse histories produce unreliable cadence estimates. Include the number of observed gaps.                                                            | P1       |
| Return value ratio       | Return value divided by gross purchase value over the same window               | Customer/product returns and purchase revenue                                  | Current dataset has no returns; value is 0.0. Avoid dividing returns from one period by purchases from a different period without explicitly naming the measure.                                     | P0       |
| Return unit ratio        | Returned units divided by purchased units, under an appropriate matching policy | Signed `quantity`, product and customer                                        | Current dataset has no returns; value is 0.0. Large values can indicate returns exceeding recorded purchases, mismatched periods or partial data. Do not automatically interpret as dissatisfaction. | P1       |

### 4.3 Extended RFM

RFM should remain an interpretable business summary. It should not become the only customer representation.

The recommended customer profile includes:

- Recency: Days since last qualifying purchase.
- Frequency: Event count over the selected observation window.
- Monetary: Gross purchase value and return-adjusted net spend over the same window.
- Value per purchase: Average and median qualifying event value.
- Tenure: Time between first-observed purchase and the selected cutoff.
- Repeat experience: Repeat count, share of customers' purchases that are repeats where defined, and observed number of interpurchase gaps.
- Lapse indicators: Recency relative to a historical or customer-specific cadence baseline.
- Historical value concentration: Share of customers or sales represented by high-value groups.

Use business-readable labels for reports, but retain raw metrics for modeling and reproducibility.

RFM thresholds should be configurable and documented. Quantile-based RFM groups are relative to the population and snapshot; they should not be mistaken for stable, universal customer types.

### 4.4 Rolling-window features

The existing `features.py` already builds rolling purchase counts and gross purchase revenue for configured windows, and recent-versus-prior change features.

Retain that foundation and standardize the definitions.

For customer \\(i\\), window \\(w\\), and cutoff \\(t\\):

\\[ S\_{i,w}(t)=\sum\_{j:\\,t-w < t\_{ij}\leq t} \max(r\_{ij},0) \\]

\\[ F\_{i,w}(t)=\sum\_{j:\\,t-w < t\_{ij}\leq t} \mathbf{1}(\text{qualifying purchase event}\_{ij}) \\]

Where \\(r\_{ij}\\) is the defined purchase-line revenue and \\(t\_{ij}\\) the transaction date. These formulas apply to raw line values for spend and canonical event counts for frequency; line counts and purchase-event counts must not be confused.

| Feature                            | Formula or definition                                          | Window                            | Purpose and safeguards                                                                                  | Priority |
| ---------------------------------- | -------------------------------------------------------------- | --------------------------------- | ------------------------------------------------------------------------------------------------------- | -------- |
| Purchase counts                    | \\(F\_{30},F\_{90},F\_{365}\\)                                 | 30, 90 and 365 days               | Short-term demand versus longer-term habit; use identical cutoff boundaries throughout the pipeline.    | P0       |
| Purchase value                     | \\(S\_{30},S\_{90},S\_{365}\\)                                 | Same windows                      | Distinguishes recent high-value customers from historically valuable but inactive customers.            | P0       |
| Recent-versus-prior count change   | \\((F\_{\text{recent}}-F\_{\text{prior}})/F\_{\text{prior}}\\) | Two adjacent equal-length windows | Undefined if prior count is zero; retain a missing indicator and optionally a separate absolute change. | P1       |
| Recent-versus-prior revenue change | \\((S\_{\text{recent}}-S\_{\text{prior}})/S\_{\text{prior}}\\) | Two adjacent equal-length windows | Zero prior spend should not be silently converted into infinite growth.                                 | P1       |
| Activity acceleration              | Recent event rate minus prior event rate                       | Adjacent equal-length windows     | Can flag increasing or declining activity without relying on unstable percentage growth.                | P1       |
| Revenue concentration              | Share of spend from top categories or products                 | Lifetime and recent windows       | Useful for assortment affinity, but should use a stable category mapping.                               | P1       |
| Active-month fraction              | Active months divided by observed eligible months              | Usually last 12 calendar months   | Measures consistency; the denominator must reflect the time actually available for observation.         | P1       |

### 4.5 Cadence and lifecycle features

The repository already computes gap-based regularity measures, an inactivity heuristic, interpurchase intervals, and survival curves.

Improve rather than duplicate these.

| Feature                       | Definition                                                             | Business use                                                                 | Leakage or interpretation safeguard                                                                                                  | Priority |
| ----------------------------- | ---------------------------------------------------------------------- | ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ | -------- |
| Median interpurchase gap      | Median days between consecutive purchase events                        | Approximate customer purchasing cadence                                      | Requires at least one gap; two events provide a fragile individual estimate.                                                         | P1       |
| Gap variability               | Standard deviation or coefficient of variation of purchase gaps        | Distinguish predictable replenishment from irregular shopping                | Use a minimum number of gaps and explicit missing-value semantics.                                                                   | P1       |
| Customer-specific lapse ratio | Recency divided by a robust cadence estimate                           | Identify customers whose inactivity is unusual relative to their own history | Shrink sparse estimates toward category or population-level cadence rather than treating a single gap as reliable.                   | P1       |
| Cadence deviation             | Recent purchase interval versus historical median interval             | Detect changes in shopping frequency                                         | Only use historical gaps observable by the cutoff.                                                                                   | P1       |
| Active-month trend            | Trend in monthly activity or purchase value                            | Distinguish sustained engagement from isolated spikes                        | Construct a complete calendar-month series with zero-activity months rather than regressing only over months that contain purchases. | P1       |
| Time-to-second-purchase risk  | Probability of remaining without a repeat purchase beyond time \\(t\\) | Assess early repeat behavior and define useful observation horizons          | Account for right-censoring and left-censoring in the data collection period.                                                        | P1       |

The recommendation about zero-activity months is particularly relevant to `src/retail_customer_analytics/features.py::_trend_features()`: the current slope calculation operates on observed customer-month rows, so a long gap between purchases is not represented as a zero-activity month in the trend fit.

### 4.6 Basket composition and product affinity

Basket composition features are useful for understanding purchasing behavior without constructing a dense customer-by-product interaction matrix.

| Feature                        | Formula or definition                                                               | Required columns and aggregation                   | Interpretation and safeguards                                                                    | Priority           |
| ------------------------------ | ----------------------------------------------------------------------------------- | -------------------------------------------------- | ------------------------------------------------------------------------------------------------ | ------------------ |
| Distinct products per basket   | Count of unique product IDs in each event; aggregate per customer by mean or median | `customer_id`, `product_id`, canonical basket key  | Use the same basket definition as the association engine.                                        | P1                 |
| Distinct categories per basket | Number of distinct non-missing categories per event                                 | Customer, category and basket key                  | A category-mapping change can create artificial shifts.                                          | P1                 |
| Basket size variability        | Standard deviation or coefficient of variation of event value or units              | Customer-level event values                        | Use a consistent definition and report the number of qualifying events.                          | P1                 |
| Category spend share           | Category purchase value divided by customer purchase value over the same period     | Customer, category, price, quantity                | Useful for category affinity; output width must be capped.                                       | P1                 |
| Category entropy               | \\(H_i=-\sum_c p\_{ic}\log(p\_{ic})\\) for customer category shares                 | Customer/category aggregated spend                 | Higher values indicate a broader category mix, not necessarily stronger loyalty or higher value. | P1                 |
| Category concentration         | \\(HHI_i=\sum_c p\_{ic}^2\\)                                                        | Customer/category spend shares                     | Higher values indicate more concentrated category spend.                                         | P1                 |
| Repeat product/category rate   | Share of distinct products/categories with purchases on multiple qualifying events  | Product/category identity and canonical event keys | Product breadth and true repeat behavior must not be conflated.                                  | P1                 |
| Inter-trip transition          | Count and probability of next-event item/category given current-event item/category | Chronologically ordered customer baskets           | Distinguish transitions from same-basket co-occurrence.                                          | Existing; validate |

The CLI already includes a capped category-mix feature for clustering and category-level customer affinity features. Preserve this strategy initially rather than introducing product embeddings, recommender systems, or high-dimensional encodings.

### 4.7 Discount sensitivity, returns, and product behavior

Discount sensitivity: `add_price_index()` in `retail_customer_analysis.py` computes a price index using each product's median observed price. The `promo_like` flag is a price-based proxy, not a confirmed promotion indicator.

Recommended additions include:

- Customer-level share of spend on proxy-discounted lines.
- Median and variability of price index.
- Historical purchase frequency for products when their observed price index is lower.
- Repeat-buyer rate by product/category.

These features can identify hypotheses about price behavior or product quality. They do not establish causal price sensitivity or promotion uplift.

Conditional requirement: If promotion exposure, list price, campaign ID, and treatment timestamps become available, use those fields rather than treating the price-index proxy as a substitute for observed campaign exposure.

Note: Return-related features (product-level return value/unit ratios, matched-return share, unmatched return amount, return rate changes) are not applicable to the current dataset which has no returns. The infrastructure exists in the codebase for future datasets with returns.

### 4.8 Missing-data, outlier, encoding, and scaling policy

| Issue                            | Recommended policy                                                                                                                                             |
| -------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Missing customer IDs             | Preserve them in transaction accounting and product summaries where possible; exclude them only from customer-specific analyses and report the excluded share. |
| Missing product/category IDs     | Retain usable financial totals while explicitly tracking which product- or category-level analyses cannot use the rows.                                        |
| Invalid dates and numeric values | Use the shared ingestion layer, explicit date-format controls, and issue counts; do not silently drop rows.                                                    |
| Outlier transaction values       | Investigate data validity first. Use explicit quantile or robust scaling only for model inputs where justified, without overwriting raw financial values.      |
| Zero denominators                | Return missing/undefined metrics or a named fallback rather than silently assigning extreme ratios.                                                            |
| Missing behavioral features      | Distinguish unavailable history from true zero activity. Include history-length or missingness indicators where relevant.                                      |
| Categorical variables            | Use stable category mappings or controlled encoding. Do not create a dense one-hot matrix across every customer and product.                                   |
| Numerical scaling                | Fit transformations using calibration data only. Reuse the fitted transformation when evaluating future periods or scoring new customers.                      |
| Highly correlated features       | Test feature ablations and reduce redundant inputs where they add no measurable value.                                                                         |
| Feature drift                    | Track distributions and missingness by batch and date; flag changes before automatically refitting models.                                                     |

### 4.9 Point-in-time correctness

Every training row must represent information available at its observation cutoff.

For a feature snapshot at time \\(t\\):

```
snapshot_rows = transactions[    transactions["transaction_day"] <= as_of]features = build_customer_snapshot(    snapshot_rows,    as_of=as_of,    event_grain=event_grain,)
```

The final implementation should enforce this contract internally, not depend solely on callers having correctly filtered the data.

Required tests:

- Adding rows dated after cutoff \\(t\\) does not change features at \\(t\\).
- Reordering input rows does not change results.
- A feature's window boundaries are consistent across CLI and UI.
- Transformations fitted on training data do not learn information from validation or test periods.
- Missing and degenerate cases produce documented values and stable output schemas.

The goal is a single feature definition that supports training, evaluation, customer exploration, and future batch scoring.

## 5. Algorithm and Baseline Recommendations

### 5.1 Essential additions

These improvements correct or strengthen existing capabilities. They do not require a large new machine-learning stack.

| Method or change                       | Business problem                                                    | Baseline and requirements                                                                                                                | Interpretability and computational cost                                                                   | Evidence required for adoption                                                                                 |
| -------------------------------------- | ------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| Bounded pair-counting engine           | Product associations exceed available memory                        | Exact counts on small data; candidate-filtered incremental counters on large data                                                        | Highly interpretable; avoids a global list of pair occurrences and avoids a dense co-occurrence matrix    | Exact agreement with reference results on small fixtures; explicit coverage and budget metadata on large tests |
| Return-aware customer scoring features | Historical spend overstates customer value when returns are ignored | Gross spend and return value grouped by customer; no new algorithm dependency                                                            | Very low model complexity; requires reliable temporal matching and accounting definitions                 | Reconciled totals (gross = net for current dataset), consistent features across entry points, clustering success on valid fixtures               |
| Rolling-origin forecast comparison     | A model may perform well on one date range but poorly on others     | Existing BG/NBD and empirical-rate/constant benchmarks                                                                                   | Uses current lightweight probabilistic models; runtime increases approximately with the number of origins | Stable or explainably variable metrics across several historical evaluation origins                            |
| Explicit model-status contract         | A fallback can be mistaken for a fitted prediction                  | Existing fitting status and fallback behavior in `retail_customer_analysis.py::fit_bgnbd()`, `fit_gamma_gamma()` and `score_customers()` | Small maintenance cost; greatly improves trustworthiness                                                  | Tests confirm failed, skipped, and fallback states propagate to artifacts and reports                          |
| Canonical feature and event contracts  | UI and CLI can disagree on event counts or feature values           | Existing `ingestion.py` and `features.py` modules                                                                                        | Low algorithmic complexity; reduces duplication                                                           | Parity tests across both entry points using the same data and configuration                                    |

### 5.2 Preserve and validate existing probabilistic models

BG/NBD

Problem: Forecast the number of repeat purchasing events over a future horizon from historical frequency, recency, and observation duration.

Why it fits: The repository already implements fitting and prediction in SciPy/NumPy without requiring a specialized modeling framework.

Baseline: Empirical customer event rate and a constant population forecast estimated from historical calibration information only.

Limitations: BG/NBD assumptions may be a poor fit for highly seasonal purchasing, subscriptions, contracts, large promotional shocks, or events defined inconsistently. Its \\(P(\text{alive})\\) output is not, by itself, a calibrated probability of purchasing within the next 90 days.

Gamma-Gamma

Problem: Estimate expected purchase-event value for use with expected future purchase counts.

Why it fits: It is already implemented and lightweight.

Baseline: Observed customer mean event value, with a population-level fallback where appropriate.

Limitations: The model's monetary target and assumptions must be evaluated explicitly. Its expected value is not profit and should not be described as lifetime customer value without a defined lifetime horizon and corresponding validation.

Adoption decision: Keep both methods, report fitting and convergence status, evaluate fallbacks separately, and assess each model against its baseline across historical cutoffs.

### 5.3 High-value enhancements

| Method                             | Recommended use                                                                                 | Baseline                                                                    | Data and assumptions                                                        | Why not add it unconditionally                                                                                                            |
| ---------------------------------- | ----------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- | --------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| Regularized logistic regression    | Predict whether a customer makes at least one qualifying purchase in the next specified horizon | Population purchase rate; simple recent-activity or empirical-rate baseline | Historical point-in-time snapshots with binary future labels                | It requires an explicitly defined purchase target and enough out-of-time observations.                                                    |
| Calibrated purchase propensity     | Target a group expected to purchase or repurchase within a fixed horizon                        | Existing BG/NBD and simple behavioral baselines                             | Sufficient positive outcomes, meaningful negative examples, calibration set | A useful ranking does not guarantee accurate probabilities. Calibrate only on out-of-sample or temporally appropriate calibration data.   |
| Improved survival analysis         | Estimate time to repeat purchase and time without a repeat                                      | Current Kaplan–Meier approach                                               | Correct censoring definitions and adequate observation time                 | Use a more complex survival model only if it adds reliable predictive or explanatory value beyond current curves.                         |
| RFM versus clustering comparison   | Decide whether extra segments improve decisions                                                 | RFM segmentation                                                            | Stable features and repeated historical snapshots                           | High clustering scores alone do not demonstrate actionable customer differences.                                                          |
| Candidate-filtered basket analysis | Identify interpretable product and category associations                                        | Single-product frequency, empirical pair frequency and lift                 | Valid basket identity and sufficient support                                | More complicated frequent-itemset or recommender methods should wait until exact counts are scalable and candidate budgets are validated. |

### 5.4 Recommended supervised-propensity design

Implement the first supervised model only after the feature contracts and time-aware evaluation are dependable.

Define the label for customer \\(i\\) at cutoff \\(t\\):

\\[ y_i = \begin{cases} 1, & \text{if customer } i \text{ makes a qualifying purchase in } (t,t+H] \\\ 0, & \text{otherwise} \end{cases} \\]

Use features such as:

- Recency and historical purchase counts.
- Recent and prior-window changes.
- Historical purchase value and return-aware value measures.
- Purchase cadence and variability.
- Number of active months.
- Category breadth and selected category affinity measures.

Start with regularized logistic regression, using only stable and interpretable inputs. Keep the implementation optional until a separate model module is justified.

Compare its future-horizon predictions against BG/NBD and the simple baselines using precision-recall performance, probability calibration, and top-ranked customer usefulness. Report probabilities only after assessing calibration.

See the [scikit-learn model-evaluation guide](https://scikit-learn.org/stable/modules/model_evaluation.html) and [probability calibration documentation](https://scikit-learn.org/stable/modules/calibration.html).

### 5.5 Optional research or advanced extensions

| Extension                                            | Trigger for considering it                                                                                                     | Prerequisites                                                                                                    |
| ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------- |
| Frequent-itemset mining beyond pairs                 | Pair analysis is stable, but the business has a concrete need for multi-item bundle discovery                                  | Bounded candidate generation, explicit support thresholds, holdout validation                                    |
| More sophisticated repeat-purchase models            | Existing BG/NBD forecasts show persistent, material errors not resolved by better event definitions, seasonality, or baselines | Multiple reliable historical periods and sufficient repeated purchases                                           |
| Customer-level personalized category recommendations | Cross-sell recommendations are a concrete business deliverable, not merely a request for affinity summaries                    | Sufficient interaction history, candidate catalog, future-outcome evaluation and recommendation coverage metrics |
| Contribution-margin value forecasting                | Decisions require profitability rather than revenue                                                                            | Cost/margin data, return economics, and a validated financial accounting contract                                |
| Causal campaign or retention-impact estimation       | The business needs incremental uplift rather than descriptive associations                                                     | Treatment/exposure logs, appropriate comparison groups and prospective experiments                               |

Do not add deep-learning embeddings, large collaborative-filtering matrices, or a distributed data-processing platform merely to make the project sound more advanced. Demonstrated business value and measured resource needs should determine that investment.

## 6. Evaluation and Metrics

### 6.1 Evaluation strategy

The current `retail_customer_analysis.py` already has a time-based holdout and supports rolling-origin validation. These capabilities should become the authoritative evaluation framework rather than being reimplemented separately in the application.

The recommended evaluation contract has four parts:

1. Point-in-time correctness: Every feature is constructed from information available at the cutoff.
2. Out-of-time testing: Training and calibration use earlier data; evaluation targets a strictly future window.
3. Baseline comparison: Each model is compared with a simple, leakage-free baseline for the same target and horizon.
4. Business interpretation: Metrics are accompanied by coverage, uncertainty, model status, and target definitions.

### 6.2 Time-aware train, validation, and test splits

For a forecast horizon of \\(H\\) days and an evaluation cutoff \\(T\\):

- Construct features only from eligible events up to \\(T\\).
- Train or fit the model using the designated calibration history.
- Score customers using information available at \\(T\\).
- Define observed outcomes from \\((T,T+H]\\).
- Repeat at earlier historical origins when sufficient data exists.
- Reserve a later period for final evaluation when selecting among multiple model variants.

The existing rolling-origin controls, `--rolling-origins` and `--origin-spacing-days`, should be used to produce comparable metrics across multiple forecast periods.

Prefer several distinct origins that capture different seasons or levels of retail activity rather than many highly overlapping windows that give a false impression of independent evidence.

The [scikit-learn TimeSeriesSplit documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) explains the general principle of using past observations for training and later observations for testing. The repository's fixed-duration customer snapshots require explicit cutoff and horizon semantics rather than blindly applying a row-based splitter.

### 6.3 Purchase-count and revenue forecast metrics

| Metric               | Definition                                                                      | Interpretation                                                              | Potential pitfall                                                             |
| -------------------- | ------------------------------------------------------------------------------- | --------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| MAE                  | \\(\frac{1}{n}\sum_i\|\hat y_i-y_i\|\\)                                         | Average absolute customer-level forecast error                              | Does not convey the direction of aggregate error.                             |
| RMSE                 | \\(\sqrt{\frac{1}{n}\sum_i(\hat y_i-y_i)^2}\\)                                  | Emphasizes larger errors                                                    | A small number of high-volume customers can dominate it.                      |
| WAPE                 | \\(\frac{\sum_i\|\hat y_i-y_i\|}{\sum_i\|y_i\|}\\)                              | Absolute error relative to observed total volume                            | Undefined when the denominator is zero; can conceal poor individual rankings. |
| Aggregate bias       | \\(\frac{\sum_i(\hat y_i-y_i)}{\sum_i y_i}\\), where the denominator is nonzero | Positive indicates aggregate overforecast; negative indicates underforecast | Aggregate accuracy can hide large customer-level errors that cancel out.      |
| Spearman correlation | Rank correlation between predictions and realized values                        | Measures whether high-scored customers tend to realize higher outcomes      | Good ranking does not imply good forecast calibration or absolute accuracy.   |

Compute these separately for purchase counts and future revenue. Report the evaluation horizon, customer count, observed total, predicted total, and baseline results next to each metric.

Use WAPE only with an explicit zero-total policy. Do not treat WAPE and MAE as interchangeable measures.

### 6.4 Binary purchase propensity metrics

These metrics should be used for a true supervised target such as whether a customer makes a purchase during the next 90 days. Do not report them as if BG/NBD's `P(alive)` were already a calibrated next-horizon purchase probability.

| Metric                     | Definition or purpose                                                                        | When it is useful                                                | Misleading interpretation                                                                                         |
| -------------------------- | -------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| ROC-AUC                    | Probability-based ranking of positive against negative examples                              | Measuring overall discrimination                                 | Can look strong when the positive class is rare, even if precision is poor.                                       |
| PR-AUC / Average precision | Summarizes precision-recall performance across thresholds                                    | Important when few customers purchase in the horizon             | Its baseline depends on positive-class prevalence; always report the prevalence.                                  |
| Brier score                | Mean squared error of predicted probabilities                                                | Evaluating probabilistic prediction quality                      | A lower score can reflect discrimination as well as calibration; it is not a pure calibration measure.            |
| Log loss                   | Negative log-likelihood of the observed outcomes                                             | Evaluating probability quality while penalizing confident errors | Extremely confident wrong predictions can dominate the score.                                                     |
| Calibration curve          | Observed purchase rate versus mean predicted probability by bin                              | Checking whether predicted probabilities are reliable            | Small bins, especially for rare purchasers, can appear unstable.                                                  |
| Lift\@K                    | Purchase rate among the top-scored \\(K\\) customers divided by the population purchase rate | Evaluating a limited-capacity targeting list                     | Lift depends on \\(K\\), the evaluation population and positive prevalence; it does not estimate campaign uplift. |

For probability calibration, use a distinct calibration period and preserve the final test period for evaluation. Where possible, compare reliability diagrams, Brier score, log loss and lift at operationally meaningful list sizes.

### 6.5 Segmentation evaluation

The existing clustering code uses robust scaling, silhouette scores and bootstrap stability through adjusted Rand index. It also has cross-snapshot stability analysis.

Retain those checks and supplement them with:

- Customer count and percentage per cluster.
- Median and distribution of recency, frequency and return-aware value.
- Category affinity and basket behavior profiles.
- Cluster sizes and profile changes across snapshots.
- A comparison of cluster assignments with the existing RFM segments.
- A clear selected, rejected, skipped or failed status with reason.

A high silhouette score is not sufficient to claim business usefulness. Tiny clusters, unstable cluster membership, redundant features, or segments that differ only in historical spend may have little operational value.

Before creating additional clustering algorithms, test whether stable, easily interpretable RFM groups and existing K-means segments already answer the intended business question.

### 6.6 Basket-affinity evaluation

Basket analysis should be evaluated in both computational and statistical terms.

Association metrics

For two products \\(A\\) and \\(B\\):

\\[ \operatorname{support}(A,B)=\frac{n\_{AB}}{N} \\]

\\[ \operatorname{confidence}(A\rightarrow B)=\frac{n\_{AB}}{n_A} \\]

\\[ \operatorname{lift}(A\rightarrow B)= \frac{n\_{AB}/n_A}{n_B/N} =\frac{n\_{AB}N}{n_A n_B} \\]

Where \\(N\\) is the number of eligible baskets and \\(n_A,n_B,n\_{AB}\\) are the basket counts containing each item or item pair.

Retain the existing CLI's customer-level association tests, false-discovery-rate correction, confidence intervals and persistence checks where their denominators and assumptions are appropriate.

Computational evaluation

- Exact agreement with an exhaustive reference implementation on small, deterministic fixtures.
- Peak RAM as basket and product counts increase.
- Runtime and memory versus basket size, catalog size, basket count and pair-operation budget.
- Repeatability under a fixed seed and unchanged input.
- Explicit partial-coverage status whenever sampling or truncation occurs.

Statistical safeguards

- Apply minimum pair counts in addition to relative support where appropriate.
- Report support and counts beside lift.
- Avoid interpreting extreme lift from rare item pairs without considering uncertainty.
- Evaluate whether association patterns persist in a later time period.
- Distinguish customer-level co-occurrence from basket-level co-occurrence.

A strong observational association is a hypothesis for further business testing, not an estimate of causal cross-sell uplift.

### 6.7 Survival and retention evaluation

The repository's survival and retention capabilities should be preserved.

Recommended measures:

- Number of customers at risk at each time point.
- Kaplan–Meier repeat-purchase survival estimates and confidence intervals.
- Fixed-horizon repeat-purchase or lapse rates.
- Cohort retention rates at comparable cohort ages.
- Cohort sizes and explicit rules for incomplete months.
- For a future predictive survival model, out-of-time survival calibration and discrimination metrics suitable for censored observations.

A cohort with a shorter observation period should not be compared directly with a cohort that has had a full year of follow-up.

### 6.8 Robustness, uncertainty and reproducibility

Every model comparison should include the following safeguards:

| Safeguard                     | Required implementation                                                                                                                             |
| ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| Bootstrap uncertainty         | Continue using the existing seeded bootstrap approach where appropriate; report the number of bootstrap samples and confidence interval definition. |
| Evaluation-origin variability | Report the metric at each origin and a summary across origins. Do not report only the pooled average.                                               |
| Feature ablation              | Compare the baseline feature set against one added feature family at a time on identical origins.                                                   |
| Data-quality sensitivity      | Repeat important comparisons on clearly defined clean-data subsets where practical.                                                                 |
| Model status                  | Separate successfully fitted models from fallbacks, skipped analyses and failures.                                                                  |
| Reproducibility               | Record configuration, cutoff, horizon, event grain, seed, input/schema identity and software versions.                                              |
| Resource use                  | Log input size, candidate counts, elapsed time and peak-memory measurements for the main bottlenecks.                                               |

### 6.9 Offline model performance versus business impact

Offline model quality does not establish business impact.

For example, a propensity model may rank customers by their likelihood of buying without identifying who is likely to change their behavior after receiving a marketing message.

To assess incremental business value, run a prospective controlled experiment where appropriate. Define the treatment, comparison group, eligibility, primary outcome, time window and treatment costs before analyzing results.

Use randomized holdout or other defensible experimental assignment where feasible. Report incremental outcomes with uncertainty and use contribution margin rather than revenue alone when the decision concerns profitability and the necessary financial data exists.

## 7. Visualization and Dashboard Plan

### 7.1 Business-facing views

The repository already contains many useful charts in `insight_engine.py`, and Streamlit exposes multiple analytical tabs in `app.py`. Improve consistency, decision relevance, denominators, and diagnostic context before adding more chart types.

| Business question                                  | Recommended chart                                                                             | Required data                                                             | Filters and grouping                                                        | Decision supported and interpretation risks                                                                                    |
| -------------------------------------------------- | --------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| What is happening to retail revenue and purchases? | Monthly line chart with revenue, purchase events and active customers in separate panels      | `monthly_summary.csv` and event/customer counts                           | Date range, category, department; select gross or net definition explicitly | Shows growth or contraction. Revenue alone may increase because of prices or mix, rather than more customers.                  |
| What are the major customer groups?                | Segment profile table plus dot/bar charts of median recency, frequency, value and return rate | Customer features and `rfm_segment_profiles.csv` / `cluster_profiles.csv` | Segment, event grain, snapshot                                              | Identifies groups with different observable behaviors. Segment profiles are descriptive and relative to a selected population. |
| Which customers are valuable but inactive?         | Scatter plot of recency versus future value estimate or historic value                        | Customer features with model status                                       | RFM/cluster group, value range, date cutoff                                 | Supports investigation of customer opportunities. Distinguish predicted horizon revenue from historical revenue and profit.    |
| How quickly do new customers repeat purchase?      | Kaplan–Meier survival plot with confidence intervals and number-at-risk table                 | `repeat_survival.csv` and cohort attributes                               | New versus previously observed customers; acquisition/first-observed cohort | Makes repeat timing and censoring visible. First-observed cohorts are not necessarily true acquisition cohorts.                |
| Are recent customer cohorts retaining better?      | Cohort retention heatmap with cohort size annotations                                         | `cohort_retention_new_customers.csv`                                      | Cohort month, cohort age                                                    | Compares behavior at equivalent cohort ages. Incomplete cohorts should be masked rather than shown as zero retention.          |
| Which categories contribute to growth or decline?  | Growth-versus-revenue-share scatter with labeled material categories                          | `category_summary.csv` or period-based category features                  | Current and comparison periods, category                                    | Helps prioritize assortment or category investigations. Low baseline revenue can create misleading percentage growth.          |
| Which product pairs merit further investigation?   | Ranked horizontal bar chart of supported pairs, with support and lift                         | `basket_affinity.csv`                                                     | Basket grain, category/product level, minimum support                       | Shows strongest supported relationships. Ranking by lift alone favors rare associations.                                       |
| What products tend to appear on subsequent trips?  | Directed transition chart or top-transition table                                             | `next_trip_affinity.csv`                                                  | Product/category, minimum transition count                                  | Supports replenishment or cross-sell hypotheses. A transition is a temporal association, not evidence of intervention impact.  |
| Where are returns concentrated?                    | Product/category return-rate table with denominators and confidence intervals                 | Purchase/return summaries with quantity and value                         | Product, category, month                                                    | Helps prioritize investigations. High return rates can reflect return windows or product mix and require further context.      |
| Which categories have unusual seasonal patterns?   | Monthly trend or small-multiple chart with calendar periods clearly aligned                   | Monthly and category summaries                                            | Category, department, time range                                            | Identifies seasonal and changing patterns. Year-over-year comparisons should align comparable calendar periods.                |

### 7.2 Diagnostic and trust views

| Diagnostic question                               | Recommended visualization                                                                                     | Required data                                                      | Use and safeguards                                                                                                                         |
| ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------ |
| Are forecasts reliable across historical periods? | Small multiples of forecast metrics by validation origin                                                      | `holdout_metrics.csv` with origin/date, model and target columns   | Exposes seasonality, drift and unstable model performance. Each point must use the same target and horizon definition.                     |
| Are predicted purchase probabilities calibrated?  | Reliability diagram with bin counts and a baseline                                                            | Out-of-time supervised propensity probabilities and outcomes       | Shows whether stated probabilities correspond to observed frequencies. Do not use this chart for `P(alive)` as if it were the same target. |
| Is model error concentrated in some customers?    | Actual-versus-predicted plot and error distribution by value/frequency group                                  | Holdout actual and predicted counts/revenue                        | Identifies systematic underforecasting or overforecasting. Use common axis scales and show aggregate bias.                                 |
| Are the segments stable?                          | Segment-size chart, profile comparison and bootstrap/time-snapshot stability plot                             | `cluster_diagnostics.csv`, profiles and cross-snapshot assignments | Shows whether segment definitions are reproducible, not just well separated on one dataset.                                                |
| Does a basket run stay within budget?             | Run diagnostic panel for baskets, candidates, processed pair operations, retained pairs, runtime and peak RAM | Basket metadata/diagnostic output                                  | Proves the system respects resource constraints and identifies when results are sampled or truncated.                                      |
| Where is data quality deteriorating?              | Bar chart of issue counts, with row denominators and run-over-run trend                                       | `data_quality.json` and issue table                                | Shows missingness, malformed records, duplicate counts and invalid values. Counts should be interpreted relative to data volume.           |
| Are financial totals consistent?                  | Gross-to-return-to-net reconciliation table and chart                                                         | Accounting and transaction summaries                               | Verifies financial roll-ups and identifies discrepancies. All amounts must use a consistent currency and accounting basis.                 |
| Which features are missing or drifting?           | Feature distribution and missingness summaries by snapshot                                                    | Canonical features and feature metadata                            | Makes preprocessing differences and new-data drift visible without relying on model performance alone.                                     |

### 7.3 Basket visualizations after refactoring

The calculation engine should produce aggregated pair data, not a full product-by-product matrix.

Retain small display visualizations such as a top-20 lift heatmap or a top-pair ranking. In particular:

- `app.py::_plot_basket_heatmap()` already restricts its visualization to a small set of products.
- `insight_engine.py::chart_rules()` also limits its visualization to a bounded number of items.

These bounded display matrices are not the primary RAM concern. The expensive step is generating the complete list of all pair occurrences before aggregation.

The new UI should load the shared affinity output and build the optional visualization from the resulting small association table. It should never recalculate the global pair list solely to create a chart.

## 8. Architecture and Engineering Improvements

### 8.1 Recommended target architecture

The current repository is already moving toward a modular package. Continue that migration incrementally.

```
retail_customer_analytics/
├── app.py
├── retail_customer_analysis.py
├── insight_engine.py
├── show_all_results.py
├── data_generator.py
├── src/
│   └── retail_customer_analytics/
│       ├── ingestion.py
│       ├── features.py
│       ├── events.py
│       ├── affinity.py
│       ├── models.py
│       ├── validation.py
│       ├── segmentation.py
│       ├── metrics.py
│       ├── period_manager.py
│       └── metadata.py
├── tests/
│   ├── test_events.py
│   ├── test_features.py
│   ├── test_basket_affinity.py
│   ├── test_models.py
│   ├── test_temporal_validation.py
│   ├── test_resource_limits.py
│   └── test_end_to_end.py
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
└── .github/
    └── workflows/
        └── ci.yml
```

Do not create every module immediately. The first extractions should be driven by the urgent basket and feature-contract problems.

### 8.2 Data ingestion and contracts

Keep `src/retail_customer_analytics/ingestion.py` as the authoritative normalization layer.

Strengthen it by formalizing:

- Required and optional fields.
- Data types and identifier normalization.
- Currency and price/quantity semantics.
- Date-format and timestamp precision requirements.
- Event-identity and transaction-key rules.
- Treatment of missing customer/product IDs.
- Return conventions and matching precision.
- Exact accounting definitions and exclusions.

Add stable schema-version metadata and explicit compatibility handling for uploaded CSV and Parquet data.

A schema-validation failure should identify the missing or invalid fields. Cleaning decisions should be recorded as issue counts and excluded-row counts, not hidden by silent row removal.

### 8.3 Canonical event construction

Create `src/retail_customer_analytics/events.py` and migrate the event-building policy out of the large CLI module.

It should own:

- Customer-day trip construction.
- Per-transaction event construction.
- Transaction-key uniqueness policy.
- Qualifying purchase definitions.
- Basket identity definitions.
- Event timestamps and ordering.
- Return matching and matching-coverage diagnostics.

Both the Streamlit application and CLI should call these functions. Their only differences should be explicit configuration, not independently implemented business rules.

The selected event grain is a business definition. Changing from customer-day trips to transaction events should not silently change historical metrics in an existing report.

### 8.4 Shared basket-affinity engine

The new `src/retail_customer_analytics/affinity.py` should have a small, testable interface.

Conceptually:

```
basket_result = analyze_affinity(    transactions=transactions,    as_of=as_of,    basket_grain=event_grain,    item_level="category",    min_support=0.01,    max_candidates=100,    max_baskets=50_000,    max_items_per_basket=30,    max_pair_operations=1_000_000,    random_seed=42,)
```

The actual parameters can be aligned with the existing CLI. This is an illustrative interface, not a claim that the function exists today.

The engine must:

- Produce stable and correctly keyed baskets.
- Deduplicate repeated item IDs inside each basket.
- Exclude or separately handle returns according to the declared association definition.
- Increment pair counts rather than materializing all pair occurrences.
- Compute support, confidence, lift and applicable statistical diagnostics from aggregated counts.
- Bound candidate and retained-pair counts.
- Report any sampled or incomplete analysis.
- Return structured results and run diagnostics.

Do not use a dense customer-by-product or basket-by-product matrix as an intermediate calculation requirement.

If a sparse binary representation is later shown to improve a specific computation, consider it separately. It must be benchmarked against the incremental counting implementation, and it must not reintroduce a product universe whose size makes the matrix impractical.

### 8.5 Canonical customer features and model contracts

Keep `features.py` as the authoritative feature builder.

The expected contract should explicitly include, where calculable:

- Core RFM, frequency, recency and tenure.
- Rolling-window counts and purchase revenue.
- Cadence and behavioral trends.
- Breadth and category concentration.
- Category affinity and bounded basket features.
- Gross purchase revenue and return-aware net spend.
- Return ratios and matching-coverage measures.
- Price-index proxies with clear missing-data semantics.

Downstream algorithms must declare which fields they require. If a feature is missing, the run should fail early or skip the affected component with a reason instead of failing deep inside clustering or report generation.

Maintain stable column names and dtypes across snapshots.

### 8.6 Configuration and dependencies

The repository currently defines settings in multiple places. Consolidate authoritative defaults for:

- Event grain and transaction key policy.
- Date and currency handling.
- Observation cutoff and forecast horizon.
- Feature windows.
- Basket support, candidate and resource budgets.
- Clustering thresholds.
- Validation origins.
- Random seed.
- Output format and path conventions.

`pyproject.toml`, `requirements.txt`, and `requirements-dev.txt` should have documented responsibilities. Minimum-version requirements such as `pandas>=2.1.0` are not strict version pins, despite the README's description of pinned dependencies.

Recommended policy:

- Keep `pyproject.toml` as the authoritative project and tool configuration.
- Keep production dependency declarations consistent with the install path.
- Add a tested constraints or lock file if exact reproducibility is required.
- Avoid adding a new runtime dependency without a measurable benefit.
- Test the Python versions explicitly declared as supported; do not imply that every declared classifier has been verified in CI.

### 8.7 Testing and CI

The repository already has a test suite and CI. Expand the existing framework rather than rebuilding it.

New or strengthened tests should cover:

1. Exact pair count and lift agreement against a simple reference implementation on small baskets.
2. Cross-customer invoice ID reuse.
3. Empty baskets, duplicate products within a basket, missing identifiers and unusually large baskets.
4. Pair-operation and candidate-budget behavior.
5. Explicit sampled/truncated status and metadata.
6. Peak-memory and runtime tests on deterministic generated datasets.
7. Return-aware feature calculations and canonical output-column contracts.
8. Successful segmentation when the minimum required features exist.
9. Point-in-time feature invariance, including new rolling and trend features.
10. End-to-end runs through CLI and Streamlit-facing helpers with consistent results.

Use deterministic correctness tests in every CI run. Use larger memory benchmarks as a separate regression benchmark or scheduled validation task when their runtime is inappropriate for every pull request.

### 8.8 Performance engineering

Prioritize measurement over a wholesale pandas-to-Polars rewrite.

For each representative dataset, record:

- Input rows and unique customers.
- Unique products and categories.
- Basket count and basket-size distribution.
- Candidate product count and unique retained pair count.
- Elapsed time by pipeline stage.
- Peak process memory.
- Output table sizes.
- Number of completed, skipped, sampled, truncated and failed analyses.

Use these measurements to identify the bottleneck rather than assuming that a single library conversion solves it.

Particular attention should be paid to:

- Global lists of repeated pair occurrences in `app.py`.
- Large customer-by-category allocations in `category_mix()` and `_category_affinity_features()`.
- Repeated joins and materialized copies.
- Temporary CSV serialization between Streamlit and the CLI in `app.py::run_pipeline()`.
- Generated data retained in memory in `data_generator.py`.
- Repeated report generation when the source data and configuration have not changed.

Where useful, reduce intermediate columns, process independent aggregates in chunks, and avoid constructing an object that will immediately be reduced to a small summary.

### 8.9 Artifact management and reproducibility

The CLI already writes tables and metadata. Standardize and enrich the metadata instead of introducing an experiment-tracking platform immediately.

Every run should record:

- Input data identity or a privacy-safe content fingerprint.
- Schema and feature versions.
- Analysis cutoff, forecast horizon and event grain.
- Date parsing and currency settings.
- Cleaning and exclusion counts.
- Model parameters and fitting status.
- Configured and observed computational budgets.
- Random seed and software versions.
- Output artifacts and their schemas.
- Runtime and resource diagnostics.

Avoid exposing absolute local input paths or sensitive row-level customer information in shareable output metadata.

### 8.10 Privacy and responsible use

Minimum requirements for customer-level analytics:

- Minimize collection and output of direct identifiers.
- Use pseudonymous customer identifiers in analytical outputs where feasible.
- Keep customer-level exports separate from aggregated dashboard outputs.
- Clean up temporary work directories and control access to generated files.
- Record data retention and deletion expectations.
- Avoid presenting descriptive segments as immutable customer characteristics.
- Restrict campaign or high-impact decisions from being based solely on a model score without context and appropriate review.
- Make missingness, model fallback, sampling and forecast uncertainty visible to users.

A formal access-control or multi-user deployment architecture is conditional on how the application will be hosted and who will access identifiable customer data.

## 9. Prioritized Roadmap and Backlog

All effort and impact estimates in this section are provisional. They should be updated after profiling representative data and confirming the application's target deployment environment.

Priority definitions

- P0: Correctness or resource-safety defect that should be resolved before scaling the application.
- P1: Foundational reliability or high-value analytical improvement.
- P2: Useful capability or reporting enhancement after the foundation is dependable.
- P3: Conditional extension requiring additional data, business processes, or evidence of need.

### Phase 0 — Quick wins and resource safety

| ID / Priority | Current gap                                                                                                                                    | Proposed change and location                                                                                                        | Expected value                                                                                      | Effort and dependencies                                                               | Computational / maintenance impact                                               | Testable acceptance criteria                                                                                                                                       |
| ------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| R01 / P0      | `app.py::build_basket_associations()` and `_compute_basket_analysis()` materialize global pair lists.                                          | Create a shared bounded analyzer in `src/retail_customer_analytics/affinity.py`; replace both implementations and their call sites. | Eliminates the main avoidable basket RAM bottleneck and reduces duplicated logic.                   | Medium. Existing CLI affinity logic and its budget parameters are the starting point. | Lowers intermediate memory; adds centralized budget and status logic.            | Exact small-data agreement; no global pair-occurrence list; candidate and pair budgets enforced; status and coverage recorded for incomplete runs.                 |
| R02 / P0      | UI basket builders use `transaction_id` alone as a grouping key.                                                                               | Use the shared event/basket-key policy in `affinity.py` and `events.py`.                                                            | Prevents different customers' baskets from being merged if transaction IDs are not globally unique. | Small–Medium. Requires the explicit key policy.                                       | Low runtime cost; reduces inconsistent results.                                  | Fixture with a reused invoice ID across two customers produces separate baskets under customer-transaction mode.                                                   |
| R03 / P0      | `features.py` sets `net_spend = gross_spend`, and downstream clustering expects `return_value_ratio` feature not supplied by the canonical builder. | Repair `src/retail_customer_analytics/features.py`; add `return_value_ratio` (0.0 when no returns) and maintain `net_spend` semantic distinction. Audit `retail_customer_analysis.py::cluster_customers()` and `run_analysis()`.  | Restores correct customer financial features and reliable segmentation.                             | Medium. Requires adding return_value_ratio with default 0.0.                           | Minor aggregation cost; reduces failures and incorrect financial interpretation. | Required feature columns exist (return_value_ratio=0.0); clustering handles valid and degenerate cases explicitly. |
| R04 / P0      | Existing test suite does not yet establish the required large-data basket resource contract.                                                   | Extend `tests/test_basket_affinity.py`; add `tests/test_resource_limits.py`; extend `.github/workflows/ci.yml`.                     | Prevents the RAM problem from returning during refactoring.                                         | Small–Medium. Depends on R01.                                                         | Small CI overhead for correctness tests; benchmark tests may run separately.     | Small-data exactness, candidate-budget, operation-budget, deterministic-sampling and incomplete-status tests pass.                                                 |
| R05 / P1      | The UI may duplicate basket work when generating charts and tables.                                                                            | Update the basket sections and call sites in `app.py` to consume a single affinity result.                                          | Removes redundant computation and improves latency.                                                 | Small–Medium. Depends on R01.                                                         | Lowers runtime and maintenance burden.                                           | One user-selected analysis configuration invokes the shared analyzer once; table and chart use the same result.                                                    |
| R06 / P1      | Resource limits and incomplete coverage are not consistent across implementations.                                                             | Share affinity configuration and run metadata through the CLI and Streamlit pipeline.                                               | Makes memory use and analytical coverage predictable.                                               | Small. Depends on R01 and R05.                                                        | Negligible computation; introduces a stable result contract.                     | Run metadata declares all relevant budgets, candidate counts, coverage and final status.                                                                           |

### Phase 1 — Foundational reliability

| ID / Priority | Current gap                                                                                                             | Proposed change and location                                                                                                                                                             | Expected value                                                                   | Effort and dependencies                                | Computational / maintenance impact                                      | Testable acceptance criteria                                                                                                                 |
| ------------- | ----------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- | ------------------------------------------------------ | ----------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| R07 / P1      | Feature computation and model inputs do not have a single enforced feature contract.                                    | Define expected features and validations in `src/retail_customer_analytics/features.py` and the relevant model/segmentation modules.                                                     | Prevents silent loss of downstream capabilities after refactors.                 | Medium. Depends on R03.                                | Small validation overhead; reduces debugging effort.                    | Valid feature snapshots contain required fields; missing fields produce actionable skip/failure status.                                      |
| R08 / P1      | Feature and event logic remains distributed between legacy functions and the canonical package.                         | Introduce `src/retail_customer_analytics/events.py`; gradually move event definitions out of the CLI and UI.                                                                             | Ensures consistent frequency, recency, and basket denominators.                  | Medium. Depends on R01–R03.                            | Lower duplicate maintenance; small refactor risk.                       | The same input and configuration produce identical event keys, counts and eligible-basket totals across entry points.                        |
| R09 / P1      | Dense category-mix allocations can grow with customer count even when category count is capped.                         | Profile `retail_customer_analysis.py::category_mix()` and `features.py::_category_affinity_features()`; compute aggregates in long form and avoid unnecessary dense intermediate arrays. | Extends scalability beyond the basket calculation.                               | Medium. Depends on a representative-data benchmark.    | Lower peak memory; may require modest implementation changes.           | Measurements document baseline and improved peak RAM; numerical outputs agree with a reference on small data.                                |
| R10 / P1      | The existing rolling-origin evaluation needs consistent reporting of performance across origins.                        | Improve validation output in `retail_customer_analysis.py` and reporting in `insight_engine.py`.                                                                                         | Reduces reliance on a single period and exposes temporal instability.            | Medium. Rolling-origin support already exists.         | Runtime grows with number of origins; no new model dependency required. | Each origin records cutoff, horizon, sample size, per-target metric and model status; HTML output shows variation across origins.            |
| R11 / P1      | `features.py` trend calculations omit months with zero recorded purchasing activity from the observed-month regression. | Correct monthly trend construction and add feature-specific tests in `src/retail_customer_analytics/features.py`.                                                                        | Makes declining activity more interpretable and improves lapse-related features. | Small–Medium. Depends on existing point-in-time tests. | Modest extra aggregation; no new dependency.                            | A customer with long inactive gaps has a different, correctly defined trend from a customer purchasing continuously across the same horizon. |
| R12 / P1      | Dependency declarations are not strictly pinned and documented requirements can overstate reproducibility.              | Reconcile `pyproject.toml`, `requirements.txt`, `requirements-dev.txt` and CI configuration.                                                                                             | Makes analysis environments easier to reproduce and maintain.                    | Small.                                                 | Low maintenance cost.                                                   | A clean environment installs from the documented route; CI verifies supported Python versions and dependencies.                              |

### Phase 2 — Core analytical enhancements

| ID / Priority | Current gap                                                                                           | Proposed change and location                                                                                                                                          | Expected value                                                                     | Effort and dependencies                                                                    | Computational / maintenance impact                               | Testable acceptance criteria                                                                                                               |
| ------------- | ----------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ | ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| R13 / P1      | Forecast comparison needs systematic robustness reporting and feature ablation.                       | Extend validation artifacts and tests for `retail_customer_analysis.py` and `insight_engine.py`.                                                                      | Shows whether additional feature families or models provide real predictive value. | Medium. Depends on R07 and R10.                                                            | More evaluation runtime; low inference complexity.               | Baselines and ablations are compared on identical origins; metric definitions and denominators are documented.                             |
| R14 / P1      | Segment usefulness must remain distinguishable from clustering quality.                               | Strengthen cross-snapshot stability and segment-profile diagnostics using existing clustering outputs.                                                                | Reduces unstable or redundant segment definitions.                                 | Small–Medium. Existing silhouette, bootstrap and cross-snapshot machinery provides a base. | Small additional reporting cost.                                 | Reports include segment sizes, behavior profiles, stability measures and explicit selection reasons.                                       |
| R15 / P1      | Basket affinity results need time-period and resource diagnostics alongside statistical significance. | Add support counts, budget coverage, persistent-association summaries and date-range information to the affinity artifact.                                            | Makes product-association outputs easier to evaluate and interpret.                | Small–Medium. Depends on R01 and R06.                                                      | Low cost; improves output quality.                               | Associations report denominator definitions and sample coverage; small fixtures reproduce exact metrics.                                   |
| R16 / P2      | Customer value reporting is based on revenue and may be confused with profitability.                  | Strengthen naming, report wording and documentation in `insight_engine.py` and `README.md`; add a separate contribution target only when cost data becomes available. | Prevents misleading commercial interpretation.                                     | Small initially; large for a true contribution model.                                      | Negligible for documentation; higher for new financial modeling. | No output calls revenue profit; any contribution forecast has separate definitions, data requirements and validation.                      |
| R17 / P2      | Existing diagnostic views do not yet provide a consolidated model/feature quality panel.              | Add forecast error, calibration where appropriate, missingness, model-status and feature-drift charts.                                                                | Helps users distinguish poor models from data problems.                            | Medium. Depends on standardized metadata.                                                  | Low-to-moderate reporting cost.                                  | Charts have correct denominators, explicit target definitions, and skipped reasons when data is insufficient.                              |
| R18 / P2      | Run artifacts need a canonical schema and input identity.                                             | Add `src/retail_customer_analytics/metadata.py` and standardize `run_metadata.json`.                                                                                  | Improves traceability and reproducibility.                                         | Medium. Depends on configuration and output-contract decisions.                            | Minimal analytical runtime; adds metadata.                       | Each run records schema version, cutoff, horizon, event grain, seed, versions, status and outputs without leaking unnecessary local paths. |

### Phase 3 — Optional capabilities

| ID / Priority | Current gap                                                           | Proposed change and location                                                                     | Expected value                                                                  | Effort and dependencies                            | Computational / maintenance impact                                   | Testable acceptance criteria                                                                                       |
| ------------- | --------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------- | -------------------------------------------------- | -------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| R19 / P2–P3   | No completed supervised purchase-propensity model.                    | Add optional regularized logistic regression in `src/retail_customer_analytics/models.py`.       | Provides next-horizon probability estimates for targeted customer selection.    | Medium. Depends on R07, R10 and R13.               | Adds model fitting and possibly an optional scikit-learn dependency. | Out-of-time comparison beats a simple baseline on predefined metrics; probabilities are evaluated for calibration. |
| R20 / P3      | Multi-item recommendations beyond pair association are not validated. | Consider bounded frequent-itemset mining only after shared pair counts are correct and scalable. | May reveal bundle opportunities when pairwise affinity is insufficient.         | Medium–Large. Depends on R01 and R15.              | Potentially high candidate-growth and maintenance cost.              | Candidate budget is enforced; patterns reproduce on small fixtures and persist in later data.                      |
| R21 / P3      | No experimental framework for evaluating actual action impact.        | Define a controlled pilot measurement workflow once treatment and outcome data exist.            | Measures incremental outcomes rather than relying on observational association. | Large. Requires operational and experimental data. | Additional data process and analysis requirements.                   | A prespecified outcome, comparison group, treatment cost and uncertainty estimate are reported.                    |

### Recommended execution order

1. First: R01–R04, eliminating unbounded UI basket calculations, repairing the canonical feature contract, and adding resource regression tests.
2. Second: R05–R09, unifying outputs and event identity while measuring and improving remaining memory-heavy feature paths.
3. Third: R10–R15, strengthening evaluation, feature robustness, segmentation and affinity reporting.
4. Fourth: R16–R18, improving business interpretation, dashboards and run provenance.
5. Later: R19–R21, conditional on the business need and supporting data.

## 10. First Five Recommended Changes

These are the five changes with the greatest near-term value, balancing the explicit RAM concern with correctness and maintainability.

### 1. Eliminate unbounded basket-pair materialization from Streamlit

Location: `app.py::build_basket_associations()`, `app.py::_compute_basket_analysis()`, new `src/retail_customer_analytics/affinity.py`.

Replace the global list of pair occurrences and subsequent DataFrame aggregation with incremental counting. Reuse the CLI's candidate limits, basket limits and pair-operation budget.

Why first: This is the most directly verified path that can cause the user's reported memory failure. It also removes duplicated implementations.

Acceptance criteria:

- No global list containing every pair occurrence is constructed.
- Both Streamlit and CLI use the same analyzer and basket-key policy.
- Candidate count, pair-operation count and retained pair count respect explicit budgets.
- Peak memory and runtime are measured on small and large synthetic datasets.
- Incomplete processing is labeled and reported.
- Exact association counts agree with a reference implementation on small fixtures.

Provisional effort: Medium. Expected impact: High.

### 2. Repair canonical customer feature contract (net spend, return_value_ratio)

Location: `src/retail_customer_analytics/features.py::build_customer_snapshot()`, `retail_customer_analysis.py::cluster_customers()`, and `retail_customer_analysis.py::run_analysis()`.

The canonical feature builder currently sets `net_spend = gross_spend` and does not supply the `return_value_ratio` feature expected by downstream clustering. While the current dataset has no returns (all quantity > 0), the feature builder should maintain the semantic distinction: `net_spend = gross_spend - return_value` (where `return_value = 0`) and include `return_value_ratio = 0.0`.

Why second: Missing required features can make segmentation fail without aborting the entire pipeline. The semantic distinction between gross and net spend should be maintained for future extensibility.

Acceptance criteria:

- `net_spend` maintains semantic distinction from `gross_spend` (net = gross - returns, with returns = 0 for current data).
- `return_value_ratio` feature is present with value 0.0.
- Features honor the selected snapshot cutoff.
- Required clustering and reporting features are present or generate explicit skip reasons.
- Valid synthetic fixtures complete clustering successfully where minimum eligibility requirements are met.

Provisional effort: Medium. Expected impact: High.

### 3. Add a dedicated resource and basket-correctness regression suite

Location: `tests/test_basket_affinity.py`, new `tests/test_resource_limits.py`, `.github/workflows/ci.yml`.

Protect the new implementation against the specific failure mode instead of relying on general end-to-end smoke tests.

Acceptance criteria:

- Exact small-data pair counts, support, confidence and lift are tested.
- Large-basket behavior respects all defined budgets.
- A fixed seed produces reproducible sampling.
- Cross-customer invoice reuse is handled correctly.
- Output statuses accurately distinguish completed, sampled, truncated, skipped and failed analyses.
- Peak RAM and runtime are measured in a repeatable benchmark procedure.

Provisional effort: Small–Medium. Expected impact: High.

### 4. Consolidate event and analytical output definitions

Location: New `src/retail_customer_analytics/events.py`; existing `src/retail_customer_analytics/ingestion.py` and `features.py`; `app.py`; `retail_customer_analysis.py`.

Ensure the same event grain, customer identity and monetary definitions are used in scoring, product affinity, retention, segment reporting and UI summaries.

Acceptance criteria:

- Identical input and configuration yield equivalent event counts in the CLI and Streamlit-facing analytics.
- Transaction ID reuse follows the explicit key policy.
- Purchase counts and denominators are consistent across basket and customer analytics.
- Feature and event outputs have stable column definitions.
- Accounting totals reconcile between source transactions and derived summaries.

Provisional effort: Medium. Expected impact: High.

### 5. Improve rolling-origin model evaluation and reporting

Location: `retail_customer_analysis.py::rolling_origin_validation()`, `holdout_validation()`, and `insight_engine.py`.

Build on the existing temporal-validation implementation. Display per-origin metrics and baseline comparisons instead of relying on a single overall performance summary.

Acceptance criteria:

- Every origin records its cutoff and future evaluation horizon.
- All model-target pairs use the same eligible evaluation population.
- Count, revenue and probability metrics are clearly distinguished.
- Failed and fallback model states are visible.
- Final report charts show variability across origins and include sample counts and uncertainty where appropriate.
- Feature ablations use identical evaluation origins.

Provisional effort: Medium. Expected impact: High.

## 11. Assumptions, Limitations, and Open Questions

### 11.1 Assumptions used in this plan

- The main dataset is line-item retail transaction data in CSV or Parquet format, using the documented customer, transaction, product, category, date, price and quantity fields.
- **Current dataset has no returns (all quantity > 0).** The codebase supports returns via negative quantities for future extensibility.
- Retail forecasting refers to expected future purchase counts and revenue over a declared horizon unless further financial data becomes available.
- Event grain can be defined as customer-day trips or transaction-level events.
- The application should remain lightweight and usable without mandatory cloud infrastructure or distributed computing.
- Existing analytical implementations should be preserved where they are sound and improve maintainability or reliability.

### 11.2 Findings requiring execution or additional evidence

Peak memory: The source code confirms unbounded pair-occurrence materialization in the UI basket implementations. It does not establish the exact peak RAM or minimum dataset size at which the application will fail. Benchmarking is needed to quantify the improvement after refactoring.

Feature and clustering behavior: The canonical feature contract does not currently match the return-related inputs expected by clustering. A regression test should establish the actual end-to-end effect in the current environment and confirm that the corrected pipeline produces valid segment results.

Forecast suitability: BG/NBD and Gamma-Gamma are already implemented, but their commercial suitability cannot be established from source inspection alone. It requires representative transaction history, sufficient observation time, model-status information and honest future-period evaluation.

Customer-level category representation: The category-mix and affinity features have bounded category dimensions, but customer-related arrays and feature tables can still be expensive at very high customer counts. The required optimization depends on real customer count, category cardinality, available RAM and the analytical importance of category mix to clustering.

Business value: Without a stated business decision and outcome measure, it is not possible to determine whether a supervised model would improve on the existing analytics. Its adoption should depend on predefined evaluation criteria rather than a general preference for more algorithms.

### 11.3 Open business and data questions

The following questions should be resolved as part of implementation, without blocking the immediate RAM and feature-contract fixes:

1. What exactly defines a basket? Is it an invoice, a customer-day trip, or another purchase occasion? The choice affects support, confidence and event counts.
2. Are transaction IDs globally unique? If not, should every transaction key be customer-scoped?
3. **Return handling is not needed for current dataset (no returns). What is the return-accounting policy if returns are added in future data?** Is the target based on return date or original purchase date, how should partial returns be assigned, and what is the treatment of unmatched returns?
4. What is the primary commercial objective? Repeat purchase, retention, revenue forecasting, product affinity, or campaign targeting?
5. How large is the intended input? Expected rows, unique customers/products, largest baskets and available memory determine appropriate candidate and resource budgets.
6. Are margin, product cost, promotion exposure and intervention outcomes available? These determine whether profitability forecasts, price-response analysis or causal action measurement are justified.
7. What runtime and coverage guarantees are needed? Some users may prefer an exact analysis of a smaller candidate universe, while others may prefer a reproducible sampled analysis at larger scale.

### 11.4 Final recommendation

The project has enough analytical breadth to justify concentrating on correctness and resource use before adding more modeling techniques.

The first implementation milestone should deliver three concrete outcomes: a bounded shared basket-affinity engine that eliminates global pair-occurrence materialization, a correct and complete canonical customer-feature contract, and automated tests that verify numerical correctness and resource limits.

Once those are stable, invest in consistent point-in-time features, rolling-origin evaluation, interpretable segment diagnostics, and better business-facing reports. Add supervised propensity or more advanced affinity methods only when they answer a defined business question and beat the existing, computationally efficient baselines.

### References

- [Retail Customer Analytics repository](https://github.com/hrczggyrgy/retail_customer_analytics)
- [BG/NBD — Counting Your Customers the Easy Way](https://www.brucehardie.com/papers/018/fader_et_al_mksc_05.pdf)
- [Gamma-Gamma customer-value model reference](https://www.brucehardie.com/papers/020/gg_nbd.pdf)
- [scikit-learn TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
- [scikit-learn model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [scikit-learn probability calibration](https://scikit-learn.org/stable/modules/calibration.html)
- [pandas sparse data structures](https://pandas.pydata.org/docs/user_guide/sparse.html)
- [NIST Privacy Framework](https://www.nist.gov/privacy-framework)