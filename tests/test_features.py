"""Tests for canonical feature builder (features.py)."""

import pandas as pd
import pytest

from retail_customer_analytics.features import build_customer_snapshot


class TestBuildCustomerSnapshot:
    """Tests for the canonical feature builder."""

    def _make_transactions(self):
        """Create sample transaction data with returns and multiple categories."""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        data = []
        for i, d in enumerate(dates):
            data.append(
                {
                    "customer_id": "C1",
                    "transaction_id": f"T{i}",
                    "transaction_day": d,
                    "transaction_date": d + pd.Timedelta(hours=10),
                    "product_id": f"P{i % 3}",
                    "product_description": f"Product {i % 3}",
                    "category": "CatA" if i % 3 == 0 else "CatB",
                    "department": "Dept1",
                    "price": 10.0 + (i % 3) * 5.0,
                    "quantity": 1,
                    "line_revenue": (10.0 + (i % 3) * 5.0),
                }
            )
        # Add second customer
        for i, d in enumerate(dates[:5]):
            data.append(
                {
                    "customer_id": "C2",
                    "transaction_id": f"T{i + 10}",
                    "transaction_day": d,
                    "transaction_date": d + pd.Timedelta(hours=14),
                    "product_id": f"P{i}",
                    "product_description": f"Product {i}",
                    "category": "CatB",
                    "department": "Dept2",
                    "price": 20.0,
                    "quantity": 2,
                    "line_revenue": 40.0,
                }
            )
        return pd.DataFrame(data)

    def test_build_snapshot_returns_dataframe(self):
        """build_customer_snapshot returns a DataFrame with one row per customer."""
        transactions = self._make_transactions()
        as_of = pd.Timestamp("2024-01-10")

        cust = build_customer_snapshot(transactions, as_of)

        assert isinstance(cust, pd.DataFrame)
        assert len(cust) == 2  # C1 and C2
        assert "C1" in cust.index
        assert "C2" in cust.index

    def test_core_rfm_features_present(self):
        """Core RFM features are present and correctly computed."""
        transactions = self._make_transactions()
        as_of = pd.Timestamp("2024-01-10")

        cust = build_customer_snapshot(transactions, as_of)

        # C1: 8 purchases, 2 returns, last purchase Jan 8, first Jan 1
        assert "n_trips" in cust.columns
        assert "recency_days" in cust.columns
        assert "gross_spend" in cust.columns
        assert "net_spend" in cust.columns
        assert "x" in cust.columns
        assert "T" in cust.columns
        assert "heuristic_inactive_flag" in cust.columns

    def test_rolling_window_features_present(self):
        """Rolling window features are present."""
        transactions = self._make_transactions()
        as_of = pd.Timestamp("2024-01-10")

        cust = build_customer_snapshot(transactions, as_of, windows_days=(30, 90))

        assert "gross_purchase_revenue_30d" in cust.columns
        assert "trips_30d" in cust.columns
        assert "gross_purchase_revenue_90d" in cust.columns

    def test_breadth_features_present(self):
        """Breadth features (distinct products/categories) are present."""
        transactions = self._make_transactions()
        as_of = pd.Timestamp("2024-01-10")

        cust = build_customer_snapshot(transactions, as_of)

        assert "unique_products_purchased" in cust.columns
        assert "unique_categories_purchased" in cust.columns
        assert "unique_departments_purchased" in cust.columns
        assert "top_category_by_spend" in cust.columns
        assert "category_spend_hhi" in cust.columns

    def test_promo_features_present(self):
        """Price-index promo proxy features are present."""
        transactions = self._make_transactions()
        as_of = pd.Timestamp("2024-01-10")

        cust = build_customer_snapshot(transactions, as_of)

        assert "promo_spend_share" in cust.columns
        assert "mean_price_index" in cust.columns
        assert "price_index_median" in cust.columns

    def test_category_affinity_features_present(self):
        """Category affinity features are present."""
        transactions = self._make_transactions()
        as_of = pd.Timestamp("2024-01-10")

        cust = build_customer_snapshot(transactions, as_of)

        # Should have spend share columns for each category
        cat_cols = [
            c for c in cust.columns if c.startswith("category_") and c.endswith("_spend_share")
        ]
        assert len(cat_cols) >= 1
        assert "top_category_by_spend_affinity" in cust.columns
        assert "category_entropy" in cust.columns

    def test_trend_features_present(self):
        """Trend features are present."""
        transactions = self._make_transactions()
        as_of = pd.Timestamp("2024-01-10")

        cust = build_customer_snapshot(transactions, as_of)

        assert "monthly_revenue_slope" in cust.columns
        assert "monthly_revenue_cv" in cust.columns
        assert "active_months_fraction" in cust.columns

    def test_basket_composition_features_present(self):
        """Basket composition features are present."""
        transactions = self._make_transactions()
        as_of = pd.Timestamp("2024-01-10")

        cust = build_customer_snapshot(transactions, as_of)

        assert "mean_distinct_products_per_trip" in cust.columns
        assert "mean_distinct_categories_per_trip" in cust.columns
        assert "mean_units_per_trip" in cust.columns
        assert "trip_revenue_cv" in cust.columns

    def test_event_grain_transaction(self):
        """transaction grain keeps separate transactions."""
        transactions = self._make_transactions()
        as_of = pd.Timestamp("2024-01-10")

        cust = build_customer_snapshot(transactions, as_of, event_grain="transaction")

        # C1 has 10 positive transactions -> 10 trips
        assert cust.loc["C1", "n_trips"] == 10

    def test_event_grain_trip(self):
        """trip grain merges same-day transactions."""
        # Create transactions with multiple per day
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        data = []
        for i, d in enumerate(dates):
            data.append(
                {
                    "customer_id": "C1",
                    "transaction_id": f"T{i}a",
                    "transaction_day": d,
                    "transaction_date": d + pd.Timedelta(hours=10),
                    "product_id": "P1",
                    "category": "CatA",
                    "department": "Dept1",
                    "price": 10.0,
                    "quantity": 1,
                    "line_revenue": 10.0,
                }
            )
            data.append(
                {
                    "customer_id": "C1",
                    "transaction_id": f"T{i}b",
                    "transaction_day": d,
                    "transaction_date": d + pd.Timedelta(hours=14),
                    "product_id": "P2",
                    "category": "CatB",
                    "department": "Dept1",
                    "price": 20.0,
                    "quantity": 1,
                    "line_revenue": 20.0,
                }
            )
        transactions = pd.DataFrame(data)
        as_of = pd.Timestamp("2024-01-03")

        cust = build_customer_snapshot(transactions, as_of, event_grain="trip")

        # 3 days -> 3 trips
        assert cust.loc["C1", "n_trips"] == 3

    def test_point_in_time_cutoff(self):
        """Features only use data up to as_of date."""
        transactions = self._make_transactions()
        as_of = pd.Timestamp("2024-01-05")  # Only first 5 days

        cust = build_customer_snapshot(transactions, as_of)

        # C1 should have 5 purchases (Jan 1-5), not 8
        assert cust.loc["C1", "n_trips"] == 5

    def test_point_in_time_cutoff_with_timestamps(self):
        """Regression test: cutoff date with timestamps uses transaction_day for inclusion.

        Bug: _return_features() was using transaction_date <= as_of for filtering,
        which excluded purchases on the cutoff date after midnight when as_of is normalized.
        This test ensures that purchases on the cutoff date are included consistently
        across all features, regardless of time-of-day.
        """
        # Create a purchase on Jan 10 at 10:00 AM
        data = [
            {
                "customer_id": "C1",
                "transaction_id": "T1",
                "transaction_day": pd.Timestamp("2024-01-10"),
                "transaction_date": pd.Timestamp("2024-01-10 10:00:00"),
                "product_id": "P1",
                "product_description": "Product 1",
                "category": "CatA",
                "department": "Dept1",
                "price": 10.0,
                "quantity": 1,
                "line_revenue": 10.0,
            },
            {
                "customer_id": "C1",
                "transaction_id": "T2",
                "transaction_day": pd.Timestamp("2024-01-09"),
                "transaction_date": pd.Timestamp("2024-01-09 14:00:00"),
                "product_id": "P2",
                "product_description": "Product 2",
                "category": "CatB",
                "department": "Dept1",
                "price": 20.0,
                "quantity": 1,
                "line_revenue": 20.0,
            },
        ]
        transactions = pd.DataFrame(data)

        # Cutoff at Jan 10 00:00 (midnight) - should include the Jan 10 purchase
        as_of = pd.Timestamp("2024-01-10")

        cust = build_customer_snapshot(transactions, as_of)

        # Both purchases should be included (Jan 9 and Jan 10)
        assert cust.loc["C1", "n_trips"] == 2
        assert cust.loc["C1", "gross_spend"] == 30.0

        # gross_purchase_revenue is now in rolling window features
        assert cust.loc["C1", "gross_purchase_revenue_30d"] == 30.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
