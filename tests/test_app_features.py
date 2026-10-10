"""Tests for app.py feature builder wrapper."""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import app


class TestAppBuildCustomerFeatures:
    """Tests for app.py's build_customer_features wrapper."""

    def _make_transactions(self):
        """Create sample transaction data."""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        data = []
        for i, d in enumerate(dates):
            data.append({
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
                "revenue": 10.0 + (i % 3) * 5.0,
            })
        # Add second customer
        for i, d in enumerate(dates[:5]):
            data.append({
                "customer_id": "C2",
                "transaction_id": f"T{i+10}",
                "transaction_day": d,
                "transaction_date": d + pd.Timedelta(hours=14),
                "product_id": f"P{i}",
                "product_description": f"Product {i}",
                "category": "CatB",
                "department": "Dept2",
                "price": 20.0,
                "quantity": 2,
                "revenue": 40.0,
            })
        return pd.DataFrame(data)

    def test_build_features_returns_dataframe(self):
        """build_customer_features returns a DataFrame with expected columns."""
        transactions = self._make_transactions()
        as_of = pd.Timestamp("2024-01-10")
        
        cust = app.build_customer_features(transactions, as_of)
        
        assert isinstance(cust, pd.DataFrame)
        assert len(cust) == 2  # C1 and C2
        
    def test_required_columns_present(self):
        """Required txn_ prefixed columns are present."""
        transactions = self._make_transactions()
        as_of = pd.Timestamp("2024-01-10")
        
        cust = app.build_customer_features(transactions, as_of)
        
        required = [
            "customer_id", "txn_frequency", "txn_monetary", "txn_recency_days",
            "txn_first_purchase", "txn_last_purchase", "txn_tenure_days",
            "txn_aov", "txn_distinct_products", "txn_distinct_categories",
            "txn_distinct_departments", "segment_rfm", "segment_value_tier",
            "lifecycle_status", "txn_rfm_score"
        ]
        
        for col in required:
            assert col in cust.columns, f"Missing column: {col}"

    def test_segment_rfm_values(self):
        """RFM segments have expected values."""
        transactions = self._make_transactions()
        as_of = pd.Timestamp("2024-01-10")
        
        cust = app.build_customer_features(transactions, as_of)
        
        # C1 has 10 purchases, C2 has 5
        # Both should have valid segments
        assert cust["segment_rfm"].notna().all()
        assert set(cust["segment_rfm"]).issubset({
            "Champions", "Loyal Customers", "New Customers", "Potential Loyalists",
            "At Risk High Value", "At Risk", "Hibernating", "Needs Attention"
        })

    def test_value_tier_values(self):
        """Value tier has expected categories."""
        transactions = self._make_transactions()
        as_of = pd.Timestamp("2024-01-10")
        
        cust = app.build_customer_features(transactions, as_of)
        
        assert cust["segment_value_tier"].notna().all()
        assert set(cust["segment_value_tier"].astype(str)).issubset({
            "Top 1%", "Next 4%", "Next 5%", "Next 10%", "Middle 50%", "Bottom 30%"
        })

    def test_lifecycle_status_present(self):
        """Lifecycle status is computed."""
        transactions = self._make_transactions()
        as_of = pd.Timestamp("2024-01-10")
        
        cust = app.build_customer_features(transactions, as_of)
        
        assert "lifecycle_status" in cust.columns
        assert cust["lifecycle_status"].notna().all()

    def test_invoice_level_frequency(self):
        """Frequency should be at transaction_id level (invoice-level)."""
        # Create data with multiple items per transaction
        data = [
            {
                "customer_id": "C1",
                "transaction_id": "T1",
                "transaction_day": pd.Timestamp("2024-01-01"),
                "transaction_date": pd.Timestamp("2024-01-01 10:00"),
                "product_id": "P1",
                "category": "CatA",
                "department": "Dept1",
                "price": 10.0,
                "quantity": 1,
                "revenue": 10.0,
            },
            {
                "customer_id": "C1",
                "transaction_id": "T1",  # Same transaction
                "transaction_day": pd.Timestamp("2024-01-01"),
                "transaction_date": pd.Timestamp("2024-01-01 10:00"),
                "product_id": "P2",
                "category": "CatB",
                "department": "Dept1",
                "price": 20.0,
                "quantity": 1,
                "revenue": 20.0,
            },
            {
                "customer_id": "C1",
                "transaction_id": "T2",  # Different transaction same day
                "transaction_day": pd.Timestamp("2024-01-01"),
                "transaction_date": pd.Timestamp("2024-01-01 14:00"),
                "product_id": "P1",
                "category": "CatA",
                "department": "Dept1",
                "price": 10.0,
                "quantity": 1,
                "revenue": 10.0,
            },
        ]
        transactions = pd.DataFrame(data)
        as_of = pd.Timestamp("2024-01-01")
        
        cust = app.build_customer_features(transactions, as_of)
        
        # Should have 2 transactions (T1 and T2) for frequency
        assert cust.loc[cust["customer_id"] == "C1", "txn_frequency"].iloc[0] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])