"""Tests for metrics computation and bootstrap reproducibility."""
import pytest
import numpy as np
import pandas as pd

import retail_customer_analysis as rca


class TestComputeMetrics:
    """Tests for _compute_metrics function."""

    def test_metrics_basic(self):
        """Basic metrics are computed correctly."""
        pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([1.2, 1.8, 3.1, 3.9, 5.2])
        bought = np.array([True, True, True, True, True])
        
        result = rca._compute_metrics("test_model", "trips_in_horizon", pred, y, bought, seed=42)
        
        assert result["model"] == "test_model"
        assert result["target"] == "trips_in_horizon"
        assert result["n_customers"] == 5
        assert result["mae"] is not None
        assert result["rmse"] is not None
        assert result["wape"] is not None
        assert result["aggregate_bias"] is not None
        assert result["spearman"] is not None
        assert result["auc_any_purchase"] is not None
        assert result["predicted_total"] == pytest.approx(15.0)
        assert result["actual_total"] == pytest.approx(15.2)

    def test_metrics_with_nan_predictions(self):
        """NaN predictions are handled correctly."""
        pred = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        y = np.array([1.2, 1.8, 3.1, 3.9, 5.2])
        bought = np.array([True, True, True, True, True])
        
        result = rca._compute_metrics("test_model", "trips_in_horizon", pred, y, bought, seed=42)
        
        assert result["n_customers"] == 4  # One NaN excluded

    def test_metrics_all_nan_predictions(self):
        """All NaN predictions returns zero customers."""
        pred = np.array([np.nan, np.nan, np.nan])
        y = np.array([1.0, 2.0, 3.0])
        bought = np.array([True, True, True])
        
        result = rca._compute_metrics("test_model", "trips_in_horizon", pred, y, bought, seed=42)
        
        assert result["n_customers"] == 0
        assert result["mae"] is None

    def test_binary_target_auc(self):
        """AUC is computed for binary target."""
        pred = np.array([0.1, 0.4, 0.35, 0.8, 0.9])
        y = np.array([1.0, 0.0, 1.0, 0.0, 1.0])  # Not used for AUC
        bought = np.array([False, False, True, True, True])
        
        result = rca._compute_metrics("test_model", "any_purchase_in_horizon", pred, y, bought, seed=42)
        
        assert result["auc_any_purchase"] is not None
        assert 0.0 <= result["auc_any_purchase"] <= 1.0

    def test_bootstrap_reproducibility(self):
        """Repeated calls with same seed produce identical confidence intervals."""
        np.random.seed(123)
        n = 100
        pred = np.random.randn(n) * 2 + 5
        y = pred + np.random.randn(n) * 0.5
        bought = y > 5
        
        result1 = rca._compute_metrics("test_model", "trips_in_horizon", pred, y, bought, seed=42)
        result2 = rca._compute_metrics("test_model", "trips_in_horizon", pred, y, bought, seed=42)
        
        assert result1["mae_ci95_lower"] == pytest.approx(result2["mae_ci95_lower"])
        assert result1["mae_ci95_upper"] == pytest.approx(result2["mae_ci95_upper"])
        assert result1["bootstrap_seed"] == 42
        assert result2["bootstrap_seed"] == 42

    def test_different_seeds_produce_different_cis(self):
        """Different seeds produce different confidence intervals (with high probability)."""
        np.random.seed(123)
        n = 100
        pred = np.random.randn(n) * 2 + 5
        y = pred + np.random.randn(n) * 0.5
        bought = y > 5
        
        result1 = rca._compute_metrics("test_model", "trips_in_horizon", pred, y, bought, seed=42)
        result2 = rca._compute_metrics("test_model", "trips_in_horizon", pred, y, bought, seed=123)
        
        # Very unlikely to be exactly equal with different seeds
        assert result1["mae_ci95_lower"] != result2["mae_ci95_lower"] or \
               result1["mae_ci95_upper"] != result2["mae_ci95_upper"]

    def test_revenue_target_no_bought(self):
        """Revenue target works without binary bought array."""
        pred = np.array([100.0, 200.0, 150.0])
        y = np.array([110.0, 190.0, 160.0])
        
        result = rca._compute_metrics("test_model", "revenue_in_horizon", pred, y, seed=42)
        
        assert result["target"] == "revenue_in_horizon"
        assert result["auc_any_purchase"] is None
        assert result["mae"] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])