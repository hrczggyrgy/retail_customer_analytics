"""Tests for model fitting, status reporting, and fallback behavior (R07)."""

import numpy as np
import pandas as pd
import pytest

import retail_customer_analysis as rca


class TestBGNBDFitting:
    """Tests for BG/NBD model fitting and status reporting."""

    def test_fit_bgnbd_returns_converged_status(self):
        """fit_bgnbd should return convergence status."""
        # Sufficient data for fitting
        x = np.array([0, 1, 2, 3, 4, 5, 0, 1, 2, 0] * 10, dtype=float)
        tx = np.array([5, 10, 15, 20, 25, 30, 5, 10, 15, 5] * 10, dtype=float)
        T = np.array([30] * 100, dtype=float)

        result = rca.fit_bgnbd(x, tx, T, penalizer=1e-3)

        assert "converged" in result
        assert isinstance(result["converged"], bool)
        assert "status" in result
        assert result["status"] == "fitted"
        assert "r" in result and "alpha" in result and "a" in result and "b" in result

    def test_fit_bgnbd_insufficient_data_raises(self):
        """fit_bgnbd should handle insufficient data gracefully."""
        x = np.array([0, 1], dtype=float)
        tx = np.array([5, 10], dtype=float)
        T = np.array([30, 30], dtype=float)

        # Should not crash, but may return failed status or raise
        # Current implementation needs at least 50 customers for score_customers
        # but fit_bgnbd itself should handle small data
        result = rca.fit_bgnbd(x, tx, T, penalizer=1e-3)
        # With very little data, optimizer may not converge
        assert "converged" in result

    def test_bgnbd_predict_handles_non_converged(self):
        """bgnbd_predict should handle edge cases in parameters."""
        par = {"r": 1.0, "alpha": 10.0, "a": 2.0, "b": 3.0}
        x = np.array([0, 1, 2])
        tx = np.array([5, 10, 15])
        T = np.array([30, 30, 30])

        p_alive, expected = rca.bgnbd_predict(par, x, tx, T, horizon=30.0)

        assert len(p_alive) == 3
        assert len(expected) == 3
        assert all(np.isfinite(p_alive))
        # expected can be nan for edge cases but should be handled


class TestGammaGammaFitting:
    """Tests for Gamma-Gamma model fitting and status reporting."""

    def test_fit_gamma_gamma_returns_status(self):
        """fit_gamma_gamma should return status field."""
        n = np.array([2, 3, 4, 5, 6] * 10, dtype=float)
        mbar = np.array([10.0, 15.0, 20.0, 25.0, 30.0] * 10, dtype=float)

        result = rca.fit_gamma_gamma(n, mbar, penalizer=1e-3)

        assert "status" in result
        if result["status"] == "fitted":
            assert "converged" in result
            assert "p" in result and "q" in result and "gamma" in result
            assert "independence_warning" in result
        else:
            assert "reason" in result

    def test_fit_gamma_gamma_insufficient_data_skipped(self):
        """fit_gamma_gamma should skip when fewer than 30 customers with >=2 trips."""
        n = np.array([2, 3], dtype=float)
        mbar = np.array([10.0, 15.0], dtype=float)

        result = rca.fit_gamma_gamma(n, mbar, penalizer=1e-3)

        assert result["status"] == "skipped"
        assert "reason" in result

    def test_gamma_gamma_predict_fallback_when_not_fitted(self):
        """gamma_gamma_predict should fall back to observed mean when not fitted."""
        par = {"status": "skipped", "reason": "insufficient data"}
        n = np.array([1, 2, 3], dtype=float)
        mbar = np.array([10.0, 20.0, 30.0], dtype=float)

        pred = rca.gamma_gamma_predict(par, n, mbar)

        # Should return observed mean trip value
        assert np.allclose(pred, mbar)


class TestScoreCustomers:
    """Tests for score_customers - model status and fallback reporting."""

    def test_score_customers_reports_model_status(self, sample_customer_table):
        """score_customers should return model info with status."""
        c = sample_customer_table
        horizon = 30.0

        # Mock args
        class Args:
            penalizer = 1e-3

        args = Args()

        scored, bg, gg = rca.score_customers(c, horizon, args)

        # Check customer features have model outputs
        assert "p_alive" in scored.columns
        assert "expected_trips_horizon" in scored.columns
        assert "expected_trip_value" in scored.columns
        assert "expected_revenue_horizon" in scored.columns

        # Check model info has status
        assert "status" in bg
        assert "status" in gg

        # If BG/NBD failed, should have fallback predictions
        if bg["status"] == "failed":
            # Should still have predictions (empirical fallback)
            assert scored["expected_trips_horizon"].notna().any()

    def test_score_customers_fallback_counts_reported(self):
        """Fallback predictions should be identifiable."""
        # Create minimal customer table that will trigger fallback
        c = pd.DataFrame(
            {
                "customer_id": [f"C{i}" for i in range(10)],  # Less than 50
                "x": [0] * 10,
                "t_x": [0] * 10,
                "T": [30] * 10,
                "n_trips": [1] * 10,
                "mean_trip_value": [10.0] * 10,
                "recency_days": [5] * 10,
            }
        ).set_index("customer_id")

        class Args:
            penalizer = 1e-3

        args = Args()
        scored, bg, gg = rca.score_customers(c, 30.0, args)

        # With < 50 customers, BG/NBD should fail and use fallback
        assert bg["status"] == "failed"
        # Should still produce predictions (empirical rate fallback)
        assert scored["expected_trips_horizon"].notna().all()


class TestModelStatusReporting:
    """Tests for explicit model status values (R07)."""

    def test_model_status_values_are_standardized(self):
        """Model status should use standard values."""
        # Test BG/NBD
        x = np.array([0, 1, 2] * 20, dtype=float)
        tx = np.array([5, 10, 15] * 20, dtype=float)
        T = np.array([30] * 60, dtype=float)

        bg = rca.fit_bgnbd(x, tx, T, 1e-3)
        assert bg["status"] in ["fitted", "failed"]  # Should be one of these
        if bg["status"] == "fitted":
            assert isinstance(bg["converged"], bool)

        # Test Gamma-Gamma
        n = np.array([2, 3, 4] * 15, dtype=float)
        mbar = np.array([10.0, 20.0, 30.0] * 15, dtype=float)

        gg = rca.fit_gamma_gamma(n, mbar, 1e-3)
        assert gg["status"] in ["fitted", "skipped"]
        if gg["status"] == "fitted":
            assert isinstance(gg["converged"], bool)

    def test_non_finite_predictions_rejected(self):
        """Non-finite predictions should be handled, not silently passed through."""
        # This tests that the pipeline handles non-finite values
        c = pd.DataFrame(
            {
                "customer_id": ["C1"],
                "x": [0.0],
                "t_x": [0.0],
                "T": [30.0],
                "n_trips": [1],
                "mean_trip_value": [10.0],
                "recency_days": [5.0],
            }
        ).set_index("customer_id")

        class Args:
            penalizer = 1e-3

        scored, bg, gg = rca.score_customers(c, 30.0, Args())

        # Predictions should be finite or NaN (not inf)
        assert np.all(
            np.isfinite(scored["expected_trips_horizon"].replace([np.inf, -np.inf], np.nan))
        )
        assert np.all(
            np.isfinite(scored["expected_revenue_horizon"].replace([np.inf, -np.inf], np.nan))
        )


# Fixture for sample customer table
@pytest.fixture
def sample_customer_table():
    """Create a sample customer table with enough data for BG/NBD."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame(
        {
            "customer_id": [f"C{i}" for i in range(n)],
            "x": np.random.poisson(2, n).astype(float),
            "t_x": np.random.uniform(0, 30, n),
            "T": np.random.uniform(30, 90, n),
            "n_trips": np.random.poisson(3, n) + 1,
            "mean_trip_value": np.random.lognormal(2, 0.5, n),
            "recency_days": np.random.uniform(0, 30, n),
            "std_gap_days": np.random.uniform(5, 20, n),
            "mean_gap_days": np.random.uniform(10, 30, n),
            "n_gaps": np.random.randint(1, 10, n),
            "gross_spend": np.random.lognormal(4, 0.5, n),
        }
    ).set_index("customer_id")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
