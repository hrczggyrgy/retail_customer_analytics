"""Tests for purchase event construction and return matching (R02, R04)."""

import pandas as pd
import pytest

import retail_customer_analysis as rca


class TestBuildTrips:
    """Tests for build_trips function - event grain consistency (R02)."""

    def test_trip_grain_merges_same_day_transactions(self):
        """trip grain should merge multiple transactions on same customer-day."""
        lines = pd.DataFrame(
            {
                "customer_id": ["C1", "C1", "C1"],
                "transaction_id": ["T1", "T2", "T3"],
                "transaction_day": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"]),
                "product_id": ["P1", "P2", "P3"],
                "quantity": [1, 2, 1],
                "price": [10.0, 20.0, 15.0],
                "line_revenue": [10.0, 40.0, 15.0],
            }
        )

        trips = rca.build_trips(lines, grain="trip")

        # Should have 2 trips (Jan 1 and Jan 2)
        assert len(trips) == 2
        # Jan 1 trip value = 10 + 40 = 50
        day1 = trips[trips["transaction_day"] == pd.Timestamp("2024-01-01")]
        assert day1["trip_value"].iloc[0] == 50.0
        assert day1["n_lines"].iloc[0] == 2
        assert day1["n_products"].iloc[0] == 2

    def test_transaction_grain_keeps_separate_transactions(self):
        """transaction grain should keep each transaction_id separate."""
        lines = pd.DataFrame(
            {
                "customer_id": ["C1", "C1", "C1"],
                "transaction_id": ["T1", "T2", "T3"],
                "transaction_day": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"]),
                "product_id": ["P1", "P2", "P3"],
                "quantity": [1, 2, 1],
                "price": [10.0, 20.0, 15.0],
                "line_revenue": [10.0, 40.0, 15.0],
            }
        )

        trips = rca.build_trips(lines, grain="transaction")

        # Should have 3 trips (one per transaction_id)
        assert len(trips) == 3
        # Each should have its own trip_value
        trip_t1 = trips[trips["transaction_id"] == "T1"]
        assert trip_t1["trip_value"].iloc[0] == 10.0

    def test_trip_grain_excludes_zero_negative_quantity(self):
        """Only positive quantity lines should create trips."""
        lines = pd.DataFrame(
            {
                "customer_id": ["C1", "C1", "C1"],
                "transaction_id": ["T1", "T2", "T3"],
                "transaction_day": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
                "product_id": ["P1", "P2", "P3"],
                "quantity": [1, 0, -1],
                "price": [10.0, 20.0, 15.0],
                "line_revenue": [10.0, 0.0, -15.0],
            }
        )

        trips = rca.build_trips(lines, grain="trip")

        # Only the positive quantity line should create a trip
        assert len(trips) == 1
        assert trips.iloc[0]["customer_id"] == "C1"

    def test_trip_grain_excludes_missing_customer_id(self):
        """Lines without customer_id should be excluded from trips."""
        lines = pd.DataFrame(
            {
                "customer_id": ["C1", None, "C1"],
                "transaction_id": ["T1", "T2", "T3"],
                "transaction_day": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
                "product_id": ["P1", "P2", "P3"],
                "quantity": [1, 1, 1],
                "price": [10.0, 20.0, 15.0],
                "line_revenue": [10.0, 20.0, 15.0],
            }
        )

        trips = rca.build_trips(lines, grain="trip")

        assert len(trips) == 2
        assert all(trips["customer_id"] == "C1")

    def test_trip_grain_excludes_missing_transaction_day(self):
        """Lines without transaction_day should be excluded."""
        lines = pd.DataFrame(
            {
                "customer_id": ["C1", "C1", "C1"],
                "transaction_id": ["T1", "T2", "T3"],
                "transaction_day": [pd.Timestamp("2024-01-01"), pd.NaT, pd.Timestamp("2024-01-03")],
                "product_id": ["P1", "P2", "P3"],
                "quantity": [1, 1, 1],
                "price": [10.0, 20.0, 15.0],
                "line_revenue": [10.0, 20.0, 15.0],
            }
        )

        trips = rca.build_trips(lines, grain="trip")

        assert len(trips) == 2


class TestMatchReturns:
    """Tests for return matching (R04 - temporal matching)."""

    def test_return_matched_to_earlier_purchase(self):
        """Return should be matched to latest earlier purchase of same customer/product."""
        lines = pd.DataFrame(
            {
                "customer_id": ["C1", "C1", "C1", "C1"],
                "product_id": ["P1", "P1", "P1", "P1"],
                "transaction_day": pd.to_datetime(
                    ["2024-01-01", "2024-01-10", "2024-01-15", "2024-01-20"]
                ),
                "transaction_id": ["T1", "T2", "T3", "T4"],
                "quantity": [1, 1, -1, 1],
                "price": [10.0, 10.0, 10.0, 10.0],
                "line_revenue": [10.0, 10.0, -10.0, 10.0],
            }
        )

        matched, info = rca.match_returns(lines)

        # Return on Jan 15 (qty=-1) should match to purchase on Jan 10 (latest earlier)
        ret_row = matched[matched["transaction_id"] == "T3"]
        assert len(ret_row) == 1
        assert ret_row["matched"].iloc[0]
        assert ret_row["purchase_time"].iloc[0] == pd.Timestamp("2024-01-10")

    def test_return_not_matched_to_later_same_day_purchase(self):
        """Return should NOT be matched to a purchase on the SAME day that occurs LATER when timestamps available."""
        # With full timestamps: purchase at 14:00, return at 10:00 (earlier same day)
        # Should NOT match because return happened BEFORE purchase
        lines = pd.DataFrame(
            {
                "customer_id": ["C1", "C1"],
                "product_id": ["P1", "P1"],
                "transaction_date": pd.to_datetime(["2024-01-15 14:00:00", "2024-01-15 10:00:00"]),
                "transaction_day": pd.to_datetime(["2024-01-15", "2024-01-15"]),  # Normalized day
                "transaction_id": ["T1", "T2"],
                "quantity": [1, -1],
                "price": [10.0, 10.0],
                "line_revenue": [10.0, -10.0],
            }
        )

        matched, info = rca.match_returns(lines)

        # With timestamps, return at 10:00 should NOT match to purchase at 14:00 (later same day)
        ret_row = matched[matched["transaction_id"] == "T2"]
        assert len(ret_row) == 1
        assert not ret_row["matched"].iloc[0]
        assert info["matching_precision"] == "timestamp"

    def test_return_matched_to_earlier_same_day_purchase(self):
        """Return SHOULD be matched to a purchase on the SAME day that occurs EARLIER when timestamps available."""
        # Purchase at 10:00, return at 14:00 (later same day) - should match
        lines = pd.DataFrame(
            {
                "customer_id": ["C1", "C1"],
                "product_id": ["P1", "P1"],
                "transaction_date": pd.to_datetime(["2024-01-15 10:00:00", "2024-01-15 14:00:00"]),
                "transaction_day": pd.to_datetime(["2024-01-15", "2024-01-15"]),
                "transaction_id": ["T1", "T2"],
                "quantity": [1, -1],
                "price": [10.0, 10.0],
                "line_revenue": [10.0, -10.0],
            }
        )

        matched, info = rca.match_returns(lines)

        # With timestamps, return at 14:00 should match to purchase at 10:00 (earlier same day)
        ret_row = matched[matched["transaction_id"] == "T2"]
        assert len(ret_row) == 1
        assert ret_row["matched"].iloc[0]
        assert info["matching_precision"] == "timestamp"

    def test_return_fallback_to_calendar_day_when_no_timestamps(self):
        """Without timestamps, falls back to calendar-day matching (may match same-day)."""
        lines = pd.DataFrame(
            {
                "customer_id": ["C1", "C1"],
                "product_id": ["P1", "P1"],
                "transaction_day": pd.to_datetime(["2024-01-15", "2024-01-15"]),
                "transaction_id": ["T1", "T2"],
                "quantity": [1, -1],
                "price": [10.0, 10.0],
                "line_revenue": [10.0, -10.0],
            }
        )

        matched, info = rca.match_returns(lines)

        # Without timestamps, matches on calendar day
        ret_row = matched[matched["transaction_id"] == "T2"]
        assert len(ret_row) == 1
        assert ret_row["matched"].iloc[0]
        assert info["matching_precision"] == "calendar_day"

    def test_return_unmatched_when_no_earlier_purchase(self):
        """Return with no earlier purchase should be unmatched."""
        lines = pd.DataFrame(
            {
                "customer_id": ["C1"],
                "product_id": ["P1"],
                "transaction_day": pd.to_datetime(["2024-01-15"]),
                "transaction_id": ["T1"],
                "quantity": [-1],
                "price": [10.0],
                "line_revenue": [-10.0],
            }
        )

        matched, info = rca.match_returns(lines)

        assert info["matched_return_lines"] == 0
        assert info["matched_return_value_share"] == 0.0

    def test_return_matching_across_customers(self):
        """Returns should only match to same customer's purchases."""
        lines = pd.DataFrame(
            {
                "customer_id": ["C1", "C2", "C2"],
                "product_id": ["P1", "P1", "P1"],
                "transaction_day": pd.to_datetime(["2024-01-10", "2024-01-05", "2024-01-15"]),
                "transaction_id": ["T1", "T2", "T3"],
                "quantity": [1, 1, -1],
                "price": [10.0, 10.0, 10.0],
                "line_revenue": [10.0, 10.0, -10.0],
            }
        )

        matched, info = rca.match_returns(lines)

        # C2's return should match to C2's purchase, not C1's
        ret_row = matched[matched["transaction_id"] == "T3"]
        assert ret_row["matched"].iloc[0]
        assert ret_row["purchase_time"].iloc[0] == pd.Timestamp("2024-01-05")

    def test_return_matching_across_products(self):
        """Returns should only match to same product's purchases."""
        lines = pd.DataFrame(
            {
                "customer_id": ["C1", "C1", "C1"],
                "product_id": ["P1", "P2", "P1"],
                "transaction_day": pd.to_datetime(["2024-01-05", "2024-01-10", "2024-01-15"]),
                "transaction_id": ["T1", "T2", "T3"],
                "quantity": [1, 1, -1],
                "price": [10.0, 10.0, 10.0],
                "line_revenue": [10.0, 10.0, -10.0],
            }
        )

        matched, info = rca.match_returns(lines)

        # P1 return should match to P1 purchase, not P2
        ret_row = matched[matched["transaction_id"] == "T3"]
        assert ret_row["matched"].iloc[0]
        assert ret_row["purchase_time"].iloc[0] == pd.Timestamp("2024-01-05")


class TestTransactionSummary:
    """Tests for transaction_summary - key for R02 event definitions."""

    def test_transaction_summary_customer_transaction_mode(self):
        """customer-transaction mode should group by (customer_id, transaction_id)."""
        lines = pd.DataFrame(
            {
                "customer_id": ["C1", "C1", "C2"],
                "transaction_id": ["T1", "T1", "T1"],  # Same ID, different customers
                "transaction_day": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"]),
                "product_id": ["P1", "P2", "P3"],
                "quantity": [1, 2, 1],
                "price": [10.0, 20.0, 15.0],
                "line_revenue": [10.0, 40.0, 15.0],
            }
        )

        summary = rca.transaction_summary(lines, mode="customer-transaction")

        # Should have 2 rows: (C1,T1) and (C2,T1)
        assert len(summary) == 2
        c1t1 = summary[(summary["customer_id"] == "C1") & (summary["transaction_id"] == "T1")]
        assert c1t1["gross_purchase_revenue"].iloc[0] == 50.0
        c2t1 = summary[(summary["customer_id"] == "C2") & (summary["transaction_id"] == "T1")]
        assert c2t1["gross_purchase_revenue"].iloc[0] == 15.0

    def test_transaction_summary_transaction_only_mode(self):
        """transaction-only mode should group by transaction_id globally."""
        lines = pd.DataFrame(
            {
                "customer_id": ["C1", "C1", "C2"],
                "transaction_id": ["T1", "T1", "T1"],  # Same ID, different customers
                "transaction_day": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"]),
                "product_id": ["P1", "P2", "P3"],
                "quantity": [1, 2, 1],
                "price": [10.0, 20.0, 15.0],
                "line_revenue": [10.0, 40.0, 15.0],
            }
        )

        summary = rca.transaction_summary(lines, mode="transaction-only")

        # Should have 1 row: T1 (merged across customers!)
        assert len(summary) == 1
        assert summary.iloc[0]["transaction_id"] == "T1"
        assert summary.iloc[0]["gross_purchase_revenue"] == 65.0  # 50 + 15
        assert summary.iloc[0]["distinct_customers"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
