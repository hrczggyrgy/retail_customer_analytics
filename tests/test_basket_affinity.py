"""Tests for basket affinity with different frequency grains."""

import pandas as pd
import pytest

import retail_customer_analysis as rca


class TestBasketAffinity:
    """Tests for basket_affinity function with trip vs transaction grain."""

    def _make_transactions(self):
        """Create sample transaction data with multiple items per day.

        Note: Need at least 30 customers to meet the association analysis threshold.
        """
        data = []
        # Customer C1: 2 days, 2 transactions per day
        # Day 1: T1 (CatA, CatB), T2 (CatA)
        data.append(
            {
                "customer_id": "C1",
                "transaction_id": "T1",
                "transaction_day": pd.Timestamp("2024-01-01"),
                "transaction_date": pd.Timestamp("2024-01-01 10:00"),
                "product_id": "P1",
                "product_description": "Product 1",
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
                "transaction_id": "T1",
                "transaction_day": pd.Timestamp("2024-01-01"),
                "transaction_date": pd.Timestamp("2024-01-01 10:00"),
                "product_id": "P2",
                "product_description": "Product 2",
                "category": "CatB",
                "department": "Dept1",
                "price": 20.0,
                "quantity": 1,
                "line_revenue": 20.0,
            }
        )
        data.append(
            {
                "customer_id": "C1",
                "transaction_id": "T2",
                "transaction_day": pd.Timestamp("2024-01-01"),
                "transaction_date": pd.Timestamp("2024-01-01 14:00"),
                "product_id": "P1",
                "product_description": "Product 1",
                "category": "CatA",
                "department": "Dept1",
                "price": 10.0,
                "quantity": 1,
                "line_revenue": 10.0,
            }
        )
        # Day 2: T3 (CatB, CatC)
        data.append(
            {
                "customer_id": "C1",
                "transaction_id": "T3",
                "transaction_day": pd.Timestamp("2024-01-02"),
                "transaction_date": pd.Timestamp("2024-01-02 10:00"),
                "product_id": "P2",
                "product_description": "Product 2",
                "category": "CatB",
                "department": "Dept1",
                "price": 20.0,
                "quantity": 1,
                "line_revenue": 20.0,
            }
        )
        data.append(
            {
                "customer_id": "C1",
                "transaction_id": "T3",
                "transaction_day": pd.Timestamp("2024-01-02"),
                "transaction_date": pd.Timestamp("2024-01-02 10:00"),
                "product_id": "P3",
                "product_description": "Product 3",
                "category": "CatC",
                "department": "Dept2",
                "price": 30.0,
                "quantity": 1,
                "line_revenue": 30.0,
            }
        )
        # Customer C2: 1 day, 1 transaction
        data.append(
            {
                "customer_id": "C2",
                "transaction_id": "T4",
                "transaction_day": pd.Timestamp("2024-01-01"),
                "transaction_date": pd.Timestamp("2024-01-01 10:00"),
                "product_id": "P1",
                "product_description": "Product 1",
                "category": "CatA",
                "department": "Dept1",
                "price": 10.0,
                "quantity": 2,
                "line_revenue": 20.0,
            }
        )
        data.append(
            {
                "customer_id": "C2",
                "transaction_id": "T4",
                "transaction_day": pd.Timestamp("2024-01-01"),
                "transaction_date": pd.Timestamp("2024-01-01 10:00"),
                "product_id": "P2",
                "product_description": "Product 2",
                "category": "CatB",
                "department": "Dept1",
                "price": 20.0,
                "quantity": 1,
                "line_revenue": 20.0,
            }
        )

        # Add 28 more customers to meet the 30-customer threshold
        for i in range(3, 31):
            data.append(
                {
                    "customer_id": f"C{i}",
                    "transaction_id": f"T{i}a",
                    "transaction_day": pd.Timestamp("2024-01-01"),
                    "transaction_date": pd.Timestamp("2024-01-01 10:00"),
                    "product_id": "P1",
                    "product_description": "Product 1",
                    "category": "CatA",
                    "department": "Dept1",
                    "price": 10.0,
                    "quantity": 1,
                    "line_revenue": 10.0,
                }
            )
            data.append(
                {
                    "customer_id": f"C{i}",
                    "transaction_id": f"T{i}b",
                    "transaction_day": pd.Timestamp("2024-01-01"),
                    "transaction_date": pd.Timestamp("2024-01-01 10:00"),
                    "product_id": "P2",
                    "product_description": "Product 2",
                    "category": "CatB",
                    "department": "Dept1",
                    "price": 20.0,
                    "quantity": 1,
                    "line_revenue": 20.0,
                }
            )

        return pd.DataFrame(data)

    def _make_args(self, frequency_grain="trip"):
        """Create mock args for basket_affinity."""

        class Args:
            def __init__(self, frequency_grain):
                self.basket_level = "category"
                self.min_support = 0.01
                self.min_confidence = 0.20
                self.min_lift = 1.0
                self.fdr_alpha = 0.05
                self.max_affinity_items = 100
                self.max_affinity_baskets = 50000
                self.max_items_per_basket = 30
                self.max_affinity_pair_operations = 1_000_000
                self.frequency_grain = frequency_grain
                self.random_seed = 42

        return Args(frequency_grain)

    def test_trip_grain_merges_same_day_transactions(self):
        """trip grain should merge transactions on same day into one basket."""
        transactions = self._make_transactions()
        as_of = pd.Timestamp("2024-01-02")
        args = self._make_args("trip")

        aff, nxt, det = rca.basket_affinity(transactions, as_of, args)

        assert det["status"] == "completed"
        # C1: 2 days -> 2 baskets (Jan 1 and Jan 2)
        # C2: 1 day -> 1 basket
        # C3-C30: 28 customers * 1 day = 28 baskets
        # Total baskets = 31
        assert det["eligible_baskets"] == 31
        # Check that CatA+CatB co-occurred on Jan 1 (merged T1+T2)
        # CatA appears in T1 and T2 on same day, CatB in T1
        # So CatA+CatB pair should exist in Jan 1 basket

    def test_transaction_grain_keeps_separate_transactions(self):
        """transaction grain should keep each transaction_id as separate basket."""
        transactions = self._make_transactions()
        as_of = pd.Timestamp("2024-01-02")
        args = self._make_args("transaction")

        aff, nxt, det = rca.basket_affinity(transactions, as_of, args)

        assert det["status"] == "completed"
        # C1: 3 transactions (T1, T2, T3) -> 3 baskets
        # C2: 1 transaction (T4) -> 1 basket
        # C3-C30: 28 customers * 2 transactions = 56 baskets
        # Total baskets = 60
        assert det["eligible_baskets"] == 60
        assert det["basket_grain"] == "transaction"

    def test_trip_grain_basket_content(self):
        """Verify trip grain basket content is correct."""
        transactions = self._make_transactions()
        as_of = pd.Timestamp("2024-01-02")
        args = self._make_args("trip")

        aff, nxt, det = rca.basket_affinity(transactions, as_of, args)

        # With trip grain: C1 Jan 1 basket has CatA, CatB (merged T1+T2)
        # C1 Jan 2 basket has CatB, CatC (T3)
        # C2 Jan 1 basket has CatA, CatB (T4)
        # C3-C30 Jan 1: each has CatA, CatB (T{i}a + T{i}b merged)

        # CatA+CatB should appear in 30 baskets (C1 Jan 1, C2 Jan 1, C3-C30)
        # CatB+CatC should appear in 1 basket (C1 Jan 2)
        catA_catB = aff[(aff["antecedent"] == "CatA") & (aff["consequent"] == "CatB")]
        catB_catC = aff[(aff["antecedent"] == "CatB") & (aff["consequent"] == "CatC")]

        assert len(catA_catB) == 1
        assert catA_catB.iloc[0]["pair_baskets"] == 30

        assert len(catB_catC) == 1
        assert catB_catC.iloc[0]["pair_baskets"] == 1

    def test_transaction_grain_basket_content(self):
        """Verify transaction grain basket content is correct."""
        transactions = self._make_transactions()
        as_of = pd.Timestamp("2024-01-02")
        args = self._make_args("transaction")

        aff, nxt, det = rca.basket_affinity(transactions, as_of, args)

        # With transaction grain:
        # C1 T1: CatA, CatB -> 1 CatA+CatB pair
        # C1 T2: CatA -> no pair
        # C1 T3: CatB, CatC -> 1 CatB+CatC pair
        # C2 T4: CatA, CatB -> 1 CatA+CatB pair
        # C3-C30: 28 customers * 2 transactions each = 56 transactions
        #   Each transaction has only one category (CatA or CatB), so no co-occurrence

        # CatA+CatB appears in 2 baskets (C1 T1, C2 T4)
        # CatB+CatC appears in 1 basket (C1 T3)
        catA_catB = aff[(aff["antecedent"] == "CatA") & (aff["consequent"] == "CatB")]
        catB_catC = aff[(aff["antecedent"] == "CatB") & (aff["consequent"] == "CatC")]

        assert len(catA_catB) == 1
        assert catA_catB.iloc[0]["pair_baskets"] == 2

        assert len(catB_catC) == 1
        assert catB_catC.iloc[0]["pair_baskets"] == 1

    def test_next_trip_transitions_trip_grain(self):
        """Next-trip transitions should work with trip grain."""
        transactions = self._make_transactions()
        as_of = pd.Timestamp("2024-01-02")
        args = self._make_args("trip")

        aff, nxt, det = rca.basket_affinity(transactions, as_of, args)

        # C1 has 2 trips: Jan 1 -> Jan 2
        # Jan 1: CatA, CatB -> Jan 2: CatB, CatC
        # Transitions: CatA->CatB, CatA->CatC, CatB->CatB, CatB->CatC
        if not nxt.empty:
            # Check CatB -> CatC transition exists
            catB_to_catC = nxt[(nxt["from_item"] == "CatB") & (nxt["next_trip_item"] == "CatC")]
            assert len(catB_to_catC) >= 1

    def test_next_trip_transitions_transaction_grain(self):
        """Next-trip transitions should work with transaction grain."""
        transactions = self._make_transactions()
        as_of = pd.Timestamp("2024-01-02")
        args = self._make_args("transaction")

        aff, nxt, det = rca.basket_affinity(transactions, as_of, args)

        # C1 has 3 transactions: T1 -> T2 -> T3
        # T1: CatA, CatB -> T2: CatA -> T3: CatB, CatC
        if not nxt.empty:
            # Should have more transitions due to more granular baskets
            assert len(nxt) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
