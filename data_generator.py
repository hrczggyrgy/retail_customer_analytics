#!/usr/bin/env python3
"""
Behavioral retail transaction generator.

Requires:
    Python 3.10+
    pip install numpy pandas

Examples:
    python retail_generator.py

    python retail_generator.py \
        --customers 5000 \
        --start 2025-01-01 \
        --end 2026-12-31 \
        --seed 42 \
        --output retail_output

    python retail_generator.py --no-returns

Output:
    transactions.csv       Exactly the nine requested columns.
    products.csv           Stable product catalog.
    customer_latents.csv   Simulation-only customer attributes.
    return_links.csv       Links refunds to original purchases.
    diagnostics.json       Summary statistics and configuration.

Conventions:
    One row = one product line in one transaction.
    price = paid UNIT price, expressed in EUR.
    quantity < 0 = return.
    transaction_date = local, timezone-naive timestamp.
    Customer population includes customers who never purchase.
    Row count emerges from behavior; it is not fixed in advance.

Important:
    Parameters are illustrative, NOT fitted estimates.
    No real customer data is used.
    This is a research-informed custom simulator, not an implementation
    of RetailSynth or an exact BG/NBD model.
    The simulator holds generated records in memory before exporting.

Research informing the design:
    https://arxiv.org/abs/2312.14095
    https://archive.ics.uci.edu/dataset/352/online+retail
    https://www.pymc-marketing.io/en/0.16.0/notebooks/clv/gamma_gamma.html
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


COLUMNS = [
    "customer_id",
    "transaction_date",
    "transaction_id",
    "product_id",
    "product_description",
    "department",
    "category",
    "price",
    "quantity",
]

# Each category:
# department, category, reference unit price, replenishment days,
# return probability per purchased line, four product templates.
#
# The price/return/cycle values are editable simulation assumptions.
# The default catalog has 16 categories x 6 brands x 4 products = 384 SKUs.
SPECS = [
    (
        "Grocery", "Bread", 2.2, 4, 0.002,
        ["Wholegrain loaf 500g", "Sourdough loaf 500g",
         "White loaf 500g", "Bread rolls 6-pack"],
    ),
    (
        "Grocery", "Milk", 1.6, 5, 0.002,
        ["Whole milk 1L", "Semi-skimmed milk 1L",
         "Oat drink 1L", "Lactose-free milk 1L"],
    ),
    (
        "Grocery", "Produce", 2.8, 5, 0.003,
        ["Apples 1kg", "Bananas 1kg", "Tomatoes 500g", "Carrots 1kg"],
    ),
    (
        "Grocery", "Meat", 5.8, 6, 0.004,
        ["Chicken breast 500g", "Turkey mince 500g",
         "Pork chops 500g", "Beef mince 500g"],
    ),
    (
        "Grocery", "Pasta", 1.8, 21, 0.002,
        ["Penne pasta 500g", "Spaghetti 500g",
         "Fusilli pasta 500g", "Wholegrain pasta 500g"],
    ),
    (
        "Grocery", "Sauces", 2.4, 21, 0.002,
        ["Tomato pasta sauce 400g", "Basil pasta sauce 400g",
         "Pesto 190g", "Arrabbiata sauce 400g"],
    ),
    (
        "Grocery", "Coffee", 5.5, 25, 0.003,
        ["Ground coffee 250g", "Coffee beans 250g",
         "Espresso capsules 10-pack", "Instant coffee 100g"],
    ),
    (
        "Grocery", "Snacks", 2.1, 10, 0.002,
        ["Potato crisps 150g", "Dark chocolate 100g",
         "Salted nuts 150g", "Oat biscuits 200g"],
    ),
    (
        "Grocery", "Soft drinks", 1.7, 8, 0.002,
        ["Cola 1L", "Sparkling water 1L",
         "Orange soda 1L", "Still water 1L"],
    ),
    (
        "Household", "Laundry", 7.5, 35, 0.008,
        ["Laundry liquid 1L", "Laundry capsules 20-pack",
         "Laundry powder 1kg", "Fabric softener 1L"],
    ),
    (
        "Household", "Cleaning", 3.1, 28, 0.008,
        ["Dishwashing liquid 500ml", "Surface cleaner 750ml",
         "Bathroom cleaner 750ml", "Glass cleaner 500ml"],
    ),
    (
        "Personal Care", "Hair care", 4.5, 35, 0.012,
        ["Daily shampoo 400ml", "Repair shampoo 400ml",
         "Conditioner 400ml", "Dry shampoo 200ml"],
    ),
    (
        "Personal Care", "Oral care", 3.2, 40, 0.008,
        ["Fluoride toothpaste 75ml", "Sensitive toothpaste 75ml",
         "Toothbrush 2-pack", "Mouthwash 500ml"],
    ),
    (
        "Apparel", "Clothing", 22.0, 100, 0.16,
        ["Cotton crew-neck T-shirt", "Long-sleeve cotton top",
         "Casual sweatshirt", "Cotton polo shirt"],
    ),
    (
        "Apparel", "Footwear", 55.0, 220, 0.18,
        ["Everyday trainers", "Walking shoes",
         "Canvas sneakers", "Casual loafers"],
    ),
    (
        "Electronics", "Accessories", 24.0, 180, 0.07,
        ["USB-C wall charger 30W", "Wireless mouse",
         "Bluetooth speaker", "USB-C power bank 10000mAh"],
    ),
]

# Fictional brands, ordered as two value, two standard, two premium.
BRANDS = {
    "Grocery": [
        "Pantry", "Field", "Daily", "Harvest", "Reserve", "Orchard"
    ],
    "Household": [
        "BasicHome", "PureHome", "HomeWorks", "Bright", "CleanPro", "EcoHome"
    ],
    "Personal Care": [
        "CareBasics", "Fresh", "DailyCare", "Balance", "Botanica", "Salon"
    ],
    "Apparel": [
        "Essential", "CottonCo", "Urban", "Thread", "Studio", "Atelier"
    ],
    "Electronics": [
        "EntryTech", "Wire", "Connect", "Volt", "Circuit", "Apex"
    ],
}

# Category weights for six shopping missions.
# Column order matches SPECS.
# Rows: main shop, top-up, household, fashion, technology, bulk.
MISSIONS = np.array([
    [4, 4, 4, 3, 2, 2, 1, 2, 2, 1, 1, 1, .7, .10, .03, .05],
    [4, 4, 3, 1, .5, .5, 1, 3, 3, .1, .1, .1, .1, .02, .01, .02],
    [.1, .1, .1, .1, .2, .2, .3, .3, .3, 4, 4, 3, 3, .1, .05, .1],
    [.05, .05, .05, .05, .05, .05, .05, .2, .1, .05, .05, .2, .2, 5, 3, .3],
    [.05, .05, .05, .05, .05, .05, .1, .2, .1, .1, .1, .1, .1, .3, .1, 7],
    [2, 2, 1, 1, 3, 3, 2, 3, 3, 2, 2, 1, 1, .03, .01, .1],
], dtype=float)

SEGMENTS = [
    "regular",
    "convenience",
    "deal_seeker",
    "premium",
    "occasional",
    "bulk",
]

# Mission probabilities by customer segment.
MISSION_PROBS = np.array([
    [.51, .27, .13, .06, .02, .01],
    [.20, .65, .08, .04, .02, .01],
    [.40, .24, .21, .08, .02, .05],
    [.42, .22, .14, .14, .07, .01],
    [.20, .23, .19, .25, .12, .01],
    [.11, .08, .12, .02, .02, .65],
])

BASKET_CATS = np.array([6, 2, 3, 2, 1, 5])
BUDGETS = np.array([48, 14, 35, 95, 85, 120])

# Explicit complementary category pairs.
COMPLEMENTS = [
    (0, 1),    # bread + milk
    (4, 5),    # pasta + sauce
    (6, 7),    # coffee + snacks
    (9, 10),   # laundry + cleaning
    (11, 12),  # hair + oral care
    (13, 14),  # clothing + footwear
    (7, 8),    # snacks + soft drinks
]


def softmax(x):
    y = np.exp(x - np.max(x))
    return y / y.sum()


def generate(
    customers=5000,
    start="2025-01-01",
    end=None,
    seed=42,
    output="retail_output",
    annual_inflation=0.025,
    enable_returns=True,
):
    start = pd.Timestamp(start).normalize()
    # Default end to today's date (normalized) to avoid future-dated demo data
    if end is None:
        end = pd.Timestamp.now().normalize()
    else:
        end = pd.Timestamp(end).normalize()

    if customers < 1 or end < start or annual_inflation <= -1:
        raise ValueError(
            "Require customers >= 1, end >= start, inflation > -1."
        )

    days = pd.date_range(start, end, freq="D")
    n, k = customers, len(SPECS)
    rng = np.random.default_rng(seed)

    records = []
    catalog_rows = []

    # Stable product master: product descriptions do not change per purchase.
    for cat, (dept, name, base, cycle, ret, templates) in enumerate(SPECS):
        for brand in range(6):
            tier = brand // 2

            for item, template in enumerate(templates):
                cents = max(
                    30,
                    int(round(
                        100
                        * base
                        * [0.72, 1.0, 1.65][tier]
                        * [0.9, 1.0, 1.12, 1.25][item]
                        * rng.lognormal(0, 0.09)
                    )),
                )

                catalog_rows.append((
                    f"P{len(catalog_rows) + 1:05d}",
                    f"{BRANDS[dept][brand]} {template}",
                    dept,
                    name,
                    cat,
                    brand,
                    tier,
                    cents,
                    rng.lognormal(0, 0.7),
                ))

    catalog = pd.DataFrame(catalog_rows, columns=[
        "product_id",
        "product_description",
        "department",
        "category",
        "category_index",
        "brand_index",
        "tier",
        "base_price_cents",
        "popularity",
    ])

    pids = catalog.product_id.to_numpy()
    desc = catalog.product_description.to_numpy()
    depts = catalog.department.to_numpy()
    cats = catalog.category.to_numpy()
    base_cents = catalog.base_price_cents.to_numpy()
    tiers = catalog.tier.to_numpy()
    brands = catalog.brand_index.to_numpy()
    pop = np.log(catalog.popularity.to_numpy())
    cat_idx = catalog.category_index.to_numpy()

    sku_by_cat = [
        np.flatnonzero(cat_idx == c)
        for c in range(k)
    ]

    cycles = np.array([s[3] for s in SPECS], dtype=float)
    ret_probs = np.array([s[4] for s in SPECS])
    reference = np.array([s[2] for s in SPECS])

    # Persistent customer characteristics.
    segment = rng.choice(
        6, n, p=[.38, .20, .19, .10, .10, .03]
    )
    household = rng.choice(
        [1, 2, 3, 4, 5], n, p=[.25, .34, .18, .17, .06]
    )

    affluence = (
        rng.lognormal(0, .4, n)
        * np.array([1, .9, .8, 1.7, 1.05, 1.2])[segment]
    )
    sensitivity = (
        rng.lognormal(0, .3, n)
        * np.array([1, .8, 1.9, .4, 1, 1.4])[segment]
    )
    loyalty = rng.beta(4, 2, n)

    # Gamma-distributed base rates create heterogeneous purchase frequency.
    rates = (
        rng.gamma(1.8, 1 / 1.8, n)
        * np.array([.12, .19, .10, .075, .018, .11])[segment]
    )
    rates = np.clip(
        rates * np.sqrt(household / 2),
        .002,
        .8,
    )

    # Permanent attrition can occur after a completed purchase.
    dropout = (
        rng.beta(1.2, 75, n)
        * np.array([.7, .8, 1.0, .6, 2.4, .4])[segment]
    )

    preferred_day = rng.choice(
        7, n, p=[.10, .09, .10, .10, .18, .28, .15]
    )
    preferred_brand = rng.integers(0, 6, size=(n, k))
    affinity = rng.lognormal(0, .6, size=(n, k))

    # A persistent dietary preference affects baskets across transactions.
    vegetarian = rng.random(n) < .12
    affinity[vegetarian, 3] *= .015
    affinity[vegetarian, 2] *= 1.6
    affinity[:, 13:15] *= affluence[:, None] ** .3

    # Existing customers plus acquisition throughout the simulation.
    arrival = rng.integers(0, len(days), n)
    arrival[rng.random(n) < .60] = 0

    active = np.ones(n, dtype=bool)
    churn_day = np.full(n, -1, dtype=int)

    # Household supply state controls category replenishment pressure.
    stock_until = (
        rng.uniform(0, 1, size=(n, k))
        * cycles[None, :]
    )
    last_sku = np.full((n, k), -1, dtype=int)
    last_visit = np.full(n, -1000, dtype=int)

    vacation_start = rng.integers(0, len(days), n)
    vacation_length = rng.integers(4, 19, n)
    vacation = rng.random(n) < .35

    discount = np.zeros(len(catalog))
    promo_until = np.full(len(catalog), -1)
    unavailable_until = np.full(len(catalog), -1)

    pending = defaultdict(list)
    return_links = []
    tx_counter = 0
    censored_returns = 0

    def add_rows(cid, stamp, tid, lines, sign=1):
        for sku, cents, qty in lines:
            records.append((
                f"C{cid + 1:07d}",
                stamp,
                tid,
                pids[sku],
                desc[sku],
                depts[sku],
                cats[sku],
                cents / 100,
                sign * qty,
            ))

    for d, date in enumerate(days):
        # Promotions are shared across customers and persist for several days.
        expired = promo_until < d
        discount[expired] = 0

        new_promo = (
            expired
            & (rng.random(len(catalog)) < .012)
        )
        promo_until[new_promo] = (
            d + rng.integers(2, 8, new_promo.sum())
        )
        discount[new_promo] = rng.choice(
            [.10, .15, .20, .30],
            new_promo.sum(),
            p=[.3, .3, .3, .1],
        )

        doy = date.dayofyear

        # Illustrative Northern Hemisphere retail calendar.
        annual_peak = np.exp(
            -.5 * ((doy - 350) / 18) ** 2
        )
        black_friday = (
            date.month == 11
            and date.weekday() == 4
            and 23 <= date.day <= 29
        )

        event_discount = np.zeros(k)

        if black_friday:
            event_discount[13:] = .25

        if date.month == 1 and date.day <= 14:
            event_discount[13:15] = .15

        effective_discount = np.maximum(
            discount,
            event_discount[cat_idx],
        )

        inflation = (
            (1 + annual_inflation) ** (d / 365.25)
        )

        # Integer cents avoid accumulating floating-point currency errors.
        paid_cents = np.maximum(
            1,
            np.rint(
                base_cents * inflation * (1 - effective_discount)
            ),
        ).astype(int)

        # Temporary availability shocks, not a full supply-chain model.
        shocks = (
            (unavailable_until < d)
            & (rng.random(len(catalog)) < .004)
        )
        unavailable_until[shocks] = (
            d + rng.integers(1, 5, shocks.sum())
        )
        available = unavailable_until < d

        cat_discounts = np.array([
            effective_discount[s].mean()
            for s in sku_by_cat
        ])

        summer = max(
            0,
            np.cos(2 * np.pi * (doy - 200) / 365.25),
        )

        cat_season = np.ones(k)
        cat_season[[7, 15]] += (
            annual_peak * np.array([.45, .8])
        )
        cat_season[8] += .55 * summer
        cat_season[14] += .25 * summer
        cat_season[13] += (
            .3 * np.exp(-.5 * ((doy - 255) / 25) ** 2)
        )

        weekday = np.where(
            preferred_day == date.dayofweek,
            1.7,
            .8833333333,
        )

        urgency = np.clip(
            (d - stock_until[:, :9]) / cycles[None, :9],
            0,
            1,
        ).mean(axis=1)

        exposed_discount = (
            (affinity * cat_discounts).sum(axis=1)
            / affinity.sum(axis=1)
        )

        rate = (
            rates
            * weekday
            * (1 + .3 * annual_peak + .12 * summer)
        )
        rate *= (
            (1 + .25 * urgency)
            * np.exp(.7 * sensitivity * exposed_discount)
        )
        rate *= np.where(
            d - last_visit == 1,
            .65,
            1.0,
        )

        eligible = active & (arrival <= d)
        eligible &= ~(
            vacation
            & (d >= vacation_start)
            & (d < vacation_start + vacation_length)
        )

        visits = rng.poisson(
            np.where(eligible, rate, 0)
        )
        cids = np.repeat(np.arange(n), visits)

        # Visits are processed chronologically.
        # Multiple same-day visits are possible.
        if len(cids):
            after_work = (
                (date.dayofweek < 5)
                & (rng.random(len(cids)) < .6)
            )
            hours = rng.normal(
                np.where(after_work, 18.0, 13.0),
                np.where(after_work, 1.1, 2.4),
            )
            seconds = np.clip(
                (hours * 3600).astype(int),
                8 * 3600,
                21 * 3600 - 1,
            )
            seconds += rng.integers(
                0, 60, len(cids)
            )
            order = np.argsort(
                seconds, kind="stable"
            )
        else:
            order, seconds = [], []

        for pos in order:
            cid = int(cids[pos])

            if not active[cid]:
                continue

            mp = MISSION_PROBS[segment[cid]].copy()

            # Promotions can encourage stock-up trips.
            mp[5] *= (
                1
                + 2 * exposed_discount[cid] * sensitivity[cid]
            )
            mission = rng.choice(
                6, p=mp / mp.sum()
            )

            budget = (
                BUDGETS[mission]
                * affluence[cid]
                * rng.lognormal(-.08, .35)
            )
            budget *= (
                1 + .15 * (household[cid] - 2)
                if mission in (0, 5)
                else 1
            )

            # Overdispersed basket size, not a fixed number of items.
            target = int(np.clip(
                1 + rng.negative_binomial(
                    3,
                    3 / (
                        3 + max(.1, BASKET_CATS[mission] - 1)
                    ),
                ),
                1,
                k,
            ))

            due = np.clip(
                (d - stock_until[cid]) / cycles,
                -2,
                2,
            )

            category_weights = (
                MISSIONS[mission]
                * affinity[cid]
                * cat_season
            )
            category_weights *= np.exp(
                .7 * due
                + sensitivity[cid] * cat_discounts
            )

            selected = set()
            lines = []
            spend = 0

            # Occasional customer-specific coupon on top of shared promotions.
            coupon = (
                .05
                if segment[cid] == 2 and rng.random() < .12
                else 0.0
            )

            for _ in range(target):
                weights = category_weights.copy()

                if selected:
                    weights[list(selected)] = 0

                if weights.sum() <= 0:
                    break

                c = int(rng.choice(
                    k,
                    p=weights / weights.sum(),
                ))
                selected.add(c)

                skus = sku_by_cat[c][
                    available[sku_by_cat[c]]
                ]

                if not len(skus):
                    continue

                wanted_skus = (
                    1
                    + int(
                        mission in (0, 5)
                        and rng.random() < .3
                    )
                )

                for _ in range(
                    min(wanted_skus, len(skus))
                ):
                    prices = np.maximum(
                        1,
                        np.rint(
                            paid_cents[skus] * (1 - coupon)
                        ),
                    ).astype(int)

                    affordable = (
                        prices
                        <= int(round((budget - spend) * 100))
                    )
                    skus = skus[affordable]
                    prices = prices[affordable]

                    if not len(skus):
                        break

                    # Product choice balances popularity, affordability,
                    # premium preference, brand loyalty, and repeat habit.
                    utility = (
                        pop[skus]
                        - sensitivity[cid]
                        * np.log(
                            prices / (100 * reference[c])
                        )
                    )
                    utility += (
                        .8
                        * np.log(affluence[cid])
                        * tiers[skus]
                    )
                    utility += (
                        1.3
                        * loyalty[cid]
                        * (
                            brands[skus]
                            == preferred_brand[cid, c]
                        )
                    )
                    utility += (
                        1.8
                        * loyalty[cid]
                        * (skus == last_sku[cid, c])
                    )

                    choice = int(rng.choice(
                        len(skus),
                        p=softmax(utility),
                    ))

                    sku = int(skus[choice])
                    cents = int(prices[choice])

                    if c >= 13:
                        qty = (
                            1
                            + int(
                                mission == 5
                                and rng.random() < .25
                            )
                        )
                    else:
                        mean_extra = (
                            .25 + .22 * (household[cid] - 1)
                        )
                        mean_extra += (
                            2.8 if mission == 5 else 0
                        )
                        mean_extra += (
                            2
                            * sensitivity[cid]
                            * effective_discount[sku]
                        )
                        qty = 1 + rng.negative_binomial(
                            2,
                            2 / (2 + mean_extra),
                        )

                    qty = min(
                        int(qty),
                        24,
                        int(round((budget - spend) * 100))
                        // cents,
                    )

                    if qty < 1:
                        break

                    lines.append((sku, cents, qty))
                    spend += cents * qty / 100
                    last_sku[cid, c] = sku

                    if rng.random() < .08:
                        preferred_brand[cid, c] = brands[sku]

                    # Purchasing more delays the next replenishment need.
                    supply = (
                        cycles[c]
                        * qty
                        * (
                            2 / household[cid]
                            if c < 13
                            else 1
                        )
                    )
                    stock_until[cid, c] = (
                        max(float(d), stock_until[cid, c])
                        + supply * rng.lognormal(-.05, .22)
                    )

                    # Prevent duplicate product lines within the basket.
                    skus = skus[skus != sku]

                # Buying a category changes the odds of its complements.
                for left, right in COMPLEMENTS:
                    if c == left and right not in selected:
                        category_weights[right] *= 3.5
                    elif c == right and left not in selected:
                        category_weights[left] *= 3.5

            # A visit can result in no purchase.
            if not lines:
                continue

            tx_counter += 1
            tid = f"T{tx_counter:010d}"
            stamp = date + pd.Timedelta(
                seconds=int(seconds[pos])
            )

            add_rows(cid, stamp, tid, lines)
            last_visit[cid] = d

            if enable_returns:
                returned = []

                for sku, cents, qty in lines:
                    probability = min(
                        .6,
                        ret_probs[cat_idx[sku]]
                        * (
                            1.25
                            if segment[cid] == 4
                            else 1
                        ),
                    )

                    if rng.random() < probability:
                        returned.append((
                            sku,
                            cents,
                            int(rng.integers(1, qty + 1)),
                        ))

                if returned:
                    delay = int(np.clip(
                        np.ceil(
                            rng.lognormal(2.0, .6)
                        ),
                        1,
                        45,
                    ))

                    if d + delay < len(days):
                        pending[d + delay].append((
                            cid,
                            tid,
                            returned,
                        ))
                    else:
                        censored_returns += len(returned)

            if rng.random() < dropout[cid]:
                active[cid] = False
                churn_day[cid] = d

        # Returns remain possible after purchase attrition.
        # Refunds use original prices, not prices on the return date.
        for cid, original, lines in pending.pop(d, []):
            rid = "R" + original[1:]
            stamp = date + pd.Timedelta(
                seconds=int(
                    rng.integers(9 * 3600, 20 * 3600)
                )
            )

            add_rows(
                cid, stamp, rid, lines, sign=-1
            )

            for sku, cents, qty in lines:
                return_links.append((
                    rid,
                    original,
                    f"C{cid + 1:07d}",
                    pids[sku],
                    qty,
                ))

    df = pd.DataFrame.from_records(
        records,
        columns=COLUMNS,
    )
    df["transaction_date"] = pd.to_datetime(
        df["transaction_date"]
    )
    df["price"] = df["price"].astype(float)
    df["quantity"] = df["quantity"].astype("int64")

    df = df.sort_values([
        "transaction_date",
        "transaction_id",
        "product_id",
    ]).reset_index(drop=True)

    links = pd.DataFrame(return_links, columns=[
        "return_transaction_id",
        "original_transaction_id",
        "customer_id",
        "product_id",
        "returned_quantity",
    ])

    validate(df, links)

    # These are simulation ground truth, not production model features.
    customer_table = pd.DataFrame({
        "customer_id": [
            f"C{i + 1:07d}" for i in range(n)
        ],
        "segment": np.array(SEGMENTS)[segment],
        "household_size": household,
        "affluence_index": affluence,
        "price_sensitivity": sensitivity,
        "brand_loyalty": loyalty,
        "baseline_daily_visit_rate": rates,
        "arrival_date": [
            days[i] for i in arrival
        ],
        "churn_date": [
            days[i] if i >= 0 else pd.NaT
            for i in churn_day
        ],
    })

    sales = df[df.quantity > 0]

    counts = (
        sales.groupby("customer_id")
        .transaction_id.nunique()
    )
    baskets = (
        sales.assign(
            amount=sales.price * sales.quantity
        )
        .groupby("transaction_id")
        .amount.sum()
    )

    report = {
        "seed": seed,
        "start": str(start.date()),
        "end": str(end.date()),
        "currency": "EUR",
        "price_definition": (
            "paid unit price; inclusive of assumed taxes"
        ),
        "timezone": "unspecified local wall-clock time",
        "annual_inflation": annual_inflation,
        "registered_customers": n,
        "purchasing_customers": int(len(counts)),
        "catalog_products": len(catalog),
        "rows": len(df),
        "purchase_transactions": int(
            sales.transaction_id.nunique()
        ),
        "return_transactions": int(
            df.loc[
                df.quantity < 0, "transaction_id"
            ].nunique()
        ),
        "net_revenue": round(
            float((df.price * df.quantity).sum()),
            2,
        ),
        "repeat_customer_fraction_among_buyers": (
            float((counts > 1).mean())
            if len(counts)
            else None
        ),
        "basket_value_quantiles": (
            {
                str(q): float(baskets.quantile(q))
                for q in [.1, .5, .9, .99]
            }
            if len(baskets)
            else {}
        ),
        "return_lines_censored_after_window": (
            censored_returns
        ),
        "warning": (
            "Illustrative behavioral simulation, not empirically "
            "calibrated. Customer latents contain future information; "
            "do not use them as production model features."
        ),
    }

    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        out / "transactions.csv",
        index=False,
        float_format="%.2f",
        date_format="%Y-%m-%d %H:%M:%S",
    )
    catalog.to_csv(
        out / "products.csv",
        index=False,
    )
    customer_table.to_csv(
        out / "customer_latents.csv",
        index=False,
    )
    links.to_csv(
        out / "return_links.csv",
        index=False,
    )
    (out / "diagnostics.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))
    return df, report


def validate(df, links):
    """Check schema, basket integrity, product integrity, and refunds."""

    if (
        list(df.columns) != COLUMNS
        or df.isna().any().any()
    ):
        raise AssertionError(
            "Unexpected schema or missing values."
        )

    if not (
        (df.price > 0).all()
        and (df.quantity != 0).all()
    ):
        raise AssertionError(
            "Invalid unit price or quantity."
        )

    if df.duplicated([
        "transaction_id", "product_id"
    ]).any():
        raise AssertionError(
            "Duplicate product lines in a transaction."
        )

    if not df.empty:
        inconsistent_transactions = (
            df.groupby("transaction_id")[
                ["customer_id", "transaction_date"]
            ].nunique() > 1
        ).any().any()

        if inconsistent_transactions:
            raise AssertionError(
                "A transaction has inconsistent customer or date."
            )

        inconsistent_products = (
            df.groupby("product_id")[
                [
                    "product_description",
                    "department",
                    "category",
                ]
            ].nunique() > 1
        ).any().any()

        if inconsistent_products:
            raise AssertionError(
                "Inconsistent product master data."
            )

        if not (
            (df.quantity < 0)
            == df.transaction_id.str.startswith("R")
        ).all():
            raise AssertionError(
                "Return ID/sign mismatch."
            )

    if not links.empty:
        sales = df[df.quantity > 0]

        original = sales[[
            "transaction_id",
            "product_id",
            "customer_id",
            "price",
            "quantity",
            "transaction_date",
        ]]

        check = links.merge(
            original,
            left_on=[
                "original_transaction_id",
                "product_id",
                "customer_id",
            ],
            right_on=[
                "transaction_id",
                "product_id",
                "customer_id",
            ],
            validate="many_to_one",
        )

        refunds = df[df.quantity < 0][[
            "transaction_id",
            "product_id",
            "customer_id",
            "price",
            "quantity",
            "transaction_date",
        ]]

        check = check.merge(
            refunds,
            left_on=[
                "return_transaction_id",
                "product_id",
                "customer_id",
            ],
            right_on=[
                "transaction_id",
                "product_id",
                "customer_id",
            ],
            suffixes=("_sale", "_refund"),
            validate="one_to_one",
        )

        if (
            len(check) != len(links)
            or len(check) != len(refunds)
        ):
            raise AssertionError(
                "Unmatched return."
            )

        ok = (
            (check.price_sale == check.price_refund)
            & (
                check.returned_quantity
                <= check.quantity_sale
            )
            & (
                check.returned_quantity
                == -check.quantity_refund
            )
            & (
                check.transaction_date_refund
                > check.transaction_date_sale
            )
        )

        if not ok.all():
            raise AssertionError(
                "Return price, quantity, or chronology is inconsistent."
            )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--customers", type=int, default=5000
    )
    parser.add_argument(
        "--start", default="2025-01-01"
    )
    parser.add_argument(
        "--end", default=None, help="End date (default: today)"
    )
    parser.add_argument(
        "--seed", type=int, default=42
    )
    parser.add_argument(
        "--output", default="retail_output"
    )
    parser.add_argument(
        "--annual-inflation", type=float, default=.025
    )
    parser.add_argument(
        "--no-returns", action="store_true"
    )

    args = parser.parse_args()

    generate(
        customers=args.customers,
        start=args.start,
        end=args.end,
        seed=args.seed,
        output=args.output,
        annual_inflation=args.annual_inflation,
        enable_returns=not args.no_returns,
    )


if __name__ == "__main__":
    main()