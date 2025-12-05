# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

"""
test_high_dimensional_sampling.py

===========================================================
HOW TO USE THIS TEST MODULE:
-----------------------------------------------------------
- Define your datasets and expected ordering in helper functions.
- Each test prints:
    * Original DataFrame
    * Expected ordering
    * Actual ordering
    * Intermediate steps (orders, indices)
- Run with:
    pytest -s -vv test_high_dimensional_sampling.py
  to see all pretty-printed outputs in the terminal.
===========================================================
"""

import itertools
import logging
from pprint import pprint

import pandas as pd
import trim.utils.order as order  # Replace with your actual module path

# ============================================================
# Helper: Define datasets and expected ordering
# ============================================================


def make_full_grid_df():
    """
    Build a full Cartesian grid for properties:
    A in [0, 1], B in [0, 1, 2]
    plus an explicit duplicate to test filtering.
    """
    A = [0, 1]
    B = [0, 1, 2]
    rows = [(a, b) for a, b in itertools.product(A, B)]
    rows.append((1, 2))  # duplicate
    return pd.DataFrame(rows, columns=["A", "B"])


# ============================================================
# TESTS
# ============================================================


def test_order_df_for_get_index_list_nn_high_dimensional_sorts_lexicographically():
    df = pd.DataFrame(
        [
            {"A": 1, "B": 2},
            {"A": 0, "B": 1},
            {"A": 1, "B": 0},
            {"A": 0, "B": 0},
            {"A": 0, "B": 2},
            {"A": 1, "B": 1},
        ]
    )
    constitutive_properties = ["A", "B"]

    ordered = order.order_df_for_get_index_list_nn_high_dimensional(
        df.copy(), constitutive_properties, dims=[2, 3]
    )

    expected = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    got = list(map(tuple, ordered[constitutive_properties].values))

    # Pretty print
    print("\n=== TEST: Lexicographic Ordering ===")
    print("Original DataFrame:")
    print(df.to_string(index=False))
    print("\nExpected Order:")
    pprint(expected)
    print("\nActual Order:")
    pprint(got)

    assert got == expected


def test_get_index_list_nn_high_dimensional_linear_indexing():
    """
    For properties A in [0,1] and B in [0,1,2], verify that the mapping computes:
      index = b + a * len(B)
    With orders ordered as [[0,0],[0,1],[0,2],[1,0],[1,1],[1,2]]
    The indices must be [0,1,2,3,4,5].
    """
    orders_to_sample = [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]]
    dims = [2, 3]  # A has 2 values, B has 3 values

    indices = order.get_index_list_nn_high_dimensional(orders_to_sample, dims)
    expected_indices = [0, 1, 2, 3, 4, 5]

    print("\n=== TEST: Index Mapping ===")
    print("Orders to Sample:")
    pprint(orders_to_sample)
    print("\nDims:", dims)
    print("Expected Indices:", expected_indices)
    print("Actual Indices:", indices)

    assert indices == expected_indices


def test_order_df_for_sampling_with_no_priors_integration_and_order_preservation(
    monkeypatch, caplog
):
    caplog.set_level(logging.WARNING)
    df = make_full_grid_df()
    constitutive_properties = ["A", "B"]

    def fake_get_order_list_nn_high_dimensional(dims, space, n, refined):
        return [[1, 2], [0, 0]]

    monkeypatch.setattr(
        order,
        "get_order_list_nn_high_dimensional",
        fake_get_order_list_nn_high_dimensional,
    )

    sampled_df = order.order_df_for_sampling_with_no_priors(
        df=df, constitutive_properties=constitutive_properties, n=2, refined=True
    )

    got = list(map(tuple, sampled_df[constitutive_properties].values))
    expected = [(1, 2), (0, 0)]

    print("\n=== TEST: Integration & Order Preservation ===")
    print("Original DataFrame:")
    print(df.to_string(index=False))
    print("\nExpected Sampled Rows:")
    pprint(expected)
    print("\nActual Sampled Rows:")
    pprint(got)

    assert got == expected
    assert "Removing 1 duplicate configurations" in caplog.text


def test_order_df_for_sampling_with_no_priors_adjusts_n(monkeypatch, caplog):
    caplog.set_level(logging.WARNING)
    df = make_full_grid_df()
    constitutive_properties = ["A", "B"]

    def fake_get_order_list_nn_high_dimensional(dims, space, n, refined):
        return [[a, b] for a, b in itertools.product([0, 1], [0, 1, 2])]

    monkeypatch.setattr(
        order,
        "get_order_list_nn_high_dimensional",
        fake_get_order_list_nn_high_dimensional,
    )

    sampled_df = order.order_df_for_sampling_with_no_priors(
        df=df, constitutive_properties=constitutive_properties, n=10, refined=False
    )

    expected = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    got = list(map(tuple, sampled_df[constitutive_properties].values))

    print("\n=== TEST: Adjust n ===")
    print("Original DataFrame:")
    print(df.to_string(index=False))
    print("\nExpected Rows:")
    pprint(expected)
    print("\nActual Rows:")
    pprint(got)

    assert got == expected
    assert (
        "Requested 10 samples, but DataFrame has only 6 rows. Adjusting n."
        in caplog.text
    )


def test_iloc_preserves_order_in_indices_to_sample(monkeypatch):
    df = pd.DataFrame({"X": [0, 1]})
    constitutive_properties = ["X"]

    def fake_get_order_list_nn_high_dimensional(dims, space, n, refined):
        return [[1], [0]]

    monkeypatch.setattr(
        order,
        "get_order_list_nn_high_dimensional",
        fake_get_order_list_nn_high_dimensional,
    )

    sampled_df = order.order_df_for_sampling_with_no_priors(
        df=df, constitutive_properties=constitutive_properties, n=2, refined=False
    )

    print("\n=== TEST: iloc Preserves Order ===")
    print("Original DataFrame:")
    print(df.to_string(index=False))
    print("\nExpected Order: [1, 0]")
    print("Actual Order:", list(sampled_df["X"].values))
