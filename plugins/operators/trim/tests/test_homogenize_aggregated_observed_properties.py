# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Tests for homogenize_aggregated_observed_properties()."""

import pandas as pd
import pytest
from trim.samplers.no_priors_utils import homogenize_aggregated_observed_properties

# ---------------------------------------------------------------------------
# Branch 1 — el already present, no {el}-mean column → no-op
# ---------------------------------------------------------------------------


def test_el_present_no_mean_col_unchanged() -> None:
    """When el exists and no {el}-mean column is present, the DataFrame is unchanged."""
    df = pd.DataFrame({"a": [1, 2], "may_fail-value": [0.5, 0.8]})
    result = homogenize_aggregated_observed_properties(df.copy(), "may_fail-value")
    pd.testing.assert_frame_equal(result, df)


# ---------------------------------------------------------------------------
# Branch 2 — only {el}-mean present, el absent → rename
# ---------------------------------------------------------------------------


def test_only_mean_col_present_renames() -> None:
    """When only {el}-mean exists, it should be renamed to el."""
    df = pd.DataFrame({"a": [1, 2], "may_fail-value-mean": [0.5, 0.8]})
    result = homogenize_aggregated_observed_properties(df, "may_fail-value")
    assert "may_fail-value" in result.columns
    assert "may_fail-value-mean" not in result.columns
    assert list(result["may_fail-value"]) == [0.5, 0.8]


def test_only_mean_col_other_cols_untouched() -> None:
    """Renaming {el}-mean to el must not affect other columns."""
    df = pd.DataFrame({"a": [10, 20], "may_fail-value-mean": [1.0, 2.0], "b": [3, 4]})
    result = homogenize_aggregated_observed_properties(df, "may_fail-value")
    assert list(result.columns) == ["a", "may_fail-value", "b"]


# ---------------------------------------------------------------------------
# Branch 3 — both {el}-mean and el present:
# {el}-mean overwrites el wherever {el}-mean is non-null;
# pre-existing el is kept only where {el}-mean is null/empty.
# ---------------------------------------------------------------------------


def test_both_present_mean_overwrites_el_where_mean_set() -> None:
    """{el}-mean overwrites el on every row where {el}-mean is non-null."""
    df = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "may_fail-value": [0.5, 0.6, None],
            "may_fail-value-mean": [0.9, 0.7, 0.8],
        }
    )
    result = homogenize_aggregated_observed_properties(df, "may_fail-value")
    assert result["may_fail-value"].iloc[0] == pytest.approx(0.9)
    assert result["may_fail-value"].iloc[1] == pytest.approx(0.7)
    assert result["may_fail-value"].iloc[2] == pytest.approx(0.8)


def test_both_present_el_kept_where_mean_is_null() -> None:
    """Pre-existing el value is preserved only where {el}-mean is null."""
    df = pd.DataFrame(
        {
            "a": [1, 2],
            "may_fail-value": [0.5, 0.6],
            "may_fail-value-mean": [None, 0.9],
        }
    )
    result = homogenize_aggregated_observed_properties(df, "may_fail-value")
    # row 0: mean is null → keep original el = 0.5
    assert result["may_fail-value"].iloc[0] == pytest.approx(0.5)
    # row 1: mean is non-null → overwrite with 0.9
    assert result["may_fail-value"].iloc[1] == pytest.approx(0.9)


def test_both_present_mean_col_dropped() -> None:
    """After merging, {el}-mean column must not remain in the DataFrame."""
    df = pd.DataFrame(
        {
            "a": [1, 2],
            "may_fail-value": [0.5, None],
            "may_fail-value-mean": [0.9, 0.8],
        }
    )
    result = homogenize_aggregated_observed_properties(df, "may_fail-value")
    assert "may_fail-value-mean" not in result.columns


def test_both_present_all_mean_null_el_unchanged() -> None:
    """When all {el}-mean rows are null, el is entirely unchanged."""
    df = pd.DataFrame(
        {
            "a": [1, 2],
            "may_fail-value": [0.1, 0.2],
            "may_fail-value-mean": [None, None],
        }
    )
    result = homogenize_aggregated_observed_properties(df, "may_fail-value")
    assert list(result["may_fail-value"]) == pytest.approx([0.1, 0.2])
    assert "may_fail-value-mean" not in result.columns


def test_both_present_el_null_mean_null_row_stays_null() -> None:
    """When both el and {el}-mean are null for a row, el remains null."""
    df = pd.DataFrame(
        {
            "a": [1],
            "may_fail-value": [None],
            "may_fail-value-mean": [None],
        }
    )
    result = homogenize_aggregated_observed_properties(df, "may_fail-value")
    assert pd.isna(result["may_fail-value"].iloc[0])
    assert "may_fail-value-mean" not in result.columns


# ---------------------------------------------------------------------------
# Unrelated columns are never touched
# ---------------------------------------------------------------------------


def test_unrelated_columns_untouched() -> None:
    """Columns unrelated to el are never modified."""
    df = pd.DataFrame(
        {
            "x": [1, 2],
            "may_fail-value-mean": [0.5, 0.6],
            "other-mean": [7.0, 8.0],
        }
    )
    result = homogenize_aggregated_observed_properties(df, "may_fail-value")
    assert "other-mean" in result.columns
    assert list(result["other-mean"]) == [7.0, 8.0]


# ---------------------------------------------------------------------------
# el not found anywhere → DataFrame unchanged, no KeyError
# ---------------------------------------------------------------------------


def test_el_not_found_returns_unchanged() -> None:
    """When el is not present in any form, the DataFrame is returned unchanged."""
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    result = homogenize_aggregated_observed_properties(df.copy(), "may_fail-value")
    pd.testing.assert_frame_equal(result, df)
