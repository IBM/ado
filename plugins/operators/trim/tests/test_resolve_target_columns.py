# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import pandas as pd
from trim.samplers.no_priors_utils import _resolve_target_columns

from ado.schema.virtual_property import PropertyAggregationMethodEnum

MEAN = PropertyAggregationMethodEnum.mean


def test_aggregated_column_renamed_to_target() -> None:
    """Should expose '<target>-mean' under the bare target identifier."""
    df = pd.DataFrame({"x": [1, 2], "pressure-mean": [10.0, 20.0]})

    resolved, found = _resolve_target_columns(df, ["pressure"], MEAN)

    assert found == ["pressure"]
    assert list(resolved.columns) == ["x", "pressure"]
    assert resolved["pressure"].tolist() == [10.0, 20.0]


def test_aggregated_and_bare_columns_merged() -> None:
    """Should keep values of rows that were not aggregated."""
    df = pd.DataFrame(
        {
            "x": [1, 2, 3],
            "pressure": [None, 5.0, None],
            "pressure-mean": [10.0, None, 30.0],
        }
    )

    resolved, found = _resolve_target_columns(df, ["pressure"], MEAN)

    assert found == ["pressure"]
    assert list(resolved.columns) == ["x", "pressure"]
    assert resolved["pressure"].tolist() == [10.0, 5.0, 30.0]


def test_bare_column_kept() -> None:
    """Should keep a bare target column when no aggregated column exists."""
    df = pd.DataFrame({"x": [1], "pressure": [5.0]})

    resolved, found = _resolve_target_columns(df, ["pressure"], MEAN)

    assert found == ["pressure"]
    assert resolved.equals(df)


def test_missing_target_not_found() -> None:
    """Should omit targets absent from the table and leave the input unchanged."""
    df = pd.DataFrame({"x": [1], "other-mean": [1.0]})

    resolved, found = _resolve_target_columns(df, ["pressure"], MEAN)

    assert found == []
    assert list(resolved.columns) == ["x", "other-mean"]
    assert list(df.columns) == ["x", "other-mean"]
