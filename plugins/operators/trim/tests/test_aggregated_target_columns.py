# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import pandas as pd
import pytest
from trim.samplers.no_priors_utils import _resolve_target_columns

from ado.schema.virtual_property import PropertyAggregationMethodEnum

MEAN = PropertyAggregationMethodEnum.mean


@pytest.mark.parametrize(
    "method",
    [PropertyAggregationMethodEnum.mean, PropertyAggregationMethodEnum.median],
)
def test_aggregated_column_renamed_to_target(
    method: PropertyAggregationMethodEnum,
) -> None:
    """Should expose '<target>-<method>' under the bare target identifier."""
    df = pd.DataFrame({"x": [1, 2], f"pressure-{method.value}": [10.0, 20.0]})
    original = df.copy()

    resolved, found_targets = _resolve_target_columns(df, ["pressure"], method)

    assert found_targets == ["pressure"]
    assert list(resolved.columns) == ["x", "pressure"]
    assert resolved["pressure"].tolist() == [10.0, 20.0]
    pd.testing.assert_frame_equal(df, original)


def test_aggregated_column_of_other_method_ignored() -> None:
    """Should not treat a column aggregated with another method as the target."""
    df = pd.DataFrame({"x": [1], "pressure-median": [10.0]})

    resolved, found_targets = _resolve_target_columns(df, ["pressure"], MEAN)

    assert found_targets == []
    assert list(resolved.columns) == ["x", "pressure-median"]


def test_aggregated_and_bare_columns_merged() -> None:
    """Should fill the target from the bare column where no aggregated value exists."""
    df = pd.DataFrame(
        {
            "x": [1, 2, 3, 4],
            "pressure": [None, 5.0, None, 7.0],
            "pressure-mean": [10.0, None, 30.0, 40.0],
        }
    )
    original = df.copy()

    resolved, found_targets = _resolve_target_columns(df, ["pressure"], MEAN)

    assert found_targets == ["pressure"]
    assert list(resolved.columns) == ["x", "pressure"]
    assert resolved["pressure"].tolist() == [10.0, 5.0, 30.0, 40.0]
    pd.testing.assert_frame_equal(df, original)


def test_bare_column_kept() -> None:
    """Should keep a bare target column when no aggregated column exists."""
    df = pd.DataFrame({"x": [1], "pressure": [5.0]})

    resolved, found_targets = _resolve_target_columns(df, ["pressure"], MEAN)

    assert found_targets == ["pressure"]
    pd.testing.assert_frame_equal(resolved, df)


def test_missing_target_not_found() -> None:
    """Should omit targets absent from the table."""
    df = pd.DataFrame({"x": [1], "other-mean": [1.0]})

    resolved, found_targets = _resolve_target_columns(df, ["pressure"], MEAN)

    assert found_targets == []
    assert list(resolved.columns) == ["x", "other-mean"]
