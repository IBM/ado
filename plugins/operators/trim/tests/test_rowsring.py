# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import pandas as pd
import pytest
from trim.utils.rowsring import RowsRing


def test_append_and_index_reset_matches_spec() -> None:
    # NOTE: these tests must be passed by the structure (as provided)
    yielded_rows = RowsRing(maxlen=3)

    yielded_rows += pd.DataFrame({"id": 3, "value": 13.0}, index=[2])
    yielded_rows += pd.DataFrame({"id": 3, "value": 13.0}, index=[3])
    yielded_rows += pd.DataFrame({"id": 4, "value": 13.0}, index=[4])
    yielded_rows += pd.DataFrame({"id": 5, "value": 13.0}, index=[5])

    # indices of the df added are discarded; .df resets to 0..size-1
    e2 = pd.DataFrame(
        {"id": [3.0, 4.0, 5.0], "value": [13.0, 13.0, 13.0]}, index=[0, 1, 2]
    )
    assert yielded_rows.df.equals(e2)

    # another append; capacity=3, oldest drops, contents are [4,5,3]; index reset to 0..2
    yielded_rows += pd.DataFrame({"id": 3, "value": 13.0}, index=[2])

    enext = pd.DataFrame(
        {"id": [4.0, 5.0, 3.0], "value": [13.0, 13.0, 13.0]}, index=[0, 1, 2]
    )
    assert yielded_rows.df.equals(enext)


def test_iadd_keeps_instance_type() -> None:
    ring = RowsRing(maxlen=1)
    ring += pd.DataFrame({"id": 1, "value": 1.0}, index=[10])
    # Ensure `+=` didn't turn the object into a DataFrame
    assert isinstance(ring, RowsRing)


def test_capacity_fifo_behavior() -> None:
    ring = RowsRing(maxlen=2)
    ring += pd.DataFrame({"id": 1, "value": 1.0}, index=[0])
    ring += pd.DataFrame({"id": 2, "value": 2.0}, index=[1])
    ring += pd.DataFrame({"id": 3, "value": 3.0}, index=[2])  # evicts the oldest (id=1)

    expected = pd.DataFrame({"id": [2.0, 3.0], "value": [2.0, 3.0]}, index=[0, 1])
    assert ring.df.equals(expected)


def test_schema_mismatch_raises() -> None:
    ring = RowsRing(maxlen=2)
    ring += pd.DataFrame({"id": 1, "value": 1.0}, index=[0])
    with pytest.raises(ValueError, match="mismatch"):
        ring += pd.DataFrame({"id": 2, "value": 2.0, "extra": 99}, index=[1])


def test_multi_row_dataframe_raises() -> None:
    ring = RowsRing(maxlen=3)
    with pytest.raises(ValueError, match="one row"):
        ring += pd.DataFrame({"id": [1, 2], "value": [1.0, 2.0]})  # >1 row not allowed


def test_accepts_dict_and_series() -> None:
    ring = RowsRing(maxlen=2)
    ring += {"id": 1, "value": 1.0}  # dict
    ring += pd.Series({"id": 2, "value": 2.0})  # series

    expected = pd.DataFrame({"id": [1.0, 2.0], "value": [1.0, 2.0]}, index=[0, 1])
    assert ring.df.equals(expected)


def test_columns_order_is_canonical() -> None:
    # Provide row with columns out of order; ring should reorder to canonical order
    ring = RowsRing(maxlen=2)
    ring += pd.DataFrame({"id": 10, "value": 1.5}, index=[0])
    # Swap order of columns in input
    ring += pd.Series({"value": 2.5, "id": 11})

    df = ring.df
    assert list(df.columns) == ["id", "value"]
    assert df.iloc[1].to_dict() == {"id": 11.0, "value": 2.5}


def test_empty_df_has_known_columns_after_first_append() -> None:
    ring = RowsRing(maxlen=2)
    # Initial empty df has no columns
    df0 = ring.df
    assert list(df0.columns) == []

    ring += {"id": 1, "value": 2.0}
    # Columns are established and persist
    df1 = ring.df
    assert list(df1.columns) == ["id", "value"]


def test_getitem_and_len_and_to_list() -> None:
    ring = RowsRing(maxlen=3)
    ring += {"id": 1, "value": 1.0}
    ring += {"id": 2, "value": 2.0}
    ring += {"id": 3, "value": 3.0}

    assert len(ring) == 3
    s = ring[1]
    assert isinstance(s, pd.Series)
    assert s["id"] == 2

    lst = ring.to_list()
    assert len(lst) == 3
    assert lst[0]["id"] == 1
    assert lst[-1]["id"] == 3


def test_validator_hook_blocks_invalid_rows() -> None:
    # Validator that prevents duplicate id values
    def no_duplicate_id(prev_df: pd.DataFrame, new_row: pd.Series) -> bool:
        if prev_df.empty:
            return True
        return new_row["id"] not in set(prev_df["id"])

    ring = RowsRing(maxlen=3, validator=no_duplicate_id)
    ring += {"id": 1, "value": 1.0}
    ring += {"id": 2, "value": 2.0}

    # Adding a duplicate id should raise
    with pytest.raises(ValueError, match="failed"):
        ring += {"id": 2, "value": 99.0}

    # Adding a new id is fine
    ring += {"id": 3, "value": 3.0}

    expected = pd.DataFrame(
        {"id": [1.0, 2.0, 3.0], "value": [1.0, 2.0, 3.0]}, index=[0, 1, 2]
    )
    assert ring.df.equals(expected)
