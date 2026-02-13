# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def split_common_and_diff(
    shorter_df_that_you_subtract: pd.DataFrame,
    longer_df_from_which_you_subtract: pd.DataFrame,
    on: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    shorter_df_that_you_subtract = d1
    longer_df_from_which_you_subtract = d2

    Return two DataFrames:
      - common: rows from df1 that match df2 on given columns
      - diff: rows from df2 that do NOT match df1 on those columns

    Parameters
    ----------
    df1, df2 : pd.DataFrame
        DataFrames to compare.
    on : list[str] | None
        Columns to match on. If None, uses intersection of columns.

    Example
    -------
    >>> df1 = pd.DataFrame({'a':[1,2], 'b':[4,5], 'c':[0,0]})
    >>> df2 = pd.DataFrame({'a':[1,2,3], 'b':[4,5,5], 'c':[1,1,1]})
    >>> common, diff = split_common_and_diff(df1, df2, on=['a','b'])
    >>> common.equals(df1)  # True
    >>> diff
       a  b  c
    0  3  5  1
    """

    if len(longer_df_from_which_you_subtract) < len(shorter_df_that_you_subtract):
        logging.warning(
            f"Warning, you are finding the rows of a dataframe of len={len(longer_df_from_which_you_subtract)}"
            f"That are also in a dataset of len = {len(shorter_df_that_you_subtract)}"
        )

    if on is None:
        on = list(
            set(shorter_df_that_you_subtract.columns)
            & set(longer_df_from_which_you_subtract.columns)
        )
    # Common rows: those in df1 whose keys exist in df2
    common_keys = longer_df_from_which_you_subtract[on].drop_duplicates()
    common = shorter_df_that_you_subtract.merge(common_keys, on=on, how="inner")
    # Diff rows: those in df2 whose keys do NOT exist in df1
    diff = longer_df_from_which_you_subtract.merge(
        shorter_df_that_you_subtract[on].drop_duplicates(),
        on=on,
        how="left",
        indicator=True,
    )
    diff = diff[diff["_merge"] == "left_only"].drop(columns=["_merge"])

    return common, diff
