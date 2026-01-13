# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT
import typing

if typing.TYPE_CHECKING:
    import pandas as pd


def reorder_dataframe_columns(
    df: "pd.DataFrame", move_to_start: list[str], move_to_end: list[str]
) -> "pd.DataFrame":
    """
    Reorder the columns of a pandas DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to reorder the columns of.
        move_to_start (list[str]): A list of column names to move to the start of the DataFrame.
        move_to_end (list[str]): A list of column names to move to the end of the DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with the columns reordered.
    """
    known_columns = move_to_start + move_to_end
    property_columns = [c for c in df.columns if c not in known_columns]
    ordered_columns = move_to_start + property_columns + move_to_end
    return df.reindex(columns=ordered_columns)


def filter_dataframe_columns(
    df: "pd.DataFrame", columns_to_keep: list[str]
) -> "pd.DataFrame":
    """
    Filters a pandas DataFrame based on a list of column names to keep.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns_to_keep (list[str]): A list of column names to keep.

    Returns:
        pd.DataFrame: A new DataFrame with only the specified columns.

    Raises:
        ValueError: If the list of columns to keep is empty.
    """
    if len(columns_to_keep) == 0:
        raise ValueError("Cannot drop all columns from the dataframe")

    columns_to_drop = [col for col in df.columns if col not in columns_to_keep]
    return df.drop(columns=columns_to_drop)
