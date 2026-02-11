# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import itertools
import logging
import math
from typing import Literal

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor

from trim.trim_pydantic import AutoGluonArgs
from trim.utils.high_dimensional_sampling import get_sampling_indices_multi_dimensional
from trim.utils.miscellaneous import delete_dir

logger = logging.getLogger(__name__)


def get_feature_importance_order(
    source_df: pd.DataFrame,
    target_output: str,
    min_measured_entities: int,
    autoGluonArgs: AutoGluonArgs,
) -> tuple[tuple[str, ...], dict[str, float]]:
    """
    Train a TabularPredictor on the source space and return:
      - ordered_tuple_most_important_first: tuple of features sorted by importance
      - importance_dict: feature -> importance score

    Minimal checks:
      - source_df has at least min_measured_entities rows
      - 'identifier' is dropped if present
      - model directory is removed after use
    """
    # 1) Guardrail
    if len(source_df) < min_measured_entities:
        msg = (
            f"Not enough measured entities: {len(source_df)} < {min_measured_entities}"
        )
        logger.warning(msg)

    # 2) Train predictor
    train_df = source_df.drop(
        columns=[c for c in ["identifier"] if c in source_df.columns]
    )

    predictor = TabularPredictor(
        label=target_output, **autoGluonArgs.tabularPredictorArgs
    )

    logger.info(
        f"Fitting AutoGluon TabularPredictor; train cols: {list(train_df.columns)}"
    )
    predictor.fit(train_data=train_df, **autoGluonArgs.fitArgs)

    # 3) Feature importances
    fi_df = predictor.feature_importance(train_df).sort_values(
        "importance", ascending=False
    )
    importance_dict = fi_df["importance"].to_dict()
    ordered_tuple_most_important_first = tuple(fi_df.index)

    logger.info(f"Top features: {list(ordered_tuple_most_important_first[:5])}")

    # 4) Cleanup model directory
    logger.info(f"AutoGluon model directory: {predictor.path}")
    delete_dir(predictor.path)
    del predictor

    return ordered_tuple_most_important_first, importance_dict


def reorder_df_by_importance(
    df: pd.DataFrame,
    importance_feature_list: tuple[str, ...] | list[str],
) -> pd.DataFrame:
    """
    Reorder df rows by feature importance (descending order priority of columns).
    Minimal checks only:
      - ensure importance_feature_list is not empty
      - warn if some features are missing in df
      - sort by the features that exist in df
    """
    if not importance_feature_list:
        raise ValueError("importance_feature_list is empty.")

    missing = [c for c in importance_feature_list if c not in df.columns]
    if missing:
        logger.error("Columns not present in target df: %s", missing)

    sort_cols = [c for c in importance_feature_list if c in df.columns]
    if not sort_cols:
        raise ValueError("None of the importance features are present in df.")

    return df.sort_values(by=sort_cols).reset_index(drop=True)


def order_df_for_sampling_with_no_priors(
    df: pd.DataFrame,
    constitutive_properties: list[str],
    n: int,
    strategy: Literal["random", "clhs", "sobol"],
) -> pd.DataFrame:
    """
    Orders a DataFrame for high-dimensional sampling without prior knowledge.

    Args:
        df (pd.DataFrame):
            The input dataset containing at least the columns specified in `constitutive_properties`.
            May contain duplicate configurations across the constitutive properties, which will
            be removed before sampling.
        constitutive_properties (list[str]):
            Column names that define the configuration space (axes of the high-dimensional grid).
            Uniqueness is enforced over the Cartesian product of these properties.
        n (int):
            Number of samples (orders) to generate. If larger than the deduplicated DataFrame length,
            it is reduced to fit and a warning is logged.
        strategy (str):
            Sampling subroutine identifier. Passed through to `get_order_list_nn_high_dimensional`;
            see that function's documentation for supported strategies and behavior.

    Returns:
        pd.DataFrame:
            A view of the ordered, deduplicated DataFrame (`df_unique`) restricted to the rows
            at `indices_to_sample`. The returned DataFrame preserves the column schema of `df`
            and has `n` rows (after any adjustment). Index is positional (0..n-1) because
            `.iloc` is used and `df_unique` was reset with `drop=True`.

    Steps:

    0. Filter dataset so that for each combination of constitutive properties you only have one row
    1. Extract unique values for each constitutive property.
    2. Build:
        - value_dict_unordered: keys = properties, values = unique unordered lists.
        - value_dict: same as above but ordered ascending.
        - space_dict: keys = properties, values = length of each list.
        - dimensions: list of lengths (dimensionality).
    3. Order the DataFrame so that index mapping aligns with high-dimensional sampling.
    4. Generate orders_to_sample using get_order_list_nn_high_dimensional().
    5. Map these orders to actual DataFrame indices.
    6. Return a DataFrame with rows corresponding to sampled indices.
        Row ith is the row corresponding to indeces_to_sample[i]

    If n > len(df), log a warning and adjust n to min(n, len(df)).
    """

    # Filtering
    len_original = len(df)
    df_unique = df.drop_duplicates(subset=constitutive_properties).reset_index(
        drop=True
    )
    delta_len = len_original - len(df_unique)
    if delta_len > 0:
        logging.warning(
            f"Removing {delta_len} duplicate configurations."
            f"They are characterized by the same combination of constitutive properties = {constitutive_properties}"
        )

    if n > len(df_unique):
        logging.warning(
            f"Requested {n} samples, but DataFrame has only {len(df_unique)} rows. Adjusting n."
        )
        n = min(n, len(df_unique))

    # Build dictionaries
    def _get_sorted_uniques(prop: str) -> list:
        """Helper to safely sort unique values for a property."""
        # Note: using set() handles duplicates, but be aware that set({nan, nan})
        # can result in multiple NaNs. consider df_unique[prop].unique() for safer handling.
        vals = set(df_unique[prop].values)
        try:
            return sorted(vals)
        except TypeError:
            logging.warning(
                f"Cannot sort mixed types for property '{prop}'. "
                "Keeping original order (it may be inconsistent due to the use of sets)."
            )
            return list(vals)

    value_dict = {prop: _get_sorted_uniques(prop) for prop in constitutive_properties}

    space_dict = {prop: len(vals) for prop, vals in value_dict.items()}

    dimensions = list(space_dict.values())

    # Order DataFrame for index mapping
    # NOTE: just added .reset_index(drop=True)
    df_unique = order_df_for_get_index_list_nn_high_dimensional(
        df_unique, constitutive_properties, dimensions=dimensions
    ).reset_index(drop=True)

    # Generate sampling orders
    orders_to_sample = get_sampling_indices_multi_dimensional(
        dimensions=dimensions, space=space_dict, n=n, strategy=strategy
    )

    # Map orders to DataFrame indices
    indices_to_sample = get_index_list_nn_high_dimensional(orders_to_sample, dimensions)

    logger.info(f"Indexes are:\n {indices_to_sample}")
    try:
        return df_unique.iloc[indices_to_sample]
    except IndexError:
        logging.error(
            f"Index Error detected. Length of the dataframe is {len(df_unique)}."
            "The indices that cause the error are:"
        )
        max_len = len(df_unique)
        out_of_bounds_list = [i for i in indices_to_sample if i < 0 or i >= max_len]

        logging.error(out_of_bounds_list)
        logging.error("Returning empty dataset")
        return pd.DataFrame({})


def order_df_for_get_index_list_nn_high_dimensional(
    df: pd.DataFrame, constitutive_properties: list[str], dimensions: list[int]
) -> pd.DataFrame:
    """
    Ensure a DataFrame is ordered and complete for high-dimensional index generation.

    This utility prepares `df` so that its rows align with the Cartesian product
    implied by `constitutive_properties` and `dimensions`. Specifically:

    1. Sort rows by the provided constitutive properties in the given order.
    2. Validate that the DataFrame length matches the expected size:
        `expected_len = product(dimensions)`.
    3. If rows are missing:
        - Log a warning.
        - Generate all possible combinations of unique values for each constitutive property.
        - Identify missing combinations and inject them as new rows.
            Non-constitutive columns in injected rows are filled with `NaN`.
    4. Return the augmented and re-sorted DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing at least the columns listed in `constitutive_properties`.
    constitutive_properties : list[str]
        Column names that define the high-dimensional space (e.g., factors or grid axes).
        The order determines the sort priority.
    dimensions : list[int]
        Expected cardinality for each constitutive property. Used to compute
        `expected_len = math.prod(dimensions)` for consistency checks.

    Returns
    -------
    pd.DataFrame
        A DataFrame sorted by `constitutive_properties` and augmented with any missing
        combinations, ensuring coverage of the full Cartesian product implied by `dimensions`.

    Notes
    -----
    - Injected rows will have `NaN` for all non-constitutive columns.
    - If `dimensions` and the actual unique values in `df` disagree, the function uses
        observed unique values to generate combinations.
    - This function is useful for downstream routines that assume a complete
        and ordered representation of the sampling space.

    Examples
    --------
    >>> df = pd.DataFrame({'A': [1, 2], 'B': ['x', 'y'], 'value': [0.1, 0.2]})
    >>> order_df_for_get_index_list_nn_high_dimensional(df, ['A', 'B'], [2, 2])
    # Returns a DataFrame with 4 rows covering all (A,B) pairs, sorted by A then B.
    """

    # Sort by constitutive properties
    df = df.sort_values(by=constitutive_properties).reset_index(drop=True)

    expected_len = math.prod(dimensions)
    if len(df) != expected_len:
        logger.warning(
            f"DataFrame length mismatch: expected {expected_len} (product of {dimensions}), "
            f"but got {len(df)}."
        )

        # Generate all possible combinations of constitutive properties
        unique_values = [
            sorted(df[prop].dropna().unique()) for prop in constitutive_properties
        ]
        all_combinations = list(itertools.product(*unique_values))

        # Identify existing combinations
        existing_combinations = {
            tuple(row[prop] for prop in constitutive_properties)
            for _, row in df.iterrows()
        }

        # Find missing combinations
        missing_combinations = [
            comb for comb in all_combinations if comb not in existing_combinations
        ]

        if missing_combinations:
            logger.info(
                f"Injecting {len(missing_combinations)} missing rows to satisfy the property."
            )
            injected_rows = []
            for comb in missing_combinations:
                row_data = dict(zip(constitutive_properties, comb, strict=False))
                # Fill other columns with NaN
                for col in df.columns:
                    if col not in constitutive_properties:
                        row_data[col] = pd.NA
                injected_rows.append(row_data)

            # Append missing rows
            df = pd.concat([df, pd.DataFrame(injected_rows)], ignore_index=True)

            # Sort again after injection
            df = df.sort_values(by=constitutive_properties).reset_index(drop=True)

            logger.info(f"Injected rows: {injected_rows}")

    return df


def get_index_list_nn_high_dimensional(
    orders_to_sample: list[list[int]], dimensions: list[int]
) -> list[int]:
    """
    Map high-dimensional sampling orders to linear (flattened) indices.

    Converts multi-dimensional coordinates to linear indices using row-major ordering,
    where the last dimension varies fastest.

    Args:
        orders_to_sample: List of multi-dimensional coordinates [i0, i1, ..., ik]
        dimensions: Size of each dimension [d0, d1, ..., dk]

    Returns:
        List of linear indices corresponding to the input coordinates

    Warns:
        If duplicate or out-of-bounds indices are detected
    """
    indices = []
    cprod = np.cumprod(np.array(dimensions), dtype=int).tolist()
    maximum_n = cprod[-1]

    for order in orders_to_sample:
        index = 0
        multiplier = 1
        # Iterate reversed so last dimension varies fastest
        for i in reversed(range(len(dimensions))):
            index += order[i] * multiplier
            multiplier *= dimensions[i]

        if index > maximum_n:
            logging.warning(
                f"Out of bound index {index} computed from order {order}, dimensions are {dimensions}"
            )
        indices.append(index)

    if len(set(indices)) != len(indices):
        logger.error(f"{len(indices) - len(set(indices))} Duplicated indices!")

    out_of_bounds_list = [i for i in indices if i > maximum_n]
    if out_of_bounds_list:
        logger.error(
            f"The following indices are out of bound: {out_of_bounds_list}, maximum admissible value is {maximum_n-1}"
        )

    return indices
