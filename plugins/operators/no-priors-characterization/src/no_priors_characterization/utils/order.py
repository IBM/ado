# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import itertools
import logging
import math
from typing import Literal

import numpy as np
import pandas as pd

from no_priors_characterization.utils.high_dimensional_sampling import (
    get_sampling_indices_multi_dimensional,
)

logger = logging.getLogger(__name__)


def order_df_for_sampling_with_no_priors(
    df: pd.DataFrame,
    constitutive_properties: list[str],
    n: int,
    strategy: Literal["random", "clhs", "sobol"],
) -> pd.DataFrame:
    """
    Orders a DataFrame for high-dimensional sampling without prior knowledge.

    Deduplicates rows based on constitutive properties, orders them for sampling,
    and returns a subset of n samples using the specified strategy.

    Args:
        df: Input dataset containing at least the columns specified in
            constitutive_properties. May contain duplicate configurations.
        constitutive_properties: Column names defining the configuration space.
            Uniqueness is enforced over the Cartesian product of these properties.
        n: Number of samples to generate. Adjusted if larger than available
            unique configurations.
        strategy: Sampling strategy - "random", "clhs", or "sobol".

    Returns:
        DataFrame with n sampled rows, preserving the original column schema.
        Index is positional (0..n-1).

    Raises:
        ValueError: If n <= 0 after adjustment or no samples are available.
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
            f"Requested {n} samples, but DataFrame has only {len(df_unique)} rows. Adjusting n to {len(df_unique)}."
        )
        n = len(df_unique)

    if n <= 0:
        logging.error(
            f"No samples available to select. DataFrame has {len(df_unique)} rows and {n} samples were requested."
        )
        # Return empty DataFrame with same columns as input
        return pd.DataFrame(columns=df_unique.columns)

    # Build dictionaries
    def _get_sorted_uniques(prop: str) -> list:
        """Helper to safely sort unique values for a property."""
        vals = df_unique[prop].unique()
        try:
            return sorted(vals)
        except TypeError:
            logging.warning(
                f"Cannot sort mixed types for property '{prop}'. "
                "Keeping original order."
            )
            return list(vals)

    value_dict = {prop: _get_sorted_uniques(prop) for prop in constitutive_properties}

    space_dict = {prop: len(vals) for prop, vals in value_dict.items()}

    dimensions = list(space_dict.values())

    # Order DataFrame for index mapping
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
    Ensure DataFrame is ordered and complete for high-dimensional index generation.

    Prepares the DataFrame so rows align with the Cartesian product implied by
    constitutive_properties and dimensions. Sorts rows, validates completeness,
    and injects missing combinations if needed.

    Args:
        df: Input DataFrame containing at least the columns in constitutive_properties.
        constitutive_properties: Column names defining the high-dimensional space.
            Order determines sort priority.
        dimensions: Expected cardinality for each constitutive property.
            Used to compute expected_len = product(dimensions).

    Returns:
        DataFrame sorted by constitutive_properties and augmented with any missing
        combinations. Injected rows have NaN for non-constitutive columns.

    Notes:
        If dimensions and actual unique values disagree, uses observed unique
        values to generate combinations.
    """
    # Sort by constitutive properties
    df = df.sort_values(by=constitutive_properties).reset_index(drop=True)

    expected_len = math.prod(dimensions)

    # Return early if already complete
    if len(df) == expected_len:
        return df

    # Generate all possible combinations based on actual unique values
    unique_values = [
        sorted(df[prop].dropna().unique()) for prop in constitutive_properties
    ]
    all_combinations = list(itertools.product(*unique_values))
    actual_expected_len = len(all_combinations)

    logger.warning(
        f"DataFrame length mismatch: expected {expected_len} (product of {dimensions}), "
        f"but got {len(df)}. Actual unique combinations: {actual_expected_len}."
    )

    # Identify existing combinations
    existing_combinations = {
        tuple(row[prop] for prop in constitutive_properties) for _, row in df.iterrows()
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
