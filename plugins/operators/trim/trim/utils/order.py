# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import itertools
import logging
import math

import pandas as pd
from autogluon.tabular import TabularPredictor

from trim.utils.high_dimensional_sampling import get_order_list_nn_high_dimensional
from trim.utils.miscellaneous import delete_dir

logger = logging.getLogger("trim")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    logger.addHandler(_h)


def get_feature_importance_order(
    source_df: pd.DataFrame,
    target_output: str,
    min_measured_entities: int,
    autoGluonArgs,
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
        label=target_output,
        **(getattr(autoGluonArgs, "tabularPredictorArgs", None) or {}),
    )

    fit_kwargs = getattr(autoGluonArgs, "fitArgs", None) or {}
    logger.info(
        f"Fitting AutoGluon TabularPredictor; train cols: {list(train_df.columns)}"
    )
    predictor.fit(train_data=train_df, **fit_kwargs)

    # 3) Feature importances
    fi_df = predictor.feature_importance(train_df).sort_values(
        "importance", ascending=False
    )
    importance_dict = fi_df["importance"].to_dict()
    ordered_tuple_most_important_first = tuple(fi_df.index)

    logger.info(f"Top features: {list(ordered_tuple_most_important_first[:5])}")

    # 4) Cleanup model directory
    model_dir = getattr(predictor, "path", None)
    logger.info(f"AutoGluon model directory: {model_dir}")
    del predictor
    delete_dir(model_dir)

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
    df: pd.DataFrame, constitutive_properties: list[str], n: int, refined: bool = True
) -> pd.DataFrame:
    """
    Orders a DataFrame for high-dimensional sampling without prior knowledge.

    Steps:

    0. Filter dataset so that for each combination of constitutive properties you only have one row
    1. Extract unique values for each constitutive property.
    2. Build:
        - value_dict_unordered: keys = properties, values = unique unordered lists.
        - value_dict: same as above but ordered ascending.
        - space_dict: keys = properties, values = length of each list.
        - dims: list of lengths (dimensionality).
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
            f"""Removing {delta_len} duplicate configurations.
                        They are characterized by the same combination of constitutive properties = {constitutive_properties}"""
        )

    if n > len(df_unique):
        logging.warning(
            f"Requested {n} samples, but DataFrame has only {len(df_unique)} rows. Adjusting n."
        )
        n = min(n, len(df_unique))

    # Build dictionaries
    value_dict_unordered = {
        prop: list(set(df_unique[prop].values)) for prop in constitutive_properties
    }

    value_dict = {}
    for prop, vals in value_dict_unordered.items():
        try:
            value_dict[prop] = sorted(vals)
        except TypeError:
            logging.warning(
                f"Cannot sort mixed types for property '{prop}'. Keeping original order (it may be inconsistent due to the use of sets)."
            )
            value_dict[prop] = list(vals)

    space_dict = {prop: len(vals) for prop, vals in value_dict.items()}
    dims = list(space_dict.values())

    # Order DataFrame for index mapping
    # NOTE: just added .reset_index(drop=True)
    df_unique = order_df_for_get_index_list_nn_high_dimensional(
        df_unique, constitutive_properties, dims=dims
    ).reset_index(drop=True)

    # Generate sampling orders
    orders_to_sample = get_order_list_nn_high_dimensional(
        dims=dims,
        space=space_dict,
        n=n,
    )

    # Map orders to DataFrame indices
    indices_to_sample = get_index_list_nn_high_dimensional(orders_to_sample, dims)

    # Select rows # IndexError("positional indexers are out-of-bounds")
    logger.info(f"Indexes are:\n {indices_to_sample}")
    print(df_unique)
    df_unique.to_csv("df_unique.csv")

    return df_unique.iloc[indices_to_sample]


def order_df_for_get_index_list_nn_high_dimensional(
    df: pd.DataFrame, constitutive_properties: list[str], dims: list[int]
) -> pd.DataFrame:
    """
    Sorts by constitutive_properties in the order provided.
    If the length of df does not match the product of dims, log a warning and inject missing rows.
    Injected rows will have NaN for non-constitutive columns.
    """

    # Sort by constitutive properties
    df = df.sort_values(by=constitutive_properties).reset_index(drop=True)

    expected_len = math.prod(dims)
    if len(df) != expected_len:
        logger.warning(
            f"DataFrame length mismatch: expected {expected_len} (product of {dims}), "
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
                row_data = dict(zip(constitutive_properties, comb))
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
    orders_to_sample: list[list[int]], dims: list[int]
) -> list[int]:
    """
    Maps high-dimensional sampling orders to linear indices.
    Each order is a list of positions [i0, i1, ..., ik].
    dims is a list of sizes for each dimension [d0, d1, ..., dk].
    """
    indices = []
    for order in orders_to_sample:
        index = 0
        multiplier = 1
        # Iterate reversed so last dimension varies fastest
        for i in reversed(range(len(dims))):
            index += order[i] * multiplier
            multiplier *= dims[i]
        indices.append(index)
    return indices
