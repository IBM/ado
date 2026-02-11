# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import logging
import os

import pandas as pd

from trim.trim_pydantic import TrimParameters
from trim.utils.rowsring import RowsRing

logger = logging.getLogger(__name__)


def log_after_split_common_and_diff(
    iter_index: int,
    previous_source_from_split_df: pd.DataFrame,
    previous_source_df: pd.DataFrame,
    one_additional_row: pd.DataFrame,
    directory: str,
) -> None:
    if not previous_source_from_split_df.reset_index(drop=True).equals(
        previous_source_df.reset_index(drop=True)
    ):
        logger.warning(
            f"Length of the source dataframe obtained from comparing the entities retrieved before and after making a measurement= {len(previous_source_from_split_df)},"
            f"Length of the source dataframe at the previous iteration = {len(previous_source_df)}"
        )
        logger.error(
            f"Unexpected behaviour of dataframes, saving data in the directory: {directory}"
        )
        previous_source_from_split_df.to_csv(
            os.path.join(directory, f"Mismatch_iter{iter_index}_{iter_index}.csv")
        )
        previous_source_df.to_csv(
            os.path.join(directory, f"Mismatch_iter{iter_index}_{iter_index-1}.csv")
        )
    else:
        logger.debug(
            "Equality of these two dataframes after resetting the index has been checked."
            "These datasets are:"
            "\t - The source dataframe obtained from comparing the entities retrieved before and after making a measurement"
            "\t - The source dataframe at the previous iteration."
        )

    if len(one_additional_row) != 1:
        logger.error(
            f"{len(one_additional_row)} point(s) sampled (expected 1), saving data in {directory}"
        )
        one_additional_row.to_csv(f"one_additional_row_{iter_index}.csv")
    else:
        logger.debug(
            "The number of rows that we are adding to the previous source space is 1, as expected"
        )


def log_after_first_holdout_creation(
    current_holdout_df: pd.DataFrame,
    yielded_rows: RowsRing,
    iter_index: int,
    params: TrimParameters,
) -> None:
    logger.debug(
        f"First holdout set created, it contains the following {len(current_holdout_df)} rows:"
    )
    logger.debug(current_holdout_df)
    if current_holdout_df.empty:
        logger.error("Empty Holdout Dataset!")
    if len(current_holdout_df) != params.holdoutSize:
        logger.error(
            f"The holdout df contains {len(current_holdout_df)} rows (expected { params.holdoutSize})"
        )
    same = yielded_rows.df.columns.equals(
        current_holdout_df.columns
    ) and yielded_rows.df.value_counts(dropna=False).equals(
        current_holdout_df.value_counts(dropna=False)
    )  # True if they contain exactly the same rows (multiset equality), regardless of order
    if not same:
        logger.error(
            f"Unexpected behaviour of holdout dfs, saving data in {params.debugDirectory}"
        )
        yielded_rows.df.to_csv(f"Mismatch_yielded_rows_{iter_index}.csv")
        current_holdout_df.to_csv(f"Mismatch_current_holdout_df_{iter_index}.csv")
    else:
        logger.debug(
            "Check passed! Every row of yielded_rows is also in current_holdout_df"
        )


def log_and_save_characterization(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
) -> None:
    """
    Logs characterization details and saves source/target DataFrames.
    Assumes caller already checked logger level.
    """

    # Log basic stats
    logger.debug(
        f"[Characterization] source_df rows: {len(source_df)}, target_df rows: {len(target_df)}"
    )

    # Log unique identifier counts if present
    for df, name in [(source_df, "source_df"), (target_df, "target_df")]:
        if "identifier" in df.columns:
            logger.debug(
                f"[Characterization] {name} 'identifier' unique count: {df['identifier'].nunique(dropna=True)}"
            )
        else:
            logger.debug(f"[Characterization] {name} has no 'identifier' column.")


def log_before_first_holdout_update(
    one_additional_row: pd.DataFrame,
    current_source_df: pd.DataFrame,
    previous_source_df: pd.DataFrame,
    iter_index: int,
    debugDirectory: str,
    batchsize: int = 1,
) -> None:
    if len(one_additional_row) != 1:
        logger.error(
            f"{len(one_additional_row)} point(s) sampled (expected 1), saving data in {debugDirectory}"
        )
        one_additional_row.to_csv(os.path.join(f"one_additional_row_{iter_index}.csv"))
    else:
        logger.info(
            f"Check on the length of the additional row to be added to holdout passed at iter {iter_index}"
        )
    if len(current_source_df) != len(previous_source_df) + batchsize:
        logger.warning(
            f"Length of source df at iter {iter_index}: {len(current_source_df)}"
            f"It is NOT 1 unit greater than length of source df for {iter_index} - {batchsize}: {len(previous_source_df)}"
        )


def training_guardrail(train_df: pd.DataFrame, targetOutput: str) -> pd.DataFrame:
    if not train_df.equals(train_df.dropna(subset=[str(targetOutput)])):
        logger.warning(
            "There are rows in train dataframe where the target is NaN! Dropping them now.\n\n"
        )
        train_df = train_df.dropna(subset=[targetOutput])

    if train_df.empty:
        logger.warning(
            "Empty training dataframe, this means either that "
            "the operation configuration or the inference problem is ill posed"
        )
        raise ValueError("Empty dataframe!")
    return train_df


def save_source_train_holdout_dfs(
    current_source_df: pd.DataFrame,
    train_df: pd.DataFrame,
    current_holdout_df: pd.DataFrame,
    iter: int,
    directory: str,
) -> None:
    source_path = os.path.join(directory, f"source_at_iter_{iter}.csv")
    train_path = os.path.join(directory, f"train_at_iter_{iter}.csv")
    holdout_path = os.path.join(directory, f"holdout_at_iter_{iter}.csv")

    current_source_df.to_csv(source_path, index=False)
    train_df.to_csv(train_path, index=False)
    current_holdout_df.to_csv(holdout_path, index=False)
