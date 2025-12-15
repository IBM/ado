# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def describe_source_spaces(
    this_iteration_source_df,
    previous_iteration_source_df,
    filter_cols=None,
):
    """
    Logs information about the current and previous source DataFrames:
      - lengths (rows, columns)
      - warns if column sets differ
      - counts common rows on shared columns
      - counts common rows restricted to filter_cols (if provided)
    """

    # lengths and shapes
    logger.info(
        f"this_iteration_source_df shape={this_iteration_source_df.shape}; "
        f"previous_iteration_source_df shape={previous_iteration_source_df.shape}"
    )
    logger.info(
        f"len(this_iteration_source_df)={len(this_iteration_source_df)}; "
        f"len(previous_iteration_source_df)={len(previous_iteration_source_df)}"
    )

    # column set comparison
    set_this, set_prev = set(this_iteration_source_df.columns), set(
        previous_iteration_source_df.columns
    )
    if set_this != set_prev:
        logger.warning(
            "describe_source_spaces: Column sets differ.\n"
            f" - Only in this_iteration_source_df: {sorted(set_this - set_prev)}\n"
            f" - Only in previous_iteration_source_df: {sorted(set_prev - set_this)}"
        )

    # common rows on shared columns
    common_cols = sorted(set_this & set_prev)
    if not common_cols:
        logger.warning(
            "describe_source_spaces: No common columns available to compare rows."
        )
    else:
        this_tuples = set(
            map(
                tuple,
                this_iteration_source_df[common_cols].itertuples(
                    index=False, name=None
                ),
            )
        )
        prev_tuples = set(
            map(
                tuple,
                previous_iteration_source_df[common_cols].itertuples(
                    index=False, name=None
                ),
            )
        )
        logger.info(
            f"describe_source_spaces: Common rows (on shared columns, n_cols={len(common_cols)}): "
            f"{len(this_tuples & prev_tuples)}"
        )

    # common rows restricted to filter_cols
    if filter_cols is not None:
        try:
            filter_cols = list(filter_cols)
        except Exception:
            logger.warning(
                "describe_source_spaces: filter_cols is not list-like; skipping filtered comparison."
            )
            filter_cols = None

        if filter_cols:
            present_in_both = [
                c for c in filter_cols if c in set_this and c in set_prev
            ]
            ignored = sorted(set(filter_cols) - set(present_in_both))
            if ignored:
                logger.warning(
                    f"describe_source_spaces: Some filter_cols not present in both DataFrames: {ignored}"
                )
            if not present_in_both:
                logger.warning(
                    "describe_source_spaces: No valid filter_cols present in both DataFrames."
                )
            else:
                this_tuples_f = set(
                    map(
                        tuple,
                        this_iteration_source_df[present_in_both].itertuples(
                            index=False, name=None
                        ),
                    )
                )
                prev_tuples_f = set(
                    map(
                        tuple,
                        previous_iteration_source_df[present_in_both].itertuples(
                            index=False, name=None
                        ),
                    )
                )
                logger.info(
                    f"describe_source_spaces: Common rows (restricted to filter_cols): "
                    f"{len(this_tuples_f & prev_tuples_f)}  |  cols={present_in_both}"
                )

    # final note if previous df is missing/empty
    if previous_iteration_source_df is None or len(previous_iteration_source_df) == 0:
        logger.warning(
            "describe_source_spaces: previous_iteration_source_df is None or empty; "
            "downstream code may raise errors."
        )


def log_and_save_characterization(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    logging: logging.Logger,
    base_dir: str = "initial_dataframes",
):
    """
    Logs characterization details and saves source/target DataFrames.
    Assumes caller already checked logger level.
    """

    # Log basic stats
    logging.debug(
        f"[Characterization] source_df rows: {len(source_df)}, target_df rows: {len(target_df)}"
    )

    # Log unique identifier counts if present
    if "identifier" in source_df.columns:
        logging.debug(
            f"[Characterization] source_df 'identifier' unique count: {source_df['identifier'].nunique(dropna=True)}"
        )
    else:
        logging.debug("[Characterization] source_df has no 'identifier' column.")

    if "identifier" in target_df.columns:
        logging.debug(
            f"[Characterization] target_df 'identifier' unique count: {target_df['identifier'].nunique(dropna=True)}"
        )
    else:
        logging.debug("[Characterization] target_df has no 'identifier' column.")

    # Prepare output directory
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(base_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save DataFrames
    src_path = out_dir / f"source_df_{ts}.csv"
    tgt_path = out_dir / f"target_df_{ts}.csv"

    try:
        source_df.to_csv(src_path, index=False)
        target_df.to_csv(tgt_path, index=False)
        logging.debug(f"[Characterization] Saved source_df -> {src_path}")
        logging.debug(f"[Characterization] Saved target_df -> {tgt_path}")
    except Exception:
        logging.exception


def split_common_and_diff(
    shorter_df_that_you_subtract: pd.DataFrame,
    longer_df_from_which_you_subtract: pd.DataFrame,
    on=None,
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


# TODO: monitor
def get_train_and_holdout_df_from_source_dfs_of_last_iters(
    this_iteration_source_df,
    previous_iteration_source_df,
    train_cols,
    train_target_cols,
    holdout_size,
):
    # TODO: chose the best approach here
    # previous_iteration_source_df and this_iteration_source_df have the same columns.
    # how can I obtain a dataset that has
    # the rows of source that are NOT present in previous_iteration_source_df (NOTE: should I look only at features here)?
    prev_key_tuples = set(
        map(
            tuple,
            previous_iteration_source_df[train_cols].itertuples(index=False, name=None),
        )
    )
    mask = (
        ~this_iteration_source_df[train_cols].apply(tuple, axis=1).isin(prev_key_tuples)
    )

    # The last batch goes into holdout
    holdout_df_tuple = this_iteration_source_df.loc[mask].copy()
    train_df_tuple = this_iteration_source_df.loc[~mask, train_target_cols].copy()
    logging.info(
        f"[Tuple Approach] Holdout rows: {len(holdout_df_tuple)}, Train rows: {len(train_df_tuple)}"
    )

    if len(holdout_df_tuple) == 0:
        logging.warning("No holdout dataframe selected by tuple-based approach")

    # --- Approach 2: Merge-based (recommended) ---
    merged = this_iteration_source_df.merge(
        previous_iteration_source_df[train_cols],
        on=train_cols,
        how="left",
        indicator=True,
    )

    # Rows only in this_iteration_source_df
    holdout_df = merged.query('_merge == "left_only"').drop(columns=["_merge"]).copy()

    # Rows present in both → train set (restricted to train_target_cols)
    train_df = merged.query('_merge == "both"')[train_target_cols].copy()

    logging.info(
        f"[Merge Approach] Holdout rows: {len(holdout_df)}, Train rows: {len(train_df)}"
    )

    # --- Ensure minimum holdout size ---
    if len(holdout_df) < holdout_size:
        deficit = holdout_size - len(holdout_df)
        logging.warning(
            f"Holdout too small ({len(holdout_df)}). Adding {deficit} extra rows from source."
        )

        # Sample additional rows from this_iteration_source_df that are NOT already in holdout_df
        remaining_candidates = this_iteration_source_df.loc[
            ~this_iteration_source_df.index.isin(holdout_df.index)
        ]

        # If deficit > remaining candidates, take all
        extra_rows = remaining_candidates.sample(
            n=min(deficit, len(remaining_candidates)), random_state=42
        )

        holdout_df = pd.concat([holdout_df, extra_rows], ignore_index=True)

    # OUTPUT THE NEW
    return train_df, holdout_df
