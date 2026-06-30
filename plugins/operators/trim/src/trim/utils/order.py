# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import logging

import pandas as pd
from autogluon.tabular import TabularPredictor

from trim.trim_pydantic import AutoGluonArgs
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
