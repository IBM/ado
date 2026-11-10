# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Validation of finite-pool training data and raw feature schemas."""

from __future__ import annotations

import typing

import numpy as np
import pandas as pd

if typing.TYPE_CHECKING:
    from collections.abc import Sequence


def _validate_training_data(
    feature_frame: pd.DataFrame,
    labels: pd.Series,
    target_output: str,
    *,
    minimum_labels: int = 1,
    minimum_labels_message: str = "at least one training label is required",
) -> tuple[np.ndarray, np.ndarray]:
    """Validate a raw feature pool and return labelled positions and targets.

    Args:
        feature_frame: Nonempty features with unique string names and row IDs.
        labels: Finite real scalar targets indexed by a subset of feature row IDs.
        target_output: Nonempty target name assigned to predictions.
        minimum_labels: Minimum number of labelled rows required for training.
        minimum_labels_message: Error for an invalid or undersized label series.

    Returns:
        Feature row positions and numeric targets in the supplied label order.

    Raises:
        ValueError: If features, row identities, labels, or the target name are invalid.
    """

    if not isinstance(feature_frame, pd.DataFrame) or feature_frame.empty:
        raise ValueError("feature_frame must contain at least one row and feature")
    if not feature_frame.columns.is_unique or not feature_frame.index.is_unique:
        raise ValueError("feature names and feature row indices must be unique")
    if not all(isinstance(column, str) for column in feature_frame.columns):
        raise ValueError("feature names must be strings")
    if not isinstance(target_output, str) or not target_output.strip():
        raise ValueError("target_output must be a nonempty string")
    if not isinstance(labels, pd.Series) or len(labels) < minimum_labels:
        raise ValueError(minimum_labels_message)
    if not labels.index.is_unique:
        raise ValueError("training label indices must be unique")
    positions = feature_frame.index.get_indexer(labels.index)
    if np.any(positions < 0):
        raise ValueError("each label index must identify a feature row")
    raw_labels = labels.to_numpy()
    if np.iscomplexobj(raw_labels):
        raise ValueError("training labels must be finite real scalar values")
    try:
        targets = np.asarray(raw_labels, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("training labels must be finite real scalar values") from error
    if targets.ndim != 1 or not np.all(np.isfinite(targets)):
        raise ValueError("training labels must be finite real scalar values")
    return positions, targets


def _resolve_categorical_columns(
    frame: pd.DataFrame,
    categorical_columns: Sequence[str] | None = None,
    *,
    missing_message: str = "categorical columns are missing from features",
) -> tuple[str, ...]:
    """Return explicit and inferred nonnumeric categories in stable column order.

    Args:
        frame: Raw feature columns whose numeric types determine inferred categories.
        categorical_columns: Additional columns treated as categories.
        missing_message: Error prefix for categorical columns absent from the frame.

    Returns:
        Unique categorical names in explicit order followed by inferred frame order.

    Raises:
        ValueError: If an explicit categorical column is absent from the frame.
    """

    explicit = list(categorical_columns or [])
    missing = sorted(set(explicit) - set(frame.columns))
    if missing:
        raise ValueError(f"{missing_message}: {missing}")
    inferred = [
        column
        for column in frame.columns
        if not pd.api.types.is_numeric_dtype(frame[column].dtype)
    ]
    return tuple(dict.fromkeys([*explicit, *inferred]))


def _prepare_feature_frame(
    frame: pd.DataFrame,
    feature_names: np.ndarray,
    category_dtypes: dict[str, pd.CategoricalDtype],
) -> pd.DataFrame:
    """Restore a raw feature schema and validate category levels and numeric values.

    Args:
        frame: Raw features with all required columns and any extra columns.
        feature_names: Required columns in their fitted order.
        category_dtypes: Fitted category levels for categorical features.

    Returns:
        Required columns with fitted categories and finite floating-point numbers.

    Raises:
        ValueError: If a required column is absent or ambiguous, a category is
            unknown, or a numeric value is invalid or non-finite.
    """

    missing = [name for name in feature_names if name not in frame.columns]
    if missing:
        raise ValueError(f"missing required feature columns: {missing}")
    selected = frame.loc[:, feature_names].copy()
    if not selected.columns.is_unique:
        raise ValueError("feature names must be unique")
    for column in feature_names:
        values = selected[column]
        if column in category_dtypes:
            dtype = category_dtypes[column]
            unknown = values.notna() & ~values.isin(dtype.categories)
            if unknown.any():
                raise ValueError(f"unknown category in feature {column!r}")
            selected[column] = pd.Categorical(values, dtype=dtype)
        else:
            if np.iscomplexobj(values.to_numpy()):
                raise ValueError(f"feature {column!r} must contain real numeric values")
            try:
                numeric = values.to_numpy(dtype=float)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"feature {column!r} must contain numeric values"
                ) from error
            if not np.all(np.isfinite(numeric)):
                raise ValueError(f"feature {column!r} must contain finite values")
            selected[column] = numeric
    return selected
