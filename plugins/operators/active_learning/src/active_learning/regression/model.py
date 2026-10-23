# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Reusable random-forest predictions with a persisted raw-feature schema."""

from __future__ import annotations

import json
import pickle
import platform
import shutil
import tempfile
import typing
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor

if typing.TYPE_CHECKING:
    import os
    from collections.abc import Sequence


def _atomic_write(path: Path, content: bytes) -> None:
    """Replace a file atomically with bytes written in its parent directory."""

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary_path = Path(stream.name)
            stream.write(content)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


@dataclass
class RandomForestPredictor:
    """A fitted regression forest and the raw-feature schema used for prediction.

    Attributes:
        forest_: Fitted scikit-learn random-forest regressor.
        feature_names_in_: Original raw-feature names in their fitted order.
        n_training_samples_: Number of labelled finite-pool rows used to fit.
        target_output: Name assigned to the prediction series.
    """

    forest_: RandomForestRegressor
    feature_names_in_: np.ndarray
    n_training_samples_: int
    target_output: str
    _categorical_columns: tuple[str, ...]
    _category_dtypes: dict[str, pd.CategoricalDtype]

    @classmethod
    def fit(
        cls,
        feature_frame: pd.DataFrame,
        labels: pd.Series,
        *,
        target_output: str,
        categorical_columns: Sequence[str] | None = None,
        forest_parameters: dict | None = None,
    ) -> RandomForestPredictor:
        """Fit on labelled rows with a categorical schema from the complete pool.

        Args:
            feature_frame: Complete finite pool of raw features with unique row IDs.
            labels: Finite target values indexed by a subset of pool row IDs.
            target_output: Name of the target represented by predictions.
            categorical_columns: Features encoded as categories, including numeric
                categories. Nonnumeric columns are inferred automatically.
            forest_parameters: Keyword parameters for ``RandomForestRegressor``.

        Returns:
            A fitted predictor that accepts raw feature dataframes.

        Raises:
            ValueError: If features or training labels are empty or ambiguous,
                values are invalid, or labelled indices are outside the pool.
        """

        if not isinstance(feature_frame, pd.DataFrame) or feature_frame.empty:
            raise ValueError("feature_frame must contain at least one row and feature")
        if not feature_frame.columns.is_unique or not feature_frame.index.is_unique:
            raise ValueError("feature names and feature row indices must be unique")
        if not all(isinstance(column, str) for column in feature_frame.columns):
            raise ValueError("feature names must be strings")
        if not isinstance(target_output, str) or not target_output.strip():
            raise ValueError("target_output must be a nonempty string")
        if not isinstance(labels, pd.Series) or labels.empty:
            raise ValueError("at least one training label is required")
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
            raise ValueError(
                "training labels must be finite real scalar values"
            ) from error
        if targets.ndim != 1 or not np.all(np.isfinite(targets)):
            raise ValueError("training labels must be finite real scalar values")

        explicit_categorical = list(categorical_columns or [])
        missing = sorted(set(explicit_categorical) - set(feature_frame.columns))
        if missing:
            raise ValueError(
                f"categorical columns are missing from features: {missing}"
            )
        inferred_categorical = [
            column
            for column in feature_frame.columns
            if not pd.api.types.is_numeric_dtype(feature_frame[column].dtype)
        ]
        categorical = tuple(
            dict.fromkeys([*explicit_categorical, *inferred_categorical])
        )
        predictor = cls(
            forest_=RandomForestRegressor(**(forest_parameters or {})),
            feature_names_in_=feature_frame.columns.to_numpy(copy=True),
            n_training_samples_=len(labels),
            target_output=target_output,
            _categorical_columns=categorical,
            _category_dtypes={
                column: pd.Categorical(feature_frame[column]).dtype
                for column in categorical
            },
        )
        encoded = predictor._encode(feature_frame)
        predictor.forest_.fit(encoded[positions], targets)
        return predictor

    def _encode(self, frame: pd.DataFrame) -> np.ndarray:
        """Apply the fitted raw-feature schema and validate finite numeric values."""

        missing = [name for name in self.feature_names_in_ if name not in frame.columns]
        if missing:
            raise ValueError(f"missing required feature columns: {missing}")
        selected = frame.loc[:, self.feature_names_in_].copy()
        if not selected.columns.is_unique:
            raise ValueError("feature names must be unique")
        for column in self.feature_names_in_:
            values = selected[column]
            if column in self._category_dtypes:
                dtype = self._category_dtypes[column]
                unknown = values.notna() & ~values.isin(dtype.categories)
                if unknown.any():
                    raise ValueError(f"unknown category in feature {column!r}")
                selected[column] = pd.Categorical(values, dtype=dtype)
            else:
                if np.iscomplexobj(values.to_numpy()):
                    raise ValueError(
                        f"feature {column!r} must contain real numeric values"
                    )
                try:
                    numeric = values.to_numpy(dtype=float)
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"feature {column!r} must contain numeric values"
                    ) from error
                if not np.all(np.isfinite(numeric)):
                    raise ValueError(f"feature {column!r} must contain finite values")
                selected[column] = numeric
        encoded = pd.get_dummies(
            selected,
            columns=list(self._categorical_columns),
            dummy_na=True,
            dtype=float,
        )
        return np.asarray(encoded.to_numpy(dtype=float), order="C")

    def predict(self, feature_frame: pd.DataFrame) -> pd.Series:
        """Predict raw rows, preserving their index and ignoring extra columns.

        Args:
            feature_frame: Raw features containing all fitted feature columns.

        Returns:
            Predicted values indexed like the input and named ``target_output``.

        Raises:
            ValueError: If required columns are missing, a category is unknown,
                or a numeric value is non-finite or invalid.
        """

        if not isinstance(feature_frame, pd.DataFrame):
            raise ValueError("prediction features must be a dataframe")
        if len(feature_frame) == 0:
            return pd.Series(
                index=feature_frame.index, dtype=float, name=self.target_output
            )
        return pd.Series(
            self.forest_.predict(self._encode(feature_frame)),
            index=feature_frame.index,
            name=self.target_output,
        )

    def save(self, directory: str | os.PathLike[str]) -> Path:
        """Save the fitted predictor and metadata into a new directory.

        Args:
            directory: Artifact directory to create, including missing parents.

        Returns:
            The created directory containing ``model.pkl`` and ``metadata.json``.

        Raises:
            FileExistsError: If the requested directory already exists.
        """

        directory = Path(directory)
        directory.mkdir(parents=True)
        try:
            metadata = {
                "backend": "random_forest",
                "target_output": self.target_output,
                "feature_names": self.feature_names_in_.tolist(),
                "n_training_samples": self.n_training_samples_,
                "python_version": platform.python_version(),
                "sklearn_version": sklearn.__version__,
            }
            _atomic_write(
                directory / "metadata.json",
                (json.dumps(metadata, indent=2) + "\n").encode("utf-8"),
            )
            _atomic_write(directory / "model.pkl", pickle.dumps(self, protocol=5))
        except BaseException:
            shutil.rmtree(directory)
            raise
        return directory

    @classmethod
    def load(cls, directory: str | os.PathLike[str]) -> RandomForestPredictor:
        """Load a predictor from a trusted artifact directory.

        Pickle can execute code while loading. Only load files from trusted
        sources, using compatible Python and scikit-learn versions.

        Args:
            directory: Directory containing a saved ``model.pkl`` artifact.

        Returns:
            The fitted predictor and its frozen feature schema.

        Raises:
            ValueError: If the artifact contains an unexpected object type.
        """

        with (Path(directory) / "model.pkl").open("rb") as stream:
            predictor = pickle.load(stream)  # noqa: S301
        if not isinstance(predictor, cls):
            raise ValueError("model artifact does not contain a RandomForestPredictor")
        return predictor
