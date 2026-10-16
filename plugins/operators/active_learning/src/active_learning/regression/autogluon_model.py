# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Optional AutoGluon predictions with a persisted raw-feature schema."""

from __future__ import annotations

import copy
import importlib.metadata
import json
import pickle
import platform
import shutil
import sys
import tempfile
import typing
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from active_learning.regression.model import _atomic_write

if typing.TYPE_CHECKING:
    import os
    from collections.abc import Sequence

    from autogluon.tabular import TabularPredictor


def _require_autogluon() -> type[TabularPredictor]:
    """Load the optional backend or explain its installation requirements."""

    message = (
        "The AutoGluon final predictor requires Python 3.11-3.13 and the optional "
        "AutoGluon dependency. Install it with: python -m pip install -e "
        "'plugins/operators/active_learning[autogluon]'"
    )
    if not (3, 11) <= sys.version_info[:2] < (3, 14):
        raise RuntimeError(message)
    try:
        from autogluon.tabular import TabularPredictor
    except (ImportError, OSError) as error:
        raise RuntimeError(message) from error
    return TabularPredictor


@dataclass
class AutoGluonPredictor:
    """A fitted AutoGluon regressor and its complete finite-pool feature schema.

    Attributes:
        predictor_: Native AutoGluon predictor with its model artifact directory.
        feature_names_in_: Raw feature names in their fitted order.
        n_training_samples_: Number of measured labels supplied to AutoGluon.
        n_fit_samples_: Number of rows in AutoGluon's internal training dataset.
        n_validation_samples_: Number of rows in its separate validation dataset.
        target_output: Name assigned to returned prediction series.
    """

    predictor_: typing.Any
    feature_names_in_: np.ndarray
    n_training_samples_: int
    n_fit_samples_: int
    n_validation_samples_: int
    target_output: str
    _categorical_columns: tuple[str, ...]
    _category_dtypes: dict[str, pd.CategoricalDtype]
    _random_state: int
    _fit_parameters: dict
    _temporary_directory: tempfile.TemporaryDirectory | None = None

    @classmethod
    def fit(
        cls,
        feature_frame: pd.DataFrame,
        labels: pd.Series,
        *,
        target_output: str,
        categorical_columns: Sequence[str] | None = None,
        fit_parameters: dict | None = None,
        random_state: int = 42,
    ) -> AutoGluonPredictor:
        """Fit measured rows while preserving the complete raw-feature schema.

        Args:
            feature_frame: Complete finite pool with unique feature names and row IDs.
            labels: Finite measured targets indexed by a subset of pool row IDs.
            target_output: Name of the target represented by predictions.
            categorical_columns: Features treated as categories, including numeric
                categories. Nonnumeric features are inferred automatically.
            fit_parameters: AutoGluon fit settings excluding external input datasets.
            random_state: Seed for AutoGluon's learner and internal data split.

        Returns:
            A fitted predictor that accepts the original raw feature columns.

        Raises:
            ValueError: If the schema or labels are invalid or fewer than four
                measured labels are supplied.
            RuntimeError: If AutoGluon is unavailable in the current environment.
        """

        if not isinstance(feature_frame, pd.DataFrame) or feature_frame.empty:
            raise ValueError("feature_frame must contain at least one row and feature")
        if not feature_frame.columns.is_unique or not feature_frame.index.is_unique:
            raise ValueError("feature names and feature row indices must be unique")
        if not all(isinstance(column, str) for column in feature_frame.columns):
            raise ValueError("feature names must be strings")
        if not isinstance(target_output, str) or not target_output.strip():
            raise ValueError("target_output must be a nonempty string")
        if not isinstance(labels, pd.Series) or len(labels) < 4:
            raise ValueError(
                "AutoGluon requires at least four measured training labels"
            )
        if not labels.index.is_unique:
            raise ValueError("training label indices must be unique")
        if np.any(feature_frame.index.get_indexer(labels.index) < 0):
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
            predictor_=None,
            feature_names_in_=feature_frame.columns.to_numpy(copy=True),
            n_training_samples_=len(labels),
            n_fit_samples_=0,
            n_validation_samples_=0,
            target_output=target_output,
            _categorical_columns=categorical,
            _category_dtypes={
                column: pd.Categorical(feature_frame[column]).dtype
                for column in categorical
            },
            _random_state=random_state,
            _fit_parameters={},
        )
        prepared = predictor._prepare_features(feature_frame)
        training = prepared.loc[labels.index].copy()
        label_name = "__ado_target__"
        while label_name in training.columns:
            label_name += "_"
        training[label_name] = targets
        fit_parameters = copy.deepcopy(fit_parameters or {})
        external_data = {
            "train_data",
            "tuning_data",
            "unlabeled_data",
        } & fit_parameters.keys()
        if external_data:
            raise ValueError(
                "fit_parameters cannot supply external datasets: "
                f"{sorted(external_data)}"
            )
        try:
            json.dumps(fit_parameters)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "fit_parameters must contain JSON-compatible settings"
            ) from error
        predictor._fit_parameters = copy.deepcopy(fit_parameters)
        predictor_class = _require_autogluon()
        workspace = tempfile.TemporaryDirectory(prefix="ado-autogluon-")
        try:
            predictor.predictor_ = predictor_class(
                label=label_name,
                problem_type="regression",
                eval_metric="root_mean_squared_error",
                path=str(Path(workspace.name) / "predictor"),
                verbosity=0,
                learner_kwargs={"random_state": random_state},
            )
            predictor.predictor_.fit(train_data=training, **fit_parameters)
            information = predictor.predictor_.info()
            predictor.n_fit_samples_ = int(information["num_rows_train"])
            predictor.n_validation_samples_ = int(information.get("num_rows_val") or 0)
        except BaseException:
            workspace.cleanup()
            raise
        predictor._temporary_directory = workspace
        return predictor

    def _prepare_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Validate raw values and restore the fitted numeric/category schema."""

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
        return selected

    def predict(self, feature_frame: pd.DataFrame) -> pd.Series:
        """Predict raw rows, preserving their index and ignoring extra columns."""

        if not isinstance(feature_frame, pd.DataFrame):
            raise ValueError("prediction features must be a dataframe")
        if len(feature_frame) == 0:
            return pd.Series(
                index=feature_frame.index, dtype=float, name=self.target_output
            )
        predictions = self.predictor_.predict(self._prepare_features(feature_frame))
        values = np.asarray(predictions, dtype=float)
        if values.shape != (len(feature_frame),) or not np.all(np.isfinite(values)):
            raise ValueError("AutoGluon produced invalid regression predictions")
        return pd.Series(values, index=feature_frame.index, name=self.target_output)

    def save(self, directory: str | os.PathLike[str]) -> Path:
        """Save schema and native model artifacts into a new portable directory."""

        directory = Path(directory)
        directory.mkdir(parents=True)
        try:
            native_predictor = self.predictor_.clone(
                path=str(directory / "autogluon"), return_clone=True
            )
            schema = {
                "format": "ado-autogluon-predictor-v1",
                "feature_names_in_": self.feature_names_in_,
                "n_training_samples_": self.n_training_samples_,
                "n_fit_samples_": self.n_fit_samples_,
                "n_validation_samples_": self.n_validation_samples_,
                "target_output": self.target_output,
                "_categorical_columns": self._categorical_columns,
                "_category_dtypes": self._category_dtypes,
                "_random_state": self._random_state,
                "_fit_parameters": self._fit_parameters,
            }
            metadata = {
                "backend": "autogluon",
                "target_output": self.target_output,
                "feature_names": self.feature_names_in_.tolist(),
                "n_training_samples": self.n_training_samples_,
                "n_fit_samples": self.n_fit_samples_,
                "n_validation_samples": self.n_validation_samples_,
                "python_version": platform.python_version(),
                "autogluon_version": importlib.metadata.version("autogluon.tabular"),
                "random_state": self._random_state,
                "fit_parameters": self._fit_parameters,
            }
            _atomic_write(directory / "model.pkl", pickle.dumps(schema, protocol=5))
            _atomic_write(
                directory / "metadata.json",
                (json.dumps(metadata, indent=2) + "\n").encode("utf-8"),
            )
        except BaseException:
            shutil.rmtree(directory)
            raise
        self.predictor_ = native_predictor
        if self._temporary_directory is not None:
            self._temporary_directory.cleanup()
            self._temporary_directory = None
        return directory

    @classmethod
    def load(cls, directory: str | os.PathLike[str]) -> AutoGluonPredictor:
        """Load a trusted artifact directory, resolving native model paths locally."""

        predictor_class = _require_autogluon()
        directory = Path(directory)
        with (directory / "model.pkl").open("rb") as stream:
            schema = pickle.load(stream)  # noqa: S301
        if (
            not isinstance(schema, dict)
            or schema.get("format") != "ado-autogluon-predictor-v1"
        ):
            raise ValueError("model artifact does not contain an AutoGluonPredictor")
        schema = {name: value for name, value in schema.items() if name != "format"}
        return cls(
            predictor_=predictor_class.load(str(directory / "autogluon")),
            **schema,
        )
