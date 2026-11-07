# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT
# ruff: noqa: S101

"""Prediction and persistence tests for fitted finite-pool regression models."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from active_learning.regression._shared import _encode_features
from active_learning.regression.model import RandomForestPredictor
from sklearn.ensemble import RandomForestRegressor


def _training_data() -> tuple[pd.DataFrame, pd.Series, dict]:
    """Return mixed finite-pool features and a labelled subset in acquisition order."""

    frame = pd.DataFrame(
        {
            "size": [1.0, 2.0, 4.0, 8.0, 16.0],
            "material": pd.Categorical(
                ["b", "a", None, "b", "a"], categories=["unused", "b", "a"]
            ),
            "code": [10, 20, 30, 20, 10],
        },
        index=pd.Index(["a", "b", "c", "d", "e"], name="entity"),
    )
    labels = pd.Series([12.0, 3.0, 1.0], index=["e", "c", "a"])
    parameters = {
        "n_estimators": 13,
        "min_samples_leaf": 1,
        "max_features": 1.0,
        "bootstrap": False,
        "random_state": 17,
        "n_jobs": 1,
    }
    return frame, labels, parameters


def _predictor() -> tuple[RandomForestPredictor, pd.DataFrame]:
    """Fit a small deterministic forest with an explicit numeric categorical column."""

    frame, labels, parameters = _training_data()
    predictor = RandomForestPredictor.fit(
        frame,
        labels,
        target_output="response",
        categorical_columns=["code"],
        forest_parameters=parameters,
    )
    return predictor, frame


def test_fit_uses_labelled_rows_and_complete_pool_encoding() -> None:
    """The fitted forest matches labelled-only training with the original encoder."""

    frame, labels, parameters = _training_data()
    encoded = _encode_features(frame, categorical_columns=["code"])
    expected = RandomForestRegressor(**parameters).fit(
        encoded[frame.index.get_indexer(labels.index)], labels.to_numpy()
    )
    predictor, _ = _predictor()
    np.testing.assert_array_equal(predictor.predict(frame), expected.predict(encoded))
    np.testing.assert_array_equal(predictor.feature_names_in_, frame.columns)
    assert predictor.forest_.n_features_in_ == encoded.shape[1]
    assert predictor.n_training_samples_ == len(labels)
    assert predictor.target_output == "response"
    assert predictor.predict(frame).index.equals(frame.index)
    assert predictor.predict(frame).name == "response"


def test_prediction_schema_is_stable_for_single_rows_and_reordered_columns() -> None:
    """Subsets and reordered columns retain the fitted numeric/category mapping."""

    predictor, frame = _predictor()
    expected = predictor.predict(frame)
    for index in frame.index:
        pd.testing.assert_series_equal(
            predictor.predict(frame.loc[[index]]), expected.loc[[index]]
        )
    reordered = frame.loc[["e", "b"], ["code", "material", "size"]].copy()
    reordered["irrelevant"] = np.inf
    pd.testing.assert_series_equal(
        predictor.predict(reordered), expected.loc[["e", "b"]]
    )
    empty = predictor.predict(frame.iloc[:0])
    assert empty.empty
    assert empty.index.equals(frame.iloc[:0].index)
    assert empty.name == "response"


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("material", "unknown", "unknown category"),
        ("code", 40, "unknown category"),
        ("size", np.nan, "finite"),
        ("size", np.inf, "finite"),
        ("size", "invalid", "numeric"),
    ],
)
def test_prediction_rejects_unknown_categories_and_invalid_numbers(
    column: str, value: object, message: str
) -> None:
    """Invalid raw rows fail before entering the fitted forest."""

    predictor, frame = _predictor()
    row = frame.iloc[[0]].astype(object)
    row.loc[row.index[0], column] = value
    with pytest.raises(ValueError, match=message):
        predictor.predict(row)


def test_prediction_requires_original_feature_columns() -> None:
    """Every original feature must be present at prediction time."""

    predictor, frame = _predictor()
    with pytest.raises(ValueError, match=r"missing.*size"):
        predictor.predict(frame.drop(columns="size"))


@pytest.mark.parametrize(
    ("labels", "message"),
    [
        (pd.Series(dtype=float), "at least one.*label"),
        (pd.Series([1.0], index=["outside"]), "index.*feature"),
        (pd.Series([1.0, 2.0], index=["a", "a"]), "unique"),
        (pd.Series([np.nan], index=["a"]), "finite"),
        (pd.Series([np.inf], index=["a"]), "finite"),
        (pd.Series([1.0 + 2.0j], index=["a"]), "real"),
    ],
)
def test_fit_rejects_invalid_label_subsets(labels: pd.Series, message: str) -> None:
    """Training labels must uniquely identify measured rows with finite real values."""

    frame, _, parameters = _training_data()
    with pytest.raises(ValueError, match=message):
        RandomForestPredictor.fit(
            frame, labels, target_output="response", forest_parameters=parameters
        )


@pytest.mark.parametrize(
    "invalid_schema", ["empty", "duplicate_columns", "duplicate_index"]
)
def test_fit_rejects_ambiguous_feature_schema(invalid_schema: str) -> None:
    """Empty pools and ambiguous row or column identities are rejected."""

    frame, labels, parameters = _training_data()
    if invalid_schema == "empty":
        frame = frame.iloc[:0]
        message = "at least one"
    elif invalid_schema == "duplicate_columns":
        frame.columns = ["size", "size", "code"]
        message = "unique"
    else:
        frame.index = ["a", "a", "c", "d", "e"]
        message = "unique"
    with pytest.raises(ValueError, match=message):
        RandomForestPredictor.fit(
            frame, labels, target_output="response", forest_parameters=parameters
        )


def test_saved_model_round_trip_preserves_predictions_and_metadata(
    tmp_path: Path,
) -> None:
    """Saved forests retain their raw-feature schema and labelled training count."""

    predictor, frame = _predictor()
    directory = tmp_path / "trained-model"
    assert predictor.save(directory) == directory
    assert (directory / "model.pkl").is_file()
    metadata = json.loads((directory / "metadata.json").read_text())
    assert metadata["target_output"] == "response"
    assert metadata["feature_names"] == list(frame.columns)
    assert metadata["n_training_samples"] == 3
    assert metadata["python_version"]
    assert metadata["sklearn_version"]
    restored = RandomForestPredictor.load(directory)
    pd.testing.assert_series_equal(restored.predict(frame), predictor.predict(frame))
    pd.testing.assert_series_equal(
        restored.predict(frame.iloc[[2]]), predictor.predict(frame.iloc[[2]])
    )
    before = (directory / "model.pkl").read_bytes()
    with pytest.raises(FileExistsError, match="trained-model"):
        predictor.save(directory)
    assert (directory / "model.pkl").read_bytes() == before


def test_inferred_object_categories_preserve_batch_encoding() -> None:
    """String columns retain full-pool categories for individual prediction rows."""

    frame = pd.DataFrame({"kind": ["z", "a", None], "size": [1, 2, 3]})
    labels = pd.Series([1.0, 3.0], index=[0, 2])
    predictor = RandomForestPredictor.fit(
        frame,
        labels,
        target_output="response",
        forest_parameters={"n_estimators": 5, "random_state": 42},
    )
    expected = predictor.predict(frame)
    assert predictor.forest_.n_features_in_ == _encode_features(frame).shape[1]
    for index in frame.index:
        pd.testing.assert_series_equal(
            predictor.predict(frame.loc[[index]]), expected.loc[[index]]
        )


def test_fit_checks_numeric_features_on_unlabelled_rows() -> None:
    """The complete raw pool must have finite numeric features."""

    frame = pd.DataFrame({"size": [1.0, np.inf]})
    with pytest.raises(ValueError, match="finite"):
        RandomForestPredictor.fit(
            frame, pd.Series([2.0], index=[0]), target_output="response"
        )


def test_failed_save_removes_new_incomplete_artifact(tmp_path: Path) -> None:
    """A real pickle serialization failure removes the new artifact directory."""

    predictor, _ = _predictor()
    predictor.forest_.serialization_obstacle = (index for index in range(1))
    directory = tmp_path / "incomplete-model"
    with pytest.raises(TypeError, match=r"pickle.*generator"):
        predictor.save(directory)
    assert not directory.exists()
