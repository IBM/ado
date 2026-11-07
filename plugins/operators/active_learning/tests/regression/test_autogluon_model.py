# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT
# ruff: noqa: S101

"""Real AutoGluon fits and portable raw-feature prediction artifacts."""

import importlib.util
import json
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from active_learning.regression.autogluon_model import AutoGluonPredictor

FittedModel = tuple[AutoGluonPredictor, pd.DataFrame, pd.Series, Path]
HAS_AUTOGLUON = (
    importlib.util.find_spec("autogluon") is not None
    and importlib.util.find_spec("autogluon.tabular") is not None
)
requires_autogluon = pytest.mark.skipif(
    not HAS_AUTOGLUON, reason="optional AutoGluon dependency is not installed"
)


def _fit_parameters() -> dict:
    """Keep real AutoGluon training small and independent of optional model libraries."""

    return {
        "presets": "medium_quality",
        "hyperparameters": {"RF": {"n_estimators": 8}},
        "fit_weighted_ensemble": False,
        "num_cpus": 1,
    }


def _training_data() -> tuple[pd.DataFrame, pd.Series]:
    """Return a mixed pool and 122 labels with nonsequential row identities."""

    frame = pd.DataFrame(
        {
            "size": np.arange(160, dtype=float) + 1,
            "material": np.resize(np.array(["a", "b", None], dtype=object), 160),
            "code": np.resize(np.array([10, 20, 30], dtype=np.int64), 160),
        },
        index=pd.Index([f"entity-{index}" for index in range(160)], name="entity"),
    )
    frame.loc["entity-159", "material"] = "pool-only"
    frame.loc["entity-159", "code"] = 2**53 + 1
    labelled = np.random.default_rng(7).permutation(122)
    labels = pd.Series(
        2 * frame.iloc[labelled]["size"].to_numpy() + labelled % 5,
        index=frame.index[labelled],
    )
    return frame, labels


@pytest.fixture(scope="module")
def fitted_model(tmp_path_factory: pytest.TempPathFactory) -> FittedModel:
    """Fit and save one real predictor for schema, split, and reload checks."""

    if not HAS_AUTOGLUON:
        pytest.skip("optional AutoGluon dependency is not installed")
    frame, labels = _training_data()
    predictor = AutoGluonPredictor.fit(
        frame,
        labels,
        target_output="size",
        categorical_columns=["code"],
        fit_parameters=_fit_parameters(),
        random_state=42,
    )
    temporary_directory = Path(predictor.predictor_.path)
    directory = tmp_path_factory.mktemp("autogluon-model") / "finalized"
    predictor.save(directory)
    assert not temporary_directory.exists()
    return predictor, frame, labels, directory


def test_import_keeps_autogluon_optional() -> None:
    """Importing the wrapper leaves the AutoGluon package unloaded."""

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import active_learning.regression.autogluon_model; "
                "assert not any(name == 'autogluon' or name.startswith('autogluon.') "
                "for name in sys.modules)"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


@requires_autogluon
def test_fit_uses_only_labelled_rows_and_records_actual_holdout(
    fitted_model: FittedModel,
) -> None:
    """AutoGluon splits the 122 supplied labels into 97 fit and 25 validation rows."""

    predictor, frame, labels, _ = fitted_model
    train_features, train_labels = predictor.predictor_.load_data_internal("train")
    validation_features, validation_labels = predictor.predictor_.load_data_internal(
        "val"
    )
    assert predictor.n_training_samples_ == 122
    assert predictor.n_fit_samples_ == len(train_features) == 97
    assert predictor.n_validation_samples_ == len(validation_features) == 25
    assert not set(train_features.index) & set(validation_features.index)
    assert set(train_features.index) | set(validation_features.index) == set(
        labels.index
    )
    observed_labels = pd.concat([train_labels, validation_labels]).sort_index()
    np.testing.assert_array_equal(observed_labels.to_numpy(), labels.sort_index())
    np.testing.assert_array_equal(predictor.feature_names_in_, frame.columns)
    assert predictor.predictor_.label not in frame.columns
    assert predictor.predictor_.info()["random_state"] == 42


@requires_autogluon
def test_raw_schema_preserves_categories_column_order_and_index(
    fitted_model: FittedModel,
) -> None:
    """Raw subsets retain categorical levels and the caller's row identities."""

    predictor, frame, _, _ = fitted_model
    selected = frame.loc[["entity-159", "entity-14", "entity-2"]]
    expected = predictor.predict(selected)
    assert expected.name == "size"
    assert expected.index.equals(selected.index)
    assert np.isfinite(expected).all()
    reordered = selected[["code", "material", "size"]].copy()
    reordered["unused"] = np.inf
    pd.testing.assert_series_equal(predictor.predict(reordered), expected)
    for identifier in selected.index:
        pd.testing.assert_series_equal(
            predictor.predict(selected.loc[[identifier]]), expected.loc[[identifier]]
        )
    assert predictor.predict(frame.iloc[:0]).empty


@requires_autogluon
@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("material", "outside-pool", "unknown category"),
        ("code", 40, "unknown category"),
        ("size", np.inf, "finite"),
        ("size", "invalid", "numeric"),
    ],
)
def test_predict_rejects_invalid_raw_values(
    fitted_model: FittedModel, column: str, value: object, message: str
) -> None:
    """Malformed raw values fail before reaching the AutoGluon model."""

    predictor, frame, _, _ = fitted_model
    invalid = frame.iloc[[0]].astype(object)
    invalid.loc[invalid.index[0], column] = value
    with pytest.raises(ValueError, match=message):
        predictor.predict(invalid)


@requires_autogluon
def test_predict_requires_original_feature_columns(fitted_model: FittedModel) -> None:
    """Prediction reports missing raw columns with a useful error."""

    predictor, frame, _, _ = fitted_model
    with pytest.raises(ValueError, match=r"missing.*size"):
        predictor.predict(frame.drop(columns="size"))


@requires_autogluon
def test_saved_artifact_survives_relocation(
    fitted_model: FittedModel, tmp_path: Path
) -> None:
    """Native model paths resolve under the moved artifact directory."""

    predictor, frame, _, directory = fitted_model
    metadata = json.loads((directory / "metadata.json").read_text())
    assert metadata["backend"] == "autogluon"
    assert metadata["n_training_samples"] == 122
    assert metadata["n_fit_samples"] == 97
    assert metadata["n_validation_samples"] == 25
    assert metadata["feature_names"] == list(frame.columns)
    assert metadata["autogluon_version"]
    assert metadata["random_state"] == 42
    assert metadata["fit_parameters"] == _fit_parameters()
    assert (directory / "model.pkl").is_file()
    assert (directory / "autogluon/predictor.pkl").is_file()
    copied = tmp_path / "before-move"
    moved = tmp_path / "after-move"
    shutil.copytree(directory, copied)
    copied.rename(moved)
    restored = AutoGluonPredictor.load(moved)
    assert Path(restored.predictor_.path) == moved / "autogluon"
    pd.testing.assert_series_equal(restored.predict(frame), predictor.predict(frame))
    before = (directory / "model.pkl").read_bytes()
    with pytest.raises(FileExistsError):
        predictor.save(directory)
    assert (directory / "model.pkl").read_bytes() == before


@pytest.mark.parametrize("count", [1, 2, 3])
def test_fit_requires_four_labels(count: int) -> None:
    """An explicit minimum avoids AutoGluon's empty-training-split error."""

    frame, labels = _training_data()
    with pytest.raises(ValueError, match="at least four"):
        AutoGluonPredictor.fit(
            frame,
            labels.iloc[:count],
            target_output="response",
            fit_parameters=_fit_parameters(),
        )


@requires_autogluon
def test_four_label_budget_can_fit_and_predict(tmp_path: Path) -> None:
    """A pilot-only run at the supported minimum yields a loadable predictor."""

    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    predictor = AutoGluonPredictor.fit(
        frame,
        pd.Series([2.0, 4.0, 6.0, 8.0], index=[0, 1, 2, 3]),
        target_output="response",
        fit_parameters=_fit_parameters(),
    )
    directory = predictor.save(tmp_path / "pilot-only")
    restored = AutoGluonPredictor.load(directory)
    assert restored.n_training_samples_ == 4
    assert restored.n_fit_samples_ == restored.n_validation_samples_ == 2
    assert np.isfinite(restored.predict(frame)).all()


@pytest.mark.parametrize(
    ("labels", "message"),
    [
        (
            pd.Series(
                [1.0, 2.0, 3.0, 4.0],
                index=["missing", "entity-1", "entity-2", "entity-3"],
            ),
            "index.*feature",
        ),
        (pd.Series([1.0, 2.0, 3.0, 4.0], index=["entity-1"] * 4), "unique"),
        (
            pd.Series(
                [1.0, 2.0, 3.0, np.inf],
                index=["entity-0", "entity-1", "entity-2", "entity-3"],
            ),
            "finite",
        ),
    ],
)
def test_fit_rejects_invalid_labels_before_training(
    labels: pd.Series, message: str
) -> None:
    """Only finite, uniquely identified measured targets can enter training."""

    frame, _ = _training_data()
    with pytest.raises(ValueError, match=message):
        AutoGluonPredictor.fit(frame, labels, target_output="response")


@requires_autogluon
def test_failed_save_removes_incomplete_artifact(
    fitted_model: FittedModel, tmp_path: Path
) -> None:
    """A real serialization failure removes the newly copied native artifact."""

    predictor, frame, _, _ = fitted_model
    invalid = replace(
        predictor,
        _category_dtypes={
            **predictor._category_dtypes,
            "unserializable": (value for value in range(1)),
        },
    )
    directory = tmp_path / "incomplete"
    with pytest.raises(TypeError, match=r"pickle.*generator"):
        invalid.save(directory)
    assert not directory.exists()
    assert np.isfinite(predictor.predict(frame)).all()
