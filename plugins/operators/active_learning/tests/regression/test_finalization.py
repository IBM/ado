# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT
# ruff: noqa: S101

"""Tests for final-model training from stored target measurements."""

from pathlib import Path

import pytest
from active_learning.regression._finalize import (
    _final_model_directory,
    _finalize_predictive_model,
)
from active_learning.regression.model import RandomForestPredictor
from active_learning.regression.pkh.parameters import PKHOperatorParameters
from test_samplers import _adaptive_test_space, _entity_with_label


def test_final_model_includes_existing_and_latest_measurements(tmp_path: Path) -> None:
    """A fresh fit uses all stored labels and refreshes the entity cache."""

    space, store, target, entities = _adaptive_test_space()
    for index in (2, 4):
        entity = _entity_with_label(entities[index], target, float(index + 1))
        store.entities[entity.identifier] = entity
    directory = tmp_path / "finalized"
    metadata = _finalize_predictive_model(
        space,
        parameters=PKHOperatorParameters(
            targetOutput="latency", numberEntities=2, nEstimators=3, nJobs=1
        ),
        directory=directory,
    )
    predictor = RandomForestPredictor.load(directory)

    assert predictor.n_training_samples_ == 3
    assert metadata["trainingEntityCount"] == 3
    assert set(metadata["trainingEntityIdentifiers"]) == {
        entities[index].identifier for index in (0, 2, 4)
    }
    assert metadata["observedProperty"] == target.identifier
    assert metadata["modelPath"] == str(directory / "model.pkl")
    assert all(tree.tree_.weighted_n_node_samples[0] == 3 for tree in predictor.forest_)
    assert (None, False, True) in store.reads


def test_final_model_requires_target_measurements(tmp_path: Path) -> None:
    """An empty training set cannot produce a saved predictor."""

    space, store, _, _ = _adaptive_test_space()
    store.entities.clear()
    directory = tmp_path / "finalized"

    with pytest.raises(ValueError, match="no target measurements"):
        _finalize_predictive_model(
            space,
            parameters=PKHOperatorParameters(targetOutput="latency", numberEntities=2),
            directory=directory,
        )
    assert not directory.exists()


def test_final_model_rejects_an_incomplete_target_set(tmp_path: Path) -> None:
    """A missing final target cannot silently produce a partial training set."""

    space, _, _, _ = _adaptive_test_space()
    directory = tmp_path / "finalized"

    with pytest.raises(ValueError, match="expected at least 2 target measurements"):
        _finalize_predictive_model(
            space,
            parameters=PKHOperatorParameters(targetOutput="latency", numberEntities=1),
            directory=directory,
            minimum_training_count=2,
        )
    assert not directory.exists()


def test_final_model_directory_preserves_existing_artifacts(tmp_path: Path) -> None:
    """Reject an occupied final-model path before measuring more entities."""

    base = tmp_path / "model"
    directory = _final_model_directory(str(base))
    assert directory == tmp_path / "model_finalized"
    directory.mkdir()
    marker = directory / "existing-model"
    marker.write_text("retained")

    with pytest.raises(FileExistsError, match="outputDirectory"):
        _final_model_directory(str(base))
    assert marker.read_text() == "retained"


def test_autogluon_requires_enough_labels_before_measurement(tmp_path: Path) -> None:
    """A campaign too small for AutoGluon's validation split fails before sampling."""

    from active_learning.regression._shared import _run_predictive_operator
    from active_learning.regression.pkh.sampler import PKHSampleSelector

    space, store, _, _ = _adaptive_test_space()
    before = dict(store.entities)
    parameters = PKHOperatorParameters(
        targetOutput="latency", numberEntities=1, finalPredictor="autogluon"
    )
    with pytest.raises(ValueError, match="at least 4"):
        _run_predictive_operator(
            discovery_space=space,
            operation_info=None,
            parameters=parameters,
            number_entities=1,
            sampler_class=PKHSampleSelector,
            output_directory=str(tmp_path / "model"),
        )
    assert store.entities == before
    assert not (tmp_path / "model_finalized").exists()


def test_generic_loader_reads_existing_random_forest_artifacts(tmp_path: Path) -> None:
    """Existing model.pkl artifacts retain their predictions with the common loader."""

    import json

    import pandas as pd
    from active_learning.regression import load_predictor

    space, _, _, _ = _adaptive_test_space()
    directory = tmp_path / "finalized"
    _finalize_predictive_model(
        space,
        parameters=PKHOperatorParameters(
            targetOutput="latency", numberEntities=1, nEstimators=3, nJobs=1
        ),
        directory=directory,
    )
    metadata_path = directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.pop("backend", None)
    metadata_path.write_text(json.dumps(metadata))
    legacy = RandomForestPredictor.load(directory)
    restored = load_predictor(directory)
    frame = pd.DataFrame({"workers": [1, 2, 3]})
    pd.testing.assert_series_equal(restored.predict(frame), legacy.predict(frame))
