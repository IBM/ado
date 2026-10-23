# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT
# ruff: noqa: S101

"""Tests for final-model training from stored target measurements."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from active_learning.regression import load_predictor
from active_learning.regression._finalize import (
    _final_model_directory,
    _finalize_predictive_model,
)
from active_learning.regression._shared import _finite_pool, _measured_labels
from active_learning.regression.model import RandomForestPredictor
from active_learning.regression.pkh.parameters import PKHOperatorParameters
from test_samplers import _adaptive_test_space, _entity_with_label

from ado.schema.domain import PropertyDomain, VariableTypeEnum
from ado.schema.entityspace import EntitySpaceRepresentation
from ado.schema.property import ConstitutiveProperty


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


@pytest.mark.parametrize("backend", ["random_forest", "autogluon"])
def test_final_model_preserves_pool_schema_and_label_alignment(
    tmp_path: Path, backend: str
) -> None:
    """Saved predictors retain raw numeric features, categories, and measured rows."""

    if backend == "autogluon":
        pytest.importorskip("autogluon.tabular")
    space, store, target, _ = _adaptive_test_space()
    category_values = [2**53, 2**53 + 1]
    domains = {
        "workers": PropertyDomain(values=[1, 2, 3]),
        "fraction": PropertyDomain(values=[0.5, 1.5]),
        "material": PropertyDomain(values=["a", "b"]),
        "code": PropertyDomain(
            values=category_values,
            variableType=VariableTypeEnum.CATEGORICAL_VARIABLE_TYPE,
        ),
    }
    space.entitySpace = EntitySpaceRepresentation(
        constitutiveProperties=[
            ConstitutiveProperty(identifier=name, propertyDomain=domain)
            for name, domain in domains.items()
        ]
    )
    store.entities.clear()
    pool = _finite_pool(space)
    labels = pd.Series([11.0, 22.0, 33.0, 44.0], index=[1, 6, 13, 22])
    for index, label in labels.items():
        entity = pool.entities[index]
        if index == 13:
            entity = entity.model_copy(update={"identifier": "external-measured-row"})
        entity = _entity_with_label(entity, target, label)
        store.entities[entity.identifier] = entity
    pool = _finite_pool(space, refresh=True)
    assert pool.feature_frame.columns.tolist() == list(domains)
    assert pool.categorical_columns == ("material", "code")
    assert _measured_labels(space, pool, target) == labels.to_dict()
    assert pool.entities[13].identifier == "external-measured-row"

    forest_parameters = {
        "n_estimators": 8,
        "min_samples_leaf": 1,
        "max_features": 1.0,
        "random_state": 42,
        "n_jobs": 1,
    }
    directory = tmp_path / backend
    metadata = _finalize_predictive_model(
        space,
        parameters=PKHOperatorParameters(
            targetOutput="latency",
            numberEntities=4,
            nEstimators=8,
            minSamplesLeaf=1,
            maxFeatures=1.0,
            seed=42,
            nJobs=1,
            finalPredictor=backend,
            autoGluon={"numCpus": 1, "hyperparameters": {"RF": {"n_estimators": 8}}},
        ),
        directory=directory,
        minimum_training_count=4,
    )
    predictor = load_predictor(directory)
    assert predictor.feature_names_in_.tolist() == list(domains)
    assert predictor._categorical_columns == ("material", "code")
    assert predictor._category_dtypes["code"].categories.tolist() == category_values
    assert predictor.n_training_samples_ == metadata["trainingEntityCount"] == 4
    assert metadata["fitEntityCount"] + metadata["validationEntityCount"] == 4
    assert metadata["trainingEntityIdentifiers"] == [
        pool.entities[index].identifier for index in labels.index
    ]
    probe = pool.feature_frame.iloc[[1]].copy()
    probe["workers"] = 4
    probe["fraction"] = 2.5
    assert np.isfinite(predictor.predict(probe).to_numpy()).all()
    if backend == "random_forest":
        expected = RandomForestPredictor.fit(
            pool.feature_frame,
            labels,
            target_output="latency",
            categorical_columns=pool.categorical_columns,
            forest_parameters=forest_parameters,
        )
        np.testing.assert_array_equal(
            predictor.predict(pool.feature_frame), expected.predict(pool.feature_frame)
        )


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
