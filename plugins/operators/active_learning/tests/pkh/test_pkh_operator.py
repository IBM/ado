# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Regression tests for PKH measurements, output protection, and saved predictors."""

import pickle
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from active_learning.operator import PredictorExport, _output_after_sampling
from active_learning.pkh.data import measured_labels, validate_target_output
from active_learning.pkh.operator import _finalize_model, pkh
from active_learning.pkh.parameters import (
    ActiveLearningAutoGluonParameters,
    PKHParameters,
)
from active_learning.pkh.sampler import PKHSampleSelector

from ado.core.discoveryspace.space import DiscoverySpace
from ado.core.operation.operation import OperationOutput
from ado.core.operation.resource import OperationResource, OperationResourceStatus
from ado.schema.domain import PropertyDomain
from ado.schema.measurementspace import MeasurementSpace, MeasurementSpaceConfiguration
from ado.schema.property import ConstitutiveProperty


def test_rejects_multiple_experiments_before_reserving_output(
    empty_space: DiscoverySpace, tmp_path: Path
) -> None:
    """PKH validates the experiment count before it reserves an output directory."""
    experiment = empty_space.measurementSpace.experiments[0]
    space = DiscoverySpace(
        entitySpace=empty_space.entitySpace,
        measurementSpace=MeasurementSpace(
            MeasurementSpaceConfiguration(
                experiments=[
                    experiment,
                    experiment.model_copy(update={"identifier": "other"}),
                ]
            )
        ),
        sample_store=empty_space.sample_store,
        project_context=empty_space.project_context,
    )
    (tmp_path / "model_finalized").mkdir()
    parameters = PKHParameters(
        targetOutput="self",
        numberEntities=1,
        outputDirectory=str(tmp_path / "model"),
    )
    with pytest.raises(ValueError, match="exactly 1 experiment"):
        pkh.__wrapped__(space, parameters=parameters)


@pytest.mark.parametrize(
    ("target", "error"),
    [
        ("is_predicted", "output column names conflict"),
        ("selected-self", "must identify exactly one observed property"),
    ],
)
def test_rejects_unusable_target_names(
    empty_space: DiscoverySpace, target: str, error: str
) -> None:
    """PKH requires one identifiable target with its own output column."""
    with pytest.raises(ValueError, match=error):
        validate_target_output(target, empty_space)


def test_existing_model_is_preserved(
    empty_space: DiscoverySpace, tmp_path: Path
) -> None:
    """A rerun must leave the previous predictor intact."""
    directory = tmp_path / "model_finalized"
    directory.mkdir()
    model = directory / "model.pkl"
    model.write_bytes(b"previous model")
    parameters = PKHParameters(
        targetOutput="self",
        numberEntities=1,
        outputDirectory=str(tmp_path / "model"),
    )
    with pytest.raises(FileExistsError):
        pkh.__wrapped__(empty_space, parameters=parameters)
    assert model.read_bytes() == b"previous model"


def test_failed_sampling_preserves_status_and_skips_publication(
    categorical_space: DiscoverySpace,
    make_test_space: Callable[[list[ConstitutiveProperty]], DiscoverySpace],
    tmp_path: Path,
) -> None:
    """PKH preserves the failed walk's status even when every candidate has a label."""
    kind, count = categorical_space.entitySpace.constitutiveProperties
    space = make_test_space(
        [
            kind.model_copy(update={"propertyDomain": PropertyDomain(values=["b"])}),
            count,
        ]
    )
    assert len(measured_labels(space, "selected-self")) == space.entitySpace.size
    walk = OperationOutput(
        operation=OperationResource(
            operationType="explore",
            operatorIdentifier="random_walk",
            config={"inputs": {"discoverySpace": space.reference}, "operation": {}},
        ),
        exitStatus=OperationResourceStatus(
            event="finished", exit_state="fail", message="Measurement run failed"
        ),
    )
    directory = tmp_path / "model_finalized"
    directory.mkdir()
    parameters = PKHParameters(targetOutput="self", numberEntities=1)
    target = validate_target_output(parameters.targetOutput, space)
    result, export = _output_after_sampling(
        walk, space, parameters, target, directory, PKHSampleSelector
    )
    assert export is None
    assert result.exitStatus == walk.exitStatus
    assert not list(directory.iterdir())


@pytest.mark.parametrize("backend", ["random_forest", "autogluon"])
def test_saved_predictor_reproduces_categorical_predictions(
    categorical_space: DiscoverySpace, tmp_path: Path, backend: str
) -> None:
    """Reload the predictor and compare its output with the exported predictions."""
    settings = None
    if backend == "autogluon":
        pytest.importorskip("autogluon.tabular")
        settings = ActiveLearningAutoGluonParameters(
            timeLimit=10, hyperparameters={"RF": {"n_estimators": 3}}
        )
    parameters = PKHParameters(
        targetOutput="self",
        numberEntities=1,
        nEstimators=3,
        finalPredictor=backend,
        autoGluon=settings,
    )
    sampled = categorical_space.entitySpace.entity_for_point(
        {"kind": "b", "count": 2**53 + 1}
    )
    target = validate_target_output(parameters.targetOutput, categorical_space)
    export = PredictorExport(
        categorical_space,
        target,
        parameters,
        tmp_path,
        {sampled.identifier},
        PKHSampleSelector,
    )
    _finalize_model(export, parameters)
    table = pd.read_csv(
        tmp_path / "combined_predictions.csv",
        dtype={"count": str},
        keep_default_na=False,
    )
    assert table.loc[~table.is_predicted, "self"].tolist() == [0, 1, 2, 10, 20] * 2
    unmeasured = table.loc[table.is_predicted]
    assert unmeasured.kind.tolist() == ["c"] * 5 + ["d"] * 5
    raw_counts = [0.5, 2**53, 2**53 + 1, None, b"code"] * 2
    inputs = pd.DataFrame(
        {"kind": unmeasured.kind.tolist(), "count": pd.Series(raw_counts, dtype=object)}
    )
    if backend == "autogluon":
        from autogluon.tabular import TabularPredictor

        model = TabularPredictor.load(str(tmp_path / "autogluon"))
        assert model.problem_type == "regression"
        assert model.transform_features(inputs)["count"].nunique() == 5
    else:
        with (tmp_path / "model.pkl").open("rb") as model_file:
            model = pickle.load(model_file)  # noqa: S301 - reload this test's model
        # Each known category activates one indicator per input column.
        features = model.named_steps["features"].transform(inputs)
        assert (features.sum(axis=1) == len(inputs.columns)).all()
        assert len(np.unique(features, axis=0)) == len(inputs)
    assert model.predict(inputs).tolist() == pytest.approx(unmeasured["self"].tolist())
    assert unmeasured["count"].tolist() == [str(value) for value in raw_counts]


def test_incomplete_measurements_cannot_publish_a_model(
    categorical_space: DiscoverySpace, tmp_path: Path
) -> None:
    """PKH checks the new measurements before it publishes a model."""
    directory = tmp_path / "model_finalized"
    directory.mkdir()
    parameters = PKHParameters(targetOutput="self", numberEntities=1)
    missing = categorical_space.entitySpace.entity_for_point(
        {"kind": "d", "count": 2**53 + 1}
    )
    target = validate_target_output(parameters.targetOutput, categorical_space)
    with pytest.raises(ValueError, match="Missing finite target measurements"):
        PredictorExport(
            categorical_space,
            target,
            parameters,
            directory,
            {missing.identifier},
            PKHSampleSelector,
        )
    with pytest.raises(ValueError, match="budget is incomplete"):
        PredictorExport(
            categorical_space, target, parameters, directory, set(), PKHSampleSelector
        )
    assert not list(directory.iterdir())


def test_autogluon_requires_enough_labels_for_the_selected_target(
    categorical_space: DiscoverySpace, tmp_path: Path
) -> None:
    """PKH checks the attainable and collected target counts before training."""
    pytest.importorskip("autogluon.tabular")
    parameters = PKHParameters(
        targetOutput="selected-selected-self",
        numberEntities=1,
        finalPredictor="autogluon",
        outputDirectory=str(tmp_path / "model"),
    )
    with pytest.raises(ValueError, match=r"at least 4.*at most 2"):
        pkh.__wrapped__(categorical_space, parameters=parameters)
    sampled = categorical_space.entitySpace.entity_for_point({"kind": 1, "count": 0.5})
    target = validate_target_output(parameters.targetOutput, categorical_space)
    with pytest.raises(ValueError, match=r"at least 4.*collected 1"):
        PredictorExport(
            categorical_space,
            target,
            parameters,
            tmp_path,
            {sampled.identifier},
            PKHSampleSelector,
        )
