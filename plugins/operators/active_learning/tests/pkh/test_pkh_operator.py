# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Regression tests for PKH measurements, output protection, and saved predictors."""

import pickle
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from active_learning.pkh.data import validate_target_output
from active_learning.pkh.operator import _finalize_model, pkh
from active_learning.pkh.parameters import (
    ActiveLearningAutoGluonParameters,
    PKHParameters,
)
from active_learning.pkh.sampler import PKHSampleSelector

from ado.core.discoveryspace.space import DiscoverySpace
from ado.core.samplestore.sql import SQLSampleStore
from ado.metastore.project import ProjectContext
from ado.schema.domain import PropertyDomain
from ado.schema.entityspace import EntitySpaceRepresentation
from ado.schema.experiment import Experiment
from ado.schema.measurementspace import MeasurementSpace, MeasurementSpaceConfiguration
from ado.schema.observed_property import ObservedPropertyValue
from ado.schema.property import ConcretePropertyDescriptor, ConstitutiveProperty
from ado.schema.result import ValidMeasurementResult
from ado.utilities.location import SQLiteStoreConfiguration


@pytest.fixture
def categorical_space(tmp_path: Path) -> DiscoverySpace:
    """Six measured points, with a seventh measured only by another experiment."""
    kind = ConstitutiveProperty(
        identifier="kind", propertyDomain=PropertyDomain(values=[1, "b", "c", "d"])
    )
    count = ConstitutiveProperty(
        identifier="count", propertyDomain=PropertyDomain(values=[0, 1])
    )
    entity_space = EntitySpaceRepresentation(constitutiveProperties=[kind, count])
    experiment = Experiment(
        identifier="selected",
        actuatorIdentifier="test",
        requiredProperties=(kind, count),
        targetProperties=[
            ConcretePropertyDescriptor(identifier="response"),
            ConcretePropertyDescriptor(identifier="is_predicted"),
        ],
    )
    other = experiment.model_copy(update={"identifier": "other"})
    location = SQLiteStoreConfiguration(path=str(tmp_path / "pkh.db"), database="pkh")
    store = SQLSampleStore(identifier="labels", storageLocation=location, parameters={})
    entities = [
        entity_space.entity_for_point({"kind": value, "count": number})
        for value in kind.propertyDomain.domain_values
        for number in (0, 1)
    ]
    store.addEntities(entities)
    store.get_entities(require_measurements=True)  # Cache the empty measurement set.
    writer = SQLSampleStore(
        identifier="labels", storageLocation=location, parameters={}
    )
    writer.add_measurement_results(
        [
            ValidMeasurementResult(
                entityIdentifier=entity.identifier,
                measurements=[ObservedPropertyValue(property=observed, value=value)],
            )
            for entity, observed, value in zip(
                entities[:7],
                [experiment.observedProperties[0]] * 6 + [other.observedProperties[0]],
                [0, 1] * 3 + [999],
                strict=True,
            )
        ],
        skip_relationship_to_request=True,
    )
    return DiscoverySpace(
        entitySpace=entity_space,
        measurementSpace=MeasurementSpace(
            MeasurementSpaceConfiguration(experiments=[experiment])
        ),
        sample_store=store,
        project_context=ProjectContext(project="pkh", metadataStore=location),
    )


def test_rejects_multiple_experiments_before_reserving_output(
    categorical_space: DiscoverySpace, tmp_path: Path
) -> None:
    """An invalid space must fail before PKH claims an output directory."""
    experiment = categorical_space.measurementSpace.experiments[0]
    space = DiscoverySpace(
        entitySpace=categorical_space.entitySpace,
        measurementSpace=MeasurementSpace(
            MeasurementSpaceConfiguration(
                experiments=[
                    experiment,
                    experiment.model_copy(update={"identifier": "other"}),
                ]
            )
        ),
        sample_store=categorical_space.sample_store,
        project_context=categorical_space.project_context,
    )
    (tmp_path / "model_finalized").mkdir()
    parameters = PKHParameters(
        targetOutput="response",
        numberEntities=1,
        outputDirectory=str(tmp_path / "model"),
    )
    with pytest.raises(ValueError, match="exactly 1 experiment"):
        pkh.__wrapped__(space, parameters=parameters)


def test_rejects_target_that_would_overwrite_prediction_flag(
    categorical_space: DiscoverySpace,
) -> None:
    """A measured target must have its own output column."""
    with pytest.raises(ValueError, match="output column names conflict"):
        validate_target_output("is_predicted", categorical_space)


def test_existing_model_is_preserved(
    categorical_space: DiscoverySpace, tmp_path: Path
) -> None:
    """A rerun must leave the previous predictor intact."""
    directory = tmp_path / "model_finalized"
    directory.mkdir()
    model = directory / "model.pkl"
    model.write_bytes(b"previous model")
    parameters = PKHParameters(
        targetOutput="response",
        numberEntities=1,
        outputDirectory=str(tmp_path / "model"),
    )
    with pytest.raises(FileExistsError):
        pkh.__wrapped__(categorical_space, parameters=parameters)
    assert model.read_bytes() == b"previous model"


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
        targetOutput=categorical_space.measurementSpace.observedProperties[
            0
        ].identifier,
        numberEntities=1,
        nEstimators=3,
        finalPredictor=backend,
        autoGluon=settings,
    )
    sampled = categorical_space.entitySpace.entity_for_point({"kind": "c", "count": 1})
    _finalize_model(categorical_space, parameters, tmp_path, {sampled.identifier})
    table = pd.read_csv(tmp_path / "combined_predictions.csv")
    assert table.loc[~table.is_predicted, "response"].tolist() == [0, 1] * 3
    unmeasured = table.loc[table.is_predicted]
    assert unmeasured.kind.tolist() == ["d", "d"]
    if backend == "autogluon":
        from autogluon.tabular import TabularPredictor

        model = TabularPredictor.load(str(tmp_path / "autogluon"))
        assert model.problem_type == "regression"
    else:
        with (tmp_path / "model.pkl").open("rb") as model_file:
            model = pickle.load(model_file)  # noqa: S301 - reload this test's model
    assert model.predict(unmeasured[["kind", "count"]]).tolist() == pytest.approx(
        unmeasured.response.tolist()
    )


def test_incomplete_measurements_cannot_publish_a_model(
    categorical_space: DiscoverySpace, tmp_path: Path
) -> None:
    """Prior labels cannot make an incomplete operation appear successful."""
    directory = tmp_path / "model_finalized"
    directory.mkdir()
    parameters = PKHParameters(targetOutput="response", numberEntities=1)
    missing = categorical_space.entitySpace.entity_for_point({"kind": "d", "count": 1})
    with pytest.raises(ValueError, match="Missing finite target measurements"):
        _finalize_model(categorical_space, parameters, directory, {missing.identifier})
    with pytest.raises(ValueError, match="budget is incomplete"):
        _finalize_model(categorical_space, parameters, directory, set())
    assert not list(directory.iterdir())


@pytest.mark.asyncio
async def test_sampler_requires_current_target_before_next_entity(
    categorical_space: DiscoverySpace,
) -> None:
    """Only entities without target labels remain eligible for sampling."""

    async def get_space() -> DiscoverySpace:
        """Return the space through the asynchronous manager interface."""
        return categorical_space

    manager = SimpleNamespace(discoverySpace=SimpleNamespace(remote=get_space))
    sampler = PKHSampleSelector(
        PKHParameters(targetOutput="response", numberEntities=2)
    )
    stream = await sampler.remoteEntityIterator(manager)
    [first] = await anext(stream)
    assert first.valueForConstitutivePropertyIdentifier("kind").value == "d"
    with pytest.raises(ValueError, match="Missing finite target measurement"):
        await anext(stream)
