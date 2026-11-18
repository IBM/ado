# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Regression tests for PKH target selection and persisted predictors."""

import asyncio
import pickle
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from active_learning.pkh.data import (
    candidate_entities,
    target_label,
    validate_target_output,
)
from active_learning.pkh.operator import _finalize_model, pkh
from active_learning.pkh.parameters import (
    ActiveLearningAutoGluonParameters,
    FinalPredictorMetadata,
    PKHParameters,
)
from active_learning.pkh.sampler import PKHSampleSelector

from ado.core.discoveryspace.space import DiscoverySpace
from ado.core.operation.operation import OperationOutput, OperationResource
from ado.core.operation.resource import (
    OperationExitStateEnum,
    OperationResourceEventEnum,
    OperationResourceStatus,
)
from ado.core.samplestore.sql import SQLSampleStore
from ado.metastore.project import ProjectContext
from ado.modules.operators.collections import explore
from ado.schema.domain import PropertyDomain, VariableTypeEnum
from ado.schema.entityspace import EntitySpaceRepresentation
from ado.schema.experiment import Experiment
from ado.schema.measurementspace import MeasurementSpace, MeasurementSpaceConfiguration
from ado.schema.observed_property import ObservedPropertyValue
from ado.schema.property import ConcretePropertyDescriptor, ConstitutiveProperty
from ado.schema.result import ValidMeasurementResult
from ado.utilities.location import SQLiteStoreConfiguration


@pytest.fixture
def categorical_space(tmp_path: Path) -> DiscoverySpace:
    """Create a cached space with externally written, experiment-specific labels."""
    feature = ConstitutiveProperty(
        identifier="kind",
        propertyDomain=PropertyDomain(
            variableType=VariableTypeEnum.DISCRETE_VARIABLE_TYPE,
            values=[1, "b", "c", "d"],
        ),
    )
    count = ConstitutiveProperty(
        identifier="count",
        propertyDomain=PropertyDomain(values=[0, 1]),
    )
    entity_space = EntitySpaceRepresentation(constitutiveProperties=[feature, count])
    entities = [
        entity_space.entity_for_point({"kind": value, "count": number})
        for value in feature.propertyDomain.domain_values
        for number in [0, 1]
    ]
    experiment = Experiment(
        identifier="selected",
        version="1.0.0",
        actuatorIdentifier="test",
        requiredProperties=(feature, count),
        targetProperties=[ConcretePropertyDescriptor(identifier="response")],
    )
    measurement_space = MeasurementSpace(
        configuration=MeasurementSpaceConfiguration(experiments=[experiment])
    )
    other_observed = measurement_space.observedProperties[0].model_copy(
        update={
            "experimentReference": experiment.reference.model_copy(
                update={"experimentIdentifier": "other"}
            )
        }
    )
    location = SQLiteStoreConfiguration(path=str(tmp_path / "pkh.db"), database="pkh")
    sample_store = SQLSampleStore(
        identifier="labels", storageLocation=location, parameters={}
    )
    sample_store.addEntities(entities)
    sample_store.get_entities(require_measurements=True)
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
                [measurement_space.observedProperties[0]] * 6 + [other_observed],
                [0, 1] * 3 + [999],
                strict=True,
            )
        ],
        skip_relationship_to_request=True,
    )
    return DiscoverySpace(
        identifier="categorical",
        entitySpace=entity_space,
        measurementSpace=measurement_space,
        sample_store=sample_store,
        project_context=ProjectContext(project="pkh", metadataStore=location),
    )


def test_target_resolution(
    categorical_space: DiscoverySpace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolve the target and reject zero or multiple experiments before sampling."""
    observed = categorical_space.measurementSpace.observedProperties[0]
    for identifier in (observed.identifier, observed.targetProperty.identifier):
        assert validate_target_output(identifier, categorical_space) == observed
    calls, _ = _stub_random_walk(monkeypatch, OperationExitStateEnum.SUCCESS)
    parameters = PKHParameters(
        targetOutput=observed.identifier,
        numberEntities=1,
        outputDirectory=str(tmp_path / "rejected"),
    )
    experiment = categorical_space.measurementSpace.experiments[0]
    for experiments in (
        [],
        [experiment, experiment.model_copy(update={"identifier": "other"})],
    ):
        space = DiscoverySpace(
            entitySpace=categorical_space.entitySpace if experiments else None,
            measurementSpace=MeasurementSpace(
                configuration=MeasurementSpaceConfiguration(experiments=experiments)
            ),
            sample_store=categorical_space.sample_store,
            project_context=categorical_space.project_context,
        )
        with pytest.raises(ValueError, match="exactly 1 experiment"):
            pkh.__wrapped__(space, parameters=parameters)
    assert calls == []
    assert not (tmp_path / "rejected_finalized").exists()


@pytest.mark.parametrize("backend", ["random_forest", "autogluon"])
def test_finalizer_refreshes_labels_and_saves_categorical_predictor(
    categorical_space: DiscoverySpace,
    tmp_path: Path,
    backend: str,
) -> None:
    """Export fresh labels for the selected experiment and reusable categorical predictions."""
    if backend == "autogluon":
        pytest.importorskip("autogluon.tabular")
    parameters = PKHParameters(
        targetOutput=categorical_space.measurementSpace.observedProperties[
            0
        ].identifier,
        numberEntities=6,
        nEstimators=3,
        outputDirectory=str(tmp_path / "model"),
        finalPredictor=backend,
        autoGluon=(
            ActiveLearningAutoGluonParameters(
                timeLimit=10, hyperparameters={"RF": {"n_estimators": 3}}
            )
            if backend == "autogluon"
            else None
        ),
    )
    final_directory = tmp_path / "model_finalized"
    final_directory.mkdir()
    sampled_identifiers = {
        categorical_space.entitySpace.entity_for_point(
            {"kind": kind, "count": count}
        ).identifier
        for kind in (1, "b", "c")
        for count in (0, 1)
    }
    final_directory, discrepancy = _finalize_model(
        categorical_space, parameters, final_directory, sampled_identifiers
    )
    metadata = FinalPredictorMetadata.model_validate_json(
        (final_directory / "metadata.json").read_text()
    )
    assert metadata.backend == backend
    assert metadata.measuredCount == 6
    assert metadata.targetOutput == "response"
    assert metadata.categoricalColumns == ["kind"]
    assert metadata.mmdSquared == discrepancy
    assert discrepancy is not None
    assert discrepancy >= 0
    assert FinalPredictorMetadata.model_validate(metadata.model_dump()) == metadata
    predictions = pd.read_csv(final_directory / "combined_predictions.csv")
    assert predictions.loc[~predictions.is_predicted, "response"].tolist() == [0, 1] * 3
    unmeasured = predictions.loc[predictions.is_predicted]
    assert unmeasured.kind.tolist() == ["d", "d"]
    if backend == "random_forest":
        with (final_directory / "model.pkl").open("rb") as model_file:
            model = pickle.load(model_file)  # noqa: S301 - reload this test's model
    else:
        from autogluon.tabular import TabularPredictor

        model = TabularPredictor.load(str(final_directory / "autogluon"))
        assert model.problem_type == "regression"
    assert model.predict(unmeasured[["kind", "count"]]).tolist() == pytest.approx(
        unmeasured.response.tolist()
    )
    typed_query = pd.DataFrame({"kind": [1, "1"], "count": [0, 0]})
    if backend == "random_forest":
        encoded = model["features"].transform(typed_query)
        assert not (encoded[0] == encoded[1]).all()
    else:
        encoded = model.transform_features(typed_query)["kind"]
        assert pd.notna(encoded.iloc[0])
        assert pd.isna(encoded.iloc[1])


def test_finalizer_requires_labels_for_all_sampled_entities(
    categorical_space: DiscoverySpace,
    tmp_path: Path,
) -> None:
    """Reject a failed final measurement even when earlier labels are available."""
    parameters = PKHParameters(
        targetOutput=categorical_space.measurementSpace.observedProperties[
            0
        ].identifier,
        numberEntities=1,
        outputDirectory=str(tmp_path / "model"),
    )
    final_directory = tmp_path / "model_finalized"
    final_directory.mkdir()
    missing_entity = categorical_space.entitySpace.entity_for_point(
        {"kind": "d", "count": 1}
    )
    with pytest.raises(ValueError, match="Missing finite target measurements"):
        _finalize_model(
            categorical_space, parameters, final_directory, {missing_entity.identifier}
        )
    with pytest.raises(ValueError, match="budget is incomplete"):
        _finalize_model(categorical_space, parameters, final_directory, set())
    assert list(final_directory.iterdir()) == []


def _stub_random_walk(
    monkeypatch: pytest.MonkeyPatch, exit_state: OperationExitStateEnum
) -> tuple[list[dict], OperationResource]:
    """Replace random_walk with a stub; return its recorded calls and operation."""
    operation = OperationResource.model_construct(identifier="operation-random_walk-1")
    calls: list[dict] = []

    def random_walk(
        _space: DiscoverySpace, _info: object, *, parameters: object
    ) -> OperationOutput:
        calls.append(parameters.model_dump())
        return OperationOutput(
            operation=operation,
            exitStatus=OperationResourceStatus(
                event=OperationResourceEventEnum.FINISHED, exit_state=exit_state
            ),
        )

    monkeypatch.setitem(
        explore.operators, "random_walk", SimpleNamespace(function=random_walk)
    )
    return calls, operation


@pytest.mark.parametrize(
    ("exit_state", "finalized"),
    [(OperationExitStateEnum.SUCCESS, True), (OperationExitStateEnum.FAIL, False)],
)
def test_pkh_wraps_random_walk_and_finalizes_on_success(
    categorical_space: DiscoverySpace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exit_state: OperationExitStateEnum,
    finalized: bool,  # noqa: FBT001
) -> None:
    """Drive random_walk with the PKH sampler and only finalize successful runs."""
    calls, operation = _stub_random_walk(monkeypatch, exit_state)
    observed = categorical_space.measurementSpace.observedProperties[0]
    categorical_space.sample_store.add_measurement_results(
        [
            ValidMeasurementResult(
                entityIdentifier=categorical_space.entitySpace.entity_for_point(
                    {"kind": "d", "count": count}
                ).identifier,
                measurements=[ObservedPropertyValue(property=observed, value=5)],
            )
            for count in (0, 1)
        ],
        skip_relationship_to_request=True,
    )
    parameters = PKHParameters(
        targetOutput=observed.identifier,
        numberEntities=2,
        nEstimators=3,
        outputDirectory=str(tmp_path / "model"),
    )
    output = pkh.__wrapped__(
        categorical_space,
        None,
        parameters=parameters,
    )
    assert len(calls) == 1
    assert calls[0]["batchSize"] == 1
    assert calls[0]["samplerConfig"]["parameters"]["targetOutput"] == (
        observed.identifier
    )
    assert output.resources == [operation]
    assert output.exitStatus.exit_state == exit_state
    assert ("pkhMmdSquared" in output.metadata) is finalized
    final_directory = tmp_path / "model_finalized"
    assert final_directory.exists() is finalized
    if finalized:
        metadata_file = final_directory / "metadata.json"
        previous_metadata = metadata_file.read_bytes()
    else:
        final_directory.mkdir()
    with pytest.raises(FileExistsError, match="model_finalized"):
        pkh.__wrapped__(categorical_space, None, parameters=parameters)
    assert len(calls) == 1
    assert final_directory.is_dir()
    if finalized:
        assert metadata_file.read_bytes() == previous_metadata


def test_sampler_rejects_unsupported_iteration() -> None:
    """Require single-entity batches and asynchronous iteration."""
    sampler = PKHSampleSelector(
        PKHParameters(targetOutput="response", numberEntities=1)
    )
    with pytest.raises(NotImplementedError):
        next(sampler.entityIterator(None))
    with pytest.raises(ValueError, match="batchsize=1"):
        asyncio.run(sampler.remoteEntityIterator(None, batchsize=2))


@pytest.mark.asyncio
async def test_sampler_stream_yields_unmeasured_entities_once(
    categorical_space: DiscoverySpace,
) -> None:
    """Yield each unlabelled entity once, resuming as its measurement lands."""
    observed = categorical_space.measurementSpace.observedProperties[0]

    async def get_space() -> DiscoverySpace:
        return categorical_space

    remote = SimpleNamespace(discoverySpace=SimpleNamespace(remote=get_space))
    sampler = PKHSampleSelector(
        PKHParameters(
            targetOutput=observed.identifier,
            numberEntities=2,
            nEstimators=3,
        )
    )
    stream = await sampler.remoteEntityIterator(remote)
    await anext(stream)
    with pytest.raises(ValueError, match="Missing finite target measurement"):
        await anext(stream)

    seen = []
    async for batch in await sampler.remoteEntityIterator(remote):
        assert len(batch) == 1
        seen.append(batch[0])
        categorical_space.sample_store.add_measurement_results(
            [
                ValidMeasurementResult(
                    entityIdentifier=batch[0].identifier,
                    measurements=[ObservedPropertyValue(property=observed, value=5)],
                )
            ],
            skip_relationship_to_request=True,
        )
    assert len({entity.identifier for entity in seen}) == 2
    assert {entity.constitutive_property_values[0].value for entity in seen} == {"d"}


def test_property_identity_and_export_names(categorical_space: DiscoverySpace) -> None:
    """Keep observed labels separate from inputs and require unambiguous output columns."""
    observed = categorical_space.measurementSpace.observedProperties[0]
    alias = ConstitutiveProperty(
        identifier=observed.identifier, propertyDomain=PropertyDomain(values=[99])
    )
    entity = EntitySpaceRepresentation(constitutiveProperties=[alias]).entity_for_point(
        {observed.identifier: 99}
    )
    assert target_label(entity, observed.identifier) is None
    for target_name, feature_name in [
        ("kind", "kind"),
        ("entityIdentifier", "kind"),
        ("is_predicted", "kind"),
        ("response", "entityIdentifier"),
        ("response", "is_predicted"),
        ("response", "mixed"),
    ]:
        feature = ConstitutiveProperty(
            identifier=feature_name,
            propertyDomain=PropertyDomain(
                values=[1, "1"] if feature_name == "mixed" else ["a"]
            ),
        )
        experiment = categorical_space.measurementSpace.experiments[0].model_copy(
            update={
                "requiredProperties": (feature,),
                "targetProperties": [
                    ConcretePropertyDescriptor(identifier=target_name)
                ],
            }
        )
        space = DiscoverySpace(
            entitySpace=EntitySpaceRepresentation(constitutiveProperties=[feature]),
            measurementSpace=MeasurementSpace(
                configuration=MeasurementSpaceConfiguration(experiments=[experiment])
            ),
            sample_store=categorical_space.sample_store,
            project_context=categorical_space.project_context,
        )
        if feature_name == "mixed":
            with pytest.raises(ValueError, match="unique entity identifiers"):
                candidate_entities(space)
        else:
            with pytest.raises(ValueError, match="column"):
                validate_target_output(target_name, space)
