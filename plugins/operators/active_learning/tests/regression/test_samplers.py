# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT
# ruff: noqa: S101

"""Shared tests for the regression-based finite-pool sample selectors.

Tests that exercise only one strategy's acquisition logic live next to that
strategy, in ``tests/regression/pkh/``. This file covers the machinery in
``active_learning.regression._shared`` and any behaviour that is
parametrized across strategies.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pydantic
import pytest
from active_learning.regression._shared import (
    _encode_features,
    _finite_pool,
    _FinitePool,
    _measured_labels,
    _resolve_target,
    _SequentialSelectionState,
)
from active_learning.regression.pkh.parameters import PKHParameters
from active_learning.regression.pkh.sampler import PKHSampleSelector, _PKHSelectionState

from ado.modules.operators.randomwalk import (
    CustomSamplerConfiguration,
    SamplerModuleConf,
)
from ado.schema.domain import PropertyDomain
from ado.schema.entity import Entity
from ado.schema.entityspace import EntitySpaceRepresentation
from ado.schema.experiment import Experiment
from ado.schema.measurementspace import (
    MeasurementSpace,
    MeasurementSpaceConfiguration,
)
from ado.schema.observed_property import ObservedProperty, ObservedPropertyValue
from ado.schema.property import AbstractPropertyDescriptor, ConstitutiveProperty
from ado.schema.result import ValidMeasurementResult


class _MutableSampleStore:
    """Small in-memory implementation of the public sample-store read API."""

    def __init__(self, entities: list[Entity]) -> None:
        """Index the supplied entities by identifier.

        Args:
            entities: Entities initially held by the test store.
        """

        self.entities = {
            entity.identifier: entity
            for entity in entities
            if entity.identifier is not None
        }
        self.reads: list[tuple[set[str] | None, bool, bool]] = []

    def get_entities(
        self,
        identifiers: str | set[str] | None = None,
        *,
        require_measurements: bool,
        refresh: bool = False,
    ) -> list[Entity]:
        """Return requested entities and record the public API arguments."""

        selected_identifiers = (
            {identifiers} if isinstance(identifiers, str) else identifiers
        )
        self.reads.append((selected_identifiers, require_measurements, refresh))
        if selected_identifiers is None:
            return list(self.entities.values())
        return [
            entity
            for identifier, entity in self.entities.items()
            if identifier in selected_identifiers
        ]


class _RemoteValue:
    """Asynchronous value accessor matching a Ray actor method."""

    def __init__(self, value: SimpleNamespace) -> None:
        """Retain the discovery-space value returned by ``remote``.

        Args:
            value: Discovery-space test double to return.
        """

        self.value = value

    async def remote(self) -> SimpleNamespace:
        """Return the retained value asynchronously."""

        return self.value


class _RemoteDiscoverySpace:
    """Manager exposing a remote discovery-space accessor."""

    def __init__(self, discovery_space: SimpleNamespace) -> None:
        """Expose a discovery space through a Ray-like accessor.

        Args:
            discovery_space: Discovery-space test double to expose.
        """

        self.discoverySpace = _RemoteValue(discovery_space)


def _measurement_space(*experiment_names: str) -> MeasurementSpace:
    """Create experiments that each measure the scalar test target.

    Args:
        *experiment_names: Identifiers for the experiments to create.

    Returns:
        A measurement space containing the requested experiments.
    """

    target = AbstractPropertyDescriptor(identifier="latency")
    experiments = [
        Experiment(
            actuatorIdentifier="test",
            identifier=name,
            targetProperties=[target],
        )
        for name in experiment_names
    ]
    return MeasurementSpace(
        configuration=MeasurementSpaceConfiguration(experiments=experiments)
    )


def _entity_with_label(
    entity: Entity,
    target_property: ObservedProperty,
    value: float,
) -> Entity:
    """Return a copy of an entity with one target measurement.

    Args:
        entity: Entity to label.
        target_property: Observed property recorded by the measurement.
        value: Scalar measurement value.

    Returns:
        A copy of ``entity`` containing the measurement.
    """

    result = ValidMeasurementResult(
        entityIdentifier=entity.identifier,
        measurements=[
            ObservedPropertyValue(property=target_property, value=value),
        ],
    )
    return entity.model_copy(update={"measurement_results": [result]})


def _adaptive_test_space() -> tuple[
    SimpleNamespace,
    _MutableSampleStore,
    ObservedProperty,
    list[Entity],
]:
    """Create a finite discovery space with one initially labelled entity.

    Returns:
        The discovery space, mutable store, target property, and finite entities.
    """

    measurement_space = _measurement_space("benchmark")
    target_property = measurement_space.observedProperties[0]
    entity_space = EntitySpaceRepresentation(
        constitutiveProperties=[
            ConstitutiveProperty(
                identifier="workers",
                propertyDomain=PropertyDomain(values=[1, 2, 3, 4, 5]),
            )
        ]
    )
    entities = [
        entity_space.entity_for_point({"workers": workers})
        for workers in [1, 2, 3, 4, 5]
    ]
    store = _MutableSampleStore([_entity_with_label(entities[0], target_property, 1.0)])
    discovery_space = SimpleNamespace(
        entitySpace=entity_space,
        measurementSpace=measurement_space,
        sample_store=store,
    )
    return discovery_space, store, target_property, entities


@pytest.mark.parametrize(
    ("parameters_class", "expected"),
    [
        (
            PKHParameters,
            {
                "targetOutput": "latency",
                "nEstimators": 100,
                "minSamplesLeaf": 2,
                "epochLength": 10,
                "seed": 42,
                "criterion": "squared_error",
                "nJobs": -1,
                "labelWaitTimeoutSeconds": 300.0,
                "labelPollIntervalSeconds": 0.1,
            },
        ),
    ],
    ids=["pkh"],
)
def test_parameter_defaults_round_trip(
    parameters_class: type[PKHParameters],
    expected: dict[str, object],
) -> None:
    """Verify defaults survive a model dump and validation round trip."""

    parameters = parameters_class(targetOutput="latency")

    assert parameters.model_dump() == expected
    assert parameters_class.model_validate(parameters.model_dump()) == parameters


@pytest.mark.parametrize("parameters_class", [PKHParameters])
def test_parameter_models_reject_invalid_options(
    parameters_class: type[PKHParameters],
) -> None:
    """Verify sampler models reject unknown options and zero workers."""

    with pytest.raises(pydantic.ValidationError):
        parameters_class(targetOutput="latency", unknownOption=True)
    with pytest.raises(pydantic.ValidationError):
        parameters_class(targetOutput="latency", nJobs=0)


@pytest.mark.parametrize(
    ("selector_class", "parameters_class"),
    [
        (PKHSampleSelector, PKHParameters),
    ],
    ids=["pkh"],
)
def test_selectors_expose_and_load_their_parameter_models(
    selector_class: type[PKHSampleSelector],
    parameters_class: type[PKHParameters],
) -> None:
    """Verify ADO loads each selector with its concrete parameter model."""

    configuration = CustomSamplerConfiguration(
        module=SamplerModuleConf(
            moduleName=selector_class.__module__,
            moduleClass=selector_class.__name__,
        ),
        parameters={"targetOutput": "latency", "nJobs": 1},
    )

    selector = configuration.sampler()
    restored = CustomSamplerConfiguration.model_validate(
        configuration.model_dump()
    ).sampler()

    assert selector_class.parameters_model() is parameters_class
    assert isinstance(selector, selector_class)
    assert isinstance(restored, selector_class)
    assert isinstance(restored.params, parameters_class)


def test_target_resolution_requires_one_observed_property() -> None:
    """Verify shared targets require a full observed-property identifier."""

    measurement_space = _measurement_space("fast", "accurate")

    with pytest.raises(ValueError, match="measured by multiple experiments"):
        _resolve_target(measurement_space, "latency")

    expected = measurement_space.observedProperties[0]
    assert _resolve_target(measurement_space, expected.identifier) == expected


def test_measured_labels_use_refreshable_sample_store_api() -> None:
    """Verify label reads use the refreshable public sample-store API."""

    measurement_space = _measurement_space("benchmark")
    target_property = measurement_space.observedProperties[0]
    entity = Entity(identifier="point-0", constitutive_property_values=())
    store = _MutableSampleStore([_entity_with_label(entity, target_property, 3.5)])
    discovery_space = SimpleNamespace(sample_store=store)
    pool = _FinitePool(
        features=np.array([[0.0]]),
        entities=(entity,),
        index_by_identifier={"point-0": 0},
    )

    labels = _measured_labels(discovery_space, pool, target_property)

    assert labels == {0: 3.5}
    assert store.reads == [({"point-0"}, True, True)]


def test_finite_pool_preserves_external_stored_entity_identifiers() -> None:
    """Verify pool construction preserves an externally assigned entity ID."""

    discovery_space, store, target_property, entities = _adaptive_test_space()
    external_entity = entities[0].model_copy(update={"identifier": "external-1"})
    store.entities = {
        "external-1": _entity_with_label(external_entity, target_property, 1.0)
    }

    pool = _finite_pool(discovery_space)
    labels = _measured_labels(discovery_space, pool, target_property)

    assert pool.entities[0].identifier == "external-1"
    assert pool.index_by_identifier["external-1"] == 0
    assert labels == {0: 1.0}


def test_public_iterator_refreshes_each_returned_label() -> None:
    """Verify the synchronous iterator waits for each requested label."""

    discovery_space, store, target_property, entities = _adaptive_test_space()
    selector = PKHSampleSelector(
        PKHParameters(
            targetOutput="latency",
            nEstimators=5,
            minSamplesLeaf=1,
            nJobs=1,
        )
    )
    iterator = selector.entityIterator(discovery_space)

    first = next(iterator)[0]
    assert first.identifier not in {entities[0].identifier, None}
    store.entities[first.identifier] = _entity_with_label(first, target_property, 2.0)
    second = next(iterator)[0]

    assert second.identifier not in {entities[0].identifier, first.identifier, None}
    assert store.reads[-1] == ({first.identifier}, True, True)


@pytest.mark.asyncio
async def test_remote_iterator_refreshes_each_returned_label() -> None:
    """Verify the remote iterator observes feedback before acquisition."""

    discovery_space, store, target_property, entities = _adaptive_test_space()
    selector = PKHSampleSelector(
        PKHParameters(
            targetOutput="latency",
            nEstimators=5,
            minSamplesLeaf=1,
            nJobs=1,
            labelPollIntervalSeconds=0.001,
        )
    )
    iterator = await selector.remoteEntityIterator(
        _RemoteDiscoverySpace(discovery_space)
    )

    first = (await anext(iterator))[0]
    assert first.identifier not in {entities[0].identifier, None}
    store.entities[first.identifier] = _entity_with_label(first, target_property, 2.0)
    second = (await anext(iterator))[0]

    assert second.identifier not in {entities[0].identifier, first.identifier, None}
    assert store.reads[-1] == ({first.identifier}, True, True)


def _complete_queries(
    state: _SequentialSelectionState,
    labels: np.ndarray,
    count: int,
) -> tuple[int, ...]:
    """Run adaptive query/update transitions and return selected indices.

    Args:
        state: Selection state to advance.
        labels: Complete target array used to reveal requested labels.
        count: Number of adaptive transitions to complete.

    Returns:
        Indices selected in acquisition order.
    """

    queried: list[int] = []
    for _ in range(count):
        selected_before = state.selected_indices
        index = state.query_one()
        assert state.pending_index == index
        assert state.selected_indices == selected_before

        state.update(index, float(labels[index]))
        assert state.pending_index is None
        assert state.selected_indices == (*selected_before, index)
        queried.append(index)
    return tuple(queried)


@pytest.mark.parametrize(
    ("state_class", "additional_arguments"),
    [
        (_PKHSelectionState, {"epoch_length": 3}),
    ],
    ids=["pkh"],
)
def test_unlabelled_pool_uses_one_deterministic_pilot_before_fitting(
    state_class: type[_PKHSelectionState],
    additional_arguments: dict[str, int],
) -> None:
    """Verify an unlabelled pool yields a deterministic pilot before fitting."""

    features = np.arange(8, dtype=float).reshape(-1, 1)
    labels = np.square(features[:, 0])
    state = state_class(
        features,
        np.array([], dtype=int),
        np.array([], dtype=float),
        n_estimators=5,
        min_samples_leaf=1,
        seed=5,
        n_jobs=1,
        **additional_arguments,
    )

    pilot = state.query_one()

    assert pilot == 0
    assert not state._forest_fitted
    state.update(pilot, labels[pilot])
    state.query_one()
    assert state._forest_fitted


@pytest.mark.parametrize(
    ("state_class", "additional_arguments"),
    [
        (_PKHSelectionState, {"epoch_length": 3}),
    ],
    ids=["pkh"],
)
def test_real_forest_selection_is_reproducible_and_unique(
    state_class: type[_PKHSelectionState],
    additional_arguments: dict[str, int],
) -> None:
    """Verify seeded adaptive selections are reproducible and duplicate-free."""

    axis = np.linspace(-2.0, 2.0, 24)
    features = np.column_stack((axis, axis**2, np.sin(2.0 * axis)))
    labels = axis**3 - 0.5 * axis + np.cos(axis)
    initial_indices = np.array([0, 4, 9, 14, 19, 23])
    common_arguments = {
        "n_estimators": 13,
        "min_samples_leaf": 1,
        "seed": 19,
        "criterion": "squared_error",
        "n_jobs": 1,
    }
    states = [
        state_class(
            features.copy(),
            initial_indices.copy(),
            labels[initial_indices].copy(),
            **common_arguments,
            **additional_arguments,
        )
        for _ in range(2)
    ]

    sequences = [_complete_queries(state, labels, count=6) for state in states]

    assert sequences[0] == sequences[1]
    assert len(set(sequences[0])) == len(sequences[0])
    assert set(sequences[0]).isdisjoint(initial_indices)


def test_encode_features_handles_categories_deterministically() -> None:
    """Verify categorical feature encoding is finite and deterministic."""

    frame = pd.DataFrame(
        {
            "workers": [1, 2, 3, 4],
            "accelerator": ["cpu", "gpu", None, "cpu"],
            "region": pd.Categorical(["eu", "eu", "us", "us"]),
        }
    )

    encoded = _encode_features(
        frame,
        categorical_columns=["accelerator", "region"],
    )

    assert encoded.ndim == 2
    assert encoded.shape[0] == len(frame)
    assert np.isfinite(encoded).all()
    assert len(np.unique(encoded, axis=0)) == len(frame)
    np.testing.assert_array_equal(
        encoded,
        _encode_features(frame, categorical_columns=["accelerator", "region"]),
    )
