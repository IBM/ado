# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Finite discovery spaces for active-learning selection and publication tests."""

from collections.abc import Callable
from pathlib import Path

import pytest

from ado.core.discoveryspace.space import DiscoverySpace
from ado.core.samplestore.sql import SQLSampleStore
from ado.metastore.project import ProjectContext
from ado.schema.domain import PropertyDomain, VariableTypeEnum
from ado.schema.entityspace import EntitySpaceRepresentation
from ado.schema.experiment import Experiment
from ado.schema.measurementspace import MeasurementSpace, MeasurementSpaceConfiguration
from ado.schema.observed_property import ObservedPropertyValue
from ado.schema.property import ConcretePropertyDescriptor, ConstitutiveProperty
from ado.schema.result import ValidMeasurementResult
from ado.utilities.location import SQLiteStoreConfiguration


@pytest.fixture
def make_test_space(
    tmp_path: Path,
) -> Callable[[list[ConstitutiveProperty]], DiscoverySpace]:
    """Build finite spaces over one SQLite store per test."""
    location = SQLiteStoreConfiguration(path=str(tmp_path / "pkh.db"), database="pkh")
    store = SQLSampleStore(identifier="labels", storageLocation=location, parameters={})

    def create(properties: list[ConstitutiveProperty]) -> DiscoverySpace:
        """Use one experiment with a reserved name and overlapping target names."""
        experiment = Experiment(
            identifier="selected",
            actuatorIdentifier="test",
            requiredProperties=tuple(properties),
            targetProperties=[
                ConcretePropertyDescriptor(identifier=name)
                for name in ("self", "is_predicted", "selected-self")
            ],
        )
        return DiscoverySpace(
            entitySpace=EntitySpaceRepresentation(constitutiveProperties=properties),
            measurementSpace=MeasurementSpace(
                MeasurementSpaceConfiguration(experiments=[experiment])
            ),
            sample_store=store,
            project_context=ProjectContext(project="pkh", metadataStore=location),
        )

    return create


@pytest.fixture
def empty_space(
    make_test_space: Callable[[list[ConstitutiveProperty]], DiscoverySpace],
) -> DiscoverySpace:
    """Create a one-candidate space for checks that precede measurement."""
    kind = ConstitutiveProperty(
        identifier="kind", propertyDomain=PropertyDomain(values=["b"])
    )
    return make_test_space([kind])


@pytest.fixture
def categorical_space(
    request: pytest.FixtureRequest,
    make_test_space: Callable[[list[ConstitutiveProperty]], DiscoverySpace],
) -> DiscoverySpace:
    """Label two kinds and leave two unmeasured, including a distractor observation."""
    kinds = [1, "b", "c", "d"]
    codes, labels = getattr(
        request, "param", ([0.5, 2**53, 2**53 + 1, None, b"code"], [0, 1, 2, 10, 20])
    )
    space = make_test_space(
        [
            ConstitutiveProperty(
                identifier="kind", propertyDomain=PropertyDomain(values=kinds)
            ),
            ConstitutiveProperty(
                identifier="count",
                propertyDomain=PropertyDomain(
                    values=codes,
                    variableType=VariableTypeEnum.CATEGORICAL_VARIABLE_TYPE,
                ),
            ),
        ]
    )
    entities = [
        space.entitySpace.entity_for_point({"kind": kind, "count": code})
        for kind in kinds
        for code in codes
    ]
    space.sample_store.addEntities(entities)
    # Prime the cache before another writer adds the target measurements.
    space.sample_store.get_entities(require_measurements=True)
    writer = SQLSampleStore(
        identifier=space.sample_store.identifier,
        storageLocation=space.project_context.metadataStore,
        parameters={},
    )
    experiment = space.measurementSpace.experiments[0]
    target, _, alternate = experiment.observedProperties
    other_target = experiment.model_copy(
        update={"identifier": "other"}
    ).observedProperties[0]
    observations = [
        (entity, target, value)
        for entity, value in zip(entities[: 2 * len(codes)], labels * 2, strict=True)
    ]
    observations.extend(
        [(entities[2 * len(codes)], other_target, 999), (entities[0], alternate, 10)]
    )
    writer.add_measurement_results(
        [
            ValidMeasurementResult(
                entityIdentifier=entity.identifier,
                measurements=[ObservedPropertyValue(property=observed, value=value)],
            )
            for entity, observed, value in observations
        ],
        skip_relationship_to_request=True,
    )
    return space
