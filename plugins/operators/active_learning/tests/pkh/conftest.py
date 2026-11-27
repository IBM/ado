# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Finite discovery spaces for PKH selection and publication tests."""

from collections.abc import Callable
from pathlib import Path

import pytest

from ado.core.discoveryspace.space import DiscoverySpace
from ado.core.samplestore.sql import SQLSampleStore
from ado.metastore.project import ProjectContext
from ado.schema.domain import PropertyDomain
from ado.schema.entity import Entity
from ado.schema.entityspace import EntitySpaceRepresentation
from ado.schema.experiment import Experiment
from ado.schema.measurementspace import MeasurementSpace, MeasurementSpaceConfiguration
from ado.schema.observed_property import ObservedProperty, ObservedPropertyValue
from ado.schema.property import ConcretePropertyDescriptor, ConstitutiveProperty
from ado.schema.request import MeasurementRequest, MeasurementRequestStateEnum
from ado.schema.result import ValidMeasurementResult
from ado.utilities.location import SQLiteStoreConfiguration


@pytest.fixture
def make_pkh_space(
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
def categorical_space(
    make_pkh_space: Callable[[list[ConstitutiveProperty]], DiscoverySpace],
) -> DiscoverySpace:
    """Five distinct coordinates, four labeled by the selected experiment."""
    codes = [1, "1", b"code", 2**53 + 1, None]
    prop = ConstitutiveProperty(
        identifier="code", propertyDomain=PropertyDomain(values=codes)
    )
    space = make_pkh_space([prop])
    entities = [space.entitySpace.entity_for_point({"code": value}) for value in codes]
    # Generated identifiers for 1 and "1" coincide.
    entities[0] = entities[0].model_copy(update={"identifier": "stored-number"})
    space.sample_store.addEntities(entities)
    experiment = space.measurementSpace.experiments[0]
    target, _, alternate = experiment.observedProperties
    other_lab = experiment.model_copy(update={"actuatorIdentifier": "other_lab"})
    observations = [
        (entities[0], target, 0),
        (entities[1], target, 2**53 + 1),
        (entities[2], target, 2),
        (entities[3], target, 3),
        (entities[0], alternate, 10),
        (entities[4], other_lab.observedProperties[-1], 999),
    ]
    space.sample_store.add_measurement_results(
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


@pytest.fixture
def record_pkh_measurement() -> Callable[
    [DiscoverySpace, Entity, str, ObservedProperty, int], None
]:
    """Store a measurement with its originating operation in the SQLite catalog."""

    def record(
        space: DiscoverySpace,
        entity: Entity,
        operation_id: str,
        target: ObservedProperty,
        value: int,
    ) -> None:
        """Write one successful result for the specified operation and property."""
        space.addMeasurement(
            MeasurementRequest(
                operation_id=operation_id,
                requestIndex=0,
                experimentReference=target.experimentReference,
                entities=[entity],
                status=MeasurementRequestStateEnum.SUCCESS,
                measurements=(
                    ValidMeasurementResult(
                        entityIdentifier=entity.identifier,
                        measurements=[
                            ObservedPropertyValue(property=target, value=value)
                        ],
                    ),
                ),
            )
        )

    return record
