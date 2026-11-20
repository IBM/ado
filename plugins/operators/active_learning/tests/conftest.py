# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Discovery-space fixtures for active-learning operators."""

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
def categorical_space(tmp_path: Path, request: pytest.FixtureRequest) -> DiscoverySpace:
    """Six measured points, with a seventh measured only by another experiment."""
    kind = ConstitutiveProperty(
        identifier="kind",
        propertyDomain=PropertyDomain(
            values=getattr(request, "param", [1, "b", "c", "d"]),
            variableType=VariableTypeEnum.CATEGORICAL_VARIABLE_TYPE,
        ),
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
