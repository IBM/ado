# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT


import pytest

from ado.schema.entity import Entity
from ado.schema.experiment import Experiment
from ado.schema.observed_property import ObservedPropertyValue
from ado.schema.property_value import ConstitutivePropertyValue
from ado.schema.reference import ExperimentReference
from ado.schema.result import InvalidMeasurementResult, ValidMeasurementResult


@pytest.fixture
def valid_measurement_result(
    property_values: list[ObservedPropertyValue | ConstitutivePropertyValue],
    entity: Entity,
) -> ValidMeasurementResult:

    return ValidMeasurementResult(
        entityIdentifier=entity.identifier, measurements=property_values
    )


@pytest.fixture
def invalid_measurement_result(
    property_values: list[ObservedPropertyValue | ConstitutivePropertyValue],
    entity: Entity,
) -> InvalidMeasurementResult:

    return InvalidMeasurementResult(
        entityIdentifier=entity.identifier,
        reason="Insufficient memory",
        experimentReference=ExperimentReference(
            experimentIdentifier="testexp", actuatorIdentifier="testact"
        ),
    )


@pytest.fixture
def valid_measurement_result_and_entity(
    entity_for_parameterized_experiment: tuple[Entity, Experiment],
) -> [Entity, ValidMeasurementResult]:

    import numpy as np

    test_entity, exp = entity_for_parameterized_experiment
    ref = exp.reference
    assert not test_entity.observedPropertiesFromExperimentReference(ref)

    # Add a result
    values = [
        ObservedPropertyValue(value=np.random.default_rng().random(), property=op)
        for op in exp.observedProperties
    ]

    return test_entity, ValidMeasurementResult(
        entityIdentifier=test_entity.identifier,
        measurements=values,
    )
