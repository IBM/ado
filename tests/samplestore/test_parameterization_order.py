# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Tests that the order of an experiment parameterization does not affect memoization."""

from collections.abc import Callable

import numpy.random

from ado.core.discoveryspace.samplers import (
    ExplicitEntitySpaceGridSampleGenerator,
    WalkModeEnum,
)
from ado.core.samplestore.sql import SQLSampleStore
from ado.modules.actuators.registry import ActuatorRegistry
from ado.schema.experiment import Experiment
from ado.schema.measurementspace import MeasurementSpace
from ado.schema.observed_property import ObservedPropertyValue
from ado.schema.property_value import ConstitutivePropertyValue
from ado.schema.reference import ExperimentReference
from ado.schema.result import ValidMeasurementResult


def test_measurements_found_for_reversed_parameterization_after_store_round_trip(
    random_sql_sample_store: Callable[[], SQLSampleStore],
    global_registry: ActuatorRegistry,
    mock_parameterizable_experiment: Experiment,
    customParameterization: list[ConstitutivePropertyValue],
) -> None:
    """Results stored for one parameterization order are found using the reverse order."""

    measured_ref = ExperimentReference(
        actuatorIdentifier=mock_parameterizable_experiment.actuatorIdentifier,
        experimentIdentifier=mock_parameterizable_experiment.identifier,
        parameterization=customParameterization,
    )
    requested_ref = ExperimentReference(
        actuatorIdentifier=mock_parameterizable_experiment.actuatorIdentifier,
        experimentIdentifier=mock_parameterizable_experiment.identifier,
        parameterization=list(reversed(customParameterization)),
    )

    measurement_space = MeasurementSpace.measurementSpaceFromExperimentReferences(
        [measured_ref]
    )
    entity = next(
        iter(
            ExplicitEntitySpaceGridSampleGenerator(
                mode=WalkModeEnum.RANDOM
            ).entitySpaceIterator(measurement_space.compatibleEntitySpace())
        )
    )[0]

    experiment = measurement_space.experimentForReference(measured_ref)
    result = ValidMeasurementResult(
        entityIdentifier=entity.identifier,
        measurements=[
            ObservedPropertyValue(
                property=op, value=numpy.random.default_rng().random()
            )
            for op in experiment.observedProperties
        ],
    )
    entity.add_measurement_result(result)

    store = random_sql_sample_store()
    store.addEntities([entity])
    store.add_measurement_results([result], skip_relationship_to_request=True)

    stored_entity = store.get_entities(
        identifiers=entity.identifier, require_measurements=True
    )[0]

    assert stored_entity.measurement_results_for_experiment_reference(requested_ref)
    assert stored_entity.observedPropertiesFromExperimentReference(
        requested_ref
    ) == stored_entity.observedPropertiesFromExperimentReference(measured_ref)
