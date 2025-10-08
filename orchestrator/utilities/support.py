# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import logging
from typing import Any, NamedTuple

from orchestrator.schema.entity import Entity
from orchestrator.schema.experiment import Experiment, ParameterizedExperiment
from orchestrator.schema.measurementspace import MeasurementSpace
from orchestrator.schema.observed_property import ObservedPropertyValue
from orchestrator.schema.property_value import ValueTypeEnum
from orchestrator.schema.reference import ExperimentReference
from orchestrator.schema.request import MeasurementRequest, MeasurementRequestStateEnum
from orchestrator.schema.result import (
    InvalidMeasurementResult,
    MeasurementResult,
    ValidMeasurementResult,
)

logger = logging.getLogger(__name__)


# convert a dictionary of measurements to AD measurements
def dict_to_measurements(
    results: dict[str, Any],
    experiment: Experiment | ParameterizedExperiment,
) -> list[ObservedPropertyValue]:
    """
    Extracts the results for experiment from a dictionary of  measurements (property id:value pairs) and returns as PropertyValues
    :param results: dictionary of observation results
    :param experiment: Experiment definition. Only properties in result that are listed by the experiment are returned
    :return: A list of PropertyValue instances
    """
    measured_values = []
    # Get observed properties
    observed = experiment.observedProperties
    for op in observed:
        # for every observed property
        target = op.targetProperty.identifier
        # get measured value
        value = results.get(target)
        if value is None:
            # default non-measured property
            value = -1
        # Set the type
        if isinstance(value, str):
            value_type = ValueTypeEnum.STRING_VALUE_TYPE
        elif isinstance(value, bytes):
            value_type = ValueTypeEnum.BLOB_VALUE_TYPE
        elif isinstance(value, list):
            value_type = ValueTypeEnum.VECTOR_VALUE_TYPE
        else:
            value_type = ValueTypeEnum.NUMERIC_VALUE_TYPE
        # build property value
        property_value = ObservedPropertyValue(
            value=value,
            property=op,
            valueType=value_type,
        )
        measured_values.append(property_value)
    return measured_values


# Create measurement result
def create_measurement_result(
    identifier: str,
    measurements: list[ObservedPropertyValue],
    reference: ExperimentReference,
    error: str | None = None,
) -> MeasurementResult:
    """
    Creating measurement result
    :param identifier: entity identifier
    :param measurements: measurements
    :param error: execution error
    :param reference: experiment reference
    :return: Measurements result
    """
    if error is None:
        # No errors
        return ValidMeasurementResult(
            entityIdentifier=identifier, measurements=measurements
        )
    # error
    return InvalidMeasurementResult(
        entityIdentifier=identifier, reason=error, experimentReference=reference
    )


# Compute execution status
def compute_measurement_status(
    measurements: list[MeasurementResult],
) -> MeasurementRequestStateEnum:
    """
    Compute execution status
    :param measurements: list of measurements
    :return: status
    """
    if len(measurements) <= 0:
        # No measurements collected
        return MeasurementRequestStateEnum.FAILED
    failed = True
    for m in measurements:
        # for every collected measurement
        if isinstance(m, ValidMeasurementResult):
            failed = False
            break
    if failed:
        return MeasurementRequestStateEnum.FAILED
    return MeasurementRequestStateEnum.SUCCESS


# Get values from entity
def get_experiment_input_values(
    experiment: Experiment, entity: Entity
) -> dict[str, Any]:
    """
    Get values from entity
    :param experiment: experiment
    :param entity: entity
    :return: parameters
    """
    return experiment.propertyValuesFromEntity(entity)


class DependentExperimentInput(NamedTuple):
    experimentReference: ExperimentReference  # The dependent experiment to run
    entities: list[Entity]  # The entities to run on
    requestIndex: int  # The request index to use for the dependnent experiment request


def prepare_dependent_experiment_input(
    measurement_request: MeasurementRequest, measurement_space: MeasurementSpace
) -> list[DependentExperimentInput]:
    """Given a request and a measurementspace prepares the inputs for any dependent experiments that can be run

    In particular this function ensures that the Entities being passed to a dependent experiment only
    contain one instance of the required observed property value -> the one that comes from measurementRequest

    This is to avoid the case where measurementRequest has value A for the required property, but the entity
    already has N other values of A.
    The expectation is that the dependent experiment triggered by the measurementRequest should use value A
    just measured and not any of the other N values.

    Returns: A list of DependentExperimentInput NamedTuples:
    """

    logging.getLogger("prepare_dependent_experiment_input").debug(
        f"Checking if dependent experiments can be calculated based on result of {measurement_request}"
    )

    experiment_entity_map = (
        measurement_space.dependentExperimentsThatCanBeAppliedAfterMeasurementRequest(
            measurement_request
        )
    )
    prepared_inputs = []
    for experiment in experiment_entity_map:
        # Important: All measurements for same entity must have same request index

        # TODO: The following should be done in the custom experiments actuator
        # e.g. by a parameter to submit that indicates the input result to use in case of dependent experiments
        #
        # We need to prepare the entity so it only has the measurements of the required property from THIS REQUEST
        # i.e. it doesn't have measurements of that property made by other requests
        # This ensures the dependent experiment only sees one value for each of the observed properties
        # it depends on
        prepared_entities = []
        for entity in experiment_entity_map[experiment]:
            prepared_entity = Entity(
                constitutive_property_values=entity.constitutive_property_values,
                generatorid=entity.generatorid,
                identifier=entity.identifier,
            )
            prepared_entity.add_measurement_result(
                measurement_request.measurement_for_entity(entity.identifier)
            )
            # We need to add all the other results as these may be needed by memoization downstream
            # For example, if the dependent experiment has already been run, if we don't do this
            # that result will be removed, and it will be run again.
            for measurement_result in entity.measurement_results:
                if (
                    measurement_result.experimentReference
                    != measurement_request.experimentReference
                ):
                    prepared_entity.add_measurement_result(measurement_result)

            prepared_entities.append(prepared_entity)

        prepared_inputs.append(
            DependentExperimentInput(
                experimentReference=experiment.reference,
                entities=prepared_entities,
                requestIndex=measurement_request.requestIndex,
            )
        )

    return prepared_inputs
