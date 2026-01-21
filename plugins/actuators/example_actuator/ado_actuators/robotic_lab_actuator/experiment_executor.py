# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT
import json
import typing

import numpy as np
import ray

from orchestrator.modules.actuators.measurement_queue import MeasurementQueue
from orchestrator.schema.experiment import Experiment, ParameterizedExperiment
from orchestrator.schema.observed_property import ObservedPropertyValue
from orchestrator.schema.request import MeasurementRequest, MeasurementRequestStateEnum
from orchestrator.schema.result import ValidMeasurementResult
from orchestrator.utilities.support import get_experiment_input_values


# For initial testing you can place your experiment logic in this function
# kwargs will be a dictionary with whose keys are the required and optional input parameters of your experiment defined in experiment.yaml
# the values will be the values to use for those parameters
def my_experiment(**kwargs: typing.Any) -> dict[str, typing.Any]:  # noqa: ANN401

    # Put your logic here
    # IMPORTANT: The keys of this dict must match the identifiers given for the outputs of the experiment in experiments.yaml
    # Here we just put some random values for testing purposes
    rng = np.random.default_rng()
    return {
        "adsorption_timeseries": [rng.random() for i in range(10)],
        "adsorption_plateau_value": rng.random(),
    }


@ray.remote
def run_experiment(
    request: MeasurementRequest,
    experiment: Experiment | ParameterizedExperiment,
    measurement_queue: MeasurementQueue,
) -> None:

    # This function
    # 1. Performs the measurement represented by MeasurementRequest
    # 2. Updates MeasurementRequest with the results of the measurement and status
    # 3. Puts it in the measurement_queue

    measurements = []
    for entity in request.entities:

        #
        # Retrieve the input parameters to run your experiment on the entity
        #
        input_parameters = get_experiment_input_values(
            experiment=experiment, entity=entity
        )
        # You can implement the logic of your experiment inside the my_experiment function above
        # Feel free to change the name
        #
        # NOTE: This simple case assumes there is only ONE experiment
        # If you want to support multiple experiments you would have to e.g. call different functions based on the passed experiment
        measured_values = my_experiment(**input_parameters)

        # Augment the values returned by my_experiment to the structure used by ado
        measuredValues = [
            ObservedPropertyValue(
                value=identifier,
                property=experiment.observedPropertyForTargetIdentifier(identifier),
            )
            for identifier, value in measured_values.items()
        ]

        print(
            "Values for entity",
            entity.identifier,
            "and experiment",
            experiment.identifier,
            "experiment type is",
            type(experiment),
            "are",
            json.dumps(measured_values),
        )

        # Create a MeasurementResult to hold the results
        # This is used to
        # (a) Separate results from multiple entities
        # (b) Distinguish Valid and Invalid measurements -> especially in latter case to provide info on failure reasons

        # Here we use ValidMeasurementResult but if the experiment failed for some reason
        # We can use InvalidMeasurementResult and give a reason etc.
        measurements.append(
            ValidMeasurementResult(
                entityIdentifier=entity.identifier, measurements=measuredValues
            )
        )

    # For multi entity experiments if ONE entity had ValidResults the status must be SUCCESS
    request.status = MeasurementRequestStateEnum.SUCCESS
    request.measurements = measurements  # Note we don't set empty above as this would raise a validation error (can't have empty measurements)

    # Push the request to the state updates queue
    measurement_queue.put(request, block=False)
