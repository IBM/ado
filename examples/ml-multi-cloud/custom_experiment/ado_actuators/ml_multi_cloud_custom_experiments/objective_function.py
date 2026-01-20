# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import logging

from orchestrator.schema.entity import Entity
from orchestrator.schema.experiment import Experiment
from orchestrator.schema.observed_property import ObservedPropertyValue

moduleLog = logging.getLogger()

# Mapping cpu family to dollar per  node/hour
# THESE VALUES ARE JUST FOR ILLUSTRATION

cost_map = {0.0: 10, 1.0: 20}


# For illustration purposes we implement the core objective function here
# NOTE: The parameter names must match the required and optional properties defined in custom_experiments.yaml
def objective_function(
    cpu_family: int, nodes: int, wallClockRuntime: float
) -> dict[str, float]:

    # You can calculate multiple property values
    return {
        "total_cost": cost_map[cpu_family] * nodes * (wallClockRuntime / (60.0 * 60))
    }


# This is the wrapper function called by ado
def cost(
    entity: Entity,
    experiment: Experiment,
    parameters: dict | None = None,
) -> list[ObservedPropertyValue]:
    """

    :param entity: The entity to be measured
    :param experiment: The Experiment object representing the exact Experiment to perform
        Required as multiple experiments can measure this property
    :param parameters:
    :return:
    """

    moduleLog.debug(f"Measuring {entity} with {experiment}")

    # Get a dict of experiment parameter (name:value) pairs
    # target=True means we will have a key wallClockRuntime rather than benchmarkPerformance-wallClockRuntime
    parameters = experiment.propertyValuesFromEntity(entity, target=True)

    # Call the custom objective function
    results = objective_function(**parameters)

    # Augment the results with data required by ado
    return [
        ObservedPropertyValue(
            value=value,
            property=experiment.observedPropertyForTargetIdentifier(
                targetIdentifier=identifier
            ),
        )
        for identifier, value in results.items()
    ]
