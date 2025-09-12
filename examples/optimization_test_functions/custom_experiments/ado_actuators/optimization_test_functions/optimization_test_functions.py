# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import logging

from nevergrad.functions import ArtificialFunction

from orchestrator.schema.entity import Entity
from orchestrator.schema.experiment import Experiment
from orchestrator.schema.observed_property import ObservedPropertyValue
from orchestrator.schema.property_value import ValueTypeEnum

moduleLog = logging.getLogger()


def artificial_function(
    entity: Entity, experiment: Experiment, parameters: dict | None
):

    import numpy as np

    # parameters is a dictionary of key:value pairs of the experiment required/optional inputs
    # defined in custom_experiments.yaml
    parameters = experiment.propertyValuesFromEntity(entity)

    # Get the function from nevergrad.functions.ArtificialFunction
    func = ArtificialFunction(
        name=parameters["name"],
        num_blocks=parameters["num_blocks"],
        block_dimension=int(
            len(entity.constitutive_property_values) / parameters["num_blocks"]
        ),
    )

    # Call the nevergrad function
    value = func(np.asarray([v.value for v in entity.constitutive_property_values]))

    # Return the function value to ado
    pv = ObservedPropertyValue(
        value=value,
        property=experiment.observedPropertyForTargetIdentifier("function_value"),
        valueType=ValueTypeEnum.NUMERIC_VALUE_TYPE,
    )
    return [pv]
