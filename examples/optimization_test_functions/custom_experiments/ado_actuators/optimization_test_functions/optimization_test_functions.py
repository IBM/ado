# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import logging

from nevergrad.functions import ArtificialFunction

from orchestrator.modules.actuators.custom_experiments import custom_experiment
from orchestrator.schema.domain import PropertyDomain
from orchestrator.schema.property import ConstitutiveProperty

moduleLog = logging.getLogger()


@custom_experiment(
    [
        ConstitutiveProperty(
            identifier="x0",
            propertyDomain=PropertyDomain(variableType="CONTINUOUS_VARIABLE_TYPE"),
        ),
        ConstitutiveProperty(
            identifier="x1",
            propertyDomain=PropertyDomain(variableType="CONTINUOUS_VARIABLE_TYPE"),
        ),
        ConstitutiveProperty(
            identifier="x2",
            propertyDomain=PropertyDomain(variableType="CONTINUOUS_VARIABLE_TYPE"),
        ),
        ConstitutiveProperty(
            identifier="name",
            propertyDomain=PropertyDomain(
                values=["discus", "sphere", "cigar", "griewank", "rosenbrock", "st1"]
            ),
        ),
        ConstitutiveProperty(
            identifier="num_blocks",
            propertyDomain=PropertyDomain(
                domainRange=[1, 10], variableType="DISCRETE_VARIABLE_TYPE", interval=1
            ),
        ),
    ]
)
def artificial_function(x0: float, x1: float, x2: float, name: str, num_blocks: int):

    import numpy as np

    # Get the function from nevergrad.functions.ArtificialFunction
    func = ArtificialFunction(
        name=name,
        num_blocks=num_blocks,
        block_dimension=int(3 / num_blocks),
        translation_factor=0.0,
    )

    # Call the nevergrad function
    value = func(np.asarray([x0, x1, x2]))

    return {"function_value": value}
