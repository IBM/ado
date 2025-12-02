# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import logging
import typing

from orchestrator.modules.actuators.custom_experiments import custom_experiment
from orchestrator.schema.domain import PropertyDomain, VariableTypeEnum
from orchestrator.schema.property import (
    ConstitutiveProperty,
)

moduleLog = logging.getLogger()


# Decorate the python function with @custom_experiment
# This tells ado
# - The domains of all your parameters i.e. what are valid values they can take
# - The name of the output variable
# - If you use keyword args -> Can extract optional parameters and parameterization
@custom_experiment(
    required_properties=[
        ConstitutiveProperty(
            identifier="x0",
            propertyDomain=PropertyDomain(
                variableType=VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE
            ),
        ),
        ConstitutiveProperty(
            identifier="x1",
            propertyDomain=PropertyDomain(
                variableType=VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE
            ),
        ),
        ConstitutiveProperty(
            identifier="x2",
            propertyDomain=PropertyDomain(
                variableType=VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE
            ),
        ),
    ],
    optional_properties=[
        ConstitutiveProperty(
            identifier="num_blocks",
            propertyDomain=PropertyDomain(
                variableType=VariableTypeEnum.DISCRETE_VARIABLE_TYPE,
                domainRange=[1, 10],
                interval=1,
            ),
        ),
        ConstitutiveProperty(
            identifier="name",
            propertyDomain=PropertyDomain(
                variableType=VariableTypeEnum.CATEGORICAL_VARIABLE_TYPE,
                values=["discus", "sphere", "cigar", "griewank", "rosenbrock", "st1"],
            ),
        ),
    ],
    parameterization={"num_blocks": 1, "name": "rosenbrock"},
    output_property_identifiers=["function_value"],
)
def nevergrad_opt_3d_test_func(
    x0: float, x1: float, x2: float, name: str, num_blocks: int
) -> dict[str, typing.Any]:

    import numpy as np
    from nevergrad.functions import ArtificialFunction

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
