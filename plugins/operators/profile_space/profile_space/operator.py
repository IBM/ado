# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT


import pandas as pd

from orchestrator.core.discoveryspace.space import DiscoverySpace
from orchestrator.core.operation.config import FunctionOperationInfo
from orchestrator.core.operation.operation import OperationOutput
from orchestrator.modules.operators.collections import characterize_operation


# See https://ibm.github.io/ado/operators/creating-operators/#ado-operator-functions
# for documentation on the decorator and its parameters
@characterize_operation(
    name="profile",
    configuration_model=None,  # You can use this field to define the option of your operator if any - see https://ibm.github.io/ado/operators/creating-operators/#describing-your-operation-input-parameters
    configuration_model_default=None,  # Use this field to provide default/example values for your operator
    description="Returns a ydata_profiling ProfileReport for the space",
)
# operator function can have any name but have similar parameters - see https://ibm.github.io/ado/operators/creating-operators/#operator-function-parameters
def profile(
    discoverySpace: DiscoverySpace,
    operationInfo: FunctionOperationInfo | None = None,
    **kwargs: dict,
) -> OperationOutput:
    import ydata_profiling

    if discoverySpace.sample_store.numberOfEntities == 0:
        raise ValueError(
            f"The discovery space {discoverySpace.uri} contains no entities"
        )
    # The potential issue here is if some entities have one value for a property and others have more
    # we will get two columns - one for those requiring the average and one for the others.
    df = pd.DataFrame(
        data=[
            e.seriesRepresentation()
            for e in discoverySpace.sample_store.entities
            if len(e.observedPropertyValues) > 0
        ]
    )

    # See https://ibm.github.io/ado/operators/creating-operators/#returning-data-from-your-operation
    # for documentation on the return value of an operator function
    return OperationOutput(other=[ydata_profiling.ProfileReport(df)])
