# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT
from importlib.metadata import version

import pandas as pd
import pydantic

from orchestrator.core.discoveryspace.space import DiscoverySpace
from orchestrator.core.operation.config import FunctionOperationInfo
from orchestrator.core.operation.operation import OperationOutput
from orchestrator.modules.operators.collections import characterize_operation


class ProfileParameters(pydantic.BaseModel):
    """Parameters for the profile operator (no configurable options)."""


# See https://ibm.github.io/ado/operators/creating-operators/#ado-operator-functions
# for documentation on the decorator and its parameters
@characterize_operation(
    name="profile",
    version=version("ado-core"),
    configuration_model=ProfileParameters,
    configuration_model_default=ProfileParameters(),
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
