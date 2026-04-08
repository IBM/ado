# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT
from importlib.metadata import version

from orchestrator.core.discoveryspace.space import DiscoverySpace
from orchestrator.core.operation.config import (
    DiscoveryOperationEnum,
    FunctionOperationInfo,
    OperatorFunctionConf,
)
from orchestrator.core.operation.operation import OperationOutput
from orchestrator.modules.operators.collections import explore_operation
from orchestrator.modules.operators.orchestrate import (
    orchestrate_explore_operation,
)

from .config import RayTuneConfiguration
from .operator import RayTune


@explore_operation(
    name="ray_tune",
    description=RayTune.description(),
    configuration_model=RayTuneConfiguration,
    configuration_model_default=RayTune.defaultOperationParameters(),
    version=version("ado-ray-tune"),
    operator_class=RayTune,
)
def ray_tune(
    discoverySpace: DiscoverySpace,
    operationInfo: FunctionOperationInfo | None = None,
    **kwargs: dict,
) -> OperationOutput:
    """Performs a ray_tune operation on a given discoverySpace."""
    return orchestrate_explore_operation(
        discovery_space=discoverySpace,
        operator_reference=OperatorFunctionConf(
            operationType=DiscoveryOperationEnum.SEARCH,
            operatorName="ray_tune",
        ),
        parameters=kwargs,
        operation_info=operationInfo or FunctionOperationInfo(),
    )
