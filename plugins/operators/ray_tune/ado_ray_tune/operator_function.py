# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT


import orchestrator.core
import orchestrator.modules.module
from orchestrator.core.discoveryspace.space import DiscoverySpace
from orchestrator.core.operation.config import FunctionOperationInfo
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
)
def ray_tune(
    discoverySpace: DiscoverySpace,
    operationInfo: FunctionOperationInfo | None = None,
    **kwargs: dict,
) -> OperationOutput:
    """
    Performs a random_walk operation on a given discoverySpace

    """
    if operationInfo is None:
        operationInfo = FunctionOperationInfo()

    module = orchestrator.core.operation.config.OperatorModuleConf(
        moduleName="ado_ray_tune.operator",
        moduleClass="RayTune",
        moduleType=orchestrator.modules.module.ModuleTypeEnum.OPERATION,
    )

    # validate parameters
    RayTuneConfiguration.model_validate(kwargs)

    return orchestrate_explore_operation(
        discovery_space=discoverySpace,
        operator_module=module,
        parameters=kwargs,
        operation_info=operationInfo,
    )
