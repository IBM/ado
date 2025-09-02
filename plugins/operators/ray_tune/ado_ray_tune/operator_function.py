# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT


import orchestrator.core
import orchestrator.modules.module
from ado_ray_tune.config import RayTuneConfiguration
from ado_ray_tune.operator import RayTune
from orchestrator.core.discoveryspace.space import DiscoverySpace
from orchestrator.core.operation.config import FunctionOperationInfo
from orchestrator.core.operation.operation import OperationOutput
from orchestrator.modules.operators.collections import explore_operation
from orchestrator.modules.operators.orchestrate import (
    explore_operation_function_wrapper,
)


@explore_operation(
    name="ray_tune",
    description=RayTune.description(),
    configuration_model=RayTuneConfiguration,
    configuration_model_default=RayTune.defaultOperationParameters(),
)
def ray_tune(
    discoverySpace: DiscoverySpace,
    operationInfo: FunctionOperationInfo = FunctionOperationInfo(),
    **kwargs: dict,
) -> OperationOutput:
    """
    Performs a random_walk operation on a given discoverySpace

    """

    import uuid

    module = orchestrator.core.operation.config.OperatorModuleConf(
        moduleName="ado_ray_tune.operator",
        moduleClass="RayTune",
        moduleType=orchestrator.modules.module.ModuleTypeEnum.OPERATION,
    )

    # validate parameters
    RayTuneConfiguration.model_validate(kwargs)

    return explore_operation_function_wrapper(
        discovery_space=discoverySpace,
        module=module,
        parameters=kwargs,
        namespace=f"namespace-{str(uuid.uuid4())[:8]}",
        operation_info=operationInfo,
    )
