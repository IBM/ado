# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import logging
import signal
import typing

import pydantic

import orchestrator.core
import orchestrator.modules
import orchestrator.modules.operators._cleanup
from orchestrator.core.discoveryspace.space import DiscoverySpace
from orchestrator.core.operation.config import (
    FunctionOperationInfo,
    OperatorFunctionConf,
    get_actuator_configurations,
    validate_actuator_configurations_against_space_configuration,
)
from orchestrator.core.operation.operation import OperationOutput
from orchestrator.modules.operators._cleanup import (
    graceful_operation_shutdown,
    graceful_operation_shutdown_handler,
)
from orchestrator.modules.operators._orchestrate_core import (
    _run_operation_harness,
    log_space_details,
)

moduleLog = logging.getLogger("general_orchestration")


def run_general_operation_core_closure(
    operation_function: typing.Callable[
        [
            DiscoverySpace,
            FunctionOperationInfo,
            ...,
        ],
        OperationOutput,
    ],
    discovery_space: DiscoverySpace,
    operationInfo: FunctionOperationInfo,
    operation_parameters: dict,
):

    def _run_general_operation_core() -> OperationOutput:
        return operation_function(
            discovery_space, operationInfo, **operation_parameters
        )  # type: OperationOutput

    return _run_general_operation_core


def orchestrate_general_operation(
    operator_function: typing.Callable[
        [
            DiscoverySpace,
            FunctionOperationInfo,
            ...,
        ],
        OperationOutput,
    ],
    operation_parameters: dict,
    parameters_model: type[pydantic.BaseModel] | None,
    discovery_space: DiscoverySpace,
    operation_info: FunctionOperationInfo,
    operation_type: orchestrator.core.operation.config.DiscoveryOperationEnum,
) -> OperationOutput:
    """Orchestrates a general operation (non-explore)

    * Checks params and space
    * creates OperationResource and adds to metastore
    * updates OperationResource with status updates,
    * stores any OperationOutput
    * insert graceful shutdown handler for keyboard interrupts
    * catches exceptions from the operation and handles them

    Used for all Operation types except Explore which requires a different setup

    Exceptions:
        ValueError: if the MeasurementSpace is not consistent with EntitySpace
        pydantic.ValidationError: if the operation parameters are not valid
        OperationException: If there is an error during the operation
    """

    import uuid

    if not operation_info.namespace:
        operation_info.namespace = (
            f"{operator_function.__name__}-namespace-{str(uuid.uuid4())[:8]}"
        )

    operator_module = OperatorFunctionConf(
        operatorName=operator_function.__name__,
        operationType=operation_type,
    )

    if parameters_model:
        parameters_model.model_validate(operation_parameters)

    # Check the space
    if not discovery_space.measurementSpace.isConsistent:
        moduleLog.critical("Measurement space is inconsistent - aborting")
        raise ValueError("Measurement space is inconsistent")

    log_space_details(discovery_space)

    # Validate the actuator configurations given
    # before calling the operation
    actuator_configurations = get_actuator_configurations(
        actuator_configuration_identifiers=operation_info.actuatorConfigurationIdentifiers,
        project_context=discovery_space.project_context,
    )
    validate_actuator_configurations_against_space_configuration(
        actuator_configurations=actuator_configurations,
        discovery_space_configuration=discovery_space.config,
    )

    operation_run_closure = run_general_operation_core_closure(
        operator_function,
        discovery_space=discovery_space,
        operationInfo=operation_info,
        operation_parameters=operation_parameters,
    )

    orchestrator.modules.operators._cleanup.shutdown = False

    signal.signal(
        signalnum=signal.SIGTERM, handler=graceful_operation_shutdown_handler()
    )

    output = _run_operation_harness(
        run_closure=operation_run_closure,
        discovery_space=discovery_space,
        operator_module=operator_module,
        operation_parameters=operation_parameters,
        operation_info=operation_info,
    )

    graceful_operation_shutdown()

    return output
