# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import logging
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
        OperationOutput | None,
    ],
    discovery_space: DiscoverySpace,
    operationInfo: FunctionOperationInfo,
    operation_parameters: dict,
) -> typing.Callable[[], OperationOutput | None]:

    def _run_general_operation_core() -> OperationOutput | None:
        return operation_function(
            discovery_space, operationInfo, **operation_parameters
        )

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

    This function handles the orchestration of non-explore operations (characterize, compare,
    modify, fuse, learn, etc.). It performs the following:
    - Validates operation parameters if a parameters model is provided
    - Checks measurement space consistency
    - Validates actuator configurations against the space
    - Inserts graceful shutdown handler for keyboard interrupts

    It calls run_operation_harness to create, store, and update the operation resource,
    execute the operation, handle exceptions, and stores the operation results.

    Params:
        operator_function: The function that implements the operation. Must accept
            DiscoverySpace and FunctionOperationInfo as first two arguments, followed
            by operation-specific parameters
        operation_parameters: Dictionary of parameters to pass to the operator function
        parameters_model: Optional Pydantic model to validate operation_parameters against
        discovery_space: The discovery space to operate on
        operation_info: Information about the operation including metadata, actuator
            configuration identifiers, and namespace
        operation_type: The type of operation being executed

    Returns:
        OperationOutput containing the results and status of the operation

    Raises:
        ValueError: If the MeasurementSpace is not consistent with EntitySpace or if
            actuator configurations are invalid
        pydantic.ValidationError: If the operation parameters are not valid
        OperationException: If there is an error during the operation
        ResourceDoesNotExistError: If an actuator configuration cannot be retrieved from the database

    """

    import uuid

    # Note on signals: Since there is no specific cleanup logic
    # for general operations it makes no difference
    # if a signal handler for SIGTERM is in place or not

    if not operation_info.ray_namespace:
        operation_info.ray_namespace = (
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

    return _run_operation_harness(
        run_closure=operation_run_closure,
        discovery_space=discovery_space,
        operator_module=operator_module,
        operation_parameters=operation_parameters,
        operation_info=operation_info,
    )
