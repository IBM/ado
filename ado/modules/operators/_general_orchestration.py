# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import inspect
import logging
import typing

from ado.core.discoveryspace.space import DiscoverySpace
from ado.core.operation.config import (
    FunctionOperationInfo,
    OperatorMetadata,
    get_actuator_configurations,
    validate_actuator_configurations_against_space_configuration,
)
from ado.core.operation.operation import OperationOutput
from ado.modules.operators._orchestrate_core import (
    _run_operation_harness,
    log_space_details,
)
from ado.modules.operators.base import OperatorFunction

moduleLog = logging.getLogger("general_orchestration")


def _operator_callable_for_harness(registered: OperatorFunction) -> OperatorFunction:
    """Resolve the callable to execute inside :func:`_run_operation_harness`.

    Operators registered via ``characterize_operation`` / ``modify_operation`` /
    ``export_operation`` store a *wrapper* in :class:`~ado.core.operation.config.OperatorMetadata`
    that delegates to :func:`orchestrate_general_operation`. The harness must run the
    underlying implementation (``functools.wraps`` sets ``__wrapped__``); otherwise
    ``run_closure`` re-invokes the wrapper and recurses without bound.

    Args:
        registered: The callable stored on the operator metadata (wrapper or not).

    Returns:
        The innermost unwrapped callable, or *registered* if there is no wrapper chain.
    """
    return typing.cast("OperatorFunction", inspect.unwrap(registered))


def run_general_operation_core_closure(
    operation_function: OperatorFunction,
    discovery_space: DiscoverySpace,
    operationInfo: FunctionOperationInfo,
    operation_parameters: dict,
) -> typing.Callable[[], OperationOutput | None]:

    def _run_general_operation_core() -> OperationOutput | None:
        implementation = _operator_callable_for_harness(operation_function)
        return implementation(discovery_space, operationInfo, **operation_parameters)

    return _run_general_operation_core


def orchestrate_general_operation(
    operator_metadata: OperatorMetadata,
    operation_parameters: dict,
    discovery_space: DiscoverySpace,
    operation_info: FunctionOperationInfo,
) -> OperationOutput:
    """Orchestrates a general operation (non-explore)

    This function handles the orchestration of non-explore operations (characterize, compare,
    modify, fuse, learn, etc.). It performs the following:
    - Validates operation parameters against the configuration model
    - Checks measurement space consistency
    - Validates actuator configurations against the space
    - Inserts graceful shutdown handler for keyboard interrupts

    It calls run_operation_harness to create, store, and update the operation resource,
    execute the operation, handle exceptions, and stores the operation results.

    Params:
        operator_metadata: Registered metadata for the operator, carrying the callable,
            configuration model, operation type, and name.
        operation_parameters: Dictionary of parameters to pass to the operator function
        discovery_space: The discovery space to operate on
        operation_info: Information about the operation including metadata, actuator
            configuration identifiers, and namespace

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

    if operator_metadata.function is None:
        raise ValueError(
            f"Operator '{operator_metadata.name}' has no function registered"
        )
    operator_function = typing.cast("OperatorFunction", operator_metadata.function)

    if not operation_info.ray_namespace:
        operation_info.ray_namespace = (
            f"{operator_metadata.name}-namespace-{str(uuid.uuid4())[:8]}"
        )

    operator_metadata.configuration_model.model_validate(operation_parameters)

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
        operator_metadata=operator_metadata,
        operation_parameters=operation_parameters,
        operation_info=operation_info,
    )
