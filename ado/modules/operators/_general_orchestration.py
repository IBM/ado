# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import inspect
import logging
import typing

from ado.core.discoveryspace.space import DiscoverySpace
from ado.core.operation.config import (
    FunctionOperationInfo,
    GenericOperatorParameters,
    OperatorMetadata,
    get_actuator_configurations,
    validate_actuator_configurations_against_space_configuration,
)
from ado.core.operation.inputs import OperatorInputType
from ado.core.operation.operation import OperationOutput
from ado.metastore.sqlstore import SQLStore
from ado.modules.operators._orchestrate_core import (
    _run_operation_harness,
    log_space_details,
)
from ado.modules.operators.base import OperatorFunction
from ado.modules.operators.provenance import operation_provenance

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
    inputs: dict[str, OperatorInputType],
    operationInfo: FunctionOperationInfo,
    operation_parameters: GenericOperatorParameters,
) -> typing.Callable[[], OperationOutput | None]:
    """Return a closure that calls the operator implementation with resource inputs.

    Args:
        operation_function: The callable stored on the operator metadata.
        inputs: Mapping of parameter name to rich resource, passed as keyword
            arguments.
        operationInfo: Runtime operation context.
        operation_parameters: Validated instance of the operator's configuration model.

    Returns:
        A zero-argument callable that invokes the operator implementation.
    """

    def _run_general_operation_core() -> OperationOutput | None:
        implementation = _operator_callable_for_harness(operation_function)
        return implementation(
            **inputs,
            operationInfo=operationInfo,
            parameters=operation_parameters,
        )

    return _run_general_operation_core


def orchestrate_general_operation(
    operator_metadata: OperatorMetadata,
    operation_parameters: GenericOperatorParameters,
    inputs: dict[str, OperatorInputType],
    operation_info: FunctionOperationInfo,
    metastore: SQLStore,
) -> OperationOutput:
    """Orchestrate a general (non-explore) operation.

    Validates parameters, checks each discovery-space input, then runs the harness.

    Args:
        operator_metadata: Registered metadata for the operator.
        operation_parameters: Validated configuration model.
        inputs: Mapping of parameter name to rich resource.
        operation_info: Operation metadata, including project context.
        metastore: Metastore that contains every input.

    Returns:
        OperationOutput containing the results and status of the operation.

    Raises:
        ValueError: If a measurement space is inconsistent or actuator
            configurations are invalid.
        pydantic.ValidationError: If the operation parameters are not valid.
        OperationException: If there is an error during the operation.
        ResourceDoesNotExistError: If an actuator configuration cannot be retrieved.
    """

    import uuid

    spaces = [value for value in inputs.values() if isinstance(value, DiscoverySpace)]

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

    if spaces:
        actuator_configurations = get_actuator_configurations(
            actuator_configuration_identifiers=operation_info.actuatorConfigurationIdentifiers,
            metastore=metastore,
        )
        for space in spaces:
            if not space.measurementSpace.isConsistent:
                moduleLog.critical("Measurement space is inconsistent - aborting")
                raise ValueError("Measurement space is inconsistent")

            log_space_details(space)

            validate_actuator_configurations_against_space_configuration(
                actuator_configurations=actuator_configurations,
                discovery_space_configuration=space.config,
            )

    operation_run_closure = run_general_operation_core_closure(
        operator_function,
        inputs=inputs,
        operationInfo=operation_info,
        operation_parameters=operation_parameters,
    )

    return _run_operation_harness(
        run_closure=operation_run_closure,
        inputs=inputs,
        operator_metadata=operator_metadata,
        operation_parameters=operation_parameters,
        operation_info=operation_info,
        metastore=metastore,
        provenance=operation_provenance(operator_metadata.reference),
    )
