# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import logging
import sys
import time
import typing

from ray.exceptions import RayTaskError

import orchestrator.utilities.output
from orchestrator.core import OperationResource
from orchestrator.core.discoveryspace.space import DiscoverySpace
from orchestrator.core.operation.config import (
    FunctionOperationInfo,
    OperatorFunctionConf,
    OperatorModuleConf,
)
from orchestrator.core.operation.operation import OperationException, OperationOutput
from orchestrator.core.operation.resource import (
    OperationExitStateEnum,
    OperationResourceEventEnum,
    OperationResourceStatus,
)
from orchestrator.modules.operators._cleanup import shutdown_signal_received
from orchestrator.modules.operators.base import (
    InterruptedOperationError,
    add_operation_output_to_metastore,
    create_operation_and_add_to_metastore,
)

# Global variable to track if graceful shutdown was called
moduleLog = logging.getLogger("orchestrate_core")


def log_space_details(discovery_space: "DiscoverySpace") -> None:

    from rich.console import Console

    console = Console()

    console.print("=========== Discovery Space ===========\n")
    console.print(discovery_space)


def _run_operation_harness(
    run_closure: typing.Callable[[], OperationOutput],
    discovery_space: DiscoverySpace,
    operator_module: OperatorModuleConf | OperatorFunctionConf,
    operation_parameters: dict,
    operation_info: FunctionOperationInfo,
    operation_identifier: str | None = None,
    finalize_callback: typing.Callable[[OperationResource], None] | None = None,
) -> OperationOutput:
    """Performs common orchestration for general and explore operations

    This function handles the common orchestration logic shared between general and explore
    operations. It creates the operation resource, executes the operation via the run_closure,
    handles exceptions, and stores the results.

    Params:
        run_closure: Callable that executes the operation and returns OperationOutput
        discovery_space: The discovery space the operation is running on
        operator_module: Configuration for the operator (either module or function-based)
        operation_parameters: Dictionary of parameters for the operation
        operation_info: Information about the operation including metadata and actuator configs
        operation_identifier: Optional pre-existing identifier for the operation resource
        finalize_callback: Optional callback to execute on the operation resource after
            completion, before final status update

    Returns:
        OperationOutput containing the results and status of the operation

    Raises:
        OperationException: If there is an error during the operation execution
    """

    #
    # OPERATION RESOURCE
    # Create and add OperationResource to metastore
    #

    operation_resource = create_operation_and_add_to_metastore(
        discovery_space=discovery_space,
        operator_module=operator_module,
        operation_parameters=operation_parameters,
        metastore=discovery_space.metadataStore,
        operation_info=operation_info,
        operation_identifier=operation_identifier,
    )

    #
    # START THE OPERATION
    #

    print(
        f"\n=========== Starting Operation {operation_resource.identifier} ===========\n"
    )

    operation_output = None

    interrupted_nested_operation: str | None = None
    operationStatus = OperationResourceStatus(
        event=OperationResourceEventEnum.FINISHED,
        exit_state=OperationExitStateEnum.ERROR,
        message="Operation exited due uncaught exception)",
    )
    try:
        operation_resource.status.append(
            OperationResourceStatus(event=OperationResourceEventEnum.STARTED)
        )
        discovery_space.metadataStore.updateResource(operation_resource)
        operation_output: OperationOutput | None = run_closure()
    except InterruptedOperationError as error:
        # This will occur if a nested operation caught SIGINT first.
        sys.stdout.flush()
        moduleLog.warning(
            f"Caught interrupt from nested operation {error.operation_identifier} "
            f"during operation {operation_resource.identifier}."
        )

        operationStatus = OperationResourceStatus(
            event=OperationResourceEventEnum.FINISHED,
            exit_state=OperationExitStateEnum.ERROR,
            message="Operation exited due to SIGINT propagated from nested operation",
        )

        # Record the identifier of the interrupted nested operation
        interrupted_nested_operation = error.operation_identifier
        if error.resources:
            # Create an OperationOutput to hold the resources created before interrupt
            operation_output = OperationOutput(
                operation=operation_resource,
                resources=error.resources,
                exitStatus=operationStatus,
            )

        raise InterruptedOperationError(operation_resource.identifier) from error
    except KeyboardInterrupt as error:
        sys.stdout.flush()
        moduleLog.warning(
            f"Caught keyboard interrupt during operation {operation_resource.identifier} - initiating graceful shutdown"
        )
        operationStatus = OperationResourceStatus(
            event=OperationResourceEventEnum.FINISHED,
            exit_state=OperationExitStateEnum.ERROR,
            message="Operation exited due to SIGINT",
        )
        raise InterruptedOperationError(operation_resource.identifier) from error
    except RayTaskError as error:
        sys.stdout.flush()
        e = error.as_instanceof_cause()
        operationStatus = OperationResourceStatus(
            event=OperationResourceEventEnum.FINISHED,
            exit_state=OperationExitStateEnum.ERROR,
            message=f"Operation exited due to the following error from a Ray Task: {e}.",
        )
        raise OperationException(
            message=f"Error raised while executing operation {operation_resource.identifier}",
            operation=operation_resource,
        ) from e
    except BaseException as error:
        import traceback

        sys.stdout.flush()
        operationStatus = OperationResourceStatus(
            event=OperationResourceEventEnum.FINISHED,
            exit_state=OperationExitStateEnum.ERROR,
            message=f"Operation exited due to the following error: {error}.\n\n"
            f"{''.join(traceback.format_exception(error))}",
        )
        raise OperationException(
            message=f"Error raised while executing operation {operation_resource.identifier}",
            operation=operation_resource,
        ) from error
    else:
        time.sleep(1)
        sys.stdout.flush()
        if shutdown_signal_received:
            moduleLog.warning(
                f"Operation {operation_identifier} exited normally but an external event e.g. SIGTERM, has already initiated shutdown"
            )
            if operation_output:
                moduleLog.info("Operation returned output - will save")

            operationStatus = (
                OperationResourceStatus(
                    event=OperationResourceEventEnum.FINISHED,
                    exit_state=OperationExitStateEnum.ERROR,
                    message="An external event e.g. SIGTERM, initiated shutdown. "
                    "This may have caused the operation to exit early",
                ),
            )
        else:
            if not operation_output:
                moduleLog.info(
                    "No output or exit status returned - setting an exit status to SUCCESS"
                )
                operationStatus = OperationResourceStatus(
                    event=OperationResourceEventEnum.FINISHED,
                    exit_state=OperationExitStateEnum.SUCCESS,
                )
            else:
                moduleLog.debug(
                    f"Operation {operation_identifier} exited normally with status {operation_output.exitStatus}"
                )
    finally:
        if operation_output:
            # Add the operation resource if not present
            if not operation_output.operation:
                operation_output.operation = operation_resource

            # Add it to metastore
            moduleLog.info(
                f"Adding output for operation {operation_identifier} to metastore"
            )
            add_operation_output_to_metastore(
                operation=operation_resource,
                output=operation_output,
                metastore=discovery_space.metadataStore,
            )
        else:
            # Create an output instance with a status
            # This is for returning, and so we have status to store below
            operation_output = OperationOutput(
                operation=operation_resource, exitStatus=operationStatus
            )

        # Add the final status to the operation resource
        moduleLog.info(
            f"Sending final status for operation {operation_identifier} to metastore"
        )
        operation_resource.status.append(operation_output.exitStatus)

        if not shutdown_signal_received and finalize_callback:
            finalize_callback(operation_resource)

        discovery_space.metadataStore.updateResource(operation_resource)

        # Establish relationships with interrupted nested operations
        if interrupted_nested_operation:
            try:
                discovery_space.metadataStore.addRelationship(
                    subjectIdentifier=operation_resource.identifier,
                    objectIdentifier=interrupted_nested_operation,
                )
            except Exception as e:
                moduleLog.warning(
                    f"Failed to establish relationship with nested operation "
                    f"{interrupted_nested_operation}: {e}"
                )

        print("=========== Operation Details ============\n")
        print(f"Space ID: {operation_resource.config.spaces[0]}")
        print(f"Sample Store ID:  {discovery_space.sample_store.identifier}")
        print(
            f"Operation:\n "
            f"{orchestrator.utilities.output.pydantic_model_as_yaml(operation_resource, exclude_none=True)}"
        )

    return operation_output
