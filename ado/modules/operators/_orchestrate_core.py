# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import logging
import sys
import time
import typing

import ray
from ray.exceptions import RayTaskError

import ado.utilities.output
from ado.core import OperationResource
from ado.core.discoveryspace.space import DiscoverySpace
from ado.core.operation.config import (
    FunctionOperationInfo,
    OperatorMetadata,
    OperatorReference,
)
from ado.core.operation.operation import OperationException, OperationOutput
from ado.core.operation.resource import (
    OperationExitStateEnum,
    OperationResourceEventEnum,
    OperationResourceStatus,
)
from ado.modules.operators import _cleanup
from ado.modules.operators.base import (
    InterruptedOperationError,
    add_operation_output_to_metastore,
    create_operation_and_add_to_metastore,
)

# Global variable to track if graceful shutdown was called
moduleLog = logging.getLogger("orchestrate_core")


def _operation_status_for_sigterm_initiated_shutdown(
    *, underlying_error: BaseException | None = None
) -> OperationResourceStatus:
    """Return a FINISHED/error status for SIGTERM-initiated shutdown.

    Args:
        underlying_error: Optional underlying exception to append to the message.

    Returns:
        OperationResourceStatus with the SIGTERM shutdown message.
    """
    message = (
        "An external event e.g. SIGTERM, initiated shutdown. "
        "This may have caused the operation to exit early"
    )
    if underlying_error is not None:
        message = f"{message}. Underlying Ray error: {underlying_error}"
    return OperationResourceStatus(
        event=OperationResourceEventEnum.FINISHED,
        exit_state=OperationExitStateEnum.ERROR,
        message=message,
    )


def log_space_details(discovery_space: "DiscoverySpace") -> None:
    from rich.console import Console

    console = Console()

    console.print("=========== Discovery Space ===========\n")
    console.print(discovery_space)


def _run_operation_harness(
    run_closure: typing.Callable[[], OperationOutput],
    discovery_space: DiscoverySpace,
    operator_metadata: OperatorMetadata,
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
        operator_metadata: Metadata for the registered operator.
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

    operator_reference = OperatorReference(
        operatorName=operator_metadata.name,
        operationType=operator_metadata.type,
        operatorVersion=operator_metadata.version,
    )
    operation_resource = create_operation_and_add_to_metastore(
        discovery_space=discovery_space,
        operator_module=operator_reference,
        operation_parameters=operation_parameters,
        metastore=discovery_space.metadataStore,
        operation_info=operation_info,
        operation_identifier=operation_identifier,
    )

    if ray.is_initialized():
        try:
            ray_job_id = ray.get_runtime_context().get_job_id()
            operation_resource.metadata["ray_job_id"] = ray_job_id
        except Exception:
            moduleLog.debug("Could not retrieve Ray job ID", exc_info=True)
            moduleLog.warning(
                "Could not retrieve Ray job ID - operation will proceed without it"
            )
    else:
        moduleLog.debug("Operation is not running as a Ray job - no ray_job_id set")

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
        message="Operation exited due to uncaught exception)",
    )
    sigterm_status_callback_key = f"{operation_resource.identifier}_sigterm_status"
    sigterm_status_was_recorded = False

    # This updates operation status on SIGTERM
    # in cases where the finally: block is not executed
    def record_sigterm_shutdown_status() -> None:
        nonlocal sigterm_status_was_recorded
        sigterm_status = _operation_status_for_sigterm_initiated_shutdown()
        operation_resource.status.append(sigterm_status)
        discovery_space.metadataStore.updateResource(operation_resource)
        sigterm_status_was_recorded = True
        moduleLog.debug(
            f"Recorded SIGTERM shutdown status for {operation_resource.identifier}"
        )

    _cleanup.cleanup_callback_functions[sigterm_status_callback_key] = (
        record_sigterm_shutdown_status
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
        # This is a fallback in case the SIGTERM callback above failed
        if _cleanup.shutdown_signal_received:
            operationStatus = _operation_status_for_sigterm_initiated_shutdown(
                underlying_error=e
            )
        else:
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
        # This is a fallback in case the SIGTERM callback above failed
        if _cleanup.shutdown_signal_received:
            operationStatus = _operation_status_for_sigterm_initiated_shutdown()
        else:
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
        # This is a fallback in case the SIGTERM callback above failed
        if _cleanup.shutdown_signal_received:
            moduleLog.warning(
                f"Operation {operation_resource.identifier} exited normally but an external event e.g. SIGTERM, has already initiated shutdown"
            )
            if operation_output:
                moduleLog.info("Operation returned output - will save")

            operationStatus = _operation_status_for_sigterm_initiated_shutdown()
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
                    f"Operation {operation_resource.identifier} exited normally with status {operation_output.exitStatus}"
                )
    finally:
        _cleanup.cleanup_callback_functions.pop(sigterm_status_callback_key, None)
        if operation_output:
            # Add the operation resource if not present
            if not operation_output.operation:
                operation_output.operation = operation_resource

            # Add it to metastore
            moduleLog.info(
                f"Adding output for operation {operation_resource.identifier} to metastore"
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

        # If no signal OR sigterm status was not recorded update status
        # The "not sigterm_status_was_recorded" engages the fallback
        if not (_cleanup.shutdown_signal_received and sigterm_status_was_recorded):
            # Add the final status to the operation resource
            moduleLog.info(
                f"Sending final status for operation {operation_identifier} to metastore"
            )
            operation_resource.status.append(operation_output.exitStatus)

        if not _cleanup.shutdown_signal_received and finalize_callback:
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
            f"{ado.utilities.output.pydantic_model_as_yaml(operation_resource, exclude_none=True)}"
        )

    return operation_output
