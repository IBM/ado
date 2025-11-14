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
from orchestrator.core.operation.config import BaseOperationRunConfiguration
from orchestrator.core.operation.operation import OperationException, OperationOutput
from orchestrator.core.operation.resource import (
    OperationExitStateEnum,
    OperationResourceEventEnum,
    OperationResourceStatus,
)
from orchestrator.modules.operators._cleanup import shutdown
from orchestrator.modules.operators.base import (
    add_operation_from_base_config_to_metastore,
    add_operation_output_to_metastore,
)

# Global variable to track if graceful shutdown was called
moduleLog = logging.getLogger("orchestrate_core")


def log_space_details(discovery_space: "DiscoverySpace"):

    from IPython.lib import pretty

    print("=========== Discovery Space ===========\n")
    print(pretty.pretty(discovery_space))
    numberEntities = discovery_space.sample_store.numberOfEntities
    if numberEntities > 0:
        e = discovery_space.sample_store.entities[0]

        print("Example entity (first retrieved from sample store):\n")
        print(
            orchestrator.utilities.output.pydantic_model_as_yaml(e, exclude_unset=True)
        )
        print("\n")


def _run_operation_harness(
    run_closure: typing.Callable[[], OperationOutput],
    base_operation_configuration: BaseOperationRunConfiguration,
    discovery_space: DiscoverySpace,
    operation_identifier: str | None = None,
    finalize_callback: typing.Callable[[OperationResource], None] | None = None,
) -> OperationOutput:
    """Performs common orchestration for general and explore operations

    Use run_closure and finalize_callback to contain differences"""

    #
    # OPERATION RESOURCE
    # Create and add OperationResource to metastore
    #

    operation_resource = add_operation_from_base_config_to_metastore(
        base_operation_configuration=base_operation_configuration,
        metastore=discovery_space.metadataStore,
        space_id=discovery_space.uri,
        operation_identifier=operation_identifier,
    )

    #
    # START THE OPERATION
    #

    print("\n=========== Starting Discovery Operation ===========\n")

    operation_output = None
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
        operation_output = run_closure()
    except KeyboardInterrupt:
        sys.stdout.flush()
        moduleLog.warning("Caught keyboard interrupt - initiating graceful shutdown")
        operationStatus = OperationResourceStatus(
            event=OperationResourceEventEnum.FINISHED,
            exit_state=OperationExitStateEnum.ERROR,
            message="Operation exited due to SIGINT",
        )
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
        if shutdown:
            moduleLog.warning("Operation exited normally but a signal was sent")
            operation_output = None
            operationStatus = OperationResourceStatus(
                event=OperationResourceEventEnum.FINISHED,
                exit_state=OperationExitStateEnum.ERROR,
                message="Operation exited due to SIGTERM)",
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
                    f"Operation exited normally with status {operation_output.exitStatus}"
                )
    finally:
        if operation_output:
            # Add the operation resource if not present
            if not operation_output.operation:
                operation_output.operation = operation_resource

            # Add it to metastore
            moduleLog.info("Adding operation output to metastore")
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
        operation_resource.status.append(operation_output.exitStatus)

        if not shutdown and finalize_callback:
            finalize_callback(operation_resource)

        discovery_space.metadataStore.updateResource(operation_resource)

        print("=========== Operation Details ============\n")
        print(f"Space ID: {operation_resource.config.spaces[0]}")
        print(f"Sample Store ID:  {discovery_space.sample_store.identifier}")
        print(
            f"Operation:\n "
            f"{orchestrator.utilities.output.pydantic_model_as_yaml(operation_resource, exclude_none=True)}"
        )

    return operation_output
