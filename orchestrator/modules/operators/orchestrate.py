# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

"""This module defines the main loop of an optimization process"""

import logging
import os
import signal

import pydantic
import ray
import ray.util.queue
from ray.runtime_env import RuntimeEnv

from orchestrator.core.discoveryspace.space import DiscoverySpace
from orchestrator.core.operation.config import (
    DiscoveryOperationResourceConfiguration,
    FunctionOperationInfo,
)
from orchestrator.core.operation.operation import OperationException, OperationOutput
from orchestrator.metastore.base import ResourceDoesNotExistError
from orchestrator.metastore.project import ProjectContext
from orchestrator.modules.operators._cleanup import (
    CLEANER_ACTOR,  # noqa: F401
    ResourceCleaner,  # noqa: F401
    cleanup_callback_functions,
    graceful_operation_shutdown_signal_handler,
)
from orchestrator.modules.operators._explore_orchestration import (
    orchestrate_explore_operation,
)

# Want this function to be accessed via this module not the private module
from orchestrator.modules.operators._general_orchestration import (
    orchestrate_general_operation,  # noqa: F401
)
from orchestrator.utilities.logging import configure_logging

configure_logging()
moduleLog = logging.getLogger("orch")


def graceful_orchestrate_shutdown() -> None:
    """Clean resources set up by orchestrate()

    This includes ray.shutdown and waiting for logs to flush."""

    import time

    from rich.status import Status

    with Status("Shutdown - shutting down Ray", spinner="dots") as status:
        ray.shutdown()
        status.update("Shutdown - waiting for logs to flush")
        moduleLog.info("Waiting for logs to flush ...")
        time.sleep(10)
        moduleLog.info("Graceful shutdown complete")


def orchestrate(
    operation_resource_configuration: DiscoveryOperationResourceConfiguration,
    project_context: ProjectContext,
    discovery_space_identifier: str,
) -> OperationOutput:
    """Orchestrate the execution of an operation defined as a function or a class (OperationModule)

    This function initializes Ray, loads the discovery space from the metastore, and executes
    the operation based on its implementation type (class-based or function-based).

    Params:
        operation_resource_configuration: Configuration for the operation including module,
            parameters, metadata, actuator configurations, and target spaces
        project_context: Project context for connecting to the metastore
        discovery_space_identifier: Identifier of the discovery space to load from the metastore

    Returns:
        OperationOutput containing the results and status of the operation

    Raises:
        ValueError: If the measurement space is inconsistent
        OperationException: If there is an error during the operation
        pydantic.ValidationError: If the operation parameters are not valid
        ray.exceptions.ActorDiedError: If there was an error initializing actors
    """

    import orchestrator.modules.operators.setup

    #
    # INIT RAY
    #

    # If we are running with a ray runtime environment we need to handle env-vars differently
    if "RAY_JOB_CONFIG_JSON_ENV_VAR" in os.environ:
        ray_runtime_config = os.environ["RAY_JOB_CONFIG_JSON_ENV_VAR"]
        moduleLog.info(
            f"Runtime environment variables are set based on provided ray runtime environment - {ray_runtime_config}"
        )
        ray.init(ignore_reinit_error=True)
    else:
        # In local mode we can read a set of envvars a then export them into the ray environment
        # Currently we don't use it but keeping the code to recall how to do so if necessary
        ray_env_vars = {}
        moduleLog.debug(
            f"Setting runtime environment variables based on local environment - {ray_env_vars}"
        )
        ray.init(
            runtime_env=RuntimeEnv(env_vars=ray_env_vars),
            ignore_reinit_error=True,
        )

        moduleLog.debug("Ensuring envvars are set the main process environment")
        for key, value in ray_env_vars.items():
            os.environ[key] = value

    #
    # Register signal handler
    #
    signal.signal(
        signalnum=signal.SIGTERM, handler=graceful_operation_shutdown_signal_handler()
    )
    cleanup_callback_functions["orchestrate"] = graceful_orchestrate_shutdown

    #
    # GET SPACE
    #
    discovery_space = DiscoverySpace.from_stored_configuration(
        project_context=project_context,
        space_identifier=discovery_space_identifier,
    )

    if not discovery_space.measurementSpace.isConsistent:
        moduleLog.critical("The measurement space is inconsistent - aborting")
        raise ValueError("The measurement space is inconsistent")

    #
    # RUN OPERATION
    #

    operation_info = FunctionOperationInfo(
        metadata=operation_resource_configuration.metadata,
        actuatorConfigurationIdentifiers=operation_resource_configuration.actuatorConfigurationIdentifiers,
    )

    try:
        if isinstance(
            operation_resource_configuration.operation.module,
            orchestrator.core.operation.config.OperatorModuleConf,
        ):
            if (
                operation_resource_configuration.operation.module.operationType
                == orchestrator.core.operation.config.DiscoveryOperationEnum.SEARCH
            ):
                output = orchestrate_explore_operation(
                    operator_module=operation_resource_configuration.operation.module,
                    discovery_space=discovery_space,
                    parameters=operation_resource_configuration.operation.parameters,
                    operation_info=operation_info,
                )
            else:
                raise ValueError(
                    "Implementing operations as classes is only supported for explore operations"
                )
        else:
            output = (
                operation_resource_configuration.operation.module.operationFunction()(
                    discovery_space,
                    operationInfo=operation_info,
                    **operation_resource_configuration.operation.parameters,
                )
            )  # type: OperationOutput
    except KeyboardInterrupt:
        moduleLog.warning("Caught keyboard interrupt - initiating graceful shutdown")
        raise
    except OperationException as error:
        moduleLog.critical(f"Error, {error}, detected during operation")
        raise
    except (
        ValueError,
        pydantic.ValidationError,
        ray.exceptions.ActorDiedError,
        ResourceDoesNotExistError,
    ) as error:
        moduleLog.critical(
            f"Error, {error}, in operation setup. Operation resource not created - exiting"
        )
        raise
    except BaseException as error:
        moduleLog.critical(
            f"Unexpected error, {error}, in operation setup. Operation resource not created - exiting"
        )
        raise
    finally:
        if not orchestrator.modules.operators._cleanup.shutdown_signal_received:
            graceful_orchestrate_shutdown()
            cleanup_callback_functions.pop("orchestrate")

    return output
