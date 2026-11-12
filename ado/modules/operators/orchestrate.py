# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""This module defines the main loop of an optimization process"""

import logging
import os
import signal

import pydantic
import ray
import ray.util.queue

from ado.core.discoveryspace.space import DiscoverySpace
from ado.core.operation.config import (
    DiscoveryOperationResourceConfiguration,
    FunctionOperationInfo,
    OperatorModuleConf,
    OperatorReference,
    ScriptOperatorConf,
)
from ado.core.operation.operation import OperationException, OperationOutput
from ado.metastore.base import ResourceDoesNotExistError
from ado.metastore.project import ProjectContext
from ado.modules.operators._cleanup import (
    CLEANER_ACTOR,  # noqa: F401
    ResourceCleaner,  # noqa: F401
    cleanup_callback_functions,
    graceful_operation_shutdown_signal_handler,
)

# These functions are re-exported via this module — keep the imports even if
# not referenced locally.
from ado.modules.operators._explore_orchestration import (
    orchestrate_explore_operation,  # noqa: F401
)
from ado.modules.operators._general_orchestration import (
    orchestrate_general_operation,  # noqa: F401
)
from ado.utilities.logging import configure_logging

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


def _check_if_using_unsupported_operator_module_conf(
    operation_resource_configuration: DiscoveryOperationResourceConfiguration,
) -> None:
    if isinstance(
        operation_resource_configuration.operation.module, OperatorModuleConf
    ):
        moduleLog.warning(
            "The supplied operation configuration uses an unsupported legacy format for the"
            "operation.module field: Use operatorName/operationType instead "
            "of moduleName/moduleClass. See https://ibm.github.io/ado/latest/user-guide/examples/tutorials/density-example/#step-4-run-an-operation"
            "for an example. "
        )
        raise ValueError(
            "The supplied operation configuration uses an unsupported legacy format for the"
            "operation.module field: Use operatorName/operationType instead "
            "of moduleName/moduleClass. See https://ibm.github.io/ado/latest/user-guide/examples/tutorials/density-example/#step-4-run-an-operation"
            "for an example. "
        )


def operation_requires_ray(
    operation_resource_configuration: DiscoveryOperationResourceConfiguration,
) -> bool:
    """Return whether the configured operator needs Ray initialized.

    Registered :class:`~ado.core.operation.config.OperatorReference` operators
    declare this via :attr:`~ado.core.operation.config.OperatorMetadata.requiresRay`.
    Explore operators always require Ray. Unknown or legacy module forms default
    to requiring Ray so behaviour stays conservative.

    Args:
        operation_resource_configuration: The operation configuration being run.

    Returns:
        True when Ray should be initialized before running the operator.
    """
    module = operation_resource_configuration.operation.module
    if isinstance(module, OperatorReference):
        from ado.modules.operators.collections import operator_metadata_for_reference

        try:
            return operator_metadata_for_reference(module).requiresRay
        except ValueError:
            moduleLog.warning(
                "Could not resolve operator metadata for %s; assuming Ray is required",
                module,
            )
            return True
    if isinstance(module, ScriptOperatorConf | OperatorModuleConf):
        return True
    return True


def _running_as_ray_job() -> bool:
    """Return True when this process is a Ray Jobs API driver."""
    return "RAY_JOB_CONFIG_JSON_ENV_VAR" in os.environ


def should_initialize_ray(requires_ray: bool) -> bool:
    """Return whether orchestrate should initialize Ray and install signal handlers.

    Ray is initialized when the operator declares ``requiresRay``, or when this
    process is already a Ray job driver (so job connect, SIGTERM handling, and
    job metadata still work for non-Ray operators submitted via ``ray job
    submit``). Local non-Ray operators skip initialization.

    Args:
        requires_ray: Whether the operator metadata requires Ray for compute.

    Returns:
        True when Ray should be initialized for this orchestrate call.
    """
    return requires_ray or _running_as_ray_job()


def _initialize_ray_for_operation() -> None:
    """Initialize Ray for a local driver or reconnect inside a Ray job."""
    # If we are running with a ray runtime environment we need to handle env-vars differently
    if _running_as_ray_job():
        ray_runtime_config = os.environ["RAY_JOB_CONFIG_JSON_ENV_VAR"]
        moduleLog.info(
            f"Runtime environment variables are set based on provided ray runtime environment - {ray_runtime_config}"
        )
        ray.init(ignore_reinit_error=True)
        return

    # In local mode, propagate the current log level to Ray workers so that
    # remote functions and actors inherit the same log level as the CLI process.
    #
    # NOTE: runtime_env must be passed as a plain dict, not a RuntimeEnv instance.
    # RuntimeEnv(working_dir=None) does not behave the same as {"working_dir": None}
    # and causes incorrect working-directory handling. This was established in
    # commit a9a14708 (https://github.com/IBM/ado/pull/546).
    ray_env_vars = {"LOGLEVEL": logging.getLevelName(logging.getLogger().level)}
    moduleLog.debug(
        f"Setting runtime environment variables based on local environment - {ray_env_vars}"
    )
    ray.init(
        runtime_env={"env_vars": ray_env_vars},
        ignore_reinit_error=True,
    )


def orchestrate(
    operation_resource_configuration: DiscoveryOperationResourceConfiguration,
    project_context: ProjectContext,
) -> OperationOutput:
    """Orchestrate the execution of an operation defined as a function or a class.

    This function optionally initializes Ray (when the operator requires it, or
    when running as a Ray job), resolves every named input from the operation
    configuration, and calls the operator with those inputs.

    Args:
        operation_resource_configuration: Configuration for the operation including
            module, parameters, metadata, actuator configurations, and inputs.
        project_context: Project context for connecting to the metastore.

    Returns:
        OperationOutput containing the results and status of the operation.

    Raises:
        ValueError: If a measurement space is inconsistent.
        OperationException: If there is an error during the operation.
        pydantic.ValidationError: If the operation parameters are not valid.
        ray.exceptions.ActorDiedError: If there was an error initializing actors.
    """

    import ado.modules.operators.setup  # noqa: F401 — side-effect: registers Ray actors
    from ado.core.operation.inputs import resource_references_to_rich_types
    from ado.metastore.sqlstore import SQLStore

    _check_if_using_unsupported_operator_module_conf(operation_resource_configuration)

    requires_ray = operation_requires_ray(operation_resource_configuration)
    ray_initialized = False
    if should_initialize_ray(requires_ray):
        if not requires_ray:
            moduleLog.info(
                "Operator does not require Ray for compute, but process is a "
                "Ray job — connecting and installing signal handlers"
            )
        _initialize_ray_for_operation()
        ray_initialized = True
        signal.signal(
            signalnum=signal.SIGTERM,
            handler=graceful_operation_shutdown_signal_handler(),
        )
        cleanup_callback_functions["orchestrate"] = graceful_orchestrate_shutdown
    else:
        moduleLog.info(
            "Operator does not require Ray — skipping local Ray initialization"
        )

    #
    # GET INPUTS
    #

    metastore = SQLStore(project_context=project_context)
    inputs = resource_references_to_rich_types(
        resource_references=operation_resource_configuration.inputs,
        metastore=metastore,
    )

    for resource in inputs.values():
        if (
            isinstance(resource, DiscoverySpace)
            and not resource.measurementSpace.isConsistent
        ):
            moduleLog.critical("The measurement space is inconsistent - aborting")
            raise ValueError("The measurement space is inconsistent")

    #
    # RUN OPERATION
    #

    operation_info = FunctionOperationInfo(
        metadata=operation_resource_configuration.metadata,
        actuatorConfigurationIdentifiers=operation_resource_configuration.actuatorConfigurationIdentifiers,
        projectContext=project_context,
    )

    operation_parameters = operation_resource_configuration.operation.parameters

    try:
        operator_fn = (
            operation_resource_configuration.operation.module.operationFunction()
        )
        output: OperationOutput = operator_fn(
            **inputs,
            operationInfo=operation_info,
            parameters=operation_parameters,
        )
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
        if (
            ray_initialized
            and not ado.modules.operators._cleanup.shutdown_signal_received
        ):
            graceful_orchestrate_shutdown()
            cleanup_callback_functions.pop("orchestrate", None)

    return output
