# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

"""This module defines the main loop of an optimization process"""

import logging
import os
import pathlib
import typing

import pydantic
import ray
import ray.util.queue
from ray.runtime_env import RuntimeEnv

import orchestrator.core
import orchestrator.core.discoveryspace.config
import orchestrator.core.operation.config
import orchestrator.modules.operators._cleanup
import orchestrator.utilities.output
from orchestrator.core.discoveryspace.space import DiscoverySpace
from orchestrator.core.operation.config import (
    BaseOperationRunConfiguration,
    FunctionOperationInfo,
)
from orchestrator.core.operation.operation import OperationException, OperationOutput
from orchestrator.core.operation.resource import (
    OperationResource,
)
from orchestrator.metastore.project import ProjectContext
from orchestrator.modules.operators._cleanup import (
    CLEANER_ACTOR,  # noqa: F401
    ResourceCleaner,  # noqa: F401
    graceful_operation_shutdown,
    initialize_resource_cleaner,
)

# Want explore_operation_function_wrapper function to be accessed via this module not the private module
from orchestrator.modules.operators._explore_orchestration import (
    explore_operation_function_wrapper,  # noqa: F401
    orchestrate_explore_operation,
)

# Want this function to be accessed via this module not the private module
from orchestrator.modules.operators._general_orchestration import (
    orchestrate_general_operation,  # noqa: F401
)
from orchestrator.utilities.logging import configure_logging

if typing.TYPE_CHECKING:
    pass

configure_logging()
moduleLog = logging.getLogger("orch")


def orchestrate_operation_function(
    base_operation_configuration: BaseOperationRunConfiguration,
    project_configuration: ProjectContext,
    discovery_space: DiscoverySpace,
) -> tuple[
    "DiscoverySpace",
    "OperationResource",
    "OperationOutput",
]:
    """This functions orchestrate operations with function operators.

    It gets the actuator configurations (if any) and calls the function
    defined in base_operation_configuration.

    This function will either call
    - explore_operation_function_wrapper -> orchestrate_explore_operation -> _run_operation_harness
    - orchestrate_general_operation -> _run_operation_harness
    """

    import orchestrator.modules.operators.collections  # noqa: F401

    initialize_resource_cleaner()

    # TODO: Check if this is necessary
    # Because
    # They are not passed
    # if explore -> this is done again
    # If general ??
    actuator_configurations = (
        base_operation_configuration.validate_actuatorconfigurations_against_space(
            project_context=project_configuration,
            discoverySpaceConfiguration=discovery_space.config,
        )
    )

    if actuator_configurations is None:
        actuator_configurations = []

    output = base_operation_configuration.operation.module.operationFunction()(
        discovery_space,
        operationInfo=FunctionOperationInfo(
            metadata=base_operation_configuration.metadata,
            actuatorConfigurationIdentifiers=base_operation_configuration.actuatorConfigurationIdentifiers,
        ),
        **base_operation_configuration.operation.parameters,
    )  # type: OperationOutput

    return discovery_space, output.operation, output


def orchestrate(
    base_operation_configuration: BaseOperationRunConfiguration,
    project_context: ProjectContext,
    discovery_space_configuration: (
        orchestrator.core.discoveryspace.config.DiscoverySpaceConfiguration | None
    ),
    discovery_space_identifier: str | None,
    entities_output_file: str | pathlib.Path | None = None,
    queue: "ray.util.queue.Queue" = None,
    execid: str | None = None,
) -> OperationOutput:
    """orchestrate the execution of an operation defined as a function or a class (OperationModule)

    Supports
    - running with either a discovery space id OR a discovery space configuration if the operation is implemented
    as a class running ONLY with discovery space id if the operation is implemented as an OperationFunction

    How the operation is implemented is given by base_operation_configuration.operation.module
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
        ray.init(namespace=execid, ignore_reinit_error=True)
    else:
        # In local mode we can read a set of envvars a then export them into the ray environment
        # Currently we don't use it but keeping the code to recall how to do so if necessary
        ray_env_vars = {}
        moduleLog.debug(
            f"Setting runtime environment variables based on local environment - {ray_env_vars}"
        )
        ray.init(
            runtime_env=RuntimeEnv(env_vars=ray_env_vars),
            namespace=execid,
            ignore_reinit_error=True,
        )

        moduleLog.debug("Ensuring envvars are set the main process environment")
        for key, value in ray_env_vars.items():
            os.environ[key] = value

    #
    # GET SPACE
    #

    if discovery_space_configuration:
        discovery_space = DiscoverySpace.from_configuration(
            conf=discovery_space_configuration,
            project_context=project_context,
            identifier=None,
        )
        print("Storing space (if backend storage configured)")
        discovery_space.saveSpace()
    elif discovery_space_identifier:
        discovery_space = DiscoverySpace.from_stored_configuration(
            project_context=project_context,
            space_identifier=discovery_space_identifier,
        )
    else:
        raise ValueError(
            "You must provide a discovery space configuration or identifier"
        )

    if not discovery_space.measurementSpace.isConsistent:
        moduleLog.critical("The measurement space is inconsistent - aborting")
        raise ValueError("The measurement space is inconsistent")

    #
    # RUN OPERATION
    # How depends on if they are implemented as functions or classes
    #
    try:
        if isinstance(
            base_operation_configuration.operation.module,
            orchestrator.core.operation.config.OperatorModuleConf,
        ):
            if (
                base_operation_configuration.operation.module.operationType
                == orchestrator.core.operation.config.DiscoveryOperationEnum.SEARCH
            ):
                _, _, output = orchestrate_explore_operation(
                    base_operation_configuration=base_operation_configuration,
                    project_context=project_context,
                    discovery_space=discovery_space,
                    namespace=execid,
                    queue=queue,
                )
            else:
                raise ValueError(
                    "Implementing operations as classes is only supported for explore operations"
                )
        else:
            _, _, output = orchestrate_operation_function(
                base_operation_configuration=base_operation_configuration,
                project_configuration=project_context,
                discovery_space=discovery_space,
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
        if not orchestrator.modules.operators._cleanup.shutdown:
            # If we get here the exception must have been raised before the operation started.
            # Therefore, we don't need to wait in DiscoverySpaceManager, Actuators etc. to shut down
            # as they never processed any date.
            graceful_operation_shutdown()

    return output
