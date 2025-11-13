# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

"""This module defines the main loop of an optimization process"""

import logging
import os
import pathlib
import signal
import sys
import time
import typing

import pydantic
import ray
import ray.util.queue
from ray.actor import ActorHandle
from ray.exceptions import RayTaskError
from ray.runtime_env import RuntimeEnv

import orchestrator.core
import orchestrator.core.discoveryspace.config
import orchestrator.core.operation.config
import orchestrator.modules.actuators.base
import orchestrator.schema.observed_property
import orchestrator.utilities.output
from orchestrator.core.discoveryspace.space import DiscoverySpace
from orchestrator.core.operation.config import (
    BaseOperationRunConfiguration,
    DiscoveryOperationConfiguration,
    FunctionOperationInfo,
    OperatorFunctionConf,
)
from orchestrator.core.operation.operation import OperationException, OperationOutput
from orchestrator.core.operation.resource import (
    OperationExitStateEnum,
    OperationResource,
    OperationResourceEventEnum,
    OperationResourceStatus,
)
from orchestrator.metastore.project import ProjectContext
from orchestrator.modules.actuators.measurement_queue import MeasurementQueue
from orchestrator.modules.actuators.registry import ActuatorRegistry
from orchestrator.modules.module import load_module_class_or_function
from orchestrator.modules.operators.base import (
    add_operation_from_base_config_to_metastore,
    add_operation_output_to_metastore,
)
from orchestrator.modules.operators.discovery_space_manager import DiscoverySpaceManager
from orchestrator.utilities.logging import configure_logging

if typing.TYPE_CHECKING:
    from orchestrator.modules.actuators.base import ActuatorActor
    from orchestrator.modules.operators.base import OperatorActor
    from orchestrator.modules.operators.discovery_space_manager import (
        DiscoverySpaceManagerActor,
    )


# Global variable to track if graceful shutdown was called
shutdown = False

configure_logging()
moduleLog = logging.getLogger("orch")
CLEANER_ACTOR = "resource_cleaner"


@ray.remote
class ResourceCleaner:
    """
    This is a singleton allowing various custom actors to clean up before shutdown,
    """

    def __init__(self):
        """
        Constructor
        """
        # list of handles for the actors to be cleaned
        self.to_clean = []

    def add_to_cleanup(self, handle: ActorHandle) -> None:
        """
        Add to clean up
        Can be used by any custom actor to add itself to clean up list. This class has to implement cleanup method
        :param handle: handle of the actor to be cleaned
        :return: None
        """
        self.to_clean.append(handle)

    def cleanup(self) -> None:
        """
        Clean up all required classes
        :return: None
        """
        if len(self.to_clean) > 0:
            handles = [h.cleanup.remote() for h in self.to_clean]
            done, not_done = ray.wait(
                ray_waitables=handles, num_returns=len(handles), timeout=60.0
            )
            moduleLog.info(f"cleaned {len(done)}, clean failed {len(not_done)}")


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


def graceful_operation_shutdown():

    global shutdown

    if not shutdown:
        import time

        moduleLog.info("Shutting down gracefully")

        shutdown = True

        moduleLog.debug("Cleanup custom actors")
        try:
            cleaner_handle = ray.get_actor(name=CLEANER_ACTOR)
            ray.get(cleaner_handle.cleanup.remote())
            # deleting a cleaner actor. It is detached one, so has to be deleted explicitly
            ray.kill(cleaner_handle)
        except Exception as e:
            moduleLog.warning(f"Failed to cleanup custom actors {e}")

        moduleLog.info("Shutting down Ray...")
        ray.shutdown()
        moduleLog.info("Waiting for logs to flush ...")
        time.sleep(10)
        moduleLog.info("Graceful shutdown complete")
    else:
        moduleLog.info("Graceful shutdown already completed")


def graceful_operation_shutdown_handler() -> (
    typing.Callable[[int, typing.Any | None], None]
):

    def handler(sig, frame):

        moduleLog.warning(f"Got signal {sig}")
        moduleLog.warning("Calling graceful shutdown")
        graceful_operation_shutdown()

    return handler


def graceful_explore_operation_shutdown(
    operator: "OperatorActor",
    state: "DiscoverySpaceManagerActor",
    actuators: list["ActuatorActor"],
    timeout=60,
):

    global shutdown

    if not shutdown:
        import time

        moduleLog.info("Shutting down gracefully")

        shutdown = True

        #
        # Shutdown process
        # 1. Shutdown state calling onComplete on operation and metricServer and ensuring metrics are flushed
        # 2. Shutdown custom actors
        # 3. Send graceful __ray_terminate__ to metric_server, operation and actuators

        # This should not return until the metric server has processed all updates.
        moduleLog.debug("Shutting down state")
        promise = state.shutdown.remote()
        ray.get(promise)

        moduleLog.debug("Cleanup custom actors")
        try:
            cleaner_handle = ray.get_actor(name=CLEANER_ACTOR)
            ray.get(cleaner_handle.cleanup.remote())
            # deleting a cleaner actor. It is detached one, so has to be deleted explicitly
            ray.kill(cleaner_handle)
        except Exception as e:
            moduleLog.warning(f"Failed to cleanup custom actors {e}")

        wait_graceful = [
            operator.__ray_terminate__.remote(),
            state.__ray_terminate__.remote(),
        ]
        # __ray_terminate allows atexit handlers of actors to run
        # see  https://docs.ray.io/en/latest/ray-core/api/doc/ray.kill.html
        wait_graceful.extend([a.__ray_terminate__.remote() for a in actuators])
        n_actors = len(wait_graceful)
        moduleLog.debug(f"waiting for graceful shutdown of {n_actors} actors")

        actors = [operator]
        actors.extend(actuators)

        lookup = dict(zip(wait_graceful, actors))

        moduleLog.debug(f"Shutdown waiting on {lookup}")
        moduleLog.debug(
            f"Gracefully stopping actors - will wait {timeout} seconds  ..."
        )
        terminated, active = ray.wait(
            ray_waitables=wait_graceful, num_returns=n_actors, timeout=60.0
        )

        moduleLog.debug(f"Terminated: {terminated}")
        moduleLog.debug(f"Active: {active}")

        if active:
            moduleLog.warning(
                f"Some actors have not completed after {timeout} grace period - killing"
            )
            for actor_ref in active:
                print(f"... {lookup[actor_ref]}")
                ray.kill(lookup[actor_ref])

        moduleLog.info("Shutting down Ray...")
        ray.shutdown()
        moduleLog.info("Waiting for logs to flush ...")
        time.sleep(10)
        moduleLog.info("Graceful shutdown complete")
    else:
        moduleLog.info("Graceful shutdown already completed")


def graceful_explore_operation_shutdown_handler(
    operation, state, actuators, timeout=60
) -> typing.Callable[[int, typing.Any | None], None]:
    """Return a signal handler that sh."""

    def handler(sig, frame):

        moduleLog.warning(f"Got signal {sig}")
        moduleLog.warning("Calling graceful shutdown")
        graceful_explore_operation_shutdown(
            operator=operation,
            state=state,
            actuators=actuators,
            timeout=timeout,
        )

    return handler


def run_explore_operation_core_closure(
    operator: "OperatorActor", state: "DiscoverySpaceManagerActor"
) -> typing.Callable[[], OperationOutput]:

    def _run_explore_operation_core() -> OperationOutput:
        import numpy as np
        import pandas as pd
        from rich.console import Console
        from rich.live import Live
        from rich.table import Table

        discovery_space = ray.get(state.discoverySpace.remote())
        operation_id = ray.get(operator.operationIdentifier.remote())

        def output_operation_results(row_limit: int | None) -> Table:
            df: pd.DataFrame = (
                discovery_space.complete_measurement_request_with_results_timeseries(
                    operation_id=operation_id,
                    output_format="target",
                )
            )

            table_title = (
                f"Latest measurements - {operation_id}"
                if row_limit
                else f"Measurements - {operation_id}"
            )
            table = Table(title=table_title)

            if df.empty:
                return table

            # Remove the columns result_index, generatorid and entityIdentifier
            # We have the constitutive properties in the df, so we don't need to show them
            df = df.drop(
                columns=["result_index", "generatorid", "identifier"], errors="ignore"
            )
            df.insert(0, "index", np.arange(len(df)))

            # If there is only one experiment drop the experiment column
            if len(discovery_space.measurementSpace.experiments) == 1:
                df = df.drop(columns=["experiment_id"], errors="ignore")
            else:
                # Convert the experiment column - which is ExperimentReference instances
                # to experiment identifiers
                df["experiment_id"] = df["experiment_id"].apply(
                    lambda x: x.experimentIdentifier
                )

            # Dynamically determine how many columns can fit the screen
            console = Console()
            terminal_width = console.width
            min_col_width = 12  # Minimum width per column (estimate)
            max_columns = max(1, terminal_width // min_col_width)

            visible_columns = list(df.columns[:max_columns])
            hidden_columns = len(df.columns) - max_columns

            if hidden_columns > 0:
                visible_columns.append(f"... (+{hidden_columns} more)")

            # Add columns manually setting overflow="fold" - this will cause text to wrap
            # It can't be set at table level
            for col in visible_columns:
                table.add_column(col, overflow="fold")

            for row_number, (_, row) in enumerate(df[::-1].iterrows()):

                if row_limit and row_number == row_limit:
                    break

                # Format numbers to 2 significant figures
                # Add the row index from the DataFrame to the first column
                row_data = [
                    (
                        f"{row[col]:.2f}"
                        if isinstance(row[col], float)
                        else str(row[col])
                    )
                    for col in df.columns[:max_columns]
                ]

                if hidden_columns > 0:
                    row_data.append("...")

                table.add_row(*row_data)

            return table

        state.startMonitoring.remote()
        future = operator.run.remote()
        finished = []

        # Try to make the table be more or less half of the terminal height
        table_height = max(int(Console().height / 2) - 4, 4)
        with Live(output_operation_results(row_limit=table_height)) as live:
            while not finished:
                live.update(output_operation_results(row_limit=table_height))
                finished, _ = ray.wait(ray_waitables=[future], timeout=2)

            # Output the whole table before exiting
            live.update(output_operation_results(row_limit=None))
        return ray.get(future)  # type: OperationOutput

    return _run_explore_operation_core


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
            message=f"Operation exited due to the following error from a Ray task: {e}.",
        )
        raise OperationException(
            message=f"Error raised from Ray task while executing operation {operation_resource.identifier}",
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

    functionConf = OperatorFunctionConf(
        operatorName=operator_function.__name__,
        operationType=operation_type,
    )

    if parameters_model:
        parameters_model.model_validate(operation_parameters)

    # Check the space
    if not discovery_space.measurementSpace.isConsistent:
        moduleLog.critical("Measurement space is inconsistent - aborting")
        raise ValueError("Measurement space is inconsistent")

    base_configuration = BaseOperationRunConfiguration(
        operation=DiscoveryOperationConfiguration(
            module=functionConf,
            parameters=operation_parameters,
        ),
        metadata=operation_info.metadata,
        actuatorConfigurationIdentifiers=operation_info.actuatorConfigurationIdentifiers,
    )

    log_space_details(discovery_space)

    operation_run_closure = run_general_operation_core_closure(
        operator_function,
        discovery_space=discovery_space,
        operationInfo=operation_info,
        operation_parameters=operation_parameters,
    )

    global shutdown
    shutdown = False

    signal.signal(
        signalnum=signal.SIGTERM, handler=graceful_operation_shutdown_handler()
    )

    output = _run_operation_harness(
        run_closure=operation_run_closure,
        base_operation_configuration=base_configuration,
        discovery_space=discovery_space,
    )

    graceful_operation_shutdown()

    return output


def orchestrate_explore_operation(
    base_operation_configuration: BaseOperationRunConfiguration,
    project_context: ProjectContext,
    discovery_space: DiscoverySpace,
    namespace: str,
    queue: ray.util.queue.Queue,
) -> tuple[
    "DiscoverySpace",
    "OperationResource",
    "orchestrator.modules.operators.base.OperationOutput",
]:
    """Orchestrates an explore operation

    In addition to the items handles by orchestrate_general_operation this function

    - Sets up the state updating apparatus for explore operation:
       - DiscoverySpaceManager, Actuators, MeasurementQueue etc.

    Exceptions:
        ValueError: if the MeasurementSpace is not consistent with EntitySpace
        pydantic.ValidationError: if the operation parameters are not valid
        OperationException: If there is an error during the operation
        ray.exceptions.ActorDiedError: If there was an error initializing the actuators
    """

    import orchestrator.modules.operators.setup

    initialize_resource_cleaner()

    # Check the space
    if not discovery_space.measurementSpace.isConsistent:
        moduleLog.critical("Measurement space is inconsistent - aborting")
        raise ValueError("Measurement space is inconsistent")

    if issues := ActuatorRegistry.globalRegistry().checkMeasurementSpaceSupported(
        discovery_space.measurementSpace
    ):
        moduleLog.critical(
            "The measurement space is not supported by the known actuators - aborting"
        )
        for issue in issues:
            moduleLog.critical(issue)
        raise ValueError(
            "The measurement space is not supported by the known actuators"
        )

    log_space_details(discovery_space)

    actuator_configurations = (
        base_operation_configuration.validate_actuatorconfigurations_against_space(
            project_context=project_context,
            discoverySpaceConfiguration=discovery_space.config,
        )
    )

    #
    # STATE
    # Create State actor
    #
    if queue is None:
        queue = MeasurementQueue.get_measurement_queue()

    # noinspection PyUnresolvedReferences
    state = DiscoverySpaceManager.options(namespace=namespace).remote(
        queue=queue, space=discovery_space, namespace=namespace
    )  # type: "InternalStateActor"
    moduleLog.debug(f"Waiting for discovery state actor to be ready: {state}")
    _ = ray.get(state.__ray_ready__.remote())
    moduleLog.debug("Discovery state actor is ready")

    #
    #  ACTUATORS
    #
    # Will raise ray.exceptions.ActorDiedError if any actuator died
    # during init
    actuators = orchestrator.modules.operators.setup.setup_actuators(
        namespace=namespace,
        actuator_configurations=actuator_configurations,
        discovery_space=discovery_space,
        queue=queue,
    )
    # FIXME: This is only necessary for mock actuator - but does it actually need to use it?
    for actuator in actuators.values():
        actuator.setMeasurementSpace.remote(discovery_space.measurementSpace)

    #
    # OPERATOR
    #
    operator = orchestrator.modules.operators.setup.setup_operator(
        actuators=actuators,
        discovery_space=discovery_space,
        base_configuration=base_operation_configuration,
        namespace=namespace,
        state=state,
    )  # type: "OperatorActor"

    # Validate the parameters for the operation
    #
    operator_class = load_module_class_or_function(
        base_operation_configuration.operation.module
    )  # type: typing.Type[StateSubscribingDiscoveryOperation]
    operator_class.validateOperationParameters(
        base_operation_configuration.operation.parameters
    )

    identifier = operator.operationIdentifier.remote()
    identifier = ray.get(identifier)

    explore_run_closure = run_explore_operation_core_closure(operator, state)

    global shutdown
    shutdown = False

    signal.signal(
        signalnum=signal.SIGTERM,
        handler=graceful_explore_operation_shutdown_handler(
            operation=operator,
            state=state,
            actuators=actuators,
        ),
    )

    def finalize_callback_closure(operator_actor: "OperatorActor"):
        def finalize_callback(operation_resource: OperationResource):
            # Even on exception we can still get entities submitted
            operation_resource.metadata["entities_submitted"] = ray.get(
                operator_actor.numberEntitiesSampled.remote()
            )
            operation_resource.metadata["experiments_requested"] = ray.get(
                operator_actor.numberMeasurementsRequested.remote()
            )

        return finalize_callback

    output = _run_operation_harness(
        run_closure=explore_run_closure,
        base_operation_configuration=base_operation_configuration,
        discovery_space=discovery_space,
        operation_identifier=identifier,
        finalize_callback=finalize_callback_closure(operator),
    )

    graceful_explore_operation_shutdown(
        operator=operator,
        state=state,
        actuators=list(actuators.values()),
    )

    return discovery_space, output.operation, output


def explore_operation_function_wrapper(
    discovery_space: DiscoverySpace,
    module: orchestrator.core.operation.config.OperatorModuleConf,
    parameters: dict,
    namespace: str,
    operation_info: typing.Optional["FunctionOperationInfo"] = None,
    queue: typing.Optional["ray.util.queue.Queue"] = None,
) -> OperationOutput:
    """
    function implementations of explore operations must call this function.

    It is a small wrapper that converts the arguments passed to the explore function operation,
    to those required to orchestrate an explore (class) operation.
    """

    base_operation_configuration = BaseOperationRunConfiguration(
        operation=DiscoveryOperationConfiguration(
            module=module,
            parameters=parameters,
        ),
        metadata=operation_info.metadata,
        actuatorConfigurationIdentifiers=operation_info.actuatorConfigurationIdentifiers,
    )

    _, _, output = orchestrate_explore_operation(
        base_operation_configuration=base_operation_configuration,
        project_context=discovery_space.project_context,
        discovery_space=discovery_space,
        namespace=namespace,
        queue=queue,
    )

    return output


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
        if not shutdown:
            # If we get here the exception must have been raised before the operation started.
            # Therefore, we don't need to wait in DiscoverySpaceManager, Actuators etc. to shut down
            # as they never processed any date.
            graceful_operation_shutdown()

    return output


def initialize_resource_cleaner():
    # create a cleaner actor.
    # We are creating Named detached actor (https://docs.ray.io/en/latest/ray-core/actors/named-actors.html)
    # so that we do not need to pass its handle (can get it by name) and it does not go out of scope, until
    # we explicitly kill it
    ResourceCleaner.options(
        name=CLEANER_ACTOR, get_if_exists=True, lifetime="detached"
    ).remote()
    # Create a default handler that will clean up the ResourceCleaner
    # Orchestration functions that require more complex shutdown can replace this handler
    signal.signal(
        signalnum=signal.SIGTERM, handler=graceful_operation_shutdown_handler()
    )
