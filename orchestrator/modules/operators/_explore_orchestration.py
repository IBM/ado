# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import logging
import signal
import typing

import ray
import ray.util.queue

import orchestrator.core
import orchestrator.modules
import orchestrator.modules.operators._cleanup
from orchestrator.core import OperationResource
from orchestrator.core.discoveryspace.space import DiscoverySpace
from orchestrator.core.operation.config import (
    FunctionOperationInfo,
    OperatorModuleConf,
    get_actuator_configurations,
    validate_actuator_configurations_against_space_configuration,
)
from orchestrator.core.operation.operation import OperationOutput
from orchestrator.modules.actuators.measurement_queue import MeasurementQueue
from orchestrator.modules.actuators.registry import ActuatorRegistry
from orchestrator.modules.module import load_module_class_or_function
from orchestrator.modules.operators._cleanup import (
    CLEANER_ACTOR,
    initialize_resource_cleaner,
)
from orchestrator.modules.operators._orchestrate_core import (
    _run_operation_harness,
    log_space_details,
)
from orchestrator.modules.operators.discovery_space_manager import DiscoverySpaceManager

moduleLog = logging.getLogger("explore_orchestration")

if typing.TYPE_CHECKING:
    from orchestrator.modules.actuators.base import ActuatorActor
    from orchestrator.modules.operators.base import OperatorActor
    from orchestrator.modules.operators.discovery_space_manager import (
        DiscoverySpaceManagerActor,
    )


def graceful_explore_operation_shutdown(
    operator: "OperatorActor",
    state: "DiscoverySpaceManagerActor",
    actuators: list["ActuatorActor"],
    timeout=60,
):

    if not orchestrator.modules.operators._cleanup.shutdown:
        import time

        moduleLog.info("Shutting down gracefully")

        orchestrator.modules.operators._cleanup.shutdown = True

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
) -> typing.Callable[[], OperationOutput | None]:

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


def orchestrate_explore_operation(
    operator_module: OperatorModuleConf,
    discovery_space: DiscoverySpace,
    parameters: dict,
    operation_info: FunctionOperationInfo,
) -> OperationOutput:
    """Orchestrates an explore operation

    This function sets up and executes an explore (search) operation. It handles:
    - Initializing the resource cleaner
    - Validating the measurement space consistency
    - Validating actuator configurations against the space
    - Setting up DiscoverySpaceManager, Actuators, and MeasurementQueue
    - Creating and running the operator actor
    - Handling graceful shutdown

    It calls run_operation_harness to create, store, and update the operation resource,
    execute the operation, handle exceptions, and store the operation results.

    Params:
        operator_module: Configuration for the operator module (class-based operation)
        discovery_space: The discovery space to operate on
        parameters: Dictionary of parameters for the operation
        operation_info: Information about the operation including metadata, actuator
            configuration identifiers, and namespace

    Returns:
        OperationOutput containing the results and status of the operation

    Raises:
        ValueError: If the MeasurementSpace is not consistent with EntitySpace or if
            actuator configurations are invalid
        pydantic.ValidationError: If the operation parameters are not valid
        OperationException: If there is an error during the operation
        ray.exceptions.ActorDiedError: If there was an error initializing the actuators
        ResourceDoesNotExistError: If an actuator configuration cannot be retrieved from the database
    """

    import uuid

    import orchestrator.modules.operators.setup

    if not operation_info.ray_namespace:
        operation_info.ray_namespace = (
            f"{operator_module.moduleClass}-namespace-{str(uuid.uuid4())[:8]}"
        )

    initialize_resource_cleaner()

    project_context = discovery_space.project_context

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

    actuator_configurations = get_actuator_configurations(
        actuator_configuration_identifiers=operation_info.actuatorConfigurationIdentifiers,
        project_context=project_context,
    )

    validate_actuator_configurations_against_space_configuration(
        actuator_configurations=actuator_configurations,
        discovery_space_configuration=discovery_space.config,
    )

    #
    # STATE
    # Create State actor
    #
    queue = MeasurementQueue.get_measurement_queue()

    # noinspection PyUnresolvedReferences
    state = DiscoverySpaceManager.options(
        namespace=operation_info.ray_namespace
    ).remote(
        queue=queue, space=discovery_space, namespace=operation_info.ray_namespace
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
        namespace=operation_info.ray_namespace,
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

    # Validate the parameters for the operation
    operator_class = load_module_class_or_function(
        operator_module
    )  # type: typing.Type["StateSubscribingDiscoveryOperation"]
    operator_class.validateOperationParameters(parameters)

    # Create operator actor
    operator = orchestrator.modules.operators.setup.setup_operator(
        operator_module=operator_module,
        parameters=parameters,
        discovery_space=discovery_space,
        actuators=actuators,
        namespace=operation_info.ray_namespace,
        state=state,
    )  # type: "OperatorActor"
    identifier = ray.get(operator.operationIdentifier.remote())

    explore_run_closure = run_explore_operation_core_closure(operator, state)

    orchestrator.modules.operators._cleanup.shutdown = False

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

    operation_output = _run_operation_harness(
        run_closure=explore_run_closure,
        discovery_space=discovery_space,
        operator_module=operator_module,
        operation_parameters=parameters,
        operation_info=operation_info,
        operation_identifier=identifier,
        finalize_callback=finalize_callback_closure(operator),
    )

    graceful_explore_operation_shutdown(
        operator=operator,
        state=state,
        actuators=list(actuators.values()),
    )

    return operation_output
