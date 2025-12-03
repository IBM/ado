# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import logging
import signal
import typing

import ray
import ray.util.queue

from orchestrator.core import OperationResource
from orchestrator.core.discoveryspace.space import DiscoverySpace
from orchestrator.core.operation.config import (
    FunctionOperationInfo,
    OperatorModuleConf,
)
from orchestrator.core.operation.operation import OperationOutput
from orchestrator.modules.actuators.measurement_queue import MeasurementQueue
from orchestrator.modules.module import load_module_class_or_function
from orchestrator.modules.operators._cleanup import (
    graceful_operation_shutdown_signal_handler,
    initialize_ray_resource_cleaner,
    shutdown_signal_received,
    signal_cleanup_callbacks,
)
from orchestrator.modules.operators._orchestrate_core import (
    _run_operation_harness,
    log_space_details,
)
from orchestrator.modules.operators.console_output import (
    RichConsoleQueue,
    run_operation_live_updates,
)
from orchestrator.modules.operators.discovery_space_manager import DiscoverySpaceManager

moduleLog = logging.getLogger("explore_orchestration")

if typing.TYPE_CHECKING:
    from orchestrator.modules.actuators.base import ActuatorActor
    from orchestrator.modules.operators.base import (
        OperatorActor,
    )
    from orchestrator.modules.operators.discovery_space_manager import (
        DiscoverySpaceManagerActor,
    )


def graceful_explore_operation_shutdown(
    identifier: str,
    operator: "OperatorActor",
    state: "DiscoverySpaceManagerActor",
    actuators: list["ActuatorActor"],
    namespace: str,
    timeout=60,
):

    from rich.console import Console

    moduleLog.info("Shutting down gracefully")

    orchestrator.modules.operators._cleanup.shutdown_signal_received = True

    #
    # Shutdown process
    # 1. Shutdown state calling onComplete on operation and metricServer and ensuring metrics are flushed
    # 2. Shutdown custom actors
    # 3. Send graceful __ray_terminate__ to metric_server, operation and actuators

    # This should not return until the metric server has processed all updates.

    console = Console()
    with console.status(
        f"Shutdown ({identifier})- waiting on all samples to be stored", spinner="dots"
    ) as status:

        moduleLog.debug("Shutting down state")
        promise = state.shutdown.remote()
        ray.get(promise)

        status.update(
            f"Shutdown ({identifier}) - waiting for actors to terminate (max 60s)"
        )

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
            status.update(
                f"Some actors have not completed after {timeout} grace period - killing"
            )
            for actor_ref in active:
                ray.kill(lookup[actor_ref])


def run_explore_operation_core_closure(
    operator: "OperatorActor", state: "DiscoverySpaceManagerActor"
) -> typing.Callable[[], OperationOutput | None]:

    def _run_explore_operation_core() -> OperationOutput:
        import ray

        # Create RichConsoleQueue
        # this needs to be created before operation starts
        # so operators and actuators can put messages
        queue_handle = RichConsoleQueue.options(
            name="RichConsoleQueue", lifetime="detached", get_if_exists=True
        ).remote()

        discovery_space = ray.get(state.discoverySpace.remote())
        operation_id = ray.get(operator.operationIdentifier.remote())

        state.startMonitoring.remote()
        future = operator.run.remote()

        # Start the rich live updates
        run_operation_live_updates(
            discovery_space=discovery_space,
            operation_id=operation_id,
            console_queue=queue_handle,
            operation_future=future,
        )

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

    initialize_ray_resource_cleaner()

    project_context = discovery_space.project_context

    # Check the space
    if not discovery_space.measurementSpace.isConsistent:
        moduleLog.critical("Measurement space is inconsistent - aborting")
        raise ValueError("Measurement space is inconsistent")

    log_space_details(discovery_space)

    # create cleaner for this namespace
    initialize_ray_resource_cleaner(namespace=operation_info.ray_namespace)

    #
    # MEASUREMENT QUEUE
    #
    # For communication between actuators -> discovery space manager -> operator
    measurement_queue = MeasurementQueue(ray_namespace=operation_info.ray_namespace)

    #
    #  ACTUATORS
    #
    # Will raise ray.exceptions.ActorDiedError if any actuator died during init
    # Will raise ValueError if there is a mismatch between  the Actuators and
    # the actuator configurations
    actuators = orchestrator.modules.operators.setup.setup_actuators(
        actuator_configuration_identifiers=operation_info.actuatorConfigurationIdentifiers,
        discovery_space=discovery_space,
        measurement_queue=measurement_queue,
    )
    # FIXME: This is only necessary for mock actuator - but does it actually need to use it?
    for actuator in actuators.values():
        actuator.setMeasurementSpace.remote(discovery_space.measurementSpace)

    #
    # DISCOVERY SPACE MANAGER
    #

    # noinspection PyUnresolvedReferences
    discovery_space_manager = DiscoverySpaceManager.options(
        namespace=operation_info.ray_namespace
    ).remote(
        queue=measurement_queue,
        space=discovery_space,
        namespace=operation_info.ray_namespace,
    )  # type: "InternalStateActor"
    moduleLog.debug(
        f"Waiting for discovery space manager to be ready: {discovery_space_manager}"
    )
    _ = ray.get(discovery_space_manager.__ray_ready__.remote())
    moduleLog.debug("Discovery space manager is ready")

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
        state=discovery_space_manager,
    )  # type: "OperatorActor"
    identifier = ray.get(operator.operationIdentifier.remote())

    explore_run_closure = run_explore_operation_core_closure(operator, state)

    # Handling SIGTERM
    # First register a callback which will cleanup if SIGTERM is sent
    # and the handler is in place
    # Note we can't register the callback until the actors are created so there
    # is a short window where graceful cleanup is not possible on SIGTERM
    signal_cleanup_callbacks[identifier] = lambda: graceful_explore_operation_shutdown(
        identifier=identifier,
        operator=operator,
        state=state,
        actuators=list(actuators.values()),
    )
    # Next  register the handler in case it was not registered already
    # Since all operations register the same stateless handler, setting
    # it multiple times does not change behaviour
    signal.signal(
        signalnum=signal.SIGTERM, handler=graceful_operation_shutdown_signal_handler()
    )

    def finalize_callback_closure(operator_actor: "OperatorActor"):
        from ray.exceptions import GetTimeoutError

        def finalize_callback(operation_resource: OperationResource):
            # Even on exception we can still get entities submitted
            logging.warning("Finalize callback - Getting entities submitted")
            try:
                operation_resource.metadata["entities_submitted"] = ray.get(
                    operator_actor.numberEntitiesSampled.remote(), timeout=10
                )
                logging.warning("Finalize callback - Getting experiments requested")
                operation_resource.metadata["experiments_requested"] = ray.get(
                    operator_actor.numberMeasurementsRequested.remote()
                )
            except GetTimeoutError:
                logging.warning(
                    "Unable to retrieve entity/experiment submission data from operator"
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

    if not shutdown_signal_received:
        graceful_explore_operation_shutdown(
            identifier=identifier,
            operator=operator,
            state=state,
            actuators=list(actuators.values()),
        )
        signal_cleanup_callbacks.pop(identifier)

    return operation_output
