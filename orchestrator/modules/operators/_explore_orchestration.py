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
    BaseOperationRunConfiguration,
    DiscoveryOperationConfiguration,
    FunctionOperationInfo,
)
from orchestrator.core.operation.operation import OperationOutput
from orchestrator.metastore.project import ProjectContext
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
from orchestrator.modules.operators.console_output import (
    RichConsoleQueue,
    run_operation_live_updates,
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
    )  # type: typing.Type["StateSubscribingDiscoveryOperation"]
    operator_class.validateOperationParameters(
        base_operation_configuration.operation.parameters
    )

    identifier = operator.operationIdentifier.remote()
    identifier = ray.get(identifier)

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
