# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import logging
import pathlib
import typing

import pydantic

from orchestrator.core.discoveryspace.space import DiscoverySpace
from orchestrator.core.operation.config import (
    DiscoveryOperationConfiguration,
    OperatorModuleConf,
    get_actuator_configurations,
    validate_actuator_configurations_against_space_configuration,
)
from orchestrator.modules.actuators.measurement_queue import MeasurementQueue
from orchestrator.modules.module import load_module_class_or_function
from orchestrator.utilities.logging import configure_logging

if typing.TYPE_CHECKING:
    from orchestrator.modules.actuators.base import ActuatorActor
    from orchestrator.modules.operators.base import OperatorActor
    from orchestrator.modules.operators.discovery_space_manager import (
        DiscoverySpaceManagerActor,
    )

configure_logging()
moduleLog = logging.getLogger("setup")


def setup_actuators(
    actuator_configuration_identifiers: list[str],
    discovery_space: DiscoverySpace,
    measurement_queue: MeasurementQueue,
) -> dict[str, "ActuatorActor"]:
    """
    Creates all the actuators required by discovery_space

    Params:
        discovery_space: The discovery space to create the actuators for
        actuator_configuration_identifiers: A set of (optional) identifiers of configurations for actuators in the discoveryspace
        queue: the measurement queue

    Raises:
        ray.exceptions.ActorDiedError if any actuator
        raised an exception in init
    """

    import ray

    import orchestrator.modules.actuators.base
    import orchestrator.modules.actuators.registry

    moduleLog.info("Initialising requested actuators")
    registry = orchestrator.modules.actuators.registry.ActuatorRegistry.globalRegistry()
    actuators = {}
    namespace = measurement_queue.ray_namespace()

    if issues := registry.checkMeasurementSpaceSupported(
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

    actuator_configurations = get_actuator_configurations(
        actuator_configuration_identifiers=actuator_configuration_identifiers,
        project_context=discovery_space.project_context,
    )

    validate_actuator_configurations_against_space_configuration(
        actuator_configurations=actuator_configurations,
        discovery_space_configuration=discovery_space.config,
    )

    # First instantiate any actuators passed in actuatorConfigurations

    actuator_configurations = actuator_configurations if actuator_configurations else []
    for actuatorConfig in actuator_configurations:
        actuatorIdentifier = actuatorConfig.actuatorIdentifier
        actuator: ActuatorActor = (
            registry.actuatorForIdentifier(actuatorIdentifier)
            .options(name=actuatorIdentifier, namespace=namespace)
            .remote(queue=measurement_queue, params=actuatorConfig.parameters)
        )

        actuators[actuatorIdentifier] = actuator

        # VV: Uncomment this line to make sure the actuator loaded properly
        # await actuator.__ray_ready__.remote()

    # Initialise the other required actuators
    actuator_ids = [
        e.actuatorIdentifier for e in discovery_space.measurementSpace.experiments
    ]
    filtered_actuator_ids = [aid for aid in actuator_ids if aid not in actuators]
    filtered_actuator_ids = list(set(filtered_actuator_ids))

    for actuatorIdentifier in filtered_actuator_ids:
        cls = registry.actuatorForIdentifier(actuatorIdentifier)
        try:
            default_actuator_parameters = cls.default_parameters()
        except pydantic.ValidationError as error:
            moduleLog.critical(
                f"The default parameters for {actuatorIdentifier} cannot be used. Reason: \n {error} \nThey may need to be customised"
            )
            raise

        moduleLog.debug(f"Instantiating actuator: {actuatorIdentifier}")

        actuator: ActuatorActor = cls.options(
            name=actuatorIdentifier, namespace=namespace
        ).remote(
            queue=measurement_queue,
            params=default_actuator_parameters,
        )

        actuators[actuatorIdentifier] = actuator

    # Check that are all ready - this will raise ray.exceptions.ActorDiedError
    # if any died
    ray.get([a.ready.remote() for a in actuators.values()])

    return actuators


def setup_operator(
    operator_module: OperatorModuleConf,
    parameters: dict,
    discovery_space: DiscoverySpace,
    namespace: str,
    state: "DiscoverySpaceManagerActor",
    actuators: dict,
) -> "OperatorActor":
    """Sets up and creates an operator actor for class-based operations

    This function loads the operator class, creates a Ray actor instance with the
    specified namespace, and initializes it with the provided parameters, state,
    and actuators.

    Params:
        operator_module: Configuration for the operator module to load
        parameters: Dictionary of parameters to pass to the operator
        discovery_space: The discovery space the operator will operate on
        namespace: Ray namespace to create the operator actor in
        state: DiscoverySpaceManager actor handle for state management
        actuators: Dictionary of actuator actor handles keyed by actuator identifier

    Returns:
        OperatorActor handle for the created operator actor
    """

    import orchestrator.utilities.output

    moduleLog.info("Creating operation")

    operatorClass = load_module_class_or_function(operator_module)
    operator = operatorClass.options(
        name=operator_module.moduleClass, namespace=namespace
    ).remote(
        operationActorName=operator_module.moduleClass,
        namespace=namespace,
        state=state,
        params=parameters,
        actuators=actuators,
    )

    print("=========== Operation Details ============\n")
    print(f"Space ID: {discovery_space.uri}")
    print(f"Sample Store ID:  {discovery_space.sample_store.identifier}")
    conf_string = orchestrator.utilities.output.pydantic_model_as_yaml(
        DiscoveryOperationConfiguration(module=operator_module, parameters=parameters),
        exclude_none=True,
    )
    print(f"Operation Configuration:\n {conf_string}")

    return operator


def write_entities(
    entities_output_file: str | pathlib.Path | None,
    discovery_space: DiscoverySpace,
) -> None:

    print("Requested to write entities to original sample store format")
    print(
        f"Note: Entities have also been stored in active sample store at {discovery_space.uri}"
    )

    entities = discovery_space.sampledEntities()

    try:
        discovery_space.sample_store.__class__.writeEntities(
            entities, filename=entities_output_file
        )
    except AttributeError as error:
        print(
            f"Sample Store class {discovery_space.sample_store.__class__} does not support entity writing: {error}"
        )
    except Exception as error:
        moduleLog.warning(f"Unexpected exception while writing entity data: {error}")
