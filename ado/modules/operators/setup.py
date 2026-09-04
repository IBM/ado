# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import logging
import pathlib
import typing

import pydantic

from ado.core.discoveryspace.space import DiscoverySpace
from ado.core.operation.config import (
    DiscoveryOperationConfiguration,
    OperatorMetadata,
    OperatorReference,
    get_actuator_configurations,
    validate_actuator_configurations_against_space_configuration,
)
from ado.modules.actuators.measurement_queue import MeasurementQueue
from ado.modules.actuators.registry import ActuatorRegistry
from ado.schema.experiment import Experiment
from ado.schema.measurementspace import MeasurementSpace
from ado.utilities.logging import configure_logging

if typing.TYPE_CHECKING:
    from ado.modules.actuators.base import ActuatorActor
    from ado.modules.operators.base import OperatorActor
    from ado.modules.operators.discovery_space_manager import (
        DiscoverySpaceManagerActor,
    )

configure_logging()
moduleLog = logging.getLogger("setup")


def find_fqi_differences(
    registry: ActuatorRegistry, measurement_space: MeasurementSpace
) -> list[tuple[Experiment, Experiment]]:
    """Returns (registry experiment, measurement space experiment) pairs with same major version identifier but different fully qualified identifiers. 


    Returns: A list of tuples with two elements. Each tuple is a FQI difference
        The first element in the tuple is an experiment from the measurement space.
        The second is the major version matching experiment from the registry which
        does not have the same fully qualified identifier"""

    differences = []
    for space_experiment in measurement_space.experiments:
        catalog_experiment = registry.experimentForReference(space_experiment.reference)
        if (
            space_experiment.fully_qualified_identifier
            != catalog_experiment.fully_qualified_identifier
        ):
            differences.append((space_experiment, catalog_experiment))

    return differences


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

    import ado.modules.actuators.base
    import ado.modules.actuators.registry

    moduleLog.info("Initialising requested actuators")
    registry = ado.modules.actuators.registry.ActuatorRegistry.globalRegistry()
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

    for (
        space_experiment,
        catalog_experiment,
    ) in find_fqi_differences(
        registry=registry, measurement_space=discovery_space.measurementSpace
    ):
        print(
            f"Note: Will use {catalog_experiment.fully_qualified_identifier} to satisfy request"
            f" for {space_experiment.major_version_identifier}. (Major Version Match)."
            f"The space was originally created with {space_experiment.fully_qualified_identifier}"
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

    actuator_configurations = actuator_configurations or []
    for actuatorConfig in actuator_configurations:
        actuatorIdentifier = actuatorConfig.actuatorIdentifier
        actuator_class = registry.actuatorForIdentifier(actuatorIdentifier)
        actuator: ActuatorActor = (
            ray.remote(actuator_class)
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
        actuator_class = registry.actuatorForIdentifier(actuatorIdentifier)
        try:
            default_actuator_parameters = actuator_class.default_parameters()
        except pydantic.ValidationError as error:
            moduleLog.critical(
                f"The default parameters for {actuatorIdentifier} cannot be used. Reason: \n {error} \nThey may need to be customised"
            )
            raise

        moduleLog.debug(f"Instantiating actuator: {actuatorIdentifier}")

        actuator: ActuatorActor = (
            ray.remote(actuator_class)
            .options(name=actuatorIdentifier, namespace=namespace)
            .remote(
                queue=measurement_queue,
                params=default_actuator_parameters,
            )
        )

        actuators[actuatorIdentifier] = actuator

    # Check that are all ready - this will raise ray.exceptions.ActorDiedError
    # if any died
    ray.get([a.ready.remote() for a in actuators.values()])

    return actuators


def setup_operator(
    operator_metadata: OperatorMetadata,
    parameters: dict,
    discovery_space: DiscoverySpace,
    namespace: str,
    discovery_space_manager: "DiscoverySpaceManagerActor",
    actuators: dict,
) -> "OperatorActor":
    """Sets up and creates an operator actor for class-based explore operations.

    Instantiates the operator class from ``operator_metadata`` as a Ray actor.

    Params:
        operator_metadata: Registered metadata for the operator, carrying the class
            and canonical name.
        parameters: Dictionary of parameters to pass to the operator
        discovery_space: The discovery space the operator will operate on
        namespace: Ray namespace to create the operator actor in
        discovery_space_manager: DiscoverySpaceManager actor handle
        actuators: Dictionary of actuator actor handles keyed by actuator identifier

    Returns:
        OperatorActor handle for the created operator actor

    Raises:
        ValueError: If ``operator_metadata.cls`` is None (operator was not registered
            via the class decorator).
    """

    import ray

    import ado.utilities.output

    moduleLog.info("Creating operation")

    if operator_metadata.cls is None:
        raise ValueError(
            f"No operator class registered for '{operator_metadata.name}'. "
            "Ensure the operator is decorated with @explore_operation."
        )

    base_class = operator_metadata.cls
    actor_name = operator_metadata.name

    operator = (
        ray.remote(base_class)
        .options(name=actor_name, namespace=namespace)
        .remote(
            operationActorName=actor_name,
            namespace=namespace,
            discovery_space_manager=discovery_space_manager,
            params=parameters,
            actuators=actuators,
        )
    )

    print("=========== Operation Details ============\n")
    print(f"Space ID: {discovery_space.uri}")
    print(f"Sample Store ID:  {discovery_space.sample_store.identifier}")
    operator_reference = OperatorReference(
        operatorName=operator_metadata.name,
        operationType=operator_metadata.type,
        operatorVersion=operator_metadata.version,
    )
    conf_string = ado.utilities.output.pydantic_model_as_yaml(
        DiscoveryOperationConfiguration(
            module=operator_reference, parameters=parameters
        ),
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
