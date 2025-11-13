# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import logging
import os
import pathlib
import typing

import pydantic

from orchestrator.core.actuatorconfiguration.config import (
    ActuatorConfiguration,
)
from orchestrator.core.discoveryspace.space import DiscoverySpace
from orchestrator.core.operation.config import BaseOperationRunConfiguration
from orchestrator.modules.actuators.measurement_queue import MeasurementQueue
from orchestrator.modules.module import load_module_class_or_function
from orchestrator.utilities.logging import configure_logging

if typing.TYPE_CHECKING:
    from orchestrator.modules.actuators.base import ActuatorActor
    from orchestrator.modules.operators.base import OperatorActor

configure_logging()
moduleLog = logging.getLogger("setup")


def load_secrets_from_files(base_path: str, vars_to_load, env_var_dict):
    paths = []
    for env_var in vars_to_load:
        paths.append(f"{base_path}/{env_var}")

    for p in paths:

        if not os.path.exists(p):
            raise Exception(f"Secret {p} does not exist")

        with open(p) as f:
            env_var_dict[os.path.basename(p)] = f.readlines()[0].strip()


def load_secrets_from_env(vars_to_load, env_var_dict):
    for var in vars_to_load:

        value = os.getenv(var)
        if value is None:
            moduleLog.warning(
                f"Env variable {var} wasn't set - assuming config provided"
            )

        env_var_dict[var] = value


def setup_actuators(
    namespace: str,
    actuator_configurations: list[ActuatorConfiguration],
    discovery_space: DiscoverySpace,
    queue: MeasurementQueue,
) -> dict[str, "ActuatorActor"]:
    """
    Params:
        namespace: The namespace to set up in
        config: Configuration of the orchestrator
        queue: the update queue

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

    # First instantiate any actuators passed in actuatorConfigurations

    actuator_configurations = actuator_configurations if actuator_configurations else []
    for actuatorConfig in actuator_configurations:
        actuatorIdentifier = actuatorConfig.actuatorIdentifier
        print(f"\t{actuatorConfig.actuatorIdentifier}")

        actuator: ActuatorActor = (
            registry.actuatorForIdentifier(actuatorIdentifier)
            .options(name=actuatorIdentifier, namespace=namespace)
            .remote(queue=queue, params=actuatorConfig.parameters)
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
            queue=queue,
            params=default_actuator_parameters,
        )

        actuators[actuatorIdentifier] = actuator

    # Check that are all ready - this will raise ray.exceptions.ActorDiedError
    # if any died
    ray.get([a.ready.remote() for a in actuators.values()])

    return actuators


def setup_operator(
    base_configuration: BaseOperationRunConfiguration,
    discovery_space: DiscoverySpace,
    namespace: str,
    state,
    actuators: dict,
) -> "OperatorActor":
    """
    Params:
        actuators: List of actuators
        config: configuration dictionary
        namespace: Namespace to set up the actor in
        state: State actor handle
    """

    import orchestrator.utilities.output

    moduleLog.info("Creating operation")

    operatorClass = load_module_class_or_function(base_configuration.operation.module)
    operatorName = base_configuration.operation.module.moduleClass

    operator = operatorClass.options(name=operatorName, namespace=namespace).remote(
        operationActorName=operatorName,
        namespace=namespace,
        state=state,
        params=base_configuration.operation.parameters,
        actuators=actuators,
    )

    print("=========== Operation Details ============\n")
    print(f"Space ID: {discovery_space.uri}")
    print(f"Sample Store ID:  {discovery_space.sample_store.identifier}")
    print(
        f"Operation Configuration:\n {orchestrator.utilities.output.pydantic_model_as_yaml(base_configuration, exclude_none=True)}"
    )

    return operator


def write_entities(
    entities_output_file: str | pathlib.Path | None,
    discovery_space: DiscoverySpace,
):

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
