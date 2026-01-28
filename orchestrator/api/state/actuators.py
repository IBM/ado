# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT
import typing

import ray
from ray.actor import ActorHandle

from orchestrator.modules.actuators.registry import ActuatorRegistry

if typing.TYPE_CHECKING:
    from orchestrator.modules.actuators.measurement_queue import MeasurementQueue


@ray.remote
class ActuatorDictionaryActor:
    """Actor that manages a registry of actuator Ray actors.

    This actor maintains a dictionary mapping actuator identifiers to
    their corresponding Ray actor handles. The actors are lazily
    instantiated on first lookup and cached for future use.
    """

    actuators_actors: dict[str, ActorHandle]

    def __init__(self) -> None:
        self.actuators_actors = {}

    def get_actuator_actor(self, actuator_id: str) -> ActorHandle:
        """Return a Ray ActorHandle for the specified actuator.

        This function lazily creates a Ray actor for an actuator identified by
        ``actuator_id``.  If an actor for that identifier has already been
        created and cached in ``actuators_actors`` it will be returned
        directly.

        Args:
            actuator_id (str): The unique identifier for the actuator.

        Returns:
            ray.actor.ActorHandle: A handle that can be used to invoke
            methods on the underlying actuator actor.
        """
        if actuator_id not in self.actuators_actors:
            shared_queue: MeasurementQueue = ray.get_actor(
                name="QueueMonitorActor", namespace="api"
            ).get_queue.remote()
            self.actuators_actors[actuator_id] = (
                ActuatorRegistry()
                .actuatorForIdentifier(actuatorid=actuator_id)
                .options(name=actuator_id, namespace="api", get_if_exists=True)
                .remote(queue=shared_queue, params=None)
            )

        return self.actuators_actors[actuator_id]
