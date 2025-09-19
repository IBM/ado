# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

from ray.actor import ActorHandle

from orchestrator.api.state.queue import shared_queue
from orchestrator.modules.actuators.registry import ActuatorRegistry

# Dictionary mapping actuator identifiers to their corresponding Ray actor handles.
actuators_actors: dict[str, ActorHandle] = {}


def get_actuator_actor(actuator_id: str) -> ActorHandle:
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
    if actuator_id not in actuators_actors:
        actuators_actors[actuator_id] = (
            ActuatorRegistry()
            .actuatorForIdentifier(actuatorid=actuator_id)
            .options(name=actuator_id, namespace="api")
            .remote(queue=shared_queue, params=None)
        )

    return actuators_actors[actuator_id]
