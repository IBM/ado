# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

from ray.actor import ActorHandle

from orchestrator.api.state.queue import shared_queue
from orchestrator.modules.actuators.registry import ActuatorRegistry

actuators_actors: dict[str, ActorHandle] = {}


def get_actuator_actor(actuator_id: str) -> ActorHandle:

    if actuator_id not in actuators_actors:
        actuators_actors[actuator_id] = (
            ActuatorRegistry()
            .actuatorForIdentifier(actuatorid=actuator_id)
            .options(name=actuator_id, namespace="api")
            .remote(queue=shared_queue, params=None)
        )

    return actuators_actors[actuator_id]
