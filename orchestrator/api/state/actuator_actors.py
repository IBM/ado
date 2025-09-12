from ray.actor import ActorHandle

from orchestrator.modules.actuators.registry import ActuatorRegistry

actuator_registry = ActuatorRegistry.globalRegistry()
actuators_actors: dict[str, ActorHandle] = {}


def set_actuator_actor(actuator_identifier: str, actuator_actor: ActorHandle):
    if actuator_identifier not in actuators_actors:
        actuators_actors[actuator_identifier] = actuator_actor


def get_actuator_actor(actuator_identifier: str) -> ActorHandle | None:
    if actuator_identifier not in actuators_actors:
        return None

    return actuators_actors[actuator_identifier]
