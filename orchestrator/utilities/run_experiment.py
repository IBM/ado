import argparse
from collections.abc import Callable

import yaml
from ray.actor import ActorHandle

from orchestrator.modules.actuators.base import ActuatorBase
from orchestrator.modules.actuators.measurement_queue import MeasurementQueue
from orchestrator.modules.actuators.registry import ActuatorRegistry
from orchestrator.schema.entity import Entity
from orchestrator.schema.point import SpacePoint
from orchestrator.schema.reference import ExperimentReference
from orchestrator.schema.request import MeasurementRequest
from orchestrator.schema.result import InvalidMeasurementResult


def execute_local_wrapper() -> (
    Callable[[ExperimentReference, Entity], MeasurementRequest]
):
    """Create a callable that submits a local measurement request.

    The function keeps a dictionary of Actuator actors so that each actuator
    is instantiated only once.
    """
    actuators: dict[str, ActorHandle[ActuatorBase]] = {}
    queue = MeasurementQueue.get_measurement_queue()

    def execute_local(
        reference: ExperimentReference, entity: Entity
    ) -> MeasurementRequest:
        # instantiate the actuator for this experiment identifier.
        if reference.actuatorIdentifier not in actuators:
            actuator_class = ActuatorRegistry().actuatorForIdentifier(
                reference.actuatorIdentifier
            )
            actuators[reference.actuatorIdentifier] = actuator_class(queue=queue)
        actuator = actuators[reference.actuatorIdentifier]
        # Submit the measurement request asynchronously.
        actuator.submit.remote(
            entities=[entity],
            experimentReference=reference,
            requesterid="run_experiment",
            requestIndex=0,
        )
        return queue.get()

    return execute_local


def execute_remote_wrapper(
    endpoint: str,
) -> Callable[[ExperimentReference, Entity], MeasurementRequest]:
    """Execute via ado API

    This is a placeholder that returns an ``InvalidMeasurementResult`` to
    indicate that no remote execution is actually configured.
    """

    def execute_remote(
        reference: ExperimentReference, entity: Entity
    ) -> MeasurementRequest:
        return MeasurementRequest(
            operation_id="test",
            requestIndex=0,
            experimentReference=reference,
            entities=[entity],
            measurements=(
                InvalidMeasurementResult(
                    reason="Not running remote measurements",
                    entityIdentifier=entity.identifier,
                    experimentReference=reference,
                ),
            ),
        )

    return execute_remote


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run a single experiment locally or remotely on"
    )
    parser.add_argument(
        "point_file",
        help="Path to a yaml file containing an ado point definition",
    )
    parser.add_argument(
        "--remote",
        dest="is_remote",
        action="store_true",
        default=False,
        help="Execute the experiment on a remote Ray cluster (default is local run)",
    )
    parser.add_argument(
        "--endpoint",
        default="test",
        help="Endpoint URL for remote execution; only used when --local is not set",
    )

    args = parser.parse_args(argv)

    with open(args.point_file) as f:
        point = SpacePoint.model_validate(yaml.safe_load(f))

    entity = point.to_entity()

    registry = ActuatorRegistry()
    execute = (
        execute_local_wrapper()
        if not args.is_remote
        else execute_remote_wrapper(args.endpoint)
    )

    for reference in point.experiments:
        experiment = registry.experimentForReference(reference)
        try:
            experiment.validate_entity(entity)
        except Exception as err:
            print(f"Cannot execute {reference} on {entity}: {err}")
        else:
            request = execute(experiment, entity)
            print(request.series_representation(output_format="target"))


if __name__ == "__main__":
    main()
