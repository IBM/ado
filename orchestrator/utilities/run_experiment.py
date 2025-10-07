# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import logging
import os
import time
from collections.abc import Callable

import requests
import typer
import yaml
from ray.actor import ActorHandle

from orchestrator.modules.actuators.base import ActuatorBase
from orchestrator.modules.actuators.measurement_queue import MeasurementQueue
from orchestrator.modules.actuators.registry import ActuatorRegistry
from orchestrator.schema.entity import Entity
from orchestrator.schema.point import SpacePoint
from orchestrator.schema.reference import ExperimentReference
from orchestrator.schema.request import MeasurementRequest


def execute_local_wrapper(
    registry: ActuatorRegistry,
) -> Callable[[ExperimentReference, Entity], MeasurementRequest]:
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
        experiment = registry.experimentForReference(reference)
        if experiment.actuatorIdentifier not in actuators:
            actuator_class = ActuatorRegistry().actuatorForIdentifier(
                experiment.actuatorIdentifier
            )
            actuators[experiment.actuatorIdentifier] = actuator_class.remote(
                queue=queue
            )
        actuator = actuators[experiment.actuatorIdentifier]
        # Submit the measurement request asynchronously.
        actuator.submit.remote(
            entities=[entity],
            experimentReference=experiment.reference,
            requesterid="run_experiment",
            requestIndex=0,
        )
        return queue.get()

    return execute_local


def execute_remote_wrapper(
    endpoint: str, timeout: int = 300
) -> Callable[[ExperimentReference, Entity], MeasurementRequest]:
    """Execute via ado API"""

    import logging

    logger = logging.getLogger("remote_execution")

    def execute_remote(
        reference: ExperimentReference, entity: Entity
    ) -> MeasurementRequest | None:

        # Use requests to post to the endpoint
        # The route is /api/latest/actuators/{actuator_id}/experiments/{experiment_id}/requests
        # The body is a list of entities - [entity] to json

        response = requests.post(
            f"{endpoint}/api/latest/actuators/{reference.actuatorIdentifier}/experiments/{reference.experimentIdentifier}/requests",
            json=[entity.model_dump()],
            verify=False,
        )
        # The response is a MeasurementRequest identifier
        # We need to poll the measurement request route until the measurement request is completed
        request_id = response.json()[0]
        logger.info(f"Request ID: {request_id}")

        is_completed = False
        wait_time = 0
        request = None
        while not is_completed:
            time.sleep(5)
            logger.debug(f"Polling for request {request_id}")
            response = requests.get(
                f"{endpoint}/api/latest/actuators/{reference.actuatorIdentifier}/experiments/{reference.experimentIdentifier}/requests/{request_id}",
                verify=False,
            )
            if response.status_code == 200:
                logger.debug(response.json())
                request = MeasurementRequest.model_validate(response.json())
                is_completed = True
            else:
                logger.debug(f"Waiting - {wait_time}")
            wait_time += 2
            if wait_time > timeout:
                raise Exception(
                    f"Timeout waiting for measurement request {request_id} to complete"
                )

        return request

    return execute_remote


app = typer.Typer(
    help="Run ADO experiments locally or remotely.",
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=True,
    no_args_is_help=True,
)


@app.callback()
def main_callback(ctx: typer.Context):
    if ctx.invoked_subcommand is None and not ctx.args:
        typer.echo(ctx.get_help())
        raise typer.Exit


# Configure the typer app with the arguments
def run(
    point_file: str = typer.Argument(
        ..., help="Path to a yaml file containing an ado point definition"
    ),
    remote: str = typer.Option(
        None,
        "--remote",
        metavar="ENDPOINT",
        help="Execute the experiment on a remote Ray cluster at the given ENDPOINT. If not given the experiment will be run locally",
    ),
    timeout: int = typer.Option(
        300,
        "--timeout",
        metavar="TIMEOUT",
        help="Timeout for the remote experiment in seconds. If not given the default is 300 seconds",
    ),
    no_validate: bool = typer.Option(
        False,
        "--no-validate",
        is_flag=True,
        help="Do not validate the entity before executing the experiment. If executing remotely this requires the experiment to be installed locally",
    ),
) -> None:
    from orchestrator.modules.actuators.registry import ActuatorRegistry

    logging.getLogger().setLevel(int(os.environ.get("LOGLEVEL", 40)))

    with open(point_file) as f:
        point = SpacePoint.model_validate(yaml.safe_load(f))

    entity = point.to_entity()
    print(f"Point: {point.entity}")

    registry = ActuatorRegistry()
    execute = (
        execute_local_wrapper(registry=registry)
        if not remote
        else execute_remote_wrapper(remote, timeout=timeout)
    )

    for reference in point.experiments:
        valid = True
        if not no_validate:
            print("Validating entity ...")
            experiment = registry.experimentForReference(reference)
            valid = experiment.validate_entity(entity)
        else:
            print("Skipping validation")

        if valid:
            print(f"Executing: {reference.experimentIdentifier}")
            request = execute(reference, entity)
            print("Result:")
            print(f"{request.series_representation(output_format='target')}\n")
        else:
            print("Entity is not valid")


def main():
    app = typer.Typer(
        help="Run ADO experiments locally or remotely.",
        context_settings={"help_option_names": ["-h", "--help"]},
        add_completion=True,
        no_args_is_help=True,
    )

    app.command()(run)

    try:
        app()
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    main()
