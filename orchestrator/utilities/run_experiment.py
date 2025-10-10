# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import logging
import os
import pathlib
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


def local_execution_closure(
    registry: ActuatorRegistry,
) -> Callable[[ExperimentReference, Entity], MeasurementRequest]:
    """Create a callable that submits a local measurement request.

    The function keeps a dictionary of Actuator actors so that each actuator
    is instantiated only once.

    Parameters:
        registry: The ActuatorRegistry to use to get the Actuator actors

    Returns:
        A callable that submits a local measurement request.
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


def remote_execution_closure(
    endpoint: str, timeout: int = 300
) -> Callable[[ExperimentReference, Entity], MeasurementRequest]:
    """Execute via ado API

    Parameters:
        endpoint: The endpoint to use to execute the experiment
        timeout: The timeout for the experiment in seconds

    Returns:
        A callable that submits a remote measurement request to the given endpoint
        with the given timeout.
    """

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
        # If the response is successful the response is a MeasurementRequest identifier
        # If the response status is 404 then the experiment was not found
        # If the response status is 422 there was a validation error
        if response.status_code == 200:
            request_id = response.json()[0]
        elif response.status_code == 404:
            raise Exception(f"Experiment {reference.experimentIdentifier} not found")
        elif response.status_code == 422:
            raise Exception(
                f"Validation error for experiment {reference.experimentIdentifier}: {response.json()}"
            )
        else:
            raise Exception(f"Unknown error {response.status_code}")
        logger.info(f"Request ID: {request_id}")

        is_completed = False
        request = None
        import datetime

        start_time = datetime.datetime.now()
        while not is_completed:
            time.sleep(2)
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
                elapsed = (datetime.datetime.now() - start_time).total_seconds()
                logger.debug(f"Waiting - {elapsed:.1f} seconds elapsed")
                if elapsed > timeout:
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


@app.command()
def run(
    point_file: pathlib.Path = typer.Argument(
        ...,
        help="Path to a yaml file containing an ado point definition",
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
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
    validate: bool = typer.Option(
        True,
        help="Validate the entity before executing the experiment. If executing remotely this requires the experiment to be installed locally",
    ),
) -> None:
    from orchestrator.modules.actuators.registry import ActuatorRegistry

    logging.getLogger().setLevel(os.environ.get("LOGLEVEL", 40))

    point = SpacePoint.model_validate(yaml.safe_load(point_file.read_text()))

    entity = point.to_entity()
    print(f"Point: {point.entity}")

    registry = ActuatorRegistry()
    execute = (
        local_execution_closure(registry=registry)
        if not remote
        else remote_execution_closure(remote, timeout=timeout)
    )

    for reference in point.experiments:
        valid = True
        if validate:
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
    try:
        app()
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    main()
