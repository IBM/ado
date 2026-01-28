# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import logging
import os
import pathlib
import time
import typing
from collections.abc import Callable
from typing import Annotated

import ray.exceptions
import requests
import typer
import yaml

from orchestrator.cli.utils.output.prints import ERROR, WARN, console_print
from orchestrator.modules.actuators.measurement_queue import MeasurementQueue
from orchestrator.modules.actuators.registry import ActuatorRegistry
from orchestrator.modules.operators._cleanup import (
    initialize_ray_resource_cleaner,
)
from orchestrator.modules.operators.orchestrate import graceful_orchestrate_shutdown
from orchestrator.schema.entity import Entity
from orchestrator.schema.point import SpacePoint
from orchestrator.schema.reference import ExperimentReference
from orchestrator.schema.request import MeasurementRequest

if typing.TYPE_CHECKING:
    from ray.actor import ActorHandle

    from orchestrator.modules.actuators.base import ActuatorBase


def local_execution_closure(
    registry: ActuatorRegistry,
    actuator_configuration_identifiers: list[str] | None = None,
) -> Callable[[ExperimentReference, Entity], MeasurementRequest] | None:
    """Create a callable that submits a local measurement request.

    The function keeps a dictionary of Actuator actors so that each actuator
    is instantiated only once.

    Parameters:
        registry: The ActuatorRegistry to use to get the Actuator actors
        actuator_configuration_identifiers: (Optional) the actuator configuration to use

    Returns:
        A callable that submits a local measurement request.
    """
    actuators: dict[str, ActorHandle[ActuatorBase]] = {}
    queue = MeasurementQueue()

    actuator_configurations = {}
    if actuator_configuration_identifiers:
        from orchestrator.cli.core.config import AdoConfiguration
        from orchestrator.core.resources import CoreResourceKinds
        from orchestrator.metastore.sqlstore import SQLStore

        # get the metastore instance for current project context
        ado_config = AdoConfiguration.load()
        metastore = SQLStore(project_context=ado_config.project_context)
        for actuator_configuration_identifier in actuator_configuration_identifiers:
            actuator_configuration = metastore.getResource(
                identifier=actuator_configuration_identifier,
                kind=CoreResourceKinds.ACTUATORCONFIGURATION,
                raise_error_if_no_resource=True,
            ).config
            actuator_configurations[actuator_configuration.actuatorIdentifier] = (
                actuator_configuration
            )
            console_print(
                f"Loaded configuration {actuator_configuration_identifier} for actuator {actuator_configuration.actuatorIdentifier}"
            )

    def execute_local(
        reference: ExperimentReference, entity: Entity
    ) -> MeasurementRequest | None:
        # instantiate the actuator for this experiment identifier.
        if reference.actuatorIdentifier not in actuators:
            actuator_class = registry.actuatorForIdentifier(
                reference.actuatorIdentifier
            )
            if reference.actuatorIdentifier in actuator_configurations:
                config = actuator_configurations[
                    reference.actuatorIdentifier
                ].parameters
            else:
                config = actuator_class.default_parameters()

            actuators[reference.actuatorIdentifier] = actuator_class.remote(
                queue=queue, params=config
            )
        actuator = actuators[reference.actuatorIdentifier]
        # Submit the measurement request asynchronously, handle errors gracefully.
        try:
            ray.get(actuator.ready.remote())
            future = actuator.submit.remote(
                entities=[entity],
                experimentReference=reference,
                requesterid="run_experiment",
                requestIndex=0,
            )
            _ = ray.get(future)
        except ray.exceptions.ActorDiedError as error:
            console_print(
                f"{ERROR}Failed to initialize actuator '{reference.actuatorIdentifier}': {error}",
                stderr=True,
            )
            return None
        except ray.exceptions.RayTaskError as error:
            e = error.as_instanceof_cause()
            console_print(
                f"{ERROR}Failed to submit measurement request for {reference} to actuator '{reference.actuatorIdentifier}':\n {e}",
                stderr=True,
            )
            # Either skip, or return None, or propagate. Let's return None.
            return None

        return queue.get()

    return execute_local


def remote_execution_closure(
    endpoint: str,
    experiment_timeout: int = 300,
    verify_certs: bool = False,
    requests_timeout: int = 60,
) -> Callable[[ExperimentReference, Entity], MeasurementRequest]:
    """Execute via ado API

    Parameters:
        endpoint: The endpoint to use to execute the experiment
        experiment_timeout: The timeout for the experiment in seconds
        verify_certs: Enables or disables SSL certificate verification for web requests
        requests_timeout: Timeout for web requests

    Returns:
        A callable that submits a remote measurement request to the given endpoint
        with the given timeout.
    """

    logger = logging.getLogger("remote_execution")

    def execute_remote(
        reference: ExperimentReference,
        entity: Entity,
        verify_certs: bool,
        requests_timeout: int,
    ) -> MeasurementRequest | None:

        # Use requests to post to the endpoint
        # The route is /api/latest/actuators/{actuator_id}/experiments/{experiment_id}/requests
        # The body is a list of entities - [entity] to json

        response = requests.post(
            f"{endpoint}/api/latest/actuators/{reference.actuatorIdentifier}/experiments/{reference.experimentIdentifier}/requests",
            json=[entity.model_dump()],
            verify=verify_certs,
            timeout=requests_timeout,
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
                verify=verify_certs,
                timeout=requests_timeout,
            )
            if response.status_code == 200:
                logger.debug(response.json())
                request = MeasurementRequest.model_validate(response.json())
                is_completed = True
            else:
                elapsed = (datetime.datetime.now() - start_time).total_seconds()
                logger.debug(f"Waiting - {elapsed:.1f} seconds elapsed")
                if elapsed > experiment_timeout:
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
    point_file: Annotated[
        pathlib.Path,
        typer.Argument(
            help="Path to a yaml file containing an ado point definition",
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ],
    remote: Annotated[
        str | None,
        typer.Option(
            metavar="ENDPOINT",
            help="Execute the experiment on a remote Ray cluster at the given ENDPOINT. If not given the experiment will be run locally",
        ),
    ] = None,
    timeout: Annotated[
        int,
        typer.Option(
            metavar="TIMEOUT",
            help="Timeout for the remote experiment in seconds. If not given the default is 300 seconds",
        ),
    ] = 300,
    validate: Annotated[
        bool,
        typer.Option(
            help="Validate the entity before executing the experiment. "
            "If executing remotely this requires the experiment to be installed locally",
        ),
    ] = True,
    actuator_configuration_identifiers: Annotated[
        list[str] | None,
        typer.Option(
            "--actuator-config-id",
            metavar="ACTUATOR_CONFIG_IDENTIFIER",
            help="Optional actuator configuration identifier(s) to use for this experiment. "
            "May be specified multiple times.",
        ),
    ] = None,
    verify_certs: Annotated[
        bool,
        typer.Option(
            help="Enable or disable SSL certificate verification of remote hosts"
        ),
    ] = False,
    request_timeout: Annotated[
        int, typer.Option(help="Timeout for web requests.")
    ] = 60,
) -> None:
    from orchestrator.modules.actuators.registry import ActuatorRegistry

    logging.getLogger().setLevel(os.environ.get("LOGLEVEL", 40))

    point = SpacePoint.model_validate(yaml.safe_load(point_file.read_text()))

    entity = point.to_entity()
    console_print(f"Point: {point.entity}")

    registry = ActuatorRegistry()
    execute = (
        local_execution_closure(
            registry=registry,
            actuator_configuration_identifiers=actuator_configuration_identifiers,
        )
        if not remote
        else remote_execution_closure(
            remote,
            experiment_timeout=timeout,
            verify_certs=verify_certs,
            requests_timeout=request_timeout,
        )
    )

    if not remote:
        initialize_ray_resource_cleaner()

    try:
        for reference in point.experiments:
            valid = True
            if validate:
                console_print("Validating entity ...")
                experiment = registry.experimentForReference(reference)
                valid = experiment.validate_entity(entity, verbose=True)
            else:
                console_print("Skipping validation")

            if valid:
                console_print(f"Executing: {reference}")
                request = execute(reference, entity)
                if request is None:
                    console_print(
                        f"{WARN}Measurement request failed unexpectedly. Skipping this experiment."
                    )
                else:
                    console_print("Result:")
                    console_print(
                        f"{request.series_representation(output_format='target')}\n",
                        has_pandas_content=True,
                        use_markup=False,
                    )
            else:
                console_print(f"{ERROR}Entity is not valid")
    finally:
        if not remote:
            graceful_orchestrate_shutdown()


def main() -> None:
    try:
        app()
    except Exception as e:
        console_print(f"{ERROR}{e}", stderr=True)
        raise typer.Exit(code=1) from e


if __name__ == "__main__":
    main()
