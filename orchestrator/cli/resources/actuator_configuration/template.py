# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import pathlib

import typer
from rich.status import Status

from orchestrator.cli.models.parameters import AdoTemplateCommandParameters
from orchestrator.cli.utils.output.prints import (
    ADO_SPINNER_INITIALIZING_ACTUATOR_REGISTRY,
    ERROR,
    HINT,
    WARN,
    console_print,
)
from orchestrator.core.actuatorconfiguration.config import ActuatorConfiguration
from orchestrator.core.metadata import ConfigurationMetadata
from orchestrator.modules.actuators.registry import (
    ActuatorRegistry,
    UnknownActuatorError,
)


def template_actuator_configuration(parameters: AdoTemplateCommandParameters) -> None:
    from orchestrator.cli.utils.pydantic.serializers import (
        serialise_pydantic_model,
        serialise_pydantic_model_json_schema,
    )

    if not parameters.actuator_identifier:
        console_print(
            f"{ERROR}You must provide an actuator identifier when templating an actuatorconfiguration.",
            stderr=True,
        )
        raise typer.Exit(1)

    with Status(ADO_SPINNER_INITIALIZING_ACTUATOR_REGISTRY):
        actuator_registry = ActuatorRegistry.globalRegistry()

    try:
        actuator = actuator_registry.actuatorForIdentifier(
            parameters.actuator_identifier
        )
    except UnknownActuatorError as e:
        console_print(
            f"{ERROR}Actuator {parameters.actuator_identifier} is not available in the registry.\n"
            f"{HINT}Registered actuators are: {list(actuator_registry.actuatorIdentifierMap.keys())}",
            stderr=True,
        )
        raise typer.Exit(1) from e

    model_instance = ActuatorConfiguration.model_construct(
        actuatorIdentifier=parameters.actuator_identifier,
        parameters=actuator.default_parameters(is_template=True),
    )

    output_path = pathlib.Path(
        f"{parameters.actuator_identifier}_{parameters.output_path}"
    )
    serialise_pydantic_model(
        model=model_instance,
        output_path=output_path,
    )

    if parameters.include_schema:
        # AP 26/06/2025:
        # model_json_schema does not care about SerializeAsAny and
        # would output GenericActuatorParameters instead of the
        # duck-typed parameters that are implemented by the actuator.
        from pydantic import create_model

        schema_output_path = pathlib.Path(output_path.stem + "_schema.yaml")
        create_model_parameters = {
            "actuatorIdentifier": (str, ...),
            "parameters": (model_instance.parameters.__class__, ...),
            "metadata": (ConfigurationMetadata, ConfigurationMetadata()),
        }
        if create_model_parameters.keys() != model_instance.model_fields.keys():
            console_print(
                f"{WARN}The {model_instance.__class__.__name__} model has changed and the schema "
                "will not be up to date. Please open an issue in our repository.",
                stderr=True,
            )

        serialise_pydantic_model_json_schema(
            model=create_model(
                model_instance.__class__.__name__, **create_model_parameters
            ).model_construct(model_instance.model_dump()),
            output_path=schema_output_path,
        )
