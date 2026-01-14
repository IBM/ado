# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import pydantic
import typer
import yaml
from rich.status import Status

from orchestrator.cli.models.parameters import AdoCreateCommandParameters
from orchestrator.cli.utils.generic.wrappers import get_sql_store
from orchestrator.cli.utils.output.prints import (
    ADO_CREATE_DRY_RUN_CONFIG_VALID,
    ADO_SPINNER_SAVING_TO_DB,
    ERROR,
    SUCCESS,
    console_print,
    magenta,
)
from orchestrator.cli.utils.pydantic.updaters import override_values_in_pydantic_model
from orchestrator.core import CoreResourceKinds
from orchestrator.core.actuatorconfiguration.config import ActuatorConfiguration
from orchestrator.core.actuatorconfiguration.resource import (
    ActuatorConfigurationResource,
)


def create_actuator_configuration(parameters: AdoCreateCommandParameters) -> str | None:
    try:
        actuatorconfig_configuration = ActuatorConfiguration.model_validate(
            yaml.safe_load(parameters.resource_configuration_file.read_text())
        )
    except pydantic.ValidationError as error:
        console_print(
            f"{ERROR}The actuatorconfiguration provided was not valid:",
            stderr=True,
        )
        console_print(error, stderr=True, use_markup=False)
        raise typer.Exit(1) from error

    if parameters.override_values:
        actuatorconfig_configuration = override_values_in_pydantic_model(
            model=actuatorconfig_configuration,
            override_values=parameters.override_values,
        )

    if parameters.dry_run:
        console_print(ADO_CREATE_DRY_RUN_CONFIG_VALID, stderr=True)
        return None

    resource_to_be_created = ActuatorConfigurationResource(
        config=actuatorconfig_configuration
    )

    sql = get_sql_store(project_context=parameters.ado_configuration.project_context)
    with Status(ADO_SPINNER_SAVING_TO_DB):
        sql.addResource(resource_to_be_created)

    # Save the identifier of the resource we created
    # for reuse
    parameters.ado_configuration.latest_resource_ids[
        CoreResourceKinds.ACTUATORCONFIGURATION
    ] = resource_to_be_created.identifier

    console_print(
        f"{SUCCESS}Created actuator configuration with identifier "
        f"{magenta(resource_to_be_created.identifier)}",
        stderr=True,
    )

    return resource_to_be_created.identifier
