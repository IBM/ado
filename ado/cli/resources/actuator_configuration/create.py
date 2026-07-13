# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import pydantic
import typer
import yaml
from rich.status import Status

from ado.cli.models.parameters import AdoCreateCommandParameters
from ado.cli.utils.generic.wrappers import get_sql_store
from ado.cli.utils.output.prints import (
    ADO_CREATE_DRY_RUN_CONFIG_VALID,
    ADO_SPINNER_SAVING_TO_DB,
    ERROR,
    SUCCESS,
    console_print,
    magenta,
)
from ado.cli.utils.pydantic.updaters import override_values_in_pydantic_model
from ado.core.actuatorconfiguration.config import ActuatorConfiguration
from ado.core.actuatorconfiguration.resource import (
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

    from ado.core.actuatorconfiguration.resource import (
        ActuatorConfigurationProvenanceInfo,
    )
    from ado.modules.actuators.registry import ActuatorRegistry

    registry = ActuatorRegistry.globalRegistry()
    actuator_provenance = registry.provenance_for_actuator(
        actuatorconfig_configuration.actuatorIdentifier
    )
    actuators = {}
    if actuator_provenance is not None:
        actuators[actuatorconfig_configuration.actuatorIdentifier] = actuator_provenance

    resource_to_be_created = ActuatorConfigurationResource(
        config=actuatorconfig_configuration,
        provenance=ActuatorConfigurationProvenanceInfo(actuators=actuators),
    )

    sql = get_sql_store(project_context=parameters.ado_configuration.project_context)
    with Status(ADO_SPINNER_SAVING_TO_DB):
        sql.addResource(resource_to_be_created)

    console_print(
        f"{SUCCESS}Created actuator configuration with identifier "
        f"{magenta(resource_to_be_created.identifier)}",
        stderr=True,
    )

    return resource_to_be_created.identifier
