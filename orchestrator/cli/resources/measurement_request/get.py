# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import typing

import typer
from rich.status import Status

from orchestrator.cli.models.parameters import AdoGetCommandParameters
from orchestrator.cli.models.types import (
    AdoGetSupportedOutputFormats,
)
from orchestrator.cli.utils.generic.wrappers import get_sql_store
from orchestrator.cli.utils.output.prints import (
    ADO_SPINNER_QUERYING_DB,
    ERROR,
    HINT,
    INFO,
    WARN,
    console_print,
    cyan,
)
from orchestrator.metastore.base import (
    ResourceDoesNotExistError,
)

if typing.TYPE_CHECKING:
    from orchestrator.core.samplestore.sql import SQLSampleStore


def get_measurement_request(parameters: AdoGetCommandParameters) -> None:

    if not parameters.resource_id:
        console_print(
            f"{ERROR}You must provide the ID of a measurement request", stderr=True
        )
        raise typer.Exit(1)

    if not any(
        [parameters.from_sample_store, parameters.from_space, parameters.from_operation]
    ):
        console_print(
            f"{ERROR}You must specify either the "
            f"samplestore, the space, or the operation this measurement belongs to.\n"
            f"{HINT}Check out the available options with {cyan('ado get --help')}",
            stderr=True,
        )
        raise typer.Exit(1)

    supported_output_formats = {
        AdoGetSupportedOutputFormats.YAML,
        AdoGetSupportedOutputFormats.JSON,
    }
    if parameters.output_format not in supported_output_formats:
        console_print(
            f"{WARN}This resource only supports the following output format: "
            f"{[f.value for f in supported_output_formats]}.\n"
            f"{INFO}We will output using {AdoGetSupportedOutputFormats.YAML.value}.",
            stderr=True,
        )
        parameters.output_format = AdoGetSupportedOutputFormats.YAML

    sql = get_sql_store(project_context=parameters.ado_configuration.project_context)
    with Status(ADO_SPINNER_QUERYING_DB) as status:
        from orchestrator.core.samplestore.base import SampleStore

        sample_store: SQLSampleStore

        try:
            if parameters.from_sample_store:
                sample_store = SampleStore.from_identifier(
                    identifier=parameters.from_sample_store, metastore=sql
                )
            elif parameters.from_space:
                sample_store = SampleStore.from_space_identifier(
                    space_id=parameters.from_space, metastore=sql
                )
            else:
                sample_store = SampleStore.from_operation_identifier(
                    operation_id=parameters.from_operation, metastore=sql
                )
        except ResourceDoesNotExistError:
            status.stop()
            raise

        status.update("Retrieving your measurement")
        measurement_request = sample_store.measurement_request_by_id(
            parameters.resource_id
        )
        if not measurement_request:
            status.stop()
            raise ResourceDoesNotExistError(resource_id=parameters.resource_id)

        status.stop()

    from orchestrator.cli.utils.resources.handlers import handle_ado_get

    # Use unified handler for rendering
    handle_ado_get(parameters=parameters, resources=measurement_request)
