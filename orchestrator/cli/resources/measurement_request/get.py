# Copyright (c) IBM Corporation
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
    ADO_SPINNER_GETTING_OUTPUT_READY,
    ADO_SPINNER_QUERYING_DB,
    ERROR,
    HINT,
    INFO,
    WARN,
    console_print,
    cyan,
)
from orchestrator.cli.utils.resources.formatters import (
    format_resource_for_ado_get_custom_format,
)
from orchestrator.core.discoveryspace.space import DiscoverySpace
from orchestrator.core.resources import CoreResourceKinds
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
        sample_store: SQLSampleStore

        if parameters.from_sample_store:
            from orchestrator.core.samplestore.utils import (
                load_sample_store_from_resource,
            )

            resource = sql.getResource(
                identifier=parameters.from_sample_store,
                kind=CoreResourceKinds.SAMPLESTORE,
            )
            if not resource:
                status.stop()
                raise ResourceDoesNotExistError(
                    resource_id=parameters.from_sample_store,
                    kind=CoreResourceKinds.SAMPLESTORE,
                )
            sample_store = load_sample_store_from_resource(resource)

        elif parameters.from_space:
            sample_store = DiscoverySpace.from_stored_configuration(
                project_context=parameters.ado_configuration.project_context,
                space_identifier=parameters.from_space,
            ).sample_store

        else:
            sample_store = DiscoverySpace.from_operation_id(
                operation_id=parameters.from_operation,
                project_context=parameters.ado_configuration.project_context,
            ).sample_store

        status.update("Retrieving your measurement")
        measurement_request = sample_store.measurement_request_by_id(
            parameters.resource_id
        )
        if not measurement_request:
            status.stop()
            raise ResourceDoesNotExistError(resource_id=parameters.resource_id)

        status.update(ADO_SPINNER_GETTING_OUTPUT_READY)
        console_print(
            format_resource_for_ado_get_custom_format(
                to_print=measurement_request, parameters=parameters
            )
        )
