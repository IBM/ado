# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import typer
from rich.status import Status

from ado.cli.models.parameters import AdoShowTraceCommandParameters
from ado.cli.resources.trace_common import (
    REQUEST_HIDABLE_FIELDS,
    RESULT_HIDABLE_FIELDS,
)
from ado.cli.utils.generic.wrappers import get_sql_store
from ado.cli.utils.output.prints import (
    ADO_SPINNER_QUERYING_DB,
    ERROR,
    console_print,
)
from ado.cli.utils.resources.handlers import render_trace_output
from ado.core.samplestore.base import SampleStore
from ado.core.samplestore.sql import SQLSampleStore
from ado.metastore.base import (
    NoRelatedResourcesError,
    ResourceDoesNotExistError,
)


def show_operation_trace(parameters: AdoShowTraceCommandParameters) -> None:
    """Show the measurement trace (requests and results) for an operation.

    This function provides a unified view of measurement requests and results,
    with support for filtering and multiple output formats.
    """
    hidable_fields = (
        RESULT_HIDABLE_FIELDS if parameters.unroll_entities else REQUEST_HIDABLE_FIELDS
    )

    # Validate and resolve hide_fields
    if parameters.hide_fields:
        for idx, field in enumerate(parameters.hide_fields):
            if field.lower() not in hidable_fields:
                console_print(
                    f"{ERROR}You can only hide the following fields (case insensitive): "
                    f"{list(hidable_fields.keys())}",
                    stderr=True,
                )
                raise typer.Exit(1)
            parameters.hide_fields[idx] = hidable_fields[field.lower()]

    # Get SQL store (has its own spinner)
    sql_store = get_sql_store(parameters.ado_configuration.project_context)  # type: ignore[arg-type]

    with Status(ADO_SPINNER_QUERYING_DB) as status:
        try:
            samplestore = SampleStore.from_operation_identifier(
                operation_id=parameters.resource_id,
                metastore=sql_store,  # type: ignore[arg-type]
            )
        except (ResourceDoesNotExistError, NoRelatedResourcesError):
            status.stop()
            raise

        if not isinstance(samplestore, SQLSampleStore):
            console_print(
                f"{ERROR}This command requires an SQLSampleStore",
                stderr=True,
            )
            raise typer.Exit(1)

        status.update("Fetching measurements")

        measurement_requests = samplestore.measurement_requests_for_operation(
            operation_id=parameters.resource_id,
            filters=parameters.field_selectors or None,
        )

    render_trace_output(
        measurement_requests=measurement_requests,
        parameters=parameters,
    )


# Made with Bob
