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
from ado.core.resources import CoreResourceKinds
from ado.core.samplestore.base import SampleStore
from ado.core.samplestore.sql import SQLSampleStore
from ado.metastore.base import (
    NoRelatedResourcesError,
    ResourceDoesNotExistError,
)


def show_sample_store_trace(parameters: AdoShowTraceCommandParameters) -> None:
    """Show the measurement trace for all operations linked to a sample store.

    Aggregates measurement requests from every operation into a single flat
    table.
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

    store_id = parameters.resource_id

    sql_store = get_sql_store(parameters.ado_configuration.project_context)  # type: ignore[arg-type]

    with Status(ADO_SPINNER_QUERYING_DB) as status:
        # Verify the samplestore exists
        if not sql_store.containsResourceWithIdentifier(
            identifier=store_id, kind=CoreResourceKinds.SAMPLESTORE
        ):
            status.stop()
            raise ResourceDoesNotExistError(
                resource_id=store_id, kind=CoreResourceKinds.SAMPLESTORE
            )

        status.update("Resolving spaces and operations")

        # Hop 1: samplestore → discoveryspaces.
        related = sql_store.get_resources_by_relationship(
            kind=CoreResourceKinds.SAMPLESTORE,
            identifier=store_id,
            relationship="child",
            result_kinds={CoreResourceKinds.DISCOVERYSPACE},
            max_hops=1,
            identifiers_only=True,
        )
        space_ids: set[str] = related.get(CoreResourceKinds.DISCOVERYSPACE, set())  # type: ignore[union-attr]

        if not space_ids:
            status.stop()
            raise NoRelatedResourcesError(
                resource_id=store_id, kind=CoreResourceKinds.SAMPLESTORE
            )

        # Hop 2: each space → its operations, building operation_id → space_id map.
        # get_resources_by_relationship with a set identifier returns
        # {space_id: {OPERATION: {op_ids, ...}, ...}, ...}
        related_spaces: dict = sql_store.get_resources_by_relationship(  # type: ignore[assignment]
            kind=CoreResourceKinds.DISCOVERYSPACE,
            identifier=space_ids,
            relationship="child",
            result_kinds={CoreResourceKinds.OPERATION},
            max_hops=1,
            identifiers_only=True,
        )
        operation_space_map: dict[str, str] = {
            operation_id: space_id
            for space_id, related_resources_by_kind in related_spaces.items()
            for operation_id in related_resources_by_kind.get(
                CoreResourceKinds.OPERATION, set()
            )
        }

        if not operation_space_map:
            status.stop()
            raise NoRelatedResourcesError(
                resource_id=store_id, kind=CoreResourceKinds.SAMPLESTORE
            )

        operation_ids = set(operation_space_map.keys())

        status.update("Loading sample store")

        try:
            samplestore = SampleStore.from_identifier(
                identifier=store_id,
                metastore=sql_store,  # type: ignore[arg-type]
            )
        except ResourceDoesNotExistError:
            status.stop()
            raise

        if not isinstance(samplestore, SQLSampleStore):
            console_print(
                f"{ERROR}This command requires an SQLSampleStore",
                stderr=True,
            )
            raise typer.Exit(1)

        status.update("Fetching measurements")

        measurement_requests_by_op = samplestore.measurement_requests_for_operation(
            operation_id=operation_ids,
            filters=parameters.field_selectors or None,
        )

    # Flatten the per-operation dict into a single list
    all_requests = []
    for requests in measurement_requests_by_op.values():  # type: ignore[union-attr]
        all_requests.extend(requests)

    render_trace_output(
        measurement_requests=all_requests,
        parameters=parameters,
        include_operation_id=True,
        operation_space_map=operation_space_map,
    )


# Made with Bob
