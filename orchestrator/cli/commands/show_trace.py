# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import pathlib
import typing
from typing import Annotated

import typer

from orchestrator.cli.exceptions.handlers import (
    handle_no_related_resource,
    handle_resource_does_not_exist,
)
from orchestrator.cli.models.parameters import AdoShowTraceCommandParameters
from orchestrator.cli.models.types import (
    AdoShowTraceSupportedOutputFormats,
    AdoShowTraceSupportedResourceTypes,
)
from orchestrator.cli.resources.discovery_space.show_trace import (
    show_discovery_space_trace,
)
from orchestrator.cli.resources.operation.show_trace import show_operation_trace
from orchestrator.cli.resources.sample_store.show_trace import show_sample_store_trace
from orchestrator.cli.utils.generic.common import get_effective_resource_id
from orchestrator.cli.utils.input.parsers import (
    enum_choice_with_plural_parser,
    parse_key_value_pairs,
)
from orchestrator.cli.utils.output.prints import ERROR, console_print
from orchestrator.cli.utils.queries.parser import prepare_query_filters_for_db
from orchestrator.metastore.base import (
    NoRelatedResourcesError,
    ResourceDoesNotExistError,
)

if typing.TYPE_CHECKING:
    from orchestrator.cli.core.config import AdoConfiguration


def show_trace_for_resources(
    ctx: typer.Context,
    resource_type: Annotated[
        AdoShowTraceSupportedResourceTypes,
        typer.Argument(
            ...,
            help="The kind of the resource to show the measurement trace for.",
            show_default=False,
            parser=enum_choice_with_plural_parser(AdoShowTraceSupportedResourceTypes),
            metavar=f"[{'|'.join(m.value for m in AdoShowTraceSupportedResourceTypes)}]",
        ),
    ],
    resource_id: Annotated[
        str | None,
        typer.Argument(
            ...,
            help="The id of the resource to show the measurement trace for.",
            show_default=False,
        ),
    ] = None,
    use_latest: Annotated[
        bool,
        typer.Option(
            "--use-latest",
            help="Show the measurement trace for the latest identifier of the selected resource type. "
            "Ignored if a resource identifier is also specified.",
            show_default=False,
        ),
    ] = False,
    unroll_entities: Annotated[
        bool,
        typer.Option(
            "--unroll-entities",
            help="Show result-level view with unrolled entities instead of request-level view.",
            show_default=False,
        ),
    ] = False,
    filters: Annotated[
        list[str] | None,
        typer.Option(
            "--filter",
            help="Filter using YAML field names from the data model. "
            "Can be used multiple times for AND logic. "
            "Only request-level filters are supported. "
            "Available fields: requestIndex, requestid, status, timestamp, metadata (use dot notation for nested fields, e.g., metadata.key=value), experimentReference.",
            show_default=False,
        ),
    ] = None,
    output_format: Annotated[
        AdoShowTraceSupportedOutputFormats,
        typer.Option(
            "--output",
            "-o",
            help="The format in which to output the trace.",
        ),
    ] = AdoShowTraceSupportedOutputFormats.TABLE,
    output_file: Annotated[
        pathlib.Path | None,
        typer.Option(
            "--output-file",
            help="Write output to the specified file instead of stdout.",
            file_okay=True,
            dir_okay=False,
            writable=True,
            resolve_path=True,
            show_default=False,
        ),
    ] = None,
    hide_fields: Annotated[
        list[str] | None,
        typer.Option(
            "--hide",
            show_default=False,
            help="""
            Hide columns (fields) from the output. Can be used multiple times.

            Different resource types might support different fields.
            """,
        ),
    ] = None,
    no_trunc: Annotated[
        bool,
        typer.Option(
            "--no-trunc",
            help="""
            Prevent truncation of table content. When enabled, columns will be sized to fit all content
            without truncation. Only applies to console output format.
            """,
        ),
    ] = False,
) -> None:
    """
    Show the measurement trace (requests and results) for a resource.

    This command provides a unified view of measurement requests and results.

    See https://ibm.github.io/ado/getting-started/ado/#ado-show-trace
    for detailed documentation and examples.

    Examples:

    # Show request-level trace for an operation
    ado show trace operation <operation-id>

    # Show request-level trace for the latest operation
    ado show trace operation --use-latest

    # Show result-level trace with unrolled entities
    ado show trace operation <operation-id> --unroll-entities

    # Multiple filters with AND logic (YAML fields)
    ado show trace operation <operation-id> --filter status=Success --filter requestIndex=5

    # Output as YAML
    ado show trace operation <operation-id> --output yaml

    # Show trace for all operations in a discovery space
    ado show trace discoveryspace <space-id>

    # Show trace for all operations that share a sample store
    ado show trace samplestore <store-id>
    """
    ado_configuration: AdoConfiguration = ctx.obj

    if not resource_id and not use_latest:
        console_print(
            f"{ERROR}You must specify either a resource id or the --use-latest flag",
            stderr=True,
        )
        raise typer.Exit(1)

    if use_latest:
        resource_id = get_effective_resource_id(
            explicit_resource_id=resource_id,
            resource_type=resource_type.value,
            project_context=ado_configuration.project_context,  # type: ignore[arg-type]
        )

    # Parse filters and prepare for DB
    field_selectors = []
    if filters:
        try:
            parsed_filters = parse_key_value_pairs(filters)
            field_selectors = prepare_query_filters_for_db(parsed_filters)
        except ValueError as e:
            console_print(f"{ERROR}{e}", stderr=True)
            raise typer.Exit(1) from e

    parameters = AdoShowTraceCommandParameters(
        ado_configuration=ado_configuration,
        field_selectors=field_selectors,  # type: ignore[arg-type]
        hide_fields=hide_fields,
        unroll_entities=unroll_entities,
        no_trunc=no_trunc,
        output_file=output_file,
        output_format=output_format,
        resource_id=resource_id,  # type: ignore[arg-type]
    )

    method_mapping = {
        AdoShowTraceSupportedResourceTypes.OPERATION: show_operation_trace,
        AdoShowTraceSupportedResourceTypes.DISCOVERY_SPACE: show_discovery_space_trace,
        AdoShowTraceSupportedResourceTypes.SAMPLE_STORE: show_sample_store_trace,
    }

    try:
        method_mapping[resource_type](parameters=parameters)
    except ResourceDoesNotExistError as e:
        handle_resource_does_not_exist(
            error=e, project_context=ado_configuration.project_context
        )
    except NoRelatedResourcesError as e:
        handle_no_related_resource(
            error=e, project_context=ado_configuration.project_context
        )


def register_show_trace_command(app: typer.Typer) -> None:
    app.command(
        name="trace",
        no_args_is_help=True,
        options_metavar="[--unroll-entities] [--filter <key=value>] [-o | --output <format>] [--output-file <path>] [--hide <column>] [--no-trunc]",
    )(show_trace_for_resources)


# Made with Bob
