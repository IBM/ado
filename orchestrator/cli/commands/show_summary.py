# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import typing
from typing import Annotated

import typer

from orchestrator.cli.models.choice import HiddenPluralChoice
from orchestrator.cli.models.parameters import AdoShowSummaryCommandParameters
from orchestrator.cli.models.types import (
    AdoShowSummarySupportedOutputFormats,
    AdoShowSummarySupportedResourceTypes,
)
from orchestrator.cli.resources.discovery_space.show_summary import (
    show_discovery_space_summary,
)
from orchestrator.cli.utils.input.parsers import (
    parse_key_value_pairs,
)
from orchestrator.cli.utils.output.prints import (
    ERROR,
    console_print,
    latest_identifier_for_resource_not_found,
    using_latest_identifier_for_resource,
)
from orchestrator.cli.utils.queries.parser import (
    prepare_query_filters_for_db,
)
from orchestrator.core import CoreResourceKinds
from orchestrator.core.samplestore.base import (
    FailedToDecodeStoredEntityError,
    FailedToDecodeStoredMeasurementResultForEntityError,
)

if typing.TYPE_CHECKING:
    from orchestrator.cli.core.config import AdoConfiguration

TABLE_ONLY_OPTIONS = "Table-only Options"


def show_summary_for_resources(
    ctx: typer.Context,
    resource_type: Annotated[
        AdoShowSummarySupportedResourceTypes,
        typer.Argument(
            ...,
            help="The kind of the resource to show a summary for.",
            show_default=False,
            click_type=HiddenPluralChoice(AdoShowSummarySupportedResourceTypes),
        ),
    ],
    ids: Annotated[
        list[str] | None,
        typer.Argument(
            ...,
            help="The ids of the resources to show a summary for.",
            show_default=False,
        ),
    ] = None,
    use_latest: Annotated[
        bool,
        typer.Option(
            "--use-latest",
            help="Adds the latest identifier of the selected resource type to "
            "the identifiers to show a summary for.",
            show_default=False,
        ),
    ] = False,
    query: Annotated[
        list[str] | None,
        typer.Option(
            "--query",
            "-q",
            help="""
            Filter results by values contained in the resources. Will return all resources that match
            the input. Can be specified multiple times to ensure all filters are matched.

            Inputs must be specified in the form of key=value, where key is a path in the resource
            and value is a JSON document.

            Please refer to the documentation provided for more information and examples:
            https://ibm.github.io/ado/getting-started/ado/#using-the-field-level-querying-functionality
            """,
            show_default=False,
        ),
    ] = None,
    labels: Annotated[
        list[str] | None,
        typer.Option(
            "--label",
            "-l",
            help="""
            Filter results by labels contained in the resources' metadata.
            Can be specified multiple times.

            Labels need to be specified in the form of key=value.
            """,
            show_default=False,
        ),
    ] = None,
    include_properties: Annotated[
        list[str] | None,
        typer.Option(
            "--with-property",
            "-p",
            help="Add constitutive properties to the output. Can be specified multiple times.",
            show_default=False,
        ),
    ] = None,
    columns_to_hide: Annotated[
        list[str] | None,
        typer.Option(
            "--hide",
            help="Hide certain columns from the output. The following values can be used to hide default columns:"
            " id (Space ID);"
            " experiment (Experiments);"
            " matching (Matching and Measured);"
            " sampled (Sampled and Measured);"
            " name (Name);"
            " description (Description);",
            show_default=False,
            rich_help_panel=TABLE_ONLY_OPTIONS,
        ),
    ] = None,
    output_format: Annotated[
        AdoShowSummarySupportedOutputFormats,
        typer.Option(
            "--format",
            "-o",
            help="The format in which to output the summary.",
        ),
    ] = AdoShowSummarySupportedOutputFormats.TABLE.value,
    render_output: Annotated[
        bool,
        typer.Option(
            "--render",
            help="Render the output in the console. Only supported for markdown and table output.",
        ),
    ] = False,
) -> None:
    """
    Show a formatted summary of one or more discovery spaces.

    See https://ibm.github.io/ado/getting-started/ado/#ado-show-summary
    for detailed documentation and examples.



    Examples:



    # Show a high-level summary of the discovery space as a Markdown table

    ado show summary space <space-id>



    # Show a high-level summary of the latest discovery space as a Markdown table

    ado show summary space --use-latest



    # Show a high-level summary of discovery spaces matching a label

    ado show summary space -l key=value



    # Show a detailed summary of the discovery space as a Markdown document

    ado show summary space <space-id> -o md
    """
    ado_configuration: AdoConfiguration = ctx.obj

    resource_kind = CoreResourceKinds(resource_type.value)
    resource_id = ado_configuration.latest_resource_ids.get(resource_kind)
    if not resource_id:
        console_print(
            latest_identifier_for_resource_not_found(
                resource_kind=resource_kind, hide_resource_in_flag=True
            ),
            stderr=True,
        )
        raise typer.Exit(1)
    console_print(
        using_latest_identifier_for_resource(
            resource_kind=resource_kind, resource_identifier=resource_id
        ),
        stderr=True,
    )

    if ids:
        ids.append(resource_id)
    else:
        ids = [resource_id]

    try:
        query = prepare_query_filters_for_db(parse_key_value_pairs(query))
        if labels:
            for parsed_label in parse_key_value_pairs(labels):
                for k, v in parsed_label.items():
                    query.extend(
                        prepare_query_filters_for_db({"config.metadata.labels": {k: v}})
                    )
    except ValueError as e:
        console_print(f"{ERROR}{e}", stderr=True)
        raise typer.Exit(1) from e

    parameters = AdoShowSummaryCommandParameters(
        ado_configuration=ado_configuration,
        columns_to_hide=columns_to_hide,
        include_properties=include_properties,
        output_format=output_format,
        query=query,
        render_output=render_output,
        resource_ids=ids,
    )

    method_mapping = {
        AdoShowSummarySupportedResourceTypes.DISCOVERY_SPACE: show_discovery_space_summary
    }

    try:
        method_mapping[resource_type](parameters=parameters)
    except (
        FailedToDecodeStoredEntityError,
        FailedToDecodeStoredMeasurementResultForEntityError,
    ) as e:
        console_print(f"{ERROR}{e}", stderr=True)
        raise typer.Exit(1) from e


def register_show_summary_command(app: typer.Typer) -> None:
    app.command(
        name="summary",
        no_args_is_help=True,
    )(show_summary_for_resources)
