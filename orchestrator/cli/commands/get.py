# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import pathlib
import typing
from typing import Annotated

import typer

from orchestrator.cli.exceptions.handlers import (
    handle_no_related_resource,
    handle_resource_does_not_exist,
)
from orchestrator.cli.models.choice import HiddenPluralChoice
from orchestrator.cli.models.parameters import AdoGetCommandParameters
from orchestrator.cli.models.types import (
    AdoGetSupportedOutputFormats,
    AdoGetSupportedResourceTypes,
)
from orchestrator.cli.resources.actuator.get import get_actuator
from orchestrator.cli.resources.actuator_configuration.get import (
    get_actuator_configuration,
)
from orchestrator.cli.resources.context.get import get_context
from orchestrator.cli.resources.data_container.get import get_data_container
from orchestrator.cli.resources.discovery_space.get import get_discovery_space
from orchestrator.cli.resources.measurement_request.get import get_measurement_request
from orchestrator.cli.resources.operation.get import get_operation
from orchestrator.cli.resources.operator.get import get_operator
from orchestrator.cli.resources.sample_store.get import get_sample_store
from orchestrator.cli.utils.input.parsers import parse_key_value_pairs
from orchestrator.cli.utils.output.prints import (
    ERROR,
    INFO,
    WARN,
    console_print,
)
from orchestrator.cli.utils.queries.parser import (
    prepare_query_filters_for_db,
)
from orchestrator.metastore.base import (
    NoRelatedResourcesError,
    ResourceDoesNotExistError,
)

if typing.TYPE_CHECKING:
    from orchestrator.cli.core.config import AdoConfiguration

OUTPUT_CONFIGURATION_OPTIONS = "Output configuration options"
ACTUATORS_ONLY_OPTIONS = "Actuators-only options"
MEASUREMENTS_ONLY_OPTIONS = "Measurements-only options"
DISCOVERY_SPACE_ONLY_OPTIONS = "Space-only options"


def get_resource(
    ctx: typer.Context,
    resource_type: Annotated[
        AdoGetSupportedResourceTypes,
        typer.Argument(
            help="The kind of the resource(s) to get.",
            show_default=False,
            click_type=HiddenPluralChoice(AdoGetSupportedResourceTypes),
        ),
    ],
    resource_id: Annotated[
        str | None,
        typer.Argument(
            help=(
                "The id of the resource to get. "
                "If unspecified, the command will return all resources of the specified type."
            ),
            show_default=False,
        ),
    ] = None,
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
    show_details: Annotated[
        bool,
        typer.Option(
            "--details",
            help="Output additional information on each object, such as names and descriptions.",
            show_default=True,
        ),
    ] = False,
    output_format: Annotated[
        AdoGetSupportedOutputFormats,
        typer.Option(
            "--output",
            "-o",
            rich_help_panel=OUTPUT_CONFIGURATION_OPTIONS,
            show_default=False,
            help="Output information in a different format. Not all formats may be supported by all resources.",
        ),
    ] = AdoGetSupportedOutputFormats.DEFAULT.value,
    exclude_default: Annotated[
        bool,
        typer.Option(
            help="Exclude/Include fields using default values in the output.",
            rich_help_panel=OUTPUT_CONFIGURATION_OPTIONS,
        ),
    ] = True,
    exclude_unset: Annotated[
        bool,
        typer.Option(
            help="Exclude/Include fields that have not been set in the output.",
            rich_help_panel=OUTPUT_CONFIGURATION_OPTIONS,
        ),
    ] = True,
    exclude_none: Annotated[
        bool,
        typer.Option(
            help="Exclude/Include fields that have a null/None value in the output.",
            rich_help_panel=OUTPUT_CONFIGURATION_OPTIONS,
        ),
    ] = True,
    exclude_fields: Annotated[
        list[str] | None,
        typer.Option(
            "--exclude-field",
            help="""
            Exclude fields from the output using JSONPath expressions.
            Only supported when the output type is yaml, json, config.

            *NOTE*: depending on the exclusions requested, this command might output
            models that are not valid if passed as input to ado create commands.
            """,
            rich_help_panel=OUTPUT_CONFIGURATION_OPTIONS,
            show_default=False,
        ),
    ] = None,
    minimize_output: Annotated[
        bool,
        typer.Option(
            "--minimize-output",
            help="""
            Make an attempt to minimize the output produced.
            This might entail applying transformations on the model, changing it from the original.

            Ignored when the output type is default or raw.
            If set, implies --exclude-default --exclude-unset --exclude-none.
            """,
            rich_help_panel=OUTPUT_CONFIGURATION_OPTIONS,
        ),
    ] = False,
    show_deprecated: Annotated[
        bool,
        typer.Option(
            "--show-deprecated",
            help="Show deprecated experiments. Ignored unless --details is provided.",
            rich_help_panel=ACTUATORS_ONLY_OPTIONS,
        ),
    ] = False,
    from_sample_store: Annotated[
        str | None,
        typer.Option(
            help="Specify the samplestore this measurement belongs to.",
            show_default=False,
            rich_help_panel=MEASUREMENTS_ONLY_OPTIONS,
        ),
    ] = None,
    from_space: Annotated[
        str | None,
        typer.Option(
            help="Specify the space this measurement belongs to.",
            show_default=False,
            rich_help_panel=MEASUREMENTS_ONLY_OPTIONS,
        ),
    ] = None,
    from_operation: Annotated[
        str | None,
        typer.Option(
            help="Specify the operation this measurement belongs to.",
            show_default=False,
            rich_help_panel=MEASUREMENTS_ONLY_OPTIONS,
        ),
    ] = None,
    matching_point: Annotated[
        pathlib.Path | None,
        typer.Option(
            help="""
            Provide a point configuration to match a space. Only for spaces.

            If set, disregards --query and --label.
            """,
            file_okay=True,
            dir_okay=False,
            exists=True,
            show_default=False,
            rich_help_panel=DISCOVERY_SPACE_ONLY_OPTIONS,
        ),
    ] = None,
    matching_space: Annotated[
        pathlib.Path | None,
        typer.Option(
            help="""
            Provide a space configuration to match other spaces. Only for spaces.

            If set, disregards --query and --label, and uses the default output format.
            """,
            file_okay=True,
            dir_okay=False,
            exists=True,
            show_default=False,
            rich_help_panel=DISCOVERY_SPACE_ONLY_OPTIONS,
        ),
    ] = None,
    matching_space_id: Annotated[
        str | None,
        typer.Option(
            help="""
            Provide a space id to match other spaces. Only for spaces.
            Takes precedence over --matching-space.

            If set, disregards --query and --label, and uses the default output format.
            """,
            show_default=False,
            rich_help_panel=DISCOVERY_SPACE_ONLY_OPTIONS,
        ),
    ] = None,
) -> None:
    """
    List, search, and retrieve representation of resources, contexts, actuators, and operators.

    See https://ibm.github.io/ado/getting-started/ado/#ado-get
    for detailed documentation and examples.



    Examples:



    # List sample stores

    ado get samplestores



    # Save the configuration of a discovery space as YAML

    ado get space <space-id> -o yaml > space.yaml



    # List actuators and the experiments they provide

    ado get actuators --details
    """
    ado_configuration: AdoConfiguration = ctx.obj

    if resource_type != AdoGetSupportedResourceTypes.DISCOVERY_SPACE and (
        matching_point or matching_space or matching_space_id
    ):
        console_print(
            f"{ERROR}--matching-point, --matching-space-id and --matching-space can only be used with "
            f"{AdoGetSupportedResourceTypes.DISCOVERY_SPACE.value}",
            stderr=True,
        )
        raise typer.Exit(1)

    if matching_space and matching_space_id:
        console_print(
            f"{WARN}--matching-space-id should not be used together with --matching-space.\n"
            f"{INFO}--matching-space will be ignored.",
            stderr=True,
        )
        matching_space = None

    if (
        matching_space or matching_space_id
    ) and output_format != AdoGetSupportedOutputFormats.DEFAULT:
        console_print(
            f"{WARN}--matching-space and --matching-space-id only support "
            f"the {AdoGetSupportedOutputFormats.DEFAULT.value} output format.",
            stderr=True,
        )
        output_format = AdoGetSupportedOutputFormats.DEFAULT

    if exclude_fields and output_format not in {
        AdoGetSupportedOutputFormats.JSON,
        AdoGetSupportedOutputFormats.YAML,
        AdoGetSupportedOutputFormats.CONFIG,
    }:
        console_print(
            f"{INFO}The {output_format.value} output format does not support excluding fields. "
            "All fields will be displayed",
            stderr=True,
        )

    if minimize_output:
        exclude_none = True
        exclude_default = True
        exclude_unset = True

    try:
        field_selectors = prepare_query_filters_for_db(parse_key_value_pairs(query))
        if labels:
            for parsed_label in parse_key_value_pairs(labels):
                for k, v in parsed_label.items():
                    field_selectors.extend(
                        prepare_query_filters_for_db({"config.metadata.labels": {k: v}})
                    )
    except ValueError as e:
        console_print(f"{ERROR}{e}", stderr=True)
        raise typer.Exit(1) from e

    parameters = AdoGetCommandParameters(
        ado_configuration=ado_configuration,
        exclude_default=exclude_default,
        exclude_fields=exclude_fields,
        exclude_none=exclude_none,
        exclude_unset=exclude_unset,
        field_selectors=field_selectors,
        from_sample_store=from_sample_store,
        from_operation=from_operation,
        from_space=from_space,
        matching_point=matching_point,
        matching_space_id=matching_space_id,
        matching_space=matching_space,
        minimize_output=minimize_output,
        output_format=output_format,
        resource_id=resource_id,
        resource_type=resource_type,
        show_deprecated=show_deprecated,
        show_details=show_details,
    )

    method_mapping = {
        AdoGetSupportedResourceTypes.ACTUATOR: get_actuator,
        AdoGetSupportedResourceTypes.ACTUATOR_CONFIGURATION: get_actuator_configuration,
        AdoGetSupportedResourceTypes.CONTEXT: get_context,
        AdoGetSupportedResourceTypes.DATA_CONTAINER: get_data_container,
        AdoGetSupportedResourceTypes.DISCOVERY_SPACE: get_discovery_space,
        AdoGetSupportedResourceTypes.SAMPLE_STORE: get_sample_store,
        AdoGetSupportedResourceTypes.MEASUREMENT_REQUEST: get_measurement_request,
        AdoGetSupportedResourceTypes.OPERATION: get_operation,
        AdoGetSupportedResourceTypes.OPERATOR: get_operator,
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


def register_get_command(app: typer.Typer) -> None:
    app.command(
        name="get",
        no_args_is_help=True,
    )(get_resource)
