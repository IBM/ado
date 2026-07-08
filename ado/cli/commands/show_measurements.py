# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import pathlib
import typing
from typing import Annotated

import typer

from ado.cli.exceptions.handlers import (
    handle_no_related_resource,
    handle_resource_does_not_exist,
)
from ado.cli.models.parameters import AdoShowMeasurementsCommandParameters
from ado.cli.models.types import (
    AdoShowMeasurementsSupportedEntityTypes,
    AdoShowMeasurementsSupportedOutputFormats,
    AdoShowMeasurementsSupportedPropertyFormats,
    AdoShowMeasurementsSupportedResourceTypes,
)
from ado.cli.resources.discovery_space.show_measurements import (
    show_discovery_space_measurements,
)
from ado.cli.resources.operation.show_measurements import (
    show_operation_measurements,
)
from ado.cli.utils.generic.common import get_effective_resource_id
from ado.cli.utils.input.parsers import enum_choice_with_plural_parser
from ado.cli.utils.output.prints import (
    ERROR,
    console_print,
)
from ado.core.samplestore.base import (
    FailedToDecodeStoredEntityError,
    FailedToDecodeStoredMeasurementResultForEntityError,
)
from ado.metastore.base import (
    NoRelatedResourcesError,
    ResourceDoesNotExistError,
)
from ado.schema.virtual_property import PropertyAggregationMethodEnum

if typing.TYPE_CHECKING:
    from ado.cli.core.config import AdoConfiguration

SPACE_PANEL_NAME = "Space-only options"


def show_measurements_for_resources(
    ctx: typer.Context,
    resource_type: Annotated[
        AdoShowMeasurementsSupportedResourceTypes,
        typer.Argument(
            ...,
            help="The kind of the resource to show measurements for.",
            show_default=False,
            parser=enum_choice_with_plural_parser(
                AdoShowMeasurementsSupportedResourceTypes
            ),
            metavar=f"[{'|'.join(m.value for m in AdoShowMeasurementsSupportedResourceTypes)}]",
        ),
    ],
    resource_id: Annotated[
        str | None,
        typer.Argument(
            ...,
            help="The id of the resource to show measurements for.",
            show_default=False,
        ),
    ] = None,
    use_latest: Annotated[
        bool,
        typer.Option(
            "--use-latest",
            help="Show measurements for the latest identifier of the selected resource type. "
            "Ignored if a resource identifier is also specified.",
            show_default=False,
        ),
    ] = False,
    resource_configuration: Annotated[
        pathlib.Path | None,
        typer.Option(
            "--file",
            "-f",
            help="Resource configuration details as YAML.",
            show_default=False,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ] = None,
    entity_type: Annotated[
        AdoShowMeasurementsSupportedEntityTypes | None,
        typer.Option(
            "--include",
            help="The type of entities to include. Ignored for operations.",
            rich_help_panel=SPACE_PANEL_NAME,
        ),
    ] = AdoShowMeasurementsSupportedEntityTypes.MEASURED.value,
    property_format: Annotated[
        AdoShowMeasurementsSupportedPropertyFormats,
        typer.Option(
            help="The naming format to be used when displaying measured properties."
        ),
    ] = AdoShowMeasurementsSupportedPropertyFormats.TARGET.value,
    output_format: Annotated[
        AdoShowMeasurementsSupportedOutputFormats,
        typer.Option(
            "--output",
            "-o",
            help="The format in which to output the measurements.",
        ),
    ] = AdoShowMeasurementsSupportedOutputFormats.TABLE.value,
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
    properties: Annotated[
        list[str] | None,
        typer.Option(
            "--property",
            show_default=False,
            help="Return only certain property values. Can be used multiple times.",
        ),
    ] = None,
    aggregation_method: Annotated[
        PropertyAggregationMethodEnum | None,
        typer.Option(
            "--aggregate",
            help="Aggregate the results in case of multiple values. "
            "By default, no aggregation will be applied.",
            show_default=False,
            rich_help_panel=SPACE_PANEL_NAME,
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
    Show measurements related to a space or an operation.

    See https://ibm.github.io/ado/getting-started/ado/#ado-show-measurements
    for detailed documentation and examples.

    Examples:

    # Show the measurements for entities that have been sampled in a space
    ado show measurements space <space-id> --include sampled

    # Show the measurements for entities in the latest space
    ado show measurements space --use-latest

    # Show the measurements for an operation, one row per entity
    ado show measurements operation <operation-id> --property-format target
    """
    ado_configuration: AdoConfiguration = ctx.obj

    if use_latest:
        resource_id = get_effective_resource_id(
            explicit_resource_id=resource_id,
            resource_type=resource_type.value,
            project_context=ado_configuration.project_context,
        )

    if not (resource_id or resource_configuration) or (
        resource_id and resource_configuration
    ):
        console_print(
            f"{ERROR}You must specify exactly one resource id or resource configuration",
            stderr=True,
        )
        raise typer.Exit(1)

    if (
        resource_type != AdoShowMeasurementsSupportedResourceTypes.DISCOVERY_SPACE
        and not resource_id
    ):
        console_print(
            f"{ERROR}You must specify a resource id when showing measurements for {resource_type.value}",
            stderr=True,
        )
        raise typer.Exit(1)

    parameters = AdoShowMeasurementsCommandParameters(
        ado_configuration=ado_configuration,
        aggregation_method=aggregation_method,
        measurements_output_format=output_format,
        measurements_property_format=property_format,
        measurements_type=entity_type,
        no_trunc=no_trunc,
        output_file=output_file,
        properties=properties,
        resource_configuration=resource_configuration,
        resource_id=resource_id,
    )

    method_mapping = {
        AdoShowMeasurementsSupportedResourceTypes.DISCOVERY_SPACE: show_discovery_space_measurements,
        AdoShowMeasurementsSupportedResourceTypes.OPERATION: show_operation_measurements,
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
    except (
        FailedToDecodeStoredEntityError,
        FailedToDecodeStoredMeasurementResultForEntityError,
    ) as e:
        console_print(f"{ERROR}{e}", stderr=True)
        raise typer.Exit(1) from e


def register_show_measurements_command(app: typer.Typer) -> None:
    app.command(
        name="measurements",
        no_args_is_help=True,
    )(show_measurements_for_resources)
