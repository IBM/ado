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
from orchestrator.cli.models.parameters import AdoShowEntitiesCommandParameters
from orchestrator.cli.models.types import (
    AdoShowEntitiesSupportedEntityTypes,
    AdoShowEntitiesSupportedOutputFormats,
    AdoShowEntitiesSupportedPropertyFormats,
    AdoShowEntitiesSupportedResourceTypes,
)
from orchestrator.cli.resources.discovery_space.show_entities import (
    show_discovery_space_entities,
)
from orchestrator.cli.resources.operation.show_entities import show_operation_entities
from orchestrator.cli.utils.generic.common import get_effective_resource_id
from orchestrator.cli.utils.output.prints import (
    ERROR,
    console_print,
)
from orchestrator.core.samplestore.base import (
    FailedToDecodeStoredEntityError,
    FailedToDecodeStoredMeasurementResultForEntityError,
)
from orchestrator.metastore.base import (
    NoRelatedResourcesError,
    ResourceDoesNotExistError,
)
from orchestrator.schema.virtual_property import PropertyAggregationMethodEnum

if typing.TYPE_CHECKING:
    from orchestrator.cli.core.config import AdoConfiguration

SPACE_PANEL_NAME = "Space-only options"


def show_entities_for_resources(
    ctx: typer.Context,
    resource_type: Annotated[
        AdoShowEntitiesSupportedResourceTypes,
        typer.Argument(
            ...,
            help="The kind of the resource to show entities for.",
            show_default=False,
            click_type=HiddenPluralChoice(AdoShowEntitiesSupportedResourceTypes),
        ),
    ],
    resource_id: Annotated[
        str | None,
        typer.Argument(
            ...,
            help="The id of the resource to show entities for.",
            show_default=False,
        ),
    ] = None,
    use_latest: Annotated[
        bool,
        typer.Option(
            "--use-latest",
            help="Show entities for the latest identifier of the selected resource type. "
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
        AdoShowEntitiesSupportedEntityTypes | None,
        typer.Option(
            "--include",
            help="The type of entities to include. Ignored for operations.",
            rich_help_panel=SPACE_PANEL_NAME,
        ),
    ] = AdoShowEntitiesSupportedEntityTypes.MEASURED.value,
    property_format: Annotated[
        AdoShowEntitiesSupportedPropertyFormats,
        typer.Option(
            help="The naming format to be used when displaying measured properties."
        ),
    ] = AdoShowEntitiesSupportedPropertyFormats.TARGET.value,
    output_format: Annotated[
        AdoShowEntitiesSupportedOutputFormats,
        typer.Option(help="The format in which to output the entities."),
    ] = AdoShowEntitiesSupportedOutputFormats.CONSOLE.value,
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
) -> None:
    """
    Show entities related to a space or an operation and their measurements.

    See https://ibm.github.io/ado/getting-started/ado/#ado-show-entities
    for detailed documentation and examples.



    Examples:



    # Show the entities that have been sampled in a space

    ado show entities space <space-id> --include sampled




    # Show the entities that have been sampled in the latest space
    ado show entities space --use-latest



    # Show the entities measured in an operation, one row per entity

    ado show entities operation <operation-id> --property-format target
    """
    ado_configuration: AdoConfiguration = ctx.obj

    if use_latest:
        resource_id = get_effective_resource_id(
            explicit_resource_id=resource_id,
            resource_type=resource_type.value,
            ado_configuration=ado_configuration,
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
        resource_type != AdoShowEntitiesSupportedResourceTypes.DISCOVERY_SPACE
        and not resource_id
    ):
        console_print(
            f"{ERROR}You must specify a resource id when showing entities for {resource_type.value}",
            stderr=True,
        )
        raise typer.Exit(1)

    parameters = AdoShowEntitiesCommandParameters(
        ado_configuration=ado_configuration,
        aggregation_method=aggregation_method,
        entities_output_format=output_format,
        entities_property_format=property_format,
        entities_type=entity_type,
        properties=properties,
        resource_configuration=resource_configuration,
        resource_id=resource_id,
    )

    method_mapping = {
        AdoShowEntitiesSupportedResourceTypes.DISCOVERY_SPACE: show_discovery_space_entities,
        AdoShowEntitiesSupportedResourceTypes.OPERATION: show_operation_entities,
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


def register_show_entities_command(app: typer.Typer) -> None:
    app.command(
        name="entities",
        no_args_is_help=True,
    )(show_entities_for_resources)
