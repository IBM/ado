# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import typing
from typing import Annotated

import typer

from ado.cli.exceptions.handlers import (
    handle_no_related_resource,
    handle_resource_does_not_exist,
)
from ado.cli.models.parameters import AdoShowRelatedCommandParameters
from ado.cli.models.types import AdoShowRelatedSupportedResourceTypes
from ado.cli.resources.actuator_configuration.show_related import (
    show_resources_related_to_actuator_configuration,
)
from ado.cli.resources.data_container.show_related import (
    show_resources_related_to_data_container,
)
from ado.cli.resources.discovery_space.show_related import (
    show_resources_related_to_discovery_space,
)
from ado.cli.resources.document.show_related import (
    show_resources_related_to_document,
)
from ado.cli.resources.operation.show_related import (
    show_resources_related_to_operation,
)
from ado.cli.resources.sample_store.show_related import (
    show_resources_related_to_sample_store,
)
from ado.cli.utils.generic.common import get_effective_resource_id
from ado.cli.utils.input.parsers import enum_choice_with_plural_parser
from ado.cli.utils.output.prints import ERROR, console_print
from ado.metastore.base import (
    NoRelatedResourcesError,
    ResourceDoesNotExistError,
)
from ado.metastore.sql.statements import _MAX_HIERARCHY_HOPS

if typing.TYPE_CHECKING:
    from ado.cli.core.config import AdoConfiguration


def show_related_for_resources(
    ctx: typer.Context,
    resource_type: Annotated[
        AdoShowRelatedSupportedResourceTypes,
        typer.Argument(
            ...,
            help="The kind of the resource to show related resources for.",
            show_default=False,
            parser=enum_choice_with_plural_parser(AdoShowRelatedSupportedResourceTypes),
            metavar=f"[{'|'.join(m.value for m in AdoShowRelatedSupportedResourceTypes)}]",
        ),
    ],
    resource_id: Annotated[
        str | None,
        typer.Argument(
            help="The id of the resource to show related resources for.",
            show_default=False,
        ),
    ] = None,
    use_latest: Annotated[
        bool,
        typer.Option(
            "--use-latest",
            help="Show related resources for the latest identifier of the selected resource type. "
            "Ignored if a resource identifier is also specified.",
            show_default=False,
        ),
    ] = False,
    max_hops: Annotated[
        int | None,
        typer.Option(
            "--max-hops",
            help=f"Maximum number of relationship hops to follow from the start resource "
            f"(1-{_MAX_HIERARCHY_HOPS}). Defaults to the full graph depth.",
            show_default=False,
            min=1,
            max=_MAX_HIERARCHY_HOPS,
        ),
    ] = 1,
) -> None:
    """
    Show resources related to the requested resource, grouped by type.

    By default the full resource graph is traversed in both directions.
    Use --max-hops to limit the traversal depth.

    See https://ibm.github.io/ado/latest/cli-reference/#ado-show-related
    for detailed documentation and examples.

    Examples:

    # Show the resources related to a space
    ado show related space <space-id>

    # Show the resources related to the latest space
    ado show related space --use-latest

    # Show only directly linked (1-hop) resources
    ado show related space <space-id> --max-hops 1
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
            project_context=ado_configuration.project_context,
        )

    parameters = AdoShowRelatedCommandParameters(
        ado_configuration=ado_configuration, resource_id=resource_id, max_hops=max_hops
    )

    method_mapping = {
        AdoShowRelatedSupportedResourceTypes.ACTUATOR_CONFIGURATION: show_resources_related_to_actuator_configuration,
        AdoShowRelatedSupportedResourceTypes.DATA_CONTAINER: show_resources_related_to_data_container,
        AdoShowRelatedSupportedResourceTypes.DISCOVERY_SPACE: show_resources_related_to_discovery_space,
        AdoShowRelatedSupportedResourceTypes.DOCUMENT: show_resources_related_to_document,
        AdoShowRelatedSupportedResourceTypes.SAMPLE_STORE: show_resources_related_to_sample_store,
        AdoShowRelatedSupportedResourceTypes.OPERATION: show_resources_related_to_operation,
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


def register_show_related_command(app: typer.Typer) -> None:
    app.command(
        name="related",
        no_args_is_help=True,
        options_metavar="",
    )(show_related_for_resources)
