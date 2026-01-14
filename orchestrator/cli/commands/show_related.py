# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import typing
from typing import Annotated

import typer

from orchestrator.cli.exceptions.handlers import (
    handle_no_related_resource,
    handle_resource_does_not_exist,
)
from orchestrator.cli.models.choice import HiddenPluralChoice
from orchestrator.cli.models.parameters import AdoShowRelatedCommandParameters
from orchestrator.cli.models.types import AdoShowRelatedSupportedResourceTypes
from orchestrator.cli.resources.actuator_configuration.show_related import (
    show_resources_related_to_actuator_configuration,
)
from orchestrator.cli.resources.data_container.show_related import (
    show_resources_related_to_data_container,
)
from orchestrator.cli.resources.discovery_space.show_related import (
    show_resources_related_to_discovery_space,
)
from orchestrator.cli.resources.operation.show_related import (
    show_resources_related_to_operation,
)
from orchestrator.cli.resources.sample_store.show_related import (
    show_resources_related_to_sample_store,
)
from orchestrator.cli.utils.generic.common import get_effective_resource_id
from orchestrator.cli.utils.output.prints import ERROR, console_print
from orchestrator.metastore.base import (
    NoRelatedResourcesError,
    ResourceDoesNotExistError,
)

if typing.TYPE_CHECKING:
    from orchestrator.cli.core.config import AdoConfiguration


def show_related_for_resources(
    ctx: typer.Context,
    resource_type: Annotated[
        AdoShowRelatedSupportedResourceTypes,
        typer.Argument(
            ...,
            help="The kind of the resource to show related resources for.",
            show_default=False,
            click_type=HiddenPluralChoice(AdoShowRelatedSupportedResourceTypes),
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
) -> None:
    """
    Show resources directly (one-hop) related to the requested resource, grouped by type.

    See https://ibm.github.io/ado/getting-started/ado/#ado-show-related
    for detailed documentation and examples.



    Examples:



    # Show the resources related to a space

    ado show related space <space-id>



    # Show the resources related to the latest space

    ado show related space --use-latest
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
            ado_configuration=ado_configuration,
        )

    parameters = AdoShowRelatedCommandParameters(
        ado_configuration=ado_configuration, resource_id=resource_id
    )

    method_mapping = {
        AdoShowRelatedSupportedResourceTypes.ACTUATOR_CONFIGURATION: show_resources_related_to_actuator_configuration,
        AdoShowRelatedSupportedResourceTypes.DATA_CONTAINER: show_resources_related_to_data_container,
        AdoShowRelatedSupportedResourceTypes.DISCOVERY_SPACE: show_resources_related_to_discovery_space,
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
