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
from orchestrator.cli.models.parameters import AdoEditCommandParameters
from orchestrator.cli.models.types import (
    AdoEditSupportedEditors,
    AdoEditSupportedResourceTypes,
)
from orchestrator.cli.resources.actuator_configuration.edit import (
    edit_actuator_configuration,
)
from orchestrator.cli.resources.data_container.edit import edit_data_container
from orchestrator.cli.resources.discovery_space.edit import edit_discovery_space
from orchestrator.cli.resources.operation.edit import edit_operation
from orchestrator.cli.resources.sample_store.edit import edit_sample_store
from orchestrator.metastore.base import (
    NoRelatedResourcesError,
    ResourceDoesNotExistError,
)

if typing.TYPE_CHECKING:
    from orchestrator.cli.core.config import AdoConfiguration


def edit_resource(
    ctx: typer.Context,
    resource_type: Annotated[
        AdoEditSupportedResourceTypes,
        typer.Argument(
            ...,
            help="The kind of the resource to edit metadata of.",
            show_default=False,
            click_type=HiddenPluralChoice(AdoEditSupportedResourceTypes),
        ),
    ],
    resource_id: Annotated[
        str,
        typer.Argument(
            ...,
            help="The id of the resource to edit metadata of.",
            show_default=False,
        ),
    ],
    editor: Annotated[
        AdoEditSupportedEditors,
        typer.Option(envvar="ADO_EDITOR", help="The editor to use to edit metadata"),
    ] = AdoEditSupportedEditors.NANO.value,
) -> None:
    """
    Edit resources' metadata.

    See https://ibm.github.io/ado/getting-started/ado/#ado-edit
    for detailed documentation and examples.



    Examples:



    # Edit the metadata of a sample store

    ado edit samplestore <sample-store-id>




    # Edit the metadata of a space using vim

    ado edit space --editor vim
    """
    ado_configuration: AdoConfiguration = ctx.obj
    parameters = AdoEditCommandParameters(
        ado_configuration=ado_configuration, editor=editor, resource_id=resource_id
    )

    method_mapping = {
        AdoEditSupportedResourceTypes.ACTUATOR_CONFIGURATION: edit_actuator_configuration,
        AdoEditSupportedResourceTypes.DATA_CONTAINER: edit_data_container,
        AdoEditSupportedResourceTypes.DISCOVERY_SPACE: edit_discovery_space,
        AdoEditSupportedResourceTypes.SAMPLE_STORE: edit_sample_store,
        AdoEditSupportedResourceTypes.OPERATION: edit_operation,
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


def register_edit_command(app: typer.Typer) -> None:
    app.command(name="edit", no_args_is_help=True, options_metavar="[--editor <name>]")(
        edit_resource
    )
