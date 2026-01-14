# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import typing
from typing import Annotated

import typer

from orchestrator.cli.exceptions.handlers import (
    handle_no_related_resource,
    handle_resource_deletion_error,
    handle_resource_does_not_exist,
)
from orchestrator.cli.models.choice import HiddenPluralChoice
from orchestrator.cli.models.parameters import AdoDeleteCommandParameters
from orchestrator.cli.models.types import AdoDeleteSupportedResourceTypes
from orchestrator.cli.resources.actuator_configuration.delete import (
    delete_actuator_configuration,
)
from orchestrator.cli.resources.context.delete import delete_context
from orchestrator.cli.resources.data_container.delete import delete_data_container
from orchestrator.cli.resources.discovery_space.delete import delete_discovery_space
from orchestrator.cli.resources.operation.delete import delete_operation
from orchestrator.cli.resources.sample_store.delete import delete_sample_store
from orchestrator.metastore.base import (
    DeleteFromDatabaseError,
    NoRelatedResourcesError,
    ResourceDoesNotExistError,
)

if typing.TYPE_CHECKING:
    from orchestrator.cli.core.config import AdoConfiguration

CONTEXT_ONLY_PANEL_NAME = "Context-only options"


def delete_resource(
    ctx: typer.Context,
    resource_type: Annotated[
        AdoDeleteSupportedResourceTypes,
        typer.Argument(
            ...,
            help="The kind of the resource to delete.",
            show_default=False,
            click_type=HiddenPluralChoice(AdoDeleteSupportedResourceTypes),
        ),
    ],
    resource_id: Annotated[
        str,
        typer.Argument(
            ...,
            help="The id of the resource to delete.",
            show_default=False,
        ),
    ],
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="""
            Force the deletion of a resource.

            Only supported when deleting sample stores that contain data or
            when deleting operations while other operations are executing.
            """,
            show_default=False,
        ),
    ] = False,
    delete_local_db: Annotated[
        bool | None,
        typer.Option(
            help="""
            Explicitly delete or keep the sqlite database file when deleting a
            local context.

            If not explicitly set, the user will be prompted for an option.
            """,
            show_default=False,
            rich_help_panel=CONTEXT_ONLY_PANEL_NAME,
        ),
    ] = None,
) -> None:
    """
    Delete resources and contexts.

    See https://ibm.github.io/ado/getting-started/ado/#ado-delete
    for detailed documentation and examples.



    Examples:



    # Delete an operation and its results

    ado delete operation <operation-id>




    # Delete a sample store that contains data

    ado delete samplestore <sample-store-id> --force



    # Delete a local context and its db

    ado delete context <context-name> --delete-local-db
    """

    ado_configuration: AdoConfiguration = ctx.obj

    parameters = AdoDeleteCommandParameters(
        ado_configuration=ado_configuration,
        delete_local_db=delete_local_db,
        force=force,
        resource_id=resource_id,
    )

    method_mapping = {
        AdoDeleteSupportedResourceTypes.ACTUATOR_CONFIGURATION: delete_actuator_configuration,
        AdoDeleteSupportedResourceTypes.CONTEXT: delete_context,
        AdoDeleteSupportedResourceTypes.DATA_CONTAINER: delete_data_container,
        AdoDeleteSupportedResourceTypes.DISCOVERY_SPACE: delete_discovery_space,
        AdoDeleteSupportedResourceTypes.SAMPLE_STORE: delete_sample_store,
        AdoDeleteSupportedResourceTypes.OPERATION: delete_operation,
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
    except DeleteFromDatabaseError as e:
        handle_resource_deletion_error(error=e)


def register_delete_command(app: typer.Typer) -> None:
    app.command(
        name="delete",
        no_args_is_help=True,
        options_metavar="[--force] [--delete-local-db] [--no-delete-local-db]",
    )(delete_resource)
