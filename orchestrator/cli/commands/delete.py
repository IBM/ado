# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import typing
from typing import Annotated

import typer

from orchestrator.cli.models.parameters import AdoDeleteCommandParameters
from orchestrator.cli.models.types import AdoDeleteSupportedResourceTypes
from orchestrator.cli.resources.actuator_configuration.delete import (
    delete_actuator_configuration,
)
from orchestrator.cli.resources.context.delete import delete_context
from orchestrator.cli.resources.data_container.delete import delete_data_container
from orchestrator.cli.resources.discovery_space.delete import delete_discovery_space
from orchestrator.cli.resources.document.delete import delete_document
from orchestrator.cli.resources.operation.delete import delete_operation
from orchestrator.cli.resources.sample_store.delete import delete_sample_store
from orchestrator.cli.utils.input.parsers import enum_choice_with_plural_parser
from orchestrator.cli.utils.output.prints import console_print
from orchestrator.metastore.base import (
    DeleteFromDatabaseError,
    NoRelatedResourcesError,
    ResourceDoesNotExistError,
)

if typing.TYPE_CHECKING:
    from orchestrator.cli.core.config import AdoConfiguration

CONTEXT_ONLY_PANEL_NAME = "Context-only options"


def _report_deletion_results(
    resource_type: AdoDeleteSupportedResourceTypes,
    successes: list[str],
    failures: list[tuple[str, Exception]],
    ado_configuration: "AdoConfiguration",
) -> None:
    """
    Report the results of batch deletion operations.

    Args:
        resource_type: The type of resource being deleted
        successes: List of successfully deleted resource IDs
        failures: List of (resource_id, exception) tuples for failed deletions
        ado_configuration: The ado configuration
    """
    from orchestrator.cli.utils.output.prints import (
        SUCCESS,
        console_print,
        magenta,
    )

    total = len(successes) + len(failures)

    # If only one resource and it succeeded, use simple success message
    if total == 1 and len(successes) == 1:
        console_print(SUCCESS)
        return

    # Report individual results
    for resource_id in successes:
        console_print(
            f":white_check_mark: Successfully deleted: {magenta(resource_id)}"
        )

    for resource_id, error in failures:
        error_msg = str(error)
        # Extract meaningful error message
        if isinstance(error, ResourceDoesNotExistError):
            error_msg = "Resource does not exist"
        elif isinstance(error, NoRelatedResourcesError):
            error_msg = "No related resources found"
        elif isinstance(error, DeleteFromDatabaseError):
            error_msg = "Failed to delete from database"
        elif "children" in error_msg.lower():
            error_msg = "Cannot delete due to dependent resources"

        console_print(
            f":x: Failed to delete {magenta(resource_id)}: {error_msg}", stderr=True
        )

    # Summary
    if total > 1:
        console_print("\nSummary:")
        console_print(f"  - Successfully deleted {len(successes)} resource(s)")
        if failures:
            console_print(
                f"  - Failed to delete {len(failures)} resource(s)", stderr=True
            )


def delete_resource(
    ctx: typer.Context,
    resource_type: Annotated[
        AdoDeleteSupportedResourceTypes,
        typer.Argument(
            ...,
            help="The kind of the resource to delete.",
            show_default=False,
            parser=enum_choice_with_plural_parser(AdoDeleteSupportedResourceTypes),
            metavar=f"[{'|'.join(m.value for m in AdoDeleteSupportedResourceTypes)}]",
        ),
    ],
    resource_ids: Annotated[
        list[str],
        typer.Argument(
            ...,
            help="The id(s) of the resource(s) to delete. Multiple IDs can be provided.",
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

    # Delete multiple operations
    ado delete operation <op-id-1> <op-id-2> <op-id-3>

    # Delete a sample store that contains data
    ado delete samplestore <sample-store-id> --force

    # Delete a local context and its db
    ado delete context <context-name> --delete-local-db
    """

    ado_configuration: AdoConfiguration = ctx.obj

    method_mapping = {
        AdoDeleteSupportedResourceTypes.ACTUATOR_CONFIGURATION: delete_actuator_configuration,
        AdoDeleteSupportedResourceTypes.CONTEXT: delete_context,
        AdoDeleteSupportedResourceTypes.DATA_CONTAINER: delete_data_container,
        AdoDeleteSupportedResourceTypes.DISCOVERY_SPACE: delete_discovery_space,
        AdoDeleteSupportedResourceTypes.DOCUMENT: delete_document,
        AdoDeleteSupportedResourceTypes.SAMPLE_STORE: delete_sample_store,
        AdoDeleteSupportedResourceTypes.OPERATION: delete_operation,
    }

    # Process each resource ID
    successes: list[str] = []
    failures: list[tuple[str, Exception]] = []

    for resource_id in resource_ids:
        # Create parameters with single resource_id for backward compatibility
        single_params = AdoDeleteCommandParameters(
            ado_configuration=ado_configuration,
            delete_local_db=delete_local_db,
            force=force,
            resource_ids=[resource_id],
        )

        try:
            method_mapping[resource_type](parameters=single_params)
            successes.append(resource_id)
        except (
            ResourceDoesNotExistError,
            NoRelatedResourcesError,
            DeleteFromDatabaseError,
        ) as e:
            failures.append((resource_id, e))
        except typer.Exit:
            # Resource-specific delete functions may raise typer.Exit
            # Treat this as a failure for this resource
            failures.append((resource_id, Exception("Deletion failed")))
        finally:
            console_print("")

    # Report results
    _report_deletion_results(
        resource_type=resource_type,
        successes=successes,
        failures=failures,
        ado_configuration=ado_configuration,
    )

    # Exit with error if any deletions failed
    if failures:
        raise typer.Exit(1)


def register_delete_command(app: typer.Typer) -> None:
    app.command(
        name="delete",
        no_args_is_help=True,
        options_metavar="[--force] [--delete-local-db] [--no-delete-local-db]",
    )(delete_resource)
