# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import typing
from typing import Annotated

import typer

from ado.cli.models.parameters import AdoDeleteCommandParameters
from ado.cli.models.types import AdoDeleteSupportedResourceTypes
from ado.cli.resources.actuator_configuration.delete import (
    delete_actuator_configuration,
)
from ado.cli.resources.context.delete import delete_context
from ado.cli.resources.data_container.delete import delete_data_container
from ado.cli.resources.discovery_space.delete import delete_discovery_space
from ado.cli.resources.document.delete import delete_document
from ado.cli.resources.operation.delete import delete_operation
from ado.cli.resources.sample_store.delete import delete_sample_store
from ado.cli.utils.input.parsers import enum_choice_with_plural_parser
from ado.cli.utils.output.prints import (
    cannot_delete_resource_due_to_children_resources,
    console_print,
    context_not_in_available_contexts_error_str,
    could_not_delete_resource_from_database_error_str,
)
from ado.metastore.base import (
    ContextDoesNotExistError,
    DeleteFromDatabaseError,
    NonEmptySampleStorePreventingDeletionError,
    NoRelatedResourcesError,
    NotSupportedOnSQLiteError,
    ResourceDoesNotExistError,
    ResourceHasChildrenError,
    RunningOperationsPreventingDeletionError,
)

if typing.TYPE_CHECKING:
    from ado.cli.core.config import AdoConfiguration

CONTEXT_ONLY_PANEL_NAME = "Context-only options"


def _deletion_error_message(resource_id: str, error: Exception) -> str:
    from ado.cli.utils.output.prints import (
        ERROR,
        HINT,
        cyan,
        magenta,
    )

    if isinstance(error, ResourceDoesNotExistError):
        return "Resource does not exist"
    if isinstance(error, ContextDoesNotExistError):
        return context_not_in_available_contexts_error_str(
            requested_context=error.resource_id,
            available_contexts=error.available_contexts,
        )
    if isinstance(error, NoRelatedResourcesError):
        return "No related resources found"
    if isinstance(error, ResourceHasChildrenError):
        return cannot_delete_resource_due_to_children_resources(
            resource_kind=error.kind,
            resource_id=error.resource_id,
            children_resources=error.children_resources,
        )
    if isinstance(error, NotSupportedOnSQLiteError):
        return (
            f"{ERROR}Checking for running operations using the same sample store as "
            f"operation {magenta(resource_id)} is not supported on local contexts.\n"
            f"{HINT}Make sure there are no such operations, and force the deletion by adding the "
            f"{cyan('--force')} flag."
        )
    if isinstance(error, RunningOperationsPreventingDeletionError):
        return (
            f"{ERROR}Operation {magenta(error.operation_id)} cannot be deleted "
            f"because the following operations have started and have not completed: "
            f"{error.running_operations}\n"
            f"{HINT}You can force the deletion by adding the {cyan('--force')} flag."
        )
    if isinstance(error, NonEmptySampleStorePreventingDeletionError):
        return (
            f"{ERROR}{error}\n"
            f"{HINT}You can force the deletion by adding the {cyan('--force')} flag."
        )
    if isinstance(error, DeleteFromDatabaseError):
        return could_not_delete_resource_from_database_error_str(error)
    return str(error)


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
    from ado.cli.utils.output.prints import (
        SUCCESS,
        magenta,
    )

    total = len(successes) + len(failures)

    # If only one resource, avoid batch-style summary output.
    if total == 1:
        if successes:
            console_print(SUCCESS)
        elif failures:
            resource_id, error = failures[0]
            console_print(
                f"Failed to delete {resource_id}: "
                f"{_deletion_error_message(resource_id=resource_id, error=error)}",
                stderr=True,
            )
        return

    # Report individual results
    for resource_id in successes:
        console_print(
            f":white_check_mark: Successfully deleted: {magenta(resource_id)}"
        )

    for resource_id, error in failures:
        console_print(
            f"Failed to delete {resource_id}: "
            f"{_deletion_error_message(resource_id=resource_id, error=error)}",
            stderr=True,
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
            ContextDoesNotExistError,
            ResourceDoesNotExistError,
            ResourceHasChildrenError,
            NoRelatedResourcesError,
            NotSupportedOnSQLiteError,
            RunningOperationsPreventingDeletionError,
            NonEmptySampleStorePreventingDeletionError,
            DeleteFromDatabaseError,
        ) as e:
            failures.append((resource_id, e))
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
