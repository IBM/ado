# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import typer

from orchestrator.cli.utils.output.prints import (
    ADO_SPINNER_QUERYING_DB,
    WARN,
    console_print,
    latest_identifier_for_resource_not_found,
    magenta,
    using_latest_identifier_for_resource,
)
from orchestrator.core import CoreResourceKinds
from orchestrator.metastore.project import ProjectContext


def get_effective_resource_id(
    explicit_resource_id: str | None,
    resource_type: str,
    project_context: ProjectContext,
) -> str:
    """
    Determines the effective resource ID to use, prioritizing an explicitly provided ID.

    If an explicit resource ID is provided, it takes precedence over the latest resource IDs.
    Otherwise, the method queries the database to retrieve the latest resource ID for the
    given resource type. If no ID is found, the program exits with an error.

    Args:
        explicit_resource_id (str | None): The resource ID explicitly provided by the user.
        resource_type (str): The type of resource (i.e., a cli resource type).
        project_context (ProjectContext): The project context for database access.

    Returns:
        str: The effective resource ID to use.

    Raises:
        typer.Exit: If no latest resource ID is found for the given resource type.
    """
    from rich.status import Status

    from orchestrator.cli.utils.generic.wrappers import get_sql_store

    if explicit_resource_id:
        console_print(
            f"{WARN}explicitly specified resource ids take precedence over the --use-latest flag\n"
            f"\tThis command will use {resource_type} identifier {magenta(explicit_resource_id)}"
        )
        return explicit_resource_id

    resource_kind = CoreResourceKinds(resource_type)

    # Query database for latest resource ID
    sql_store = get_sql_store(project_context)
    with Status(ADO_SPINNER_QUERYING_DB):
        latest_ids = sql_store.get_latest_resource_identifiers_of_kinds(
            kinds=[resource_kind]
        )

    resource_id = latest_ids.get(resource_kind)
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
    return resource_id
