# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import typer

from orchestrator.cli.core.config import AdoConfiguration
from orchestrator.cli.utils.output.prints import (
    WARN,
    console_print,
    latest_identifier_for_resource_not_found,
    magenta,
    using_latest_identifier_for_resource,
)
from orchestrator.core import CoreResourceKinds


def get_effective_resource_id(
    explicit_resource_id: str, resource_type: str, ado_configuration: AdoConfiguration
) -> str:
    """
    Determines the effective resource ID to use, prioritizing an explicitly provided ID.

    If an explicit resource ID is provided, it takes precedence over any configuration or flags.
    Otherwise, the method attempts to retrieve the latest resource ID from the ADO configuration
    based on the resource type. If no ID is found, the program exits with an error.

    Args:
        explicit_resource_id (str): The resource ID explicitly provided by the user.
        resource_type (str): The type of resource (i.e., a cli resource type).
        ado_configuration (AdoConfiguration): Configuration object containing latest resource IDs.

    Returns:
        str: The effective resource ID to use.

    Raises:
        typer.Exit: If no latest resource ID is found for the given resource type.
    """

    if explicit_resource_id:
        console_print(
            f"{WARN}explicitly specified resource ids take precedence over the --use-latest flag\n\t"
            f"The command will show details for {resource_type} {magenta(explicit_resource_id)}"
        )
        return explicit_resource_id

    resource_kind = CoreResourceKinds(resource_type)
    resource_id = ado_configuration.latest_resource_ids.get(resource_kind)
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
