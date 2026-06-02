# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import typing
from typing import Annotated

import typer

from orchestrator.cli.models.parameters import (
    AdoUpgradeCommandParameters,
)
from orchestrator.cli.models.types import (
    AdoUpgradeSupportedResourceTypes,
)
from orchestrator.cli.resources.actuator_configuration.upgrade import (
    upgrade_actuator_configuration,
)
from orchestrator.cli.resources.data_container.upgrade import upgrade_data_container
from orchestrator.cli.resources.discovery_space.upgrade import upgrade_discovery_space
from orchestrator.cli.resources.operation.upgrade import upgrade_operation
from orchestrator.cli.resources.sample_store.upgrade import upgrade_sample_store
from orchestrator.cli.utils.input.parsers import enum_choice_with_plural_parser

if typing.TYPE_CHECKING:
    from orchestrator.cli.core.config import AdoConfiguration


def upgrade_resource(
    ctx: typer.Context,
    resource_type: Annotated[
        AdoUpgradeSupportedResourceTypes,
        typer.Argument(
            ...,
            help="The kind of the resource to upgrade.",
            show_default=False,
            parser=enum_choice_with_plural_parser(AdoUpgradeSupportedResourceTypes),
        ),
    ],
    apply_legacy_migrator: Annotated[
        list[str] | None,
        typer.Option(
            "--apply-legacy-migrator",
            help="Apply legacy migrators by identifier (e.g., 'samplestore_kind_entitysource_to_samplestore'). "
            "Can be specified multiple times.",
        ),
    ] = None,
    list_legacy_migrators: Annotated[
        bool,
        typer.Option(
            "--list-legacy-migrators",
            help="List available legacy migrators for this resource type",
        ),
    ] = False,
) -> None:
    """
    Upgrade resources and contexts.

    See https://ibm.github.io/ado/getting-started/ado/#ado-upgrade
    for detailed documentation and examples.

    Examples:

    # Upgrade all operations

    ado upgrade operations

    # List available legacy migrators for sample stores

    ado upgrade samplestores --list-legacy-migrators

    # Apply a legacy migrator during upgrade

    ado upgrade samplestores --apply-legacy-migrator samplestore_kind_entitysource_to_samplestore
    """

    ado_configuration: AdoConfiguration = ctx.obj

    parameters = AdoUpgradeCommandParameters(
        ado_configuration=ado_configuration,
        apply_legacy_migrator=apply_legacy_migrator,
        list_legacy_migrators=list_legacy_migrators,
    )

    method_mapping = {
        AdoUpgradeSupportedResourceTypes.ACTUATOR_CONFIGURATION: upgrade_actuator_configuration,
        AdoUpgradeSupportedResourceTypes.DATA_CONTAINER: upgrade_data_container,
        AdoUpgradeSupportedResourceTypes.DISCOVERY_SPACE: upgrade_discovery_space,
        AdoUpgradeSupportedResourceTypes.SAMPLE_STORE: upgrade_sample_store,
        AdoUpgradeSupportedResourceTypes.OPERATION: upgrade_operation,
    }

    method_mapping[resource_type](parameters=parameters)


def register_upgrade_command(app: typer.Typer) -> None:
    app.command(
        name="upgrade",
        no_args_is_help=True,
        options_metavar="",
    )(upgrade_resource)
