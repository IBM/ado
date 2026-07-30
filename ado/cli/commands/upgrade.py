# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import typing
from typing import Annotated

import typer

from ado.cli.models.parameters import (
    AdoUpgradeCommandParameters,
)
from ado.cli.models.types import (
    AdoUpgradeSupportedResourceTypes,
)
from ado.cli.resources.actuator_configuration.upgrade import (
    upgrade_actuator_configuration,
)
from ado.cli.resources.data_container.upgrade import upgrade_data_container
from ado.cli.resources.discovery_space.upgrade import upgrade_discovery_space
from ado.cli.resources.operation.upgrade import upgrade_operation
from ado.cli.resources.sample_store.upgrade import upgrade_sample_store
from ado.cli.utils.input.parsers import enum_choice_with_plural_parser

if typing.TYPE_CHECKING:
    from ado.cli.core.config import AdoConfiguration


def upgrade_resource(
    ctx: typer.Context,
    resource_type: Annotated[
        AdoUpgradeSupportedResourceTypes,
        typer.Argument(
            ...,
            help="The kind of the resource to upgrade.",
            show_default=False,
            parser=enum_choice_with_plural_parser(AdoUpgradeSupportedResourceTypes),
            metavar=f"[{'|'.join(m.value for m in AdoUpgradeSupportedResourceTypes)}]",
        ),
    ],
    upgrade_entities_and_results: Annotated[
        bool,
        typer.Option(
            "--upgrade-entities-and-results",
            help=(
                "Also upgrade stored entities and measurement results in each sample store. "
                "Only applies to the samplestore resource type. "
                "This can take a long time for large stores."
            ),
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
    """

    ado_configuration: AdoConfiguration = ctx.obj

    parameters = AdoUpgradeCommandParameters(
        ado_configuration=ado_configuration,
        upgrade_entities_and_results=upgrade_entities_and_results,
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
