# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import typer
import yaml
from rich.status import Status

from ado.cli.models.parameters import AdoDescribeCommandParameters
from ado.cli.utils.generic.wrappers import get_sql_store
from ado.cli.utils.output.prints import (
    ADO_SPINNER_QUERYING_DB,
    ERROR,
    console_print,
)
from ado.core import DiscoverySpaceResource
from ado.core.discoveryspace.config import DiscoverySpaceConfiguration
from ado.core.resources import CoreResourceKinds
from ado.metastore.base import ResourceDoesNotExistError


def describe_discovery_space(parameters: AdoDescribeCommandParameters) -> None:

    if parameters.resource_id:
        sql = get_sql_store(
            project_context=parameters.ado_configuration.project_context
        )
        with Status(ADO_SPINNER_QUERYING_DB) as status:
            space = sql.getResource(
                identifier=parameters.resource_id, kind=CoreResourceKinds.DISCOVERYSPACE
            )
            if not space:
                status.stop()
                raise ResourceDoesNotExistError(
                    resource_id=parameters.resource_id,
                    kind=CoreResourceKinds.DISCOVERYSPACE,
                )

    else:
        try:
            configuration = DiscoverySpaceConfiguration.model_validate(
                yaml.safe_load(parameters.resource_configuration.read_text())
            )
        except ValueError as e:
            console_print(
                f"{ERROR}The space configuration provided is not valid:\n{e}",
                stderr=True,
            )
            raise typer.Exit(1) from e
        except OSError as e:
            console_print(
                f"{ERROR}There was a problem while reading the space configuration provided:\n\t{e}",
                stderr=True,
            )
            raise typer.Exit(1) from e

        space = DiscoverySpaceResource(config=configuration)

    console_print(space)
