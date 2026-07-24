# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import pathlib
import typing
from typing import Annotated

import typer

from ado.cli.exceptions.handlers import (
    handle_no_related_resource,
    handle_resource_does_not_exist,
    handle_unknown_experiment_error,
)
from ado.cli.models.parameters import AdoDescribeCommandParameters
from ado.cli.models.types import AdoDescribeSupportedResourceTypes
from ado.cli.resources.data_container.describe import describe_data_container
from ado.cli.resources.discovery_space.describe import describe_discovery_space
from ado.cli.resources.document.describe import describe_document
from ado.cli.resources.experiment.describe import describe_experiment
from ado.cli.utils.generic.common import get_effective_resource_id
from ado.cli.utils.input.parsers import enum_choice_with_plural_parser
from ado.cli.utils.output.prints import (
    ERROR,
    console_print,
    cyan,
)
from ado.metastore.base import (
    NoRelatedResourcesError,
    ResourceDoesNotExistError,
)
from ado.modules.actuators.errors import (
    UnknownActuatorError,
    UnknownExperimentError,
)

if typing.TYPE_CHECKING:
    from ado.cli.core.config import AdoConfiguration


def describe_resource(
    ctx: typer.Context,
    resource_type: Annotated[
        AdoDescribeSupportedResourceTypes,
        typer.Argument(
            help="The kind of the resource to describe.",
            show_default=False,
            parser=enum_choice_with_plural_parser(AdoDescribeSupportedResourceTypes),
            metavar=f"[{'|'.join(m.value for m in AdoDescribeSupportedResourceTypes)}]",
        ),
    ],
    resource_id: Annotated[
        str | None,
        typer.Argument(
            ...,
            help="The id of the resource to describe.",
            show_default=False,
        ),
    ] = None,
    use_latest: Annotated[
        bool,
        typer.Option(
            "--use-latest",
            help="Describe the resource using the latest identifier created. "
            "Not supported for experiments. "
            "Ignored if a resource identifier is also specified.",
            show_default=False,
        ),
    ] = False,
    resource_configuration: Annotated[
        pathlib.Path | None,
        typer.Option(
            "--file",
            "-f",
            help="Resource configuration details as YAML. Supported only for spaces.",
            show_default=False,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ] = None,
) -> None:
    """
    Print a human-friendly description of a resource or an experiment.

    See https://ibm.github.io/ado/getting-started/ado/#ado-describe
    for detailed documentation and examples.

    Examples:

    # Describe an existing space
    ado describe space <space-id>

    # Describe a space from a space configuration file
    ado describe space -f <space.yaml>

    # Describe an experiment, optionally with actuator prefix
    ado describe experiment <actuator-id>.<experiment-id>

    # Describe a document (markdown in-terminal; HTML opens in browser)
    ado describe document <document-id>
    """
    ado_configuration: AdoConfiguration = ctx.obj

    if use_latest:
        if resource_type == AdoDescribeSupportedResourceTypes.EXPERIMENT:
            console_print(
                f"{ERROR}The {cyan('--use-latest')} flag is not supported for experiments.",
                stderr=True,
            )
            raise typer.Exit(1)

        resource_id = get_effective_resource_id(
            explicit_resource_id=resource_id,
            resource_type=resource_type.value,
            project_context=ado_configuration.project_context,
        )

    if not (resource_id or resource_configuration) or (
        resource_id and resource_configuration
    ):
        console_print(
            f"{ERROR}You must specify exactly one resource id or resource configuration",
            stderr=True,
        )
        raise typer.Exit(1)

    if (
        resource_type != AdoDescribeSupportedResourceTypes.DISCOVERY_SPACE
        and not resource_id
    ):
        console_print(
            f"{ERROR}You must specify a resource id when describing a {resource_type.value}",
            stderr=True,
        )
        raise typer.Exit(1)

    parameters = AdoDescribeCommandParameters(
        ado_configuration=ado_configuration,
        resource_id=resource_id,
        resource_configuration=resource_configuration,
    )

    method_mapping = {
        AdoDescribeSupportedResourceTypes.DATA_CONTAINER: describe_data_container,
        AdoDescribeSupportedResourceTypes.DISCOVERY_SPACE: describe_discovery_space,
        AdoDescribeSupportedResourceTypes.DOCUMENT: describe_document,
        AdoDescribeSupportedResourceTypes.EXPERIMENT: describe_experiment,
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
    except UnknownActuatorError as e:
        console_print(f"{ERROR}{e}", stderr=True)
        raise typer.Exit(1) from e
    except UnknownExperimentError as e:
        handle_unknown_experiment_error(error=e)


def register_describe_command(app: typer.Typer) -> None:
    app.command(
        name="describe",
        no_args_is_help=True,
        options_metavar="[-f | --file <file.yaml>]",
    )(describe_resource)
