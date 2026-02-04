# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import pathlib
import typing
from typing import Annotated

import typer

from orchestrator.cli.exceptions.handlers import (
    handle_no_related_resource,
    handle_resource_does_not_exist,
    handle_unknown_experiment_error,
)
from orchestrator.cli.models.choice import HiddenPluralChoice
from orchestrator.cli.models.parameters import AdoDescribeCommandParameters
from orchestrator.cli.models.types import AdoDescribeSupportedResourceTypes
from orchestrator.cli.resources.data_container.describe import describe_data_container
from orchestrator.cli.resources.discovery_space.describe import describe_discovery_space
from orchestrator.cli.resources.experiment.describe import describe_experiment
from orchestrator.cli.utils.generic.common import get_effective_resource_id
from orchestrator.cli.utils.output.prints import (
    ERROR,
    console_print,
    cyan,
)
from orchestrator.metastore.base import (
    NoRelatedResourcesError,
    ResourceDoesNotExistError,
)
from orchestrator.modules.actuators.registry import (
    UnknownActuatorError,
    UnknownExperimentError,
)

if typing.TYPE_CHECKING:
    from orchestrator.cli.core.config import AdoConfiguration

EXPERIMENT_ONLY_OPTIONS = "Experiment-only options"
SPACE_ONLY_OPTIONS = "Space-only options"


def describe_resource(
    ctx: typer.Context,
    resource_type: Annotated[
        AdoDescribeSupportedResourceTypes,
        typer.Argument(
            help="The kind of the resource to describe.",
            show_default=False,
            click_type=HiddenPluralChoice(AdoDescribeSupportedResourceTypes),
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
            help="Describe the space using the latest space identifier created. "
            "Ignored if a resource identifier is also specified.",
            rich_help_panel=SPACE_ONLY_OPTIONS,
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
    actuator_id: Annotated[
        str | None,
        typer.Option(
            help="""
            The id of the actuator that provides the experiment. Auto-determined if not provided.

            Only used when resource_type is experiment.""",
            show_default=False,
            rich_help_panel=EXPERIMENT_ONLY_OPTIONS,
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



    # Describe an experiment and explicitly specify the actuator id

    ado describe experiment <experiment-id> --actuator-id <actuator-id>
    """
    ado_configuration: AdoConfiguration = ctx.obj

    if use_latest:

        if resource_type != AdoDescribeSupportedResourceTypes.DISCOVERY_SPACE:
            console_print(
                f"{ERROR}The {cyan('--use-latest')} flag is available only for spaces.",
                stderr=True,
            )
            raise typer.Exit(1)

        resource_id = get_effective_resource_id(
            explicit_resource_id=resource_id,
            resource_type=resource_type.value,
            ado_configuration=ado_configuration,
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
        actuator_id=actuator_id,
        ado_configuration=ado_configuration,
        resource_id=resource_id,
        resource_configuration=resource_configuration,
    )

    method_mapping = {
        AdoDescribeSupportedResourceTypes.DATA_CONTAINER: describe_data_container,
        AdoDescribeSupportedResourceTypes.DISCOVERY_SPACE: describe_discovery_space,
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
        options_metavar="[-f | --file <file.yaml>] [--actuator-id <id>]",
    )(describe_resource)
