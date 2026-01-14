# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT
import os
import pathlib
from typing import Annotated

import typer

from orchestrator.cli.core.config import AdoConfiguration
from orchestrator.cli.exceptions.handlers import (
    handle_no_related_resource,
    handle_resource_does_not_exist,
    handle_unknown_experiment_error,
)
from orchestrator.cli.models.choice import HiddenPluralChoice, HiddenShorthandChoice
from orchestrator.cli.models.parameters import AdoCreateCommandParameters
from orchestrator.cli.models.types import (
    AdoCreateSupportedResourceTypes,
    AdoCreateWithResourceSupportedResourceTypes,
)
from orchestrator.cli.resources.actuator_configuration.create import (
    create_actuator_configuration,
)
from orchestrator.cli.resources.context.create import create_context
from orchestrator.cli.resources.discovery_space.create import create_discovery_space
from orchestrator.cli.resources.operation.create import create_operation
from orchestrator.cli.resources.sample_store.create import create_sample_store
from orchestrator.cli.utils.input.parsers import (
    parse_key_value_pairs,
    resource_shorthands_to_full_names,
)
from orchestrator.cli.utils.output.prints import (
    ERROR,
    HINT,
    bold,
    console_print,
    magenta,
)
from orchestrator.core import CoreResourceKinds
from orchestrator.metastore.base import (
    NoRelatedResourcesError,
    ResourceDoesNotExistError,
)
from orchestrator.modules.actuators.registry import UnknownExperimentError

CREATE_OPERATION_PANEL_NAME = "Operation-specific options"
CREATE_SPACE_PANEL_NAME = "Space-specific options"


def resource_type_callback(
    ctx: typer.Context,
    param: typer.CallbackParam,
    value: AdoCreateSupportedResourceTypes,
) -> str:
    # AP: 27/05/2025
    # To avoid making it impossible to run ado create context
    # when there is no default context, we need to disable
    # failing on existing context in cli.py.
    # This is because we can only see that we are running "ado create"
    # and not what we are creating. Here, we perform the stricter
    # validation if we're not running ado create context
    if value != AdoCreateSupportedResourceTypes.CONTEXT:
        project_context_param = ctx.parent.params["project_context_file"]
        project_context_file = (
            pathlib.Path(project_context_param) if project_context_param else None
        )

        override_ado_app_dir_param = ctx.parent.params["override_ado_app_dir"]
        override_ado_app_dir = (
            pathlib.Path(override_ado_app_dir_param)
            if override_ado_app_dir_param
            else None
        )

        ctx.obj = AdoConfiguration.load(
            from_project_context=project_context_file,
            _override_config_dir=override_ado_app_dir,
        )

    # We need to return a string
    return value.value


def create_resource(
    ctx: typer.Context,
    resource_type: Annotated[
        AdoCreateSupportedResourceTypes,
        typer.Argument(
            help="The kind of the resource to create.",
            show_default=False,
            click_type=HiddenPluralChoice(AdoCreateSupportedResourceTypes),
            callback=resource_type_callback,
        ),
    ],
    resource_configuration: Annotated[
        pathlib.Path | None,
        typer.Option(
            "--file",
            "-f",
            help="""
            Resource configuration details as YAML.

            If creating a sample store and using --new-sample-store, this is optional.
            """,
            show_default=False,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ] = None,
    new_sample_store: Annotated[
        bool,
        typer.Option(
            "--new-sample-store",
            help="Request and use a new, empty sample store. Available only for space and sample store. "
            "Ignored if --with or --use-latest are used.",
        ),
    ] = False,
    set_values: Annotated[
        list[str] | None,
        typer.Option(
            "--set",
            show_default=False,
            help="""
            Override fields in the resource configuration. Can be used multiple times.


            Requires using JSONPath syntax.
            See https://github.com/h2non/jsonpath-ng?tab=readme-ov-file#jsonpath-syntax
            for more detailed information.
            """,
        ),
    ] = None,
    with_resources: Annotated[
        list[str] | None,
        typer.Option(
            "--with",
            help="""Specify additional resources to use or create via key=value pairs,
            where the key is a resource type and the value is a resource ID or a resource configuration file.

            Supports creating spaces with sample stores and operations with spaces and actuator configurations.""",
        ),
    ] = None,
    use_latest: Annotated[
        list[CoreResourceKinds] | None,
        typer.Option(
            show_default=False,
            click_type=HiddenShorthandChoice(CoreResourceKinds),
            help="""
            Reuse the latest identifier of a resource kind. Can be used multiple times.

            Only supported for spaces and operations. Ignored if --with is used.""",
        ),
    ] = None,
    use_default_sample_store: Annotated[
        bool,
        typer.Option(
            "--use-default-sample-store",
            rich_help_panel=CREATE_SPACE_PANEL_NAME,
            help="Request and use the default sample store. Available only for spaces. "
            "Ignored if --with, --use-latest, or --new-sample-store are used. "
            "Alias for --set sampleStoreIdentifier=default.",
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Validate the resource configuration file without creating the associated resource.",
        ),
    ] = False,
) -> None:
    """
    Create resources, contexts, and start operations.

    See https://ibm.github.io/ado/getting-started/ado/#ado-create
    for detailed documentation and examples.



    Examples:



    # Create a space with a new sample store

    ado create space -f <space.yaml> --new-sample-store



    # Start an operation

    ado create operation -f <operation.yaml>



    # Create a context

    ado create context -f <file.yaml>
    """

    # AP 29/05/2025:
    # Having --new-sample-store means that the resource_configuration
    # has become optional. At the same time, however, we must enforce
    # it being specified in all cases other than
    # ado create samplestore --new-sample-store
    if not resource_configuration and not (
        resource_type == AdoCreateSupportedResourceTypes.SAMPLE_STORE
        and new_sample_store
    ):
        console_print(
            f"{ERROR}You must specify a resource configuration file using the -f flag.",
            stderr=True,
        )
        raise typer.Exit(1)

    # Unfortunately it looks like we lose the typer-provided checks
    # on the parameter by making it optional. We have to do some
    # manual validation.
    if resource_configuration and not all(
        (resource_configuration.is_file(), resource_configuration.exists())
    ):
        console_print(
            f"{ERROR}{resource_configuration.absolute()} must exist and be a file.",
            stderr=True,
        )
        raise typer.Exit(1)

    ado_configuration: AdoConfiguration = ctx.obj
    override_values = parse_key_value_pairs(set_values)
    with_resources = parse_with_resource_options(with_resources)

    parameters = AdoCreateCommandParameters(
        ado_configuration=ado_configuration,
        dry_run=dry_run,
        new_sample_store=new_sample_store,
        override_values=override_values,
        resource_configuration_file=resource_configuration,
        resource_type=resource_type,
        use_default_sample_store=use_default_sample_store,
        use_latest=use_latest,
        with_resources=with_resources,
    )

    method_mapping = {
        AdoCreateSupportedResourceTypes.ACTUATOR_CONFIGURATION: create_actuator_configuration,
        AdoCreateSupportedResourceTypes.CONTEXT: create_context,
        AdoCreateSupportedResourceTypes.DISCOVERY_SPACE: create_discovery_space,
        AdoCreateSupportedResourceTypes.SAMPLE_STORE: create_sample_store,
        AdoCreateSupportedResourceTypes.OPERATION: create_operation,
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
    except UnknownExperimentError as e:
        handle_unknown_experiment_error(error=e)

    ado_configuration.store()


def register_create_command(app: typer.Typer) -> None:
    app.command(
        name="create",
        no_args_is_help=True,
        options_metavar="[-f | --file <configuration>] [--set <path=document> ...] "
        "[--new-sample-store] [--dry-run]",
    )(create_resource)


def parse_with_resource_options(
    user_provided_options: list[str] | None,
) -> dict[CoreResourceKinds, pathlib.Path | str]:

    parsed_options = {}
    if not user_provided_options:
        return parsed_options

    # To simplify checking whether the resource type is supported
    supported_resource_types = {
        v.value for v in AdoCreateWithResourceSupportedResourceTypes
    }

    for with_option in parse_key_value_pairs(user_provided_options):
        for resource_type, value in with_option.items():

            resource_type = resource_shorthands_to_full_names(resource_type)
            if resource_type not in supported_resource_types:
                console_print(
                    f"{ERROR}{bold(resource_type)} is not a supported resource type "
                    f"for the {magenta('--with')} option.\n"
                    f"{HINT}The only supported resource types are "
                    f"[b cyan]{', '.join(supported_resource_types)}[/b cyan] and their shorthands.",
                    stderr=True,
                )
                raise typer.Exit(1)

            # Now we know the resource_type can be parsed as a CoreResourceKinds
            resource_type = CoreResourceKinds(resource_type)

            # Since we are dealing with a dictionary, we can only support one --with option per resource type
            if resource_type in parsed_options:
                console_print(
                    f"{ERROR}The {resource_type.value} resource type was specified more than once.",
                    stderr=True,
                )
                raise typer.Exit(1)

            # We allow users to provide either paths to a resource configuration file or a resource identifier.
            # In this function we only check whether it's one or the other (with validation on the path). Both
            # schema validation for configuration files and resource retrieval for ids will be performed in the
            # resource-specific create functions.
            #
            # If a dot or a path separator character is present in the value, we assume it must be a path and
            # fail if it doesn't exist, or it's not a file.
            value_must_be_path = "." in value or os.sep in value
            value_as_path = pathlib.Path(value)
            if value_as_path.exists() and value_as_path.is_file():
                parsed_options[resource_type] = value_as_path
            elif value_must_be_path:
                console_print(
                    f"{ERROR}{value} does not exist or is not a file.", stderr=True
                )
                raise typer.Exit(1)
            else:
                parsed_options[resource_type] = value

    return parsed_options
