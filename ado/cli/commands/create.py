# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT
import os
import pathlib
from typing import Annotated

import typer

from ado.cli.core.config import AdoConfiguration
from ado.cli.exceptions.handlers import (
    handle_no_related_resource,
    handle_resource_does_not_exist,
    handle_unknown_experiment_error,
)
from ado.cli.models.parameters import (
    AdoCreateCommandParameters,
    OperationInputResourceOption,
)
from ado.cli.models.types import (
    AdoCreateSupportedResourceTypes,
    AdoCreateWithResourceSupportedResourceTypes,
)
from ado.cli.resources.actuator_configuration.create import (
    create_actuator_configuration,
)
from ado.cli.resources.context.create import create_context
from ado.cli.resources.discovery_space.create import create_discovery_space
from ado.cli.resources.document.create import create_document
from ado.cli.resources.operation.create import create_operation
from ado.cli.resources.sample_store.create import create_sample_store
from ado.cli.utils.input.parsers import (
    enum_choice_with_plural_parser,
    parse_key_value_pairs,
    resource_kind_with_input_parser,
    resource_shorthands_to_full_names,
    split_input_qualifier,
)
from ado.cli.utils.output.prints import (
    ERROR,
    HINT,
    bold,
    console_print,
    magenta,
)
from ado.core import CoreResourceKinds
from ado.core.operation.inputs import KIND_FOR_OPERATOR_INPUT_TYPE
from ado.metastore.base import (
    NoRelatedResourcesError,
    ResourceDoesNotExistError,
)
from ado.modules.actuators.errors import UnknownExperimentError

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
            parser=enum_choice_with_plural_parser(AdoCreateSupportedResourceTypes),
            metavar=f"[{'|'.join(m.value for m in AdoCreateSupportedResourceTypes)}]",
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
            help="""Specify additional resources to use or create via KIND=VALUE pairs,
            where VALUE is a resource ID or a resource configuration file.

            Supports creating spaces with sample stores and operations with
            spaces, data containers, and actuator configurations.

            For operations, use KIND:INPUT=VALUE to set the operation input
            named INPUT, e.g. space:reference=space-abc.""",
        ),
    ] = None,
    use_latest: Annotated[
        list[str] | None,
        typer.Option(
            show_default=False,
            parser=resource_kind_with_input_parser(CoreResourceKinds),
            metavar=f"[{'|'.join(m.value for m in CoreResourceKinds)}][:INPUT]",
            help="""
            Reuse the latest identifier of a resource kind. Can be used multiple times.

            For operations, use KIND:INPUT to set the operation input named INPUT,
            e.g. space:reference.

            Only supported for spaces and operations.""",
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

    See https://ibm.github.io/ado/latest/cli-reference/#ado-create
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
    with_resources, with_inputs = parse_with_resource_options(with_resources)
    use_latest_kinds, use_latest_inputs = parse_use_latest_options(use_latest)

    if (with_inputs or use_latest_inputs) and (
        resource_type != AdoCreateSupportedResourceTypes.OPERATION
    ):
        console_print(
            f"{ERROR}KIND:INPUT qualifiers for {magenta('--with')} and "
            f"{magenta('--use-latest')} are only supported when creating operations.",
            stderr=True,
        )
        raise typer.Exit(1)

    parameters = AdoCreateCommandParameters(
        ado_configuration=ado_configuration,
        dry_run=dry_run,
        new_sample_store=new_sample_store,
        override_values=override_values,
        resource_configuration_file=resource_configuration,
        resource_type=resource_type,
        use_default_sample_store=use_default_sample_store,
        use_latest=use_latest_kinds,
        with_resources=with_resources,
        useLatestInputs=use_latest_inputs,
        withInputs=with_inputs,
    )

    method_mapping = {
        AdoCreateSupportedResourceTypes.ACTUATOR_CONFIGURATION: create_actuator_configuration,
        AdoCreateSupportedResourceTypes.CONTEXT: create_context,
        AdoCreateSupportedResourceTypes.DISCOVERY_SPACE: create_discovery_space,
        AdoCreateSupportedResourceTypes.DOCUMENT: create_document,
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


def register_create_command(app: typer.Typer) -> None:
    app.command(
        name="create",
        no_args_is_help=True,
        options_metavar="[-f | --file <configuration>] [--set <path=document> ...] "
        "[--new-sample-store] [--dry-run]",
    )(create_resource)


def _resolve_with_value(value: str) -> pathlib.Path | str:
    """Interpret a ``--with`` value as a configuration file path or an identifier.

    A value containing a dot or a path separator must be an existing file.

    Args:
        value: The value part of a ``--with`` option.

    Returns:
        The file path when *value* names an existing file, otherwise *value*.

    Raises:
        typer.Exit: If *value* looks like a path but is not an existing file.
    """
    value_as_path = pathlib.Path(value)
    if value_as_path.exists() and value_as_path.is_file():
        return value_as_path
    if "." in value or os.sep in value:
        console_print(f"{ERROR}{value} does not exist or is not a file.", stderr=True)
        raise typer.Exit(1)
    return value


def parse_with_resource_options(
    user_provided_options: list[str] | None,
) -> tuple[
    dict[CoreResourceKinds, pathlib.Path | str], list[OperationInputResourceOption]
]:
    """Parse ``--with`` options into kind-keyed values and named operation inputs.

    ``KIND=VALUE`` options are keyed by kind. ``KIND:INPUT=VALUE`` options
    become named operation inputs. ``KIND`` may be a shorthand.

    Args:
        user_provided_options: Raw ``--with`` option values.

    Returns:
        The kind-keyed values and the named operation inputs, in option order.

    Raises:
        typer.Exit: If a kind is unsupported, a kind is given twice without a
            qualifier, a qualifier is invalid, or a path value does not exist.
    """
    parsed_options: dict[CoreResourceKinds, pathlib.Path | str] = {}
    named_inputs: list[OperationInputResourceOption] = []
    if not user_provided_options:
        return parsed_options, named_inputs

    supported_resource_types = {
        v.value for v in AdoCreateWithResourceSupportedResourceTypes
    }
    supported_input_kinds = set(KIND_FOR_OPERATOR_INPUT_TYPE.values())

    for with_option in parse_key_value_pairs(user_provided_options):
        for key, value in with_option.items():
            try:
                resource_type, input_name = split_input_qualifier(key)
            except ValueError as error:
                console_print(f"{ERROR}{error}", stderr=True)
                raise typer.Exit(1) from error

            resource_type = resource_shorthands_to_full_names(resource_type)
            if resource_type not in supported_resource_types:
                console_print(
                    f"{ERROR}{bold(resource_type)} is not a supported resource type "
                    f"for the {magenta('--with')} option.\n"
                    f"{HINT}The only supported resource types are "
                    f"[b cyan]{', '.join(sorted(supported_resource_types))}[/b cyan] "
                    "and their shorthands.",
                    stderr=True,
                )
                raise typer.Exit(1)

            kind = CoreResourceKinds(resource_type)

            if input_name is not None:
                if kind not in supported_input_kinds:
                    console_print(
                        f"{ERROR}{bold(kind.value)} cannot be used as an operation "
                        "input. Supported input kinds are "
                        f"[b cyan]{', '.join(sorted(k.value for k in supported_input_kinds))}"
                        "[/b cyan].",
                        stderr=True,
                    )
                    raise typer.Exit(1)
                named_inputs.append(
                    OperationInputResourceOption(
                        inputName=input_name,
                        kind=kind,
                        value=_resolve_with_value(value),
                    )
                )
                continue

            if kind in parsed_options:
                console_print(
                    f"{ERROR}The {kind.value} resource type was specified more than once.\n"
                    f"{HINT}Use {kind.value}:INPUT=VALUE to set several "
                    "operation inputs of the same kind.",
                    stderr=True,
                )
                raise typer.Exit(1)

            parsed_options[kind] = _resolve_with_value(value)

    return parsed_options, named_inputs


def parse_use_latest_options(
    user_provided_options: list[str] | None,
) -> tuple[list[CoreResourceKinds], list[tuple[str, CoreResourceKinds]]]:
    """Parse ``--use-latest`` values into kinds and named operation inputs.

    Args:
        user_provided_options: ``KIND`` or ``KIND:INPUT`` values with the kind
            already resolved to its full name.

    Returns:
        The unqualified kinds and the ``(input name, kind)`` pairs, in option order.

    Raises:
        typer.Exit: If a qualified kind cannot be an operation input.
    """
    kinds: list[CoreResourceKinds] = []
    named_inputs: list[tuple[str, CoreResourceKinds]] = []
    supported_input_kinds = set(KIND_FOR_OPERATOR_INPUT_TYPE.values())

    for option in user_provided_options or []:
        kind_value, input_name = split_input_qualifier(option)
        kind = CoreResourceKinds(kind_value)
        if input_name is None:
            kinds.append(kind)
            continue
        if kind not in supported_input_kinds:
            console_print(
                f"{ERROR}{bold(kind.value)} cannot be used as an operation input. "
                "Supported input kinds are "
                f"[b cyan]{', '.join(sorted(k.value for k in supported_input_kinds))}"
                "[/b cyan].",
                stderr=True,
            )
            raise typer.Exit(1)
        named_inputs.append((input_name, kind))

    return kinds, named_inputs
