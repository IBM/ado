# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import pathlib
import typing
from typing import Annotated

import typer

from ado.cli.exceptions.handlers import (
    handle_no_related_resource,
    handle_resource_does_not_exist,
)
from ado.cli.models.parameters import AdoEditCommandParameters
from ado.cli.models.types import (
    AdoEditSupportedEditors,
    AdoEditSupportedResourceTypes,
)
from ado.cli.resources.actuator_configuration.edit import (
    edit_actuator_configuration,
)
from ado.cli.resources.data_container.edit import edit_data_container
from ado.cli.resources.discovery_space.edit import edit_discovery_space
from ado.cli.resources.operation.edit import edit_operation
from ado.cli.resources.sample_store.edit import edit_sample_store
from ado.cli.utils.input.parsers import enum_choice_with_plural_parser
from ado.cli.utils.output.prints import ERROR, console_print
from ado.metastore.base import (
    NoRelatedResourcesError,
    ResourceDoesNotExistError,
)

if typing.TYPE_CHECKING:
    from ado.cli.core.config import AdoConfiguration


def edit_resource(
    ctx: typer.Context,
    resource_type: Annotated[
        AdoEditSupportedResourceTypes,
        typer.Argument(
            ...,
            help="The kind of the resource to edit metadata of.",
            show_default=False,
            parser=enum_choice_with_plural_parser(AdoEditSupportedResourceTypes),
            metavar=f"[{'|'.join(m.value for m in AdoEditSupportedResourceTypes)}]",
        ),
    ],
    resource_id: Annotated[
        str,
        typer.Argument(
            ...,
            help="The id of the resource to edit metadata of.",
            show_default=False,
        ),
    ],
    editor: Annotated[
        AdoEditSupportedEditors,
        typer.Option(
            "--editor",
            envvar="ADO_EDITOR",
            help=(
                "The editor to use to edit metadata in interactive mode. "
                "Ignored when --patch or --patch-file is specified."
            ),
        ),
    ] = AdoEditSupportedEditors.NANO.value,
    patch: Annotated[
        str | None,
        typer.Option(
            "-p",
            "--patch",
            help=(
                "YAML/JSON to merge into metadata (strategic merge). "
                "Non-interactive mode; --editor is ignored if specified."
            ),
            show_default=False,
        ),
    ] = None,
    patch_file: Annotated[
        pathlib.Path | None,
        typer.Option(
            "--patch-file",
            help=(
                "File with YAML/JSON to merge into metadata. "
                "Non-interactive mode; --editor is ignored if specified."
            ),
            file_okay=True,
            dir_okay=False,
            exists=True,
            readable=True,
            resolve_path=True,
            show_default=False,
        ),
    ] = None,
) -> None:
    """
    Edit resources' metadata.

    See https://ibm.github.io/ado/getting-started/ado/#ado-edit
    for detailed documentation and examples.

    Examples:

    # Edit the metadata of a sample store
    ado edit samplestore <sample-store-id>

    # Edit the metadata of a space using vim
    ado edit space <space-id> --editor vim

    # Merge metadata with an inline patch (oc-style) or a file
    ado edit space <space-id> -p "labels: { team: core }"
    ado edit space <space-id> --patch-file meta.yaml
    """
    if patch is not None and patch_file is not None:
        console_print(
            f"{ERROR}Use only one of --patch / -p and --patch-file.",
            stderr=True,
        )
        raise typer.Exit(1)

    ado_configuration: AdoConfiguration = ctx.obj
    parameters = AdoEditCommandParameters(
        ado_configuration=ado_configuration,
        editor=editor,
        resource_id=resource_id,
        metadata_patch=patch,
        metadata_path=patch_file,
    )

    method_mapping = {
        AdoEditSupportedResourceTypes.ACTUATOR_CONFIGURATION: edit_actuator_configuration,
        AdoEditSupportedResourceTypes.DATA_CONTAINER: edit_data_container,
        AdoEditSupportedResourceTypes.DISCOVERY_SPACE: edit_discovery_space,
        AdoEditSupportedResourceTypes.SAMPLE_STORE: edit_sample_store,
        AdoEditSupportedResourceTypes.OPERATION: edit_operation,
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


def register_edit_command(app: typer.Typer) -> None:
    app.command(
        name="edit",
        no_args_is_help=True,
        options_metavar="[-p | --patch <yaml>] [--patch-file <file>] [--editor <name>]",
    )(edit_resource)
