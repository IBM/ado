# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import pathlib
import typing
from typing import Annotated

import typer
from click.core import ParameterSource

from orchestrator.cli.exceptions.handlers import (
    handle_no_related_resource,
    handle_resource_does_not_exist,
)
from orchestrator.cli.models.choice import HiddenPluralChoice
from orchestrator.cli.models.parameters import AdoEditCommandParameters
from orchestrator.cli.models.types import (
    AdoEditSupportedEditors,
    AdoEditSupportedResourceTypes,
)
from orchestrator.cli.resources.actuator_configuration.edit import (
    edit_actuator_configuration,
)
from orchestrator.cli.resources.data_container.edit import edit_data_container
from orchestrator.cli.resources.discovery_space.edit import edit_discovery_space
from orchestrator.cli.resources.operation.edit import edit_operation
from orchestrator.cli.resources.sample_store.edit import edit_sample_store
from orchestrator.cli.utils.output.prints import ERROR, console_print
from orchestrator.metastore.base import (
    NoRelatedResourcesError,
    ResourceDoesNotExistError,
)

if typing.TYPE_CHECKING:
    from orchestrator.cli.core.config import AdoConfiguration


def _parse_ado_edit_editor_name(value: str) -> AdoEditSupportedEditors:
    """Map *value* from :class:`HiddenPluralChoice` to the editor enum member."""
    token = value.removesuffix("s")
    for m in AdoEditSupportedEditors:
        if m.value == token:
            return m
    raise RuntimeError(
        "HiddenPluralChoice should have already validated the editor"
    )  # pragma: no cover


def edit_resource(
    ctx: typer.Context,
    resource_type: Annotated[
        AdoEditSupportedResourceTypes,
        typer.Argument(
            ...,
            help="The kind of the resource to edit metadata of.",
            show_default=False,
            click_type=HiddenPluralChoice(AdoEditSupportedResourceTypes),
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
        str,
        typer.Option(
            "--editor",
            envvar="ADO_EDITOR",
            help=(
                "The editor to use to edit metadata (interactive mode only; "
                "not with --patch or --patch-file)."
            ),
            click_type=AdoEditSupportedEditors,
        ),
    ] = AdoEditSupportedEditors.NANO.value,
    patch: Annotated[
        str | None,
        typer.Option(
            "-p",
            "--patch",
            help=(
                "YAML/JSON to merge into metadata (strategic merge; default "
                "non-interactive input, like oc -p)."
            ),
            show_default=False,
        ),
    ] = None,
    patch_file: Annotated[
        pathlib.Path | None,
        typer.Option(
            "--patch-file",
            help="File with YAML/JSON to merge into metadata.",
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

    non_interactive = patch is not None or patch_file is not None
    if non_interactive and ctx.get_parameter_source("editor") in (
        ParameterSource.COMMANDLINE,
        ParameterSource.PROMPT,
    ):
        console_print(
            f"{ERROR}The options --patch / -p and --patch-file "
            "may not be used with an explicit --editor flag.",
            stderr=True,
        )
        raise typer.Exit(1)

    ado_configuration: AdoConfiguration = ctx.obj
    if non_interactive:
        parameters = AdoEditCommandParameters(
            ado_configuration=ado_configuration,
            editor=None,
            resource_id=resource_id,
            metadata_patch=patch,
            metadata_path=patch_file,
        )
    else:
        parameters = AdoEditCommandParameters(
            ado_configuration=ado_configuration,
            editor=_parse_ado_edit_editor_name(editor),
            resource_id=resource_id,
            metadata_patch=None,
            metadata_path=None,
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
        options_metavar="[-p, --patch <yaml>] [--patch-file <file>] "
        "[--editor <name>]",
    )(edit_resource)
