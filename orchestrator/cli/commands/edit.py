# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import os
import pathlib
import typing
from typing import Annotated

import typer

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


def _resolve_interactive_ado_edit_editor(
    editor_cli: str | None,
) -> AdoEditSupportedEditors:
    """
    Select an editor: ``--editor`` if given, else ``$ADO_EDITOR``, else nano.
    """
    raw: str | None = editor_cli
    if raw is None:
        raw = os.environ.get("ADO_EDITOR")
    if raw is None:
        return AdoEditSupportedEditors.NANO
    token = str(raw).removesuffix("s")
    for member in AdoEditSupportedEditors:
        if member.value == token:
            return member
    valid = ", ".join(sorted(m.value for m in AdoEditSupportedEditors))
    console_print(
        f"{ERROR}Invalid --editor (or $ADO_EDITOR) value {raw!r}. "
        f"Use one of: {valid}.",
        stderr=True,
    )
    raise typer.Exit(1)


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
        str | None,
        typer.Option(
            "--editor",
            help=(
                "Editor for interactive mode (not used with --metadata). "
                "Default: nano, or the ADO_EDITOR environment variable if set."
            ),
            show_default=False,
        ),
    ] = None,
    metadata: Annotated[
        pathlib.Path | None,
        typer.Option(
            "--metadata",
            help=(
                "YAML file whose contents are merged into this resource's "
                "metadata (non-interactive; cannot be combined with --editor)."
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

    # Merge additional metadata from a file (e.g. labels) without an editor
    ado edit space <space-id> --metadata meta.yaml
    """
    if metadata is not None and editor is not None:
        console_print(
            f"{ERROR}The options --metadata and --editor may not be used together.",
            stderr=True,
        )
        raise typer.Exit(1)

    ado_configuration: AdoConfiguration = ctx.obj
    if metadata is not None:
        parameters = AdoEditCommandParameters(
            ado_configuration=ado_configuration,
            editor=None,
            resource_id=resource_id,
            metadata_path=metadata,
        )
    else:
        parameters = AdoEditCommandParameters(
            ado_configuration=ado_configuration,
            editor=_resolve_interactive_ado_edit_editor(editor),
            resource_id=resource_id,
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
        options_metavar="[--metadata <file>] [--editor <name>]",
    )(edit_resource)
