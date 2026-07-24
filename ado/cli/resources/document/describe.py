# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import pathlib
import sys
import tempfile
import webbrowser

from rich.console import Group
from rich.markdown import Markdown
from rich.status import Status
from rich.text import Text

from ado.cli.models.parameters import AdoDescribeCommandParameters
from ado.cli.utils.generic.wrappers import get_sql_store
from ado.cli.utils.output.prints import (
    ADO_SPINNER_QUERYING_DB,
    INFO,
    console_print,
    stdout_console,
)
from ado.core.resources import CoreResourceKinds
from ado.metastore.base import ResourceDoesNotExistError


def _stdout_is_terminal() -> bool:
    """Return True when stdout is an interactive terminal."""
    return stdout_console.is_terminal


def describe_document(parameters: AdoDescribeCommandParameters) -> None:
    """Print a human-friendly description of a document resource.

    When stdout is a terminal, markdown is rendered with rich and HTML is opened
    in the default browser. When stdout is redirected (pipe or file), the raw
    ``content`` body is written with no styling.
    """
    sql = get_sql_store(project_context=parameters.ado_configuration.project_context)
    to_terminal = _stdout_is_terminal()

    if to_terminal:
        with Status(ADO_SPINNER_QUERYING_DB) as status:
            document_resource = sql.getResource(
                identifier=parameters.resource_id, kind=CoreResourceKinds.DOCUMENT
            )
            if not document_resource:
                status.stop()
                raise ResourceDoesNotExistError(
                    resource_id=parameters.resource_id, kind=CoreResourceKinds.DOCUMENT
                )
    else:
        document_resource = sql.getResource(
            identifier=parameters.resource_id, kind=CoreResourceKinds.DOCUMENT
        )
        if not document_resource:
            raise ResourceDoesNotExistError(
                resource_id=parameters.resource_id, kind=CoreResourceKinds.DOCUMENT
            )

    config = document_resource.config

    if not to_terminal:
        sys.stdout.write(config.content)
        if not config.content.endswith("\n"):
            sys.stdout.write("\n")
        return

    header_parts: list[Text | str] = [
        Text.assemble(
            ("Identifier: ", "bold"), (document_resource.identifier, "bold green")
        ),
        Text.assemble(("Content type: ", "bold"), (config.contentType,)),
        "",
    ]
    if config.metadata.name:
        header_parts.append(Text.assemble(("Name: ", "bold"), (config.metadata.name,)))
    if config.metadata.description:
        header_parts.append(
            Text.assemble(("Description: ", "bold"), (config.metadata.description,))
        )
    if config.relatedResources:
        header_parts.append(
            Text.assemble(
                ("Related resources: ", "bold"),
                (", ".join(config.relatedResources),),
            )
        )
    header_parts.append("")

    if config.contentType == "html":
        console_print(Group(*header_parts))
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".html",
            prefix=f"{document_resource.identifier}-",
            delete=False,
            encoding="utf-8",
        ) as html_file:
            html_file.write(config.content)
            html_path = pathlib.Path(html_file.name)

        webbrowser.open(html_path.as_uri())
        console_print(
            f"{INFO}Opened HTML document content in the default browser ({html_path})."
        )
        return

    console_print(Group(*header_parts, Markdown(config.content)))
