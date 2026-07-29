# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

from rich.console import Group
from rich.markdown import Markdown
from rich.status import Status
from rich.text import Text

from ado.cli.models.parameters import AdoDescribeCommandParameters
from ado.cli.utils.generic.wrappers import get_sql_store
from ado.cli.utils.output.prints import (
    ADO_SPINNER_QUERYING_DB,
    console_print,
)
from ado.core.resources import CoreResourceKinds
from ado.metastore.base import ResourceDoesNotExistError


def describe_document(parameters: AdoDescribeCommandParameters) -> None:
    """Print a human-friendly description of a document resource.

    Markdown content is rendered with rich. HTML content is printed as the HTML
    source. Rich handles terminal vs redirected formatting.
    """
    sql = get_sql_store(project_context=parameters.ado_configuration.project_context)

    with Status(ADO_SPINNER_QUERYING_DB) as status:
        document_resource = sql.getResource(
            identifier=parameters.resource_id, kind=CoreResourceKinds.DOCUMENT
        )
        if not document_resource:
            status.stop()
            raise ResourceDoesNotExistError(
                resource_id=parameters.resource_id, kind=CoreResourceKinds.DOCUMENT
            )

    config = document_resource.config

    header_parts: list[Text | str] = [
        Text.assemble(
            ("Identifier: ", "bold"), (document_resource.identifier, "bold green")
        ),
        "",
    ]
    if config.metadata.name:
        header_parts.append(Text.assemble(("Name: ", "bold"), (config.metadata.name,)))
    if config.metadata.description:
        header_parts.append(
            Text.assemble(("Description: ", "bold"), (config.metadata.description,))
        )
    if config.relatedResources:
        related_summary = ", ".join(
            f"{related.id} ({related.role})" for related in config.relatedResources
        )
        header_parts.append(
            Text.assemble(("Related resources: ", "bold"), (related_summary,))
        )
    header_parts.append("")

    if config.contentType == "html":
        console_print(Group(*header_parts, config.content))
        return

    console_print(Group(*header_parts, Markdown(config.content)))
