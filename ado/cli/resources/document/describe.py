# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

from rich.status import Status

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

    console_print(document_resource)
