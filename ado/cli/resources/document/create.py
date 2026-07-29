# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import pydantic
import typer
import yaml
from rich.status import Status

from ado.cli.models.parameters import AdoCreateCommandParameters
from ado.cli.utils.generic.wrappers import get_sql_store
from ado.cli.utils.output.prints import (
    ADO_CREATE_DRY_RUN_CONFIG_VALID,
    ADO_SPINNER_SAVING_TO_DB,
    ERROR,
    SUCCESS,
    console_print,
    magenta,
)
from ado.cli.utils.pydantic.updaters import override_values_in_pydantic_model
from ado.core.document.config import DocumentConfiguration, RelatedResource
from ado.core.document.resource import DocumentResource
from ado.metastore.sqlstore import SQLStore


def _add_document_relationships(
    sql: SQLStore,
    document_identifier: str,
    related_resources: list[RelatedResource],
) -> None:
    """Write parent/child edges for a document (subject=parent, object=child)."""
    for related in related_resources:
        if related.role == "parent":
            sql.addRelationship(
                subjectIdentifier=related.id,
                objectIdentifier=document_identifier,
            )
        else:
            sql.addRelationship(
                subjectIdentifier=document_identifier,
                objectIdentifier=related.id,
            )


def create_document(parameters: AdoCreateCommandParameters) -> str | None:
    """Create a document resource from a YAML configuration file."""
    try:
        document_configuration = DocumentConfiguration.model_validate(
            yaml.safe_load(parameters.resource_configuration_file.read_text())
        )
    except pydantic.ValidationError as error:
        console_print(
            f"{ERROR}The document provided was not valid:",
            stderr=True,
        )
        console_print(error, stderr=True, use_markup=False)
        raise typer.Exit(1) from error

    if parameters.override_values:
        document_configuration = override_values_in_pydantic_model(
            model=document_configuration,
            override_values=parameters.override_values,
        )

    if parameters.dry_run:
        console_print(ADO_CREATE_DRY_RUN_CONFIG_VALID, stderr=True)
        return None

    resource_to_be_created = DocumentResource(config=document_configuration)

    sql = get_sql_store(project_context=parameters.ado_configuration.project_context)

    missing_related = [
        related.id
        for related in document_configuration.relatedResources
        if not sql.containsResourceWithIdentifier(identifier=related.id)
    ]
    if missing_related:
        console_print(
            f"{ERROR}Unknown related resource identifier(s): "
            f"{', '.join(missing_related)}",
            stderr=True,
        )
        raise typer.Exit(1)

    with Status(ADO_SPINNER_SAVING_TO_DB):
        sql.addResource(resource_to_be_created)
        _add_document_relationships(
            sql=sql,
            document_identifier=resource_to_be_created.identifier,
            related_resources=document_configuration.relatedResources,
        )

    console_print(
        f"{SUCCESS}Created document with identifier "
        f"{magenta(resource_to_be_created.identifier)}",
        stderr=True,
    )

    return resource_to_be_created.identifier
