# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT


import pathlib
import typing

import pydantic
import rich.rule
import typer
import yaml
from rich.status import Status

from orchestrator.cli.models.types import (
    AdoEditSupportedEditors,
    AdoGetSupportedOutputFormats,
)
from orchestrator.cli.utils.generic.wrappers import get_sql_store
from orchestrator.cli.utils.output.prints import (
    ADO_GET_CONFIG_ONLY_WHEN_SINGLE_RESOURCE,
    ADO_INFO_EMPTY_DATAFRAME,
    ADO_SPINNER_GETTING_OUTPUT_READY,
    ADO_SPINNER_QUERYING_DB,
    ADO_SPINNER_SAVING_TO_DB,
    ERROR,
    SUCCESS,
    console_print,
    cyan,
)
from orchestrator.cli.utils.resources.formatters import (
    format_default_ado_get_multiple_resources,
    format_default_ado_get_single_resource,
    format_resource_for_ado_get_custom_format,
)
from orchestrator.core.metadata import ConfigurationMetadata
from orchestrator.metastore.base import ResourceDoesNotExistError

if typing.TYPE_CHECKING:
    from orchestrator.cli.models.parameters import (
        AdoGetCommandParameters,
        AdoUpgradeCommandParameters,
    )
    from orchestrator.core import CoreResourceKinds
    from orchestrator.metastore.project import ProjectContext
    from orchestrator.metastore.sqlstore import SQLStore


def handle_ado_get_special_formats(
    parameters: "AdoGetCommandParameters",
    resource_type: "CoreResourceKinds",
) -> None:

    if (
        parameters.output_format == AdoGetSupportedOutputFormats.CONFIG
        and not parameters.resource_id
    ):
        console_print(f"{ERROR}{ADO_GET_CONFIG_ONLY_WHEN_SINGLE_RESOURCE}", stderr=True)
        raise typer.Exit(1)

    sql_store = get_sql_store(
        project_context=parameters.ado_configuration.project_context
    )
    with Status(ADO_SPINNER_QUERYING_DB) as status:

        if parameters.output_format == AdoGetSupportedOutputFormats.RAW:

            if not parameters.resource_id:
                status.stop()
                console_print(
                    f"{ERROR}Raw output mode is available only when specifying a resource_id",
                    stderr=True,
                )
                raise typer.Exit(1)

            resources = sql_store.getResourceRaw(parameters.resource_id)

        else:
            if parameters.resource_id:
                resources = sql_store.getResource(
                    identifier=parameters.resource_id, kind=resource_type
                )
                if not resources:
                    status.stop()
                    raise ResourceDoesNotExistError(
                        resource_id=parameters.resource_id, kind=resource_type
                    )
            else:
                resources = list(
                    sql_store.getResourcesOfKind(
                        kind=resource_type.value,
                        field_selectors=parameters.field_selectors,
                    ).values()
                )

        status.stop()
        console_print(
            format_resource_for_ado_get_custom_format(
                to_print=resources, parameters=parameters
            )
        )


def handle_ado_get_default_format(
    parameters: "AdoGetCommandParameters",
    resource_type: "CoreResourceKinds",
) -> None:

    sql_store = get_sql_store(
        project_context=parameters.ado_configuration.project_context
    )
    with Status(ADO_SPINNER_QUERYING_DB) as status:
        if not parameters.resource_id:
            resources = sql_store.getResourceIdentifiersOfKind(
                kind=resource_type.value,
                field_selectors=parameters.field_selectors,
                details=parameters.show_details,
            )

            status.update(ADO_SPINNER_GETTING_OUTPUT_READY)
            output_df = format_default_ado_get_multiple_resources(
                resources=resources,
                resource_kind=resource_type,
            )

            status.stop()
            if output_df.empty:
                console_print(ADO_INFO_EMPTY_DATAFRAME, stderr=True)
                return

            console_print(output_df, has_pandas_content=True)
            return

        resource = sql_store.getResource(
            identifier=parameters.resource_id, kind=resource_type
        )
        status.stop()

        if not resource:
            raise ResourceDoesNotExistError(
                resource_id=parameters.resource_id, kind=resource_type
            )

        output_df = format_default_ado_get_single_resource(
            resource=resource, show_details=parameters.show_details
        )

        console_print(output_df, has_pandas_content=True)


def print_related_resources(
    resource_id: str,
    resource_type: "CoreResourceKinds",
    sql: "SQLStore",
    hide_banner: bool = False,
) -> None:
    with Status(ADO_SPINNER_QUERYING_DB) as status:
        if not sql.containsResourceWithIdentifier(identifier=resource_id):
            status.stop()
            raise ResourceDoesNotExistError(resource_id=resource_id, kind=resource_type)

        status.update("Finding related resources")
        related_resources = sql.getRelatedResourceIdentifiers(resource_id)

    if related_resources.empty:
        console_print("There are no related resources", stderr=True)
        return

    if not hide_banner:
        console_print(rich.rule.Rule(title="RELATED RESOURCES"))
    previous_resource_kind = ""
    for _, row in related_resources.iterrows():
        if row["TYPE"] != previous_resource_kind:
            console_print(cyan(row["TYPE"]))
            previous_resource_kind = row["TYPE"]
        console_print(f"  - {row['IDENTIFIER']}")


def handle_edit_resource_metadata(
    resource_id: str,
    resource_type: "CoreResourceKinds",
    project_context: "ProjectContext",
    editor: AdoEditSupportedEditors,
) -> None:
    import subprocess  # noqa: S404
    import tempfile

    import orchestrator.cli.utils.pydantic.serializers

    sql = get_sql_store(project_context=project_context)
    with Status(ADO_SPINNER_QUERYING_DB) as status:
        resource = sql.getResource(identifier=resource_id, kind=resource_type)
        if not resource:
            status.stop()
            raise ResourceDoesNotExistError(resource_id=resource_id, kind=resource_type)

    with tempfile.TemporaryDirectory() as d:
        file = pathlib.Path(d) / pathlib.Path("tmp_metadata.yaml")
        orchestrator.cli.utils.pydantic.serializers.serialise_pydantic_model(
            model=resource.config.metadata,
            output_path=file,
            suppress_success_message=True,
        )

        try:
            subprocess.run([editor.value, file], check=True)  # noqa: S603
        except subprocess.CalledProcessError as e:
            console_print(f"{ERROR}The editor exited with an error: {e}", stderr=True)
            raise typer.Exit(1) from e

        try:
            new_metadata = ConfigurationMetadata.model_validate(
                yaml.safe_load(file.read_text())
            )
        except pydantic.ValidationError as e:
            console_print(f"{ERROR}The updated metadata was invalid: {e}", stderr=True)
            raise typer.Exit(1) from e

    resource.config.metadata = new_metadata
    with Status(ADO_SPINNER_SAVING_TO_DB):
        sql.updateResource(resource)

    console_print(SUCCESS, stderr=True)


def handle_ado_upgrade(
    parameters: "AdoUpgradeCommandParameters",
    resource_type: "CoreResourceKinds",
) -> None:

    sql_store = get_sql_store(
        project_context=parameters.ado_configuration.project_context
    )
    with Status(ADO_SPINNER_QUERYING_DB) as status:
        resources = sql_store.getResourcesOfKind(
            kind=resource_type.value,
        )

        for idx, resource in enumerate(resources.values()):
            status.update(ADO_SPINNER_SAVING_TO_DB + f" ({idx +1 }/{len(resources)})")
            sql_store.updateResource(resource=resource)

    console_print(SUCCESS)
