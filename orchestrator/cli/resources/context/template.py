# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import pathlib

from orchestrator.cli.models.parameters import AdoTemplateCommandParameters
from orchestrator.cli.utils.pydantic.serializers import (
    serialise_pydantic_model,
    serialise_pydantic_model_json_schema,
)
from orchestrator.metastore.project import ProjectContext
from orchestrator.utilities.location import (
    SQLiteStoreConfiguration,
    SQLStoreConfiguration,
)


def template_context(parameters: AdoTemplateCommandParameters) -> None:

    model_instance = (
        ProjectContext(
            project="your-project-name",
            metadataStore=SQLiteStoreConfiguration(
                path=str(
                    parameters.ado_configuration.local_db_path_for_context(
                        "your-project-name"
                    )
                ),
            ),
        )
        if parameters.template_local_context
        else ProjectContext(
            project="your-project-name",
            metadataStore=SQLStoreConfiguration(
                scheme="mysql+pymysql",
                host="your-db-endpoint",
                database="your-project-name",
                user="your-user",
                password="your-password",  # noqa: S106
            ),
        )
    )

    serialise_pydantic_model(
        model=model_instance,
        output_path=parameters.output_path,
    )

    if parameters.include_schema:
        schema_output_path = pathlib.Path(parameters.output_path.stem + "_schema.yaml")
        serialise_pydantic_model_json_schema(model_instance, schema_output_path)
