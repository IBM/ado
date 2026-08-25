# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import pathlib

from ado.cli.models.parameters import AdoTemplateCommandParameters
from ado.core.document.config import DocumentConfiguration


def template_document(parameters: AdoTemplateCommandParameters) -> None:
    """Emit a starter document configuration YAML file."""
    from ado.cli.utils.pydantic.serializers import (
        serialise_pydantic_model,
        serialise_pydantic_model_json_schema,
    )

    model_instance = DocumentConfiguration(
        content="# Report title\n\nReport body.",
        contentType="markdown",
        relatedResources=[],
    )
    serialise_pydantic_model(
        model=model_instance,
        output_path=parameters.output_file,
    )

    if parameters.include_schema:
        if parameters.output_file is None:
            serialise_pydantic_model_json_schema(model_instance, None)
        else:
            schema_output_path = pathlib.Path(
                parameters.output_file.stem + "_schema.yaml"
            )
            serialise_pydantic_model_json_schema(model_instance, schema_output_path)
