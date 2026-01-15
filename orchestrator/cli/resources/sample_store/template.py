# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import pathlib

from orchestrator.cli.models.parameters import AdoTemplateCommandParameters
from orchestrator.core.samplestore.config import (
    SampleStoreConfiguration,
    SampleStoreModuleConf,
    SampleStoreSpecification,
)


def template_sample_store(parameters: AdoTemplateCommandParameters) -> None:
    from orchestrator.cli.utils.pydantic.serializers import (
        serialise_pydantic_model,
        serialise_pydantic_model_json_schema,
    )

    model_instance = SampleStoreConfiguration(
        specification=SampleStoreSpecification(
            module=SampleStoreModuleConf(
                moduleClass="SQLSampleStore",
                moduleName="orchestrator.core.samplestore.sql",
            )
        )
    )
    serialise_pydantic_model(
        model=model_instance,
        output_path=parameters.output_path,
    )

    if parameters.include_schema:
        schema_output_path = pathlib.Path(parameters.output_path.stem + "_schema.yaml")
        serialise_pydantic_model_json_schema(model_instance, schema_output_path)
