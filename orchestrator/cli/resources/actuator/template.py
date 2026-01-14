# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import pathlib

import pydantic

from orchestrator.cli.models.parameters import AdoTemplateCommandParameters


def template_actuator(parameters: AdoTemplateCommandParameters) -> None:
    import orchestrator.modules.actuators.base
    from orchestrator.cli.utils.pydantic.serializers import (
        serialise_pydantic_model,
        serialise_pydantic_model_json_schema,
    )

    ActuatorFileModel = pydantic.RootModel[
        list[orchestrator.modules.actuators.base.ActuatorModuleConf]
    ]

    model_instance = ActuatorFileModel(
        [orchestrator.modules.actuators.base.ActuatorModuleConf()]
    )
    serialise_pydantic_model(
        model=model_instance,
        output_path=parameters.output_path,
    )

    if parameters.include_schema:
        schema_output_path = pathlib.Path(parameters.output_path.stem + "_schema.yaml")
        serialise_pydantic_model_json_schema(model_instance, schema_output_path)
