# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import pathlib
import typing

from rich.status import Status

from orchestrator.cli.models.parameters import AdoTemplateCommandParameters
from orchestrator.cli.utils.output.prints import (
    ADO_SPINNER_GETTING_OUTPUT_READY,
)
from orchestrator.cli.utils.resources.experiments import (
    _ado_resolved_experiment_reference_from_cli_resource_id,
)
from orchestrator.core.discoveryspace.config import DiscoverySpaceConfiguration
from orchestrator.schema.measurementspace import MeasurementSpace

if typing.TYPE_CHECKING:
    from orchestrator.schema.entityspace import EntitySpaceRepresentation


def template_discovery_space(parameters: AdoTemplateCommandParameters) -> None:
    from orchestrator.cli.utils.pydantic.serializers import (
        serialise_pydantic_model,
        serialise_pydantic_model_json_schema,
    )

    with Status(ADO_SPINNER_GETTING_OUTPUT_READY):
        if parameters.from_experiments:
            experiment_references = [
                _ado_resolved_experiment_reference_from_cli_resource_id(experiment_id)
                for experiment_id in parameters.from_experiments
            ]

            measurement_space = (
                MeasurementSpace.measurementSpaceFromExperimentReferences(
                    experimentReferences=experiment_references
                )
            )
            entity_space: EntitySpaceRepresentation = (
                measurement_space.compatibleEntitySpace()
            )

            model_instance = DiscoverySpaceConfiguration(
                sampleStoreIdentifier="ID",
                entitySpace=entity_space.constitutiveProperties,
                experiments=experiment_references,
            )
        else:
            model_instance = DiscoverySpaceConfiguration(sampleStoreIdentifier="ID")

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
