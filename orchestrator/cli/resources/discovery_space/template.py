# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import pathlib
import typing

from rich.status import Status

from orchestrator.cli.models.parameters import AdoTemplateCommandParameters
from orchestrator.cli.utils.output.prints import (
    ADO_SPINNER_GETTING_OUTPUT_READY,
)
from orchestrator.cli.utils.resources.experiments import (
    _ado_get_actuator_from_experiment_id,
)
from orchestrator.core.discoveryspace.config import DiscoverySpaceConfiguration
from orchestrator.schema.measurementspace import MeasurementSpace
from orchestrator.schema.reference import ExperimentReference

if typing.TYPE_CHECKING:
    from orchestrator.schema.entityspace import EntitySpaceRepresentation


def template_discovery_space(parameters: AdoTemplateCommandParameters) -> None:
    from orchestrator.cli.utils.pydantic.serializers import (
        serialise_pydantic_model,
        serialise_pydantic_model_json_schema,
    )

    with Status(ADO_SPINNER_GETTING_OUTPUT_READY):
        if parameters.from_experiments:

            experiment_references = []
            for pair in parameters.from_experiments:
                for experiment_id, actuator_id in pair.items():
                    actuator_id = (
                        actuator_id
                        if actuator_id
                        else _ado_get_actuator_from_experiment_id(
                            experiment_id=experiment_id, actuator_id=actuator_id
                        )
                    )
                    experiment_references.append(
                        ExperimentReference(
                            actuatorIdentifier=actuator_id,
                            experimentIdentifier=experiment_id,
                        )
                    )

            measurement_space = (
                MeasurementSpace.measurementSpaceFromExperimentReferences(
                    experimentReferences=experiment_references
                )
            )
            entity_space: EntitySpaceRepresentation = (
                measurement_space.compatibleEntitySpace()
            )
            experiment_references = [
                ExperimentReference(
                    experimentIdentifier=e.experimentIdentifier,
                    actuatorIdentifier=e.actuatorIdentifier,
                )
                for e in experiment_references
            ]

            model_instance = DiscoverySpaceConfiguration(
                sampleStoreIdentifier="ID",
                entitySpace=entity_space.constitutiveProperties,
                experiments=experiment_references,
            )
        else:
            model_instance = DiscoverySpaceConfiguration(sampleStoreIdentifier="ID")

    serialise_pydantic_model(
        model=model_instance,
        output_path=parameters.output_path,
    )

    if parameters.include_schema:
        schema_output_path = pathlib.Path(parameters.output_path.stem + "_schema.yaml")
        serialise_pydantic_model_json_schema(model_instance, schema_output_path)
