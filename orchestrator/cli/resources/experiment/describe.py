# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import typer
from rich.status import Status

from orchestrator.cli.models.parameters import AdoDescribeCommandParameters
from orchestrator.cli.utils.output.prints import (
    ADO_SPINNER_INITIALIZING_ACTUATOR_REGISTRY,
    ERROR,
    HINT,
    console_print,
)
from orchestrator.cli.utils.resources.experiments import (
    _ado_get_actuator_from_experiment_id,
)
from orchestrator.modules.actuators.errors import (
    AmbiguousExperimentIdentifierError,
    UnknownExperimentError,
)
from orchestrator.modules.actuators.registry import (
    ActuatorRegistry,
)
from orchestrator.schema.reference import (
    ExperimentReference,
    _parse_experiment_part_from_string,
)


def describe_experiment(parameters: AdoDescribeCommandParameters) -> None:

    with Status(ADO_SPINNER_INITIALIZING_ACTUATOR_REGISTRY):
        registry = ActuatorRegistry.globalRegistry()

    if (
        parameters.actuator_id
        and parameters.actuator_id not in registry.actuatorIdentifierMap
    ):
        console_print(
            f"{ERROR}Actuator {parameters.actuator_id} does not exist.\n"
            f"{HINT}Available ones are: {list(registry.actuatorIdentifierMap.keys())}",
            stderr=True,
        )
        raise typer.Exit(1)

    try:
        if parameters.actuator_id is None:
            base_experiment_identifier, _, _ = _parse_experiment_part_from_string(
                parameters.resource_id
            )
            actuator_id = _ado_get_actuator_from_experiment_id(
                experiment_id=base_experiment_identifier,
                actuator_id=None,
            )
        else:
            actuator_id = parameters.actuator_id

        # Need to use referenceFromString in case resource id contains version
        reference = ExperimentReference.referenceFromString(
            f"{actuator_id}.{parameters.resource_id}"
        )
    except ValueError as error:
        console_print(f"{ERROR}{error}", stderr=True)
        raise typer.Exit(1) from error

    if reference.experimentVersion is not None:
        experiment = registry.experimentForReference(
            reference, match_on="fully_qualified_version"
        )
    else:
        catalog = registry.catalogForActuatorIdentifier(reference.actuatorIdentifier)
        matches = catalog.experiments_matching_identifier(reference)
        if len(matches) == 0:
            raise UnknownExperimentError(
                f"The {reference.actuatorIdentifier} actuator was found but it did not "
                f"contain the {reference.experimentIdentifier} experiment."
            )
        if len(matches) > 1:
            available_versions = ", ".join(
                sorted({e.version for e in matches if e.version is not None})
            )
            raise AmbiguousExperimentIdentifierError(
                f"The given identifier, {reference.experimentIdentifier!r}, is ambiguous: "
                f"catalog contains {len(matches)} versions "
                f"({available_versions}). "
                f"Specify a version suffix, e.g. "
                f"{reference.experimentIdentifier}@<version>."
            )
        experiment = matches[0]

    console_print(experiment)
