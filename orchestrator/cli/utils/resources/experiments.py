# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import typer

from orchestrator.cli.exceptions.actuators import (
    NoActuatorWithExperimentError,
    TooManyActuatorsWithExperimentError,
)
from orchestrator.cli.utils.output.prints import ERROR, console_print, magenta
from orchestrator.modules.actuators.errors import (
    AmbiguousExperimentIdentifierError,
    UnknownExperimentError,
)
from orchestrator.modules.actuators.registry import ActuatorRegistry
from orchestrator.schema.experiment import Experiment
from orchestrator.schema.reference import (
    ExperimentReference,
    _parse_experiment_part_from_string,
)


def _split_resource_id_prefix(resource_id: str) -> tuple[str | None, str]:
    """Split ``resource_id`` on the first ``.`` unless ``@`` precedes it."""
    dot_index = resource_id.find(".")
    if dot_index == -1:
        return None, resource_id

    at_index = resource_id.find("@")
    if at_index != -1 and at_index < dot_index:
        return None, resource_id

    return resource_id[:dot_index], resource_id[dot_index + 1 :]


def get_actuators_implementing_experiment(experiment_id: str) -> set[str]:
    """
    Returns a set of actuators that implement a given experiment.

    Args:
        experiment_id (str): The identifier of the experiment.

    Returns:
        set[str]: A set of actuator identifiers that implement the experiment.
    """
    registry = ActuatorRegistry.globalRegistry()
    actuators_with_target_experiment: set[str] = set()

    for actuator_id in registry.actuatorIdentifierMap:
        catalog = registry.catalogForActuatorIdentifier(actuator_id)
        for experiment in catalog.experiments:
            if experiment.identifier == experiment_id:
                actuators_with_target_experiment.add(actuator_id)

    return actuators_with_target_experiment


def get_actuators_from_experiment_id(
    experiment_id: str,
) -> set[str]:
    """
    Retrieves a set of actuators that implement a given experiment ID.

    Args:
        experiment_id (str): The ID of the experiment.

    Returns:
        set[str]: A set of actuator identifiers that implement the experiment.

    Raises:
        NoActuatorWithExperimentError: If no actuators implement the experiment.
    """
    actuators_with_target_experiment = get_actuators_implementing_experiment(
        experiment_id
    )

    if len(actuators_with_target_experiment) == 0:
        raise NoActuatorWithExperimentError
    return actuators_with_target_experiment


def get_actuator_from_experiment_id(experiment_id: str) -> str:
    """
    Retrieves the ID of the actuator that implements an experiment ID.

    Parameters:
    - experiment_id (str): The experiment ID to retrieve the actuator ID for.

    Returns:
    str: The actuator ID that implements the experiment.

    Raises:
    NoActuatorWithExperimentError: If no actuators implement the experiment.
    TooManyActuatorsWithExperimentError: If multiple actuators implement the experiment.
    """
    actuators_with_target_experiment = get_actuators_from_experiment_id(experiment_id)

    if len(actuators_with_target_experiment) > 1:
        raise TooManyActuatorsWithExperimentError(actuators_with_target_experiment)

    return next(iter(actuators_with_target_experiment))


def experiment_reference_from_cli_resource_id(
    resource_id: str,
    *,
    registry: ActuatorRegistry | None = None,
) -> ExperimentReference:
    """Build an experiment reference from a CLI experiment resource identifier.

    Args:
        resource_id: Bare experiment identifier, versioned identifier with an
            ``@MAJOR.MINOR.PATCH`` suffix, or a fully-qualified reference
            string of the form ``actuator_id.experiment_id``.
        registry: Optional actuator registry. Defaults to the global registry.

    Returns:
        Parsed experiment reference.

    Raises:
        NoActuatorWithExperimentError: If no actuator implements the experiment.
        TooManyActuatorsWithExperimentError: If multiple actuators implement the
            experiment.
        ValueError: If ``resource_id`` is not a valid experiment reference string.
    """
    registry = registry or ActuatorRegistry.globalRegistry()

    prefix, _ = _split_resource_id_prefix(resource_id)
    if prefix is not None and prefix in registry.actuatorIdentifierMap:
        return ExperimentReference.referenceFromString(
            resource_id, allow_parameterization=False
        )

    base_experiment_identifier, _, _ = _parse_experiment_part_from_string(
        resource_id, allow_parameterization=False
    )
    actuator_id = get_actuator_from_experiment_id(base_experiment_identifier)
    return ExperimentReference.referenceFromString(
        f"{actuator_id}.{resource_id}",
        allow_parameterization=False,
    )


def lookup_experiment_for_reference(
    reference: ExperimentReference,
    *,
    registry: ActuatorRegistry | None = None,
) -> Experiment:
    """Look up a catalog experiment for a parsed reference.

    Args:
        reference: Experiment reference to resolve.
        registry: Optional actuator registry. Defaults to the global registry.

    Returns:
        Matching catalog experiment.

    Raises:
        UnknownExperimentError: If the experiment is not present in the catalog.
        AmbiguousExperimentIdentifierError: If multiple catalog versions match.
    """
    registry = registry or ActuatorRegistry.globalRegistry()
    catalog = registry.catalogForActuatorIdentifier(reference.actuatorIdentifier)

    experiment = catalog.experimentForReference(
        reference, match_on="fully_qualified_version"
    )
    if experiment is not None:
        return experiment

    experiment = catalog.experimentForReference(reference, match_on="major_version")
    if experiment is not None:
        return experiment

    matches = catalog.experiments_matching_identifier(reference)
    if len(matches) == 0:
        raise UnknownExperimentError(
            f"The {reference.actuatorIdentifier} actuator was found but it did not "
            f"contain the {reference.experimentIdentifier} experiment."
        )
    if len(matches) > 1:
        available_versions = ", ".join(
            sorted({matched.version for matched in matches if matched.version})
        )
        raise AmbiguousExperimentIdentifierError(
            f"The given identifier, {reference.experimentIdentifier!r}, is ambiguous: "
            f"catalog contains {len(matches)} versions "
            f"({available_versions}). "
            f"Specify a version suffix, e.g. "
            f"{reference.experimentIdentifier}@<version>."
        )

    return matches[0]


def experiment_from_cli_resource_id(
    resource_id: str,
    *,
    registry: ActuatorRegistry | None = None,
) -> Experiment:
    """Parse a CLI experiment resource identifier and return the catalog entry."""
    reference = experiment_reference_from_cli_resource_id(
        resource_id, registry=registry
    )
    return lookup_experiment_for_reference(reference, registry=registry)


def resolved_experiment_reference_from_cli_resource_id(
    resource_id: str,
    *,
    registry: ActuatorRegistry | None = None,
) -> ExperimentReference:
    """Parse a CLI experiment resource identifier and return a resolved reference."""
    experiment = experiment_from_cli_resource_id(resource_id, registry=registry)
    return experiment.reference


def _ado_experiment_from_cli_resource_id(
    resource_id: str,
    *,
    registry: ActuatorRegistry | None = None,
) -> Experiment:
    """Parse CLI input and look up the experiment, exiting on lookup failure."""
    try:
        return experiment_from_cli_resource_id(resource_id, registry=registry)
    except NoActuatorWithExperimentError as error:
        base_experiment_identifier, _, _ = _parse_experiment_part_from_string(
            resource_id, allow_parameterization=False
        )
        console_print(
            f"{ERROR}Experiment {magenta(base_experiment_identifier)} does not exist",
            stderr=True,
        )
        raise typer.Exit(1) from error
    except TooManyActuatorsWithExperimentError as error:
        base_experiment_identifier, _, _ = _parse_experiment_part_from_string(
            resource_id, allow_parameterization=False
        )
        console_print(
            f"{ERROR}Experiment {magenta(base_experiment_identifier)} was found in "
            f"multiple actuators: {error.actuators_with_experiments}. "
            "Specify the actuator in the resource id, e.g. "
            f"{next(iter(error.actuators_with_experiments))}."
            f"{base_experiment_identifier}.",
            stderr=True,
        )
        raise typer.Exit(1) from error
    except (UnknownExperimentError, AmbiguousExperimentIdentifierError) as error:
        console_print(f"{ERROR}{error}", stderr=True)
        raise typer.Exit(1) from error
    except ValueError as error:
        console_print(f"{ERROR}{error}", stderr=True)
        raise typer.Exit(1) from error


def _ado_resolved_experiment_reference_from_cli_resource_id(
    resource_id: str,
    *,
    registry: ActuatorRegistry | None = None,
) -> ExperimentReference:
    """Parse CLI input and return a resolved reference, exiting on lookup failure."""
    try:
        return resolved_experiment_reference_from_cli_resource_id(
            resource_id, registry=registry
        )
    except NoActuatorWithExperimentError as error:
        base_experiment_identifier, _, _ = _parse_experiment_part_from_string(
            resource_id, allow_parameterization=False
        )
        console_print(
            f"{ERROR}Experiment {magenta(base_experiment_identifier)} does not exist",
            stderr=True,
        )
        raise typer.Exit(1) from error
    except TooManyActuatorsWithExperimentError as error:
        base_experiment_identifier, _, _ = _parse_experiment_part_from_string(
            resource_id, allow_parameterization=False
        )
        console_print(
            f"{ERROR}Experiment {magenta(base_experiment_identifier)} was found in "
            f"multiple actuators: {error.actuators_with_experiments}. "
            "Specify the actuator in the resource id, e.g. "
            f"{next(iter(error.actuators_with_experiments))}."
            f"{base_experiment_identifier}.",
            stderr=True,
        )
        raise typer.Exit(1) from error
    except (UnknownExperimentError, AmbiguousExperimentIdentifierError) as error:
        console_print(f"{ERROR}{error}", stderr=True)
        raise typer.Exit(1) from error
    except ValueError as error:
        console_print(f"{ERROR}{error}", stderr=True)
        raise typer.Exit(1) from error
