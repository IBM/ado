# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import re

import typer

from ado.cli.utils.output.prints import ERROR, console_print, magenta
from ado.modules.actuators.errors import (
    AmbiguousExperimentIdentifierError,
    NoActuatorWithExperimentError,
    TooManyActuatorsWithExperimentError,
    UnknownExperimentError,
)
from ado.modules.actuators.registry import ActuatorRegistry
from ado.schema.experiment import Experiment
from ado.schema.reference import ExperimentReference
from ado.utilities.pydantic import _STRICT_SEMVER_PATTERN

_MAJOR_VERSION_PATTERN = re.compile(r"^v(0|[1-9]\d*)$")


def parse_cli_experiment_id(
    s: str,
) -> tuple[str | None, str, str | None]:
    """Parse a CLI experiment identifier into its constituent parts.

    Accepts the following forms:
    - ``experiment`` — bare name, no actuator prefix, no version
    - ``experiment@1.0.0`` — bare name with SemVer version
    - ``experiment@v1`` — bare name with major-version shorthand
    - ``actuator.experiment`` — fully-qualified, no version
    - ``actuator.experiment@1.0.0`` — fully-qualified with SemVer version
    - ``actuator.experiment@v1`` — fully-qualified with major-version shorthand

    ``@v<N>`` is normalised to ``"<N>.0.0"`` so the result can be passed
    directly to ``ExperimentReference(experimentVersion=...)``.

    Args:
        s: The CLI identifier string to parse.

    Returns:
        A tuple of ``(actuator_id_or_None, experiment_id, version_or_None)``.

    Raises:
        ValueError: If the ``@`` suffix is present but not a valid SemVer or
            major-version identifier.
    """
    # Split actuator prefix on the first '.' only when it precedes any '@'.
    dot_index = s.find(".")
    at_index = s.find("@")
    if dot_index != -1 and (at_index == -1 or dot_index < at_index):
        actuator_id: str | None = s[:dot_index]
        remainder = s[dot_index + 1 :]
    else:
        actuator_id = None
        remainder = s

    # Split optional version suffix on '@'.
    if "@" in remainder:
        experiment_id, version_str = remainder.split("@", maxsplit=1)
        if _STRICT_SEMVER_PATTERN.match(version_str):
            version: str | None = version_str
        else:
            major_match = _MAJOR_VERSION_PATTERN.match(version_str)
            if major_match:
                version = f"{major_match.group(1)}.0.0"
            else:
                raise ValueError(
                    f"Invalid version suffix '@{version_str}' in {s!r}. "
                    "Version must be SemVer MAJOR.MINOR.PATCH (e.g. @1.0.0) "
                    "or a major version identifier (e.g. @v1)."
                )
    else:
        experiment_id = remainder
        version = None

    return actuator_id, experiment_id, version


def _ado_lookup_cli_experiment(
    resource_id: str,
    *,
    registry: ActuatorRegistry | None = None,
) -> Experiment:
    """Parse CLI input, look up the experiment, and exit on lookup failure."""
    try:
        actuator_id, experiment_id, version = parse_cli_experiment_id(resource_id)
    except ValueError as error:
        console_print(f"{ERROR}{error}", stderr=True)
        raise typer.Exit(1) from error

    try:
        registry = registry or ActuatorRegistry.globalRegistry()

        if actuator_id is None:
            # Bare name: search all actuators for the experiment.
            actuators_with_experiment = (
                registry.actuators_containing_experiment_with_base_identifier(
                    experiment_id
                )
            )
            if len(actuators_with_experiment) == 0:
                raise NoActuatorWithExperimentError
            if len(actuators_with_experiment) > 1:
                raise TooManyActuatorsWithExperimentError(actuators_with_experiment)
            actuator_id = next(iter(actuators_with_experiment))

        reference = ExperimentReference(
            actuatorIdentifier=actuator_id,
            experimentIdentifier=experiment_id,
            experimentVersion=version,
        )
        return registry.experimentForReference(reference, match_on="any")

    except NoActuatorWithExperimentError as error:
        console_print(
            f"{ERROR}Experiment {magenta(experiment_id)} does not exist",
            stderr=True,
        )
        raise typer.Exit(1) from error
    except TooManyActuatorsWithExperimentError as error:
        sorted_actuators = sorted(error.actuators_with_experiments)
        example = f"{sorted_actuators[0]}.{experiment_id}"
        console_print(
            f"{ERROR}Experiment {magenta(experiment_id)} was found in "
            f"multiple actuators: {sorted_actuators}. "
            f"Specify the actuator in the resource id, e.g. {example}",
            stderr=True,
        )
        raise typer.Exit(1) from error
    except (UnknownExperimentError, AmbiguousExperimentIdentifierError) as error:
        console_print(f"{ERROR}{error}", stderr=True)
        raise typer.Exit(1) from error
    except ValueError as error:
        console_print(f"{ERROR}{error}", stderr=True)
        raise typer.Exit(1) from error
