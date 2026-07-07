# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import typer

from orchestrator.cli.utils.output.prints import ERROR, console_print, magenta
from orchestrator.modules.actuators.errors import (
    AmbiguousExperimentIdentifierError,
    NoActuatorWithExperimentError,
    TooManyActuatorsWithExperimentError,
    UnknownExperimentError,
)
from orchestrator.modules.actuators.registry import ActuatorRegistry
from orchestrator.schema.experiment import Experiment
from orchestrator.schema.reference import (
    _parse_experiment_part_from_string,
)


def _ado_lookup_cli_experiment(
    resource_id: str,
    *,
    registry: ActuatorRegistry | None = None,
) -> Experiment:
    """Parse CLI input, look up the experiment, and exit on lookup failure."""
    try:
        registry = registry or ActuatorRegistry.globalRegistry()
        return registry.experiment_for_experiment_identifier(
            resource_id, match_on="any", resolve=False
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
        sorted_actuators_with_experiment = sorted(error.actuators_with_experiments)
        example = f"{sorted_actuators_with_experiment[0]}.{base_experiment_identifier}"
        console_print(
            f"{ERROR}Experiment {magenta(base_experiment_identifier)} was found in "
            f"multiple actuators: {sorted_actuators_with_experiment}. "
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
