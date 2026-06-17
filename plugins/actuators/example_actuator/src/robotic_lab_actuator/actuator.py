# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""RoboticLab actuator — example actuator using StandardActuator.

This actuator demonstrates the simple implementation path: override
``_experiment_implementations()`` to return a mapping from experiment identifier
to a callable.  The callable receives constitutive property values as keyword
arguments and returns a dict of observed property values.
"""

import os
import typing
from typing import Annotated, Any

import numpy as np
import yaml
from pydantic import Field

from orchestrator.core.actuatorconfiguration.config import GenericActuatorParameters
from orchestrator.modules.actuators.catalog import ExperimentCatalog
from orchestrator.modules.actuators.standard import (
    StandardActuator,
    StandardActuatorParameters,
)
from orchestrator.schema.experiment import Experiment


class RoboticLabParameters(StandardActuatorParameters):
    """Configuration parameters for the RoboticLab actuator."""

    my_parameter: Annotated[str, Field()] = "hello world"


def my_experiment(**kwargs: Any) -> dict[str, Any]:  # noqa: ANN401
    """Simulate a robotic-lab experiment.

    Args:
        **kwargs: Constitutive property values for the entity under test.

    Returns:
        Dict mapping observed property identifiers to their measured values.
        Keys must match the identifiers defined for the outputs in experiments.yaml.
    """
    rng = np.random.default_rng()
    return {
        "adsorption_timeseries": [rng.random() for _ in range(10)],
        "adsorption_plateau_value": rng.random(),
    }


class RoboticLab(StandardActuator):
    """Example actuator that wraps a simple experiment function.

    Demonstrates the ``_experiment_implementations`` hook: each experiment is
    mapped to a callable that takes entity property values as kwargs and returns
    a results dict.

    Subclasses with more complex setup (e.g. external connections, environment
    managers) should capture that state in closures or ``functools.partial``
    objects and return them from ``_experiment_implementations``, or override
    ``_get_request_executor`` for full control.
    """

    identifier = "robotic_lab"
    parameters_class = RoboticLabParameters

    @classmethod
    def catalog(
        cls, actuator_configuration: GenericActuatorParameters | None = None
    ) -> ExperimentCatalog:
        """Return the experiments provided by this actuator.

        Reads experiment definitions from the experiments.yaml file co-located
        with this module.

        Args:
            actuator_configuration: Unused; accepted for interface compatibility.

        Returns:
            ExperimentCatalog with all experiments declared in experiments.yaml.
        """
        path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "experiments.yaml"
        )
        with open(path) as f:
            data = yaml.safe_load(f)
        experiments = [Experiment(**data[e]) for e in data]
        return ExperimentCatalog(
            catalogIdentifier=cls.identifier,
            experiments={e.identifier: e for e in experiments},
        )

    def _experiment_implementations(
        self,
    ) -> dict[str, typing.Callable[..., dict[str, Any]]]:
        """Return the experiment implementation mapping.

        Returns:
            Dict mapping experiment identifier to the experiment callable.
        """
        return {
            "peptide_mineralization": my_experiment,
        }
