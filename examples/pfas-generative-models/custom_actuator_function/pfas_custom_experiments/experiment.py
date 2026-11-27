# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

from ado.modules.actuators.custom_experiments import custom_experiment
from ado.schema.observed_property import ObservedProperty
from ado.schema.property import AbstractPropertyDescriptor
from ado.schema.reference import ExperimentReference

pka = ObservedProperty(
    targetProperty=AbstractPropertyDescriptor(identifier="pka"),
    experimentReference=ExperimentReference(
        experimentIdentifier="transformer-toxicity-inference-experiment",
        actuatorIdentifier="replay",
    ),
)


@custom_experiment(
    required_properties=[pka],
    output_property_identifiers=["isAcid"],
)
def acid_test(pka: float) -> dict[str, int]:
    """Classify a molecule as an acid from its pKa.

    Args:
        pka: The molecule's pKa, measured by the replay actuator's
            transformer-toxicity-inference-experiment experiment.

    Returns:
        A dictionary with the key "isAcid": 1 if pka is negative, otherwise 0.
    """
    return {"isAcid": 0 if pka >= 0 else 1}
