# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Ideal-gas pressure experiment for adaptive sampling examples."""

from typing import Literal

from ado.modules.actuators.custom_experiments import custom_experiment
from ado.schema.domain import PropertyDomain, VariableTypeEnum
from ado.schema.property import ConstitutiveProperty


@custom_experiment(
    required_properties=[
        ConstitutiveProperty(
            identifier="mol",
            propertyDomain=PropertyDomain(
                variableType=VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE,
                domainRange=[0.01, 10],
            ),
        ),
        ConstitutiveProperty(
            identifier="temperature",
            propertyDomain=PropertyDomain(
                variableType=VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE,
                domainRange=[1, 400],
            ),
        ),
        ConstitutiveProperty(
            identifier="volume",
            propertyDomain=PropertyDomain(
                variableType=VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE,
                domainRange=[1, 100],
            ),
        ),
    ],
    output_property_identifiers=["pressure"],
)
def ideal_gas_pressure(
    mol: float, temperature: float, volume: float
) -> dict[Literal["pressure"], float]:
    """Calculate ideal-gas pressure in pascals.

    Args:
        mol: Amount of gas in moles.
        temperature: Temperature in kelvin.
        volume: Volume in cubic metres.

    Returns:
        Calculated pressure keyed by ``pressure``.
    """
    return {"pressure": mol * 8.314462618 * temperature / volume}
