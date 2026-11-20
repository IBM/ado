# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import zlib
from typing import Literal

import numpy as np

from ado.modules.actuators.custom_experiments import custom_experiment
from ado.schema.domain import PropertyDomain, VariableTypeEnum
from ado.schema.property import ConstitutiveProperty

acid_concentration = ConstitutiveProperty(
    identifier="acid_concentration",
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE,
        domainRange=[0.001, 1.0],
    ),
)

base_concentration = ConstitutiveProperty(
    identifier="base_concentration",
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE,
        domainRange=[0.001, 1.0],
    ),
)

titrant_volume = ConstitutiveProperty(
    identifier="titrant_volume",
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE,
        domainRange=[0.0, 100.0],
    ),
)

SAMPLE_VOLUME_ML = 25.0
WATER_ION_PRODUCT = 1e-14  # Kw at 25 °C
PH_METER_NOISE = 0.05  # standard deviation of the pH reading


@custom_experiment(
    required_properties=[
        acid_concentration,
        base_concentration,
        titrant_volume,
    ],
    output_property_identifiers=["ph"],
    metadata={
        "description": (
            "Simulated pH of a 25 mL strong acid sample titrated with "
            "a strong base at 25 °C."
        ),
    },
)
def titrate_strong_acid(
    acid_concentration: float, base_concentration: float, titrant_volume: float
) -> dict[Literal["ph"], float]:
    """Read the pH of a strong acid titrated with a strong base.

    The equilibrium equations are:
        c = (c_acid * V_s - c_base * V_titrant) / (V_s + V_titrant)
        [H+] = (c + sqrt(c^2 + 4 * Kw)) / 2
        pH = -log10([H+]) + noise

    V_s is 25 mL and Kw is 1e-14 at 25 °C. The equivalence point is at
    c_acid * V_s / c_base mL of titrant.

    The pH meter adds Gaussian noise with a standard deviation of
    0.05 pH units. The noise is seeded from the inputs, so the same
    configuration always gives the same reading.

    Args:
        acid_concentration: Concentration of the acid sample in mol/L.
        base_concentration: Concentration of the base titrant in mol/L.
        titrant_volume: Volume of base added to the acid sample in mL.

    Returns:
        The simulated pH reading under the "ph" key.
    """
    acid_mmol = acid_concentration * SAMPLE_VOLUME_ML
    base_mmol = base_concentration * titrant_volume
    net_acid = (acid_mmol - base_mmol) / (SAMPLE_VOLUME_ML + titrant_volume)
    hydrogen = (net_acid + np.sqrt(net_acid**2 + 4 * WATER_ION_PRODUCT)) / 2

    # A volume of 0, 0.0 or -0.0 represents the same meter reading.
    seed_inputs = (
        float(acid_concentration),
        float(base_concentration),
        0.0 if titrant_volume == 0 else float(titrant_volume),
    )
    seed = zlib.crc32(repr(seed_inputs).encode())
    noise = np.random.default_rng(seed).normal(scale=PH_METER_NOISE)

    return {"ph": float(-np.log10(hydrogen) + noise)}
