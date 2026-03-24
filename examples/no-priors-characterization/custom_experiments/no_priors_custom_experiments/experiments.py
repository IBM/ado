# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

from typing import Literal

import numpy as np

from orchestrator.modules.actuators.custom_experiments import custom_experiment
from orchestrator.schema.domain import PropertyDomain, VariableTypeEnum
from orchestrator.schema.property import ConstitutiveProperty

# ---------------------------
# Properties for Reaction Yield
# ---------------------------

temperature = ConstitutiveProperty(
    identifier="temperature",
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE,
        domainRange=[273, 473],  # 0-200°C in Kelvin
    ),
)

concentration = ConstitutiveProperty(
    identifier="concentration",
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE,
        domainRange=[0.1, 5.0],  # mol/L
    ),
)

catalyst_amount = ConstitutiveProperty(
    identifier="catalyst_amount",
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE,
        domainRange=[0.0, 10.0],  # grams
    ),
)

# ---------------------------
# Properties for Material Strength
# ---------------------------

composition_a = ConstitutiveProperty(
    identifier="composition_a",
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE,
        domainRange=[0, 100],  # percentage
    ),
)

composition_b = ConstitutiveProperty(
    identifier="composition_b",
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE,
        domainRange=[0, 100],  # percentage
    ),
)

temperature_celsius = ConstitutiveProperty(
    identifier="temperature_celsius",
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE,
        domainRange=[-50, 200],  # Celsius
    ),
)

grain_size = ConstitutiveProperty(
    identifier="grain_size",
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE,
        domainRange=[1, 100],  # micrometers
    ),
)

# ---------------------------
# Reaction Yield Experiment
# ---------------------------


@custom_experiment(
    required_properties=[temperature, concentration, catalyst_amount],
    output_property_identifiers=["yield"],
)
def calculate_reaction_yield(
    temperature: float, concentration: float, catalyst_amount: float
) -> dict[Literal["yield"], float]:
    """
    Calculate chemical reaction yield using Arrhenius-like equation with catalyst effect.

    The yield is calculated using:
        k = A * exp(-Ea / (R * T)) * (1 + 0.1 * catalyst_amount)
        yield = 100 * (1 - exp(-k * concentration * time))

    where:
        A = 1e10 (pre-exponential factor)
        Ea = 50000 J/mol (activation energy)
        R = 8.314 J/(mol·K) (gas constant)
        time = 3600 s (reaction time)

    Args:
        temperature: Reaction temperature in Kelvin
        concentration: Reactant concentration in mol/L
        catalyst_amount: Catalyst amount in grams

    Returns:
        dict: Dictionary containing the calculated yield as a percentage (0-100)
    """
    A = 1e10  # pre-exponential factor
    Ea = 50000  # J/mol, activation energy
    R = 8.314  # J/(mol·K), gas constant
    time = 3600  # seconds, reaction time

    # Calculate rate constant with catalyst effect
    k = A * np.exp(-Ea / (R * temperature)) * (1 + 0.1 * catalyst_amount)

    # Calculate yield
    reaction_yield = 100 * (1 - np.exp(-k * concentration * time))

    # Ensure yield is between 0 and 100
    reaction_yield = np.clip(reaction_yield, 0, 100)

    return {"yield": float(reaction_yield)}


# ---------------------------
# Material Strength Experiment
# ---------------------------


@custom_experiment(
    required_properties=[composition_a, composition_b, temperature_celsius, grain_size],
    output_property_identifiers=["tensile_strength"],
)
def calculate_material_strength(
    composition_a: float,
    composition_b: float,
    temperature_celsius: float,
    grain_size: float,
) -> dict[Literal["tensile_strength"], float]:
    """
    Calculate material tensile strength using Hall-Petch relationship with composition effects.

    The strength is calculated using:
        base_strength = composition_a * 500 + composition_b * 300 + (100 - composition_a - composition_b) * 200
        temp_factor = 1 - 0.002 * (temperature_celsius - 20)
        grain_factor = 1 + 100 / sqrt(grain_size)
        tensile_strength = base_strength * temp_factor * grain_factor / 1000

    Args:
        composition_a: Percentage of component A (0-100)
        composition_b: Percentage of component B (0-100)
        temperature_celsius: Testing temperature in Celsius
        grain_size: Grain size in micrometers

    Returns:
        dict: Dictionary containing the calculated tensile strength in MPa
    """
    # Calculate base strength from composition
    composition_c = 100 - composition_a - composition_b
    base_strength = composition_a * 500 + composition_b * 300 + composition_c * 200

    # Temperature effect (strength decreases with temperature)
    temp_factor = 1 - 0.002 * (temperature_celsius - 20)
    temp_factor = np.clip(temp_factor, 0.1, 2.0)  # Prevent unrealistic values

    # Hall-Petch relationship (strength increases with smaller grain size)
    grain_factor = 1 + 100 / np.sqrt(grain_size)

    # Calculate final tensile strength in MPa
    tensile_strength = base_strength * temp_factor * grain_factor / 1000

    # Ensure positive strength
    tensile_strength = np.maximum(tensile_strength, 0)

    return {"tensile_strength": float(tensile_strength)}
