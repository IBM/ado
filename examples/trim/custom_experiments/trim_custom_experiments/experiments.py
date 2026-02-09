# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

from typing import Literal

from orchestrator.modules.actuators.custom_experiments import custom_experiment
from orchestrator.schema.domain import PropertyDomain, VariableTypeEnum
from orchestrator.schema.property import ConstitutiveProperty

# ---------------------------
# Properties
# ---------------------------

mass = ConstitutiveProperty(
    identifier="mass",  # expected as function arg name
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE, domainRange=[1, 100]
    ),
)

volume = ConstitutiveProperty(
    identifier="volume",  # expected as function arg name
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE, domainRange=[1, 100]
    ),
)

temperature = ConstitutiveProperty(
    identifier="temperature",  # expected as function arg name
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE, domainRange=[1, 400]
    ),
)

mol = ConstitutiveProperty(
    identifier="mol",  # expected as function arg name
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE, domainRange=[0.01, 10]
    ),
)

# ---------------------------
# Ideal gas pressure experiment
# ---------------------------


@custom_experiment(
    required_properties=[mol, temperature, volume],
    output_property_identifiers=["pressure"],
)
def calculate_pressure_ideal_gas(
    mol: float, temperature: float, volume: float
) -> dict[Literal["pressure"], float]:
    """
    Compute pressure from the Ideal Gas Law:
        p = n * R * T / V
    where:
        n = mol (mol)
        R = 8.314462618 J/(mol·K)
        T = temperature (K)
        V = volume (m^3 or consistent arbitrary units)

    For discrete inputs, the law applies pointwise.
    """
    R = 8.314462618  # J/(mol·K)
    pressure = (mol * R * temperature) / volume

    return {"pressure": pressure}


# ---------------------------
# Real vs Ideal gas pressure experiment (continuous, SI units)
# ---------------------------

gas = ConstitutiveProperty(
    identifier="gas",
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.CATEGORICAL_VARIABLE_TYPE,
        values=["ideal", "CO2", "N2", "O2", "CH4", "NH3", "H2", "He", "Ar"],
    ),
)


@custom_experiment(
    required_properties=[mol, temperature, volume, gas],
    output_property_identifiers=["pressure"],
)
def calculate_pressure_gas(
    mol: float,
    temperature: float,
    volume: float,
    gas: Literal["ideal", "CO2", "N2", "O2", "CH4", "NH3", "H2", "He", "Ar"],
) -> dict[Literal["pressure"], float]:
    """
    Compute pressure using Van der Waals equation in SI units (no checks):
        P = (n R T) / (V - n b) - a (n / V)^2

    Args:
      mol (n): Amount of substance of the gas in moles (mol)
      temperature (T): Temperature of the gas in Kelvin (K)
      volume (V): Volume of the gas in cubic meters (m³)
      gas: Identifier of the gas (one of the literals: "ideal", "CO2", "N2", "O2", "CH4", "NH3", "H2", "He", "Ar").
           Used to determine the Van der Waals constants to be used.

    Returns:
      dict: Dictionary containing the calculated pressure in Pascals (Pa)
    """

    # Van der Waals constants in SI:
    # a [Pa·m^6·mol^-2], b [m^3·mol^-1]
    # Converted from common tables (a in L^2·atm·mol^-2, b in L·mol^-1) via:
    #   a_SI = a_atmL2 * 101325 * (1e-3)^2
    #   b_SI = b_L * 1e-3
    LOOKUP_TABLE = {
        "ideal": {"a": 0.0, "b": 0.0},
        "CO2": {"a": 0.364, "b": 4.27e-5},
        "N2": {"a": 0.139, "b": 3.91e-5},
        "O2": {"a": 0.136, "b": 3.18e-5},
        "CH4": {"a": 0.225, "b": 4.28e-5},
        "NH3": {"a": 0.417, "b": 3.71e-5},
        "H2": {"a": 0.0244, "b": 2.66e-5},
        "He": {"a": 0.00341, "b": 2.37e-5},
        "Ar": {"a": 0.134, "b": 3.22e-5},
    }

    # OTHER UNIT SYSTEM
    # a in L²·atm·mol⁻², bbb in L·mol⁻¹,
    # LOOKUP_TABLE = {
    #     "ideal": {"a": 0.0,    "b": 0.0},
    #     "CO2":   {"a": 3.59,   "b": 0.0427},
    #     "N2":    {"a": 1.39,   "b": 0.0391},
    #     "O2":    {"a": 1.36,   "b": 0.0318},
    #     "CH4":   {"a": 2.25,   "b": 0.0428},
    #     "NH3":   {"a": 4.17,   "b": 0.0371},
    #     "H2":    {"a": 0.244,  "b": 0.0266},
    #     "He":    {"a": 0.0341, "b": 0.0237},
    #     "Ar":    {"a": 1.34,   "b": 0.0322},
    # }

    R = 8.314462618  # Pa·m^3·mol^-1·K^-1 (SI)
    n = mol
    V = volume
    T = temperature

    params = LOOKUP_TABLE.get(gas, LOOKUP_TABLE["ideal"])
    a = params["a"]
    b = params["b"]

    # Van der Waals pressure
    pressure = (n * R * T) / (V - n * b) - a * (n / V) ** 2

    return {"pressure": pressure}
