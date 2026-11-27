# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

from ado.modules.actuators.custom_experiments import custom_experiment
from ado.schema.domain import PropertyDomain
from ado.schema.observed_property import ObservedProperty
from ado.schema.property import AbstractPropertyDescriptor, ConstitutiveProperty
from ado.schema.reference import ExperimentReference

# Mapping cpu family to dollar per node/hour
# THESE VALUES ARE JUST FOR ILLUSTRATION
cost_map = {0: 10, 1: 20}

nodes = ConstitutiveProperty(
    identifier="nodes",
    propertyDomain=PropertyDomain(interval=1, domainRange=[0, 1000]),
)
cpu_family = ConstitutiveProperty(
    identifier="cpu_family",
    propertyDomain=PropertyDomain(values=[0, 1]),
)
wall_clock_runtime = ObservedProperty(
    targetProperty=AbstractPropertyDescriptor(identifier="wallClockRuntime"),
    experimentReference=ExperimentReference(
        experimentIdentifier="benchmark_performance",
        actuatorIdentifier="replay",
    ),
)


@custom_experiment(
    required_properties=[nodes, cpu_family, wall_clock_runtime],
    output_property_identifiers=["total_cost"],
    version="1.0.0",
)
def ml_multicloud_cost(
    nodes: int, cpu_family: int, wallClockRuntime: float
) -> dict[str, float]:
    """Calculate the cost of running a workload.

    Args:
        nodes: Number of nodes the workload ran on.
        cpu_family: CPU family of the nodes, which sets the price per node-hour.
        wallClockRuntime: Run time of the workload in seconds, measured by the
            replay actuator's benchmark_performance experiment.

    Returns:
        A dictionary with the total cost in dollars under the key "total_cost".
    """
    return {"total_cost": cost_map[cpu_family] * nodes * (wallClockRuntime / 3600.0)}
