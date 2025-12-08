# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

from typing import Any

from orchestrator.modules.actuators.custom_experiments import custom_experiment


@custom_experiment(output_property_identifiers=["density"])
def calculate_density(mass: float, volume: float) -> dict[str, Any]:
    density_value = mass / volume if volume else None
    return {"density": density_value}
