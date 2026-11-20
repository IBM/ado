# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import pytest
from titration_custom_experiments.experiments import titrate_strong_acid


@pytest.mark.parametrize(
    ("integer_volume", "float_volume"),
    [(10, 10.0), (0, -0.0)],
)
def test_titration_noise_is_independent_of_numeric_input_types(
    integer_volume: int, float_volume: float
) -> None:
    """Equivalent integer and float volumes give the same pH."""
    integer_reading = titrate_strong_acid(0.1, 0.2, integer_volume)
    float_reading = titrate_strong_acid(0.1, 0.2, float_volume)
    assert integer_reading["ph"] == float_reading["ph"]


@pytest.mark.parametrize(
    ("volume_ml", "expected_ph"), [(0.0, 1.0), (25.0, 7.0), (50.0, 12.52)]
)
def test_ph_across_equivalence(volume_ml: float, expected_ph: float) -> None:
    """Equal acid and base concentrations reach equivalence at 25 mL."""
    reading = titrate_strong_acid(0.1, 0.1, volume_ml)
    assert reading["ph"] == pytest.approx(expected_ph, abs=0.2)
