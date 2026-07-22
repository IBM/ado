# Copyright IBM Corporation 2026
# SPDX-License-Identifier: MIT

import pytest
from ado_ray_tune.space_analysis import mi_pareto_selection


@pytest.mark.parametrize(
    "mutual_information",
    [
        {"provider": 0.0, "cpu_family": 0.0},
        {"provider": 0.0, "nodes": 0.454},
    ],
)
def test_mi_pareto_selection_with_fewer_than_two_significant_dimensions(
    mutual_information: dict[str, float],
) -> None:
    assert mi_pareto_selection(mutual_information) == []  # noqa: S101
