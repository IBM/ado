# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""CLI tests for the run_experiment command."""

import pytest
from typer.testing import CliRunner

from ado.utilities.run_experiment import app

_DENSITY_POINT = "examples/density_example/point.yaml"
_OPT_FUNCTIONS_POINT = "examples/optimization_test_functions/point.yaml"


@pytest.mark.parametrize(
    "point_file",
    [_DENSITY_POINT, _OPT_FUNCTIONS_POINT],
)
def test_run_experiment_point_yaml_succeeds(point_file: str) -> None:
    """run_experiment <point.yaml> must exit 0 and report a valid measurement."""
    runner = CliRunner()
    result = runner.invoke(app, [point_file])
    assert result.exit_code == 0, result.output
