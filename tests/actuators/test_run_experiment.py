# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""CLI tests for the run_experiment command."""

import pytest
import typer.core
import typer.main
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


def test_run_experiment_option_names_are_lowercase() -> None:
    """Every option must be lowercase, so a metavar cannot leak into its name."""
    command = typer.main.get_command(app)
    options = [
        option
        for parameter in command.params
        if isinstance(parameter, typer.core.TyperOption)
        for option in parameter.opts + parameter.secondary_opts
    ]
    assert options
    assert [option for option in options if option != option.lower()] == []


def test_run_experiment_timeout_option_is_recognised() -> None:
    """--timeout must be accepted, as documented."""
    runner = CliRunner()
    result = runner.invoke(app, ["--timeout", "5", "--help"])
    assert result.exit_code == 0, result.output
