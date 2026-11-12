# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""CLI tests for the run_experiment command."""

from pathlib import Path

import pytest
import typer.core
import typer.main
import yaml
from typer.testing import CliRunner

from ado.modules.actuators import custom_experiments
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


@pytest.mark.parametrize(
    "point_file",
    [_DENSITY_POINT, _OPT_FUNCTIONS_POINT],
)
def test_run_experiment_no_ray_point_yaml_succeeds(point_file: str) -> None:
    """run_experiment --no-ray must exit 0 without starting Ray actors."""
    runner = CliRunner()
    result = runner.invoke(app, ["--no-ray", point_file])
    assert result.exit_code == 0, result.output


def test_run_experiment_no_ray_rejects_sandbox_venv_experiment(
    tmp_path: Path,
) -> None:
    """--no-ray must refuse experiments that require a Ray sandbox venv."""

    @custom_experiments.custom_experiment(
        output_property_identifiers=["y"],
        ray_options={"runtime_env": {"pip": ["requests"]}},
    )
    def run_experiment_sandbox_venv_exp(x: float) -> dict[str, float]:
        return {"y": x}

    point_file = tmp_path / "point.yaml"
    point_file.write_text(
        yaml.dump(
            {
                "entity": {"x": 1.0},
                "experiments": [
                    {
                        "actuatorIdentifier": "custom_experiments",
                        "experimentIdentifier": "run_experiment_sandbox_venv_exp",
                    }
                ],
            }
        )
    )

    runner = CliRunner()
    result = runner.invoke(app, ["--no-ray", str(point_file)])
    assert result.exit_code != 0, result.output
    assert "sandbox Ray venv" in result.output
    # Keep the decorated function referenced so the catalog entry stays meaningful.
    assert run_experiment_sandbox_venv_exp._is_custom_experiment


def test_run_experiment_no_ray_incompatible_with_remote() -> None:
    """--no-ray and --remote cannot be combined."""
    runner = CliRunner()
    result = runner.invoke(
        app, ["--no-ray", "--remote", "http://example.com", _DENSITY_POINT]
    )
    assert result.exit_code != 0, result.output
    assert "--no-ray" in result.output
    assert "--remote" in result.output


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


def test_run_experiment_no_ray_option_is_recognised() -> None:
    """--no-ray must be accepted, as documented."""
    runner = CliRunner()
    result = runner.invoke(app, ["--no-ray", "--help"])
    assert result.exit_code == 0, result.output
    assert "--no-ray" in result.output
