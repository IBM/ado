# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""CLI tests for the run_experiment command."""

import os
from collections.abc import Callable
from pathlib import Path

import pytest
import ray
import typer.core
import typer.main
from typer.testing import CliRunner

from ado.metastore.project import ProjectContext
from ado.modules.actuators import custom_experiments
from ado.schema.point import SpacePoint
from ado.schema.reference import ExperimentReference
from ado.utilities.output import pydantic_model_as_yaml
from ado.utilities.run_experiment import app

_DENSITY_POINT = "examples/density_example/point.yaml"
_OPT_FUNCTIONS_POINT = "examples/optimization_test_functions/point.yaml"


def _isolate_custom_experiment_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Register test experiments on copies restored when the test ends.

    The shared catalog is loaded first. Restoring the copies then keeps those
    experiments and drops only the ones this test adds.

    Args:
        monkeypatch: Pytest monkeypatch fixture for the calling test.
    """
    custom_experiments.CustomExperiments.catalog()
    catalog = custom_experiments._custom_experiments_catalog
    monkeypatch.setattr(catalog, "_experiments", dict(catalog._experiments))
    monkeypatch.setattr(
        custom_experiments,
        "_custom_experiment_functions",
        dict(custom_experiments._custom_experiment_functions),
    )


def _write_point_file(
    directory: Path,
    entity: dict[str, object],
    actuator_identifier: str,
    experiment_identifier: str,
) -> Path:
    """Write a point file that runs one experiment on one entity.

    Args:
        directory: Directory to write the file into.
        entity: Constitutive property values for the point.
        actuator_identifier: Actuator that provides the experiment.
        experiment_identifier: Experiment to run.

    Returns:
        Path of the written point file.
    """
    point_file = directory / "point.yaml"
    point_file.write_text(
        pydantic_model_as_yaml(
            SpacePoint(
                entity=entity,
                experiments=[
                    ExperimentReference(
                        actuatorIdentifier=actuator_identifier,
                        experimentIdentifier=experiment_identifier,
                    )
                ],
            )
        )
    )
    return point_file


@pytest.mark.parametrize(
    "point_file",
    [_DENSITY_POINT, _OPT_FUNCTIONS_POINT],
)
def test_run_experiment_point_yaml_succeeds(point_file: str) -> None:
    """run_experiment <point.yaml> must exit 0 and report a valid measurement."""
    runner = CliRunner()
    result = runner.invoke(app, [point_file])
    assert result.exit_code == 0, result.output


def test_run_experiment_no_ray_point_yaml_succeeds() -> None:
    """run_experiment --no-ray must exit 0 for the optimization example point."""
    runner = CliRunner()
    result = runner.invoke(app, ["--no-ray", _OPT_FUNCTIONS_POINT])
    assert result.exit_code == 0, result.output


def test_run_experiment_no_ray_runs_in_calling_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--no-ray runs the experiment in the calling process."""
    _isolate_custom_experiment_registry(monkeypatch)

    @custom_experiments.custom_experiment(output_property_identifiers=["pid"])
    def run_experiment_pid_exp(x: float) -> dict[str, int]:
        return {"pid": os.getpid()}

    point_file = _write_point_file(
        tmp_path,
        {"x": 1.0},
        "custom_experiments",
        "run_experiment_pid_exp",
    )

    runner = CliRunner()
    result = runner.invoke(app, ["--no-ray", str(point_file)])
    assert result.exit_code == 0, result.output
    assert str(os.getpid()) in result.output


def test_run_experiment_no_ray_does_not_initialise_ray() -> None:
    """--no-ray must leave Ray uninitialised when it was not already running."""
    was_initialized = ray.is_initialized()
    if was_initialized:
        ray.shutdown()
    try:
        runner = CliRunner()
        result = runner.invoke(app, ["--no-ray", _DENSITY_POINT])
        assert result.exit_code == 0, result.output
        assert not ray.is_initialized()
    finally:
        if was_initialized:
            ray.init(
                ignore_reinit_error=True,
                object_store_memory=200 * 1024 * 1024,
            )


def test_run_experiment_no_ray_rejects_non_standard_actuator(tmp_path: Path) -> None:
    """--no-ray must refuse actuators that are not StandardActuator subclasses."""
    point_file = _write_point_file(tmp_path, {"x": 1.0}, "replay", "replay_experiment")

    runner = CliRunner()
    result = runner.invoke(app, ["--no-ray", "--no-validate", str(point_file)])
    assert result.exit_code == 1, result.output
    assert "'replay'" in result.output
    assert "StandardActuator" in result.output


def test_run_experiment_no_ray_rejects_sandbox_venv_experiment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--no-ray must refuse experiments that require a Ray sandbox venv."""
    _isolate_custom_experiment_registry(monkeypatch)

    @custom_experiments.custom_experiment(
        output_property_identifiers=["y"],
        ray_options={"runtime_env": {"pip": ["requests"]}},
    )
    def run_experiment_sandbox_venv_exp(x: float) -> dict[str, float]:
        return {"y": x}

    point_file = _write_point_file(
        tmp_path,
        {"x": 1.0},
        "custom_experiments",
        "run_experiment_sandbox_venv_exp",
    )

    runner = CliRunner()
    result = runner.invoke(app, ["--no-ray", str(point_file)])
    assert result.exit_code == 1, result.output
    assert "sandbox Ray venv" in result.output


def test_run_experiment_no_ray_reports_actuator_initialization_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--no-ray reports an actuator that cannot be constructed and skips it."""

    class RequiredFieldParameters(custom_experiments.CustomExperimentsParameters):
        required_value: str

    monkeypatch.setattr(
        custom_experiments.CustomExperiments,
        "parameters_class",
        RequiredFieldParameters,
    )

    runner = CliRunner()
    result = runner.invoke(app, ["--no-ray", _DENSITY_POINT])
    assert result.exit_code == 0, result.output
    assert "Failed to initialize actuator" in result.output
    assert "Measurement request failed unexpectedly" in result.output


def test_run_experiment_no_ray_reports_execute_failure(tmp_path: Path) -> None:
    """--no-ray reports an exception from execute() and skips the experiment."""
    point_file = _write_point_file(
        tmp_path,
        {"x": 1.0},
        "custom_experiments",
        "run_experiment_unknown_exp",
    )

    runner = CliRunner()
    result = runner.invoke(app, ["--no-ray", "--no-validate", str(point_file)])
    assert result.exit_code == 0, result.output
    assert "Failed to submit measurement request" in result.output
    assert "Measurement request failed unexpectedly" in result.output


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


def test_run_experiment_no_ray_missing_actuator_configuration(
    tmp_path: Path,
    valid_ado_sqlite_project_context: ProjectContext,
    create_active_ado_context: Callable[[CliRunner, Path, ProjectContext], None],
) -> None:
    """--no-ray must fail when the actuator configuration identifier is absent."""

    create_active_ado_context(CliRunner(), tmp_path, valid_ado_sqlite_project_context)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "--no-ray",
            "--actuator-config-id",
            "missing-config",
            _DENSITY_POINT,
        ],
    )
    assert result.exit_code != 0, result.output
    assert result.exception is not None
    assert "missing-config" in str(result.exception)


def test_run_experiment_timeout_option_is_recognised() -> None:
    """--timeout must be accepted, as documented."""
    runner = CliRunner()
    result = runner.invoke(app, ["--timeout", "5", "--help"])
    assert result.exit_code == 0, result.output
