# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import pathlib
from collections.abc import Callable

import yaml
from typer.testing import CliRunner

from orchestrator.cli.core.cli import app as ado
from orchestrator.metastore.project import ProjectContext


def test_create_actuator_configuration_dry_run_success(tmp_path: pathlib.Path) -> None:
    actuator_configuration_file = (
        "tests/resources/actuatorconfiguration/robotic_lab.yaml"
    )
    runner = CliRunner()
    result = runner.invoke(
        ado,
        [
            "--override-ado-app-dir",
            tmp_path,
            "create",
            "actuatorconfiguration",
            "-f",
            actuator_configuration_file,
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    expected_output = (
        "INFO:   Initializing contexts - local is now your default context.\n"
        "The configuration passed is valid!\n"
    )
    assert result.output == expected_output


def test_create_actuator_configuration_dry_run_failure(tmp_path: pathlib.Path) -> None:
    actuator_configuration_file = pathlib.Path(
        "tests/resources/actuatorconfiguration/robotic_lab.yaml"
    )

    invalid_actuator_configuration_file = tmp_path / "invalid.yaml"
    actuator_configuration = yaml.safe_load(actuator_configuration_file.read_text())
    actuator_configuration["actuatorIdentifier"] = "robotic-lab-fake"
    invalid_actuator_configuration_file.write_text(
        yaml.safe_dump(actuator_configuration)
    )

    runner = CliRunner()
    result = runner.invoke(
        ado,
        [
            "--override-ado-app-dir",
            tmp_path,
            "create",
            "actuatorconfiguration",
            "-f",
            invalid_actuator_configuration_file,
            "--dry-run",
        ],
    )
    assert result.exit_code == 1
    assert result.output.strip().startswith(
        "INFO:   Initializing contexts - local is now your default context.\n"
        "ERROR:  The actuatorconfiguration provided was not valid:"
    )


def test_create_actuator_configuration(
    tmp_path: pathlib.Path,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
) -> None:
    actuator_configuration_file = (
        "tests/resources/actuatorconfiguration/robotic_lab.yaml"
    )

    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )
    result = runner.invoke(
        ado,
        [
            "--override-ado-app-dir",
            tmp_path,
            "create",
            "actuatorconfiguration",
            "-f",
            actuator_configuration_file,
        ],
    )

    assert result.exit_code == 0
    expected_output = "Success! Created actuator configuration with identifier"
    assert result.output.startswith(expected_output)
