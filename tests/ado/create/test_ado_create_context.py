# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import pathlib

import yaml
from typer.testing import CliRunner

from orchestrator.cli.core.cli import app as ado
from orchestrator.metastore.project import ProjectContext
from orchestrator.utilities.output import pydantic_model_as_yaml


def test_create_context_dry_run_success(
    tmp_path: pathlib.Path, valid_ado_mysql_context_yaml: str
) -> None:
    context_file = tmp_path / "temp_context.yaml"
    context_file.write_text(valid_ado_mysql_context_yaml)

    runner = CliRunner()
    result = runner.invoke(
        ado,
        [
            "--override-ado-app-dir",
            tmp_path,
            "create",
            "context",
            "-f",
            context_file,
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    expected_output = (
        "INFO:   Initializing contexts - local is now your default context.\n"
        "The configuration passed is valid!\n"
    )
    assert result.output == expected_output


def test_create_context_dry_run_failure(
    tmp_path: pathlib.Path, valid_ado_sqlite_context_yaml: str
) -> None:
    from orchestrator.utilities.output import pydantic_model_as_yaml

    # Changing the project name will make the context file invalid
    invalid_project_context = ProjectContext.model_validate(
        yaml.safe_load(valid_ado_sqlite_context_yaml)
    )
    invalid_project_context.project = "something-else"
    context_file = tmp_path / "temp_context.yaml"
    context_file.write_text(pydantic_model_as_yaml(invalid_project_context))

    runner = CliRunner()
    result = runner.invoke(
        ado,
        [
            "--override-ado-app-dir",
            tmp_path,
            "create",
            "context",
            "-f",
            context_file,
            "--dry-run",
        ],
    )
    assert result.exit_code == 1, result.output
    assert result.output.strip().startswith(
        "INFO:   Initializing contexts - local is now your default context.\n"
        "ERROR:  The context provided was not valid:"
    )


def test_create_context(
    tmp_path: pathlib.Path, valid_ado_project_context: ProjectContext
) -> None:
    context_file = tmp_path / "temp_context.yaml"
    context_file.write_text(pydantic_model_as_yaml(valid_ado_project_context))

    runner = CliRunner()
    result = runner.invoke(
        ado,
        ["--override-ado-app-dir", tmp_path, "create", "context", "-f", context_file],
    )

    assert result.exit_code == 0
    expected_output = (
        "INFO:   Initializing contexts - local is now your default context.\n"
        "Success! \n"
        "INFO:   To set it as your default context, run:\n"
        f"        ado context {valid_ado_project_context.project}\n"
    )
    assert result.output == expected_output
