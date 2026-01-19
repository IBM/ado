# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import pathlib
from collections.abc import Callable

from testcontainers.mysql import MySqlContainer
from typer.testing import CliRunner

from orchestrator.cli.core.cli import app as ado
from orchestrator.core.samplestore.config import SampleStoreConfiguration
from orchestrator.metastore.project import ProjectContext
from orchestrator.utilities.output import pydantic_model_as_yaml


def test_create_sample_store_dry_run_success(tmp_path: pathlib.Path) -> None:
    sample_store_configuration_file = "tests/resources/ml_multicloud_sample_store.yaml"
    runner = CliRunner()
    result = runner.invoke(
        ado,
        [
            "--override-ado-app-dir",
            tmp_path,
            "create",
            "samplestore",
            "-f",
            sample_store_configuration_file,
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    expected_output = (
        "INFO:   Initializing contexts - local is now your default context.\n"
        "The configuration passed is valid!\n"
    )
    assert result.output == expected_output


def test_create_sample_store_dry_run_failure(tmp_path: pathlib.Path) -> None:
    sample_store_configuration_file = pathlib.Path(
        "tests/resources/ml_multicloud_sample_store.yaml"
    )
    invalid_sample_store_configuration_file = tmp_path / "invalid_sample_store.yaml"
    invalid_sample_store_configuration_file.write_text(
        sample_store_configuration_file.read_text() + "\nnonexistent-key: hello"
    )

    runner = CliRunner()
    result = runner.invoke(
        ado,
        [
            "--override-ado-app-dir",
            tmp_path,
            "create",
            "samplestore",
            "-f",
            invalid_sample_store_configuration_file,
            "--dry-run",
        ],
    )

    assert result.exit_code == 1
    expected_output = (
        "INFO:   Initializing contexts - local is now your default context.\n"
        "ERROR:  The sample store configuration provided was not valid:\n"
    )
    assert result.output.startswith(expected_output)


def test_create_sample_store_success(
    tmp_path: pathlib.Path,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
) -> None:
    sample_store_configuration_file = pathlib.Path(
        "tests/resources/ml_multicloud_sample_store.yaml"
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
            "samplestore",
            "-f",
            sample_store_configuration_file,
        ],
    )

    assert result.exit_code == 0, result.output
    expected_output = "Success! Created sample store with identifier"
    assert result.output.startswith(expected_output)


def test_create_sample_store_success_new_sample_store(
    tmp_path: pathlib.Path,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
) -> None:
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
            "samplestore",
            "--new-sample-store",
        ],
    )

    assert result.exit_code == 0, result.output
    expected_output = (
        "INFO:   A new SQLSampleStore was requested.\n"
        "Success! Created sample store with identifier"
    )
    assert result.output.startswith(expected_output)


def test_create_sample_store_failure_because_hardcoded_storage_location(
    tmp_path: pathlib.Path,
    mysql_test_instance: MySqlContainer,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
    ado_sql_sample_store_with_storagelocation: SampleStoreConfiguration,
) -> None:

    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    sample_store_file = tmp_path / "temp_es.yaml"
    sample_store_file.write_text(
        pydantic_model_as_yaml(ado_sql_sample_store_with_storagelocation)
    )
    result = runner.invoke(
        ado,
        [
            "--override-ado-app-dir",
            tmp_path,
            "create",
            "samplestore",
            "-f",
            sample_store_file,
        ],
    )
    assert result.exit_code == 1
    expected_output = (
        "ERROR:  The storageLocation section must be empty when creating sample stores. "
        "It will be set automatically based on the context information.\n"
        "HINT:   Remove the storageLocation field and retry.\n"
    )
    assert result.output == expected_output
