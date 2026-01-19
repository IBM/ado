# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import pathlib
from collections.abc import Callable

import pytest
from testcontainers.mysql import MySqlContainer
from typer.testing import CliRunner

from orchestrator.cli.core.cli import app as ado
from orchestrator.core.discoveryspace.space import DiscoverySpace
from orchestrator.core.operation.config import DiscoveryOperationResourceConfiguration
from orchestrator.metastore.project import ProjectContext
from orchestrator.utilities.output import pydantic_model_as_yaml


@pytest.fixture(
    params=[
        "ml_multi_cloud_operation_configuration",
        "peptide_mineralization_basic_operation_configuration",
    ]
)
def operations_to_be_run(
    request: pytest.FixtureRequest,
) -> DiscoveryOperationResourceConfiguration:
    return request.getfixturevalue(request.param)


def test_create_operation_dry_run_success(
    tmp_path: pathlib.Path,
    mysql_test_instance: MySqlContainer,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
    operations_to_be_run: DiscoveryOperationResourceConfiguration,
) -> None:
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    operation_configuration_file = tmp_path / "operation.yaml"
    operation_configuration_file.write_text(
        pydantic_model_as_yaml(operations_to_be_run)
    )

    result = runner.invoke(
        ado,
        [
            "--override-ado-app-dir",
            tmp_path,
            "create",
            "operation",
            "-f",
            operation_configuration_file,
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    expected_output = "The configuration passed is valid!\n"
    assert result.output == expected_output


def test_create_operation_dry_run_failure(
    tmp_path: pathlib.Path,
    mysql_test_instance: MySqlContainer,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
    ml_multi_cloud_operation_configuration: DiscoveryOperationResourceConfiguration,
) -> None:
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    invalid_operation_configuration_file = tmp_path / "invalid_operation.yaml"
    invalid_operation_configuration_file.write_text(
        pydantic_model_as_yaml(ml_multi_cloud_operation_configuration)
        + "\nnonexistent-key: hello"
    )

    result = runner.invoke(
        ado,
        [
            "--override-ado-app-dir",
            tmp_path,
            "create",
            "operation",
            "-f",
            invalid_operation_configuration_file,
            "--dry-run",
        ],
    )

    assert result.exit_code == 1
    expected_output = "ERROR:  The operation configuration provided was not valid:\n"
    assert result.output.startswith(expected_output)


def test_create_operation_success(
    tmp_path: pathlib.Path,
    mysql_test_instance: MySqlContainer,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
    operations_to_be_run: DiscoveryOperationResourceConfiguration,
) -> None:
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    operation_file = tmp_path / "temp_op.yaml"
    operation_file.write_text(pydantic_model_as_yaml(operations_to_be_run))

    result = runner.invoke(
        ado,
        [
            "--override-ado-app-dir",
            tmp_path,
            "create",
            "operation",
            "-f",
            operation_file,
        ],
    )
    assert result.exit_code == 0


def test_create_operation_success_set_spaces(
    tmp_path: pathlib.Path,
    mysql_test_instance: MySqlContainer,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
    ml_multi_cloud_space: DiscoverySpace,
) -> None:
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    operation_configuration_file = pathlib.Path(
        "examples/ml-multi-cloud/randomwalk_ml_multicloud_operation.yaml"
    )

    result = runner.invoke(
        ado,
        [
            "--override-ado-app-dir",
            tmp_path,
            "create",
            "operation",
            "-f",
            operation_configuration_file,
            "--set",
            f'spaces=["{ml_multi_cloud_space.uri}"]',
        ],
    )
    assert result.exit_code == 0, result.output


def test_create_operation_success_with_discovery_space(
    tmp_path: pathlib.Path,
    mysql_test_instance: MySqlContainer,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
) -> None:
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    space_configuration_file = pathlib.Path(
        "examples/optimization_test_functions/space.yaml"
    )

    operation_configuration_file = pathlib.Path(
        "examples/optimization_test_functions/operation_bayesopt.yaml"
    )

    result = runner.invoke(
        ado,
        [
            "--override-ado-app-dir",
            tmp_path,
            "create",
            "operation",
            "-f",
            operation_configuration_file,
            "--set",
            "operation.parameters.tuneConfig.num_samples=2",
            "--with",
            f"space={space_configuration_file}",
        ],
    )
    assert result.exit_code == 0, result.output
