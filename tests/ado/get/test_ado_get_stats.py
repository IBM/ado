# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT
import os
import pathlib
from collections.abc import Callable

import pytest
from testcontainers.mysql import MySqlContainer
from typer.testing import CliRunner

from orchestrator.cli.core.cli import app as ado
from orchestrator.core.samplestore.sql import SQLSampleStore
from orchestrator.metastore.project import ProjectContext
from orchestrator.metastore.sqlstore import SQLStore
from orchestrator.schema.request import MeasurementRequest
from tests.conftest import requires_sqlite_3_38


@requires_sqlite_3_38
def test_ado_get_operations_stats_columns_present(
    tmp_path: pathlib.Path,
    mysql_test_instance: MySqlContainer,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
    simulate_ml_multi_cloud_random_walk_operation: Callable[
        [int, int, int, str | None],
        tuple[SQLSampleStore, list[MeasurementRequest], list[str]],
    ],
) -> None:
    """ado get operations -o stats exits 0 and shows result stats column headers."""
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    # Create one operation with known measurements.
    simulate_ml_multi_cloud_random_walk_operation(
        number_entities=2,
        number_requests=3,
        measurements_per_result=1,
    )

    result = runner.invoke(
        ado,
        [
            "--override-ado-app-dir",
            tmp_path,
            "get",
            "operations",
            "-o",
            "stats",
            "--no-trunc",
        ],
    )

    assert result.exit_code == 0, result.output
    if os.environ.get("CI", "false") != "true":
        for col in [
            "TOTAL_RESULTS",
            "SUCCESSFUL_RESULTS",
            "FAILED_RESULTS",
            "MEASURED_ENTITIES",
        ]:
            assert col in result.output, f"Column {col!r} missing from output"


@requires_sqlite_3_38
def test_ado_get_operations_stats_values(
    tmp_path: pathlib.Path,
    mysql_test_instance: MySqlContainer,
    sql_store: SQLStore,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
    simulate_ml_multi_cloud_random_walk_operation: Callable[
        [int, int, int, str | None],
        tuple[SQLSampleStore, list[MeasurementRequest], list[str]],
    ],
) -> None:
    """Stats values for a known operation match expected counts."""
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    number_entities = 3
    number_requests = 4
    operation_id = "op-stats-test-abc123"

    simulate_ml_multi_cloud_random_walk_operation(
        number_entities=number_entities,
        number_requests=number_requests,
        measurements_per_result=1,
        operation_id=operation_id,
    )

    result = runner.invoke(
        ado,
        [
            "--override-ado-app-dir",
            tmp_path,
            "get",
            "operation",
            operation_id,
            "-o",
            "stats",
        ],
    )

    assert result.exit_code == 0, result.output
    if os.environ.get("CI", "false") != "true":
        # TOTAL_RESULTS == number_requests * number_entities
        assert str(number_requests * number_entities) in result.output


@requires_sqlite_3_38
def test_ado_get_operation_stats_single_resource(
    tmp_path: pathlib.Path,
    mysql_test_instance: MySqlContainer,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
    simulate_ml_multi_cloud_random_walk_operation: Callable[
        [int, int, int, str | None],
        tuple[SQLSampleStore, list[MeasurementRequest], list[str]],
    ],
) -> None:
    """ado get operation <id> -o stats exits 0 and shows result stats columns."""
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    operation_id = "op-stats-single-xyz789"
    simulate_ml_multi_cloud_random_walk_operation(
        number_entities=2,
        number_requests=2,
        measurements_per_result=1,
        operation_id=operation_id,
    )

    result = runner.invoke(
        ado,
        [
            "--override-ado-app-dir",
            tmp_path,
            "get",
            "operation",
            operation_id,
            "-o",
            "stats",
            "--no-trunc",
        ],
    )

    assert result.exit_code == 0, result.output
    if os.environ.get("CI", "false") != "true":
        assert operation_id in result.output
        assert "TOTAL_RESULTS" in result.output


@requires_sqlite_3_38
@pytest.mark.parametrize("resource_kind", ["spaces", "samplestores"])
def test_ado_get_stats_unsupported_resource_type_exits_1(
    tmp_path: pathlib.Path,
    mysql_test_instance: MySqlContainer,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
    resource_kind: str,
) -> None:
    """ado get <non-operation> -o stats exits 1 with an error message."""
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    result = runner.invoke(
        ado,
        [
            "--override-ado-app-dir",
            tmp_path,
            "get",
            resource_kind,
            "-o",
            "stats",
        ],
    )

    assert result.exit_code == 1
    if os.environ.get("CI", "false") != "true":
        assert "stats" in result.output.lower()
        assert "operation" in result.output.lower()
