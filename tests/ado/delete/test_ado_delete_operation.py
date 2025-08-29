# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import pathlib
import sqlite3

import pytest
from typer.testing import CliRunner

from orchestrator.cli.core.cli import app as ado

sqlite3_version = sqlite3.sqlite_version_info


# AP: the -> and ->> syntax in SQLite is only supported from version 3.38.0
# ref: https://sqlite.org/json1.html#jptr
@pytest.mark.skipif(
    sqlite3_version < (3, 38, 0), reason="SQLite version 3.38.0 or higher is required"
)
def test_delete_ml_multi_cloud_operation(
    tmp_path: pathlib.Path,
    valid_ado_project_context,
    create_active_ado_context,
    sql_store,
    ml_multi_cloud_benchmark_performance_experiment,
    random_ml_multi_cloud_benchmark_performance_measurement_requests,
    simulate_ml_multi_cloud_random_walk_operation,
    random_sql_sample_store,
    random_identifier,
):
    assert ml_multi_cloud_benchmark_performance_experiment is not None
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    number_entities = 3
    number_requests = 3
    measurements_per_result = 2
    operation_id = random_identifier()

    sample_store, _, _ = simulate_ml_multi_cloud_random_walk_operation(
        number_entities=number_entities,
        number_requests=number_requests,
        measurements_per_result=measurements_per_result,
        operation_id=operation_id,
    )

    # Check expected status for the setup
    assert (
        sample_store.measurement_requests_count_for_operation(operation_id=operation_id)
        == number_requests
    )
    assert (
        sample_store.measurement_results_count_for_operation(operation_id=operation_id)
        == number_requests * number_entities
    )

    # Delete the operation
    result = runner.invoke(
        ado,
        [
            "--override-ado-app-dir",
            tmp_path,
            "delete",
            "operation",
            operation_id,
            "--force",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (
        sample_store.measurement_requests_count_for_operation(operation_id=operation_id)
        == 0
    )
    assert (
        sample_store.measurement_results_count_for_operation(operation_id=operation_id)
        == 0
    )
