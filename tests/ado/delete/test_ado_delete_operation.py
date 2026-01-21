# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import pathlib
import sqlite3
from collections.abc import Callable

import pytest
from typer.testing import CliRunner

from orchestrator.cli.core.cli import app as ado
from orchestrator.core.samplestore.sql import SQLSampleStore
from orchestrator.metastore.project import ProjectContext
from orchestrator.metastore.sqlstore import SQLStore
from orchestrator.schema.experiment import Experiment
from orchestrator.schema.request import (
    MeasurementRequest,
    MeasurementRequestStateEnum,
    ReplayedMeasurement,
)

sqlite3_version = sqlite3.sqlite_version_info


# AP: the -> and ->> syntax in SQLite is only supported from version 3.38.0
# ref: https://sqlite.org/json1.html#jptr
@pytest.mark.skipif(
    sqlite3_version < (3, 38, 0), reason="SQLite version 3.38.0 or higher is required"
)
def test_delete_ml_multi_cloud_operation(
    tmp_path: pathlib.Path,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
    sql_store: SQLStore,
    ml_multi_cloud_benchmark_performance_experiment: Experiment,
    random_ml_multi_cloud_benchmark_performance_measurement_requests: Callable[
        [int, int, MeasurementRequestStateEnum | None, str | None],
        ReplayedMeasurement,
    ],
    simulate_ml_multi_cloud_random_walk_operation: Callable[
        [int, int, int, str | None],
        tuple[SQLSampleStore, list[MeasurementRequest], list[str]],
    ],
    random_sql_sample_store: Callable[[], SQLSampleStore],
    random_identifier: Callable[[], str],
) -> None:
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
