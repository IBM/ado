# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Tests for `ado show trace samplestore`."""

import pathlib
from collections.abc import Callable

from typer.testing import CliRunner

from orchestrator.cli.core.cli import app as ado
from orchestrator.cli.resources.trace_common import REQUEST_COLUMN
from orchestrator.core.discoveryspace.space import DiscoverySpace
from orchestrator.core.operation.config import DiscoveryOperationResourceConfiguration
from orchestrator.core.samplestore.sql import SQLSampleStore
from orchestrator.metastore.project import ProjectContext
from orchestrator.schema.experiment import Experiment
from orchestrator.schema.request import MeasurementRequest
from tests.conftest import requires_sqlite_3_38


@requires_sqlite_3_38
def test_show_trace_samplestore_single_operation(
    tmp_path: pathlib.Path,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
    ml_multi_cloud_benchmark_performance_experiment: Experiment,
    simulate_ml_multi_cloud_random_walk_operation: Callable[
        [int, int, int, str | None],
        tuple[SQLSampleStore, list[MeasurementRequest], list[str]],
    ],
    ml_multi_cloud_sample_store: SQLSampleStore,
) -> None:
    """Single operation: exit 0, Operation ID and Space ID columns always present."""
    assert ml_multi_cloud_benchmark_performance_experiment is not None
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    simulate_ml_multi_cloud_random_walk_operation(
        number_entities=2,
        number_requests=2,
        measurements_per_result=1,
    )

    result = runner.invoke(
        ado,
        [
            "--override-ado-app-dir",
            str(tmp_path),
            "show",
            "trace",
            "samplestore",
            ml_multi_cloud_sample_store.identifier,
            "--no-trunc",
        ],
    )

    assert result.exit_code == 0, result.output
    assert REQUEST_COLUMN.OPERATION_ID.value in result.output
    assert REQUEST_COLUMN.SPACE_ID.value in result.output


@requires_sqlite_3_38
def test_show_trace_samplestore_multi_operation(
    tmp_path: pathlib.Path,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
    ml_multi_cloud_benchmark_performance_experiment: Experiment,
    ml_multi_cloud_operation_configuration: DiscoveryOperationResourceConfiguration,
    simulate_ml_multi_cloud_random_walk_operation: Callable[
        [int, int, int, str | None],
        tuple[SQLSampleStore, list[MeasurementRequest], list[str]],
    ],
    random_identifier: Callable[[], str],
    ml_multi_cloud_space: DiscoverySpace,
    ml_multi_cloud_sample_store: SQLSampleStore,
) -> None:
    """Two operations: Operation ID and Space ID columns present in output."""
    assert ml_multi_cloud_benchmark_performance_experiment is not None
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    simulate_ml_multi_cloud_random_walk_operation(
        number_entities=2,
        number_requests=2,
        measurements_per_result=1,
    )

    # Second operation: simulate handles addResourceWithRelationships internally
    simulate_ml_multi_cloud_random_walk_operation(
        number_entities=2,
        number_requests=2,
        measurements_per_result=1,
        operation_id=random_identifier(),
    )

    result = runner.invoke(
        ado,
        [
            "--override-ado-app-dir",
            str(tmp_path),
            "show",
            "trace",
            "samplestore",
            ml_multi_cloud_sample_store.identifier,
            "--output",
            "csv",
        ],
    )

    assert result.exit_code == 0, result.output
    assert REQUEST_COLUMN.OPERATION_ID.value in result.output
    assert REQUEST_COLUMN.SPACE_ID.value in result.output


@requires_sqlite_3_38
def test_show_trace_samplestore_nonexistent(
    tmp_path: pathlib.Path,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
) -> None:
    """Non-existent samplestore ID: non-zero exit code."""
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    result = runner.invoke(
        ado,
        [
            "--override-ado-app-dir",
            str(tmp_path),
            "show",
            "trace",
            "samplestore",
            "does-not-exist",
        ],
    )

    assert result.exit_code != 0


# Made with Bob
