# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Tests for `ado show trace discoveryspace`."""

import pathlib
from collections.abc import Callable

from typer.testing import CliRunner

from ado.cli.core.cli import app as ado
from ado.cli.resources.trace_common import REQUEST_COLUMN
from ado.core.discoveryspace.space import DiscoverySpace
from ado.core.operation.config import DiscoveryOperationResourceConfiguration
from ado.core.samplestore.sql import SQLSampleStore
from ado.metastore.project import ProjectContext
from ado.schema.experiment import Experiment
from ado.schema.request import MeasurementRequest
from tests.conftest import requires_sqlite_3_38


@requires_sqlite_3_38
def test_show_trace_discoveryspace_single_operation(
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
    ml_multi_cloud_space: DiscoverySpace,
) -> None:
    """Single operation: exit 0 and no Operation ID column."""
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
            "show",
            "trace",
            "discoveryspace",
            ml_multi_cloud_space.uri,
        ],
    )

    assert result.exit_code == 0, result.output
    assert REQUEST_COLUMN.OPERATION_ID.value not in result.output


@requires_sqlite_3_38
def test_show_trace_discoveryspace_multi_operation(
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
) -> None:
    """Two operations: Operation ID column present in output."""
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
            "show",
            "trace",
            "discoveryspace",
            ml_multi_cloud_space.uri,
            "--output",
            "csv",
        ],
    )

    assert result.exit_code == 0, result.output
    assert REQUEST_COLUMN.OPERATION_ID.value in result.output


@requires_sqlite_3_38
def test_show_trace_discoveryspace_nonexistent(
    tmp_path: pathlib.Path,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
) -> None:
    """Non-existent space ID: non-zero exit code."""
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    result = runner.invoke(
        ado,
        [
            "show",
            "trace",
            "discoveryspace",
            "does-not-exist",
        ],
    )

    assert result.exit_code != 0


# Made with Bob
