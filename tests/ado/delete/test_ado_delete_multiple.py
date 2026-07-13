# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import pathlib
from collections.abc import Callable

from typer.testing import CliRunner

from ado.cli.core.cli import app as ado
from ado.core.resources import CoreResourceKinds
from ado.core.samplestore.sql import SQLSampleStore
from ado.metastore.project import ProjectContext
from ado.metastore.sqlstore import SQLStore
from ado.schema.experiment import Experiment
from ado.schema.request import (
    MeasurementRequest,
)
from tests.conftest import requires_sqlite_3_38


@requires_sqlite_3_38
def test_delete_multiple_operations_success(
    tmp_path: pathlib.Path,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
    sql_store: SQLStore,
    ml_multi_cloud_benchmark_performance_experiment: Experiment,
    simulate_ml_multi_cloud_random_walk_operation: Callable[
        [int, int, int, str | None],
        tuple[SQLSampleStore, list[MeasurementRequest], list[str]],
    ],
    random_identifier: Callable[[], str],
) -> None:
    """Test deleting multiple operations successfully."""
    assert ml_multi_cloud_benchmark_performance_experiment is not None
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    # Create three operations
    operation_ids = [random_identifier() for _ in range(3)]
    sample_stores = []

    for operation_id in operation_ids:
        sample_store, _, _ = simulate_ml_multi_cloud_random_walk_operation(
            number_entities=2,
            number_requests=2,
            measurements_per_result=1,
            operation_id=operation_id,
        )
        sample_stores.append(sample_store)

    # Verify operations exist
    for i, operation_id in enumerate(operation_ids):
        assert sql_store.containsResourceWithIdentifier(
            identifier=operation_id,
            kind=CoreResourceKinds.OPERATION,
        )
        assert (
            sample_stores[i].measurement_requests_count_for_operation(
                operation_id=operation_id
            )
            == 2
        )

    # Delete all three operations
    result = runner.invoke(
        ado,
        [
            "delete",
            "operation",
            *operation_ids,
            "--force",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Successfully deleted" in result.output
    assert operation_ids[0] in result.output
    assert operation_ids[1] in result.output
    assert operation_ids[2] in result.output

    # Verify all operations are deleted
    for i, operation_id in enumerate(operation_ids):
        assert not sql_store.containsResourceWithIdentifier(
            identifier=operation_id,
            kind=CoreResourceKinds.OPERATION,
        )
        assert (
            sample_stores[i].measurement_requests_count_for_operation(
                operation_id=operation_id
            )
            == 0
        )


@requires_sqlite_3_38
def test_delete_multiple_operations_partial_failure(
    tmp_path: pathlib.Path,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
    sql_store: SQLStore,
    ml_multi_cloud_benchmark_performance_experiment: Experiment,
    simulate_ml_multi_cloud_random_walk_operation: Callable[
        [int, int, int, str | None],
        tuple[SQLSampleStore, list[MeasurementRequest], list[str]],
    ],
    random_identifier: Callable[[], str],
) -> None:
    """Test deleting multiple operations where some don't exist."""
    assert ml_multi_cloud_benchmark_performance_experiment is not None
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    # Create two operations
    valid_op_1 = random_identifier()
    valid_op_2 = random_identifier()
    invalid_op = "non-existent-operation-id"

    sample_store_1, _, _ = simulate_ml_multi_cloud_random_walk_operation(
        number_entities=2,
        number_requests=2,
        measurements_per_result=1,
        operation_id=valid_op_1,
    )
    sample_store_2, _, _ = simulate_ml_multi_cloud_random_walk_operation(
        number_entities=2,
        number_requests=2,
        measurements_per_result=1,
        operation_id=valid_op_2,
    )

    # Try to delete two valid and one invalid operation
    result = runner.invoke(
        ado,
        [
            "delete",
            "operation",
            valid_op_1,
            invalid_op,
            valid_op_2,
            "--force",
        ],
    )

    # Should exit with error code due to partial failure
    assert result.exit_code == 1, result.output

    # Check output contains success and failure messages
    assert "Successfully deleted" in result.output
    assert "Failed to delete" in result.output
    assert valid_op_1 in result.output
    assert valid_op_2 in result.output
    assert invalid_op in result.output

    # Verify valid operations are deleted
    assert not sql_store.containsResourceWithIdentifier(
        identifier=valid_op_1,
        kind=CoreResourceKinds.OPERATION,
    )
    assert (
        sample_store_1.measurement_requests_count_for_operation(operation_id=valid_op_1)
        == 0
    )
    assert not sql_store.containsResourceWithIdentifier(
        identifier=valid_op_2,
        kind=CoreResourceKinds.OPERATION,
    )
    assert (
        sample_store_2.measurement_requests_count_for_operation(operation_id=valid_op_2)
        == 0
    )


@requires_sqlite_3_38
def test_delete_single_operation_backward_compatible(
    tmp_path: pathlib.Path,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
    sql_store: SQLStore,
    ml_multi_cloud_benchmark_performance_experiment: Experiment,
    simulate_ml_multi_cloud_random_walk_operation: Callable[
        [int, int, int, str | None],
        tuple[SQLSampleStore, list[MeasurementRequest], list[str]],
    ],
    random_identifier: Callable[[], str],
) -> None:
    """Test that single operation deletion still works (backward compatibility)."""
    assert ml_multi_cloud_benchmark_performance_experiment is not None
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    operation_id = random_identifier()
    sample_store, _, _ = simulate_ml_multi_cloud_random_walk_operation(
        number_entities=2,
        number_requests=2,
        measurements_per_result=1,
        operation_id=operation_id,
    )

    # Delete single operation (original behavior)
    result = runner.invoke(
        ado,
        [
            "delete",
            "operation",
            operation_id,
            "--force",
        ],
    )
    assert result.exit_code == 0, result.output

    # Verify operation is deleted
    assert not sql_store.containsResourceWithIdentifier(
        identifier=operation_id,
        kind=CoreResourceKinds.OPERATION,
    )
    assert (
        sample_store.measurement_requests_count_for_operation(operation_id=operation_id)
        == 0
    )


@requires_sqlite_3_38
def test_delete_multiple_operations_all_fail(
    tmp_path: pathlib.Path,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
    sql_store: SQLStore,
) -> None:
    """Test deleting multiple non-existent operations."""
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    # Try to delete three non-existent operations
    result = runner.invoke(
        ado,
        [
            "delete",
            "operation",
            "non-existent-1",
            "non-existent-2",
            "non-existent-3",
            "--force",
        ],
    )

    # Should exit with error code
    assert result.exit_code == 1, result.output

    # Check output contains failure messages
    assert "Failed to delete" in result.output
    assert "non-existent-1" in result.output
    assert "non-existent-2" in result.output
    assert "non-existent-3" in result.output
    assert "Summary" in result.output
    assert "Failed to delete 3 resource(s)" in result.output


# Made with Bob
