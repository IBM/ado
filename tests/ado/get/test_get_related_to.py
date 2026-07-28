# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Tests for `ado get <resource> --related-to <kind>=<id>`."""

import pathlib
from collections.abc import Callable

from typer.testing import CliRunner

from ado.cli.core.cli import app as ado
from ado.core.discoveryspace.space import DiscoverySpace
from ado.core.operation.config import DiscoveryOperationResourceConfiguration
from ado.core.samplestore.sql import SQLSampleStore
from ado.metastore.project import ProjectContext
from ado.schema.experiment import Experiment
from ado.schema.request import MeasurementRequest
from tests.conftest import requires_sqlite_3_38


@requires_sqlite_3_38
def test_get_operation_related_to_samplestore(
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
    """ado get operation --related-to samplestore=<id> returns the operation."""
    assert ml_multi_cloud_benchmark_performance_experiment is not None
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    _sample_store, _requests, _request_ids = (
        simulate_ml_multi_cloud_random_walk_operation(
            number_entities=2,
            number_requests=2,
            measurements_per_result=1,
        )
    )

    result = runner.invoke(
        ado,
        [
            "get",
            "operation",
            "--related-to",
            f"samplestore={ml_multi_cloud_sample_store.identifier}",
            "-o",
            "name",
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(result.output.strip()) > 0


@requires_sqlite_3_38
def test_get_discoveryspace_related_to_samplestore(
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
    ml_multi_cloud_sample_store: SQLSampleStore,
) -> None:
    """ado get discoveryspace --related-to samplestore=<id> returns the space."""
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
            "get",
            "discoveryspace",
            "--related-to",
            f"samplestore={ml_multi_cloud_sample_store.identifier}",
            "-o",
            "name",
        ],
    )

    assert result.exit_code == 0, result.output
    assert ml_multi_cloud_space.uri in result.output


@requires_sqlite_3_38
def test_get_operation_related_to_samplestore_with_matching_filter(
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
    ml_multi_cloud_operation_configuration: DiscoveryOperationResourceConfiguration,
    ml_multi_cloud_sample_store: SQLSampleStore,
) -> None:
    """--related-to combined with --filter returns intersection: match → non-empty, no-match → empty."""
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

    op_name = ml_multi_cloud_operation_configuration.metadata.name

    # Matching filter → operation returned
    result_match = runner.invoke(
        ado,
        [
            "get",
            "operation",
            "--related-to",
            f"samplestore={ml_multi_cloud_sample_store.identifier}",
            "--filter",
            f"config.metadata.name={op_name}",
            "-o",
            "name",
        ],
    )
    assert result_match.exit_code == 0, result_match.output
    assert len(result_match.output.strip()) > 0

    # Non-matching filter → empty result
    result_no_match = runner.invoke(
        ado,
        [
            "get",
            "operation",
            "--related-to",
            f"samplestore={ml_multi_cloud_sample_store.identifier}",
            "--filter",
            "config.metadata.name=this-name-does-not-exist",
            "-o",
            "name",
        ],
    )
    assert result_no_match.exit_code == 0, result_no_match.output


@requires_sqlite_3_38
def test_get_operation_related_to_nonexistent_samplestore(
    tmp_path: pathlib.Path,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
) -> None:
    """Non-existent anchor resource: non-zero exit code."""
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    result = runner.invoke(
        ado,
        [
            "get",
            "operation",
            "--related-to",
            "samplestore=does-not-exist",
        ],
    )

    assert result.exit_code != 0


def test_get_operation_related_to_bad_format(
    tmp_path: pathlib.Path,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
) -> None:
    """Malformed --related-to value (no '='): exits with code 1."""
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    result = runner.invoke(ado, ["get", "operation", "--related-to", "badformat"])

    assert result.exit_code == 1


def test_get_operation_related_to_unknown_kind(
    tmp_path: pathlib.Path,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
) -> None:
    """Unknown resource kind in --related-to: exits with code 1."""
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    result = runner.invoke(
        ado, ["get", "operation", "--related-to", "unknownkind=some-id"]
    )

    assert result.exit_code == 1


def test_get_operation_related_to_with_explicit_resource_id(
    tmp_path: pathlib.Path,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
) -> None:
    """--related-to with explicit resource_id: exits with code 1."""
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    result = runner.invoke(
        ado,
        [
            "get",
            "operation",
            "some-op-id",
            "--related-to",
            "samplestore=some-store-id",
        ],
    )

    assert result.exit_code == 1


def test_get_operation_related_to_same_kind(
    tmp_path: pathlib.Path,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
) -> None:
    """--related-to anchor kind same as requested kind: exits with code 1."""
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    result = runner.invoke(
        ado,
        ["get", "operation", "--related-to", "operation=some-op-id"],
    )

    assert result.exit_code == 1


@requires_sqlite_3_38
def test_get_operation_related_to_table_output(
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
    """ado get operation --related-to ... -o table exits 0."""
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
            "get",
            "operation",
            "--related-to",
            f"samplestore={ml_multi_cloud_sample_store.identifier}",
            "-o",
            "table",
        ],
    )

    assert result.exit_code == 0, result.output


@requires_sqlite_3_38
def test_get_operation_related_to_yaml_output(
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
    """ado get operation --related-to ... -o yaml exits 0 and outputs YAML."""
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
            "get",
            "operation",
            "--related-to",
            f"samplestore={ml_multi_cloud_sample_store.identifier}",
            "-o",
            "yaml",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "identifier" in result.output


# Made with Bob
