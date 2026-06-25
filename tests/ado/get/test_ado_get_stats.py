# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT
import datetime
import os
import pathlib
from collections.abc import Callable

import pandas as pd
import pytest
import rich.box
from testcontainers.mysql import MySqlContainer
from typer.testing import CliRunner

from orchestrator.cli.core.cli import app as ado
from orchestrator.core import CoreResourceKinds
from orchestrator.core.discoveryspace.space import DiscoverySpace
from orchestrator.core.samplestore.sql import SQLSampleStore
from orchestrator.metastore.project import ProjectContext
from orchestrator.schema.request import MeasurementRequest
from orchestrator.utilities.rich import dataframe_to_rich_table, render_to_string
from tests.conftest import requires_sqlite_3_38

# A past timestamp far enough from "now" that AGE is stable across a test run.
_CREATED_7_DAYS_AGO = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
    days=7
)
_EXPECTED_AGE = "7d0h"


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
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
    ml_multi_cloud_space: DiscoverySpace,
    simulate_ml_multi_cloud_random_walk_operation: Callable[
        [int, int, int, str | None, datetime.datetime | None],
        tuple[SQLSampleStore, list[MeasurementRequest], list[str]],
    ],
) -> None:
    """Stats values for a known operation match the expected counts in the rendered table."""
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    number_entities = 3
    # Use 1 request so MEASURED_ENTITIES == number_entities (entities are sampled per-request)
    number_requests = 1
    operation_id = "op-stats-test-abc123"

    simulate_ml_multi_cloud_random_walk_operation(
        number_entities=number_entities,
        number_requests=number_requests,
        measurements_per_result=1,
        operation_id=operation_id,
        created=_CREATED_7_DAYS_AGO,
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
        expected_output = pd.DataFrame(
            data={
                "IDENTIFIER": operation_id,
                "NAME": "randomwalk-all",
                "SPACE": ml_multi_cloud_space.uri,
                "STATUS": "added",
                "EXIT_STATE": "N/A",
                "AGE": _EXPECTED_AGE,
                "TOTAL_RESULTS": number_requests * number_entities,
                "SUCCESSFUL_RESULTS": number_requests * number_entities,
                "FAILED_RESULTS": 0,
                "MEASURED_ENTITIES": number_entities,
            },
            index=pd.Index([0]),
        )
        rendered_output = render_to_string(
            dataframe_to_rich_table(
                expected_output,
                show_index=True,
                show_edge=True,
                box=rich.box.SQUARE,
                do_not_truncate_columns=True,
            ),
            auto_width=True,
        )
        assert (
            rendered_output in result.output
        ), f"Expected output:\n{rendered_output}\nnot found in:\n{result.output}"


@requires_sqlite_3_38
def test_ado_get_operation_stats_single_resource(
    tmp_path: pathlib.Path,
    mysql_test_instance: MySqlContainer,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
    ml_multi_cloud_space: DiscoverySpace,
    simulate_ml_multi_cloud_random_walk_operation: Callable[
        [int, int, int, str | None, datetime.datetime | None],
        tuple[SQLSampleStore, list[MeasurementRequest], list[str]],
    ],
) -> None:
    """ado get operation <id> -o stats renders a table with the correct operation stats."""
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    number_entities = 2
    # Use 1 request so MEASURED_ENTITIES == number_entities (entities are sampled per-request)
    number_requests = 1
    operation_id = "op-stats-single-xyz789"

    simulate_ml_multi_cloud_random_walk_operation(
        number_entities=number_entities,
        number_requests=number_requests,
        measurements_per_result=1,
        operation_id=operation_id,
        created=_CREATED_7_DAYS_AGO,
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
        expected_output = pd.DataFrame(
            data={
                "IDENTIFIER": operation_id,
                "NAME": "randomwalk-all",
                "SPACE": ml_multi_cloud_space.uri,
                "STATUS": "added",
                "EXIT_STATE": "N/A",
                "AGE": _EXPECTED_AGE,
                "TOTAL_RESULTS": number_requests * number_entities,
                "SUCCESSFUL_RESULTS": number_requests * number_entities,
                "FAILED_RESULTS": 0,
                "MEASURED_ENTITIES": number_entities,
            },
            index=pd.Index([0]),
        )
        rendered_output = render_to_string(
            dataframe_to_rich_table(
                expected_output,
                show_index=True,
                show_edge=True,
                box=rich.box.SQUARE,
                do_not_truncate_columns=True,
            ),
            auto_width=True,
        )
        assert (
            rendered_output in result.output
        ), f"Expected output:\n{rendered_output}\nnot found in:\n{result.output}"


@requires_sqlite_3_38
def test_ado_get_spaces_stats_values(
    tmp_path: pathlib.Path,
    mysql_test_instance: MySqlContainer,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
    ml_multi_cloud_space: DiscoverySpace,
    backdate_resource: Callable[[str, CoreResourceKinds, datetime.datetime], None],
    simulate_ml_multi_cloud_random_walk_operation: Callable[
        [int, int, int, str | None],
        tuple[SQLSampleStore, list[MeasurementRequest], list[str]],
    ],
) -> None:
    """Stats values for a known space match the expected counts in the rendered table."""
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    number_entities = 3
    # Use 1 request so MEASURED_ENTITIES == number_entities (entities are sampled per-request)
    number_requests = 1
    operation_id = "op-space-stats-abc123"

    simulate_ml_multi_cloud_random_walk_operation(
        number_entities=number_entities,
        number_requests=number_requests,
        measurements_per_result=1,
        operation_id=operation_id,
    )
    space_id = ml_multi_cloud_space.uri
    backdate_resource(space_id, CoreResourceKinds.DISCOVERYSPACE, _CREATED_7_DAYS_AGO)

    result = runner.invoke(
        ado,
        [
            "--override-ado-app-dir",
            tmp_path,
            "get",
            "space",
            space_id,
            "-o",
            "stats",
            "--no-trunc",
        ],
    )

    assert result.exit_code == 0, result.output
    if os.environ.get("CI", "false") != "true":
        expected_output = pd.DataFrame(
            data={
                "IDENTIFIER": space_id,
                "NAME": "ml_multicloud_basic",
                "AGE": _EXPECTED_AGE,
                "EXPERIMENTS": 1,
                "OPERATIONS": 1,
                "EXPLORE_OPERATIONS": 1,
                "MEASURED_ENTITIES": number_entities,
            },
            index=pd.Index([0]),
        )
        rendered_output = render_to_string(
            dataframe_to_rich_table(
                expected_output,
                show_index=True,
                show_edge=True,
                box=rich.box.SQUARE,
                do_not_truncate_columns=True,
            ),
            auto_width=True,
        )
        assert (
            rendered_output in result.output
        ), f"Expected output:\n{rendered_output}\nnot found in:\n{result.output}"


@requires_sqlite_3_38
def test_ado_get_space_stats_single_resource(
    tmp_path: pathlib.Path,
    mysql_test_instance: MySqlContainer,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
    ml_multi_cloud_space: DiscoverySpace,
    backdate_resource: Callable[[str, CoreResourceKinds, datetime.datetime], None],
    simulate_ml_multi_cloud_random_walk_operation: Callable[
        [int, int, int, str | None],
        tuple[SQLSampleStore, list[MeasurementRequest], list[str]],
    ],
) -> None:
    """ado get space <id> -o stats renders a table with the correct space stats."""
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    number_entities = 2
    # Use 1 request so MEASURED_ENTITIES == number_entities (entities are sampled per-request)
    number_requests = 1
    operation_id = "op-space-stats-single-xyz789"

    simulate_ml_multi_cloud_random_walk_operation(
        number_entities=number_entities,
        number_requests=number_requests,
        measurements_per_result=1,
        operation_id=operation_id,
    )
    space_id = ml_multi_cloud_space.uri
    backdate_resource(space_id, CoreResourceKinds.DISCOVERYSPACE, _CREATED_7_DAYS_AGO)

    result = runner.invoke(
        ado,
        [
            "--override-ado-app-dir",
            tmp_path,
            "get",
            "space",
            space_id,
            "-o",
            "stats",
            "--no-trunc",
        ],
    )

    assert result.exit_code == 0, result.output
    if os.environ.get("CI", "false") != "true":
        expected_output = pd.DataFrame(
            data={
                "IDENTIFIER": space_id,
                "NAME": "ml_multicloud_basic",
                "AGE": _EXPECTED_AGE,
                "EXPERIMENTS": 1,
                "OPERATIONS": 1,
                "EXPLORE_OPERATIONS": 1,
                "MEASURED_ENTITIES": number_entities,
            },
            index=pd.Index([0]),
        )
        rendered_output = render_to_string(
            dataframe_to_rich_table(
                expected_output,
                show_index=True,
                show_edge=True,
                box=rich.box.SQUARE,
                do_not_truncate_columns=True,
            ),
            auto_width=True,
        )
        assert (
            rendered_output in result.output
        ), f"Expected output:\n{rendered_output}\nnot found in:\n{result.output}"


@requires_sqlite_3_38
@pytest.mark.parametrize("resource_kind", ["actuatorconfigurations"])
def test_ado_get_stats_unsupported_resource_type_exits_1(
    tmp_path: pathlib.Path,
    mysql_test_instance: MySqlContainer,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
    resource_kind: str,
) -> None:
    """ado get <unsupported-resource> -o stats exits 1 with an error message."""
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


@requires_sqlite_3_38
def test_ado_get_samplestores_stats_values(
    tmp_path: pathlib.Path,
    mysql_test_instance: MySqlContainer,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
    ml_multi_cloud_sample_store: SQLSampleStore,
    backdate_resource: Callable[[str, CoreResourceKinds, datetime.datetime], None],
    simulate_ml_multi_cloud_random_walk_operation: Callable[
        [int, int, int, str | None],
        tuple[SQLSampleStore, list[MeasurementRequest], list[str]],
    ],
) -> None:
    """Stats values for a known samplestore match expected counts in the rendered table."""
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    number_entities = 3
    number_requests = 1

    simulate_ml_multi_cloud_random_walk_operation(
        number_entities=number_entities,
        number_requests=number_requests,
        measurements_per_result=1,
    )

    after = ml_multi_cloud_sample_store.samplestore_statistics()
    store_id = ml_multi_cloud_sample_store.identifier
    backdate_resource(store_id, CoreResourceKinds.SAMPLESTORE, _CREATED_7_DAYS_AGO)

    result = runner.invoke(
        ado,
        [
            "--override-ado-app-dir",
            tmp_path,
            "get",
            "samplestores",
            "-o",
            "stats",
            "--no-trunc",
        ],
    )

    assert result.exit_code == 0, result.output
    if os.environ.get("CI", "false") != "true":
        expected_output = pd.DataFrame(
            data={
                "IDENTIFIER": store_id,
                "NAME": "ml_multi_cloud",
                "AGE": _EXPECTED_AGE,
                "ENTITIES": after.number_of_entities,
                "RESULTS": after.number_of_results,
                "EXPERIMENTS": after.number_of_experiments,
            },
            index=pd.Index([0]),
        )
        rendered_output = render_to_string(
            dataframe_to_rich_table(
                expected_output,
                show_index=True,
                show_edge=True,
                box=rich.box.SQUARE,
                do_not_truncate_columns=True,
            ),
            auto_width=True,
        )
        assert (
            rendered_output in result.output
        ), f"Expected output:\n{rendered_output}\nnot found in:\n{result.output}"
