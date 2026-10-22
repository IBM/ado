# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT
import datetime as dt
import os
import pathlib
import typing
from collections.abc import Callable

import pandas as pd
import pytest
import yaml

try:
    from yaml import CSafeLoader as SafeLoader
except ImportError:  # pragma: nocover
    from yaml import SafeLoader  # type: ignore[assignment]
from testcontainers.community.mysql import MySqlContainer
from typer.testing import CliRunner

from ado.cli.core.cli import app as ado
from ado.core import CoreResourceKinds
from ado.core.discoveryspace.space import DiscoverySpace
from ado.core.samplestore.sql import SQLSampleStore
from ado.metastore.project import ProjectContext
from ado.schema.point import SpacePoint
from ado.schema.request import MeasurementRequest
from ado.utilities.output import pydantic_model_as_yaml
from tests.conftest import requires_sqlite_3_38
from tests.utilities.cli_rendering import render_dataframe_to_cli_output

if typing.TYPE_CHECKING:
    from ado.core import DataContainerResource

# A past timestamp far enough from "now" that AGE is stable across a test run.
_CREATED_7_DAYS_AGO = dt.datetime.now(dt.UTC) - dt.timedelta(days=7)
_EXPECTED_AGE = "7d0h"


def test_timedelta_to_string_boundary_rounding() -> None:
    """timedelta_to_string must not produce overflow values when rounding carries over."""
    from ado.cli.utils.resources.formatters import timedelta_to_string

    # 59.5s rounds to 60s → should be 1m0s, not 60s
    assert timedelta_to_string(59.5) == "1m0s"
    # Unambiguous sub-minute case
    assert timedelta_to_string(30.0) == "30s"

    # 4m 59.5s rounds seconds to 60 → should be 5m0s, not 4m60s
    assert timedelta_to_string(4 * 60 + 59.5) == "5m0s"
    # Unambiguous sub-hour case
    assert timedelta_to_string(4 * 60 + 30.0) == "4m30s"

    # 2h 59m 30s rounds minutes to 60 → should be 3h0m, not 2h60m
    assert timedelta_to_string(2 * 3600 + 59 * 60 + 30.0) == "3h0m"
    # Unambiguous sub-day case
    assert timedelta_to_string(2 * 3600 + 30 * 60) == "2h30m"

    # 189 days + 23h 59m 30s rounds hours to 24 → should be 190d0h, not 189d24h
    assert (
        timedelta_to_string(float(189 * 86400 + 23 * 3600 + 59 * 60 + 30)) == "190d0h"
    )
    # Exactly 190 days stays 190d0h
    assert timedelta_to_string(float(190 * 86400)) == "190d0h"
    # Unambiguous multi-day case
    assert timedelta_to_string(float(189 * 86400 + 3600)) == "189d1h"


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
    """ado get operations --stats exits 0 and shows result stats column headers."""
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
            "get",
            "operations",
            "--stats",
            "--no-trunc",
        ],
    )

    assert result.exit_code == 0, result.output
    if os.environ.get("CI", "false") != "true":
        for col in [
            "TOTAL_MEASUREMENTS",
            "SUCCESSFUL_MEASUREMENTS",
            "FAILED_MEASUREMENTS",
            "SAMPLED_ENTITIES",
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
        [int, int, int, str | None, dt.datetime | None],
        tuple[SQLSampleStore, list[MeasurementRequest], list[str]],
    ],
) -> None:
    """Stats values for a known operation match the expected counts in the rendered table."""
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    number_entities = 3
    # Use 1 request so SAMPLED_ENTITIES == number_entities (entities are sampled per-request)
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
            "get",
            "operation",
            operation_id,
            "--stats",
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
                "TOTAL_MEASUREMENTS": number_requests * number_entities,
                "SUCCESSFUL_MEASUREMENTS": number_requests * number_entities,
                "FAILED_MEASUREMENTS": 0,
                "SAMPLED_ENTITIES": number_entities,
            },
            index=pd.Index([0]),
        )
        rendered_output = render_dataframe_to_cli_output(
            expected_output, do_not_truncate_columns=True
        )
        assert rendered_output in result.output, (
            f"Expected output:\n{rendered_output}\nnot found in:\n{result.output}"
        )


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
        [int, int, int, str | None, dt.datetime | None],
        tuple[SQLSampleStore, list[MeasurementRequest], list[str]],
    ],
) -> None:
    """ado get operation <id> --stats renders a table with the correct operation stats."""
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    number_entities = 2
    # Use 1 request so SAMPLED_ENTITIES == number_entities (entities are sampled per-request)
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
            "get",
            "operation",
            operation_id,
            "--stats",
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
                "TOTAL_MEASUREMENTS": number_requests * number_entities,
                "SUCCESSFUL_MEASUREMENTS": number_requests * number_entities,
                "FAILED_MEASUREMENTS": 0,
                "SAMPLED_ENTITIES": number_entities,
            },
            index=pd.Index([0]),
        )
        rendered_output = render_dataframe_to_cli_output(
            expected_output, do_not_truncate_columns=True
        )
        assert rendered_output in result.output, (
            f"Expected output:\n{rendered_output}\nnot found in:\n{result.output}"
        )


@requires_sqlite_3_38
def test_ado_get_spaces_stats_values(
    tmp_path: pathlib.Path,
    mysql_test_instance: MySqlContainer,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
    ml_multi_cloud_space: DiscoverySpace,
    backdate_resource: Callable[[str, CoreResourceKinds, dt.datetime], None],
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
    # Use 1 request so SAMPLED_ENTITIES == number_entities (entities are sampled per-request)
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
            "get",
            "space",
            space_id,
            "--stats",
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
                "EXPLORE_OPS": 1,
                "SAMPLED_ENTITIES": number_entities,
            },
            index=pd.Index([0]),
        )
        rendered_output = render_dataframe_to_cli_output(
            expected_output, do_not_truncate_columns=True
        )
        assert rendered_output in result.output, (
            f"Expected output:\n{rendered_output}\nnot found in:\n{result.output}"
        )


@requires_sqlite_3_38
def test_ado_get_space_stats_single_resource(
    tmp_path: pathlib.Path,
    mysql_test_instance: MySqlContainer,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
    ml_multi_cloud_space: DiscoverySpace,
    backdate_resource: Callable[[str, CoreResourceKinds, dt.datetime], None],
    simulate_ml_multi_cloud_random_walk_operation: Callable[
        [int, int, int, str | None],
        tuple[SQLSampleStore, list[MeasurementRequest], list[str]],
    ],
) -> None:
    """ado get space <id> --stats renders a table with the correct space stats."""
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    number_entities = 2
    # Use 1 request so SAMPLED_ENTITIES == number_entities (entities are sampled per-request)
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
            "get",
            "space",
            space_id,
            "--stats",
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
                "EXPLORE_OPS": 1,
                "SAMPLED_ENTITIES": number_entities,
            },
            index=pd.Index([0]),
        )
        rendered_output = render_dataframe_to_cli_output(
            expected_output, do_not_truncate_columns=True
        )
        assert rendered_output in result.output, (
            f"Expected output:\n{rendered_output}\nnot found in:\n{result.output}"
        )


@requires_sqlite_3_38
@pytest.mark.parametrize("resource_kind", ["actuatorconfigurations"])
@pytest.mark.parametrize("output_args", [[], ["-o", "yaml"]])
def test_ado_get_stats_unsupported_resource_type_exits_1(
    tmp_path: pathlib.Path,
    mysql_test_instance: MySqlContainer,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
    resource_kind: str,
    output_args: list[str],
) -> None:
    """ado get <unsupported-resource> --stats exits 1 with an error message."""
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    result = runner.invoke(
        ado,
        [
            "get",
            resource_kind,
            "--stats",
            *output_args,
        ],
    )

    assert result.exit_code == 1
    if os.environ.get("CI", "false") != "true":
        assert "--stats" in result.output.lower()
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
    backdate_resource: Callable[[str, CoreResourceKinds, dt.datetime], None],
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
            "get",
            "samplestores",
            "--stats",
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
        rendered_output = render_dataframe_to_cli_output(
            expected_output, do_not_truncate_columns=True
        )
        assert rendered_output in result.output, (
            f"Expected output:\n{rendered_output}\nnot found in:\n{result.output}"
        )


@requires_sqlite_3_38
def test_ado_get_datacontainers_stats_columns_present(
    tmp_path: pathlib.Path,
    mysql_test_instance: MySqlContainer,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
    data_container_resource: "DataContainerResource",
    create_resources: Callable,
) -> None:
    """ado get datacontainers --stats exits 0 and shows all four stats column headers."""
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    create_resources([data_container_resource])

    result = runner.invoke(
        ado,
        [
            "get",
            "datacontainers",
            "--stats",
            "--no-trunc",
        ],
    )

    assert result.exit_code == 0, result.output
    if os.environ.get("CI", "false") != "true":
        for col in ("TABLES", "LOCATIONS", "KEY_VALUES", "DATA_BYTES"):
            assert col in result.output, f"Column {col!r} missing from output"


@requires_sqlite_3_38
def test_ado_get_datacontainers_stats_values(
    tmp_path: pathlib.Path,
    mysql_test_instance: MySqlContainer,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
    data_container_resource: "DataContainerResource",
    create_resources: Callable,
    backdate_resource: Callable[[str, CoreResourceKinds, "dt.datetime"], None],
) -> None:
    """Stats values for a known datacontainer match expected counts in the rendered table."""
    from ado.metastore.sqlstore import SQLStore

    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    create_resources([data_container_resource])

    sql_store = SQLStore(project_context=valid_ado_project_context)
    stats_by_id = sql_store.get_datacontainer_stats(
        {data_container_resource.identifier}
    )
    stats = stats_by_id[data_container_resource.identifier]

    container_id = data_container_resource.identifier
    backdate_resource(
        container_id, CoreResourceKinds.DATACONTAINER, _CREATED_7_DAYS_AGO
    )

    result = runner.invoke(
        ado,
        [
            "get",
            "datacontainers",
            "--stats",
            "--no-trunc",
        ],
    )

    assert result.exit_code == 0, result.output
    if os.environ.get("CI", "false") != "true":
        from ado.cli.utils.resources.formatters import _format_bytes

        expected_output = pd.DataFrame(
            data={
                "IDENTIFIER": container_id,
                "NAME": "",
                "AGE": _EXPECTED_AGE,
                "TABLES": stats.number_of_tables,
                "LOCATIONS": stats.number_of_locations,
                "KEY_VALUES": stats.number_of_key_values,
                "DATA_BYTES": _format_bytes(stats.total_data_bytes),
            },
            index=pd.Index([0]),
        )
        rendered_output = render_dataframe_to_cli_output(
            expected_output, do_not_truncate_columns=True
        )
        assert rendered_output in result.output, (
            f"Expected output:\n{rendered_output}\nnot found in:\n{result.output}"
        )


@requires_sqlite_3_38
def test_ado_get_operation_stats_details_columns(
    tmp_path: pathlib.Path,
    mysql_test_instance: MySqlContainer,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
    simulate_ml_multi_cloud_random_walk_operation: Callable[
        [int, int, int, str | None, dt.datetime | None],
        tuple[SQLSampleStore, list[MeasurementRequest], list[str]],
    ],
) -> None:
    """ado get operation <id> --stats --details adds DESCRIPTION and LABELS columns.

    The operation config (randomwalk_ml_multicloud_operation.yaml) carries the
    description 'Perform a random walk on all points in a space', so that string
    must appear in the rendered table when --details is given.  LABELS must also
    be present as a column header.
    """
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    operation_id = "op-stats-details-op-001"
    simulate_ml_multi_cloud_random_walk_operation(
        number_entities=1,
        number_requests=1,
        measurements_per_result=1,
        operation_id=operation_id,
    )

    result = runner.invoke(
        ado,
        [
            "get",
            "operation",
            operation_id,
            "--stats",
            "--details",
            "--no-trunc",
        ],
    )

    assert result.exit_code == 0, result.output
    if os.environ.get("CI", "false") != "true":
        assert "DESCRIPTION" in result.output, "DESCRIPTION column missing from output"
        assert "LABELS" in result.output, "LABELS column missing from output"
        assert "Perform a random walk on all points in a space" in result.output, (
            "Operation description missing from output"
        )


@requires_sqlite_3_38
def test_ado_get_space_stats_details_columns(
    tmp_path: pathlib.Path,
    mysql_test_instance: MySqlContainer,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
    ml_multi_cloud_space: DiscoverySpace,
) -> None:
    """ado get space <id> --stats --details adds DESCRIPTION and LABELS columns.

    The ml_multicloud_basic space YAML carries no description or labels, so the
    DESCRIPTION column must be present but empty, and LABELS must be present as
    a column header with no value.
    """
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    result = runner.invoke(
        ado,
        [
            "get",
            "space",
            ml_multi_cloud_space.uri,
            "--stats",
            "--details",
            "--no-trunc",
        ],
    )

    assert result.exit_code == 0, result.output
    if os.environ.get("CI", "false") != "true":
        assert "DESCRIPTION" in result.output, "DESCRIPTION column missing from output"
        assert "LABELS" in result.output, "LABELS column missing from output"


@requires_sqlite_3_38
def test_ado_get_stats_deprecated_output_format_warns_and_shows_stats(
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
    """ado get operations -o stats emits a deprecation warning and still shows stats columns."""
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
        ["get", "operations", "-o", "stats", "--no-trunc"],
    )

    assert result.exit_code == 0, result.output
    assert "deprecated" in result.output.lower()
    assert "--stats" in result.output
    assert "TOTAL_MEASUREMENTS" in result.output


_OPERATION_STATS_KEYS = {
    "total_measurements",
    "successful_measurements",
    "failed_measurements",
    "sampled_entities",
}
_SPACE_STATS_KEYS = {"experiments", "operations", "explore_ops", "sampled_entities"}
_SAMPLESTORE_STATS_KEYS = {"entities", "results", "experiments"}
_DATACONTAINER_STATS_KEYS = {"tables", "locations", "key_values", "data_bytes"}


@requires_sqlite_3_38
@pytest.mark.parametrize(
    ("resource_kind", "expected_keys", "output_format"),
    [
        ("operations", _OPERATION_STATS_KEYS, "yaml"),
        ("operations", _OPERATION_STATS_KEYS, "json"),
        ("spaces", _SPACE_STATS_KEYS, "yaml"),
        ("samplestores", _SAMPLESTORE_STATS_KEYS, "yaml"),
    ],
)
def test_ado_get_stats_structured_output(
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
    resource_kind: str,
    expected_keys: set[str],
    output_format: str,
) -> None:
    """--stats with -o yaml/json adds metadata.stats to every listed resource."""
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )
    simulate_ml_multi_cloud_random_walk_operation(
        number_entities=2, number_requests=1, measurements_per_result=1
    )

    result = runner.invoke(ado, ["get", resource_kind, "--stats", "-o", output_format])

    assert result.exit_code == 0, result.stderr
    assert "ignored" not in result.stderr.lower()
    resources = yaml.load(result.stdout, Loader=SafeLoader)
    assert len(resources) == 1
    assert set(resources[0]["metadata"]["stats"]) == expected_keys


@requires_sqlite_3_38
def test_ado_get_datacontainers_stats_structured_output(
    tmp_path: pathlib.Path,
    mysql_test_instance: MySqlContainer,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
    data_container_resource: "DataContainerResource",
    create_resources: Callable,
) -> None:
    """--stats with -o yaml adds metadata.stats to data containers."""
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )
    create_resources([data_container_resource])

    result = runner.invoke(ado, ["get", "datacontainers", "--stats", "-o", "yaml"])

    assert result.exit_code == 0, result.stderr
    resources = yaml.load(result.stdout, Loader=SafeLoader)
    assert len(resources) == 1
    assert set(resources[0]["metadata"]["stats"]) == _DATACONTAINER_STATS_KEYS


@requires_sqlite_3_38
def test_ado_get_operation_stats_yaml_single_resource_values(
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
    """ado get operation <id> --stats -o yaml has the expected metadata.stats values."""
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )
    number_entities = 3
    operation_id = "op-stats-yaml-single-123"
    simulate_ml_multi_cloud_random_walk_operation(
        number_entities=number_entities,
        number_requests=1,
        measurements_per_result=1,
        operation_id=operation_id,
    )

    result = runner.invoke(
        ado, ["get", "operation", operation_id, "--stats", "-o", "yaml"]
    )

    assert result.exit_code == 0, result.stderr
    resource = yaml.load(result.stdout, Loader=SafeLoader)
    assert resource["identifier"] == operation_id
    assert resource["metadata"]["stats"] == {
        "total_measurements": number_entities,
        "successful_measurements": number_entities,
        "failed_measurements": 0,
        "sampled_entities": number_entities,
    }


@requires_sqlite_3_38
def test_ado_get_operations_no_stats_yaml(
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
    """Without --stats the yaml output has no metadata.stats."""
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )
    simulate_ml_multi_cloud_random_walk_operation(
        number_entities=1, number_requests=1, measurements_per_result=1
    )

    result = runner.invoke(ado, ["get", "operations", "-o", "yaml"])

    assert result.exit_code == 0, result.stderr
    resources = yaml.load(result.stdout, Loader=SafeLoader)
    assert len(resources) == 1
    assert "stats" not in resources[0].get("metadata", {})


@requires_sqlite_3_38
@pytest.mark.parametrize("output_format", ["name", "raw", "config"])
def test_ado_get_stats_ignored_with_unsupported_format(
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
    output_format: str,
) -> None:
    """--stats is ignored with name/raw/config output and prints an INFO message."""
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )
    operation_id = "op-stats-ignored-456"
    simulate_ml_multi_cloud_random_walk_operation(
        number_entities=1,
        number_requests=1,
        measurements_per_result=1,
        operation_id=operation_id,
    )

    result = runner.invoke(
        ado, ["get", "operation", operation_id, "--stats", "-o", output_format]
    )

    assert result.exit_code == 0, result.stderr
    assert "--stats" in result.stderr
    assert "ignored" in result.stderr.lower()
    assert "total_measurements" not in result.stdout


@requires_sqlite_3_38
def test_ado_get_stats_structured_output_empty_listing(
    tmp_path: pathlib.Path,
    mysql_test_instance: MySqlContainer,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
) -> None:
    """--stats with -o yaml on a project without matching resources prints an empty list."""
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    result = runner.invoke(ado, ["get", "spaces", "--stats", "-o", "yaml"])

    assert result.exit_code == 0, result.stderr
    assert "ignored" not in result.stderr.lower()
    assert yaml.load(result.stdout, Loader=SafeLoader) == []


@requires_sqlite_3_38
def test_ado_get_spaces_stats_matching_point_structured_output(
    tmp_path: pathlib.Path,
    mysql_test_instance: MySqlContainer,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
    ml_multi_cloud_space: DiscoverySpace,
    ml_multi_cloud_matching_point: SpacePoint,
) -> None:
    """ado get spaces --stats --matching-point -o yaml adds metadata.stats to the matching spaces."""
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )
    point_file = tmp_path / "point.yaml"
    point_file.write_text(pydantic_model_as_yaml(ml_multi_cloud_matching_point))

    result = runner.invoke(
        ado,
        [
            "get",
            "spaces",
            "--stats",
            "--matching-point",
            str(point_file),
            "-o",
            "yaml",
        ],
    )

    assert result.exit_code == 0, result.stderr
    resources = yaml.load(result.stdout, Loader=SafeLoader)
    assert len(resources) == 1
    assert resources[0]["identifier"] == ml_multi_cloud_space.uri
    assert set(resources[0]["metadata"]["stats"]) == _SPACE_STATS_KEYS


@requires_sqlite_3_38
def test_ado_get_spaces_stats_matching_point(
    tmp_path: pathlib.Path,
    mysql_test_instance: MySqlContainer,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
    ml_multi_cloud_space: DiscoverySpace,
    ml_multi_cloud_matching_point: SpacePoint,
) -> None:
    """ado get spaces --stats --matching-point yields space stats columns, not operation stats."""
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    point_file = tmp_path / "point.yaml"
    point_file.write_text(pydantic_model_as_yaml(ml_multi_cloud_matching_point))

    result = runner.invoke(
        ado,
        [
            "get",
            "spaces",
            "--stats",
            "--matching-point",
            str(point_file),
            "--no-trunc",
        ],
    )

    assert result.exit_code == 0, result.output
    if os.environ.get("CI", "false") != "true":
        for col in ("EXPERIMENTS", "OPERATIONS", "EXPLORE_OPS", "SAMPLED_ENTITIES"):
            assert col in result.output, (
                f"Space stats column {col!r} missing from output"
            )
        assert "TOTAL_MEASUREMENTS" not in result.output
