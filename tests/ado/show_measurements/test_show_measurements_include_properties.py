# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Integration tests for `ado show measurements --include-properties` / `-p`."""

import json
import pathlib
from collections.abc import Callable

from typer.testing import CliRunner

from ado.cli.core.cli import app as ado
from ado.core.discoveryspace.space import DiscoverySpace
from ado.core.samplestore.sql import SQLSampleStore
from ado.metastore.project import ProjectContext
from ado.schema.request import MeasurementRequest
from tests.conftest import requires_sqlite_3_38


@requires_sqlite_3_38
def test_show_measurements_space_include_properties_optional(
    tmp_path: pathlib.Path,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
    peptide_mineralization_basic_space: DiscoverySpace,
) -> None:
    """Option -p optional includes default optional properties in space measurements."""
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    result = runner.invoke(
        ado,
        [
            "show",
            "measurements",
            "space",
            peptide_mineralization_basic_space.uri,
            "--include",
            "unmeasured",
            "-p",
            "optional",
            "--no-trunc",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "temperature" in result.output
    assert "23" in result.output
    assert "robot_identifier" in result.output
    assert "hermione" in result.output


@requires_sqlite_3_38
def test_show_measurements_space_include_properties_all(
    tmp_path: pathlib.Path,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
    peptide_mineralization_basic_space: DiscoverySpace,
) -> None:
    """Option -p all includes constitutive and optional properties."""
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    result = runner.invoke(
        ado,
        [
            "show",
            "measurements",
            "space",
            peptide_mineralization_basic_space.uri,
            "--include",
            "unmeasured",
            "-p",
            "all",
            "--no-trunc",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "peptide_identifier" in result.output
    assert "temperature" in result.output
    assert "23" in result.output
    assert "robot_identifier" in result.output
    assert "hermione" in result.output


@requires_sqlite_3_38
def test_show_measurements_space_include_properties_named(
    tmp_path: pathlib.Path,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
    peptide_mineralization_basic_space: DiscoverySpace,
) -> None:
    """Option -p <property_name> includes the specific named property."""
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    result = runner.invoke(
        ado,
        [
            "show",
            "measurements",
            "space",
            peptide_mineralization_basic_space.uri,
            "--include",
            "unmeasured",
            "-p",
            "temperature",
            "--no-trunc",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "temperature" in result.output
    assert "23" in result.output
    assert "robot_identifier" not in result.output


@requires_sqlite_3_38
def test_show_measurements_space_include_properties_csv_and_json(
    tmp_path: pathlib.Path,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
    peptide_mineralization_basic_space: DiscoverySpace,
) -> None:
    """Output formats csv and json include the requested properties."""
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    # CSV
    result_csv = runner.invoke(
        ado,
        [
            "show",
            "measurements",
            "space",
            peptide_mineralization_basic_space.uri,
            "--include",
            "unmeasured",
            "-p",
            "optional",
            "-o",
            "csv",
        ],
    )
    assert result_csv.exit_code == 0, result_csv.output
    assert "temperature" in result_csv.output
    assert "23" in result_csv.output

    # JSON
    result_json = runner.invoke(
        ado,
        [
            "show",
            "measurements",
            "space",
            peptide_mineralization_basic_space.uri,
            "--include",
            "unmeasured",
            "-p",
            "optional",
            "-o",
            "json",
        ],
    )
    assert result_json.exit_code == 0, result_json.output
    measurements_json_data = json.loads(result_json.output)
    assert len(measurements_json_data) > 0
    assert "temperature" in measurements_json_data
    assert any(
        property_value == 23
        for property_value in measurements_json_data["temperature"].values()
    )


@requires_sqlite_3_38
def test_show_measurements_property_and_include_properties_are_mutually_exclusive(
    tmp_path: pathlib.Path,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
    peptide_mineralization_basic_space: DiscoverySpace,
) -> None:
    """Combining --property and -p exits with an error."""
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    result = runner.invoke(
        ado,
        [
            "show",
            "measurements",
            "space",
            peptide_mineralization_basic_space.uri,
            "-p",
            "temperature",
            "--property",
            "adsorption_timeseries",
            "--no-trunc",
        ],
    )

    assert result.exit_code == 1
    assert "cannot be used together" in result.output


@requires_sqlite_3_38
def test_show_measurements_operation_include_properties(
    tmp_path: pathlib.Path,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
    simulate_ml_multi_cloud_random_walk_operation: Callable[
        [int, int, int, str | None],
        tuple[SQLSampleStore, list[MeasurementRequest], list[str]],
    ],
) -> None:
    """Operation measurements include requested experiment properties."""
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    (
        _sample_store,
        measurement_requests,
        _entity_ids,
    ) = simulate_ml_multi_cloud_random_walk_operation(
        number_entities=2,
        number_requests=2,
        measurements_per_result=1,
    )
    operation_id = measurement_requests[0].operation_id

    result = runner.invoke(
        ado,
        [
            "show",
            "measurements",
            "operation",
            operation_id,
            "-p",
            "constitutive",
            "--no-trunc",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "provider" in result.output


@requires_sqlite_3_38
def test_show_measurements_samplestore_include_properties(
    tmp_path: pathlib.Path,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
    simulate_ml_multi_cloud_random_walk_operation: Callable[
        [int, int, int, str | None],
        tuple[SQLSampleStore, list[MeasurementRequest], list[str]],
    ],
    ml_multi_cloud_sample_store: SQLSampleStore,
) -> None:
    """Samplestore measurements support --include-properties."""
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
            "measurements",
            "samplestore",
            ml_multi_cloud_sample_store.identifier,
            "-p",
            "constitutive",
            "--no-trunc",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "provider" in result.output
