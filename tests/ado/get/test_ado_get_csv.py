# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Tests for `ado get <resource> -o csv`."""

import datetime as dt
import io
import pathlib
from collections.abc import Callable

import pandas as pd
import yaml
from testcontainers.community.mysql import MySqlContainer
from typer.testing import CliRunner

from ado.cli.core.cli import app as ado
from ado.cli.core.config import AdoConfiguration
from ado.core import DiscoverySpaceResource
from ado.core.discoveryspace.space import DiscoverySpace
from ado.core.resources import ADOResource
from ado.core.samplestore.sql import SQLSampleStore
from ado.metastore.project import ProjectContext
from ado.schema.reference import ExperimentReference
from ado.schema.request import MeasurementRequest
from tests.conftest import requires_sqlite_3_38


def _parse_csv(content: str) -> pd.DataFrame:
    """Parse CSV text into a DataFrame with all values kept as strings."""
    return pd.read_csv(io.StringIO(content), dtype=str, keep_default_na=False)


###########################################################
# Resources that do not require a metastore
###########################################################
def test_get_experiments_csv_stdout_contains_only_csv() -> None:
    runner = CliRunner()
    result = runner.invoke(ado, ["get", "experiments", "-o", "csv"])
    assert result.exit_code == 0, result.stderr

    # The local-command warning goes to stderr, not into the CSV
    assert "local command" not in result.stdout
    assert "local command" in result.stderr

    # No INDEX column, unlike the table output
    assert result.stdout.splitlines()[0] == "ACTUATOR ID,EXPERIMENT ID,VERSION"
    assert len(_parse_csv(result.stdout)) > 0


def test_get_experiments_csv_does_not_truncate_and_ignores_no_trunc() -> None:
    runner = CliRunner()
    default_result = runner.invoke(ado, ["get", "experiments", "-o", "csv"])
    no_trunc_result = runner.invoke(
        ado, ["get", "experiments", "-o", "csv", "--no-trunc"]
    )
    assert default_result.exit_code == 0
    assert no_trunc_result.exit_code == 0
    assert default_result.stdout == no_trunc_result.stdout
    assert "…" not in default_result.stdout


def test_get_experiments_csv_output_file(tmp_path: pathlib.Path) -> None:
    runner = CliRunner()
    output_file = tmp_path / "experiments.csv"

    stdout_result = runner.invoke(ado, ["get", "experiments", "-o", "csv"])
    file_result = runner.invoke(
        ado,
        ["get", "experiments", "-o", "csv", "--output-file", str(output_file)],
    )

    assert file_result.exit_code == 0, file_result.stderr
    assert file_result.stdout == ""
    assert "Output written to" in file_result.stderr
    assert output_file.read_text() == stdout_result.stdout


def test_get_actuators_csv() -> None:
    runner = CliRunner()
    result = runner.invoke(ado, ["get", "actuators", "-o", "csv"])
    assert result.exit_code == 0, result.stderr

    df = _parse_csv(result.stdout)
    assert list(df.columns) == ["ACTUATOR ID", "EXPERIMENTS"]
    assert len(df) > 0


def test_get_operators_csv_has_no_banner_on_stdout() -> None:
    runner = CliRunner()
    result = runner.invoke(ado, ["get", "operators", "-o", "csv"])
    assert result.exit_code == 0, result.stderr

    assert "Available operators by type" not in result.stdout
    df = _parse_csv(result.stdout)
    assert list(df.columns) == ["OPERATOR", "VERSION", "TYPE"]
    assert {"random_walk", "ray_tune"}.issubset(set(df["OPERATOR"]))


###########################################################
# Contexts
###########################################################
def test_ado_contexts_csv(tmp_path: pathlib.Path) -> None:
    ado_configuration = AdoConfiguration.load(
        do_not_fail_on_available_contexts=True, _override_config_dir=tmp_path
    )
    ado_configuration.project_context_path_for_context("first-context").touch()
    ado_configuration.project_context_path_for_context("second-context").touch()

    runner = CliRunner()
    result = runner.invoke(ado, ["contexts", "-o", "csv"])
    assert result.exit_code == 0, result.stderr

    # The ACTIVE column holds booleans rather than the emoji used by the table
    assert (
        result.stdout == "CONTEXT,ACTIVE\nfirst-context,False\nlocal,True\n"
        "second-context,False\n"
    )


###########################################################
# Resources stored in the metastore
###########################################################
@requires_sqlite_3_38
def test_get_operations_csv(
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
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )
    simulate_ml_multi_cloud_random_walk_operation(
        number_entities=2,
        number_requests=1,
        measurements_per_result=1,
        operation_id="op-csv-test",
    )

    result = runner.invoke(ado, ["get", "operations", "-o", "csv"])
    assert result.exit_code == 0, result.stderr

    # stdout is only CSV: no spinner text, no INDEX column
    df = _parse_csv(result.stdout)
    assert list(df.columns) == [
        "IDENTIFIER",
        "NAME",
        "SPACE",
        "STATUS",
        "EXIT_STATE",
        "AGE",
    ]
    assert list(df["IDENTIFIER"]) == ["op-csv-test"]
    assert list(df["SPACE"]) == [ml_multi_cloud_space.uri]
    # Identifiers are never truncated
    assert "…" not in result.stdout


@requires_sqlite_3_38
def test_get_operations_csv_stats(
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
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )
    number_entities = 3
    simulate_ml_multi_cloud_random_walk_operation(
        number_entities=number_entities,
        number_requests=1,
        measurements_per_result=1,
        operation_id="op-csv-stats-test",
    )

    result = runner.invoke(ado, ["get", "operations", "-o", "csv", "--stats"])
    assert result.exit_code == 0, result.stderr
    assert "does not support statistics" not in result.stderr

    df = _parse_csv(result.stdout)
    assert list(df["TOTAL_MEASUREMENTS"]) == [str(number_entities)]
    assert list(df["SUCCESSFUL_MEASUREMENTS"]) == [str(number_entities)]
    assert list(df["FAILED_MEASUREMENTS"]) == ["0"]
    assert list(df["SAMPLED_ENTITIES"]) == [str(number_entities)]


@requires_sqlite_3_38
def test_get_spaces_csv_quotes_special_characters(
    tmp_path: pathlib.Path,
    mysql_test_instance: MySqlContainer,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
    create_resources: Callable[[list[ADOResource]], None],
    make_continuous_space_resource: Callable[..., DiscoverySpaceResource],
) -> None:
    """Commas, quotes and newlines in names and descriptions round-trip."""
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )
    space = make_continuous_space_resource(
        "csv-space", [0, 10], name='Name, with "quotes"'
    )
    space.config.metadata.description = "line one\nline two, with comma"
    create_resources([space])

    result = runner.invoke(ado, ["get", "spaces", "-o", "csv", "--details"])
    assert result.exit_code == 0, result.stderr

    rows = _parse_csv(result.stdout).set_index("IDENTIFIER")
    assert rows.loc["csv-space", "NAME"] == 'Name, with "quotes"'
    assert rows.loc["csv-space", "DESCRIPTION"] == "line one\nline two, with comma"


@requires_sqlite_3_38
def test_get_spaces_csv_empty_result(
    tmp_path: pathlib.Path,
    mysql_test_instance: MySqlContainer,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
) -> None:
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    result = runner.invoke(ado, ["get", "spaces", "-o", "csv"])
    assert result.exit_code == 0, result.stderr
    assert result.stdout == ""
    assert "Nothing was returned" in result.stderr


@requires_sqlite_3_38
def test_get_spaces_csv_matching_space_id(
    tmp_path: pathlib.Path,
    mysql_test_instance: MySqlContainer,
    create_resources: Callable[[list[ADOResource]], None],
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
    make_continuous_space_resource: Callable[..., DiscoverySpaceResource],
    experiment_reference: ExperimentReference,
) -> None:
    """csv is not downgraded to table when combined with --matching-space-id."""
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )
    exp_a = experiment_reference.model_copy(
        update={"experimentIdentifier": "exp1", "experimentVersion": "1.0.0"}
    )
    create_resources(
        [
            make_continuous_space_resource("csv-ref", [0, 10], [exp_a], name="Ref"),
            make_continuous_space_resource("csv-same", [0, 10], [exp_a], name="Same"),
        ]
    )

    result = runner.invoke(
        ado,
        ["get", "spaces", "-o", "csv", "--matching-space-id", "csv-ref"],
    )
    assert result.exit_code == 0, result.stderr
    assert "only support" not in result.stderr

    df = _parse_csv(result.stdout)
    assert set(df["IDENTIFIER"]) == {"csv-ref", "csv-same"}
    assert "ENTITY_SPACE_MATCH" in df.columns


@requires_sqlite_3_38
def test_get_spaces_csv_matching_point(
    tmp_path: pathlib.Path,
    mysql_test_instance: MySqlContainer,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
    ml_multi_cloud_space: DiscoverySpace,
) -> None:
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )
    point_file = tmp_path / "point.yaml"
    point_file.write_text(
        yaml.safe_dump(
            {
                "entity": {
                    "provider": "A",
                    "cpu_family": 0,
                    "vcpu_size": 0,
                    "nodes": 2,
                },
                "experiments": [
                    {
                        "experimentIdentifier": "benchmark_performance",
                        "actuatorIdentifier": "replay",
                    }
                ],
            }
        )
    )

    result = runner.invoke(
        ado,
        ["get", "spaces", "-o", "csv", "--matching-point", str(point_file)],
    )
    assert result.exit_code == 0, result.stderr
    assert list(_parse_csv(result.stdout)["IDENTIFIER"]) == [ml_multi_cloud_space.uri]
