# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Tests for `ado show measurements samplestore`."""

import pathlib
from collections.abc import Callable

from typer.testing import CliRunner

from ado.cli.core.cli import app as ado
from ado.core.samplestore.sql import SQLSampleStore
from ado.metastore.project import ProjectContext
from ado.schema.request import MeasurementRequest
from tests.conftest import requires_sqlite_3_38


@requires_sqlite_3_38
def test_show_measurements_samplestore_basic(
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
    """Measurements present: exit 0 and non-empty output."""
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
        ],
    )

    assert result.exit_code == 0, result.output
    assert result.output.strip()


@requires_sqlite_3_38
def test_show_measurements_samplestore_nonexistent(
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
            "show",
            "measurements",
            "samplestore",
            "does-not-exist",
        ],
    )

    assert result.exit_code != 0


@requires_sqlite_3_38
def test_show_measurements_samplestore_empty(
    tmp_path: pathlib.Path,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
    empty_sample_store: SQLSampleStore,
) -> None:
    """Empty samplestore (no measurements): exit 0 and informational message."""
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    result = runner.invoke(
        ado,
        [
            "show",
            "measurements",
            "samplestore",
            empty_sample_store.identifier,
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    assert "Nothing was returned for" in result.output


@requires_sqlite_3_38
def test_show_measurements_samplestore_unsupported_include(
    tmp_path: pathlib.Path,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
    ml_multi_cloud_sample_store: SQLSampleStore,
) -> None:
    """Unsupported --include value for samplestore: non-zero exit code."""
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    result = runner.invoke(
        ado,
        [
            "show",
            "measurements",
            "samplestore",
            ml_multi_cloud_sample_store.identifier,
            "--include",
            "matching",
        ],
    )

    assert result.exit_code != 0
