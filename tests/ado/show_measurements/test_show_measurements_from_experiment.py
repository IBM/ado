# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Tests for `ado show measurements --from-experiment`."""

import pathlib
import typing
from collections.abc import Callable

import yaml
from typer.testing import CliRunner

from ado.cli.core.cli import app as ado
from ado.core.samplestore.sql import SQLSampleStore
from ado.metastore.project import ProjectContext
from ado.schema.reference import ExperimentReference
from ado.schema.request import MeasurementRequest
from ado.utilities.output import pydantic_model_as_yaml
from tests.conftest import requires_sqlite_3_38

if typing.TYPE_CHECKING:
    from ado.core.discoveryspace.space import DiscoverySpace

# The experiment used by all ml_multi_cloud simulation fixtures.
_EXPERIMENT_ACTUATOR = "replay"
_EXPERIMENT_ID = "benchmark_performance"
_FULLY_QUALIFIED = f"{_EXPERIMENT_ACTUATOR}.{_EXPERIMENT_ID}"


@requires_sqlite_3_38
def test_from_experiment_samplestore_string_filter(
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
    """Filter by known experiment identifier string returns a non-empty result."""
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
            "--from-experiment",
            _FULLY_QUALIFIED,
        ],
    )

    assert result.exit_code == 0, result.output
    assert result.output.strip()


@requires_sqlite_3_38
def test_from_experiment_samplestore_multiple_filters(
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
    """Multiple --from-experiment values use OR semantics: one matching + one non-matching still returns results."""
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
            "--from-experiment",
            _FULLY_QUALIFIED,
            "--from-experiment",
            "no_such_actuator.no_such_experiment",
        ],
    )

    assert result.exit_code == 0, result.output
    assert result.output.strip()


@requires_sqlite_3_38
def test_from_experiment_samplestore_yaml_file_filter(
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
    """Filter by YAML ExperimentReference file returns a non-empty result."""
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    simulate_ml_multi_cloud_random_walk_operation(
        number_entities=2,
        number_requests=2,
        measurements_per_result=1,
    )

    ref = ExperimentReference(
        actuatorIdentifier=_EXPERIMENT_ACTUATOR,
        experimentIdentifier=_EXPERIMENT_ID,
    )
    ref_file = tmp_path / "experiment_ref.yaml"
    ref_file.write_text(pydantic_model_as_yaml(ref))

    result = runner.invoke(
        ado,
        [
            "show",
            "measurements",
            "samplestore",
            ml_multi_cloud_sample_store.identifier,
            "--from-experiment",
            str(ref_file),
        ],
    )

    assert result.exit_code == 0, result.output
    assert result.output.strip()


@requires_sqlite_3_38
def test_from_experiment_samplestore_unqualified_experiment_name(
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
    """Bare experiment name without actuator prefix exits non-zero."""
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
            "--from-experiment",
            _EXPERIMENT_ID,
        ],
    )

    assert result.exit_code != 0
    assert "Could not resolve actuator" in result.output


@requires_sqlite_3_38
def test_from_experiment_samplestore_no_match(
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
    """When a valid but non-matching experiment is supplied, exit 0 with informational message."""
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    simulate_ml_multi_cloud_random_walk_operation(
        number_entities=2,
        number_requests=2,
        measurements_per_result=1,
    )

    # Use a fully-qualified experiment that won't match any entities in the store.
    result = runner.invoke(
        ado,
        [
            "show",
            "measurements",
            "samplestore",
            ml_multi_cloud_sample_store.identifier,
            "--from-experiment",
            "replay.benchmark_performance@v99",
        ],
    )

    assert result.exit_code == 0
    assert "No entities with measurements from experiments" in result.output


@requires_sqlite_3_38
def test_from_experiment_samplestore_invalid_yaml_file(
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
    """A YAML file that cannot be parsed as ExperimentReference exits non-zero."""
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    simulate_ml_multi_cloud_random_walk_operation(
        number_entities=2,
        number_requests=2,
        measurements_per_result=1,
    )

    bad_ref_file = tmp_path / "bad_experiment_ref.yaml"
    bad_ref_file.write_text(yaml.dump({"notAField": "badValue"}))

    result = runner.invoke(
        ado,
        [
            "show",
            "measurements",
            "samplestore",
            ml_multi_cloud_sample_store.identifier,
            "--from-experiment",
            str(bad_ref_file),
        ],
    )

    assert result.exit_code != 0


@requires_sqlite_3_38
def test_from_experiment_space_string_filter(
    tmp_path: pathlib.Path,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
    simulate_ml_multi_cloud_random_walk_operation: Callable[
        [int, int, int, str | None],
        tuple[SQLSampleStore, list[MeasurementRequest], list[str]],
    ],
    ml_multi_cloud_space: "DiscoverySpace",
) -> None:
    """Filter by known experiment identifier string for a space returns non-empty result."""
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
            "space",
            ml_multi_cloud_space.uri,
            "--from-experiment",
            _FULLY_QUALIFIED,
        ],
    )

    assert result.exit_code == 0, result.output
    assert result.output.strip()


@requires_sqlite_3_38
def test_from_experiment_operation_string_filter(
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
    """Filter by known experiment identifier string for an operation returns non-empty result."""
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    _store, _requests, _ids = simulate_ml_multi_cloud_random_walk_operation(
        number_entities=2,
        number_requests=2,
        measurements_per_result=1,
    )

    # The operation_id is embedded in each request.
    operation_id = _requests[0].operation_id

    result = runner.invoke(
        ado,
        [
            "show",
            "measurements",
            "operation",
            operation_id,
            "--from-experiment",
            _FULLY_QUALIFIED,
        ],
    )

    assert result.exit_code == 0, result.output
    assert result.output.strip()
