# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Integration tests for `ado show stats`."""

import json
import pathlib
from collections.abc import Callable
from typing import TYPE_CHECKING

from typer.testing import CliRunner

if TYPE_CHECKING:
    import datetime

    from ado.schema.result import MeasurementResultStateEnum

from ado.cli.core.cli import app as ado
from ado.core.discoveryspace.space import DiscoverySpace
from ado.core.samplestore.sql import SQLSampleStore
from ado.metastore.project import ProjectContext
from ado.schema.request import MeasurementRequest
from tests.conftest import requires_sqlite_3_38

# Expected constants for the ml_multi_cloud space
# (entity space: provider x cpu_family x vcpu_size x nodes = 3x2x2x4 = 48 points)
_ENTITY_SPACE_SIZE = 48
_NUMBER_OF_MATCHING_ENTITIES = (
    42  # all CSV entities satisfy isEntityInSpace; all have valid results
)


# ---------------------------------------------------------------------------
# discoveryspace — heavy stats columns (unique to show stats)
# ---------------------------------------------------------------------------


@requires_sqlite_3_38
def test_show_stats_discoveryspace_heavy_stats_values(
    tmp_path: pathlib.Path,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
    ml_multi_cloud_space: DiscoverySpace,
    simulate_ml_multi_cloud_random_walk_operation: Callable[
        [
            int,
            int,
            int,
            "str | None",
            "datetime.datetime | None",
            "MeasurementResultStateEnum | None",
        ],
        tuple[SQLSampleStore, list[MeasurementRequest], list[str]],
    ],
) -> None:
    """show stats discoveryspace emits accurate values for all heavy stats columns.

    Setup: 1 operation, 1 request, 3 entities measured (all with the single
    benchmark_performance experiment).  Expected values are derived from the
    known ml_multi_cloud space geometry and CSV sample store:
      - ENTITY_SPACE_SIZE: 48 (3x2x2x4 entity-space points)
      - UNSAMPLED: 45 (48 - 3 measured)
      - SAMPLED_FULL: 3 (one experiment, each entity has one valid result)
      - SAMPLED_PARTIAL: 0
      - SAMPLED_FAILED: 0
      - MATCHING_FULL: 42 (CSV sample store carries observations for all matching entities)
      - MATCHING_PARTIAL: 0
      - MATCHING_FAILED: 0
    """
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    number_entities = 3
    simulate_ml_multi_cloud_random_walk_operation(
        number_entities=number_entities,
        number_requests=1,
        measurements_per_result=1,
    )

    result = runner.invoke(
        ado,
        [
            "show",
            "stats",
            "discoveryspace",
            ml_multi_cloud_space.uri,
            "-o",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert ml_multi_cloud_space.uri in data
    stats = data[ml_multi_cloud_space.uri]

    assert stats["ENTITY_SPACE_SIZE"] == _ENTITY_SPACE_SIZE
    assert isinstance(stats["ENTITY_SPACE_SIZE"], int)
    assert stats["UNSAMPLED"] == _ENTITY_SPACE_SIZE - number_entities
    assert isinstance(stats["UNSAMPLED"], int)
    assert stats["SAMPLED_FULL"] == number_entities
    assert stats["SAMPLED_PARTIAL"] == 0
    assert stats["SAMPLED_FAILED"] == 0
    assert stats["MATCHING_FULL"] == _NUMBER_OF_MATCHING_ENTITIES
    assert stats["MATCHING_PARTIAL"] == 0
    assert stats["MATCHING_FAILED"] == 0


# ---------------------------------------------------------------------------
# operation — request-level stats columns (unique to show stats)
# ---------------------------------------------------------------------------


@requires_sqlite_3_38
def test_show_stats_operation_request_level_stats_values(
    tmp_path: pathlib.Path,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
    simulate_ml_multi_cloud_random_walk_operation: Callable[
        [
            int,
            int,
            int,
            "str | None",
            "datetime.datetime | None",
            "MeasurementResultStateEnum | None",
        ],
        tuple[SQLSampleStore, list[MeasurementRequest], list[str]],
    ],
) -> None:
    """show stats operation emits accurate values for the request-level columns.

    Setup: 1 operation, 3 requests (all SUCCESS by default), 2 entities each.
    Expected values:
      - TOTAL_REQUESTS: 3
      - FAILED_REQUESTS: 0
      - SUCCESSFUL_REQUESTS: 3
    """
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    number_requests = 3
    operation_id = "show-stats-op-requests-001"
    simulate_ml_multi_cloud_random_walk_operation(
        number_entities=2,
        number_requests=number_requests,
        measurements_per_result=1,
        operation_id=operation_id,
    )

    result = runner.invoke(
        ado,
        [
            "show",
            "stats",
            "operation",
            operation_id,
            "-o",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert operation_id in data
    stats = data[operation_id]

    assert stats["TOTAL_REQUESTS"] == number_requests
    assert stats["FAILED_REQUESTS"] == 0
    assert stats["SUCCESSFUL_REQUESTS"] == number_requests
