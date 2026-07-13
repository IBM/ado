# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Integration tests for `ado show stats` — covering only behaviour not
tested elsewhere.

* Space stats: ``ado get space -o stats`` only emits the lightweight columns
  (EXPERIMENTS, OPERATIONS, EXPLORE_OPERATIONS, MEASURED_ENTITIES).
  ``ado show stats discoveryspace`` is the **only** CLI path that appends the
  heavy columns (SIZE_OF_ENTITY_SPACE, UNMEASURED_ENTITIES, MATCHING_ENTITIES,
  MATCHING_WITH_MEASUREMENTS, ENTITIES_WITH_ALL_MEASUREMENTS,
  ENTITIES_WITH_PARTIAL_MEASUREMENTS, MATCHING_ENTITIES_WITH_ALL_MEASUREMENTS).

* Operation request-level stats: ``ado get operation -o stats`` only emits the
  result-level columns (TOTAL_RESULTS, …, MEASURED_ENTITIES).
  ``ado show stats operation`` is the **only** CLI path that also appends the
  request-level columns (TOTAL_REQUESTS, FAILED_REQUESTS, SUCCESSFUL_REQUESTS).

All other aspects (output formats, --use-latest, samplestore, datacontainer,
query filters, loading all resources when no IDs are supplied, ``--details``
columns) are already covered by ``tests/ado/get/test_ado_get_stats.py`` and
``tests/core/test_space_statistics.py``.
"""

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

# Expected constants for the ml_multi_cloud space
# (entity space: provider x cpu_family x vcpu_size x nodes = 3x2x2x4 = 48 points)
_ENTITY_SPACE_SIZE = 48
_NUMBER_OF_MATCHING_ENTITIES = 42  # all CSV entities satisfy isEntityInSpace


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
        [int, int, int, str | None],
        tuple[SQLSampleStore, list[MeasurementRequest], list[str]],
    ],
) -> None:
    """show stats discoveryspace emits accurate values for all heavy stats columns.

    ``ado get space -o stats`` does **not** compute the heavy fields; this test
    verifies the values that are unique to ``ado show stats discoveryspace``.

    Setup: 1 operation, 1 request, 3 entities measured (all with the single
    benchmark_performance experiment).  Expected values are derived from the
    known ml_multi_cloud space geometry and CSV sample store:
      - SIZE_OF_ENTITY_SPACE: 48 (3x2x2x4 entity-space points)
      - UNMEASURED_ENTITIES: 45 (48 - 3 measured)
      - MATCHING_ENTITIES: 42 (all CSV entities satisfy isEntityInSpace)
      - MATCHING_WITH_MEASUREMENTS: 42 (CSV sample store carries observations)
      - ENTITIES_WITH_ALL_MEASUREMENTS: 3 (one experiment, each entity has one result)
      - ENTITIES_WITH_PARTIAL_MEASUREMENTS: 0
      - MATCHING_ENTITIES_WITH_ALL_MEASUREMENTS: 42
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

    assert stats["SIZE_OF_ENTITY_SPACE"] == _ENTITY_SPACE_SIZE
    assert isinstance(stats["SIZE_OF_ENTITY_SPACE"], int)
    assert stats["UNMEASURED_ENTITIES"] == _ENTITY_SPACE_SIZE - number_entities
    assert isinstance(stats["UNMEASURED_ENTITIES"], int)
    assert stats["MATCHING_ENTITIES"] == _NUMBER_OF_MATCHING_ENTITIES
    assert stats["MATCHING_WITH_MEASUREMENTS"] == _NUMBER_OF_MATCHING_ENTITIES
    assert stats["ENTITIES_WITH_ALL_MEASUREMENTS"] == number_entities
    assert stats["ENTITIES_WITH_PARTIAL_MEASUREMENTS"] == 0
    assert (
        stats["MATCHING_ENTITIES_WITH_ALL_MEASUREMENTS"] == _NUMBER_OF_MATCHING_ENTITIES
    )


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
        [int, int, int, str | None],
        tuple[SQLSampleStore, list[MeasurementRequest], list[str]],
    ],
) -> None:
    """show stats operation emits accurate values for the request-level columns.

    ``ado get operation -o stats`` does **not** compute the request columns;
    this test verifies the values that are unique to ``ado show stats operation``.

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
