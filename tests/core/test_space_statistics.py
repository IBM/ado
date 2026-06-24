# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import math
from collections.abc import Callable

from orchestrator.core.discoveryspace.space import DiscoverySpace
from orchestrator.core.discoveryspace.stats import (
    DiscoverySpaceStatistics,
    space_statistics_for_spaces,
)
from orchestrator.core.samplestore.sql import SQLSampleStore
from orchestrator.schema.request import MeasurementRequest
from tests.conftest import requires_sqlite_3_38

# ---------------------------------------------------------------------------
# Unit tests for DiscoverySpaceStatistics model
# ---------------------------------------------------------------------------


def test_heavy_fields_default_to_none() -> None:
    """Heavy fields are optional and default to None."""
    stats = DiscoverySpaceStatistics(
        number_of_experiments=1,
        number_of_operations=2,
        number_of_explore_operations=1,
        number_measured_entities=5,
    )
    assert stats.size_of_entity_space is None
    assert stats.number_unmeasured_entities is None
    assert stats.number_matching_entities is None
    assert stats.number_matching_entities_with_measurements is None


def test_nan_unmeasured_entities_round_trip() -> None:
    """math.nan in number_unmeasured_entities survives a model_dump/model_validate cycle."""
    stats = DiscoverySpaceStatistics(
        number_of_experiments=1,
        number_of_operations=2,
        number_of_explore_operations=1,
        number_measured_entities=5,
        size_of_entity_space=None,
        number_unmeasured_entities=math.nan,
        number_matching_entities=None,
        number_matching_entities_with_measurements=None,
    )
    restored = DiscoverySpaceStatistics.model_validate(stats.model_dump())
    assert restored.size_of_entity_space is None
    assert math.isnan(restored.number_unmeasured_entities)


def test_inf_unmeasured_entities_round_trip() -> None:
    """math.inf in number_unmeasured_entities survives a model_dump/model_validate cycle."""
    stats = DiscoverySpaceStatistics(
        number_of_experiments=1,
        number_of_operations=0,
        number_of_explore_operations=0,
        number_measured_entities=0,
        size_of_entity_space=None,
        number_unmeasured_entities=math.inf,
        number_matching_entities=None,
        number_matching_entities_with_measurements=None,
    )
    restored = DiscoverySpaceStatistics.model_validate(stats.model_dump())
    assert math.isinf(restored.number_unmeasured_entities)


# ---------------------------------------------------------------------------
# Integration tests for DiscoverySpace.space_statistics()
#
# The ml_multi_cloud_space fixture uses examples/ml-multi-cloud/ml_multicloud_space.yaml:
#   entity space: provider (3) x cpu_family (2) x vcpu_size (2) x nodes (4) = 48 points
#   experiments:  1  (benchmark_performance / replay)
#   CSV sample store: 42 distinct entities, all matching the entity space
# ---------------------------------------------------------------------------

# Expected constants for the ml_multi_cloud space
_ENTITY_SPACE_SIZE = 48
_NUMBER_OF_EXPERIMENTS = 1
_NUMBER_OF_MATCHING_ENTITIES = 42  # all CSV entities satisfy isEntityInSpace


@requires_sqlite_3_38
def test_space_statistics_lightweight_only(
    ml_multi_cloud_space: DiscoverySpace,
) -> None:
    """lightweight_only=True returns correct lightweight fields and None for heavy fields."""
    stats = ml_multi_cloud_space.space_statistics(lightweight_only=True)

    assert isinstance(stats, DiscoverySpaceStatistics)
    assert stats.number_of_experiments == _NUMBER_OF_EXPERIMENTS
    assert stats.number_of_operations == 0
    assert stats.number_of_explore_operations == 0
    assert stats.number_measured_entities == 0
    # Heavy fields must be None when lightweight_only=True
    assert stats.size_of_entity_space is None
    assert stats.number_unmeasured_entities is None
    assert stats.number_matching_entities is None
    assert stats.number_matching_entities_with_measurements is None


@requires_sqlite_3_38
def test_space_statistics_full_no_operations(
    ml_multi_cloud_space: DiscoverySpace,
) -> None:
    """Full stats on a space with no operations: entity space and matching counts are exact."""
    stats = ml_multi_cloud_space.space_statistics(lightweight_only=False)

    assert isinstance(stats, DiscoverySpaceStatistics)
    assert stats.number_of_experiments == _NUMBER_OF_EXPERIMENTS
    assert stats.number_of_operations == 0
    assert stats.number_of_explore_operations == 0
    assert stats.number_measured_entities == 0
    assert stats.size_of_entity_space == _ENTITY_SPACE_SIZE
    assert stats.number_unmeasured_entities == _ENTITY_SPACE_SIZE
    assert stats.number_matching_entities == _NUMBER_OF_MATCHING_ENTITIES
    # The CSV sample store already carries observed property values for the
    # benchmark_performance experiment, so all matching entities have measurements
    assert (
        stats.number_matching_entities_with_measurements == _NUMBER_OF_MATCHING_ENTITIES
    )


@requires_sqlite_3_38
def test_space_statistics_full_with_operation(
    ml_multi_cloud_space: DiscoverySpace,
    simulate_ml_multi_cloud_random_walk_operation: Callable[
        [int, int, int, str | None],
        tuple[SQLSampleStore, list[MeasurementRequest], list[str]],
    ],
) -> None:
    """Full stats reflect exact measured-entity counts after a single operation."""
    number_entities = 3
    simulate_ml_multi_cloud_random_walk_operation(
        number_entities=number_entities,
        number_requests=1,
    )

    stats = ml_multi_cloud_space.space_statistics(lightweight_only=False)

    assert stats.number_of_experiments == _NUMBER_OF_EXPERIMENTS
    assert stats.number_of_operations == 1
    # The simulated operation uses operationType="search", which counts as explore
    assert stats.number_of_explore_operations == 1
    assert stats.number_measured_entities == number_entities
    assert stats.size_of_entity_space == _ENTITY_SPACE_SIZE
    assert stats.number_unmeasured_entities == _ENTITY_SPACE_SIZE - number_entities
    assert stats.number_matching_entities == _NUMBER_OF_MATCHING_ENTITIES
    # The CSV sample store already carries observed property values for all entities,
    # so all matching entities have measurements regardless of the simulated operation
    assert (
        stats.number_matching_entities_with_measurements == _NUMBER_OF_MATCHING_ENTITIES
    )


def test_space_statistics_for_spaces_empty() -> None:
    """An empty list returns an empty dict without any DB access."""
    result = space_statistics_for_spaces([])
    assert result == {}


@requires_sqlite_3_38
def test_space_statistics_for_spaces_single(
    ml_multi_cloud_space: DiscoverySpace,
) -> None:
    """Single-space helper matches per-space method."""
    stats_direct = ml_multi_cloud_space.space_statistics(lightweight_only=True)
    stats_batch = space_statistics_for_spaces(
        [ml_multi_cloud_space], lightweight_only=True
    )

    assert ml_multi_cloud_space.uri in stats_batch
    batch_stats = stats_batch[ml_multi_cloud_space.uri]
    assert batch_stats.number_of_experiments == stats_direct.number_of_experiments
    assert batch_stats.number_of_operations == stats_direct.number_of_operations
    assert batch_stats.number_measured_entities == stats_direct.number_measured_entities
