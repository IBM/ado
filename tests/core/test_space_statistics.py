# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import math
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import datetime

from ado.core import OperationResource
from ado.core.discoveryspace.space import DiscoverySpace
from ado.core.discoveryspace.stats import (
    DiscoverySpaceStatistics,
    space_statistics_for_spaces,
)
from ado.core.operation.config import (
    DiscoveryOperationEnum,
    DiscoveryOperationResourceConfiguration,
)
from ado.core.samplestore.sql import SQLSampleStore
from ado.metastore.project import ProjectContext
from ado.metastore.sqlstore import SQLResourceStore
from ado.schema.reference import ExperimentReference
from ado.schema.request import (
    MeasurementRequest,
    MeasurementRequestStateEnum,
    ReplayedMeasurement,
)
from ado.schema.result import InvalidMeasurementResult, MeasurementResultStateEnum
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
    assert stats.sampled_full is None
    assert stats.sampled_partial is None
    assert stats.sampled_failed is None
    assert stats.matching_full is None
    assert stats.matching_partial is None
    assert stats.matching_failed is None


def test_nan_unmeasured_entities_round_trip() -> None:
    """math.nan in number_unmeasured_entities survives a model_dump/model_validate cycle."""
    stats = DiscoverySpaceStatistics(
        number_of_experiments=1,
        number_of_operations=2,
        number_of_explore_operations=1,
        number_measured_entities=5,
        size_of_entity_space=None,
        number_unmeasured_entities=math.nan,
        sampled_full=None,
        sampled_partial=None,
        sampled_failed=None,
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
        sampled_full=None,
        sampled_partial=None,
        sampled_failed=None,
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
    assert stats.sampled_full is None
    assert stats.sampled_partial is None
    assert stats.sampled_failed is None
    assert stats.matching_full is None
    assert stats.matching_partial is None
    assert stats.matching_failed is None


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
    # The CSV sample store already carries observed property values for the
    # benchmark_performance experiment, so all matching entities have measurements
    assert stats.matching_full == _NUMBER_OF_MATCHING_ENTITIES
    assert stats.matching_partial == 0
    assert stats.matching_failed == 0
    # No sampled entities (no operations) → all buckets are 0
    assert stats.sampled_full == 0
    assert stats.sampled_partial == 0
    assert stats.sampled_failed == 0


@requires_sqlite_3_38
def test_space_statistics_full_with_operation(
    ml_multi_cloud_space: DiscoverySpace,
    simulate_ml_multi_cloud_random_walk_operation: Callable[
        [
            int,
            int,
            int,
            str | None,
            "datetime.datetime | None",
            "MeasurementResultStateEnum | None",
        ],
        tuple[SQLSampleStore, list[MeasurementRequest], list[str]],
    ],
) -> None:
    """Full stats reflect exact measured-entity counts after a single operation."""
    number_entities = 3
    number_requests = 1
    measurements_per_result = 2
    simulate_ml_multi_cloud_random_walk_operation(
        number_entities=number_entities,
        number_requests=number_requests,
        measurements_per_result=measurements_per_result,
    )

    stats = ml_multi_cloud_space.space_statistics(lightweight_only=False)

    assert stats.number_of_experiments == _NUMBER_OF_EXPERIMENTS
    assert stats.number_of_operations == 1
    # The simulated operation uses operationType="explore"
    assert stats.number_of_explore_operations == 1
    assert stats.number_measured_entities == number_entities
    assert stats.size_of_entity_space == _ENTITY_SPACE_SIZE
    assert stats.number_unmeasured_entities == _ENTITY_SPACE_SIZE - number_entities
    # The CSV sample store already carries observed property values for all entities,
    # so all matching entities have measurements regardless of the simulated operation
    assert stats.matching_full == _NUMBER_OF_MATCHING_ENTITIES
    assert stats.matching_partial == 0
    assert stats.matching_failed == 0
    assert stats.sampled_full == number_entities
    assert stats.sampled_partial == 0
    assert stats.sampled_failed == 0


@requires_sqlite_3_38
def test_space_statistics_failure_only_entities(
    ml_multi_cloud_space: DiscoverySpace,
    ml_multi_cloud_sample_store: SQLSampleStore,
    ml_multi_cloud_operation_configuration: DiscoveryOperationResourceConfiguration,
    valid_ado_project_context: ProjectContext,
) -> None:
    """sampled_failed equals the number of sampled entities when all results are failures."""
    # The ml_multi_cloud entity space has 48 points; the CSV store seeds 42.
    # Enumerate all entity-space points and collect those whose identifier is
    # not yet in the SQL store — these have no valid measurements of any kind.
    store_identifiers = ml_multi_cloud_sample_store.entity_identifiers()
    prop_names = [
        c.identifier for c in ml_multi_cloud_space.entitySpace.constitutiveProperties
    ]
    entities_to_fail = []
    for point in ml_multi_cloud_space.entitySpace.sequential_point_iterator():
        entity = ml_multi_cloud_space.entitySpace.entity_for_point(
            dict(zip(prop_names, point, strict=True))
        )
        if entity.identifier not in store_identifiers:
            entities_to_fail.append(entity)
        if len(entities_to_fail) == 3:
            break
    assert len(entities_to_fail) == 3, (
        "Not enough entity-space points absent from the store to run this test"
    )

    # Add the entities to the store so get_entities() can find them during
    # Pass 1 of space_statistics.  They have no valid measurements at this
    # point — only the InvalidMeasurementResult added below.
    ml_multi_cloud_sample_store.addEntities(entities_to_fail)

    # Register a synthetic operation in the metastore linked to this space.
    operation_id = "regression-bug1-failure-only-001"
    sql = SQLResourceStore(project_context=valid_ado_project_context)
    resource = OperationResource(
        identifier=operation_id,
        config=ml_multi_cloud_operation_configuration,
        operationType=DiscoveryOperationEnum.EXPLORE,
        operatorIdentifier="test",
    )
    sql.addResourceWithRelationships(
        resource,
        relatedIdentifiers=ml_multi_cloud_operation_configuration.spaces,
    )

    # Add a request and invalid results for the unmeasured entities.
    exp_ref = ExperimentReference(
        experimentIdentifier="benchmark_performance",
        actuatorIdentifier="replay",
    )
    request = ReplayedMeasurement(
        operation_id=operation_id,
        requestIndex=0,
        experimentReference=exp_ref,
        entities=tuple(entities_to_fail),
        requestid="regression-req-001",
        status=MeasurementRequestStateEnum.SUCCESS,
        measurements=tuple(
            InvalidMeasurementResult(
                entityIdentifier=e.identifier,
                reason="test failure",
                experimentReference=exp_ref,
            )
            for e in entities_to_fail
        ),
    )
    request_db_id = ml_multi_cloud_sample_store.add_measurement_request(request=request)
    ml_multi_cloud_sample_store.add_measurement_results(
        results=list(request.measurements),
        skip_relationship_to_request=False,
        request_db_id=request_db_id,
    )

    stats = ml_multi_cloud_space.space_statistics(lightweight_only=False)

    assert stats.sampled_failed == len(entities_to_fail)
    assert stats.sampled_full == 0
    assert stats.sampled_partial == 0


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
