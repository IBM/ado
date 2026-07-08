# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Integration tests for measurement filtering with realistic database contents."""

import json
from collections.abc import Callable

from ado.core.samplestore.sql import SQLSampleStore
from ado.schema.request import MeasurementRequest


def test_filter_measurement_requests_by_status(
    random_identifier: Callable[[], str],
    simulate_ml_multi_cloud_random_walk_operation: Callable[
        [int, int, int, str | None],
        tuple[SQLSampleStore, list[MeasurementRequest], list[str]],
    ],
) -> None:
    """Test filtering measurement requests by status field."""
    operation_id = random_identifier()
    sample_store, _requests, _ = simulate_ml_multi_cloud_random_walk_operation(
        number_entities=3,
        number_requests=5,
        measurements_per_result=2,
        operation_id=operation_id,
    )

    # All requests should have status "Success" in the simulation
    filters = [{"status": json.dumps("Success")}]

    filtered_requests = sample_store.measurement_requests_for_operation(
        operation_id=operation_id, filters=filters
    )

    # Should return all requests since they all have status "Success"
    assert len(filtered_requests) == 5
    for req in filtered_requests:
        assert req.status == "Success"


def test_filter_measurement_requests_by_request_index(
    random_identifier: Callable[[], str],
    simulate_ml_multi_cloud_random_walk_operation: Callable[
        [int, int, int, str | None],
        tuple[SQLSampleStore, list[MeasurementRequest], list[str]],
    ],
) -> None:
    """Test filtering measurement requests by requestIndex field."""
    operation_id = random_identifier()
    sample_store, _requests, _ = simulate_ml_multi_cloud_random_walk_operation(
        number_entities=3,
        number_requests=5,
        measurements_per_result=2,
        operation_id=operation_id,
    )

    # Get all requests to find what indices exist
    all_requests = sample_store.measurement_requests_for_operation(
        operation_id=operation_id
    )
    assert len(all_requests) == 5

    # Pick the requestIndex from the second request
    target_index = all_requests[1].requestIndex

    # Count how many requests have this index
    expected_count = sum(1 for req in all_requests if req.requestIndex == target_index)

    # Filter for requests with this specific requestIndex
    filters = [{"requestIndex": json.dumps(target_index)}]

    filtered_requests = sample_store.measurement_requests_for_operation(
        operation_id=operation_id, filters=filters
    )

    # Should return exactly the requests with this index
    assert len(filtered_requests) == expected_count
    for req in filtered_requests:
        assert req.requestIndex == target_index


def test_filter_measurement_results_by_entity_identifier(
    random_identifier: Callable[[], str],
    simulate_ml_multi_cloud_random_walk_operation: Callable[
        [int, int, int, str | None],
        tuple[SQLSampleStore, list[MeasurementRequest], list[str]],
    ],
) -> None:
    """Test filtering measurement results by entityIdentifier field."""
    operation_id = random_identifier()
    sample_store, _requests, _ = simulate_ml_multi_cloud_random_walk_operation(
        number_entities=3,
        number_requests=2,
        measurements_per_result=2,
        operation_id=operation_id,
    )

    # Get all results first to find an entity identifier
    all_results = sample_store.measurement_results_for_operation(
        operation_id=operation_id
    )
    # 3 entities * 2 requests = 6 total results
    # (measurements_per_result affects measurements within each result, not result count)
    assert len(all_results) == 6

    # Pick the first entity identifier
    target_entity_id = all_results[0].entityIdentifier

    # Count how many results have this entity identifier
    expected_count = sum(
        1 for result in all_results if result.entityIdentifier == target_entity_id
    )
    # Verify we have at least one result for this entity
    assert expected_count >= 1

    # Filter for results with this specific entity identifier
    filters = [{"entityIdentifier": json.dumps(target_entity_id)}]

    filtered_results = sample_store.measurement_results_for_operation(
        operation_id=operation_id, filters=filters
    )

    # Should return exactly the results for this entity
    assert len(filtered_results) == expected_count
    for result in filtered_results:
        assert result.entityIdentifier == target_entity_id


def test_filter_with_multiple_conditions(
    random_identifier: Callable[[], str],
    simulate_ml_multi_cloud_random_walk_operation: Callable[
        [int, int, int, str | None],
        tuple[SQLSampleStore, list[MeasurementRequest], list[str]],
    ],
) -> None:
    """Test filtering with multiple filter conditions (AND logic)."""
    operation_id = random_identifier()
    sample_store, _requests, _ = simulate_ml_multi_cloud_random_walk_operation(
        number_entities=3,
        number_requests=5,
        measurements_per_result=2,
        operation_id=operation_id,
    )

    # Get all requests to find a specific index
    all_requests = sample_store.measurement_requests_for_operation(
        operation_id=operation_id
    )
    target_index = all_requests[0].requestIndex

    # Count how many requests match both conditions
    expected_count = sum(
        1
        for req in all_requests
        if req.status == "Success" and req.requestIndex == target_index
    )

    # Filter for requests with status="Success" AND specific requestIndex
    filters = [
        {"status": json.dumps("Success")},
        {"requestIndex": json.dumps(target_index)},
    ]

    filtered_requests = sample_store.measurement_requests_for_operation(
        operation_id=operation_id, filters=filters
    )

    # Should return exactly the requests matching both conditions
    assert len(filtered_requests) == expected_count
    for req in filtered_requests:
        assert req.status == "Success"
        assert req.requestIndex == target_index


def test_filter_returns_empty_when_no_matches(
    random_identifier: Callable[[], str],
    simulate_ml_multi_cloud_random_walk_operation: Callable[
        [int, int, int, str | None],
        tuple[SQLSampleStore, list[MeasurementRequest], list[str]],
    ],
) -> None:
    """Test that filtering returns empty list when no results match."""
    operation_id = random_identifier()
    sample_store, _requests, _ = simulate_ml_multi_cloud_random_walk_operation(
        number_entities=3,
        number_requests=5,
        measurements_per_result=2,
        operation_id=operation_id,
    )

    # Filter for a status that doesn't exist
    filters = [{"status": json.dumps("NonExistentStatus")}]

    filtered_requests = sample_store.measurement_requests_for_operation(
        operation_id=operation_id, filters=filters
    )

    # Should return empty list
    assert len(filtered_requests) == 0


def test_filter_with_dollar_prefix(
    random_identifier: Callable[[], str],
    simulate_ml_multi_cloud_random_walk_operation: Callable[
        [int, int, int, str | None],
        tuple[SQLSampleStore, list[MeasurementRequest], list[str]],
    ],
) -> None:
    """Test that filters work with $.prefix notation."""
    operation_id = random_identifier()
    sample_store, _requests, _ = simulate_ml_multi_cloud_random_walk_operation(
        number_entities=3,
        number_requests=5,
        measurements_per_result=2,
        operation_id=operation_id,
    )

    # Filter using $.status notation
    filters = [{"$.status": json.dumps("Success")}]

    filtered_requests = sample_store.measurement_requests_for_operation(
        operation_id=operation_id, filters=filters
    )

    # Should work the same as without the prefix
    assert len(filtered_requests) == 5
    for req in filtered_requests:
        assert req.status == "Success"


def test_filter_with_invalid_path_raises_error(
    random_identifier: Callable[[], str],
    simulate_ml_multi_cloud_random_walk_operation: Callable[
        [int, int, int, str | None],
        tuple[SQLSampleStore, list[MeasurementRequest], list[str]],
    ],
) -> None:
    """Test that filtering with an invalid path raises ValueError."""
    import pytest

    operation_id = random_identifier()
    sample_store, _requests, _ = simulate_ml_multi_cloud_random_walk_operation(
        number_entities=3,
        number_requests=5,
        measurements_per_result=2,
        operation_id=operation_id,
    )

    # Try to filter on an invalid field (not a known column, not metadata.*)
    filters = [{"invalid_field": json.dumps("some_value")}]

    with pytest.raises(ValueError, match="Invalid filter path 'invalid_field'"):
        sample_store.measurement_requests_for_operation(
            operation_id=operation_id, filters=filters
        )


# Made with Bob
