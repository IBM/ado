# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Tests for SampleStoreStatistics and SQLSampleStore.samplestore_statistics."""

from collections.abc import Callable

from ado.core.samplestore.sql import SQLSampleStore
from ado.schema.request import MeasurementRequest

# ---------------------------------------------------------------------------
# SQLSampleStore.samplestore_statistics — integration
# ---------------------------------------------------------------------------


def test_all_counters_zero_on_empty_store(
    empty_sample_store: SQLSampleStore,
) -> None:
    """All three counters are 0 for a fresh, empty store."""
    stats = empty_sample_store.samplestore_statistics()
    assert stats.number_of_entities == 0
    assert stats.number_of_results == 0
    assert stats.number_of_experiments == 0


def test_samplestore_statistics_reflect_simulation(
    simulate_ml_multi_cloud_random_walk_operation: Callable[
        [int, int, int, str | None],
        tuple[SQLSampleStore, list[MeasurementRequest], list[str]],
    ],
    ml_multi_cloud_sample_store: SQLSampleStore,
    random_identifier: Callable[[], str],
) -> None:
    """After a simulation the result count increases by number_entities * number_requests.

    The ml_multi_cloud store is pre-seeded from the CSV fixture (entities and
    results are already present, but no experiment references).  We snapshot
    the store before the simulation to capture that baseline, then assert:
    - the result delta equals number_entities * number_requests
    - number_of_experiments grows by exactly 1, because all requests share
      the same experiment_reference (deduplication).
    """
    number_entities = 3
    number_requests = 4

    before = ml_multi_cloud_sample_store.samplestore_statistics()

    sample_store, requests, _ = simulate_ml_multi_cloud_random_walk_operation(
        number_entities=number_entities,
        number_requests=number_requests,
        measurements_per_result=1,
        operation_id=random_identifier(),
    )
    after = sample_store.samplestore_statistics()

    assert (
        after.number_of_results - before.number_of_results
        == number_entities * number_requests
    )

    # All requests share the same experiment_reference, so number_of_experiments
    # grows by exactly 1 regardless of how many requests were made
    experiment_refs = {str(r.experimentReference) for r in requests}
    assert len(experiment_refs) == 1
    assert after.number_of_experiments == before.number_of_experiments + 1
