# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""PKH herding scores, refitting, and measurement feedback."""

from collections.abc import Callable

import numpy as np
import pytest
import ray
from active_learning.pkh.parameters import PKHParameters
from active_learning.pkh.sampler import PKHSampleSelector, PKHState

from ado.core.discoveryspace.space import DiscoverySpace
from ado.modules.actuators.measurement_queue import MeasurementQueue
from ado.modules.operators.discovery_space_manager import DiscoverySpaceManager
from ado.schema.entity import Entity
from ado.schema.observed_property import ObservedProperty


def test_herding_and_mmd_match_forest_proximity_kernel() -> None:
    """Fill the pilot shortfall and check herding and MMD across refits."""
    features = np.random.default_rng(7).normal(size=(12, 3))
    labels = features.sum(axis=1)
    initial = [0, 3, 7]
    state = PKHState(
        features,
        initial_measured_indices=initial,
        initial_labels=labels[initial].tolist(),
        params=PKHParameters(
            targetOutput="response",
            numberEntities=5,
            pilotSize=6,
            nEstimators=8,
            epochLength=2,
            seed=2**32 - 1,
        ),
    )
    # A measurement arriving before the first selection counts toward the pilot.
    state.record_measurement(9, labels[9])
    for _ in range(2):
        chosen = state.select_next()
        assert chosen is not None
        assert chosen not in state.measured_set
        assert state.forest is None
        state.record_measurement(chosen, labels[chosen])

    forests = []
    for _ in range(3):
        chosen = state.select_next()
        forest = state.forest
        assert forest is not None
        forests.append(forest)
        leaves = forest.apply(features)
        kernel = (leaves[:, None, :] == leaves[None, :, :]).mean(axis=2)
        measured = state.measured_indices
        scores = kernel.mean(axis=1) - kernel[:, measured].mean(axis=1)
        scores[measured] = -np.inf
        assert chosen is not None
        assert scores[chosen] == pytest.approx(scores.max())
        sample_kernel = kernel[np.ix_(measured, measured)]
        mmd = kernel.mean() - 2 * kernel[measured].mean() + sample_kernel.mean()
        assert state.mmd_squared() == pytest.approx(mmd)
        state.record_measurement(chosen, labels[chosen])
    assert forests[0] is forests[1]
    assert forests[1] is not forests[2]


def test_one_label_pilot_draws_a_second_candidate_before_herding() -> None:
    """Use a seeded draw while the sampling forest still needs a second label."""
    state = PKHState(
        np.arange(9).reshape(-1, 1),
        initial_measured_indices=[],
        initial_labels=[],
        params=PKHParameters(
            targetOutput="response", numberEntities=3, pilotSize=1, seed=42
        ),
    )
    first = state.select_next()
    assert first is not None
    state.record_measurement(first, 1.0)
    second = state.select_next()
    assert second in range(9)
    assert second != first
    assert second != min(set(range(9)) - {first})


@pytest.mark.asyncio
async def test_sampler_requires_current_target_before_another_selection(
    categorical_space: DiscoverySpace,
    record_pkh_measurement: Callable[[Entity, str, ObservedProperty, int], None],
) -> None:
    """Require the selected point's target from the active walk."""
    queue = MeasurementQueue()
    manager = DiscoverySpaceManager.remote(space=categorical_space, queue=queue)
    try:
        sampler = PKHSampleSelector(
            PKHParameters(targetOutput="self", numberEntities=2)
        )
        stream = sampler._entity_stream(manager, "pkh-walk")
        [first] = await anext(stream)
        assert (
            first.identifier
            == categorical_space.entitySpace.entity_for_point({"code": None}).identifier
        )
        target, _, sibling = categorical_space.measurementSpace.observedProperties
        record_pkh_measurement(first, "pkh-walk", sibling, 17)
        record_pkh_measurement(first, "another-walk", target, 999)
        with pytest.raises(ValueError, match="Missing finite target measurement"):
            await anext(stream)
    finally:
        ray.kill(manager)
        queue.shutdown(force=True)
