# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""PKH herding scores, refitting, and measurement feedback."""

from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager

import numpy as np
import pytest
import ray
from active_learning.pkh.parameters import PKHParameters
from active_learning.pkh.sampler import PKHSampleSelector, PKHState

from ado.core.discoveryspace.space import DiscoverySpace
from ado.modules.actuators.measurement_queue import MeasurementQueue
from ado.modules.operators.discovery_space_manager import DiscoverySpaceManager
from ado.schema.domain import PropertyDomain, VariableTypeEnum
from ado.schema.entity import Entity
from ado.schema.observed_property import ObservedProperty
from ado.schema.property import ConstitutiveProperty


@contextmanager
def _discovery_space_manager(space: DiscoverySpace) -> Iterator[DiscoverySpaceManager]:
    """Start and clean up a Ray discovery space manager and its queue."""
    with ExitStack() as cleanup:
        queue = MeasurementQueue()
        cleanup.callback(queue.shutdown, force=True)
        manager = DiscoverySpaceManager.remote(space=space, queue=queue)
        cleanup.callback(ray.kill, manager)
        yield manager


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
    first, repeat, different_seed = [
        PKHState(
            np.arange(9).reshape(-1, 1),
            initial_measured_indices=[0],
            initial_labels=[1.0],
            params=PKHParameters(
                targetOutput="response", numberEntities=3, pilotSize=1, seed=seed
            ),
        ).select_next()
        for seed in (42, 42, 2)
    ]
    assert first in range(1, 9)
    assert repeat == first
    assert different_seed in range(1, 9)
    assert different_seed != first


@pytest.mark.parametrize(
    ("domain", "expected"),
    [
        pytest.param(PropertyDomain(values=[1, 2]), True, id="finite"),
        pytest.param(
            PropertyDomain(
                variableType=VariableTypeEnum.DISCRETE_VARIABLE_TYPE, interval=1
            ),
            False,
            id="unbounded",
        ),
    ],
)
def test_sampler_compatibility_with_remote_space(
    make_pkh_space: Callable[[list[ConstitutiveProperty]], DiscoverySpace],
    domain: PropertyDomain,
    expected: bool,
) -> None:
    """Accept finite grids and reject unbounded domains through the remote API."""
    space = make_pkh_space(
        [ConstitutiveProperty(identifier="temperature", propertyDomain=domain)]
    )
    with _discovery_space_manager(space) as manager:
        assert (
            PKHSampleSelector.samplerCompatibleWithDiscoverySpaceRemote(manager)
            is expected
        )


@pytest.mark.asyncio
async def test_live_target_changes_the_next_selection(
    make_pkh_space: Callable[[list[ConstitutiveProperty]], DiscoverySpace],
    record_pkh_measurement: Callable[
        [DiscoverySpace, Entity, str, ObservedProperty, int], None
    ],
) -> None:
    """Use the child walk's newly measured target when fitting the first forest."""
    space = make_pkh_space(
        [
            ConstitutiveProperty(
                identifier=name, propertyDomain=PropertyDomain(values=list(range(4)))
            )
            for name in ("x", "y")
        ]
    )
    target = space.measurementSpace.observedProperties[0]
    for (x, y), value in [((0, 0), 1), ((0, 3), 5), ((2, 3), 9)]:
        entity = space.entitySpace.entity_for_point({"x": x, "y": y})
        record_pkh_measurement(space, entity, "prior-walk", target, value)

    sampler = PKHSampleSelector(
        PKHParameters(
            targetOutput="self",
            numberEntities=2,
            pilotSize=4,
            seed=0,
            nEstimators=20,
            minSamplesLeaf=1,
            maxFeatures=1.0,
        )
    )
    with _discovery_space_manager(space) as manager:
        low_stream = sampler._entity_stream(manager, "low-walk")
        high_stream = sampler._entity_stream(manager, "high-walk")
        [low_pilot] = await anext(low_stream)
        [high_pilot] = await anext(high_stream)
        assert low_pilot.identifier == high_pilot.identifier
        record_pkh_measurement(space, low_pilot, "low-walk", target, 0)
        record_pkh_measurement(space, high_pilot, "high-walk", target, 2000)
        [low_choice] = await anext(low_stream)
        [high_choice] = await anext(high_stream)
        assert low_choice.identifier != high_choice.identifier


@pytest.mark.asyncio
async def test_sampler_requires_current_target_before_another_selection(
    categorical_space: DiscoverySpace,
    record_pkh_measurement: Callable[
        [DiscoverySpace, Entity, str, ObservedProperty, int], None
    ],
) -> None:
    """Require the selected point's target from the active walk."""
    with _discovery_space_manager(categorical_space) as manager:
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
        record_pkh_measurement(categorical_space, first, "pkh-walk", sibling, 17)
        record_pkh_measurement(categorical_space, first, "another-walk", target, 999)
        with pytest.raises(ValueError, match="Missing finite target measurement"):
            await anext(stream)
