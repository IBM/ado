# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""PKH selection, histogram updates, and refitting against explicit leaf embeddings."""

import numpy as np
import pytest
from active_learning.pkh.parameters import PKHParameters
from active_learning.pkh.sampler import PKHState


@pytest.mark.parametrize("initial_indices", [[], [0, 1, 2, 3]])
def test_pilot_completes_with_priors(initial_indices: list[int]) -> None:
    """Prior labels count toward a distinct pilot before the first forest fit."""
    state = PKHState(
        np.arange(10).reshape(-1, 1),
        initial_measured_indices=initial_indices,
        initial_labels=[float(index) for index in initial_indices],
        params=PKHParameters(
            targetOutput="response",
            numberEntities=10 - len(initial_indices),
            pilotSize=5,
            nEstimators=3,
        ),
    )
    assert state.mmd_squared() is None
    while len(state.measured_indices) < 5:
        selected = state.select_next()
        assert selected is not None
        assert selected not in state.measured_set
        assert state.forest is None
        state.record_measurement(selected, float(selected))
    assert state.select_next() not in state.measured_set
    assert state.forest is not None


def test_herding_and_mmd_match_explicit_embeddings() -> None:
    """Selections and MMD match explicit embeddings through updates and refitting."""
    features = np.random.default_rng(7).normal(size=(12, 3))
    initial_indices = [0, 3, 7, 9]
    state = PKHState(
        features,
        initial_measured_indices=initial_indices,
        initial_labels=[float(features[index].sum()) for index in initial_indices],
        params=PKHParameters(
            targetOutput="response",
            numberEntities=3,
            pilotSize=1,
            nEstimators=8,
            epochLength=2,
            seed=2**32 - 1,
        ),
    )
    forests = []
    for _ in range(3):
        selected = state.select_next()
        forest = state.forest
        assert selected is not None
        assert forest is not None
        forests.append(forest)
        leaves = forest.apply(features)
        embeddings = np.eye(leaves.max() + 1)[leaves].reshape(len(features), -1)
        embeddings /= np.sqrt(state.params.nEstimators)
        discrepancy = embeddings.mean(axis=0) - embeddings[state.measured_indices].mean(
            axis=0
        )
        scores = embeddings @ discrepancy
        scores[state.measured_indices] = -np.inf
        assert scores[selected] == pytest.approx(scores.max())
        assert state.mmd_squared() == pytest.approx(discrepancy @ discrepancy)
        state.record_measurement(selected, float(features[selected].sum()))
        updated_discrepancy = embeddings.mean(axis=0) - embeddings[
            state.measured_indices
        ].mean(axis=0)
        assert state.mmd_squared() == pytest.approx(
            updated_discrepancy @ updated_discrepancy
        )
    assert forests[0] is forests[1]
    assert forests[1] is not forests[2]


def test_ties_are_reproducible_and_pool_is_exhausted() -> None:
    """Tied scores produce the same distinct sequence despite diagnostic reads."""
    states = [
        PKHState(
            np.zeros((6, 2)),
            initial_measured_indices=[0, 1],
            initial_labels=[1.0, 2.0],
            params=PKHParameters(
                targetOutput="response",
                numberEntities=4,
                pilotSize=1,
                nEstimators=3,
                seed=123,
            ),
        )
        for _ in range(2)
    ]
    for _ in range(4):
        states[0].mmd_squared()
        choices = [state.select_next() for state in states]
        assert choices[0] == choices[1]
        assert choices[0] is not None
        for state, selected in zip(states, choices, strict=True):
            assert selected not in state.measured_set
            state.record_measurement(selected, 1.0)
    assert all(state.select_next() is None for state in states)
