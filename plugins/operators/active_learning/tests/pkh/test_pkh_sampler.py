# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""PKH selection, histogram updates, and refitting against explicit leaf embeddings."""

import numpy as np
import pytest
from active_learning.pkh.parameters import PKHParameters
from active_learning.pkh.sampler import PKHState


def test_herding_and_mmd_match_explicit_embeddings() -> None:
    """Complete the pilot from prior labels, then check herding and MMD across refits."""
    features = np.random.default_rng(7).normal(size=(12, 3))
    initial_indices = [0, 3, 7, 9]
    state = PKHState(
        features,
        initial_measured_indices=initial_indices,
        initial_labels=[float(features[index].sum()) for index in initial_indices],
        params=PKHParameters(
            targetOutput="response",
            numberEntities=4,
            pilotSize=5,
            nEstimators=8,
            epochLength=2,
            seed=2**32 - 1,
        ),
    )
    pilot = state.select_next()
    assert pilot is not None
    assert pilot not in initial_indices
    assert state.forest is None
    assert state.mmd_squared() is None
    state.record_measurement(pilot, float(features[pilot].sum()))

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
