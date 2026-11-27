# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""FLORA's leaf gain, tie-breaking and refit schedule."""

import numpy as np
import pytest
from active_learning.flora.parameters import FLORAParameters
from active_learning.flora.sampler import FLORASelectionState


def test_selection_maximizes_risk_reduction_across_refits() -> None:
    """Check greedy choices, seeded ties and refits as measurements arrive."""
    features = np.random.default_rng(7).normal(size=(30, 3))
    labels = features.sum(axis=1)
    initial_indices = [0, 3, 7, 9, 12]
    state = FLORASelectionState(
        features,
        initial_measured_indices=initial_indices,
        initial_labels=labels[initial_indices].tolist(),
        params=FLORAParameters(
            targetOutput="response", numberEntities=8, nEstimators=8, seed=42
        ),
    )
    refit_counts = []
    forest = None
    for _ in range(8):
        selected = state.select_next()
        assert selected is not None
        if state.forest is not forest:
            refit_counts.append(len(state.measured_indices))
            forest = state.forest
        assert forest is not None
        prediction_variance = np.var(
            [tree.predict(features) for tree in forest.estimators_], axis=0
        )
        scores = np.zeros(len(features))
        for leaves in forest.apply(features).T:
            for leaf in np.unique(leaves):
                members = leaves == leaf
                measured_count = members[state.measured_indices].sum()
                leaf_size = members.sum()
                entities_per_label = leaf_size / measured_count
                after_one_more_label = leaf_size / (measured_count + 1)
                weight = prediction_variance[members].mean() / len(features)
                scores[members] += weight * (
                    entities_per_label - after_one_more_label + 1
                )
        scores[state.measured_indices] = -np.inf
        assert scores[selected] == pytest.approx(scores.max())
        state.record_measurement(selected, labels[selected])
    assert refit_counts == [5, 8, 12]
    # The first two active choices each have three equally scoring candidates.
    active_choices = state.measured_indices[len(initial_indices) :]
    assert active_choices[:2] == [14, 29]
