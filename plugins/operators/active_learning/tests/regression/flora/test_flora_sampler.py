# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT
# ruff: noqa: S101

"""Tests for FLORA's own pointwise-disagreement acquisition logic.

Tests shared across both strategies (parameter round trips, the finite-pool
and iterator machinery, ...) live in ``tests/regression/test_samplers.py``.
"""

import numpy as np
import pytest
from active_learning.regression.flora.sampler import (
    _flora_leaf_gain,
    _FLORASelectionState,
)


def test_flora_leaf_gain_is_the_exact_objective_difference() -> None:
    """Verify covered-leaf gain equals the objective's exact difference."""

    pool_count = 12
    selected_count = 3
    disagreement = 2.5
    pool_size = 40
    objective_before = (
        disagreement / pool_size * (pool_count / selected_count - selected_count)
    )
    objective_after = (
        disagreement
        / pool_size
        * (pool_count / (selected_count + 1) - (selected_count + 1))
    )

    gain = _flora_leaf_gain(
        pool_count=pool_count,
        selected_count=selected_count,
        disagreement=disagreement,
        pool_size=pool_size,
    )

    assert gain == pytest.approx(objective_before - objective_after)
    assert gain == pytest.approx(0.125)


def test_flora_averages_pointwise_tree_disagreement_within_each_leaf() -> None:
    """Verify pointwise population variance is averaged within RF leaves."""

    axis = np.linspace(-3.0, 3.0, 18)
    features = np.column_stack((axis, axis**2))
    labels = np.sin(axis) + 0.2 * axis
    initial_indices = np.array([0, 3, 6, 9, 12, 15, 17])
    state = _FLORASelectionState(
        features,
        initial_indices,
        labels[initial_indices],
        n_estimators=9,
        min_samples_leaf=1,
        seed=23,
        n_jobs=1,
    )

    state.query_one()

    tree_predictions = np.vstack(
        [tree.predict(features) for tree in state.forest_.estimators_]
    )
    expected_disagreement = np.var(tree_predictions, axis=0, ddof=0)
    np.testing.assert_allclose(state.point_disagreement_, expected_disagreement)
    for tree_index in range(state.leaf_signature_.shape[1]):
        for leaf_code in np.unique(state.leaf_signature_[:, tree_index]):
            members = state.leaf_signature_[:, tree_index] == leaf_code
            assert state.leaf_disagreement_[tree_index, leaf_code] == pytest.approx(
                np.mean(expected_disagreement[members])
            )
