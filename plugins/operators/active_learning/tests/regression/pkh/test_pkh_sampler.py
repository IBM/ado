# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT
# ruff: noqa: S101

"""Tests for PKH's own leaf-distribution-deficit acquisition logic.

Tests shared across both strategies (parameter round trips, the finite-pool
and iterator machinery, ...) live in ``tests/regression/test_samplers.py``.
"""

import numpy as np
from active_learning.regression.pkh.sampler import _PKHSelectionState


def test_pkh_uses_the_hand_computed_leaf_distribution_deficit() -> None:
    """Verify PKH maximizes exact leaf deficit with stable tie-breaking."""

    features = np.arange(5, dtype=float).reshape(-1, 1)
    state = _PKHSelectionState(
        features,
        np.array([0, 1]),
        np.array([0.0, 1.0]),
        n_estimators=2,
        min_samples_leaf=1,
        epoch_length=20,
        seed=7,
        n_jobs=1,
    )
    leaf_signature = np.array([[0, 0], [0, 1], [0, 0], [1, 0], [1, 0]])
    pool_leaf_mass = np.array([[0.6, 0.4], [0.8, 0.2]])
    selected_leaf_count = np.array([[2, 0], [1, 1]])
    candidates = np.array([4, 2, 3])
    tree_indices = np.arange(2)
    selected_count = len(state.selected_indices)
    expected_scores = np.array(
        [
            np.mean(
                pool_leaf_mass[tree_indices, leaf_signature[index]]
                - selected_leaf_count[tree_indices, leaf_signature[index]]
                / selected_count
            )
            for index in candidates
        ]
    )

    np.testing.assert_allclose(expected_scores, [0.35, -0.05, 0.35])
    state.leaf_signature_ = leaf_signature
    state.pool_leaf_mass_ = pool_leaf_mass
    state.selected_leaf_count_ = selected_leaf_count
    state._forest_fitted = True

    assert state.query_one(candidates) == 3
    assert state.pending_index == 3
    assert state.selected_indices == (0, 1)
