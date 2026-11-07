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


@pytest.mark.parametrize(
    ("seed", "expected_forest_seeds", "expected_tie_prefix"),
    [
        (
            42,
            [1319930014, 250891444, 1679823183, 1737235761, 1771475718],
            [15, 36, 10, 6, 12, 28, 38, 17, 23, 22, 24, 32],
        ),
        (
            17,
            [357838922, 207341212, 1822441986, 313021393, 1581564404],
            [14, 23, 28, 9, 32, 10, 6, 22, 20, 13, 36, 15],
        ),
        (
            2**32 + 42,
            [1319930014, 250891444, 1679823183, 1737235761, 1771475718],
            [15, 36, 10, 6, 12, 28, 38, 17, 23, 22, 24, 32],
        ),
    ],
    ids=["seed-42", "seed-17", "32-bit-seed-domain"],
)
def test_flora_replays_legacy_forest_and_tie_random_streams(
    seed: int,
    expected_forest_seeds: list[int],
    expected_tie_prefix: list[int],
) -> None:
    """Replay legacy FORE/TIES vectors across actual scheduled forest refits."""

    features = np.arange(40, dtype=float).reshape(-1, 1)
    initial_indices = np.arange(5)
    trajectories = []
    for _ in range(2):
        state = _FLORASelectionState(
            features,
            initial_indices,
            np.ones(5),
            n_estimators=5,
            min_samples_leaf=1,
            seed=seed,
            n_jobs=1,
        )
        choices = []
        forests = []
        forest_seeds = []
        refit_sizes = []
        for _ in range(20):
            index = state.query_one()
            if not forests or state.forest_ is not forests[-1]:
                forests.append(state.forest_)
                forest_seeds.append(state.forest_.random_state)
                refit_sizes.append(len(state.selected_indices))
            assert np.all(state.point_disagreement_ == 0.0)
            choices.append(index)
            state.update(index, 1.0)

        assert refit_sizes == [5, 8, 12, 17, 23]
        assert forest_seeds == expected_forest_seeds
        assert choices[:12] == expected_tie_prefix
        trajectories.append(choices)

    assert trajectories[0] == trajectories[1]


def test_flora_extra_tie_draws_do_not_change_forest_seeds() -> None:
    """Identical training labels produce matching seeds despite different ties."""

    features = np.arange(40, dtype=float).reshape(-1, 1)
    initial_indices = np.arange(5)
    tied = _FLORASelectionState(
        features,
        initial_indices,
        np.ones(5),
        n_estimators=5,
        min_samples_leaf=1,
        seed=42,
        n_jobs=1,
    )
    singleton = _FLORASelectionState(
        features,
        initial_indices,
        np.ones(5),
        n_estimators=5,
        min_samples_leaf=1,
        seed=42,
        n_jobs=1,
    )
    forest_seeds = []
    for _ in range(15):
        index = tied.query_one()
        assert singleton.query_one(np.array([index])) == index
        assert tied.forest_.random_state == singleton.forest_.random_state
        if not forest_seeds or tied.forest_.random_state != forest_seeds[-1]:
            forest_seeds.append(tied.forest_.random_state)
        tied.update(index, 1.0)
        singleton.update(index, 1.0)
        assert tied.selected_indices == singleton.selected_indices

    assert len(forest_seeds) == 4


def test_flora_extra_forest_refits_do_not_change_tie_choices() -> None:
    """Additional fits leave tie choices unchanged for constant-label forests."""

    features = np.arange(40, dtype=float).reshape(-1, 1)
    initial_indices = np.arange(5)
    scheduled = _FLORASelectionState(
        features,
        initial_indices,
        np.ones(5),
        n_estimators=5,
        min_samples_leaf=1,
        seed=42,
        n_jobs=1,
    )
    frequent = _FLORASelectionState(
        features,
        initial_indices,
        np.ones(5),
        n_estimators=5,
        min_samples_leaf=1,
        seed=42,
        n_jobs=1,
    )
    seeds_differ = False
    for step in range(12):
        if step > 0:
            frequent._fit_epoch()
        index = scheduled.query_one()
        assert frequent.query_one() == index
        seeds_differ |= scheduled.forest_.random_state != frequent.forest_.random_state
        scheduled.update(index, 1.0)
        frequent.update(index, 1.0)

    assert seeds_differ
