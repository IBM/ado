# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT
# ruff: noqa: S101

"""Tests for PKH's own leaf-distribution-deficit acquisition logic.

Tests shared across both strategies (parameter round trips, the finite-pool
and iterator machinery, ...) live in ``tests/regression/test_samplers.py``.
"""

import numpy as np
from active_learning.regression.pkh.sampler import _PKHSelectionState


def _hand_computed_state(seed: int = 7) -> _PKHSelectionState:
    """Create a PKH state whose candidate scores include two exact maximizers."""

    features = np.arange(5, dtype=float).reshape(-1, 1)
    state = _PKHSelectionState(
        features,
        np.array([0, 1]),
        np.array([0.0, 1.0]),
        n_estimators=2,
        min_samples_leaf=1,
        pilot_size=1,
        epoch_length=20,
        seed=seed,
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
    return state


def test_pkh_uses_the_hand_computed_leaf_distribution_deficit() -> None:
    """Verify exact-score ties can select either maximizer but never a lower score."""

    state = _hand_computed_state()
    candidates = np.array([4, 2, 3])
    choices = [state._choose_candidate(candidates) for _ in range(64)]
    assert set(choices) == {3, 4}
    assert state.selected_indices == (0, 1)


def test_pkh_tie_choices_replay_the_seed() -> None:
    """The same seed reproduces tied choices and another seed changes them."""

    states = [_hand_computed_state(seed) for seed in [7, 7, 19]]
    choices = [
        [state._choose_candidate(np.array([4, 2, 3])) for _ in range(32)]
        for state in states
    ]
    assert choices[0] == choices[1]
    assert choices[0] != choices[2]


def test_pkh_unique_winners_do_not_advance_tie_rng() -> None:
    """Selecting an unambiguous maximizer leaves the following tie sequence intact."""

    with_unique_winners = _hand_computed_state()
    reference = _hand_computed_state()
    for _ in range(8):
        assert with_unique_winners._choose_candidate(np.array([2, 4])) == 4
    actual = [
        with_unique_winners._choose_candidate(np.array([3, 4])) for _ in range(32)
    ]
    expected = [reference._choose_candidate(np.array([3, 4])) for _ in range(32)]
    assert actual == expected


def test_pkh_near_equal_scores_keep_the_unique_maximizer() -> None:
    """A one-step floating-point score difference is not an exact tie."""

    state = _hand_computed_state()
    state.leaf_signature_[4, 0] = 2
    state.pool_leaf_mass_ = np.column_stack(
        [state.pool_leaf_mass_, state.pool_leaf_mass_[:, 1]]
    )
    state.selected_leaf_count_ = np.column_stack(
        [state.selected_leaf_count_, state.selected_leaf_count_[:, 1]]
    )
    state.pool_leaf_mass_[0, 2] = np.nextafter(0.4, np.inf)
    assert all(state._choose_candidate(np.array([3, 4])) == 4 for _ in range(32))


def test_pkh_candidate_permutation_preserves_seeded_query_choice() -> None:
    """The public query path canonicalizes candidate ordering before a tie draw."""

    first = _hand_computed_state()
    second = _hand_computed_state()
    assert first.query_one(np.array([4, 2, 3])) == second.query_one(np.array([3, 4, 2]))


def test_pkh_ties_preserve_pilot_sequence_and_fixed_forest_seed() -> None:
    """Pilot draws and repeated forest fits remain independent of tie draws."""

    features = np.arange(100, dtype=float).reshape(-1, 1)
    state = _PKHSelectionState(
        features,
        np.array([], dtype=int),
        np.array([], dtype=float),
        pilot_size=5,
        n_estimators=2,
        min_samples_leaf=2,
        epoch_length=1,
        seed=42,
        n_jobs=1,
    )
    pilot = []
    for _ in range(5):
        chosen = state.query_one()
        pilot.append(chosen)
        state.update(chosen, 1.0)
    assert pilot == [75, 43, 64, 8, 99]

    reference = _PKHSelectionState(
        features,
        np.array(pilot),
        np.ones(5),
        pilot_size=5,
        n_estimators=2,
        min_samples_leaf=2,
        epoch_length=1,
        seed=42,
        n_jobs=1,
    )
    for _ in range(6):
        chosen = state.query_one()
        assert reference.query_one() == chosen
        assert state.forest_.random_state == reference.forest_.random_state == 42
        state.update(chosen, 1.0)
        reference.update(chosen, 1.0)


def test_pkh_tie_stream_is_independent_of_pilot_rng_consumption() -> None:
    """Unrelated pilot draws leave tied acquisition decisions unchanged."""

    advanced = _hand_computed_state()
    reference = _hand_computed_state()
    advanced._pilot_rng.random(100)
    actual = [advanced._choose_candidate(np.array([3, 4])) for _ in range(32)]
    expected = [reference._choose_candidate(np.array([3, 4])) for _ in range(32)]
    assert actual == expected


def test_pkh_query_records_the_selected_tied_candidate() -> None:
    """The sequential state records whichever tied candidate the seed selects."""

    state = _hand_computed_state()
    chosen = state.query_one(np.array([4, 2, 3]))
    assert chosen in {3, 4}
    assert state.pending_index == chosen
    assert state.selected_indices == (0, 1)
