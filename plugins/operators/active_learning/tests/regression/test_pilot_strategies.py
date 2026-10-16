# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT
# ruff: noqa: S101

"""Pilot selection over finite pools with mixed original dimensions."""

import itertools

import numpy as np
import pandas as pd
import pytest
from active_learning.regression._shared import _encode_features
from active_learning.regression.pkh.sampler import _PKHSelectionState
from scipy.stats import qmc


def _state(
    raw: np.ndarray,
    strategy: str,
    *,
    pilot_size: int,
    initial: tuple[int, ...] = (),
) -> _PKHSelectionState:
    """Build a real selector with numeric model features and original dimensions."""

    return _PKHSelectionState(
        _encode_features(pd.DataFrame(raw)),
        np.asarray(initial, dtype=int),
        np.asarray(initial, dtype=float),
        pilot_strategy=strategy,
        pilot_features=raw,
        pilot_size=pilot_size,
        seed=42,
        n_estimators=2,
        min_samples_leaf=1,
        n_jobs=1,
    )


def _draw(state: _PKHSelectionState, count: int) -> list[int]:
    """Reveal each selected entity's scalar label before the following query."""

    selected = []
    for _ in range(count):
        index = state.query_one()
        state.update(index, float(index))
        selected.append(index)
    return selected


def test_random_pilot_keeps_existing_seeded_sequence() -> None:
    """The historical random pilot retains its exact five-point sequence."""

    raw = np.asarray(list(itertools.product(range(10), repeat=2)))
    state = _state(raw, "random", pilot_size=5)
    assert _draw(state, 5) == [75, 43, 64, 8, 99]
    assert not state._forest_fitted


def test_clhs_matches_dimension_cycles_for_numeric_and_categorical_features() -> None:
    """CLHS consumes each original dimension's levels before repeating a cycle."""

    raw = np.asarray(
        list(itertools.product([2.0, 10.0, 25.0, 100.0], ["red", "blue", "green"])),
        dtype=object,
    )
    state = _state(raw, "clhs", pilot_size=6)
    order = _draw(state, 6)
    assert order == [0, 11, 4, 8, 10, 3]


def test_sobol_matches_scrambled_coordinates_on_original_dimensions() -> None:
    """Sobol coordinates map to valid finite-grid rows before feature encoding."""

    raw = np.asarray(
        list(itertools.product(range(8), ["a", "b", "c", "d"])), dtype=object
    )
    coordinates = qmc.Sobol(d=2, scramble=True, rng=42).random_base2(3)
    expected = (np.floor(coordinates * [8, 4]).astype(int) @ [4, 1]).tolist()
    assert len(set(expected)) == 8
    state = _state(raw, "sobol", pilot_size=8)
    assert _draw(state, 8) == expected


def test_clhs_skips_repeated_coordinate_and_continues_dimension_cycles() -> None:
    """An ideal-gas-sized grid gets eighteen distinct pilot measurements."""

    raw = np.asarray(list(itertools.product(range(15), range(9), range(9))))
    state = _state(raw, "clhs", pilot_size=18)
    assert _draw(state, 18) == [
        819,
        1021,
        268,
        161,
        897,
        676,
        200,
        59,
        471,
        348,
        776,
        642,
        491,
        1093,
        1151,
        151,
        945,
        31,
    ]


@pytest.mark.parametrize("strategy", ["random", "clhs", "sobol"])
def test_existing_labels_count_toward_pilot(strategy: str) -> None:
    """Existing labels are excluded and only missing pilot labels are acquired."""

    raw = np.asarray(list(itertools.product(range(4), repeat=2)))
    state = _state(raw, strategy, pilot_size=5, initial=(0, 1, 2))
    chosen = _draw(state, 2)
    assert len(set(chosen)) == 2
    assert not set(chosen).intersection({0, 1, 2})
    assert len(state.selected_indices) == 5
    assert not state._forest_fitted
    state.query_one()
    assert state._forest_fitted


@pytest.mark.parametrize("strategy", ["random", "clhs", "sobol"])
def test_pilot_larger_than_small_pool_exhausts_unique_entities(strategy: str) -> None:
    """Small pools terminate without repeating measured rows or fitting early."""

    raw = np.asarray([[0, "same"], [1, "same"], [2, "same"]], dtype=object)
    state = _state(raw, strategy, pilot_size=20)
    assert set(_draw(state, 3)) == {0, 1, 2}
    assert not state.has_candidates
    assert not state._forest_fitted


@pytest.mark.parametrize("strategy", ["clhs", "sobol"])
def test_sparse_pool_and_changing_candidates_remain_unique(strategy: str) -> None:
    """Restricted candidates and incomplete grids always yield eligible rows."""

    raw = np.asarray([[index, index] for index in range(50)])
    state = _state(raw, strategy, pilot_size=10, initial=(0, 1))
    first = state.query_one(np.array([2, 4, 6, 8]))
    assert first in {2, 4, 6, 8}
    state.update(first, float(first))
    second = state.query_one(np.array([49]))
    assert second == 49
    state.update(second, float(second))
    third = state.query_one(np.array([3, 5, 7]))
    assert third in {3, 5, 7}
    assert len({first, second, third}) == 3


@pytest.mark.parametrize("strategy", ["clhs", "sobol"])
def test_structured_pilot_is_reproducible(strategy: str) -> None:
    """Identical seeds reproduce all pilot selections, including collision handling."""

    raw = np.asarray(list(itertools.product(range(3), repeat=2)))
    first = _draw(_state(raw, strategy, pilot_size=9), 9)
    second = _draw(_state(raw, strategy, pilot_size=9), 9)
    assert first == second
    assert set(first) == set(range(9))


def test_invalid_pilot_strategy_is_rejected_by_state() -> None:
    """Direct state construction rejects an unsupported acquisition policy."""

    with pytest.raises(ValueError, match="pilot_strategy"):
        _state(np.arange(4).reshape(-1, 1), "invalid", pilot_size=2)


def test_pilot_features_must_align_with_encoded_pool() -> None:
    """Raw coordinates must describe every model-feature row."""

    with pytest.raises(ValueError, match="pilot_features"):
        _PKHSelectionState(
            np.arange(4).reshape(-1, 1),
            np.array([], dtype=int),
            np.array([], dtype=float),
            pilot_strategy="clhs",
            pilot_features=np.arange(3).reshape(-1, 1),
        )
