# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import numpy as np
import pytest
from trim.samplers.no_priors_utils import (
    _select_least_dense_index,
    get_index_list_van_der_corput,
)

# --- Error Handling Tests ---


def test_get_index_list_nn_raises_value_error_on_invalid_sampling() -> None:
    """Should raise ValueError when trying to sample more points than available."""
    with pytest.raises(ValueError, match="ValueError"):
        get_index_list_van_der_corput(0, 1)


# --- Functional Tests for get_index_list_nn ---


def test_get_index_list_nn_full_sampling() -> None:
    """Should return correct full sampling order for segment of length 9."""
    expected = [0, 8, 4, 2, 6, 3, 7, 1, 5]
    assert get_index_list_van_der_corput(9, 9) == expected


@pytest.mark.parametrize(
    ("points", "expected"),
    [
        (7, [0, 4, 6, 8, 10, 12, 16]),
        (8, [0, 2, 4, 6, 8, 10, 12, 16]),
        (9, [0, 2, 4, 6, 8, 10, 12, 14, 16]),
    ],
)
def test_get_index_list_nn_sorted_sampling(points: int, expected: list[int]) -> None:
    """Should return sorted sampling for segment of length 17."""
    assert get_index_list_van_der_corput(17, points, sort=True) == expected


@pytest.mark.parametrize(
    ("length", "points", "sampled_indices", "expected"),
    [
        (5, 3, [0, 4], [0, 4, 2]),
        (10, 4, [0, 4, 9], [0, 4, 9, 6]),
        (14, 8, None, [0, 13, 6, 9, 3, 11, 5, 2]),
        (12, 12, [3, 7], [3, 7, 0, 11, 5, 9, 2, 8, 4, 10, 1, 6]),
    ],
)
def test_get_index_list_van_der_corput_with_sampled_indices(
    length: int,
    points: int,
    sampled_indices: list[int] | None,
    expected: list[int],
) -> None:
    """Should extend already-sampled indices in least-dense-neighbourhood order."""
    assert (
        get_index_list_van_der_corput(length, points, sampled_indices=sampled_indices)
        == expected
    )


@pytest.mark.parametrize("length", [1, 2, 3, 16, 33, 100])
def test_get_index_list_van_der_corput_full_sampling_is_permutation(
    length: int,
) -> None:
    """Should return each index of the segment exactly once, extremities first."""
    order = get_index_list_van_der_corput(length, length)
    assert sorted(order) == list(range(length))
    assert order[: min(2, length)] == [0, length - 1][: min(2, length)]


@pytest.mark.parametrize(
    ("occupancy", "length_segment", "expected"),
    [
        ([1, 0, 0, 0, 1], 5, 2),
        ([1, 0, 0, 0, 0, 0, 0, 0, 1], 9, 4),
        ([1, 1, 0, 0, 1], 5, 3),
        ([1, 0, 0, 1, 0, 0, 0, 0, 1], 9, 5),
    ],
)
def test_select_least_dense_index(
    occupancy: list[int], length_segment: int, expected: int
) -> None:
    """Should return the unsampled index in the least densely sampled window."""
    assert (
        _select_least_dense_index(
            occupancy=np.array(occupancy),
            length_segment=length_segment,
            max_half_width=length_segment,
        )
        == expected
    )


def test_select_least_dense_index_ignores_positions_beyond_segment() -> None:
    """Should select only within the segment when occupancy extends beyond it."""
    occupancy = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 1])

    selected_index = _select_least_dense_index(
        occupancy=occupancy, length_segment=4, max_half_width=4
    )

    assert selected_index == 3


def test_get_index_list_van_der_corput_with_out_of_segment_sampled_index() -> None:
    """Should count an out-of-segment sampled index and select only in-segment indices."""
    assert get_index_list_van_der_corput(5, 5, sampled_indices=[7]) == [7, 0, 4, 2, 3]
