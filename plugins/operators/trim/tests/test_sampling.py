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
        (10, 4, [0, 4, 9], [0, 4, 9, 6]),
        (12, 12, [3, 7], [3, 7, 0, 11, 5, 9, 2, 8, 4, 10, 1, 6]),
    ],
)
def test_get_index_list_van_der_corput_with_sampled_indices(
    length: int,
    points: int,
    sampled_indices: list[int],
    expected: list[int],
) -> None:
    """Should extend already-sampled indices in least-dense-neighbourhood order."""
    assert (
        get_index_list_van_der_corput(length, points, sampled_indices=sampled_indices)
        == expected
    )


@pytest.mark.parametrize("sampled_indices", [[7], [-1], [0, 5]])
def test_get_index_list_van_der_corput_rejects_out_of_segment_sampled_indices(
    sampled_indices: list[int],
) -> None:
    """Should raise ValueError for sampled indices outside the segment."""
    with pytest.raises(ValueError, match="outside the segment"):
        get_index_list_van_der_corput(5, 5, sampled_indices=sampled_indices)


@pytest.mark.parametrize(
    ("occupancy", "expected"),
    [
        ([1, 1, 0, 0, 1], 3),
        ([1, 0, 0, 1, 0, 0, 0, 0, 1], 5),
        ([1, 1, 0, 0, 1, 0, 0, 1], 5),
    ],
)
def test_select_least_dense_index(occupancy: list[int], expected: int) -> None:
    """Should return the unsampled index in the least densely sampled window."""
    assert (
        _select_least_dense_index(
            occupancy=np.array(occupancy), max_half_width=len(occupancy)
        )
        == expected
    )


@pytest.mark.parametrize(("max_half_width", "expected"), [(1, 2), (2, 3), (8, 5)])
def test_select_least_dense_index_widens_window_to_break_ties(
    max_half_width: int, expected: int
) -> None:
    """Should widen the window until a single candidate is left or the cap is reached."""
    occupancy = np.array([1, 1, 0, 0, 1, 0, 0, 1])

    assert (
        _select_least_dense_index(occupancy=occupancy, max_half_width=max_half_width)
        == expected
    )
