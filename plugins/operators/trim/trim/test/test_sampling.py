# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import pytest
from trim.utils.one_dimensional_sampling import (
    get_index_list_ordered_partitions,
    get_index_list_van_der_corput,
)  # Replace with actual module name

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


# --- Functional Tests for get_index_list_ordered_partitions ---


@pytest.mark.parametrize(
    ("points", "expected"),
    [
        (7, [0, 2, 4, 8, 10, 12, 16]),
        (8, [0, 2, 4, 6, 8, 10, 12, 16]),
        (9, [0, 2, 4, 6, 8, 10, 12, 14, 16]),
    ],
)
def test_get_index_list_ordered_partitions_sampling(
    points: int, expected: list[int]
) -> None:
    """Should return correct partition-based sampling for segment of length 17."""
    assert get_index_list_ordered_partitions(17, points) == expected
