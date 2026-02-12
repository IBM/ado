# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""
Tests for high-dimensional sampling functions.

This module tests the concatenated_latin_hypercube_sampling function
to ensure it maintains proper stratification properties across dimensions.
"""

import math
from collections import Counter
from typing import Any

import pytest
from trim.test.test_data_documentation import TEST_DATAFRAMES
from trim.utils.high_dimensional_sampling import concatenated_latin_hypercube_sampling


class TestConcatenatedLatinHypercubeSampling:
    """Test suite for concatenated_latin_hypercube_sampling function."""

    @pytest.fixture(params=list(TEST_DATAFRAMES.keys()))
    def df_config(self, request: pytest.FixtureRequest) -> dict[str, Any]:
        """Fixture providing all test dataframe configurations."""
        return TEST_DATAFRAMES[request.param]

    def _verify_stratification(
        self, samples: list[list[int]], dimensions: list[int], n: int
    ) -> None:
        """
        Verify that each dimension maintains Latin Hypercube stratification.

        For each dimension k with cardinality d_k, verify that each value
        in range(d_k) appears either floor(n/d_k) or ceil(n/d_k) times.

        Args:
            samples: List of sampled points
            dimensions: Dimension cardinalities
            n: Total number of samples
        """
        for dim_idx, dim_size in enumerate(dimensions):
            # Extract values for this dimension across all samples
            dim_values = [sample[dim_idx] for sample in samples]
            value_counts = Counter(dim_values)

            # Calculate expected counts
            floor_count = math.floor(n / dim_size)
            ceil_count = math.ceil(n / dim_size)

            # Verify all values in range are present (only if n >= dim_size)
            # When n < dim_size, we only expect n unique values
            if n >= dim_size:
                assert set(value_counts.keys()) == set(
                    range(dim_size)
                ), f"Dimension {dim_idx}: Not all values present. Expected {set(range(dim_size))}, got {set(value_counts.keys())}"
            else:
                # When n < dim_size, we should have exactly n unique values
                assert (
                    len(value_counts) == n
                ), f"Dimension {dim_idx}: Expected {n} unique values, got {len(value_counts)}"

            # Verify each value appears floor(n/k) or ceil(n/k) times
            for value, count in value_counts.items():
                assert count in [
                    floor_count,
                    ceil_count,
                ], f"Dimension {dim_idx}, value {value}: count {count} not in [{floor_count}, {ceil_count}]"

            # Additional check: verify the distribution is balanced
            # The number of values with ceil_count should equal n % dim_size
            # Only check this when floor_count != ceil_count (i.e., when n % dim_size != 0)
            if floor_count != ceil_count:
                ceil_count_occurrences = sum(
                    1 for c in value_counts.values() if c == ceil_count
                )
                expected_ceil_occurrences = n % dim_size

                assert (
                    ceil_count_occurrences == expected_ceil_occurrences
                ), f"Dimension {dim_idx}: Expected {expected_ceil_occurrences} values with count {ceil_count}, got {ceil_count_occurrences}"

    def test_quarter_sampling(self, df_config: dict[str, Any]) -> None:
        """
        Test sampling with n = len(df) / 4.

        Tests all dataframes from TEST_DATAFRAMES.
        Verifies Latin Hypercube stratification for each dimension.
        """
        dimensions = df_config["dimensions"]
        total_configs = df_config["total_configs"]
        n = total_configs // 4

        samples = concatenated_latin_hypercube_sampling(
            dimensions=dimensions, final_sample_size=n, seed=42
        )

        # Basic checks
        assert len(samples) == n, f"Expected {n} samples, got {len(samples)}"
        assert all(
            len(sample) == len(dimensions) for sample in samples
        ), "All samples should have correct dimensionality"

        # Verify stratification
        self._verify_stratification(samples, dimensions, n)

    def test_half_sampling(self, df_config: dict[str, Any]) -> None:
        """
        Test sampling with n = len(df) / 2.

        Tests all dataframes from TEST_DATAFRAMES.
        Verifies Latin Hypercube stratification for each dimension.
        """
        dimensions = df_config["dimensions"]
        total_configs = df_config["total_configs"]
        n = total_configs // 2

        samples = concatenated_latin_hypercube_sampling(
            dimensions=dimensions, final_sample_size=n, seed=42
        )

        # Basic checks
        assert len(samples) == n, f"Expected {n} samples, got {len(samples)}"
        assert all(
            len(sample) == len(dimensions) for sample in samples
        ), "All samples should have correct dimensionality"

        # Verify stratification
        self._verify_stratification(samples, dimensions, n)

    def test_stratification_detailed_quarter(self, df_config: dict[str, Any]) -> None:
        """
        Detailed test for quarter sampling showing exact counts per dimension.

        Tests all dataframes and verifies the exact distribution of values.
        """
        dimensions = df_config["dimensions"]
        total_configs = df_config["total_configs"]
        n = total_configs // 4

        samples = concatenated_latin_hypercube_sampling(
            dimensions=dimensions, final_sample_size=n, seed=42
        )

        # Verify detailed stratification for each dimension
        for dim_idx, dim_size in enumerate(dimensions):
            dim_values = [sample[dim_idx] for sample in samples]
            dim_counts = Counter(dim_values)

            floor_count = n // dim_size
            ceil_count = math.ceil(n / dim_size)

            # All counts should be either floor or ceil
            for value, count in dim_counts.items():
                assert count in [floor_count, ceil_count], (
                    f"{df_config['description']}: Dimension {dim_idx}, value {value}: "
                    f"count {count} not in [{floor_count}, {ceil_count}]"
                )

    def test_stratification_detailed_half(self, df_config: dict[str, Any]) -> None:
        """
        Detailed test for half sampling showing exact counts per dimension.

        Tests all dataframes and verifies the exact distribution of values.
        """
        dimensions = df_config["dimensions"]
        total_configs = df_config["total_configs"]
        n = total_configs // 2

        samples = concatenated_latin_hypercube_sampling(
            dimensions=dimensions, final_sample_size=n, seed=42
        )

        # Verify detailed stratification for each dimension
        for dim_idx, dim_size in enumerate(dimensions):
            dim_values = [sample[dim_idx] for sample in samples]
            dim_counts = Counter(dim_values)

            floor_count = n // dim_size
            ceil_count = math.ceil(n / dim_size)

            # All counts should be either floor or ceil
            for value, count in dim_counts.items():
                assert count in [floor_count, ceil_count], (
                    f"{df_config['description']}: Dimension {dim_idx}, value {value}: "
                    f"count {count} not in [{floor_count}, {ceil_count}]"
                )

    def test_reproducibility_with_seed(self) -> None:
        """Test that using the same seed produces identical results."""
        # Use a fixed configuration for reproducibility test
        dimensions = [5, 4, 3, 2]
        n = 30
        seed = 12345

        samples1 = concatenated_latin_hypercube_sampling(
            dimensions=dimensions, final_sample_size=n, seed=seed
        )
        samples2 = concatenated_latin_hypercube_sampling(
            dimensions=dimensions, final_sample_size=n, seed=seed
        )

        assert samples1 == samples2, "Same seed should produce identical samples"

    def test_different_seeds_produce_different_results(self) -> None:
        """Test that different seeds produce different results."""
        # Use a fixed configuration for seed comparison test
        dimensions = [5, 4, 3, 2]
        n = 30

        samples1 = concatenated_latin_hypercube_sampling(
            dimensions=dimensions, final_sample_size=n, seed=42
        )
        samples2 = concatenated_latin_hypercube_sampling(
            dimensions=dimensions, final_sample_size=n, seed=123
        )

        assert samples1 != samples2, "Different seeds should produce different samples"

    def test_empty_sampling(self) -> None:
        """Test that n=0 returns empty list."""
        dimensions = [5, 4, 3, 2]
        samples = concatenated_latin_hypercube_sampling(
            dimensions=dimensions, final_sample_size=0, seed=42
        )
        assert samples == [], "n=0 should return empty list"

    def test_invalid_dimensions(self) -> None:
        """Test that invalid dimensions raise ValueError."""
        with pytest.raises(ValueError, match="All dimensions must be >= 1"):
            concatenated_latin_hypercube_sampling(
                dimensions=[5, 0, 3], final_sample_size=10, seed=42
            )

        with pytest.raises(ValueError, match="All dimensions must be >= 1"):
            concatenated_latin_hypercube_sampling(
                dimensions=[5, -1, 3], final_sample_size=10, seed=42
            )

    def test_single_dimension(self) -> None:
        """Test sampling with a single dimension."""
        dimensions = [10]
        n = 5

        samples = concatenated_latin_hypercube_sampling(
            dimensions=dimensions, final_sample_size=n, seed=42
        )

        assert len(samples) == n
        self._verify_stratification(samples, dimensions, n)

    def test_exceeds_dimension_size(self, df_config: dict[str, Any]) -> None:
        """
        Test sampling when n exceeds the smallest dimension size.

        This tests the concatenation behavior where new permutations
        are created when a dimension's pool is exhausted.
        """
        dimensions = df_config["dimensions"]
        min_dim = min(dimensions)
        # Sample more than the smallest dimension
        n = min_dim * 2 + 1

        samples = concatenated_latin_hypercube_sampling(
            dimensions=dimensions, final_sample_size=n, seed=42
        )

        assert len(samples) == n
        self._verify_stratification(samples, dimensions, n)
