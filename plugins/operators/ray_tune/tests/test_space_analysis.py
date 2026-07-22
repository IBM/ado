# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Tests for space_analysis.py, focusing on edge cases in mi_pareto_selection."""

from ado_ray_tune.space_analysis import mi_pareto_selection


class TestMiParetoSelection:
    """Tests for mi_pareto_selection."""

    def test_all_mi_below_ignore_threshold_returns_empty(self) -> None:
        """When all MI values are at or below ignore_below, return []."""
        mi = {"provider": 0.0, "cpu_family": 0.0, "vcpu_size": 0.0, "nodes": 0.00005}
        result = mi_pareto_selection(mi)
        assert result == []

    def test_empty_mi_dict_returns_empty(self) -> None:
        """When MI dict is empty, return []."""
        mi: dict[str, float] = {}
        result = mi_pareto_selection(mi)
        assert result == []

    def test_single_dimension_above_threshold_returns_empty(self) -> None:
        """With only one dimension above ignore_below, no 2-combinations exist."""
        mi = {"provider": 0.0, "cpu_family": 0.0, "vcpu_size": 0.0, "nodes": 0.454}
        result = mi_pareto_selection(mi)
        assert result == []

    def test_single_dimension_only_returns_empty(self) -> None:
        """MI dict with a single key, above threshold."""
        mi = {"nodes": 0.5}
        result = mi_pareto_selection(mi)
        assert result == []

    def test_multiple_dimensions_returns_selection(self) -> None:
        """Normal case: multiple dimensions above ignore_below returns a selection."""
        mi = {"provider": 0.3, "cpu_family": 0.4, "vcpu_size": 0.2, "nodes": 0.5}
        result = mi_pareto_selection(mi)
        assert isinstance(result, list)
        assert len(result) > 0
        # All returned dimensions must be keys in the input MI dict
        for dim in result:
            assert dim in mi

    def test_two_dimensions_returns_selection(self) -> None:
        """Two dimensions both above ignore_below."""
        mi = {"cpu_family": 0.4, "nodes": 0.6}
        result = mi_pareto_selection(mi)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_return_all_above_threshold(self) -> None:
        """With return_all_above_threshold=True, returns a tuple."""
        mi = {"provider": 0.3, "cpu_family": 0.4, "nodes": 0.5}
        result = mi_pareto_selection(mi, return_all_above_threshold=True)
        assert isinstance(result, tuple)
        dimensions, _above_threshold_df = result
        assert isinstance(dimensions, list)

    def test_return_all_above_threshold_empty(self) -> None:
        """With return_all_above_threshold=True and single dim, returns empty tuple."""
        mi = {"nodes": 0.5}
        result = mi_pareto_selection(mi, return_all_above_threshold=True)
        assert isinstance(result, tuple)
        dimensions, _above_threshold_df = result
        assert dimensions == []
