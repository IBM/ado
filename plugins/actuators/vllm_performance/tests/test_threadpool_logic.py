# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Tests for threadpool usage determination logic."""

import pytest
from ado_actuators.vllm_performance.experiment_executor import (
    _is_threadpool_requested,
)


class TestDetermineThreadpoolUsage:
    """Test cases for _is_threadpool_requested function."""

    def test_with_workers(self) -> None:
        """renderer_num_workers=32 -> True."""
        values = {"renderer_num_workers": 32}
        assert _is_threadpool_requested(values) is True

    def test_with_zero_workers(self) -> None:
        """renderer_num_workers=0 -> False (no threadpool)."""
        values = {"renderer_num_workers": 0}
        assert _is_threadpool_requested(values) is False

    def test_without_workers(self) -> None:
        """missing renderer_num_workers -> False."""
        values = {}
        assert _is_threadpool_requested(values) is False

    def test_with_string_workers(self) -> None:
        """renderer_num_workers='32' -> True."""
        values = {"renderer_num_workers": "32"}
        assert _is_threadpool_requested(values) is True

    def test_with_string_zero_workers(self) -> None:
        """renderer_num_workers='0' -> False (no threadpool)."""
        values = {"renderer_num_workers": "0"}
        assert _is_threadpool_requested(values) is False

    def test_with_none_workers(self) -> None:
        """renderer_num_workers=None -> False."""
        values = {"renderer_num_workers": None}
        assert _is_threadpool_requested(values) is False

    def test_with_negative_workers(self) -> None:
        """renderer_num_workers=-1 -> ValueError."""
        values = {"renderer_num_workers": -1}
        with pytest.raises(ValueError, match="must be non-negative"):
            _is_threadpool_requested(values)

    def test_with_positive_workers(self) -> None:
        """renderer_num_workers=1 -> True (threadpool enabled)."""
        values = {"renderer_num_workers": 1}
        assert _is_threadpool_requested(values) is True

    def test_with_negative_string_workers(self) -> None:
        """renderer_num_workers='-1' -> ValueError."""
        values = {"renderer_num_workers": "-1"}
        with pytest.raises(ValueError, match="must be non-negative"):
            _is_threadpool_requested(values)
