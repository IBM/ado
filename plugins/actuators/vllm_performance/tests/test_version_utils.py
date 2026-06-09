# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Tests for vLLM version utilities."""

from ado_actuators.vllm_performance.version_utils import VLLMVersionChecker


class TestVLLMVersionChecker:
    """Tests for VLLMVersionChecker class."""

    def test_parse_version_from_list(self) -> None:
        """Test version parsing from list format."""
        image_value = ["vllm/vllm-openai:v0.20.1", "0.20.1"]
        assert VLLMVersionChecker.parse_version(image_value) == "0.20.1"

    def test_parse_version_from_list_single_element(self) -> None:
        """Test version parsing from list with single element."""
        image_value = ["vllm/vllm-openai:v0.20.1"]
        assert VLLMVersionChecker.parse_version(image_value) is None

    def test_parse_version_from_string(self) -> None:
        """Test version parsing from string format (backward compatibility)."""
        image_value = "vllm/vllm-openai:v0.20.1"
        assert VLLMVersionChecker.parse_version(image_value) is None

    def test_supports_threadpool_disabled_by_user(self) -> None:
        """Test threadpool disabled when user sets threadpool=0."""
        image_value = ["vllm/vllm-openai:v0.20.1", "0.20.1"]
        assert not VLLMVersionChecker.supports_threadpool(image_value, 0)

    def test_supports_threadpool_version_supported(self) -> None:
        """Test threadpool enabled for vLLM >= 0.20.0."""
        image_value = ["vllm/vllm-openai:v0.20.1", "0.20.1"]
        assert VLLMVersionChecker.supports_threadpool(image_value, 1)

    def test_supports_threadpool_version_not_supported(self) -> None:
        """Test threadpool disabled for vLLM < 0.20.0."""
        image_value = ["vllm/vllm-openai:v0.18.0", "0.18.0"]
        assert not VLLMVersionChecker.supports_threadpool(image_value, 1)

    def test_supports_threadpool_no_version_info(self) -> None:
        """Test threadpool enabled when no version info (backward compatible)."""
        image_value = "vllm/vllm-openai:v0.20.1"
        assert VLLMVersionChecker.supports_threadpool(image_value, 1)

    def test_supports_threadpool_invalid_version(self) -> None:
        """Test threadpool enabled for invalid version (fail-safe)."""
        image_value = ["vllm/vllm-openai:latest", "invalid-version"]
        assert VLLMVersionChecker.supports_threadpool(image_value, 1)

    def test_supports_threadpool_edge_version(self) -> None:
        """Test threadpool enabled at exact minimum version."""
        image_value = ["vllm/vllm-openai:v0.20.0", "0.20.0"]
        assert VLLMVersionChecker.supports_threadpool(image_value, 1)


# Made with Bob