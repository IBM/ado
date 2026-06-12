# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Tests for vLLM version utilities."""

from ado_actuators.vllm_performance.version_utils import VLLMVersionChecker


class TestVLLMVersionChecker:
    """Tests for VLLMVersionChecker class."""

    def test_extract_version_from_image_with_v_prefix(self) -> None:
        """Test version extraction from image with 'v' prefix in tag."""
        image = "vllm/vllm-openai:v0.20.1"
        assert VLLMVersionChecker.extract_version_from_image(image) == "0.20.1"

    def test_extract_version_from_image_without_v_prefix(self) -> None:
        """Test version extraction from image without 'v' prefix in tag."""
        image = "vllm/vllm-openai:0.20.1"
        assert VLLMVersionChecker.extract_version_from_image(image) == "0.20.1"

    def test_extract_version_from_plain_version_string(self) -> None:
        """Test that plain version strings are returned as-is."""
        version_str = "0.20.1"
        assert VLLMVersionChecker.extract_version_from_image(version_str) == "0.20.1"

    def test_extract_version_from_plain_version_with_v(self) -> None:
        """Test that plain version strings with 'v' prefix have it removed."""
        version_str = "v0.20.1"
        assert VLLMVersionChecker.extract_version_from_image(version_str) == "0.20.1"

    def test_extract_version_from_image_latest_tag(self) -> None:
        """Test version extraction from image with 'latest' tag."""
        image = "vllm/vllm-openai:latest"
        assert VLLMVersionChecker.extract_version_from_image(image) == "latest"

    def test_supports_threadpool_version_supported(self) -> None:
        """Test threadpool enabled for vLLM >= 0.20.0."""
        version_str = "0.20.1"
        assert VLLMVersionChecker.supports_threadpool(version_str)

    def test_supports_threadpool_version_not_supported(self) -> None:
        """Test threadpool disabled for vLLM < 0.20.0."""
        version_str = "0.18.0"
        assert not VLLMVersionChecker.supports_threadpool(version_str)

    def test_supports_threadpool_invalid_version(self) -> None:
        """Test threadpool enabled for invalid version (fail-safe)."""
        version_str = "invalid-version"
        assert VLLMVersionChecker.supports_threadpool(version_str)

    def test_supports_threadpool_edge_version(self) -> None:
        """Test threadpool enabled at exact minimum version."""
        version_str = "0.20.0"
        assert VLLMVersionChecker.supports_threadpool(version_str)

    def test_supports_threadpool_with_image_extraction(self) -> None:
        """Test full workflow: extract version from image then check threadpool support."""
        image = "vllm/vllm-openai:v0.20.1"
        version_str = VLLMVersionChecker.extract_version_from_image(image)
        assert VLLMVersionChecker.supports_threadpool(version_str)

    def test_supports_threadpool_with_old_image_extraction(self) -> None:
        """Test full workflow with old version: extract then check threadpool support."""
        image = "vllm/vllm-openai:v0.18.0"
        version_str = VLLMVersionChecker.extract_version_from_image(image)
        assert not VLLMVersionChecker.supports_threadpool(version_str)
