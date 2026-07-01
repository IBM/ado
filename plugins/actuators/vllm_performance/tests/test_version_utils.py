# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Tests for vLLM version utilities."""

import pytest
from ado_actuators.vllm_performance.k8s import VLLMVersionExtractionError
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

    def test_extract_version_from_empty_string(self) -> None:
        """Test that empty string raises VLLMVersionExtractionError."""
        with pytest.raises(VLLMVersionExtractionError, match="Invalid image value"):
            VLLMVersionChecker.extract_version_from_image("")

    def test_extract_version_from_none(self) -> None:
        """Test that None raises VLLMVersionExtractionError."""
        with pytest.raises(VLLMVersionExtractionError, match="Invalid image value"):
            VLLMVersionChecker.extract_version_from_image(None)  # type: ignore[arg-type]

    def test_extract_version_from_non_string(self) -> None:
        """Test that non-string input raises VLLMVersionExtractionError."""
        with pytest.raises(VLLMVersionExtractionError, match="Invalid image value"):
            VLLMVersionChecker.extract_version_from_image(123)  # type: ignore[arg-type]

    def test_extract_version_from_empty_tag(self) -> None:
        """Test that image with empty tag after colon returns empty string."""
        image = "vllm/vllm-openai:"
        assert VLLMVersionChecker.extract_version_from_image(image) == ""

    def test_extract_version_from_only_v_tag(self) -> None:
        """Test that image with only 'v' as tag returns 'v'."""
        image = "vllm/vllm-openai:v"
        assert VLLMVersionChecker.extract_version_from_image(image) == "v"

    def test_extract_version_from_vllm_image_without_tag(self) -> None:
        """Test that image without tag returns the image name"""
        image = "vllm/vllm-openai"
        with pytest.raises(VLLMVersionExtractionError):
            VLLMVersionChecker.extract_version_from_image(image=image)  # type: ignore[arg-type]

    def test_extract_version_from_image_without_tag(self) -> None:
        """Test that image without tag (no colon) returns None"""
        image = "custom-image"
        with pytest.raises(VLLMVersionExtractionError):
            VLLMVersionChecker.extract_version_from_image(image=image)  # type: ignore[arg-type]
