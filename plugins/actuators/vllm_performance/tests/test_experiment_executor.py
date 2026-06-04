# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""
Unit tests for experiment_executor module functions.
Tests version extraction from image property values.
"""

from ado_actuators.vllm_performance.experiment_executor import (
    _get_vllm_version_from_image_value,
)


class TestGetVllmVersionFromImageValue:
    """Test suite for _get_vllm_version_from_image_value function"""

    def test_version_extraction_from_list_value(self) -> None:
        """Test extracting vLLM version from list image value"""
        image_value = [
            "icr.io/drl-nextgen/mgazz/vllm:v0.18.0-tt.v1.2.5",
            "0.18.0",
        ]

        version = _get_vllm_version_from_image_value(image_value)
        assert version == "0.18.0"

    def test_version_extraction_from_another_list_value(self) -> None:
        """Test extracting vLLM version from another list image value"""
        image_value = [
            "vllm/vllm-openai:v0.14.0",
            "0.14.0",
        ]

        version = _get_vllm_version_from_image_value(image_value)
        assert version == "0.14.0"

    def test_version_extraction_returns_none_for_string_value(self) -> None:
        """Test that None is returned when image value is a string (backward compatibility)"""
        image_value = "icr.io/drl-nextgen/mgazz/vllm:v0.18.0-tt.v1.2.5"

        version = _get_vllm_version_from_image_value(image_value)
        assert version is None

    def test_version_extraction_returns_none_for_list_without_version(self) -> None:
        """Test that None is returned when list has only one element (no version)"""
        image_value = [
            "icr.io/drl-nextgen/mgazz/vllm:v0.18.0-tt.v1.2.5",
        ]

        version = _get_vllm_version_from_image_value(image_value)
        assert version is None

    def test_version_extraction_with_latest_tag(self) -> None:
        """Test extracting version for latest tag"""
        image_value = [
            "vllm/vllm-openai:latest",
            "0.21.0",
        ]

        version = _get_vllm_version_from_image_value(image_value)
        assert version == "0.21.0"


# Made with Bob
