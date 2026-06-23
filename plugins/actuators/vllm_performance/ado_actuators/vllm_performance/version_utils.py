# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Utilities for vLLM version checking and threadpool support detection."""

from ado_actuators.vllm_performance.k8s import VLLMVersionExtractionError
from packaging import version


class VLLMVersionChecker:
    """Utility class for checking vLLM version and threadpool support."""

    THREADPOOL_MIN_VERSION = "0.20.0"

    @classmethod
    def extract_version_from_image(cls, image: str) -> str | None:
        """
        Extract version string from a container image string.

        Handles formats like:
        - "vllm/vllm-openai:v0.20.1" -> "0.20.1"
        - "vllm/vllm-openai:0.20.1" -> "0.20.1"
        - "0.20.1" -> "0.20.1" (already a version)

        Args:
            image: Container image string or version string

        Returns:
            Extracted version string

        Raises:
            VLLMVersionExtractionError: If version cannot be extracted from image string
        """
        if not image or not isinstance(image, str):
            raise VLLMVersionExtractionError(
                f"Invalid image value: {image}. Must be a non-empty string."
            )

        # If there's a colon, extract the tag part
        if ":" in image:
            tag = image.split(":")[-1]
            try:
                return version.parse(tag).base_version
            except version.InvalidVersion:
                return tag

        try:
            return version.parse(image).base_version
        except version.InvalidVersion as e:
            raise VLLMVersionExtractionError(
                f"Cannot extract version from image string: {image}. "
                f"Expected format: 'image:version' or a valid version string. "
            ) from e

    @classmethod
    def supports_threadpool(cls, vllm_version_str: str) -> bool:
        """
        Check if threadpool is supported. If version cannot be parsed we return True
        to avoid halting test campaigns when we don't have version info.

        Starting from 0.20.0, vLLM versions have threadpool support enabled by default.
        We optimistically return True unless we clearly have an unsupported version.

        If the image has a custom tag and threadpool is not supported,
        the evaluation will fail when the actuator will
        try to start the vLLM server with an unknown parameter.

        Args:
            vllm_version_str: vLLM version string (e.g., "0.20.1")

        Returns:
            True if threadpool is supported, False otherwise
        """
        try:
            vllm_ver = version.parse(vllm_version_str)
            min_ver = version.parse(cls.THREADPOOL_MIN_VERSION)
            return vllm_ver >= min_ver

        except version.InvalidVersion:
            return True
