# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Utilities for vLLM version checking and threadpool support detection."""

from packaging import version


class VLLMVersionChecker:
    """Utility class for checking vLLM version and threadpool support."""

    THREADPOOL_MIN_VERSION = "0.20.0"

    @classmethod
    def extract_version_from_image(cls, image: str) -> str:
        """
        Extract version string from a container image string.

        Handles formats like:
        - "vllm/vllm-openai:v0.20.1" -> "0.20.1"
        - "vllm/vllm-openai:0.20.1" -> "0.20.1"
        - "0.20.1" -> "0.20.1" (already a version)

        Args:
            image: Container image string or version string

        Returns:
            Extracted version string, or None if version cannot be extracted
        """
        if not image or not isinstance(image, str):
            return None

        # If there's a colon, extract the tag part
        if ":" in image:
            tag = image.split(":")[-1]
            # Remove leading 'v' if present
            tag = tag.removeprefix("v")
            return tag or None

        # If no colon, assume it's already a version string
        # Remove leading 'v' if present
        if image.startswith("v"):
            return image[1:]

        return image

    @classmethod
    def supports_threadpool(cls, vllm_version_str: str) -> bool:
        """
        Check if threadpool is supported. If version cannot be parsed we return True
        to avoid halting test campaigns when we don't have version info.

        New versions of vLLM will have threadpool support enabled by default therefore
        is more likely that the version is supported than not supported.

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

        except Exception:
            return True
