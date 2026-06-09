# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Utilities for vLLM version checking and threadpool support detection."""

from packaging import version


class VLLMVersionChecker:
    """Utility class for checking vLLM version and threadpool support."""

    THREADPOOL_MIN_VERSION = "0.20.0"

    @staticmethod
    def parse_version(image_value: list | str) -> str | None:
        """Extract vLLM version from image property value."""
        if isinstance(image_value, list) and len(image_value) > 1:
            return image_value[1]
        return None

    @classmethod
    def supports_threadpool(
        cls, image_value: list | str, threadpool_requested: int
    ) -> bool:
        """Check if threadpool is supported and should be enabled."""
        if threadpool_requested == 0:
            return False

        vllm_version_str = cls.parse_version(image_value)
        if vllm_version_str is None:
            return True

        try:
            vllm_ver = version.parse(vllm_version_str)
            min_ver = version.parse(cls.THREADPOOL_MIN_VERSION)
            return vllm_ver >= min_ver
        except Exception:
            return True


# Made with Bob