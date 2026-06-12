# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT


import pytest
from ado_actuators.vllm_performance.k8s import UnsupportedThreadpoolConfigurationError


def test_unsupported_threadpool_configuration_error_message() -> None:
    """Unsupported threadpool requests should produce a clear error message."""

    error = UnsupportedThreadpoolConfigurationError(
        "Threadpool requested but not supported by image vllm/vllm-openai:v0.18.0"
    )

    assert (
        str(error)
        == "Threadpool requested but not supported by image vllm/vllm-openai:v0.18.0"
    )


def test_unsupported_threadpool_configuration_error_is_an_exception() -> None:
    """Unsupported threadpool configuration error should be catchable as an exception."""

    with pytest.raises(UnsupportedThreadpoolConfigurationError):
        raise UnsupportedThreadpoolConfigurationError("unsupported threadpool")


def test_unsupported_threadpool_configuration_error_accepts_string_context() -> None:
    """Unsupported threadpool configuration error should preserve string context."""

    image_name = "vllm/vllm-openai:v0.18.0"
    error = UnsupportedThreadpoolConfigurationError(
        f"Threadpool requested but not supported by image {image_name}"
    )

    assert isinstance(error, Exception)
    assert image_name in str(error)
