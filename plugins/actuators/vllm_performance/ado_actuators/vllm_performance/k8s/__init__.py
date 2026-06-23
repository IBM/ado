# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT


class K8sEnvironmentCreationError(Exception):
    """Error raised when K8 environment cannot be created for some reason"""


class UnsupportedThreadpoolConfigurationError(Exception):
    """Error raised when threadpool is requested for an unsupported vLLM image."""


class K8sConnectionError(Exception):
    """Error raised when there is an issue connecting to K8s or a service its hosting"""


class VLLMVersionExtractionError(Exception):
    """Error raised when vLLM version cannot be extracted from image string"""


class InvalidImageStructureError(Exception):
    """Error raised when image value has invalid structure"""
