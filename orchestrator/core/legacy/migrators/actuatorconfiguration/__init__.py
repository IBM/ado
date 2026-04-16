# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Legacy migrators for actuatorconfiguration migrations"""

from orchestrator.core.legacy.migrators.actuatorconfiguration import (
    vllm_performance_image_secret_rename,
)

__all__ = ["vllm_performance_image_secret_rename"]

# Made with Bob
