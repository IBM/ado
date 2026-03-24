# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Legacy validators for deprecated resource formats"""

# Import all validator subpackages to trigger registration
from orchestrator.core.legacy.validators import (
    discoveryspace,
    operation,
    resource,
    samplestore,
)

__all__ = ["discoveryspace", "operation", "resource", "samplestore"]

# Made with Bob
