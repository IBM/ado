# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Legacy migrators for deprecated resource formats"""

# Import all migrator subpackages to trigger registration
from orchestrator.core.legacy.migrators import (
    discoveryspace,
    operation,
    resource,
    samplestore,
)

__all__ = ["discoveryspace", "operation", "resource", "samplestore"]

# Made with Bob
