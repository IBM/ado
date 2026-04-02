# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Legacy migrator system for handling deprecated resource formats"""

from orchestrator.core.legacy.metadata import LegacyMigratorMetadata
from orchestrator.core.legacy.registry import (
    LegacyMigratorRegistry,
    legacy_migrator,
)

__all__ = [
    "LegacyMigratorMetadata",
    "LegacyMigratorRegistry",
    "legacy_migrator",
]

# Made with Bob
