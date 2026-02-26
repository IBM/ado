# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Legacy validator system for handling deprecated resource formats"""

from orchestrator.core.legacy.metadata import LegacyValidatorMetadata
from orchestrator.core.legacy.registry import (
    LegacyValidatorRegistry,
    legacy_validator,
)

__all__ = [
    "LegacyValidatorMetadata",
    "LegacyValidatorRegistry",
    "legacy_validator",
]

# Made with Bob
