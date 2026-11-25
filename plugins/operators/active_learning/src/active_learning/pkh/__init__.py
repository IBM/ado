# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Predictive Kernel Herding (PKH) active learning operator and sampler."""

from __future__ import annotations

from typing import TYPE_CHECKING

from active_learning.pkh.parameters import (
    ActiveLearningAutoGluonParameters,
    PKHParameters,
)

if TYPE_CHECKING:
    from active_learning.pkh.sampler import PKHSampleSelector

__all__ = [
    "ActiveLearningAutoGluonParameters",
    "PKHParameters",
    "PKHSampleSelector",
]


def __getattr__(name: str) -> type[PKHSampleSelector]:
    """Load the public sampler class on first access."""
    if name == "PKHSampleSelector":
        from active_learning.pkh.sampler import PKHSampleSelector

        globals()[name] = PKHSampleSelector
        return PKHSampleSelector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
