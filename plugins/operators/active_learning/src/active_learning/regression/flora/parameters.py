# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Configuration models for the FLORA operator and sampler."""

from typing import Annotated

import pydantic

from active_learning.regression._shared import _PredictiveParameters


class FLORAParameters(_PredictiveParameters):
    """Configuration for FLORA finite-pool acquisition."""


class FLORAOperatorParameters(FLORAParameters):
    """Configuration exposed by the FLORA characterization operator."""

    numberEntities: Annotated[
        int,
        pydantic.Field(
            gt=0,
            description="Number of entities to select and measure.",
        ),
    ]

    @classmethod
    def example_configuration(cls) -> "FLORAOperatorParameters":
        """Return a minimal example operator configuration."""

        return cls(targetOutput="TO_BE_SET", numberEntities=10)
