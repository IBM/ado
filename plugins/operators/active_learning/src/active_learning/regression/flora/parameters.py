# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Configuration models for the FLORA operator and sampler."""

from typing import Annotated

import pydantic

from active_learning.regression._shared import (
    _PredictiveOperatorParameters,
    _PredictiveParameters,
)


class FLORAParameters(_PredictiveParameters):
    """Configuration for FLORA finite-pool acquisition."""


class FLORAOperatorParameters(FLORAParameters, _PredictiveOperatorParameters):
    """Configuration exposed by the FLORA characterization operator."""

    numberEntities: Annotated[
        int,
        pydantic.Field(
            gt=0,
            description="Number of entities to select and measure.",
        ),
    ]
    outputDirectory: Annotated[
        str,
        pydantic.StringConstraints(strip_whitespace=True, min_length=1),
        pydantic.Field(
            description=(
                "Base path for the saved predictor. The final model is written "
                "to {outputDirectory}_finalized; that directory must not exist."
            ),
        ),
    ] = "flora_models"

    @classmethod
    def example_configuration(cls) -> "FLORAOperatorParameters":
        """Return a minimal example operator configuration."""

        return cls(targetOutput="TO_BE_SET", numberEntities=10)
