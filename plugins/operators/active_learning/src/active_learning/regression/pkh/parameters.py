# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Configuration models for the PKH operator and sampler."""

from typing import Annotated

import pydantic

from active_learning.regression._shared import _PredictiveParameters


class PKHParameters(_PredictiveParameters):
    """Configuration for predictive kernel herding."""

    epochLength: Annotated[
        int,
        pydantic.Field(
            ge=1,
            description="Selections made before the forest is fitted again.",
        ),
    ] = 10


class PKHOperatorParameters(PKHParameters):
    """Configuration exposed by the PKH characterization operator."""

    numberEntities: Annotated[
        int,
        pydantic.Field(
            gt=0,
            description="Number of entities to select and measure.",
        ),
    ]

    @classmethod
    def example_configuration(cls) -> "PKHOperatorParameters":
        """Return a minimal example operator configuration."""

        return cls(targetOutput="TO_BE_SET", numberEntities=10)
