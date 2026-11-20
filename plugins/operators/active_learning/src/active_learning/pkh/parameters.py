# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""PKH configuration and saved predictor metadata."""

from typing import Annotated

from pydantic import Field, PositiveInt

from active_learning.parameters import (
    ActiveLearningAutoGluonParameters,
    ActiveLearningParameters,
    ActiveLearningPredictorMetadata,
    OutputDirectory,
)

__all__ = [
    "ActiveLearningAutoGluonParameters",
    "FinalPredictorMetadata",
    "PKHParameters",
]


class PKHParameters(ActiveLearningParameters):
    """Configuration for PKH selection, measurement, and final prediction."""

    epochLength: Annotated[
        PositiveInt, Field(description="Measurements between Random Forest refits.")
    ] = 10
    outputDirectory: OutputDirectory = "pkh_models"


class FinalPredictorMetadata(ActiveLearningPredictorMetadata):
    """PKH predictor metadata including the final sampling discrepancy."""

    pkhVersion: Annotated[str, Field()]
    mmdSquared: Annotated[
        float | None,
        Field(
            description="Final forest leaf-space MMD squared; null with fewer than two labels."
        ),
    ] = None
