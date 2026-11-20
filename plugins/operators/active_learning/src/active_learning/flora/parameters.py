# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Configuration for FLORA's finite-pool acquisition."""

from typing import Annotated

from pydantic import ConfigDict, Field, PositiveFloat, PositiveInt

from active_learning.parameters import (
    ActiveLearningAutoGluonParameters,
    ActiveLearningParameters,
    ForestJobCount,
    OutputDirectory,
)


class FLORAAutoGluonParameters(ActiveLearningAutoGluonParameters):
    """Resource limits for the optional final FLORA regressor."""

    timeLimit: Annotated[
        PositiveFloat,
        Field(
            serialization_alias="time_limit",
            allow_inf_nan=False,
            description="Training time limit in seconds.",
        ),
    ] = 60.0
    numCpus: Annotated[
        PositiveInt,
        Field(
            strict=True,
            serialization_alias="num_cpus",
            description="CPUs available to AutoGluon.",
        ),
    ] = 1


class FLORAParameters(ActiveLearningParameters):
    """Settings for FLORA's pilot and log-tempered forest refits."""

    model_config = ConfigDict(str_strip_whitespace=True)
    autogluon_parameters_type = FLORAAutoGluonParameters
    autoGluon: Annotated[
        FLORAAutoGluonParameters | None,
        Field(description="Fit arguments when finalPredictor is autogluon."),
    ] = None
    pilotSize: Annotated[
        PositiveInt,
        Field(
            strict=True,
            description="Total pilot labels, including existing measurements.",
        ),
    ] = 5
    seed: Annotated[
        int,
        Field(ge=0, lt=2**32, description="Random seed for pilot, forests, and ties."),
    ] = 42
    nJobs: ForestJobCount = -1
    outputDirectory: OutputDirectory = "flora_models"
