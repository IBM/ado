# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Common settings and saved predictor metadata for active learning."""

from __future__ import annotations

import importlib.util
from typing import Annotated, Any, ClassVar, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator

from ado.core.operation.config import GenericOperatorParameters


class ActiveLearningAutoGluonParameters(BaseModel):
    """Fit arguments for the optional final AutoGluon predictor."""

    model_config = ConfigDict(extra="forbid")

    presets: Annotated[
        Literal["medium_quality", "good_quality", "high_quality", "best_quality"],
        Field(description="AutoGluon training preset."),
    ] = "medium_quality"
    timeLimit: Annotated[
        PositiveInt,
        Field(
            serialization_alias="time_limit",
            description="Training time limit in seconds.",
        ),
    ] = 60
    hyperparameters: Annotated[
        dict[str, Any] | None, Field(description="AutoGluon model hyperparameters.")
    ] = None


OutputDirectory = Annotated[
    str, Field(min_length=1, description="Output path, with _finalized appended.")
]
ForestJobCount = Annotated[
    Annotated[int, Field(le=-1)] | PositiveInt,
    Field(description="Parallel forest jobs; -1 uses all processors."),
]


class ActiveLearningParameters(GenericOperatorParameters):
    """Shared settings for forest fitting, measurements, and saved predictions."""

    autogluon_parameters_type: ClassVar[type[ActiveLearningAutoGluonParameters]] = (
        ActiveLearningAutoGluonParameters
    )

    pilotSize: Annotated[
        PositiveInt,
        Field(description="Total pilot labels, including existing measurements."),
    ] = 5
    nEstimators: Annotated[
        PositiveInt, Field(description="Trees in each Random Forest.")
    ] = 100
    minSamplesLeaf: Annotated[
        PositiveInt, Field(description="Minimum training samples in a leaf.")
    ] = 2
    maxFeatures: Annotated[
        Literal["sqrt", "log2"]
        | Annotated[int, Field(ge=1)]
        | Annotated[float, Field(gt=0, le=1)]
        | None,
        Field(description="Features per split: count, fraction, sqrt, log2, or null."),
    ] = "sqrt"
    criterion: Annotated[
        Literal["squared_error", "absolute_error", "friedman_mse", "poisson"],
        Field(description="Random Forest split criterion."),
    ] = "squared_error"
    seed: Annotated[
        int | None,
        Field(ge=0, lt=2**32, description="Random seed for pilot, forests, and ties."),
    ] = 42
    nJobs: ForestJobCount = 1
    targetOutput: Annotated[
        str,
        Field(
            min_length=1,
            description="Unique target name or qualified observed property.",
        ),
    ]
    numberEntities: Annotated[
        PositiveInt,
        Field(description="New entities to measure, including any pilot additions."),
    ]
    outputDirectory: OutputDirectory
    finalPredictor: Annotated[
        Literal["random_forest", "autogluon"],
        Field(description="Final predictor backend."),
    ] = "random_forest"
    autoGluon: Annotated[
        ActiveLearningAutoGluonParameters | None,
        Field(description="Fit arguments when finalPredictor is autogluon."),
    ] = None

    @model_validator(mode="after")
    def validate_predictor_configuration(self) -> Self:
        """Validate and default arguments for the selected predictor backend."""
        if self.finalPredictor == "autogluon":
            if (
                importlib.util.find_spec("autogluon") is None
                or importlib.util.find_spec("autogluon.tabular") is None
            ):
                raise ValueError(
                    "Install 'ado-active-learning[autogluon]' to use finalPredictor='autogluon'."
                )
            self.autoGluon = self.autoGluon or self.autogluon_parameters_type()
        elif self.autoGluon is not None:
            raise ValueError(
                "autoGluon parameters must not be provided for random_forest"
            )
        return self

    @classmethod
    def example_configuration(cls) -> Self:
        """Return a minimal example for operator discovery."""
        return cls(targetOutput="objective", numberEntities=20)


class ActiveLearningPredictorMetadata(BaseModel):
    """Fields both active-learning operators write alongside their final predictor."""

    model_config = ConfigDict(extra="forbid")

    backend: Annotated[Literal["random_forest", "autogluon"], Field()]
    targetOutput: Annotated[str, Field()]
    featureNames: Annotated[list[str], Field()]
    categoricalColumns: Annotated[list[str], Field()]
    measuredCount: Annotated[int, Field(ge=0)]
    createdTimestamp: Annotated[str, Field()]
