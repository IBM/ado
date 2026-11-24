# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""PKH sampling and prediction configuration."""

from __future__ import annotations

import importlib
import sys
from typing import Annotated, Any, Literal, get_args

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
    model_validator,
)

from ado.core.operation.config import GenericOperatorParameters

TreeFamily = Literal["RF", "XT", "XGB", "CAT", "GBM"]
TreeSettings = dict[str, Any] | Annotated[list[dict[str, Any]], Field(min_length=1)]


class ActiveLearningAutoGluonParameters(BaseModel):
    """Fit arguments for AutoGluon's final tree predictor."""

    model_config = ConfigDict(extra="forbid")

    presets: Annotated[
        Literal["medium_quality", "good_quality", "high_quality", "best_quality"],
        Field(description="AutoGluon training preset."),
    ] = "medium_quality"
    timeLimit: Annotated[
        PositiveInt,
        Field(
            validation_alias=AliasChoices("timeLimit", "time_limit"),
            serialization_alias="time_limit",
            description="Training time limit in seconds.",
        ),
    ] = 60
    hyperparameters: Annotated[
        dict[TreeFamily, TreeSettings] | None,
        Field(min_length=1, description="AutoGluon hyperparameters by tree family."),
    ] = None


class PKHParameters(GenericOperatorParameters):
    """Configuration for PKH selection, measurement, and final prediction."""

    pilotSize: Annotated[
        PositiveInt,
        Field(description="Total pilot labels, including existing measurements."),
    ] = 5
    epochLength: Annotated[
        PositiveInt, Field(description="Measurements between sampling forest refits.")
    ] = 10
    nEstimators: Annotated[
        PositiveInt, Field(description="Trees in the sampling forest.")
    ] = 100
    minSamplesLeaf: Annotated[
        PositiveInt,
        Field(description="Minimum training samples per sampling forest leaf."),
    ] = 2
    maxFeatures: Annotated[
        Literal["sqrt", "log2"]
        | Annotated[int, Field(ge=1)]
        | Annotated[float, Field(gt=0, le=1)]
        | None,
        Field(
            description="Features per sampling forest split: count, fraction, sqrt, log2, or null."
        ),
    ] = "sqrt"
    criterion: Annotated[
        Literal["squared_error", "absolute_error", "friedman_mse", "poisson"],
        Field(description="Sampling forest split criterion."),
    ] = "squared_error"
    seed: Annotated[
        int | None,
        Field(ge=0, lt=2**32, description="Random seed for pilot, forests, and ties."),
    ] = 42
    nJobs: Annotated[
        Annotated[int, Field(le=-1)] | PositiveInt,
        Field(
            description="CPU jobs for the sampling forest and final predictor. Use -1 for all processors."
        ),
    ] = 1
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
    outputDirectory: Annotated[
        str | None,
        Field(
            min_length=1,
            exclude=True,
            deprecated=True,
            description="Legacy output path accepted for existing PKH configurations.",
        ),
    ] = None
    finalPredictor: Annotated[
        Literal["random_forest", "weighted_ensemble", "autogluon"],
        Field(
            description="Random Forest, five-family weighted L2 ensemble, or configured AutoGluon trees."
        ),
    ] = "random_forest"
    autoGluon: Annotated[
        ActiveLearningAutoGluonParameters | None,
        Field(description="AutoGluon fit options for completing the prediction table."),
    ] = None

    @model_validator(mode="after")
    def validate_predictor_configuration(self) -> PKHParameters:
        """Keep the Random Forest mode restricted to the RF family."""
        if (
            self.finalPredictor == "random_forest"
            and self.autoGluon
            and self.autoGluon.hyperparameters
            and self.autoGluon.hyperparameters.keys() != {"RF"}
        ):
            raise ValueError("random_forest accepts only RF hyperparameters.")
        return self

    def prediction_hyperparameters(self) -> dict[TreeFamily, TreeSettings]:
        """Return the tree families and settings submitted to AutoGluon."""
        configured = (self.autoGluon and self.autoGluon.hyperparameters) or {}
        families = (
            get_args(TreeFamily)
            if self.finalPredictor == "weighted_ensemble"
            else configured or ("RF",)
        )
        trees = {family: configured.get(family, {}) for family in families}
        if "CAT" in trees:
            # A small training split can have a constant target even when the full pool varies.
            cat = trees["CAT"]
            defaults = {"allow_const_label": True}
            trees["CAT"] = (
                [defaults | variant for variant in cat]
                if isinstance(cat, list)
                else defaults | cat
            )
        return trees

    def validate_autogluon_installation(self) -> None:
        """Check that the driver can load the requested tree model libraries."""
        if sys.version_info >= (3, 14):
            raise ValueError(
                "PKH's AutoGluon predictor requires Python 3.11 through 3.13."
            )
        modules = {"GBM": "lightgbm", "CAT": "catboost", "XGB": "xgboost"}
        required = ["autogluon.tabular"]
        required.extend(
            modules[family]
            for family in self.prediction_hyperparameters()
            if family in modules
        )
        for module in required:
            try:
                importlib.import_module(module)
            except (ImportError, OSError, ValueError) as exc:
                raise ValueError(
                    f"PKH prediction could not load {module}. Install "
                    f"'ado-active-learning[autogluon]' and its native dependencies: {exc}"
                ) from exc

    @classmethod
    def example_configuration(cls) -> PKHParameters:
        """Return a minimal example for operator discovery."""
        return cls(targetOutput="objective", numberEntities=20)
