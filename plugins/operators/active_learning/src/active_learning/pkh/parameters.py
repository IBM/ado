# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""PKH sampling and prediction configuration."""

from __future__ import annotations

import importlib
import sys
from typing import Annotated, Any, Literal

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
    model_validator,
)

from ado.core.operation.config import GenericOperatorParameters

TREE_FAMILIES = ("RF", "XT", "XGB", "CAT", "GBM")
ModelSettings = dict[str, Any] | Annotated[list[dict[str, Any]], Field(min_length=1)]


class ActiveLearningAutoGluonParameters(BaseModel):
    """Fit arguments for AutoGluon's final predictor."""

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
        dict[str, ModelSettings] | None,
        Field(min_length=1, description="AutoGluon hyperparameters by model family."),
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
            description=(
                "AutoGluon fits the final predictor. random_forest fits a Random Forest. "
                "autogluon enables automatic model selection using the chosen preset or "
                "the families in autoGluon.hyperparameters. weighted_ensemble compares "
                "five tree families. Both selection modes compare single models and "
                "their weighted ensemble by validation RMSE."
            )
        ),
    ] = "random_forest"
    autoGluon: Annotated[
        ActiveLearningAutoGluonParameters | None,
        Field(description="AutoGluon fit options for completing the prediction table."),
    ] = None

    @model_validator(mode="after")
    def validate_predictor_configuration(self) -> PKHParameters:
        """Check model-family overrides against the selected predictor mode."""
        configured = self.autoGluon and self.autoGluon.hyperparameters
        if not configured:
            return self
        if self.finalPredictor == "random_forest" and configured.keys() != {"RF"}:
            raise ValueError("random_forest accepts only RF hyperparameters.")
        if (
            self.finalPredictor == "weighted_ensemble"
            and configured.keys() - TREE_FAMILIES
        ):
            raise ValueError(
                f"weighted_ensemble accepts only the following tree families: {', '.join(TREE_FAMILIES)}."
            )
        return self

    def prediction_hyperparameters(self) -> dict[str, ModelSettings] | None:
        """Return explicit model settings, or None to use the preset's models."""
        configured = self.autoGluon and self.autoGluon.hyperparameters
        if self.finalPredictor == "autogluon" and configured is None:
            return None
        models = dict(configured or {"RF": {}})
        if self.finalPredictor == "weighted_ensemble":
            models = {family: models.get(family, {}) for family in TREE_FAMILIES}
        if "CAT" in models:
            # A small training split can have a constant target even when the full pool varies.
            cat = models["CAT"]
            defaults = {"allow_const_label": True}
            models["CAT"] = (
                [defaults | variant for variant in cat]
                if isinstance(cat, list)
                else defaults | cat
            )
        return models

    def validate_autogluon_installation(self) -> None:
        """Validate AutoGluon family names and dependencies before sampling."""
        if sys.version_info >= (3, 14):
            raise ValueError(
                "PKH's AutoGluon predictor requires Python 3.11 through 3.13."
            )
        plugin_extra = "ado-active-learning[autogluon]"
        dependencies = {
            "GBM": ("lightgbm", plugin_extra),
            "CAT": ("catboost", plugin_extra),
            "XGB": ("xgboost", plugin_extra),
            "FASTAI": ("fastai.tabular.all", "autogluon.tabular[fastai]"),
            "NN_TORCH": ("torch", "torch"),
        }
        families = self.prediction_hyperparameters() or {}
        required = [("autogluon.tabular", "ado-active-learning")]
        required.extend(
            dependencies[family] for family in families if family in dependencies
        )
        for module, package in required:
            try:
                importlib.import_module(module)
            except (ImportError, OSError, ValueError) as exc:
                raise ValueError(
                    f"PKH prediction could not load {module}. Install '{package}' "
                    f"and its native dependencies: {exc}"
                ) from exc

        if self.finalPredictor == "autogluon" and families:
            from autogluon.tabular.registry import ag_model_registry

            base_families = set(ag_model_registry.keys) - {
                "ENS_WEIGHTED",
                "SIMPLE_ENS_WEIGHTED",
            }
            if invalid := families.keys() - base_families:
                raise ValueError(
                    f"Unsupported AutoGluon base model families: {sorted(invalid)}."
                )

    @classmethod
    def example_configuration(cls) -> PKHParameters:
        """Return a minimal example for operator discovery."""
        return cls(targetOutput="objective", numberEntities=20)
