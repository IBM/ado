# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Configuration and loading of finalized regression predictors."""

from __future__ import annotations

import json
import typing
from pathlib import Path
from typing import Annotated, Literal

import pydantic

if typing.TYPE_CHECKING:
    import os

    from active_learning.regression.autogluon_model import AutoGluonPredictor
    from active_learning.regression.model import RandomForestPredictor


class AutoGluonParameters(pydantic.BaseModel):
    """Settings for fitting an AutoGluon predictor on collected measurements."""

    model_config = pydantic.ConfigDict(extra="forbid")

    presets: Annotated[
        Literal["medium_quality", "good_quality", "high_quality", "best_quality"],
        pydantic.Field(description="AutoGluon training preset."),
    ] = "medium_quality"
    timeLimit: Annotated[
        float,
        pydantic.Field(
            gt=0, allow_inf_nan=False, description="Training time limit in seconds."
        ),
    ] = 60.0
    numCpus: Annotated[
        int,
        pydantic.Field(
            gt=0, strict=True, description="CPUs available to the final AutoGluon fit."
        ),
    ] = 1
    hyperparameters: Annotated[
        dict[str, dict[str, typing.Any] | list[dict[str, typing.Any]]] | None,
        pydantic.Field(description="Optional AutoGluon model-family hyperparameters."),
    ] = None

    def fit_parameters(self) -> dict[str, typing.Any]:
        """Return arguments for AutoGluon's fit method."""

        parameters = {
            "presets": self.presets,
            "time_limit": self.timeLimit,
            "num_cpus": self.numCpus,
        }
        if self.hyperparameters is not None:
            parameters["hyperparameters"] = self.hyperparameters
        return parameters


class _FinalPredictorParameters(pydantic.BaseModel):
    """Operator settings for the predictor saved after acquisition."""

    model_config = pydantic.ConfigDict(extra="forbid")

    finalPredictor: Annotated[
        Literal["random_forest", "autogluon"],
        pydantic.Field(description="Final predictor fitted on collected measurements."),
    ] = "random_forest"
    autoGluon: Annotated[
        AutoGluonParameters,
        pydantic.Field(
            default_factory=AutoGluonParameters,
            description="Final fit settings when finalPredictor is autogluon.",
        ),
    ]


def _validate_final_predictor(
    parameters: _FinalPredictorParameters,
    *,
    expected_training_count: int,
    collected_training_count: int | None = None,
) -> None:
    """Validate backend availability and expected or collected target counts.

    Args:
        parameters: Final-predictor backend and training settings.
        expected_training_count: Projected count before measurement, or required
            minimum when a collected count is supplied.
        collected_training_count: Available target count after measurement, if known.
    """

    if parameters.finalPredictor == "autogluon":
        training_count = (
            expected_training_count
            if collected_training_count is None
            else collected_training_count
        )
        if training_count < 4:
            if collected_training_count is None:
                raise ValueError(
                    "AutoGluon requires at least 4 labelled entities, "
                    f"but this campaign would provide only {expected_training_count} "
                    "in total."
                )
            raise ValueError(
                "Cannot train the final AutoGluon model: at least 4 labelled entities "
                f"are required, but only {collected_training_count} "
                f"{'is' if collected_training_count == 1 else 'are'} available."
            )
        from active_learning.regression.autogluon_model import _require_autogluon

        _require_autogluon()


def load_predictor(
    directory: str | os.PathLike[str],
) -> RandomForestPredictor | AutoGluonPredictor:
    """Load a saved predictor and its feature schema from a trusted directory.

    Artifacts use pickle and can execute code when loaded. Use trusted files
    and the same library versions used for fitting.

    Args:
        directory: Finalized model directory containing metadata.json.

    Returns:
        The fitted predictor with a predict method accepting raw feature rows.

    Raises:
        ValueError: If the saved backend is unsupported.
    """

    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    backend = metadata.get("backend", "random_forest")
    if backend == "random_forest":
        from active_learning.regression.model import RandomForestPredictor

        return RandomForestPredictor.load(directory)
    if backend == "autogluon":
        from active_learning.regression.autogluon_model import AutoGluonPredictor

        return AutoGluonPredictor.load(directory)
    raise ValueError(f"unsupported final predictor backend: {backend!r}")
