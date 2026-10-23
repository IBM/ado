# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Final predictor training from a discovery space's stored measurements."""

from __future__ import annotations

import logging
import typing
from pathlib import Path

import pandas as pd

from active_learning.regression._predictor import _validate_final_predictor
from active_learning.regression._shared import (
    _finite_pool,
    _measured_labels,
    _resolve_target,
)
from active_learning.regression.model import RandomForestPredictor

if typing.TYPE_CHECKING:
    from active_learning.regression._shared import (
        _PredictiveOperatorParameters,
        _PredictiveParameters,
    )
    from ado.core.discoveryspace.space import DiscoverySpace


def _final_model_directory(output_directory: str) -> Path:
    """Resolve the final artifact path and reject existing files or directories."""

    base = Path(output_directory).expanduser().resolve()
    directory = base.with_name(f"{base.name}_finalized")
    if directory.exists() or directory.is_symlink():
        raise FileExistsError(
            f"final model directory already exists: {directory}; "
            "choose a different outputDirectory"
        )
    return directory


def _expected_training_count(
    discovery_space: DiscoverySpace,
    *,
    parameters: _PredictiveParameters,
    number_entities: int,
) -> int:
    """Count the target labels expected after sampling the remaining finite pool."""

    pool = _finite_pool(discovery_space, refresh=True)
    target = _resolve_target(discovery_space.measurementSpace, parameters.targetOutput)
    labels = _measured_labels(discovery_space, pool, target)
    return min(len(pool.entities), len(labels) + number_entities)


def _finalize_predictive_model(
    discovery_space: DiscoverySpace,
    *,
    parameters: _PredictiveOperatorParameters,
    directory: Path,
    minimum_training_count: int = 1,
) -> dict[str, typing.Any]:
    """Fit and save a predictor using all available finite-pool target labels.

    Args:
        discovery_space: Space containing committed experiment measurements.
        parameters: Target identifier, acquisition settings and final-predictor settings.
        directory: New directory for the finalized model artifact.
        minimum_training_count: Required number of labelled entities at completion.

    Returns:
        Artifact locations and training provenance for operation metadata.

    Raises:
        ValueError: If the space has insufficient usable target measurements.
    """

    pool = _finite_pool(discovery_space, refresh=True)
    target = _resolve_target(discovery_space.measurementSpace, parameters.targetOutput)
    labels = _measured_labels(discovery_space, pool, target)
    if not labels:
        raise ValueError(
            f"cannot fit a final predictor: no target measurements for {target.identifier!r}; "
            f"expected at least {minimum_training_count}, collected 0"
        )
    if len(labels) < minimum_training_count:
        raise ValueError(
            f"cannot fit a final predictor: expected at least {minimum_training_count} "
            f"target measurements for {target.identifier!r}, found {len(labels)}"
        )
    _validate_final_predictor(
        parameters,
        expected_training_count=minimum_training_count,
        collected_training_count=len(labels),
    )

    measured_indices = sorted(labels)
    measured_labels = pd.Series(
        {index: labels[index] for index in measured_indices}, dtype=float
    )
    if parameters.finalPredictor == "autogluon":
        from active_learning.regression.autogluon_model import AutoGluonPredictor

        predictor = AutoGluonPredictor.fit(
            pool.feature_frame,
            measured_labels,
            target_output=parameters.targetOutput,
            categorical_columns=pool.categorical_columns,
            fit_parameters=parameters.autoGluon.fit_parameters(),
            random_state=parameters.seed,
        )
        fit_count = predictor.n_fit_samples_
        validation_count = predictor.n_validation_samples_
    else:
        predictor = RandomForestPredictor.fit(
            pool.feature_frame,
            measured_labels,
            target_output=parameters.targetOutput,
            categorical_columns=pool.categorical_columns,
            forest_parameters={
                "n_estimators": parameters.nEstimators,
                "min_samples_leaf": parameters.minSamplesLeaf,
                "max_features": parameters.maxFeatures,
                "criterion": parameters.criterion,
                "random_state": parameters.seed,
                "n_jobs": parameters.nJobs,
            },
        )
        fit_count = len(labels)
        validation_count = 0
    directory = predictor.save(directory)
    logging.getLogger(__name__).info(
        "Final model supplied with %d measured entities and saved to %s",
        len(labels),
        directory,
    )
    return {
        "predictorType": parameters.finalPredictor,
        "directory": str(directory),
        "modelPath": str(directory / "model.pkl"),
        "metadataPath": str(directory / "metadata.json"),
        "trainingEntityCount": len(labels),
        "fitEntityCount": fit_count,
        "validationEntityCount": validation_count,
        "trainingEntityIdentifiers": [
            pool.entities[index].identifier for index in measured_indices
        ],
        "targetOutput": parameters.targetOutput,
        "observedProperty": target.identifier,
    }
