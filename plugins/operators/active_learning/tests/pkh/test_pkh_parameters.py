# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Tests for PKH-specific predictor configuration."""

import pytest
from active_learning.pkh.parameters import (
    ActiveLearningAutoGluonParameters,
    PKHParameters,
)


def test_tree_predictor_configuration() -> None:
    """Weighted prediction retains tree settings and rejects unrelated families."""
    parameters = PKHParameters(
        targetOutput="pressure",
        numberEntities=20,
        finalPredictor="weighted_ensemble",
        autoGluon=ActiveLearningAutoGluonParameters(
            timeLimit=15, hyperparameters={"XT": {"n_estimators": 5}}
        ),
    )
    restored = PKHParameters.model_validate(parameters.model_dump(by_alias=True))
    assert restored == parameters
    assert restored.prediction_hyperparameters() == {
        "RF": {},
        "XT": {"n_estimators": 5},
        "XGB": {},
        "CAT": {"allow_const_label": True},
        "GBM": {},
    }
    explicit = PKHParameters(
        targetOutput="pressure",
        numberEntities=20,
        finalPredictor="autogluon",
        autoGluon={
            "hyperparameters": {"CAT": [{"allow_const_label": False}, {"depth": 2}]}
        },
    )
    assert explicit.prediction_hyperparameters()["CAT"] == [
        {"allow_const_label": False},
        {"allow_const_label": True, "depth": 2},
    ]
    with pytest.raises(ValueError, match="weighted_ensemble"):
        PKHParameters(
            targetOutput="pressure",
            numberEntities=20,
            finalPredictor="weighted_ensemble",
            autoGluon={"hyperparameters": {"KNN": {}}},
        )
    with pytest.raises(ValueError, match="random_forest"):
        PKHParameters(
            targetOutput="pressure",
            numberEntities=20,
            autoGluon=ActiveLearningAutoGluonParameters(hyperparameters={"CAT": {}}),
        )


def test_automatic_model_selection_with_preset_or_explicit_families() -> None:
    """AutoGluon takes candidate models from the preset or the configured families."""
    parameters = PKHParameters(
        targetOutput="pressure",
        numberEntities=20,
        finalPredictor="autogluon",
        autoGluon={"presets": "good_quality", "timeLimit": 30},
    )
    restored = PKHParameters.model_validate(parameters.model_dump(by_alias=True))
    assert restored == parameters
    assert restored.prediction_hyperparameters() is None
    explicit = {"LR": {}, "KNN": {"weights": "distance"}, "RF": {}}
    custom = PKHParameters.model_validate(
        {**parameters.model_dump(), "autoGluon": {"hyperparameters": explicit}}
    )
    assert custom.prediction_hyperparameters() == explicit


@pytest.mark.parametrize(
    "family", ["MisspelledModel", "ENS_WEIGHTED", "SIMPLE_ENS_WEIGHTED"]
)
def test_explicit_autogluon_families_must_be_base_models(family: str) -> None:
    """Reject unknown and ensemble-only families before the experiment starts."""
    pytest.importorskip("autogluon.tabular")
    parameters = PKHParameters(
        targetOutput="pressure",
        numberEntities=20,
        finalPredictor="autogluon",
        autoGluon={"hyperparameters": {family: {}}},
    )
    with pytest.raises(ValueError, match=family):
        parameters.validate_autogluon_installation()
