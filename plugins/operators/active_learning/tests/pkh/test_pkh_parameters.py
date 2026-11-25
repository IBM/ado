# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Tests for PKH-specific predictor configuration."""

from importlib.util import find_spec

import pytest
from active_learning.pkh.parameters import (
    ActiveLearningAutoGluonParameters,
    PKHParameters,
)


def test_random_forest_configuration() -> None:
    """The default predictor accepts only Random Forest settings."""
    parameters = PKHParameters(
        targetOutput="pressure",
        numberEntities=20,
        autoGluon=ActiveLearningAutoGluonParameters(
            timeLimit=15, hyperparameters={"RF": {"n_estimators": 5}}
        ),
    )
    restored = PKHParameters.model_validate(parameters.model_dump(by_alias=True))
    assert restored == parameters
    assert restored.prediction_hyperparameters() == {"RF": {"n_estimators": 5}}
    with pytest.raises(ValueError, match="random_forest"):
        PKHParameters(
            targetOutput="pressure",
            numberEntities=20,
            autoGluon=ActiveLearningAutoGluonParameters(hyperparameters={"CAT": {}}),
        )


def test_rejects_standalone_ensemble_mode() -> None:
    """Configurations select between Random Forest and AutoGluon."""
    with pytest.raises(ValueError, match="finalPredictor"):
        PKHParameters.model_validate(
            {
                "targetOutput": "pressure",
                "numberEntities": 20,
                "finalPredictor": "weighted_ensemble",
            }
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
    explicit = {"LR": {}, "CAT": [{"allow_const_label": False}, {"depth": 2}]}
    custom = PKHParameters.model_validate(
        {**parameters.model_dump(), "autoGluon": {"hyperparameters": explicit}}
    )
    assert custom.prediction_hyperparameters() == {
        "LR": {},
        "CAT": [{"allow_const_label": False}, {"depth": 2, "allow_const_label": True}],
    }


@pytest.mark.parametrize(
    "family",
    [
        "MisspelledModel",
        "ENS_WEIGHTED",
        "SIMPLE_ENS_WEIGHTED",
        "IM_BOOSTEDRULES",
        "IM_GREEDYTREE",
        "AG_IMAGE_NN",
    ],
)
def test_explicit_autogluon_families_must_be_base_models(family: str) -> None:
    """Reject families that cannot supply a base regressor when validating the operation."""
    with pytest.raises(ValueError, match=family):
        PKHParameters.model_validate(
            {
                "targetOutput": "pressure",
                "numberEntities": 20,
                "finalPredictor": "autogluon",
                "autoGluon": {"hyperparameters": {family: {}}},
            }
        )


@pytest.mark.parametrize(
    ("family", "module"), [("IM_RULEFIT", "imodels"), ("EBM", "interpret")]
)
def test_missing_explicit_model_backend_fails_preflight(
    family: str, module: str
) -> None:
    """An unavailable optional backend fails before the measurement walk."""
    pytest.importorskip("autogluon.tabular")
    if find_spec(module) is not None:
        pytest.skip(f"{module} is installed")
    parameters = PKHParameters(
        targetOutput="pressure",
        numberEntities=20,
        finalPredictor="autogluon",
        autoGluon={"hyperparameters": {family: {}}},
    )
    with pytest.raises(ValueError, match=module):
        parameters.validate_autogluon_installation()
