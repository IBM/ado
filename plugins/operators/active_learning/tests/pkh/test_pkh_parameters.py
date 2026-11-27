# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Test PKH's configuration constraints and optional dependencies."""

from importlib.util import find_spec

import pytest
from active_learning.pkh.parameters import PKHParameters

BASE_PARAMETERS = {"targetOutput": "y", "numberEntities": 4}


def test_catboost_settings_survive_serialization() -> None:
    """PKH supplies the constant-target default while preserving an explicit override."""
    variants = [{"allow_const_label": False}, {"depth": 2}]
    parameters = PKHParameters(
        **BASE_PARAMETERS,
        finalPredictor="autogluon",
        autoGluon={"hyperparameters": {"CAT": variants}, "timeLimit": 15},
    )
    restored = PKHParameters.model_validate(parameters.model_dump(by_alias=True))
    assert restored == parameters
    assert restored.prediction_hyperparameters() == {
        "CAT": [{"allow_const_label": False}, {"depth": 2, "allow_const_label": True}]
    }


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"autoGluon": {"hyperparameters": {"CAT": {}}}}, "random_forest"),
        (
            {
                "finalPredictor": "autogluon",
                "autoGluon": {"hyperparameters": {"ENS_WEIGHTED": {}}},
            },
            "ENS_WEIGHTED",
        ),
        (
            {
                "autoGluon": {
                    "hyperparameters": {"RF": [{}, {"ag_args": {"model_type": "XT"}}]}
                }
            },
            r"RF.*ag_args\.model_type",
        ),
        (
            {
                "autoGluon": {
                    "hyperparameters": {
                        "RF": {"ag_args": {"problem_types": ["binary"]}}
                    }
                }
            },
            "RF.*regression",
        ),
        (
            {
                "autoGluon": {
                    "hyperparameters": {"RF": {"ag_args": {"valid_base": False}}}
                }
            },
            "RF.*base model",
        ),
    ],
)
def test_rejects_invalid_configuration(settings: dict, message: str) -> None:
    """Require base regressors suitable for the selected predictor mode."""
    with pytest.raises(ValueError, match=message):
        PKHParameters(**BASE_PARAMETERS, **settings)


def test_missing_backend_fails_preflight() -> None:
    """A missing optional backend is reported before PKH starts measurements."""
    pytest.importorskip("autogluon.tabular")
    if find_spec("interpret") is not None:
        pytest.skip("interpret is installed")
    parameters = PKHParameters(
        **BASE_PARAMETERS,
        finalPredictor="autogluon",
        autoGluon={"hyperparameters": {"EBM": {}}},
    )
    with pytest.raises(ValueError, match="interpret"):
        parameters.validate_autogluon_installation()
