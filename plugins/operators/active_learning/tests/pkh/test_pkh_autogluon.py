# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Test the AutoGluon predictors used to complete PKH measurements."""

import numpy as np
import pandas as pd
import pytest
from active_learning.pkh.parameters import (
    ActiveLearningAutoGluonParameters,
    PKHParameters,
)


@pytest.mark.parametrize(
    ("mode", "expected_model"),
    [("random_forest", "RandomForest_FULL"), ("autogluon", "WeightedEnsemble_L2_FULL")],
)
def test_transient_tree_predictions(mode: str, expected_model: str) -> None:
    """AutoGluon can select a blend of two tree models for a quadratic target."""
    pytest.importorskip("autogluon.tabular")
    from active_learning.pkh.autogluon import predict_unmeasured

    hyperparameters = {"RF": {"n_estimators": 10}}
    if mode == "autogluon":
        hyperparameters["XT"] = {"n_estimators": 10}
    frame = pd.DataFrame({"x": np.arange(160)})
    labels = frame["x"].astype(float) ** 2
    labels.iloc[[15, 145]] = np.nan
    parameters = PKHParameters(
        targetOutput="self",
        numberEntities=20,
        finalPredictor=mode,
        autoGluon=ActiveLearningAutoGluonParameters(hyperparameters=hyperparameters),
    )
    values, model = predict_unmeasured(frame, labels, "self", parameters)
    assert model == expected_model
    assert values.shape == (2,)
    assert np.isfinite(values).all()
    assert values[0] < values[1]


def test_single_model_wins_validation_tie_on_constant_labels() -> None:
    """AutoGluon's regression preset can complete a constant target with one model."""
    pytest.importorskip("autogluon.tabular")
    from active_learning.pkh.autogluon import predict_unmeasured

    frame = pd.DataFrame(
        {"category": pd.Categorical(["None", "9007199254740993", "a", "b", "c"])}
    )
    labels = pd.Series([5.0, 5.0, 5.0, 5.0, np.nan])
    parameters = PKHParameters(
        targetOutput="y", numberEntities=4, finalPredictor="autogluon"
    )
    values, model = predict_unmeasured(frame, labels, "y", parameters)
    assert model.endswith("_FULL")
    assert not model.startswith("WeightedEnsemble")
    # The native regression preset names its RF candidate RandomForestMSE.
    assert model != "RandomForest_FULL"
    np.testing.assert_allclose(values, [5.0])


def test_automatic_selection_can_choose_a_configured_linear_model() -> None:
    """A linear predictor wins validation and extrapolates beyond measured values."""
    pytest.importorskip("autogluon.tabular")
    from active_learning.pkh.autogluon import predict_unmeasured

    frame = pd.DataFrame({"x": np.arange(101)})
    labels = 2.0 * frame["x"] + 3
    labels.iloc[-1] = np.nan
    parameters = PKHParameters(
        targetOutput="y",
        numberEntities=100,
        finalPredictor="autogluon",
        autoGluon={"hyperparameters": {"LR": {"C": 1e6}, "RF": {"n_estimators": 10}}},
    )
    values, model = predict_unmeasured(frame, labels, "y", parameters)
    assert model == "LinearModel_FULL"
    np.testing.assert_allclose(values, [203.0], atol=1e-3)
