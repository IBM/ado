# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Test the AutoGluon tree models used to complete PKH measurements."""

import numpy as np
import pandas as pd
import pytest
from active_learning.pkh.parameters import (
    ActiveLearningAutoGluonParameters,
    PKHParameters,
)


@pytest.mark.parametrize("weighted", [False, True], ids=["rf", "ensemble-wins"])
def test_transient_tree_predictions(weighted: bool) -> None:
    """A weighted blend improves on the individual trees for this quadratic target."""
    pytest.importorskip("autogluon.tabular")
    from active_learning.pkh.autogluon import predict_unmeasured

    hyperparameters = {"RF": {"n_estimators": 10}}
    if weighted:
        for dependency in ("lightgbm", "catboost", "xgboost"):
            pytest.importorskip(dependency)
        hyperparameters.update(
            XT={"n_estimators": 10},
            GBM={"num_boost_round": 10},
            CAT={"iterations": 10},
            XGB={"n_estimators": 10},
        )
    frame = pd.DataFrame({"x": np.arange(160)})
    labels = frame["x"].astype(float) ** 2
    labels.iloc[[15, 145]] = np.nan
    parameters = PKHParameters(
        targetOutput="self",
        numberEntities=20,
        finalPredictor="weighted_ensemble" if weighted else "random_forest",
        autoGluon=ActiveLearningAutoGluonParameters(hyperparameters=hyperparameters),
    )
    values, model = predict_unmeasured(frame, labels, "self", parameters)
    assert model.removesuffix("_FULL") == (
        "WeightedEnsemble_L2" if weighted else "RandomForest"
    )
    assert values.shape == (2,)
    assert np.isfinite(values).all()
    assert values[0] < values[1]


def test_single_tree_model_wins_validation_tie_on_constant_labels() -> None:
    """A single model can match the ensemble perfectly on rare categories."""
    for dependency in ("autogluon.tabular", "lightgbm", "catboost", "xgboost"):
        pytest.importorskip(dependency)
    from active_learning.pkh.autogluon import predict_unmeasured

    frame = pd.DataFrame(
        {"category": pd.Categorical(["None", "9007199254740993", "a", "b", "c"])}
    )
    labels = pd.Series([5.0, 5.0, 5.0, 5.0, np.nan])
    parameters = PKHParameters(
        targetOutput="y", numberEntities=4, finalPredictor="weighted_ensemble"
    )
    values, model = predict_unmeasured(frame, labels, "y", parameters)
    assert model in {
        "RandomForest_FULL",
        "ExtraTrees_FULL",
        "LightGBM_FULL",
        "CatBoost_FULL",
        "XGBoost_FULL",
    }
    np.testing.assert_allclose(values, [5.0])
