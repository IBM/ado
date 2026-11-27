# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Model selection and refitting for PKH's final predictions."""

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
from active_learning.pkh.parameters import PKHParameters

from ado.core.discoveryspace.space import DiscoverySpace
from ado.schema.domain import PropertyDomain, VariableTypeEnum
from ado.schema.property import ConstitutiveProperty

pytest.importorskip("autogluon.tabular")

from active_learning.pkh.autogluon import (  # noqa: E402
    predict_unmeasured,
    validate_knn_prediction,
)
from active_learning.pkh.data import (  # noqa: E402
    candidate_coordinates,
    candidate_entities,
    feature_frame,
)
from active_learning.pkh.operator import pkh  # noqa: E402


def test_winning_ensemble_refits_on_all_measured_rows() -> None:
    """PKH fills missing rows in grid order using the best-scoring ensemble."""
    frame = pd.DataFrame({"x": range(160)})
    labels = frame["x"].astype(float) ** 2
    labels.iloc[[15, 145]] = np.nan
    parameters = PKHParameters(
        targetOutput="y",
        numberEntities=2,
        finalPredictor="autogluon",
        autoGluon={
            "hyperparameters": {"RF": {"n_estimators": 10}, "XT": {"n_estimators": 10}}
        },
    )
    values, metadata = predict_unmeasured(frame, labels, "y", parameters)
    assert metadata["predictor"].startswith("WeightedEnsemble")
    assert int(metadata["predictorTrainingRows"]) == 158
    low, high = values
    assert 0 <= low < high <= labels.max()


def test_missing_ensemble_leaves_scored_models_eligible() -> None:
    """Predict with a scored regressor when the variants have disjoint validation rows."""
    frame = pd.DataFrame({"x": range(13)})
    labels = frame["x"].astype(float)
    labels.iloc[-1] = np.nan
    variants = {}
    for family, held_out, remaining in (
        ("RF", list(range(6)), list(range(6, 12))),
        ("XT", list(range(6, 12)), list(range(6))),
    ):
        variants[family] = {
            "n_estimators": 5,
            "ag_args_ensemble": {
                "num_folds": 2,
                "max_sets": 1,
                "fold_fitting_strategy": "sequential_local",
                "custom_splits": [(remaining, held_out[:3]), (remaining, held_out[3:])],
            },
        }
    parameters = PKHParameters(
        targetOutput="y",
        numberEntities=1,
        finalPredictor="autogluon",
        autoGluon={"presets": "best_quality", "hyperparameters": variants},
    )
    values, metadata = predict_unmeasured(frame, labels, "y", parameters)
    assert metadata["predictor"] in {
        "RandomForest_BAG_L1_FULL",
        "ExtraTrees_BAG_L1_FULL",
    }
    [value] = values
    assert 0 <= value <= labels.max()


def test_validation_tie_uses_stable_model_order() -> None:
    """Prefer a single model and break base-model ties by name."""
    frame = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0, 1.4]})
    labels = pd.Series([0.0, 10.0, 0.0, 0.0, np.nan])
    first_model = {"n_estimators": 250, "max_depth": 1}
    parameters = PKHParameters(
        targetOutput="y",
        numberEntities=4,
        finalPredictor="autogluon",
        autoGluon={
            "hyperparameters": {
                "RF": [
                    first_model,
                    {"n_estimators": 10, "max_depth": 9},
                ]
            }
        },
    )
    values, metadata = predict_unmeasured(frame, labels, "y", parameters)
    assert metadata["predictor"] == "RandomForest_FULL"
    reference = PKHParameters(
        targetOutput="y",
        numberEntities=4,
        autoGluon={"hyperparameters": {"RF": first_model}},
    )
    expected, _ = predict_unmeasured(frame, labels, "y", reference)
    np.testing.assert_allclose(values, expected)


@pytest.mark.parametrize(
    ("preset", "coordinate_values", "label_count", "variant", "message"),
    [
        ("medium_quality", [0, 1, 2, 3, 4], 4, {}, "2 available training rows"),
        ("best_quality", [0, 1, 2, 3, 4, 5], 5, {}, "4 available training rows"),
        (
            "best_quality",
            [0, 1, 2, 3, 4, 5],
            5,
            {
                "ag_args_ensemble": {
                    "use_child_oof": True,
                    "custom_splits": None,
                    "num_folds": 0,
                }
            },
            "4 available training rows",
        ),
        (
            "best_quality",
            [0, 1, 2, 3, 4, 5],
            5,
            {"ag_args_ensemble": {"use_child_oof": False}},
            "KNN uses the preset's default validation",
        ),
        ("medium_quality", list("abcdefghijk"), 10, {}, "numeric feature"),
    ],
)
def test_knn_preflight_requires_enough_neighbors_and_numeric_features(
    make_pkh_space: Callable[[list[ConstitutiveProperty]], DiscoverySpace],
    preset: str,
    coordinate_values: list[int] | list[str],
    label_count: int,
    variant: dict,
    message: str,
) -> None:
    """Reject infeasible KNN settings through PKH before dispatching any measurement."""
    space = make_pkh_space(
        [
            ConstitutiveProperty(
                identifier="x", propertyDomain=PropertyDomain(values=coordinate_values)
            )
        ]
    )
    parameters = PKHParameters(
        targetOutput="self",
        numberEntities=label_count,
        finalPredictor="autogluon",
        autoGluon={"presets": preset, "hyperparameters": {"KNN": variant}},
    )
    with pytest.raises(ValueError, match=message):
        pkh.__wrapped__(space, parameters=parameters)


def test_knn_can_use_binary_coordinates(
    make_pkh_space: Callable[[list[ConstitutiveProperty]], DiscoverySpace],
) -> None:
    """Fit with a binary feature and categorical coordinates."""
    space = make_pkh_space(
        [
            ConstitutiveProperty(
                identifier="group",
                propertyDomain=PropertyDomain(values=["a", "b", "c", "d"]),
            ),
            ConstitutiveProperty(
                identifier="flag",
                propertyDomain=PropertyDomain(
                    variableType=VariableTypeEnum.BINARY_VARIABLE_TYPE
                ),
            ),
        ]
    )
    parameters = PKHParameters(
        targetOutput="y",
        numberEntities=6,
        finalPredictor="autogluon",
        autoGluon={"hyperparameters": {"KNN": {"n_neighbors": 2}}},
    )
    frame = feature_frame(space, candidate_coordinates(candidate_entities(space)))
    labels = frame["flag"].astype(float).mask(frame["group"] == 3)
    validate_knn_prediction(parameters, parameters.numberEntities, space.entitySpace)
    values, _ = predict_unmeasured(frame, labels, "y", parameters)
    np.testing.assert_allclose(values, frame.loc[labels.isna(), "flag"].to_numpy())


def test_bagged_refit_reports_retained_training_rows(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Bagging accepts the largest PKH seed and reports the copied fold's size."""
    frame = pd.DataFrame({"x": range(40)})
    labels = frame["x"].astype(float)
    labels.iloc[-1] = np.nan
    parameters = PKHParameters(
        targetOutput="y",
        numberEntities=1,
        seed=2**32 - 1,
        finalPredictor="autogluon",
        autoGluon={
            "presets": "best_quality",
            "hyperparameters": {
                "DUMMY": {
                    "ag_args_ensemble": {"fold_fitting_strategy": "sequential_local"}
                }
            },
        },
    )
    _, metadata = predict_unmeasured(frame, labels, "y", parameters)
    rows = int(metadata["predictorTrainingRows"])
    assert 0 < rows < 39
    assert f"{rows} of 39 measured rows" in caplog.text
