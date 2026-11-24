# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Saved predictors for the active-learning operators."""

import pickle
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest
from active_learning.flora.operator import _finalize_model as finalize_flora_model
from active_learning.flora.parameters import FLORAParameters
from active_learning.flora.sampler import FLORASampleSelector
from active_learning.operator import PredictorExport
from active_learning.parameters import ActiveLearningParameters
from active_learning.pkh.operator import _finalize_model as finalize_pkh_model
from active_learning.pkh.parameters import PKHParameters
from active_learning.pkh.sampler import PKHSampleSelector
from active_learning.sampler import ForestSampleSelector

from ado.core.discoveryspace.space import DiscoverySpace
from ado.schema.observed_property import ObservedPropertyValue
from ado.schema.result import ValidMeasurementResult


@pytest.mark.parametrize(
    ("parameter_type", "sampler", "finalize_model", "first_label"),
    [
        pytest.param(PKHParameters, PKHSampleSelector, finalize_pkh_model, 0, id="pkh"),
        pytest.param(
            FLORAParameters, FLORASampleSelector, finalize_flora_model, 2, id="flora"
        ),
    ],
)
@pytest.mark.parametrize("categorical_space", [([0, 1, 2], [0, 1, 2])], indirect=True)
@pytest.mark.parametrize("backend", ["random_forest", "autogluon"])
def test_saved_predictor_reproduces_categorical_predictions(
    categorical_space: DiscoverySpace,
    tmp_path: Path,
    backend: str,
    parameter_type: type[ActiveLearningParameters],
    sampler: type[ForestSampleSelector],
    finalize_model: Callable[[PredictorExport, ActiveLearningParameters], float | None],
    first_label: float,
) -> None:
    """Reload predictors and preserve the operator's repeated-measurement labels."""
    target = categorical_space.measurementSpace.observedProperties[0]
    repeated = categorical_space.entitySpace.entity_for_point({"kind": 1, "count": 0})
    categorical_space.sample_store.add_measurement_results(
        [
            ValidMeasurementResult(
                entityIdentifier=repeated.identifier,
                measurements=[ObservedPropertyValue(property=target, value=4.0)],
            )
        ],
        skip_relationship_to_request=True,
    )
    settings = None
    if backend == "autogluon":
        pytest.importorskip("autogluon.tabular")
        settings = {"timeLimit": 10, "hyperparameters": {"RF": {"n_estimators": 3}}}
    parameters = parameter_type(
        targetOutput=target.targetProperty.identifier,
        numberEntities=1,
        nEstimators=3,
        finalPredictor=backend,
        autoGluon=settings,
    )
    sampled = categorical_space.entitySpace.entity_for_point({"kind": "b", "count": 1})
    export = PredictorExport(
        categorical_space, target, parameters, tmp_path, {sampled.identifier}, sampler
    )
    finalize_model(export, parameters)
    table = pd.read_csv(tmp_path / "combined_predictions.csv")
    expected_labels = [first_label, 1, 2, 0, 1, 2]
    assert table.loc[~table.is_predicted, "self"].tolist() == expected_labels
    unmeasured = table.loc[table.is_predicted]
    assert unmeasured.kind.tolist() == ["c"] * 3 + ["d"] * 3
    if backend == "autogluon":
        from autogluon.tabular import TabularPredictor

        model = TabularPredictor.load(str(tmp_path / "autogluon"))
        assert model.problem_type == "regression"
    else:
        with (tmp_path / "model.pkl").open("rb") as model_file:
            model = pickle.load(model_file)  # noqa: S301 - reload this test's model
    assert model.predict(unmeasured[["kind", "count"]]).tolist() == pytest.approx(
        unmeasured["self"].tolist()
    )
