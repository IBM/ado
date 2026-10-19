# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT
# ruff: noqa: S101

"""Real CLI coverage for final predictive models and persisted training metadata."""

import importlib
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from ado.metastore.project import ProjectContext
from ado.utilities.location import SQLiteStoreConfiguration
from ado.utilities.output import pydantic_model_as_yaml


def _run_cli(
    repository: Path,
    config_directory: Path,
    context_path: Path,
    *arguments: str,
) -> None:
    """Execute the installed CLI with isolated configuration and capture failures."""

    environment = os.environ.copy()
    environment["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"
    result = subprocess.run(  # noqa: S603
        [
            str(Path(sys.executable).with_name("ado")),
            "--override-ado-app-dir",
            str(config_directory),
            "--context",
            str(context_path),
            *arguments,
        ],
        cwd=repository,
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, (
        f"ado {' '.join(arguments)} failed with exit code {result.returncode}\n"
        f"{result.stdout}\n{result.stderr}"
    )


@pytest.mark.timeout(300)
@pytest.mark.parametrize("algorithm", ["pkh", "flora"])
@pytest.mark.parametrize(
    ("budget", "pilot_strategy", "final_predictor"),
    [
        (20, "random", "random_forest"),
        (3, "random", "random_forest"),
        (7, "sobol", "random_forest"),
        (20, "random", "autogluon"),
    ],
    ids=["adaptive", "pilot-only", "sobol-adaptive", "autogluon-adaptive"],
)
def test_cli_saves_final_model_trained_on_every_measurement(
    tmp_path: Path,
    algorithm: str,
    budget: int,
    pilot_strategy: str,
    final_predictor: str,
) -> None:
    """Reload a final model fitted on the full campaign, including its last label."""

    repository = Path(__file__).resolve().parents[5]
    config_directory = tmp_path / "config"
    config_directory.mkdir()
    context = ProjectContext(
        project="final_model_test",
        metadataStore=SQLiteStoreConfiguration(
            database="final_model_test",
            path=str(tmp_path / "final_model_test.db"),
        ),
    )
    context_path = tmp_path / "context.yaml"
    context_path.write_text(pydantic_model_as_yaml(context))
    output_base = tmp_path / f"{algorithm}_models"
    parameters = {
        "numberEntities": budget,
        "targetOutput": "density",
        "pilotSize": 5,
        "pilotStrategy": pilot_strategy,
        "nEstimators": 11,
        "minSamplesLeaf": 2,
        "maxFeatures": "sqrt",
        "seed": 42,
        "nJobs": 1,
        "outputDirectory": str(output_base),
    }
    if algorithm == "pkh":
        parameters["epochLength"] = 10
    if final_predictor == "autogluon":
        pytest.importorskip("autogluon.tabular")
        parameters["finalPredictor"] = "autogluon"
        parameters["autoGluon"] = {
            "presets": "medium_quality",
            "timeLimit": 60,
            "numCpus": 1,
            "hyperparameters": {"RF": {"n_estimators": 8}},
        }
    parameter_module = importlib.import_module(
        f"active_learning.regression.{algorithm}.parameters"
    )
    parameter_class = getattr(
        parameter_module, f"{algorithm.upper()}OperatorParameters"
    )
    parameter_class.model_validate(parameters)
    operation_path = tmp_path / "operation.yaml"
    operation_path.write_text(
        yaml.safe_dump(
            {
                "spaces": ["replace-with-a-space-identifier"],
                "operation": {
                    "module": {
                        "operationType": "characterize",
                        "operatorName": algorithm,
                    },
                    "parameters": parameters,
                },
            }
        )
    )
    example_space = (
        repository / "plugins/operators/active_learning/examples/discoveryspace.yaml"
    )
    _run_cli(
        repository,
        config_directory,
        context_path,
        "create",
        "space",
        "-f",
        str(example_space),
        "--new-sample-store",
    )
    _run_cli(
        repository,
        config_directory,
        context_path,
        "create",
        "operation",
        "-f",
        str(operation_path),
        "--use-latest",
        "space",
    )

    from active_learning.regression import load_predictor

    final_directory = Path(f"{output_base}_finalized")
    metadata_path = final_directory / "metadata.json"
    assert metadata_path.is_file()
    assert json.loads(metadata_path.read_text())
    predictor = load_predictor(final_directory)
    assert predictor.n_training_samples_ == budget
    if final_predictor == "random_forest":
        assert (final_directory / "model.pkl").is_file()
        assert len(predictor.forest_.estimators_) == parameters["nEstimators"]
        assert all(
            estimator.tree_.weighted_n_node_samples[0] == budget
            for estimator in predictor.forest_.estimators_
        )
    else:
        assert predictor.n_fit_samples_ > 0
        assert predictor.n_validation_samples_ > 0
        assert predictor.n_fit_samples_ + predictor.n_validation_samples_ == budget

    measurements_path = tmp_path / "measurements.csv"
    _run_cli(
        repository,
        config_directory,
        context_path,
        "show",
        "measurements",
        "space",
        "--use-latest",
        "-o",
        "csv",
        "--output-file",
        str(measurements_path),
    )
    measurements = pd.read_csv(measurements_path)
    assert len(measurements) == measurements["identifier"].nunique() == budget
    assert np.isfinite(measurements["density"]).all()
    space_configuration = yaml.safe_load(example_space.read_text())
    properties = space_configuration["entitySpace"]
    raw_pool = pd.DataFrame(
        itertools.product(*(prop["propertyDomain"]["values"] for prop in properties)),
        columns=[prop["identifier"] for prop in properties],
    )
    raw_pool.index.name = "configuration"
    assert len(raw_pool) == 100
    measured_points = set(
        measurements[["mass", "volume"]].itertuples(index=False, name=None)
    )
    unmeasured = raw_pool.loc[
        [
            point not in measured_points
            for point in raw_pool[["mass", "volume"]].itertuples(index=False, name=None)
        ],
        ["volume", "mass"],
    ]
    assert len(unmeasured) == 100 - budget
    predictions = predictor.predict(unmeasured)
    assert isinstance(predictions, pd.Series)
    pd.testing.assert_index_equal(predictions.index, unmeasured.index)
    assert len(predictions) == 100 - budget
    assert np.isfinite(predictions.to_numpy()).all()
    pd.testing.assert_series_equal(
        predictions,
        load_predictor(final_directory).predict(unmeasured),
    )

    operations_path = tmp_path / "operations.json"
    _run_cli(
        repository,
        config_directory,
        context_path,
        "get",
        "operations",
        "-o",
        "json",
        "--output-file",
        str(operations_path),
    )
    outer_operations = [
        operation
        for operation in json.loads(operations_path.read_text())
        if operation["config"]["operation"]["module"]["operatorName"] == algorithm
    ]
    assert len(outer_operations) == 1
    final_model = outer_operations[0]["metadata"]["finalModel"]
    assert final_model["directory"] == str(final_directory)
    model_path = Path(final_model["modelPath"])
    assert model_path.is_relative_to(final_directory)
    assert model_path.is_file()
    assert final_model["predictorType"] == final_predictor
    assert final_model["metadataPath"] == str(metadata_path)
    assert final_model["trainingEntityCount"] == budget
    assert len(final_model["trainingEntityIdentifiers"]) == budget
    assert set(final_model["trainingEntityIdentifiers"]) == set(
        measurements["identifier"]
    )
    assert final_model["targetOutput"] == "density"
    assert final_model["observedProperty"] == "calculate_density-density"
    if final_predictor == "random_forest":
        assert model_path == final_directory / "model.pkl"
        assert final_model["fitEntityCount"] == budget
        assert final_model["validationEntityCount"] == 0
    else:
        assert final_model["fitEntityCount"] == predictor.n_fit_samples_
        assert final_model["validationEntityCount"] == predictor.n_validation_samples_
        native_predictor = predictor.predictor_
        train_x, train_y = native_predictor.load_data_internal("train")
        validation_x, validation_y = native_predictor.load_data_internal("val")
        assert len(train_x) == len(train_y) == predictor.n_fit_samples_
        assert len(validation_x) == len(validation_y) == predictor.n_validation_samples_
        assert set(train_x.index).isdisjoint(validation_x.index)
        assert len(set(train_x.index) | set(validation_x.index)) == budget
        actual_features = pd.concat([train_x, validation_x])
        actual_targets = pd.concat([train_y, validation_y])
        actual_training = actual_features.assign(test_target=actual_targets)
        ordered_measurements = measurements.set_index("identifier").loc[
            final_model["trainingEntityIdentifiers"]
        ]
        expected_features = native_predictor.transform_features(
            ordered_measurements[list(predictor.feature_names_in_)]
        )
        expected_training = expected_features.assign(
            test_target=ordered_measurements["density"]
        )
        sort_columns = list(actual_training.columns)
        pd.testing.assert_frame_equal(
            actual_training.sort_values(sort_columns).reset_index(drop=True),
            expected_training.sort_values(sort_columns).reset_index(drop=True),
            check_dtype=False,
            check_categorical=False,
            rtol=1e-6,
        )

    trace_path = tmp_path / "trace.json"
    _run_cli(
        repository,
        config_directory,
        context_path,
        "show",
        "trace",
        "space",
        "--use-latest",
        "-o",
        "json",
        "--output-file",
        str(trace_path),
    )
    trace = pd.read_json(trace_path).sort_values("Request Index")
    assert trace["Request Index"].tolist() == list(range(budget))
    assert set(trace["Status"]) == {"Success"}
    last_identifiers = trace.iloc[-1]["Entity IDs"]
    assert len(last_identifiers) == 1
    assert last_identifiers[0] in final_model["trainingEntityIdentifiers"]
