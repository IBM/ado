# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Tests for the throughput recommender's candidate and ADO contracts."""

from pathlib import Path

import pandas as pd
import pytest
from autogluon.tabular import TabularPredictor


def test_candidate_combinations_match_reference_defaults() -> None:
    """Default limits retain the POC's ordered, divisible candidates."""
    from autoconf.throughput_recommender import candidate_combinations

    candidates = candidate_combinations(max_nodes=1, gpus_per_node=8)
    assert candidates[0] == (1, 1)
    assert candidates[-1] == (2048, 8)
    assert all(batch % gpus == 0 for batch, gpus in candidates)
    assert all(batch // gpus <= 256 for batch, gpus in candidates)
    assert {gpus for _, gpus in candidates} == {1, 2, 4, 8}


def test_candidate_combinations_only_include_representable_gpu_counts() -> None:
    """Non-power-of-two node capacities cannot overstate total GPU count."""
    from autoconf.throughput_recommender import candidate_combinations

    candidates = candidate_combinations(max_nodes=2, gpus_per_node=6)
    assert {gpus for _, gpus in candidates} == {1, 2, 4}
    assert (4096, 16) in candidate_combinations(max_nodes=2, gpus_per_node=8)
    with pytest.raises(ValueError, match="positive"):
        candidate_combinations(max_nodes=0, gpus_per_node=8)


def test_prepare_regression_data_filters_and_limits_features() -> None:
    """Only successful, divisible, positive-throughput rows reach AutoGluon."""
    from autoconf.throughput_recommender import (
        REGRESSION_COLUMNS,
        prepare_regression_data,
    )

    data = pd.DataFrame(
        {
            "model_name": ["llama-7b"] * 4,
            "method": ["lora"] * 4,
            "number_gpus": [2, 2, 3, 2],
            "gpu_model": ["NVIDIA-A100-80GB-PCIe"] * 4,
            "tokens_per_sample": [2048] * 4,
            "batch_size": [8, 8, 8, 8],
            "is_valid": [1, 0, 1, 1],
            "dataset_tokens_per_second": [10.0, 12.0, 20.0, 0.0],
            "metadata.uid": ["a", "b", "c", "d"],
        }
    )
    prepared = prepare_regression_data(data)
    assert list(prepared.columns) == REGRESSION_COLUMNS
    assert prepared["dataset_tokens_per_second"].tolist() == [10.0]


def test_prepare_regression_data_rejects_missing_or_unusable_targets() -> None:
    """Training reports a useful error when no throughput observations exist."""
    from autoconf.throughput_recommender import prepare_regression_data

    with pytest.raises(ValueError, match="missing required columns"):
        prepare_regression_data(pd.DataFrame({"model_name": ["llama-7b"]}))


def test_select_best_candidate_preserves_first_tie() -> None:
    """The first highest-throughput candidate wins, as in the POC."""
    from autoconf.throughput_recommender import select_best_candidate

    candidates = pd.DataFrame({"batch_size": [8, 16, 32], "number_gpus": [2, 4, 8]})
    result = select_best_candidate(
        candidates, pd.Series([1, 1, 0]), pd.Series([50.0, 50.0])
    )
    assert result == {
        "can_recommend": True,
        "gpus": 2,
        "workers": 1,
        "effective_batch_size": 8,
        "per_device_batch_size": 4,
        "estimated_throughput": 50.0,
    }


def test_select_best_candidate_uses_index_aligned_throughput() -> None:
    """throughput_predictions with non-default index must align positionally.

    rows 0 and 2 are valid; the regressor returns a Series with index [0, 2].
    positional assignment must pair throughput[0]=10 → row 0 and
    throughput[1]=99 → row 2, so the winner is row 2 (batch_size=32).
    """
    from autoconf.throughput_recommender import select_best_candidate

    candidates = pd.DataFrame({"batch_size": [8, 16, 32], "number_gpus": [2, 2, 2]})
    throughput = pd.Series([10.0, 99.0], index=[0, 2])
    result = select_best_candidate(candidates, pd.Series([1, 0, 1]), throughput)
    assert result == {
        "can_recommend": True,
        "gpus": 2,
        "workers": 1,
        "effective_batch_size": 32,
        "per_device_batch_size": 16,
        "estimated_throughput": 99.0,
    }


def test_select_best_candidate_handles_no_valid_candidate() -> None:
    """No successful candidate has the existing recommender's failure shape."""
    from autoconf.throughput_recommender import select_best_candidate

    candidates = pd.DataFrame({"batch_size": [8], "number_gpus": [2]})
    assert select_best_candidate(
        candidates, pd.Series([0]), pd.Series(dtype=float)
    ) == {"can_recommend": False}


def test_select_best_candidate_returns_exact_multi_node_layout() -> None:
    """Returned nodes and per-node GPUs multiply to the selected total."""
    from autoconf.throughput_recommender import select_best_candidate

    candidates = pd.DataFrame({"batch_size": [4096], "number_gpus": [16]})
    result = select_best_candidate(
        candidates, pd.Series([1]), pd.Series([100.0]), gpus_per_node=8
    )
    assert result["workers"] == 2
    assert result["gpus"] == 8
    assert result["per_device_batch_size"] == 256


def test_unknown_model_warning_is_advisory() -> None:
    """An unseen model produces a warning without becoming a validation error."""
    from autoconf.throughput_recommender import warn_if_unknown_model

    with pytest.warns(UserWarning, match="unvalidated"):
        warn_if_unknown_model("new-model", frozenset({"llama-7b"}))
    warn_if_unknown_model("llama-7b", frozenset({"llama-7b"}))


def test_ado_experiment_interface_and_version() -> None:
    """ADO exposes the required fields, defaults, and six result properties."""
    from ado.schema.domain import VariableTypeEnum
    from autoconf.throughput_recommender import throughput_recommender

    experiment = throughput_recommender._experiment
    assert experiment.identifier == "throughput_recommender"
    required = {prop.identifier: prop for prop in experiment.requiredProperties}
    assert set(required) == {
        "model_name",
        "method",
        "gpu_model",
        "tokens_per_sample",
    }
    assert (
        required["model_name"].propertyDomain.variableType
        == VariableTypeEnum.OPEN_CATEGORICAL_VARIABLE_TYPE
    )
    assert {prop.identifier for prop in experiment.optionalProperties} == {
        "max_nodes",
        "gpus_per_node",
        "model_version",
    }
    assert throughput_recommender._original_func.__defaults__ == (1, 8, "4.1.0")


def test_regressor_path_does_not_change_classifier_path(tmp_path: Path) -> None:
    """The new model and the existing classifier use separate versioned paths."""
    from autoconf.model_paths import model_path
    from autoconf.throughput_recommender import regressor_path

    assert model_path(tmp_path) == tmp_path / "v4-0-0"
    assert regressor_path(tmp_path) == tmp_path / "v4-1-0-regressor"


@pytest.fixture(scope="module")
def trained_predictors(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[TabularPredictor, TabularPredictor, frozenset[str], Path]:
    """Fit small real predictors without the full first-use training budget."""
    from autoconf.throughput_recommender import build_regressor, load_regressor

    directory = tmp_path_factory.mktemp("throughput-models")
    rows = [
        {
            "model_name": "llama-7b",
            "method": "lora",
            "number_gpus": number_gpus,
            "gpu_model": "NVIDIA-A100-80GB-PCIe",
            "tokens_per_sample": 2048,
            "batch_size": number_gpus * (repeat % 4 + 1),
            "is_valid": int(number_gpus > 1),
            "dataset_tokens_per_second": float(number_gpus * 10 + repeat),
        }
        for repeat in range(8)
        for number_gpus in (1, 2, 4, 8)
    ]
    data = pd.DataFrame(rows)
    data_root = directory / "data"
    data_root.mkdir()
    data.to_csv(data_root / "dataset.csv", index=False)
    classifier = TabularPredictor(
        label="is_valid",
        problem_type="binary",
        path=str(directory / "classifier"),
        verbosity=0,
    ).fit(data.drop(columns=["dataset_tokens_per_second"]), hyperparameters={"RF": {}})
    build_regressor(
        model_root=directory,
        data_root_dir=data_root,
        fit_options={"hyperparameters": {"RF": {}}, "num_bag_folds": 0},
    )
    regressor, known_models = load_regressor(model_root=directory)
    return classifier, regressor, known_models, directory


def test_regressor_uses_saved_model_on_later_calls(
    trained_predictors: tuple[TabularPredictor, TabularPredictor, frozenset[str], Path],
) -> None:
    """Training persists model names and subsequent loads reuse the predictor."""
    from autoconf.throughput_recommender import load_regressor, regressor_path

    _, regressor, known_models, directory = trained_predictors
    assert regressor_path(directory).is_dir()
    assert known_models == frozenset({"llama-7b"})
    assert load_regressor(model_root=directory)[0] is regressor
    with pytest.raises(ValueError, match="Unknown throughput model_version"):
        load_regressor(model_version="4.0.0", model_root=directory)


def test_unseen_model_can_still_be_predicted(
    trained_predictors: tuple[TabularPredictor, TabularPredictor, frozenset[str], Path],
) -> None:
    """AutoGluon accepts an unknown category after the advisory warning."""
    from autoconf.throughput_recommender import recommend_throughput

    classifier, regressor, known_models, _ = trained_predictors
    with pytest.warns(UserWarning, match="unvalidated"):
        result = recommend_throughput(
            model_name="new-model",
            method="lora",
            gpu_model="NVIDIA-A100-80GB-PCIe",
            tokens_per_sample=2048,
            max_nodes=1,
            gpus_per_node=8,
            classifier=classifier,
            regressor=regressor,
            known_models=known_models,
        )
    assert result["can_recommend"] is True
    assert result["workers"] == 1
    assert result["effective_batch_size"] == (
        result["per_device_batch_size"] * result["gpus"]
    )
    assert isinstance(result["estimated_throughput"], float)
