# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Tests for the throughput recommender's candidate and ADO contracts."""

from pathlib import Path

import pandas as pd
import pytest
from autogluon.tabular import TabularPredictor


def test_candidate_combinations_only_include_representable_gpu_counts() -> None:
    """Non-power-of-two node capacities cannot overstate total GPU count."""
    from autoconf.throughput_recommender import candidate_combinations

    candidates = candidate_combinations(max_nodes=2, gpus_per_node=6)
    assert {gpus for _, gpus in candidates} == {1, 2, 4}
    assert (4096, 16) in candidate_combinations(max_nodes=2, gpus_per_node=8)
    with pytest.raises(ValueError, match="positive"):
        candidate_combinations(max_nodes=0, gpus_per_node=8)


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

    rows 0 and 2 are valid; the regressor returns a Series with index [0, 1]
    (reset by AutoGluon). Positional alignment must pair index-0 prediction
    10.0 → candidate row 0 and index-1 prediction 99.0 → candidate row 2,
    so the winner is row 2 (batch_size=32). A label-based assignment would
    incorrectly map index 1 → candidate row 1 (invalid) and produce no
    valid candidate, failing the assertion.
    """
    from autoconf.throughput_recommender import select_best_candidate

    candidates = pd.DataFrame({"batch_size": [8, 16, 32], "number_gpus": [2, 2, 2]})
    # valid_mask selects rows 0 and 2; AutoGluon resets the prediction index to [0, 1]
    throughput = pd.Series([10.0, 99.0], index=[0, 1])
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


def _make_build_data(tmp_path: Path) -> Path:
    """Write a minimal regression CSV under tmp_path/data/dataset.csv."""
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
        for number_gpus in (2, 4, 8)
    ]
    data_root = tmp_path / "data"
    data_root.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(data_root / "dataset.csv", index=False)
    return data_root


def test_build_regressor_cleans_up_staging_on_failure(tmp_path: Path) -> None:
    """A failed build must not leave staging artifacts behind.

    clone_for_deployment creates the staging directory before the patch raises,
    so the cleanup path actually removes an existing directory. A retry after
    cleanup must succeed.
    """
    from unittest.mock import patch

    from autoconf.throughput_recommender import build_regressor, regressor_path

    data_root = _make_build_data(tmp_path)
    staging = Path(str(regressor_path(tmp_path)) + ".staging")

    original_clone = TabularPredictor.clone_for_deployment

    def clone_then_fail(self: TabularPredictor, path: str, **kwargs: object) -> None:
        # Call the real method so staging is populated, then raise.
        original_clone(self, path, **kwargs)
        raise RuntimeError("post-clone failure")

    with (
        patch.object(TabularPredictor, "clone_for_deployment", clone_then_fail),
        pytest.raises(RuntimeError, match="post-clone failure"),
    ):
        build_regressor(
            model_root=tmp_path,
            data_root_dir=data_root,
            fit_options={"hyperparameters": {"RF": {}}, "num_bag_folds": 0},
        )

    # Staging and destination must both be absent.
    assert not staging.exists()
    assert not regressor_path(tmp_path).exists()

    # A retry must succeed now that staging is clean.
    result = build_regressor(
        model_root=tmp_path,
        data_root_dir=data_root,
        fit_options={"hyperparameters": {"RF": {}}, "num_bag_folds": 0},
    )
    assert result == regressor_path(tmp_path)
    assert regressor_path(tmp_path).is_dir()


def test_build_regressor_second_caller_skips_training(tmp_path: Path) -> None:
    """A builder that arrives after another has published must not retrain.

    This covers the re-check-under-lock path: a second call with an already-
    present destination returns immediately without touching the model files.
    """
    from autoconf.throughput_recommender import build_regressor, regressor_path

    data_root = _make_build_data(tmp_path)
    build_regressor(
        model_root=tmp_path,
        data_root_dir=data_root,
        fit_options={"hyperparameters": {"RF": {}}, "num_bag_folds": 0},
    )
    destination = regressor_path(tmp_path)
    mtime_before = destination.stat().st_mtime

    # Second call must return without modifying the destination.
    result = build_regressor(
        model_root=tmp_path,
        data_root_dir=data_root,
        fit_options={"hyperparameters": {"RF": {}}, "num_bag_folds": 0},
    )
    assert result == destination
    assert destination.stat().st_mtime == mtime_before


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
