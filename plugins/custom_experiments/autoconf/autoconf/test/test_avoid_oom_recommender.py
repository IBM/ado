# Copyright IBM Corporation 2025, 2026

# SPDX-License-Identifier: MIT

"""Unit tests for avoid_oom_recommender custom experiment."""

from pathlib import Path

import pandas as pd
import pytest
from autogluon.tabular import TabularPredictor


@pytest.fixture(scope="module")
def generated_model_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build a small real predictor for recommendation integration tests."""
    model_root = tmp_path_factory.mktemp("autoconf-models")
    model_path = model_root / "v4-0-0"
    rows = [
        {
            "model_name": "llama-7b",
            "method": "lora",
            "number_gpus": 1,
            "gpu_model": "NVIDIA-A100-80GB-PCIe",
            "tokens_per_sample": 8192,
            "batch_size": 16,
            "is_valid": 0,
        },
        {
            "model_name": "llama-7b",
            "method": "lora",
            "number_gpus": 2,
            "gpu_model": "NVIDIA-A100-80GB-PCIe",
            "tokens_per_sample": 8192,
            "batch_size": 32,
            "is_valid": 0,
        },
        {
            "model_name": "llama-7b",
            "method": "lora",
            "number_gpus": 4,
            "gpu_model": "NVIDIA-A100-80GB-PCIe",
            "tokens_per_sample": 8192,
            "batch_size": 64,
            "is_valid": 1,
        },
        {
            "model_name": "llama-7b",
            "method": "lora",
            "number_gpus": 16,
            "gpu_model": "NVIDIA-A100-80GB-PCIe",
            "tokens_per_sample": 8192,
            "batch_size": 32,
            "is_valid": 1,
        },
        {
            "model_name": "llama3.1-405b",
            "method": "full",
            "number_gpus": 64,
            "gpu_model": "NVIDIA-A100-80GB-PCIe",
            "tokens_per_sample": 8192,
            "batch_size": 8192,
            "is_valid": 0,
        },
        {
            "model_name": "llama3.1-405b",
            "method": "full",
            "number_gpus": 64,
            "gpu_model": "NVIDIA-A100-80GB-PCIe",
            "tokens_per_sample": 512,
            "batch_size": 64,
            "is_valid": 1,
        },
    ]
    training_data = pd.DataFrame(rows * 10)
    TabularPredictor(
        label="is_valid",
        path=model_path,
        problem_type="binary",
        verbosity=0,
    ).fit(
        training_data,
        hyperparameters={"RF": {}},
    )
    return model_root


@pytest.fixture
def use_generated_model(
    generated_model_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Configure the recommender to load the temporary fixture model."""
    from autoconf.min_gpu_recommender import load_model

    monkeypatch.setattr("autoconf.model_paths.DEFAULT_MODEL_ROOT", generated_model_root)
    load_model.cache_clear()


def test_avoid_oom_recommender_preserves_original_gpus_when_no_oom(
    use_generated_model: None,
) -> None:
    """Test that avoid_oom_recommender returns the original number_gpus when it won't cause OOM.

    This test uses a configuration where the user requests 16 GPUs:
    - model_name: llama-7b
    - method: lora
    - gpu_model: NVIDIA-A100-80GB-PCIe
    - tokens_per_sample: 8192
    - per_device_train_batch_size: 2 (small batch size)
    - number_gpus: 16 (original request - more than minimum needed)

    With per_device_train_batch_size=2, the minimum GPUs needed is 4.
    Since the user requests 16 GPUs (more than the minimum 4),
    avoid_oom_recommender should return number_gpus=16 unchanged.
    """
    from autoconf.min_gpu_recommender import avoid_oom_recommender

    # Call avoid_oom_recommender with 16 GPUs (more than the minimum needed)
    # The experiment should preserve this value since it won't cause OOM
    result = avoid_oom_recommender(
        model_name="llama-7b",
        method="lora",
        gpu_model="NVIDIA-A100-80GB-PCIe",
        tokens_per_sample=8192,
        per_device_train_batch_size=2,  # Small batch size - minimum is 4 GPUs
        number_gpus=16,  # User wants 16 GPUs - should be preserved
        gpus_per_worker=8,
        max_gpus=64,
        model_version="4.0.0",
    )

    # The experiment should recognize that 16 GPUs won't cause OOM
    # and return configuration that uses 16 total GPUs
    assert result["can_recommend"] is True, "Should be able to make a recommendation"
    # With 16 total GPUs and gpus_per_worker=8: workers=2, gpus_per_worker=8
    assert result["workers"] == 2, f"Expected workers=2, but got {result['workers']}"
    assert result["gpus"] == 8, (
        f"Expected gpus=8 (per worker), but got {result['gpus']}. "
        "With 2 workers and 8 GPUs per worker, total is 16 GPUs as requested."
    )
    # Verify total GPUs = workers * gpus_per_worker = 2 * 8 = 16
    total_gpus = result["workers"] * result["gpus"]
    assert total_gpus == 16, f"Expected total 16 GPUs, but got {total_gpus}"


def test_avoid_oom_recommender_finds_minimum_when_oom_expected(
    use_generated_model: None,
) -> None:
    """Test that avoid_oom_recommender finds minimum GPUs when original would cause OOM.

    If the user requests 1 GPU with a large per_device_train_batch_size that would cause OOM,
    the experiment should find the minimum number of GPUs that avoids OOM.
    """
    from autoconf.min_gpu_recommender import avoid_oom_recommender

    # Use a configuration that would cause OOM with 1 GPU but works with more
    result = avoid_oom_recommender(
        model_name="llama-7b",
        method="lora",
        gpu_model="NVIDIA-A100-80GB-PCIe",
        tokens_per_sample=8192,
        per_device_train_batch_size=16,  # Larger batch size
        number_gpus=1,  # Original request that will likely cause OOM
        gpus_per_worker=8,
        max_gpus=64,
        model_version="4.0.0",
    )

    # Should recommend more than 1 GPU
    assert result["can_recommend"] is True
    assert result["gpus"] > 1, (
        f"Expected gpus > 1 when original would OOM, but got {result['gpus']}"
    )


def test_avoid_oom_recommender_no_valid_config_exists(
    use_generated_model: None,
) -> None:
    """Test that avoid_oom_recommender returns can_recommend=False when no valid config exists."""
    from autoconf.min_gpu_recommender import avoid_oom_recommender

    # Use a configuration that's likely impossible (very large model with huge batch size)
    result = avoid_oom_recommender(
        model_name="llama3.1-405b",
        method="full",
        gpu_model="NVIDIA-A100-80GB-PCIe",
        tokens_per_sample=8192,
        per_device_train_batch_size=128,
        number_gpus=64,
        gpus_per_worker=8,
        max_gpus=64,
        model_version="4.0.0",
    )

    # Should not be able to recommend
    assert result["can_recommend"] is False
    assert "gpus" not in result
    assert "workers" not in result
