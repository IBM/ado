# Copyright IBM Corporation 2025, 2026

# SPDX-License-Identifier: MIT

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


def _make_df() -> pd.DataFrame:
    """Return a minimal DataFrame with the required classifier columns."""
    return pd.DataFrame(
        {
            "model_name": ["llama-7b"] * 4,
            "method": ["lora"] * 4,
            "number_gpus": [2, 2, 4, 4],
            "gpu_model": ["NVIDIA-A100-80GB-PCIe"] * 4,
            "tokens_per_sample": [2048] * 4,
            "batch_size": [8] * 4,
            "is_valid": [1, 0, 1, 0],
            "train_runtime": [1.0, None, 1.0, None],
        }
    )


def test_build_model_trains_and_returns_path(tmp_path: Path) -> None:
    """build_model() orchestrates download, train, clone and returns the model path."""
    from autoconf.model_paths import model_path
    from autoconf.utils.autoconf_build.ml_classifier import build_model

    expected_path = model_path(tmp_path)

    mock_predictor = MagicMock()
    mock_predictor.disk_usage.return_value = 1000
    mock_predictor.path = str(tmp_path / "tmp_model")

    # Simulate clone_for_deployment creating the model directory on disk.
    def fake_clone(path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)

    mock_predictor.clone_for_deployment.side_effect = fake_clone

    mock_clone = MagicMock()
    mock_clone.disk_usage.return_value = 800
    mock_clone.evaluate.return_value = {"accuracy": 0.9}

    dataset_path = tmp_path / "dataset.csv"

    with (
        patch(
            "autoconf.utils.autoconf_build.ml_classifier.ensure_dataset",
            return_value=dataset_path,
        ),
        patch("pandas.read_csv", return_value=_make_df()),
        patch(
            "autoconf.utils.autoconf_build.ml_classifier.fit_tabular_predictor",
            return_value=(mock_predictor, _make_df(), pd.DataFrame(), 1.0),
        ),
        patch(
            "autoconf.utils.autoconf_build.ml_classifier.TabularPredictor.load",
            return_value=mock_clone,
        ),
    ):
        result = build_model(model_root=tmp_path)

    assert result == expected_path


def test_build_model_raises_if_model_exists(tmp_path: Path) -> None:
    """build_model() raises FileExistsError when the target model directory already exists."""
    from autoconf.model_paths import model_path
    from autoconf.utils.autoconf_build.ml_classifier import build_model

    model_path(tmp_path).mkdir(parents=True)

    dataset_path = tmp_path / "dataset.csv"
    with (
        patch(
            "autoconf.utils.autoconf_build.ml_classifier.ensure_dataset",
            return_value=dataset_path,
        ),
        patch("pandas.read_csv", return_value=_make_df()),
        pytest.raises(FileExistsError),
    ):
        build_model(model_root=tmp_path)
