# Copyright (c) IBM Corporation

# SPDX-License-Identifier: MIT

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from autoconf.utils.pydantic_models import JobConfig
from autoconf.utils.recommender import (
    get_model_prediction_and_metadata,
    recommend_min_gpu,
)

# Example configurations

valid_config_dict = {
    "model_name": "granite-3.2-8b-instruct",
    "method": "lora",
    "gpu_model": "NVIDIA-A100-80GB-PCIe",
    "tokens_per_sample": 8192.0,
    "batch_size": 16.0,
    "number_gpus": 2,
    "model_version": "2.0.0",
}

invalid_config_dict = {
    "model_name": "llama-13b",
    "method": "full",
    "gpu_model": "NVIDIA-A100-80GB-PCIe",
    "tokens_per_sample": 4096.0,
    "batch_size": 128.0,
    "number_gpus": 8,
    "model_version": "2.0.0",
}

# Convert to JobConfig instances

valid_job_config = JobConfig(**valid_config_dict)
invalid_job_config = JobConfig(**invalid_config_dict)

# Mock return values

mock_prediction_valid = [1]
mock_prediction_invalid = [0]
mock_rbc_valid = (1, [])
mock_rbc_invalid = (0, ["Invalid configuration"])


@pytest.fixture
def mock_predictor_valid() -> MagicMock:
    mock = MagicMock()

    def mock_predict(df: pd.DataFrame) -> pd.Series:
        # Simulate prediction logic based on gpus_per_worker
        print(df)
        val = 1 if int(df["number_gpus"].values[0]) == 2 else 0
        return pd.Series([val])  # Mimics .values[0] behavior

    mock.predict.side_effect = mock_predict
    return mock


@pytest.fixture
def mock_predictor_invalid() -> MagicMock:
    mock = MagicMock()

    def mock_predict(df: pd.DataFrame) -> pd.Series:
        val = -1
        return pd.Series([val])

    mock.predict.side_effect = mock_predict
    return mock


@patch(
    "autoconf.utils.rule_based_classifier.is_row_valid",
    return_value=mock_rbc_valid,
)
def test_get_model_prediction_and_metadata_valid(
    mock_is_row_valid: MagicMock, mock_predictor_valid: MagicMock
) -> None:
    print(valid_job_config)
    pred, metadata = get_model_prediction_and_metadata(
        valid_job_config, predictor=mock_predictor_valid
    )
    assert pred == 1
    assert isinstance(metadata, dict)
    assert metadata["Rule-Based Classifier error"] == ""


@patch(
    "autoconf.utils.rule_based_classifier.is_row_valid",
    return_value=mock_rbc_invalid,
)
def test_get_model_prediction_and_metadata_invalid(
    mock_is_row_valid: MagicMock, mock_predictor_valid: MagicMock
) -> None:
    pred, metadata = get_model_prediction_and_metadata(
        invalid_job_config, predictor=mock_predictor_valid
    )
    assert pred is None or pred == 0
    assert isinstance(metadata, dict)


@patch(
    "autoconf.utils.rule_based_classifier.is_row_valid",
    return_value=mock_rbc_valid,
)
def test_recommend_min_gpu_valid(
    mock_is_row_valid: MagicMock, mock_predictor_valid: MagicMock
) -> None:
    min_gpu, metadata = recommend_min_gpu(
        valid_job_config, valid_n_gpu_list=[1, 2, 4], predictor=mock_predictor_valid
    )
    assert min_gpu == 2
    assert isinstance(metadata, dict)


@patch(
    "autoconf.utils.rule_based_classifier.is_row_valid",
    return_value=mock_rbc_invalid,
)
def test_recommend_min_gpu_invalid(
    mock_is_row_valid: MagicMock, mock_predictor_invalid: MagicMock
) -> None:
    min_gpu, metadata = recommend_min_gpu(
        invalid_job_config, valid_n_gpu_list=[1, 2, 4], predictor=mock_predictor_invalid
    )
    assert min_gpu == -1
    assert isinstance(metadata, dict)
